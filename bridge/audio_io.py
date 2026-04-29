"""Microphone capture and speaker playback via sounddevice.

MicStream uses a callback-based InputStream with a bounded queue:
- PortAudio calls our callback from a realtime thread with the next block
- We copy+enqueue non-blocking; if the queue is full we drop the frame rather
  than block the RT thread (blocking a PortAudio callback causes xruns and
  on Linux can wedge alsa)
- The iterator drains the queue on the main thread

Speaker is a thin wrapper around `sd.play(..., blocking=True)`. The sample
rate is per-call because the TTS output (32 kHz for GPT-SoVITS v2Pro) differs
from the mic rate (16 kHz for Whisper/VAD).
"""
from __future__ import annotations

import queue
from collections.abc import Iterator

import numpy as np
import sounddevice as sd


def _resolve_device(name: str | None, *, want_input: bool) -> int | None:
    """Substring-match a device name against the sounddevice device list.

    Returns the device index, or None to let PortAudio pick the default.
    Raises ValueError if `name` is given but no matching device is found.
    """
    if name is None:
        return None
    want_channels = "max_input_channels" if want_input else "max_output_channels"
    for idx, info in enumerate(sd.query_devices()):
        if name.lower() in info["name"].lower() and info[want_channels] > 0:
            return idx
    kind = "input" if want_input else "output"
    raise ValueError(f"no {kind} device matching {name!r}")


class MicStream:
    """Blocking mic reader that yields fixed-size int16 mono PCM frames.

    Always use as a context manager so the PortAudio stream is released
    cleanly on exit.
    """

    # Soft cap on in-flight frames. At 32 ms/frame that's ~1.6 s of audio —
    # enough to absorb GC hiccups without unbounded memory growth.
    _QUEUE_MAX = 50

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int,
        device: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._device = _resolve_device(device, want_input=True)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self._QUEUE_MAX)
        self._stream: sd.InputStream | None = None
        self._closed = False

    def __enter__(self) -> "MicStream":
        def callback(indata, frames, time, status):  # noqa: ARG001 — signature fixed by PortAudio
            # `status` can carry xrun / overflow flags; we just ignore them here.
            # `indata` is shape (frames, channels). For mono, flatten to 1D.
            # The buffer is reused by the RT thread, so we must .copy().
            try:
                self._queue.put_nowait(indata.copy().reshape(-1))
            except queue.Full:
                # Drop frames under backpressure rather than block the RT thread.
                pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            dtype="int16",
            channels=1,
            device=self._device,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        # 1.0 s timeout so we can notice close() in a reasonable time.
        while not self._closed:
            try:
                yield self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


class Speaker:
    """PCM playback sink. Accepts int16 mono; sample rate is per-call.

    Uses `sd.play(..., blocking=True)` which opens and closes an OutputStream
    per call. That's a few ms of overhead but means we don't have to care
    about rate changes between calls (mic 16 kHz vs TTS 32 kHz).
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = _resolve_device(device, want_input=False)

    def __enter__(self) -> "Speaker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Play int16 mono PCM at the given sample rate. Blocks until complete."""
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        sd.play(pcm, samplerate=sample_rate, device=self._device, blocking=True)

    def stop(self) -> None:
        """Abort any in-progress playback."""
        sd.stop()
