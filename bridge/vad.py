"""Silero VAD wrapper with a simple state-machine segmenter.

Consumes a stream of fixed-size int16 mic frames and yields one complete
utterance (int16 PCM) per "turn" — from the first speech frame (with
pre-padding from a ring buffer) through ``min_silence_ms`` of trailing
silence, tail-trimmed back to ``speech_pad_ms`` so we don't include a long
hanging silence.

Design choice: we call ``model(tensor, sr)`` directly per frame instead of
using silero's VADIterator. The state machine is simple enough to own, and
it insulates us from VADIterator's per-version quirks (its valid chunk
sizes and return shapes have churned across silero-vad releases).

Silero VAD 5.x is strict about chunk size: 512 samples @ 16 kHz or 256 @
8 kHz. The config pins ``audio.frame_ms: 32`` to land on exactly 512
samples at 16 kHz.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import numpy as np


class SileroVAD:
    """Silero VAD segmenter that emits one complete utterance per turn.

    Each mic frame is scored by Silero's LSTM; a state machine tracks
    ``idle``/``speaking``, holds a ring buffer of pre-speech padding, and
    emits a concatenated utterance once ``min_silence_ms`` of trailing
    silence is observed. Trailing silence is trimmed back to
    ``speech_pad_ms`` so downstream STT doesn't waste time on dead air.
    """

    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        min_silence_ms: int,
        speech_pad_ms: int,
        min_speech_ms: int,
    ) -> None:
        """Initialise the segmenter and load the Silero VAD model.

        Args:
            sample_rate: PCM sample rate in Hz. Must be 16000 or 8000.
            threshold: Speech probability cutoff (0.0-1.0).
            min_silence_ms: Silence duration that ends an utterance.
            speech_pad_ms: Pre/post-padding around detected speech.
            min_speech_ms: Utterances shorter than this are dropped.
        """
        # Lazy imports — torch and silero_vad are heavyweight and we don't
        # want to pay the import cost on any module that just wants types.
        from silero_vad import load_silero_vad
        import torch

        self._torch = torch
        self.model = load_silero_vad()
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_ms = min_speech_ms

    def _speech_prob(self, frame: np.ndarray) -> float:
        """Run one forward pass and return the frame's speech probability."""
        audio_f = frame.astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(audio_f)
        with self._torch.no_grad():
            return float(self.model(tensor, self.sample_rate).item())

    def segment(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        """Consume mic frames and yield one complete utterance at a time.

        Args:
            frames: An iterable of int16 mono PCM frames. All frames must
                be the same length; the first frame's length is used to
                derive per-frame duration for silence/padding calculations.

        Yields:
            One int16 mono PCM array per utterance. Each array is the
            concatenation of pre-pad + speech frames + tail pad.
        """
        state = "idle"
        frame_ms: float | None = None
        pre_pad: deque[np.ndarray] | None = None
        utterance: list[np.ndarray] = []
        silence_ms = 0.0

        for frame in frames:
            # Derive frame duration from the first frame we see, so we don't
            # have to take it as a constructor arg. All subsequent frames are
            # assumed to be the same size (enforced upstream by audio_io).
            if frame_ms is None:
                frame_ms = len(frame) * 1000.0 / self.sample_rate
                pre_pad_cap = max(1, int(round(self.speech_pad_ms / frame_ms)))
                pre_pad = deque(maxlen=pre_pad_cap)

            prob = self._speech_prob(frame)
            is_speech = prob >= self.threshold

            if state == "idle":
                # Keep a rolling window of the last ~speech_pad_ms of audio
                # so when speech begins we can include a bit of lead-in.
                assert pre_pad is not None
                pre_pad.append(frame)
                if is_speech:
                    state = "speaking"
                    utterance = list(pre_pad)
                    utterance.append(frame)
                    silence_ms = 0.0
            else:  # speaking
                utterance.append(frame)
                if is_speech:
                    silence_ms = 0.0
                else:
                    silence_ms += frame_ms
                    if silence_ms >= self.min_silence_ms:
                        # End of utterance. The tail currently has
                        # min_silence_ms of silence; trim back to
                        # speech_pad_ms so downstream STT doesn't waste
                        # time on dead air.
                        excess_ms = self.min_silence_ms - self.speech_pad_ms
                        trim_frames = max(0, int(round(excess_ms / frame_ms)))
                        if trim_frames > 0:
                            utterance = utterance[:-trim_frames]

                        total_ms = len(utterance) * frame_ms
                        if total_ms >= self.min_speech_ms:
                            yield np.concatenate(utterance).astype(np.int16)

                        # Reset for the next utterance.
                        state = "idle"
                        utterance = []
                        silence_ms = 0.0
                        assert pre_pad is not None
                        pre_pad.clear()
                        # Reset the LSTM hidden state so the next utterance
                        # starts with a clean context.
                        self.model.reset_states()
