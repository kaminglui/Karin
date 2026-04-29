"""GPT-SoVITS v2Pro inference client.

Talks to the ``api_v2.py`` server that ships with GPT-SoVITS. Schema pinned
to:

    RVC-Boss/GPT-SoVITS @ 2d9193b0d3c0eae0c3a14d8c68a839f1bae157dc
    (main HEAD as of 2026-02-09, verified 2026-04-11)

The server must already be running. Launch it (in a separate shell or
systemd unit) with::

    python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml

The ``-c`` config can be any valid ``tts_infer.yaml`` — this client overrides
the GPT and SoVITS weights at startup via ``/set_gpt_weights`` and
``/set_sovits_weights``, pointing at the checkpoints from
``config/assistant.yaml``.

Non-streaming synthesis only; streaming mode is not yet implemented.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

log = logging.getLogger("bridge.tts")


class TTSError(RuntimeError):
    """Raised when the GPT-SoVITS server rejects a request or returns garbage."""


class SoVITSTTS:
    """HTTP client for a running GPT-SoVITS ``api_v2.py`` server.

    The client optionally pushes GPT and SoVITS weight paths at construction
    time via ``/set_gpt_weights`` and ``/set_sovits_weights``, so you can
    point the server at a custom voice without editing its
    ``tts_infer.yaml``.
    """

    def __init__(
        self,
        base_url: str,
        endpoint: str,
        ref_audio_path: str,
        prompt_text: str,
        prompt_lang: str,
        text_lang: str,
        top_k: int,
        top_p: float,
        temperature: float,
        speed_factor: float,
        streaming_mode: bool,
        gpt_weights_path: str | None = None,
        sovits_weights_path: str | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        """Initialise the client and optionally push weight paths to the server.

        Args:
            base_url: Base URL of the api_v2.py server
                (e.g. ``http://127.0.0.1:9880``).
            endpoint: Synthesis endpoint path. Normally ``/tts``.
            ref_audio_path: Path to the reference WAV used to condition
                inference.
            prompt_text: Transcript of ``ref_audio_path``. Empty is allowed.
            prompt_lang: Language of the reference clip (``en``, ``ja``,
                ``zh``, ...).
            text_lang: Language of the text passed to ``synthesize``. Must
                be present in the server's ``tts_config.languages`` set.
            top_k: Sampler top-k.
            top_p: Sampler top-p.
            temperature: Sampler temperature.
            speed_factor: Output audio speed multiplier.
            streaming_mode: Stored for config compatibility. The client's
                one-shot and chunked synthesis paths currently choose
                their own request-level streaming settings.
            gpt_weights_path: Optional absolute path to a GPT .ckpt. If
                given, pushed to the server via ``/set_gpt_weights`` during
                construction.
            sovits_weights_path: Optional absolute path to a SoVITS .pth.
                If given, pushed via ``/set_sovits_weights``.
            request_timeout: Per-request HTTP timeout in seconds.

        Raises:
            TTSError: If a weight-push call fails.
        """
        self.streaming_mode = streaming_mode

        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        # When client + server share a host (localhost), resolve() makes
        # relative paths absolute. When the server runs REMOTELY (PC via
        # Tailscale), the path must already be absolute on the PC's
        # filesystem and resolve() would mangle it (e.g. a Windows
        # backslash path resolved on a Linux Jetson). Keep the path as-is
        # — the config is the source of truth for where the file lives.
        self.ref_audio_path = str(ref_audio_path)
        self.prompt_text = prompt_text
        self.prompt_lang = prompt_lang
        self.text_lang = text_lang
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.speed_factor = speed_factor

        self._client = httpx.Client(base_url=self.base_url, timeout=request_timeout)
        self._sample_rate: int | None = None  # learned from first response

        # Store weight paths and attempt to push them, but don't crash the
        # process if GPT-SoVITS isn't running yet. This lets the unified
        # web server boot regardless of 9880 state — text chat, tool calls,
        # and panels still work; TTS synthesis itself will retry weight
        # loading on first synthesize() call and fail loudly there if the
        # backend is still down.
        self._gpt_weights_path = gpt_weights_path
        self._sovits_weights_path = sovits_weights_path
        self._weights_loaded = False
        self._ensure_weights_loaded(swallow_connection_errors=True)

    def _ensure_weights_loaded(self, *, swallow_connection_errors: bool = False) -> None:
        """Idempotent weight push. If the TTS backend is unreachable at
        boot, we warn and defer; synthesize() retries each time until it
        succeeds. This keeps the main web server resilient to GPT-SoVITS
        not being up yet (common during development / first-boot)."""
        if self._weights_loaded:
            return
        try:
            if self._gpt_weights_path is not None:
                self._set_weights("/set_gpt_weights", self._gpt_weights_path)
            if self._sovits_weights_path is not None:
                self._set_weights("/set_sovits_weights", self._sovits_weights_path)
            self._weights_loaded = True
        except httpx.HTTPError as e:
            if not swallow_connection_errors:
                raise TTSError(f"TTS backend unreachable at {self.base_url}: {e}") from e
            log.warning(
                "TTS backend unreachable at %s (%s) — server will still "
                "run; voice output will activate once GPT-SoVITS is up.",
                self.base_url, e,
            )

    # -------------------------------------------------------------- weights

    def _set_weights(self, route: str, path: str) -> None:
        """GET ``route`` with ``weights_path`` set to a path on the TTS
        server's filesystem. When the server runs on the PC, this is a
        PC-side path — don't resolve it against the Jetson's FS."""
        resp = self._client.get(route, params={"weights_path": str(path)})
        if resp.status_code != 200:
            raise TTSError(f"{route} failed [{resp.status_code}]: {resp.text}")

    def switch_voice(
        self,
        ref_audio_path: str,
        gpt_weights_path: str,
        sovits_weights_path: str,
    ) -> None:
        """Hot-swap to a different voice model at runtime.

        Pushes new weights to the GPT-SoVITS server and updates the
        ref audio path used for subsequent synthesize() calls. No
        restart needed — the server re-loads the weights in place.
        """
        self._gpt_weights_path = gpt_weights_path
        self._sovits_weights_path = sovits_weights_path
        self.ref_audio_path = ref_audio_path
        self._weights_loaded = False
        self._ensure_weights_loaded(swallow_connection_errors=False)
        log.info("switched voice: ref=%s gpt=%s sovits=%s",
                 ref_audio_path, gpt_weights_path, sovits_weights_path)

    # ------------------------------------------------------------ synthesis

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text and return mono PCM audio.

        v2Pro returns WAV at 32 kHz. The sample rate is read from the WAV
        header on each call, not assumed.

        Args:
            text: Non-empty text to synthesize.

        Returns:
            A tuple ``(pcm, sample_rate)`` where ``pcm`` is a 1-D int16
            numpy array and ``sample_rate`` is the server's output rate
            in Hz.

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
            TTSError: If the server rejects the request or returns a
                non-audio response.
        """
        if not text.strip():
            raise ValueError("synthesize() received empty text")

        # Truncate at the sidecar boundary (KARIN_TTS_MAX_CHARS=1000 on
        # the PC sidecar). Without this guard, a longer reply produces
        # HTTP 413 ("text too long for sidecar TTS"); with it, we
        # synthesize the first ~950 chars + an ellipsis. The chat UI
        # still shows the full reply — only the spoken audio is
        # truncated. The 50-char headroom covers prompt-injected
        # punctuation that can push past the sidecar's exact cap.
        _MAX_TTS_CHARS = 950
        if len(text) > _MAX_TTS_CHARS:
            text = text[: _MAX_TTS_CHARS].rstrip() + "…"
            log.info(
                "TTS truncating reply from %d to %d chars for sidecar",
                len(text), _MAX_TTS_CHARS,
            )

        # Retry weight loading if __init__ couldn't reach the backend.
        # swallow_connection_errors=False so this raises TTSError cleanly
        # if GPT-SoVITS is still down when the first TTS call arrives.
        self._ensure_weights_loaded(swallow_connection_errors=False)

        body = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "speed_factor": self.speed_factor,
            "media_type": "wav",
            "streaming_mode": False,
            "repetition_penalty": 1.5,
            "parallel_infer": True,
            "text_split_method": "cut5",
        }
        resp = self._client.post(self.endpoint, json=body)

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise TTSError(f"POST {self.endpoint} failed [{resp.status_code}]: {detail}")

        content_type = resp.headers.get("content-type", "")
        if "audio" not in content_type.lower():
            raise TTSError(
                f"expected audio response, got content-type={content_type!r}, "
                f"body[:200]={resp.content[:200]!r}"
            )

        buf = io.BytesIO(resp.content)
        audio, sr = sf.read(buf, dtype="int16", always_2d=False)
        if audio.ndim != 1:
            # GPT-SoVITS outputs mono; downmix defensively in case upstream changes.
            audio = audio.mean(axis=1).astype(np.int16)

        self._sample_rate = int(sr)
        return audio, int(sr)

    # ---------------------------------------------------------- streaming

    def synthesize_stream(self, text: str):
        """Yield ``(pcm_chunk, sample_rate)`` tuples as GPT-SoVITS generates.

        The bridge uses the streaming HTTP path and yields chunks as they
        arrive from GPT-SoVITS. On this stack we intentionally post with
        ``streaming_mode=0`` because mode 1 had severe first-byte latency
        variance on short and medium utterances. The first chunk includes
        a 44-byte WAV header; subsequent chunks are raw int16 PCM.

        Args:
            text: Non-empty text to synthesize.

        Yields:
            Tuples ``(pcm, sample_rate)`` where ``pcm`` is a 1-D int16
            numpy array and ``sample_rate`` is in Hz.

        Raises:
            ValueError: If ``text`` is empty.
            TTSError: If the server rejects the request.
        """
        import struct

        if not text.strip():
            raise ValueError("synthesize_stream() received empty text")

        # Same sidecar-boundary truncation as synthesize() — see that
        # method for rationale.
        _MAX_TTS_CHARS = 950
        if len(text) > _MAX_TTS_CHARS:
            text = text[: _MAX_TTS_CHARS].rstrip() + "…"
            log.info(
                "TTS streaming: truncating reply to %d chars for sidecar",
                _MAX_TTS_CHARS,
            )

        # See synthesize() above — retry weight loading on first use in
        # case the backend wasn't up at __init__.
        self._ensure_weights_loaded(swallow_connection_errors=False)

        body = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "speed_factor": self.speed_factor,
            "media_type": "wav",
            # streaming_mode=1 had a severe TTFB pathology on short /
            # medium text on this GPT-SoVITS + Jetson+PC setup (RTF ~10
            # with 18–27 s stdev; see scripts/bench_tts.py for the
            # numbers). Batched is consistently RTF 0.5 across all
            # lengths. Long-text TTFB is the same either way
            # (~7.8 s both), so streaming never actually bought us
            # earlier audio — just variance.
            "streaming_mode": 0,
            "repetition_penalty": 1.5,
            "parallel_infer": True,
            "text_split_method": "cut5",
        }

        with self._client.stream("POST", self.endpoint, json=body) as resp:
            if resp.status_code != 200:
                error_body = b""
                for chunk in resp.iter_bytes():
                    error_body += chunk
                try:
                    detail = error_body.decode(errors="replace")
                except Exception:
                    detail = str(error_body[:500])
                raise TTSError(
                    f"POST {self.endpoint} stream failed [{resp.status_code}]: {detail}"
                )

            header_buf = b""
            sr = 32000
            header_parsed = False

            for raw_chunk in resp.iter_bytes(chunk_size=16384):
                if not header_parsed:
                    header_buf += raw_chunk
                    if len(header_buf) >= 44:
                        sr = struct.unpack("<I", header_buf[24:28])[0]
                        pcm_data = header_buf[44:]
                        header_parsed = True
                        usable = len(pcm_data) - (len(pcm_data) % 2)
                        if usable > 0:
                            yield np.frombuffer(pcm_data[:usable], dtype=np.int16), sr
                    continue

                usable = len(raw_chunk) - (len(raw_chunk) % 2)
                if usable > 0:
                    yield np.frombuffer(raw_chunk[:usable], dtype=np.int16), sr

    # --------------------------------------------------------------- utils

    @property
    def sample_rate(self) -> int | None:
        """The last observed server output sample rate, or None before the first call."""
        return self._sample_rate

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "SoVITSTTS":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
