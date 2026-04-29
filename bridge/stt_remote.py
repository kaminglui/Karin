"""Remote faster-whisper client.

Drop-in replacement for :class:`bridge.stt.WhisperSTT` that POSTs PCM
audio to a sidecar running `deploy/pc-tts/tts_server.py` (which also
exposes `/transcribe`). Keeps the Whisper memory cost off the Jetson.

The interface matches :class:`WhisperSTT` exactly — same constructor
kwargs (model/device/compute_type are accepted for call-site symmetry
but only ``language``/``beam_size``/``vad_filter`` actually travel to
the remote server; the remote decides its own model via env vars).
``transcribe(pcm)`` returns the same ``str`` the local wrapper does.

Selection between local vs remote is made by ``web/server.py`` at
startup: if ``KARIN_STT_BASE_URL`` is set, build a ``RemoteWhisperSTT``;
otherwise build a local ``WhisperSTT`` as before.
"""
from __future__ import annotations

import base64
import logging

import httpx
import numpy as np

log = logging.getLogger("bridge.stt_remote")


class RemoteWhisperSTT:
    """Transcribe via an HTTP POST to a Karin voice sidecar."""

    def __init__(
        self,
        base_url: str,
        language: str | None = None,
        beam_size: int = 1,
        vad_filter: bool = False,
        request_timeout: float = 30.0,
    ) -> None:
        """Open an httpx client against the remote sidecar.

        Args:
            base_url: Sidecar base URL, e.g.
                ``http://<pc-tailscale-ip>:9880``. Trailing slash OK.
            language: Whisper language code or ``None`` / ``auto`` for
                per-utterance detection. Forwarded to the remote call.
            beam_size: Beam width for decoding. 1 = latency-first.
            vad_filter: Whether the remote should run its own VAD
                filter. Keep False — upstream VAD already segmented.
            request_timeout: Per-request HTTP timeout in seconds.
                The remote's lazy Whisper load can take ~5 s on first
                call, so this defaults to 30 s.
        """
        if not base_url:
            raise ValueError("RemoteWhisperSTT requires a non-empty base_url")
        self.base_url = base_url.rstrip("/")
        # Normalize language → None when unspecified / "auto" so the
        # remote's per-request override doesn't override anything.
        if language is None or str(language).strip().lower() in ("", "auto"):
            self.language: str | None = None
        else:
            self.language = str(language).strip().lower()
        self.beam_size = int(beam_size)
        self.vad_filter = bool(vad_filter)
        self._client = httpx.Client(base_url=self.base_url, timeout=request_timeout)

    def transcribe(self, pcm: np.ndarray) -> str:
        """POST the PCM audio to ``{base_url}/transcribe`` and return the text.

        Args:
            pcm: int16 mono PCM at 16 kHz (same as the local wrapper expects).

        Returns:
            The transcribed text, stripped. Empty string if the remote
            returned no segments.

        Raises:
            RuntimeError: on a non-200 response. The bridge's existing
                ``except Exception`` around ``stt.transcribe(pcm)`` logs
                and either 500s the request (PTT path) or routes as
                silence (streaming path), so the caller decides how
                loud the failure should be.
        """
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        if pcm.ndim != 1:
            # Flatten stereo/extra-channel arrays to mono just in case —
            # the local wrapper accepts any 1-D so this matches.
            pcm = pcm.reshape(-1)
        payload = {
            "pcm_base64": base64.b64encode(pcm.tobytes()).decode("ascii"),
            "sample_rate": 16000,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
        }
        if self.language is not None:
            payload["language"] = self.language
        # One retry on transient connect failures. Covers the common
        # case of the PC sidecar being briefly unreachable (Tailscale
        # reconnect, sleep/wake, brief restart). Two tries total; the
        # second uses the same client so DNS/keepalive state is reused.
        # Non-connect HTTPError (4xx/5xx response, timeout mid-read,
        # protocol error) is NOT retried — those aren't transient in
        # a way a second try fixes.
        last_err: Exception | None = None
        r = None
        for attempt in range(2):
            try:
                r = self._client.post("/transcribe", json=payload)
                break
            except httpx.ConnectError as e:
                last_err = e
                if attempt == 0:
                    log.warning(
                        "remote STT connect failed (attempt 1/2), retrying: %s", e,
                    )
                    continue
                raise RuntimeError(
                    f"remote STT connect failed after 2 attempts: {e}"
                ) from e
            except httpx.HTTPError as e:
                raise RuntimeError(f"remote STT request failed: {e}") from e
        if r is None:
            # Defensive — loop either set r or raised. Keep mypy/type-
            # checkers happy without a cast.
            raise RuntimeError(
                f"remote STT had no response object: {last_err}"
            )
        if r.status_code != 200:
            detail = ""
            try:
                body = r.json()
                detail = body.get("detail") or body.get("message") or ""
            except Exception:
                detail = r.text[:200]
            raise RuntimeError(
                f"remote STT HTTP {r.status_code}: {detail}"
            )
        try:
            body = r.json()
        except Exception as e:
            raise RuntimeError(f"remote STT returned non-JSON: {e}") from e
        text = str(body.get("text") or "").strip()
        return text

    def close(self) -> None:
        """Close the underlying httpx client. Idempotent."""
        try:
            self._client.close()
        except Exception:
            pass

    def __del__(self) -> None:
        # Best-effort cleanup if the caller forgets to close().
        self.close()
