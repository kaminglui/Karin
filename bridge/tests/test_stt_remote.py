"""Tests for the remote faster-whisper client.

`bridge.stt_remote.RemoteWhisperSTT` is a drop-in for `WhisperSTT` that
POSTs PCM audio to a PC-side voice sidecar (`deploy/pc-tts/tts_server.py`).
No live server contacted — `httpx.MockTransport` canned responses.
"""
from __future__ import annotations

import base64
from typing import Callable

import httpx
import numpy as np
import pytest

from bridge.stt_remote import RemoteWhisperSTT


def _make_remote(
    responder: Callable[[httpx.Request], httpx.Response],
    **kw,
) -> RemoteWhisperSTT:
    """Build a RemoteWhisperSTT whose httpx client routes through a mock."""
    defaults = dict(
        base_url="http://mock:9880",
        language="en",
        beam_size=1,
        vad_filter=False,
        request_timeout=5.0,
    )
    defaults.update(kw)
    client = RemoteWhisperSTT(**defaults)
    # Swap the auto-built httpx client for one wired to the mock.
    client.close()
    client._client = httpx.Client(
        base_url="http://mock:9880",
        transport=httpx.MockTransport(responder),
        timeout=5.0,
    )
    return client


def _pcm16(samples: int = 16000) -> np.ndarray:
    """1 s of int16 silence (zeros) at 16 kHz — enough for the POST body
    without inflating the base64 payload."""
    return np.zeros(samples, dtype=np.int16)


# ---------------------------------------------------------------------------
# Constructor behavior
# ---------------------------------------------------------------------------
class TestConstructor:
    def test_empty_base_url_raises(self):
        with pytest.raises(ValueError, match="base_url"):
            RemoteWhisperSTT(base_url="")

    def test_trailing_slash_stripped(self):
        client = RemoteWhisperSTT(base_url="http://pc:9880/", language="en")
        try:
            assert client.base_url == "http://pc:9880"
        finally:
            client.close()

    @pytest.mark.parametrize("lang,expected", [
        ("en", "en"),
        ("EN", "en"),
        ("ja", "ja"),
        (None, None),
        ("", None),
        ("auto", None),
        ("  AUTO  ", None),
    ])
    def test_language_normalization(self, lang, expected):
        client = RemoteWhisperSTT(base_url="http://pc:9880", language=lang)
        try:
            assert client.language == expected
        finally:
            client.close()


# ---------------------------------------------------------------------------
# POST /transcribe — happy path + payload shape
# ---------------------------------------------------------------------------
class TestTranscribePayload:
    def test_request_hits_transcribe_endpoint(self):
        hits = []
        def responder(req: httpx.Request) -> httpx.Response:
            hits.append((req.method, req.url.path))
            return httpx.Response(200, json={"text": "hello world"})
        client = _make_remote(responder)
        try:
            out = client.transcribe(_pcm16(800))
        finally:
            client.close()
        assert out == "hello world"
        assert hits == [("POST", "/transcribe")]

    def test_payload_contains_base64_pcm(self):
        bodies = []
        def responder(req: httpx.Request) -> httpx.Response:
            import json as _json
            bodies.append(_json.loads(req.content))
            return httpx.Response(200, json={"text": "ok"})
        client = _make_remote(responder)
        try:
            pcm = (np.arange(100, dtype=np.int16) * 10)
            client.transcribe(pcm)
        finally:
            client.close()
        body = bodies[0]
        assert "pcm_base64" in body
        assert body["sample_rate"] == 16000
        assert body["beam_size"] == 1
        assert body["vad_filter"] is False
        # Round-trip check: the base64 must decode back to the exact PCM.
        decoded = np.frombuffer(
            base64.b64decode(body["pcm_base64"]), dtype=np.int16,
        )
        assert np.array_equal(decoded, pcm)

    def test_language_forwarded_when_set(self):
        bodies = []
        def responder(req: httpx.Request) -> httpx.Response:
            import json as _json
            bodies.append(_json.loads(req.content))
            return httpx.Response(200, json={"text": ""})
        client = _make_remote(responder, language="ja")
        try:
            client.transcribe(_pcm16(100))
        finally:
            client.close()
        assert bodies[0].get("language") == "ja"

    def test_language_omitted_when_auto(self):
        """None/auto means `let the server decide per utterance` — don't
        send a language field at all so the sidecar falls back to its env
        default."""
        bodies = []
        def responder(req: httpx.Request) -> httpx.Response:
            import json as _json
            bodies.append(_json.loads(req.content))
            return httpx.Response(200, json={"text": ""})
        client = _make_remote(responder, language="auto")
        try:
            client.transcribe(_pcm16(100))
        finally:
            client.close()
        assert "language" not in bodies[0]


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------
class TestTranscribeResponses:
    def test_text_is_stripped(self):
        def responder(req):
            return httpx.Response(200, json={"text": "  spaced out  \n"})
        client = _make_remote(responder)
        try:
            assert client.transcribe(_pcm16(100)) == "spaced out"
        finally:
            client.close()

    def test_empty_text_returns_empty_string(self):
        """Sidecar returning {} or {"text": ""} is 'no speech'.
        Caller decides how to handle (silence fallthrough)."""
        def responder(req):
            return httpx.Response(200, json={"text": ""})
        client = _make_remote(responder)
        try:
            assert client.transcribe(_pcm16(100)) == ""
        finally:
            client.close()

    def test_missing_text_key_returns_empty_string(self):
        def responder(req):
            return httpx.Response(200, json={"other": "field"})
        client = _make_remote(responder)
        try:
            assert client.transcribe(_pcm16(100)) == ""
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
class TestTranscribeErrors:
    def test_non_200_raises_runtime_error_with_detail(self):
        def responder(req):
            return httpx.Response(500, json={"detail": "whisper load failed: cuda oom"})
        client = _make_remote(responder)
        try:
            with pytest.raises(RuntimeError, match="500"):
                client.transcribe(_pcm16(100))
        finally:
            client.close()

    def test_non_200_includes_detail_message(self):
        def responder(req):
            return httpx.Response(400, json={"detail": "empty pcm_base64"})
        client = _make_remote(responder)
        try:
            with pytest.raises(RuntimeError, match="empty pcm_base64"):
                client.transcribe(_pcm16(100))
        finally:
            client.close()

    def test_non_json_error_body_still_raises(self):
        def responder(req):
            return httpx.Response(502, text="Bad Gateway")
        client = _make_remote(responder)
        try:
            with pytest.raises(RuntimeError, match="502"):
                client.transcribe(_pcm16(100))
        finally:
            client.close()

    def test_connect_error_retries_once_then_raises(self):
        """Connect errors get ONE retry — covers the common case of a
        Tailscale peer blipping or the PC briefly unreachable. Both
        attempts failing surfaces as RuntimeError so the bridge's catch
        handler can route it as silence rather than 500ing the turn."""
        attempts = {"n": 0}
        def responder(req):
            attempts["n"] += 1
            raise httpx.ConnectError("connection refused")
        client = _make_remote(responder)
        try:
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                client.transcribe(_pcm16(100))
        finally:
            client.close()
        assert attempts["n"] == 2, "expected exactly one retry, got " + str(attempts["n"])

    def test_connect_error_recovers_on_retry(self):
        """First attempt fails, second succeeds — the common case after
        a Tailscale peer briefly drops. Transcript returns normally."""
        attempts = {"n": 0}
        def responder(req):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ConnectError("temporarily unreachable")
            return httpx.Response(200, json={"text": "recovered"})
        client = _make_remote(responder)
        try:
            out = client.transcribe(_pcm16(100))
        finally:
            client.close()
        assert out == "recovered"
        assert attempts["n"] == 2

    def test_non_connect_http_error_not_retried(self):
        """Protocol errors / timeouts mid-read are NOT retried — a second
        try won't help and would double the latency before the user sees
        a failure."""
        attempts = {"n": 0}
        def responder(req):
            attempts["n"] += 1
            raise httpx.ReadTimeout("read timed out")
        client = _make_remote(responder)
        try:
            with pytest.raises(RuntimeError, match="remote STT request failed"):
                client.transcribe(_pcm16(100))
        finally:
            client.close()
        assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# PCM normalization
# ---------------------------------------------------------------------------
class TestPcmNormalization:
    def test_non_int16_dtype_is_cast(self):
        """Accept float32 / int32 arrays (legacy bridge paths) by casting.
        The local wrapper does the same."""
        bodies = []
        def responder(req):
            import json as _json
            bodies.append(_json.loads(req.content))
            return httpx.Response(200, json={"text": ""})
        client = _make_remote(responder)
        try:
            pcm_f32 = np.array([0.1, -0.2, 0.3], dtype=np.float32)
            client.transcribe(pcm_f32)
        finally:
            client.close()
        decoded = np.frombuffer(
            base64.b64decode(bodies[0]["pcm_base64"]), dtype=np.int16,
        )
        # 3 float samples → cast to int16 → 3 × 2 = 6 bytes → 3 int16.
        assert decoded.shape == (3,)

    def test_multidim_array_flattened(self):
        """Stereo or (1, N) arrays get flattened to 1-D before encoding."""
        bodies = []
        def responder(req):
            import json as _json
            bodies.append(_json.loads(req.content))
            return httpx.Response(200, json={"text": ""})
        client = _make_remote(responder)
        try:
            pcm = np.zeros((2, 100), dtype=np.int16)  # pretend-stereo
            client.transcribe(pcm)
        finally:
            client.close()
        decoded = np.frombuffer(
            base64.b64decode(bodies[0]["pcm_base64"]), dtype=np.int16,
        )
        assert decoded.shape == (200,)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
class TestCloseIdempotent:
    def test_close_can_be_called_twice(self):
        client = RemoteWhisperSTT(base_url="http://pc:9880")
        client.close()
        client.close()  # second call must not raise
