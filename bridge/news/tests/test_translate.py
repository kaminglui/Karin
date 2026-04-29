"""Tests for bridge.news.translate.

Covers:
  - Short-circuit rules (empty input, target == source).
  - Cache round-trip: first call hits LLM, second call serves from
    disk without another HTTP request.
  - Fail-soft: HTTP errors + empty LLM output return the original
    text with translated=False.
  - Output cleanup: common model-prefix noise gets stripped.
  - End-to-end HTTP path using httpx.MockTransport so we exercise
    the real request shape once.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bridge.news.translate import (
    TranslationResult,
    Translator,
    _build_messages,
    _cache_key,
    _clean_output,
)


# --- helpers --------------------------------------------------------------

def _make_translator(
    tmp_path: Path,
    *,
    transport: httpx.MockTransport | None = None,
    model: str = "qwen-test:latest",
) -> Translator:
    """Build a Translator whose cache lives in tmp_path.

    If ``transport`` is provided, monkey-patch httpx.Client so the
    translator's internal client uses it. This lets us exercise the
    actual _call_ollama code path without a live Ollama.
    """
    tr = Translator(
        base_url="http://mock",
        model=model,
        cache_path=tmp_path / "translations.json",
        request_timeout=2.0,
    )
    if transport is not None:
        # Replace the httpx.Client constructor in the translate module's
        # namespace. _call_ollama builds its own client each call, so
        # patching at the module import site covers every call.
        import bridge.news.translate as _t

        real_ctor = _t.httpx.Client

        def _factory(*args, **kwargs):
            kwargs.pop("transport", None)
            return real_ctor(*args, transport=transport, **kwargs)

        tr._httpx_factory = _factory  # type: ignore[attr-defined]
    return tr


def _chat_response(content: str) -> dict:
    """Shape matches Ollama's /api/chat non-stream response."""
    return {
        "model": "qwen-test:latest",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


# --- pure helpers ---------------------------------------------------------

class TestCacheKey:
    def test_stable_across_calls(self):
        # Same inputs -> same key. Trivial but load-bearing: a change
        # to the hashing order would invalidate every cached entry.
        a = _cache_key("hello", "zh", "en")
        b = _cache_key("hello", "zh", "en")
        assert a == b

    def test_differs_by_target_language(self):
        assert _cache_key("hello", "zh", "en") != _cache_key("hello", "ja", "en")

    def test_differs_by_source_language(self):
        # Same text + same target with a different source-language hint
        # is a different cache slot — the prompt sent to the LLM differs.
        assert _cache_key("hello", "zh", "en") != _cache_key("hello", "zh", "auto")


class TestBuildMessages:
    def test_uses_language_names_not_codes(self):
        msgs = _build_messages("hello", "en", "zh")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # Qwen responds to names better than codes; guard the mapping.
        assert "English" in msgs[1]["content"]
        assert "Simplified Chinese" in msgs[1]["content"]
        assert "zh" not in msgs[1]["content"]  # no raw codes leaked

    def test_includes_do_not_answer_guard(self):
        # Inputs that read like questions get answered, not translated,
        # without this guard. Don't let the phrase get dropped.
        msgs = _build_messages("what time is it?", "en", "zh")
        assert "do not answer" in msgs[1]["content"].lower()


class TestCleanOutput:
    def test_strips_sure_heres_prefix(self):
        assert _clean_output("Sure, here is the translation: 你好") == "你好"

    def test_strips_translation_colon_prefix(self):
        assert _clean_output("Translation: 你好世界") == "你好世界"

    def test_strips_surrounding_quotes(self):
        assert _clean_output('"你好"') == "你好"
        assert _clean_output("'hello'") == "hello"
        assert _clean_output("\u201c\u4f60\u597d\u201d") == "\u4f60\u597d"

    def test_plain_translation_passes_through(self):
        assert _clean_output("  你好世界  ") == "你好世界"

    def test_keeps_content_that_starts_with_word_translation(self):
        # "Translation of X is Y" legitimately starts with "Translation"
        # but isn't a prefix pattern. We only strip "translation:" form.
        out = _clean_output("Translation is essential for understanding")
        assert out == "Translation is essential for understanding"


# --- short-circuit behavior ----------------------------------------------

class TestShortCircuit:
    def test_empty_input_no_call(self, tmp_path, monkeypatch):
        tr = _make_translator(tmp_path)
        # Any _call_ollama would blow up — we should never reach it.
        monkeypatch.setattr(
            tr, "_call_ollama",
            lambda *a, **k: pytest.fail("should not call LLM for empty input"),
        )
        r = tr.translate("", "zh", "en")
        assert r == TranslationResult(text="", translated=False, from_cache=False)

    def test_whitespace_only_no_call(self, tmp_path, monkeypatch):
        tr = _make_translator(tmp_path)
        monkeypatch.setattr(
            tr, "_call_ollama",
            lambda *a, **k: pytest.fail("should not call LLM for whitespace"),
        )
        r = tr.translate("   \n\t", "zh", "en")
        assert r.translated is False

    def test_source_equals_target_no_call(self, tmp_path, monkeypatch):
        tr = _make_translator(tmp_path)
        monkeypatch.setattr(
            tr, "_call_ollama",
            lambda *a, **k: pytest.fail("should not call LLM when langs match"),
        )
        r = tr.translate("Already English", "en", "en")
        # translated=False lets callers distinguish "LLM worked" from
        # "already in target". Brief-builder uses this to decide
        # whether to fall back to original.
        assert r.translated is False
        assert r.text == "Already English"


# --- happy path + caching ------------------------------------------------

class TestCache:
    def test_first_call_hits_llm_second_hits_cache(self, tmp_path, monkeypatch):
        tr = _make_translator(tmp_path)
        calls = {"n": 0}

        def _fake_call(messages):
            calls["n"] += 1
            return "你好世界"

        monkeypatch.setattr(tr, "_call_ollama", _fake_call)

        r1 = tr.translate("Hello world", "zh", "en")
        r2 = tr.translate("Hello world", "zh", "en")

        assert r1.text == "你好世界"
        assert r1.translated is True
        assert r1.from_cache is False

        assert r2.text == "你好世界"
        assert r2.translated is True
        assert r2.from_cache is True

        # Exactly one LLM call — second served from cache.
        assert calls["n"] == 1

    def test_cache_persists_to_disk(self, tmp_path, monkeypatch):
        tr = _make_translator(tmp_path)
        monkeypatch.setattr(tr, "_call_ollama", lambda messages: "你好")
        tr.translate("hi", "zh", "en")

        # File must exist and hold the translation so a fresh Translator
        # instance (e.g. after a container restart) picks it up.
        on_disk = json.loads((tmp_path / "translations.json").read_text(encoding="utf-8"))
        assert "你好" in on_disk.values()

        # Fresh instance reads the same cache.
        tr2 = _make_translator(tmp_path)
        r = tr2.translate("hi", "zh", "en")
        assert r.text == "你好"
        assert r.from_cache is True

    def test_corrupted_cache_starts_fresh(self, tmp_path, monkeypatch, caplog):
        cache_path = tmp_path / "translations.json"
        cache_path.write_text("this is not json {{{", encoding="utf-8")
        tr = Translator(
            base_url="http://mock", model="x",
            cache_path=cache_path, request_timeout=2.0,
        )
        monkeypatch.setattr(tr, "_call_ollama", lambda messages: "OK")
        with caplog.at_level("WARNING"):
            r = tr.translate("hi", "zh", "en")
        assert r.translated is True
        # Loud about the bad cache so it's obvious in logs.
        assert any("unreadable" in rec.getMessage() for rec in caplog.records)


# --- fail-soft behavior --------------------------------------------------

class TestFailSoft:
    def test_http_error_returns_original(self, tmp_path, monkeypatch, caplog):
        tr = _make_translator(tmp_path)

        def _boom(messages):
            raise httpx.ConnectError("ollama down")

        monkeypatch.setattr(tr, "_call_ollama", _boom)
        with caplog.at_level("WARNING"):
            r = tr.translate("Hello", "zh", "en")
        assert r.text == "Hello"
        assert r.translated is False
        assert any("translation failed" in rec.getMessage() for rec in caplog.records)

    def test_empty_llm_response_returns_original(self, tmp_path, monkeypatch):
        tr = _make_translator(tmp_path)
        # _call_ollama raises ValueError on empty; translate catches.
        def _empty(messages):
            raise ValueError("Ollama returned empty content")
        monkeypatch.setattr(tr, "_call_ollama", _empty)
        r = tr.translate("Hello", "zh", "en")
        assert r.text == "Hello"
        assert r.translated is False

    def test_cleanup_to_empty_returns_original(self, tmp_path, monkeypatch):
        # Model returns pure noise that cleanup strips to "". Don't
        # poison the cache with an empty entry — return original instead.
        tr = _make_translator(tmp_path)
        monkeypatch.setattr(tr, "_call_ollama", lambda messages: "\"\"")
        r = tr.translate("Hello", "zh", "en")
        assert r.text == "Hello"
        assert r.translated is False


# --- end-to-end HTTP -----------------------------------------------------

class TestHTTPIntegration:
    def test_real_http_shape_through_mock_transport(self, tmp_path, monkeypatch):
        """Exercise _call_ollama against a MockTransport so we verify
        the actual request body shape — not just the translate() wrapper."""
        captured: dict = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response("你好世界"))

        # Patch httpx.Client inside the translate module so _call_ollama
        # uses the MockTransport client. Preserves the method's real
        # request-building logic.
        import bridge.news.translate as _t

        real_client_cls = _t.httpx.Client

        def _client_factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(_handler)
            return real_client_cls(*args, **kwargs)

        monkeypatch.setattr(_t.httpx, "Client", _client_factory)

        tr = Translator(
            base_url="http://mock", model="qwen-test:latest",
            cache_path=tmp_path / "c.json", request_timeout=2.0,
        )
        r = tr.translate("Hello world", "zh", "en")

        # Result flowed through unchanged.
        assert r.text == "你好世界"
        assert r.translated is True
        assert r.from_cache is False

        # Request hit /api/chat with the model + think:False + temp 0.
        assert "/api/chat" in captured["url"]
        body = captured["body"]
        assert body["model"] == "qwen-test:latest"
        assert body["stream"] is False
        assert body["think"] is False  # critical for Qwen3 latency
        assert body["options"]["temperature"] == 0.0
        # System + user message pair made it into the request.
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert "Hello world" in body["messages"][1]["content"]
