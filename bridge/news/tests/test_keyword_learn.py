"""Tests for the Phase E LLM-assisted keyword learning module.

Mocks httpx.Client so we can pin the request shape + exercise the
defensive JSON-parsing + cleanup paths without a live Ollama.
"""
from __future__ import annotations

import json

import httpx
import pytest

from bridge.news.keyword_learn import (
    LearnedEntity,
    _build_corpus,
    _coerce_entity,
    _extract_json_array,
    extract_entities,
)


def _chat_response(content: str) -> dict:
    return {
        "model": "qwen-test",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


def _with_mock_httpx(monkeypatch, handler):
    """Install a MockTransport-backed httpx.Client so _call_ollama
    goes through our handler instead of the network."""
    import bridge.news.keyword_learn as _kl
    real_ctor = _kl.httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_ctor(*args, **kwargs)

    monkeypatch.setattr(_kl.httpx, "Client", _factory)


# --- pure helpers ---------------------------------------------------------

class TestBuildCorpus:
    def test_joins_fragments_one_per_line(self):
        out = _build_corpus(["first story", "second story", "third"])
        assert out == "first story\nsecond story\nthird"

    def test_strips_blank_and_whitespace(self):
        assert _build_corpus(["  hello  ", "", "\n\t", "world"]) == "hello\nworld"

    def test_clips_at_char_budget(self):
        # 10 "aaaa...a" fragments of 1000 chars each -> budget is 8000 so
        # only 7-8 fit cleanly. Guard dropping the tail without mid-
        # fragment truncation.
        fragments = ["x" * 1000 for _ in range(10)]
        out = _build_corpus(fragments)
        assert len(out) <= 8000
        # No partial fragment survived — every line should be exactly
        # 1000 chars (no truncation mid-line).
        for line in out.splitlines():
            assert len(line) == 1000


class TestExtractJsonArray:
    def test_strict_parse_works(self):
        assert _extract_json_array('[{"label":"A"}, {"label":"B"}]') == [
            {"label": "A"}, {"label": "B"},
        ]

    def test_strips_markdown_fence(self):
        raw = "```json\n[{\"label\": \"A\"}]\n```"
        assert _extract_json_array(raw) == [{"label": "A"}]

    def test_greedy_slice_when_preamble(self):
        raw = "Here are the entities:\n[{\"label\": \"A\"}]\nHope this helps!"
        assert _extract_json_array(raw) == [{"label": "A"}]

    def test_bare_string_items_get_rescued(self):
        assert _extract_json_array('["ASML", "Nvidia"]') == [
            {"label": "ASML", "kind": "other", "confidence": 50},
            {"label": "Nvidia", "kind": "other", "confidence": 50},
        ]

    def test_non_array_raises(self):
        with pytest.raises(ValueError):
            _extract_json_array('{"not": "an array"}')


class TestCoerceEntity:
    def test_well_formed_passes_through(self):
        e = _coerce_entity({"label": "ASML", "kind": "organization", "confidence": 82})
        assert e == LearnedEntity(label="ASML", kind="organization", confidence=82)

    def test_unknown_kind_becomes_other(self):
        e = _coerce_entity({"label": "Thing", "kind": "widget", "confidence": 50})
        assert e.kind == "other"

    def test_confidence_clamped(self):
        e = _coerce_entity({"label": "X", "kind": "person", "confidence": 200})
        assert e.confidence == 100
        e = _coerce_entity({"label": "Y", "kind": "person", "confidence": -10})
        assert e.confidence == 0

    def test_non_int_confidence_falls_to_zero(self):
        e = _coerce_entity({"label": "Z", "kind": "event", "confidence": "high"})
        assert e.confidence == 0

    def test_empty_label_rejected(self):
        assert _coerce_entity({"label": "  ", "kind": "other"}) is None
        assert _coerce_entity({}) is None

    def test_whitespace_collapsed(self):
        e = _coerce_entity({"label": "Foo   Bar\nBaz", "kind": "place"})
        assert e.label == "Foo Bar Baz"


# --- extract_entities (full flow with mocked Ollama) ---------------------

class TestExtractEntities:
    def test_happy_path(self, monkeypatch):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_chat_response(
                '[{"label": "ASML", "kind": "organization", "confidence": 90},'
                ' {"label": "Nvidia", "kind": "organization", "confidence": 85}]'
            ))

        _with_mock_httpx(monkeypatch, handler)
        out = extract_entities(
            ["ASML boosts forecast", "Nvidia chip export"],
            bucket_label="AI / Tech",
            base_url="http://mock", model="qwen-test",
            request_timeout=2.0,
        )
        assert [e.label for e in out] == ["ASML", "Nvidia"]
        # Critical request shape guards:
        assert captured["body"]["think"] is False
        assert captured["body"]["options"]["temperature"] == 0.0
        assert "AI / Tech" in captured["body"]["messages"][1]["content"]

    def test_dedup_case_insensitive(self, monkeypatch):
        # Model returns the same entity twice with different casing.
        # Caller should only see one.
        _with_mock_httpx(monkeypatch, lambda req: httpx.Response(
            200, json=_chat_response(
                '[{"label": "ASML", "kind": "organization"},'
                ' {"label": "asml", "kind": "organization"}]'
            ),
        ))
        out = extract_entities(
            ["ASML news"], "US",
            base_url="http://mock", model="qwen-test",
        )
        assert len(out) == 1

    def test_http_error_returns_empty(self, monkeypatch):
        _with_mock_httpx(monkeypatch, lambda req: httpx.Response(500))
        out = extract_entities(
            ["something"], "US",
            base_url="http://mock", model="qwen-test",
        )
        assert out == []

    def test_empty_corpus_no_llm_call(self, monkeypatch):
        # Passing nothing should short-circuit before any HTTP call.
        def _boom(request):
            pytest.fail("should not have called the LLM")
        _with_mock_httpx(monkeypatch, _boom)
        assert extract_entities(
            ["", "   "], "US",
            base_url="http://mock", model="qwen-test",
        ) == []

    def test_bad_json_returns_empty(self, monkeypatch):
        _with_mock_httpx(monkeypatch, lambda req: httpx.Response(
            200, json=_chat_response("Sorry, no entities found today."),
        ))
        assert extract_entities(
            ["x"], "US",
            base_url="http://mock", model="qwen-test",
        ) == []
