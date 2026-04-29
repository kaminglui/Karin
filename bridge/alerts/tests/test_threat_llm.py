"""Tests for bridge.alerts.threat_llm.ThreatVerifier.

Covers the hallucination guardrails that were promised for Phase G.b:

  - ±1 tier hard clamp on LLM output (stops big jumps)
  - Citation-required (no citation = fall back to rule score)
  - Bad/empty JSON -> rule score
  - HTTP errors -> rule score
  - Cache hit/miss + 7-day TTL expiry
  - Empty cluster_id short-circuits (can't cache -> skip)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from bridge.alerts.threat_llm import ThreatVerifier, _extract_json
from bridge.alerts.user_context import UserContext


_CTX = UserContext(
    city="University Park", region="Pennsylvania", country="United States",
    latitude=40.79, longitude=-77.86,
)


def _payload(**kw):
    base = dict(
        cluster_id="c1",
        headline="Pennsylvania flash flood warning upgraded",
        watchlist_label="Disasters",
        watchlist_type="event",
        cluster_state="confirmed",
    )
    base.update(kw)
    return base


def _mk_verifier(tmp_path):
    return ThreatVerifier(
        base_url="http://ollama.test",
        model="qwen3:4b",
        cache_path=tmp_path / "threat_decisions.json",
        request_timeout=5.0,
    )


def _mock_ollama_response(content: str):
    """Build a mock httpx.Client context-manager whose post() returns a
    JSON body with {"message": {"content": content}}. Matches the shape
    ThreatVerifier._call_ollama expects."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"message": {"content": content}}
    client = MagicMock()
    client.post.return_value = resp
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


# --- JSON extraction ----------------------------------------------------

class TestExtractJson:
    def test_strict_parse(self):
        assert _extract_json('{"score": 3, "citation": "x"}') == {
            "score": 3, "citation": "x",
        }

    def test_code_fence_stripped(self):
        raw = '```json\n{"score": 2, "citation": "y"}\n```'
        assert _extract_json(raw) == {"score": 2, "citation": "y"}

    def test_embedded_json(self):
        raw = 'sure here:\n{"score": 1, "citation": "z"}\n(done)'
        out = _extract_json(raw)
        assert out == {"score": 1, "citation": "z"}

    def test_empty_returns_none(self):
        assert _extract_json("") is None

    def test_garbage_returns_none(self):
        assert _extract_json("the sky is blue") is None

    def test_non_dict_returns_none(self):
        assert _extract_json("[1, 2, 3]") is None


# --- clamp --------------------------------------------------------------

class TestClamp:
    def test_within_band_passes_through(self):
        # rule=3, LLM=3 stays 3
        assert ThreatVerifier._clamp(3, 3) == 3

    def test_upper_clamp(self):
        # LLM screams 4 but rule is 2 — clamp to 3 (rule+1)
        assert ThreatVerifier._clamp(4, 2) == 3

    def test_lower_clamp(self):
        # LLM says 0 but rule is 3 — clamp to 2 (rule-1)
        assert ThreatVerifier._clamp(0, 3) == 2

    def test_hallucinated_high_collapses(self):
        # LLM returns 9, rule is 2 — clamp to 3
        assert ThreatVerifier._clamp(9, 2) == 3

    def test_hallucinated_negative_collapses(self):
        assert ThreatVerifier._clamp(-5, 3) == 2

    def test_boundary_zero(self):
        # rule=0, LLM=0 -> band is [0, 1], stays 0
        assert ThreatVerifier._clamp(0, 0) == 0

    def test_boundary_four(self):
        assert ThreatVerifier._clamp(4, 4) == 4


# --- verify() pathways --------------------------------------------------

class TestVerifyHappyPath:
    def test_llm_agrees_with_rule(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": 3, "citation": "flash flood warning upgraded"}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            out = v.verify(_payload(), rule_score=3, ctx=_CTX)
        assert out == 3

    def test_llm_upgrades_one_tier(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": 3, "citation": "state of emergency declared"}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            out = v.verify(_payload(), rule_score=2, ctx=_CTX)
        assert out == 3  # allowed: rule+1

    def test_llm_tries_to_jump_two_gets_clamped(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": 4, "citation": "imminent danger"}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            out = v.verify(_payload(), rule_score=2, ctx=_CTX)
        assert out == 3  # clamped from 4 back to rule+1


class TestVerifyGuardrails:
    def test_no_cluster_id_short_circuits(self, tmp_path):
        v = _mk_verifier(tmp_path)
        # Client should never be called when cluster_id is missing.
        with patch("bridge.alerts.threat_llm.httpx.Client") as mock_cls:
            out = v.verify(_payload(cluster_id=""), rule_score=2, ctx=_CTX)
            assert mock_cls.call_count == 0
        assert out == 2

    def test_missing_citation_falls_back(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": 4, "citation": ""}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            out = v.verify(_payload(), rule_score=2, ctx=_CTX)
        assert out == 2  # no citation -> rule score unchanged

    def test_bad_json_falls_back(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response("not even JSON")
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            out = v.verify(_payload(), rule_score=3, ctx=_CTX)
        assert out == 3

    def test_http_error_falls_back(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("boom")
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            out = v.verify(_payload(), rule_score=2, ctx=_CTX)
        assert out == 2

    def test_non_int_score_falls_back(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": "three", "citation": "x"}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            out = v.verify(_payload(), rule_score=2, ctx=_CTX)
        assert out == 2


class TestVerifyCache:
    def test_second_call_hits_cache(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": 3, "citation": "upgraded"}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client) as mock_cls:
            first = v.verify(_payload(), rule_score=2, ctx=_CTX)
            second = v.verify(_payload(), rule_score=2, ctx=_CTX)
            # Client constructed exactly once — second call served
            # from the in-memory cache.
            assert mock_cls.call_count == 1
        assert first == 3
        assert second == 3

    def test_cache_persisted_to_disk(self, tmp_path):
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": 3, "citation": "upgraded"}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            v.verify(_payload(), rule_score=2, ctx=_CTX)
        cache_path = tmp_path / "threat_decisions.json"
        assert cache_path.exists()
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        assert len(raw) == 1
        entry = next(iter(raw.values()))
        assert entry["score"] == 3

    def test_stale_cache_refetches(self, tmp_path):
        # Pre-write a cache entry stamped 10 days ago (past 7-day TTL).
        cache_path = tmp_path / "threat_decisions.json"
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        cache_path.write_text(json.dumps({
            "c1:deadbeef": {"score": 2, "saved_at": old_ts},
        }), encoding="utf-8")
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": 3, "citation": "worsening"}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client) as mock_cls:
            out = v.verify(_payload(), rule_score=3, ctx=_CTX)
            # Stale -> LLM is called despite cache file existing.
            assert mock_cls.call_count == 1
        assert out == 3

    def test_corrupt_cache_starts_fresh(self, tmp_path):
        # Garbage bytes should not raise; verifier just rebuilds cache.
        cache_path = tmp_path / "threat_decisions.json"
        cache_path.write_text("not even json {", encoding="utf-8")
        v = _mk_verifier(tmp_path)
        mock_client = _mock_ollama_response(
            '{"score": 2, "citation": "contained"}'
        )
        with patch("bridge.alerts.threat_llm.httpx.Client",
                   return_value=mock_client):
            out = v.verify(_payload(), rule_score=2, ctx=_CTX)
        assert out == 2
