"""Tests for bridge.news.newsapi + bridge.alerts.nws_fetch.

Mocked transport — no network. Verifies the fetchers parse upstream
shapes correctly and fail soft on error paths.
"""
from __future__ import annotations

import os
from datetime import datetime

import httpx
import pytest

from bridge.news.newsapi import _to_raw_article, fetch_newsapi
from bridge.news.models import Source, Tier
from bridge.alerts.nws_fetch import (
    fetch_nws_alerts, is_significant, _flatten_feature,
)


# ---- test fixtures -------------------------------------------------------


def _ap_source() -> Source:
    return Source(
        id="associated-press", name="AP", domain="apnews.com",
        tier=Tier.WIRE, ownership_group="ap", is_wire_service=True,
    )


def _reuters_source() -> Source:
    return Source(
        id="reuters", name="Reuters", domain="reuters.com",
        tier=Tier.WIRE, ownership_group="reuters", is_wire_service=True,
    )


# ---- NewsAPI fetcher -----------------------------------------------------


class TestNewsAPIFetcher:
    def test_no_key_returns_empty_without_network(self, monkeypatch):
        """Opt-in by design: if the env key is unset, do NOTHING (no
        network call, no log noise). Silent gracefully."""
        monkeypatch.delenv("KARIN_NEWSAPI_KEY", raising=False)
        # Even with sources present, we should skip entirely.
        sources = {"associated-press": _ap_source()}
        # Pass a client that would explode if called — asserts no HTTP.
        def responder(req):  # pragma: no cover — should never fire
            raise AssertionError(f"unexpected HTTP: {req.url}")
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            result = fetch_newsapi(sources, client=client)
        finally:
            client.close()
        assert result == []

    def test_parses_valid_response(self, monkeypatch):
        monkeypatch.setenv("KARIN_NEWSAPI_KEY", "test-key-123")
        captured_urls: list[str] = []
        def responder(req):
            captured_urls.append(str(req.url))
            return httpx.Response(200, json={
                "status": "ok", "totalResults": 2,
                "articles": [
                    {
                        "source": {"id": "associated-press", "name": "AP"},
                        "title": "Earthquake rocks region",
                        "description": "A 6.3 magnitude quake...",
                        "url": "https://apnews.com/article/abc",
                        "publishedAt": "2026-04-15T10:00:00Z",
                    },
                    {
                        "source": {"id": "reuters", "name": "Reuters"},
                        "title": "Markets open higher",
                        "description": "Futures up on earnings...",
                        "url": "https://reuters.com/markets/xyz",
                        "publishedAt": "2026-04-15T10:05:00Z",
                    },
                ],
            })
        client = httpx.Client(transport=httpx.MockTransport(responder))
        sources = {
            "associated-press": _ap_source(),
            "reuters": _reuters_source(),
        }
        try:
            result = fetch_newsapi(sources, client=client)
        finally:
            client.close()
        assert len(result) == 2
        # Timestamps parsed as tz-aware UTC
        for art in result:
            assert art.published_at.tzinfo is not None
        # URL includes both sources
        assert any("associated-press" in u and "reuters" in u
                   for u in captured_urls)

    def test_skips_articles_from_unknown_sources(self, monkeypatch):
        """If NewsAPI returns an article from a source we haven't
        registered, drop it — we need Source metadata to tier it."""
        monkeypatch.setenv("KARIN_NEWSAPI_KEY", "k")
        def responder(req):
            return httpx.Response(200, json={
                "status": "ok",
                "articles": [
                    {
                        "source": {"id": "some-unknown-site", "name": "?"},
                        "title": "t", "description": "d",
                        "url": "https://x/1",
                        "publishedAt": "2026-04-15T00:00:00Z",
                    },
                ],
            })
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            # Sources registry DOESN'T include 'some-unknown-site'
            result = fetch_newsapi(
                {"associated-press": _ap_source()}, client=client,
            )
        finally:
            client.close()
        assert result == []

    def test_non_200_returns_empty_no_raise(self, monkeypatch):
        monkeypatch.setenv("KARIN_NEWSAPI_KEY", "k")
        def responder(req):
            return httpx.Response(429, text="rate limit")
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            result = fetch_newsapi(
                {"associated-press": _ap_source()}, client=client,
            )
        finally:
            client.close()
        assert result == []

    def test_api_status_error_returns_empty(self, monkeypatch):
        """NewsAPI uses a status:error payload with HTTP 200 for quota
        hits. Must still return []."""
        monkeypatch.setenv("KARIN_NEWSAPI_KEY", "k")
        def responder(req):
            return httpx.Response(200, json={
                "status": "error", "code": "rateLimited",
                "message": "You've exceeded your daily quota",
            })
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            result = fetch_newsapi(
                {"associated-press": _ap_source()}, client=client,
            )
        finally:
            client.close()
        assert result == []

    def test_source_selection_excludes_non_wire_sources(self, monkeypatch):
        """Only ``is_wire_service=True`` sources are sent to NewsAPI.
        RSS-only REPUTABLE sources like BBC flow through the RSS
        fetcher on their own."""
        monkeypatch.setenv("KARIN_NEWSAPI_KEY", "k")
        captured_urls: list[str] = []
        def responder(req):
            captured_urls.append(str(req.url))
            return httpx.Response(200, json={"status": "ok", "articles": []})
        client = httpx.Client(transport=httpx.MockTransport(responder))
        sources = {
            "bbc": Source(id="bbc", name="BBC", domain="bbc.com",
                          tier=Tier.REPUTABLE, ownership_group="bbc",
                          is_wire_service=False),
            "associated-press": _ap_source(),
            "reuters": _reuters_source(),
        }
        try:
            fetch_newsapi(sources, client=client)
        finally:
            client.close()
        assert captured_urls, "no HTTP call made"
        url = captured_urls[0]
        # Wire services ride; BBC does not.
        assert "bbc" not in url
        assert "associated-press" in url
        assert "reuters" in url

    def test_returns_empty_when_no_wire_sources(self, monkeypatch):
        """With a registry containing only non-wire sources, the fetch
        short-circuits without making a network call — there's nothing
        we'd ask NewsAPI about."""
        monkeypatch.setenv("KARIN_NEWSAPI_KEY", "k")
        def responder(req):  # pragma: no cover
            raise AssertionError(f"unexpected HTTP: {req.url}")
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            result = fetch_newsapi(
                {"bbc": Source(id="bbc", name="BBC", domain="bbc.com",
                               tier=Tier.REPUTABLE, ownership_group="bbc",
                               is_wire_service=False)},
                client=client,
            )
        finally:
            client.close()
        assert result == []


class TestToRawArticle:
    def test_drops_article_without_url(self):
        art = {
            "source": {"id": "associated-press"},
            "title": "t", "description": "d", "url": "",
            "publishedAt": "2026-04-15T00:00:00Z",
        }
        assert _to_raw_article(art, {"associated-press": _ap_source()}) is None

    def test_drops_article_without_title(self):
        art = {
            "source": {"id": "associated-press"},
            "title": "", "description": "d", "url": "https://x",
            "publishedAt": "2026-04-15T00:00:00Z",
        }
        assert _to_raw_article(art, {"associated-press": _ap_source()}) is None

    def test_malformed_timestamp_defaults_to_now(self):
        art = {
            "source": {"id": "associated-press"},
            "title": "t", "description": "d", "url": "https://x",
            "publishedAt": "not-a-date",
        }
        raw = _to_raw_article(art, {"associated-press": _ap_source()})
        assert raw is not None
        # Defaulted "now" -> timezone-aware and recent
        assert raw.published_at.tzinfo is not None


# ---- NWS fetcher ---------------------------------------------------------


class TestNWSFetcher:
    def _nws_feature(self, alert_id: str, severity: str, event: str) -> dict:
        """Shape one CAP feature like api.weather.gov returns."""
        return {
            "id": alert_id,
            "properties": {
                "id": alert_id,
                "event": event,
                "severity": severity,
                "certainty": "Observed",
                "urgency": "Expected",
                "headline": f"{event} issued for test area",
                "description": "Full CAP description goes here...",
                "areaDesc": "Test County, PA",
                "onset": "2026-04-15T12:00:00-04:00",
                "expires": "2026-04-15T18:00:00-04:00",
                "senderName": "NWS State College PA",
            },
        }

    def test_flatten_feature_happy_path(self):
        feat = self._nws_feature("urn:oid:1", "Severe", "Winter Storm Warning")
        flat = _flatten_feature(feat)
        assert flat is not None
        assert flat["id"] == "urn:oid:1"
        assert flat["event"] == "Winter Storm Warning"
        assert flat["severity"] == "Severe"
        assert "Test County" in flat["area_desc"]

    def test_flatten_feature_drops_missing_id_or_event(self):
        assert _flatten_feature({"properties": {"event": "x"}}) is None
        assert _flatten_feature({"properties": {"id": "x"}}) is None
        assert _flatten_feature({"properties": {}}) is None

    def test_is_significant(self):
        assert is_significant({"severity": "Moderate"})
        assert is_significant({"severity": "Severe"})
        assert is_significant({"severity": "Extreme"})
        assert not is_significant({"severity": "Minor"})
        assert not is_significant({"severity": "Unknown"})
        assert not is_significant({"severity": None})

    def test_fetch_happy_path(self):
        feature = self._nws_feature("a1", "Severe", "Flash Flood Warning")
        def responder(req):
            assert "point=40" in str(req.url)
            return httpx.Response(200, json={"features": [feature]})
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            out = fetch_nws_alerts(40.7934, -77.8600, client=client)
        finally:
            client.close()
        assert len(out) == 1
        assert out[0]["event"] == "Flash Flood Warning"

    def test_fetch_non_200_returns_empty(self):
        def responder(req):
            return httpx.Response(500, text="server error")
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            assert fetch_nws_alerts(40.0, -77.0, client=client) == []
        finally:
            client.close()

    def test_fetch_network_error_returns_empty(self):
        def responder(req):
            raise httpx.ConnectError("no route to host")
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            assert fetch_nws_alerts(40.0, -77.0, client=client) == []
        finally:
            client.close()

    def test_fetch_bad_json_returns_empty(self):
        def responder(req):
            return httpx.Response(
                200, content=b"not json",
                headers={"content-type": "text/plain"},
            )
        client = httpx.Client(transport=httpx.MockTransport(responder))
        try:
            assert fetch_nws_alerts(40.0, -77.0, client=client) == []
        finally:
            client.close()
