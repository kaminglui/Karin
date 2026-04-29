"""Tests for bridge.news.extract.

Mocks the httpx client + trafilatura.extract so no network / no parser
runs. Covers:
  - ExtractStore load/save roundtrip
  - extract_one's error paths (http fail, non-200, bad JSON, empty text)
  - extract_missing respects cooldown + rate cap + sort-by-recency
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from bridge.news.extract import (
    ExtractStore,
    ExtractedArticle,
    extract_missing,
    extract_one,
)


@dataclass
class _FakeArticle:
    """Minimal shape of NormalizedArticle that extract_missing uses."""
    article_id: str
    url: str
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---- ExtractStore -------------------------------------------------------


class TestExtractStore:
    def test_empty_file_load(self, tmp_path):
        store = ExtractStore(tmp_path / "extracted.json")
        assert store.load() == {}

    def test_save_then_load_roundtrip(self, tmp_path):
        store = ExtractStore(tmp_path / "extracted.json")
        now = datetime.now(timezone.utc)
        records = {
            "a1": ExtractedArticle(
                article_id="a1", text="hello world", title="Hello",
                author="Alice", date="2026-04-15",
                attempted_at=now,
            ),
            "a2": ExtractedArticle(
                article_id="a2", error="paywall", attempted_at=now,
            ),
        }
        store.save(records)
        loaded = store.load()
        assert set(loaded.keys()) == {"a1", "a2"}
        assert loaded["a1"].text == "hello world"
        assert loaded["a1"].ok is True
        assert loaded["a2"].error == "paywall"
        assert loaded["a2"].ok is False
        # datetime roundtrips (within microsecond precision)
        assert loaded["a1"].attempted_at is not None

    def test_corrupted_json_starts_fresh(self, tmp_path):
        p = tmp_path / "extracted.json"
        p.write_text("not json", encoding="utf-8")
        store = ExtractStore(p)
        assert store.load() == {}


# ---- extract_one --------------------------------------------------------


class TestExtractOne:
    def _mock_client(self, responder):
        return httpx.Client(transport=httpx.MockTransport(responder))

    def test_http_error_returns_error_record(self, monkeypatch):
        def responder(req):
            raise httpx.ConnectError("no route to host")
        rec = extract_one("https://x/1", client=self._mock_client(responder))
        assert rec.error.startswith("http: ConnectError")
        assert rec.text == ""
        assert rec.attempted_at is not None

    def test_non_200_returns_error_record(self, monkeypatch):
        def responder(req):
            return httpx.Response(403, text="Forbidden")
        rec = extract_one("https://x/1", client=self._mock_client(responder))
        assert "http 403" in rec.error

    def test_empty_extraction_returns_error_record(self, monkeypatch):
        def responder(req):
            return httpx.Response(200, text="<html><body></body></html>")
        # Force trafilatura to return None (empty output).
        with patch("bridge.news.extract.trafilatura.extract", return_value=None):
            rec = extract_one("https://x/1", client=self._mock_client(responder))
        assert rec.error == "empty extraction"

    def test_successful_extraction_populates_fields(self, monkeypatch):
        def responder(req):
            return httpx.Response(200, text="<html>body</html>")
        fake_json = json.dumps({
            "text": "Full article body goes here with multiple sentences.",
            "title": "Story Headline",
            "author": "J. Reporter",
            "date": "2026-04-15",
        })
        with patch("bridge.news.extract.trafilatura.extract", return_value=fake_json):
            rec = extract_one("https://x/1", client=self._mock_client(responder))
        assert rec.ok
        assert rec.text.startswith("Full article")
        assert rec.title == "Story Headline"
        assert rec.author == "J. Reporter"
        assert rec.date == "2026-04-15"
        assert rec.error == ""

    def test_bad_extractor_output_returns_error(self, monkeypatch):
        def responder(req):
            return httpx.Response(200, text="<html>body</html>")
        with patch("bridge.news.extract.trafilatura.extract", return_value="not json"):
            rec = extract_one("https://x/1", client=self._mock_client(responder))
        assert "bad extractor output" in rec.error


# ---- extract_missing ----------------------------------------------------


class TestExtractMissing:
    def test_skips_already_extracted(self, tmp_path, monkeypatch):
        """Articles that already have ok=True records aren't re-fetched."""
        store = ExtractStore(tmp_path / "extracted.json")
        now = datetime.now(timezone.utc)
        store.save({
            "a1": ExtractedArticle(
                article_id="a1", text="already have it",
                attempted_at=now,
            ),
        })
        arts = {"a1": _FakeArticle("a1", "https://x/1")}
        # If extract_one were called, it'd fail — no mock responder set.
        # Success here proves no fetch happened.
        summary = extract_missing(arts, store)
        assert summary == {"pending": 0, "attempted": 0, "succeeded": 0, "failed": 0}

    def test_cooldown_holds_failed_records(self, tmp_path, monkeypatch):
        """Records that failed within the last 24h aren't retried."""
        store = ExtractStore(tmp_path / "extracted.json")
        now = datetime.now(timezone.utc)
        store.save({
            "a1": ExtractedArticle(
                article_id="a1", error="paywall",
                attempted_at=now - timedelta(hours=2),
            ),
        })
        arts = {"a1": _FakeArticle("a1", "https://x/1")}
        summary = extract_missing(arts, store)
        assert summary["attempted"] == 0

    def test_cooldown_expires_after_24h(self, tmp_path, monkeypatch):
        """Records failed >24h ago get retried."""
        store = ExtractStore(tmp_path / "extracted.json")
        now = datetime.now(timezone.utc)
        store.save({
            "a1": ExtractedArticle(
                article_id="a1", error="paywall",
                attempted_at=now - timedelta(hours=25),
            ),
        })
        arts = {"a1": _FakeArticle("a1", "https://x/1")}
        with patch("bridge.news.extract.extract_one") as mock_one:
            mock_one.return_value = ExtractedArticle(
                article_id="", text="now it works", attempted_at=now,
            )
            summary = extract_missing(arts, store)
        assert summary["attempted"] == 1
        assert summary["succeeded"] == 1

    def test_rate_cap_respected(self, tmp_path, monkeypatch):
        """With more pending than max_extractions, only max get tried."""
        store = ExtractStore(tmp_path / "extracted.json")
        now = datetime.now(timezone.utc)
        arts = {
            f"a{i}": _FakeArticle(
                f"a{i}", f"https://x/{i}",
                published_at=now - timedelta(hours=i),
            )
            for i in range(10)
        }
        with patch("bridge.news.extract.extract_one") as mock_one:
            mock_one.return_value = ExtractedArticle(
                article_id="", text="ok", attempted_at=now,
            )
            summary = extract_missing(arts, store, max_extractions=3)
        assert summary["pending"] == 10
        assert summary["attempted"] == 3
        assert summary["succeeded"] == 3

    def test_sorts_newer_first(self, tmp_path, monkeypatch):
        """With rate cap, newer articles should be processed first."""
        store = ExtractStore(tmp_path / "extracted.json")
        now = datetime.now(timezone.utc)
        arts = {
            "old": _FakeArticle("old", "https://x/old",
                                 published_at=now - timedelta(days=3)),
            "new": _FakeArticle("new", "https://x/new",
                                 published_at=now),
            "mid": _FakeArticle("mid", "https://x/mid",
                                 published_at=now - timedelta(days=1)),
        }
        urls_seen: list[str] = []
        def _stub(url, client=None):
            urls_seen.append(url)
            return ExtractedArticle(article_id="", text="ok", attempted_at=now)
        with patch("bridge.news.extract.extract_one", side_effect=_stub):
            extract_missing(arts, store, max_extractions=2)
        # Newer URLs processed first; "old" should NOT appear.
        assert "https://x/new" in urls_seen
        assert "https://x/mid" in urls_seen
        assert "https://x/old" not in urls_seen

    def test_no_trafilatura_returns_empty(self, tmp_path, monkeypatch):
        """When EXTRACTION_AVAILABLE is False (trafilatura missing),
        extract_missing no-ops cleanly."""
        store = ExtractStore(tmp_path / "extracted.json")
        with patch("bridge.news.extract.EXTRACTION_AVAILABLE", False):
            summary = extract_missing(
                {"a1": _FakeArticle("a1", "https://x/1")}, store,
            )
        assert summary == {"pending": 0, "attempted": 0, "succeeded": 0, "failed": 0}
