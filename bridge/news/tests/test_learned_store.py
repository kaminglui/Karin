"""Tests for the Phase E learned-keyword store.

Covers:
  - round-trip save/load preserves all fields
  - first-time vs repeat sightings (count++, last_seen bumped)
  - TTL sweep drops stale rows + prunes empty buckets
  - to_ui_payload sort order (count desc, then recency)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bridge.news.keyword_learn import LearnedEntity
from bridge.news.learned_store import (
    BUCKET_KEYS,
    LearnedStore,
    record,
    sweep_expired,
    to_ui_payload,
)


_NOW = datetime(2026, 4, 16, 12, 0, 0, tzinfo=timezone.utc)


class TestRecord:
    def test_first_sighting_adds_row(self):
        data = {k: {} for k in BUCKET_KEYS}
        added = record(
            data, "regions", "US",
            [LearnedEntity(label="ASML", kind="organization", confidence=85)],
            now=_NOW,
        )
        assert added == 1
        row = data["regions"]["US"]["asml"]
        assert row.label == "ASML"
        assert row.count == 1
        assert row.first_seen == _NOW
        assert row.last_seen == _NOW

    def test_repeat_sighting_increments_count(self):
        data = {k: {} for k in BUCKET_KEYS}
        record(data, "regions", "US",
               [LearnedEntity("ASML", "organization", 80)], now=_NOW)
        later = _NOW + timedelta(hours=6)
        added = record(data, "regions", "US",
                       [LearnedEntity("asml", "organization", 90)], now=later)
        assert added == 0   # no NEW rows
        row = data["regions"]["US"]["asml"]
        assert row.count == 2
        assert row.last_seen == later
        assert row.first_seen == _NOW
        # Higher confidence kept.
        assert row.confidence == 90

    def test_unknown_bucket_raises(self):
        data = {k: {} for k in BUCKET_KEYS}
        try:
            record(data, "not_a_bucket", "US", [LearnedEntity("X", "other")], _NOW)
        except ValueError as e:
            assert "unknown bucket" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestSweep:
    def test_drops_stale_rows(self):
        data = {k: {} for k in BUCKET_KEYS}
        fresh = _NOW - timedelta(days=5)
        stale = _NOW - timedelta(days=60)
        record(data, "regions", "US",
               [LearnedEntity("Fresh", "other")], now=fresh)
        record(data, "regions", "US",
               [LearnedEntity("Stale", "other")], now=stale)
        dropped = sweep_expired(data, now=_NOW, ttl_days=30)
        assert dropped == 1
        assert "fresh" in data["regions"]["US"]
        assert "stale" not in data["regions"]["US"]

    def test_prunes_empty_labels(self):
        data = {k: {} for k in BUCKET_KEYS}
        stale = _NOW - timedelta(days=60)
        record(data, "regions", "US",
               [LearnedEntity("AllStale", "other")], now=stale)
        sweep_expired(data, now=_NOW, ttl_days=30)
        assert "US" not in data["regions"]   # empty label removed


class TestRoundTrip:
    def test_save_load_identity(self, tmp_path):
        store = LearnedStore(tmp_path / "learned.json")
        data = {k: {} for k in BUCKET_KEYS}
        record(data, "regions", "US",
               [LearnedEntity("ASML", "organization", 82)], now=_NOW)
        record(data, "topics", "AI / Tech",
               [LearnedEntity("GPT", "other", 50)], now=_NOW)
        store.save(data)
        loaded = store.load()
        assert loaded["regions"]["US"]["asml"].label == "ASML"
        assert loaded["regions"]["US"]["asml"].count == 1
        assert loaded["topics"]["AI / Tech"]["gpt"].kind == "other"

    def test_missing_file_returns_empty_buckets(self, tmp_path):
        store = LearnedStore(tmp_path / "nope.json")
        out = store.load()
        for k in BUCKET_KEYS:
            assert k in out
            assert out[k] == {}


class TestAnalyzedTracking:
    """Sidecar: which cluster_ids the LLM has already learned from.
    The service uses this to skip buckets with no new clusters since
    the last learning pass — avoids re-paying the LLM for data we've
    already distilled."""

    def test_empty_file_returns_empty_sets(self, tmp_path):
        store = LearnedStore(tmp_path / "learned.json")
        out = store.load_analyzed()
        for k in BUCKET_KEYS:
            assert out[k] == {}

    def test_save_and_reload_analyzed(self, tmp_path):
        store = LearnedStore(tmp_path / "learned.json")
        data = {k: {} for k in BUCKET_KEYS}
        # Attach a tiny analyzed sidecar to a save + read it back.
        analyzed = {
            "regions": {"US": {"cid1", "cid2"}},
            "topics": {},
            "events": {},
        }
        store.save(data, analyzed=analyzed)
        reloaded = store.load_analyzed()
        assert reloaded["regions"]["US"] == {"cid1", "cid2"}
        assert reloaded["topics"] == {}

    def test_save_without_analyzed_preserves_sidecar(self, tmp_path):
        # Calling save() without the analyzed kwarg must not wipe the
        # sidecar on disk — otherwise every entity-only save would
        # forget the learning history.
        store = LearnedStore(tmp_path / "learned.json")
        data = {k: {} for k in BUCKET_KEYS}
        analyzed = {"regions": {"US": {"cid1"}}, "topics": {}, "events": {}}
        store.save(data, analyzed=analyzed)
        # Second save without analyzed kwarg — should keep "cid1".
        store.save(data)
        assert store.load_analyzed()["regions"]["US"] == {"cid1"}


class TestUiPayload:
    def test_sort_by_count_desc_then_recency(self):
        data = {k: {} for k in BUCKET_KEYS}
        # Two entries with different counts.
        record(data, "regions", "US",
               [LearnedEntity("A", "other")], now=_NOW - timedelta(hours=2))
        record(data, "regions", "US",
               [LearnedEntity("A", "other")], now=_NOW)   # count=2
        record(data, "regions", "US",
               [LearnedEntity("B", "other")], now=_NOW)   # count=1
        payload = to_ui_payload(data)
        group = payload["regions"][0]
        assert group["watchlist_label"] == "US"
        labels = [e["label"] for e in group["entities"]]
        assert labels[0] == "A"   # higher count first
        assert labels[1] == "B"

    def test_empty_buckets_still_present(self):
        data = {k: {} for k in BUCKET_KEYS}
        payload = to_ui_payload(data)
        assert payload == {"regions": [], "topics": [], "events": []}
