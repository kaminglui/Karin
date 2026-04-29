"""Tests for bridge.trackers.store.

Covers TrackerStore persistence round-trip, append-only event log, and
the two history helpers (add_or_replace_reading, prune_history).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bridge.trackers.models import TrackerReading, TrackerRecord
from bridge.trackers.store import (
    TrackerStore,
    add_or_replace_reading,
    prune_history,
)


_EPOCH = datetime(2026, 4, 1, tzinfo=timezone.utc)


def _reading(day_offset: float, value: float) -> TrackerReading:
    return TrackerReading(
        timestamp=_EPOCH + timedelta(days=day_offset),
        value=value,
    )


# --- TrackerStore round-trip ----------------------------------------------

class TestStoreRoundTrip:
    def test_empty_file_loads_empty_dict(self, tmp_path):
        store = TrackerStore(tmp_path)
        assert store.load() == {}

    def test_save_and_load_preserves_structure(self, tmp_path):
        store = TrackerStore(tmp_path)
        record = TrackerRecord(
            id="usd_cny", label="USD/CNY", category="fx",
            history=[_reading(0, 7.23), _reading(1, 7.24)],
            last_fetched_at=_EPOCH + timedelta(days=2),
            last_fetch_error=None,
        )
        store.save({"usd_cny": record})
        loaded = store.load()
        assert set(loaded.keys()) == {"usd_cny"}
        got = loaded["usd_cny"]
        assert got.label == "USD/CNY"
        assert got.category == "fx"
        assert len(got.history) == 2
        assert got.history[0].value == 7.23
        assert got.history[1].value == 7.24
        assert got.last_fetched_at == _EPOCH + timedelta(days=2)
        assert got.last_fetch_error is None

    def test_last_fetch_error_persists(self, tmp_path):
        store = TrackerStore(tmp_path)
        record = TrackerRecord(
            id="t1", label="T1", category="fx",
            history=[], last_fetched_at=_EPOCH,
            last_fetch_error="frankfurter: status 503",
        )
        store.save({"t1": record})
        got = store.load()["t1"]
        assert got.last_fetch_error == "frankfurter: status 503"

    def test_event_log_is_append_only(self, tmp_path):
        store = TrackerStore(tmp_path)
        store.append_event("tracker_fetch_ok", {"id": "t1", "value": 1.0})
        store.append_event("tracker_fetch_error", {"id": "t1", "error": "boom"})
        contents = store.events_path.read_text(encoding="utf-8").splitlines()
        assert len(contents) == 2
        assert "tracker_fetch_ok" in contents[0]
        assert "tracker_fetch_error" in contents[1]


# --- add_or_replace_reading -----------------------------------------------

class TestAddOrReplace:
    def test_new_timestamp_appends(self):
        record = TrackerRecord(id="t", label="T", category="fx",
                               history=[_reading(0, 7.2)])
        add_or_replace_reading(record, _reading(1, 7.3))
        assert len(record.history) == 2
        assert record.history[-1].value == 7.3

    def test_same_timestamp_replaces(self):
        record = TrackerRecord(id="t", label="T", category="fx",
                               history=[_reading(0, 7.2)])
        add_or_replace_reading(record, _reading(0, 7.25))  # revised value
        assert len(record.history) == 1
        assert record.history[0].value == 7.25

    def test_history_stays_sorted_ascending(self):
        record = TrackerRecord(id="t", label="T", category="fx", history=[])
        # Insert out of order.
        add_or_replace_reading(record, _reading(5, 7.5))
        add_or_replace_reading(record, _reading(1, 7.1))
        add_or_replace_reading(record, _reading(3, 7.3))
        values = [r.value for r in record.history]
        assert values == [7.1, 7.3, 7.5]


# --- prune_history ---------------------------------------------------------

class TestPrune:
    def test_drops_older_than_threshold(self):
        # Build a record with 30 readings, 1 day apart, ending "today".
        now = _EPOCH + timedelta(days=29)
        record = TrackerRecord(
            id="t", label="T", category="fx",
            history=[_reading(i, i * 0.01) for i in range(30)],
        )
        prune_history(record, history_days=10, now=now)
        # Keep readings from day 19 onward (29-10=19).
        assert len(record.history) == 11
        assert record.history[0].timestamp == _EPOCH + timedelta(days=19)
        assert record.history[-1].timestamp == _EPOCH + timedelta(days=29)

    def test_keeps_all_when_within_window(self):
        now = _EPOCH + timedelta(days=5)
        record = TrackerRecord(
            id="t", label="T", category="fx",
            history=[_reading(i, i * 0.01) for i in range(6)],
        )
        prune_history(record, history_days=30, now=now)
        assert len(record.history) == 6

    def test_sorts_after_prune(self):
        now = _EPOCH + timedelta(days=5)
        record = TrackerRecord(
            id="t", label="T", category="fx",
            history=[_reading(3, 3.0), _reading(1, 1.0), _reading(5, 5.0)],
        )
        prune_history(record, history_days=30, now=now)
        values = [r.value for r in record.history]
        assert values == [1.0, 3.0, 5.0]
