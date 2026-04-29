"""Tests for the tracker tool layer in bridge.tools.

These exercise the FULL dispatch path via tools.execute() with a
patched singleton so no network calls happen.

The legacy `get_tracker` / `get_trackers` tool names were consolidated
into a single `tracker` tool with optional ``id`` and ``period`` args.
Migration:

* ``tools.execute("get_tracker", {"tracker_id": X})``
  → ``tools.execute("tracker", {"id": X})``
* ``tools.execute("tracker", {})``
  → ``tools.execute("tracker", {})``    (no id → all enabled)
* The retired multi-id form ``get_trackers(tracker_ids=[...])`` has no
  consolidated dispatcher equivalent. The underlying
  ``tools._get_trackers(tracker_ids=...)`` helper is still exercised
  directly in the few cases where filter-by-list semantics matter.

Test coverage list (unchanged in intent):

  1. tracker(id=…) returns a formatted single-line result
  2. alias resolution works
  3. invalid tracker id returns a clear message
  4. tracker() returns all enabled trackers
  5. _get_trackers(tracker_ids=[…]) filters correctly
  6. stale tracker formatting includes a stale note
  7. missing deltas do not produce bogus text
  8. tool path goes through the real service interface, not direct store access
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from bridge import tools
from bridge.trackers.models import (
    TrackerConfig,
    TrackerReading,
    TrackerRecord,
)
from bridge.trackers.service import TrackerService
from bridge.trackers.store import TrackerStore


# --- fixtures --------------------------------------------------------------

def _cfg(**kw) -> TrackerConfig:
    base = dict(
        id="usd_cny", label="USD/CNY", category="fx", source="frankfurter",
        params={"from": "USD", "to": "CNY"}, cadence="daily",
        stale_after_hours=72, history_days=90, enabled=True,
    )
    base.update(kw)
    return TrackerConfig(**base)


def _reading(ts: datetime, value: float) -> TrackerReading:
    return TrackerReading(timestamp=ts, value=value)


@pytest.fixture
def service_with_history(tmp_path):
    """One FX tracker with 30d of history — all deltas computable."""
    cfg = _cfg()
    now = datetime.now(timezone.utc)
    store = TrackerStore(tmp_path)
    record = TrackerRecord(
        id=cfg.id, label=cfg.label, category=cfg.category,
        history=[
            _reading(now - timedelta(days=30), 7.00),
            _reading(now - timedelta(days=7),  7.10),
            _reading(now - timedelta(days=1),  7.20),
            _reading(now,                      7.30),
        ],
        last_fetched_at=now,
    )
    store.save({cfg.id: record})
    return TrackerService(store=store, configs=[cfg])


@pytest.fixture
def service_mixed(tmp_path):
    """Three trackers: USD/CNY fresh (single reading), Gold stale, CPI monthly."""
    now = datetime.now(timezone.utc)
    configs = [
        _cfg(),
        _cfg(id="gold_usd", label="Gold", category="metal",
             source="stooq", params={"symbol": "xauusd"}),
        _cfg(id="us_cpi_food", label="US CPI: Food", category="food_index",
             source="bls", params={"series_id": "CUUR0000SAF1"},
             cadence="monthly", stale_after_hours=960, history_days=730),
    ]
    store = TrackerStore(tmp_path)
    records = {
        "usd_cny": TrackerRecord(
            id="usd_cny", label="USD/CNY", category="fx",
            history=[_reading(now, 7.30)],          # single reading -> no deltas
            last_fetched_at=now,
        ),
        # Gold reading is 96h old vs 72h stale threshold -> is_stale=True.
        # last_fetched_at=now keeps TTL gate from triggering a real fetch.
        "gold_usd": TrackerRecord(
            id="gold_usd", label="Gold", category="metal",
            history=[_reading(now - timedelta(hours=96), 2347.80)],
            last_fetched_at=now,
        ),
        "us_cpi_food": TrackerRecord(
            id="us_cpi_food", label="US CPI: Food", category="food_index",
            history=[
                _reading(now - timedelta(days=60), 300.0),
                _reading(now - timedelta(days=30), 305.0),
            ],
            last_fetched_at=now,
        ),
    }
    store.save(records)
    return TrackerService(store=store, configs=configs)


# --- tests: tracker(id=…) --------------------------------------------------

class TestTrackerSingleId:
    def test_returns_formatted_single_line(self, service_with_history):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_with_history):
            result = tools.execute("tracker", {"id": "usd_cny"})
        # Label, value, date all present.
        assert "USD/CNY is 7.3000" in result
        # Daily FX uses "latest close" wording (monthly CPI would use "as of").
        assert "latest close" in result
        # Longest available delta wins (1m, since 30d history exists).
        assert "1 month ago" in result

    def test_alias_resolution_slash(self, service_with_history):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_with_history):
            result = tools.execute("tracker", {"id": "usd/cny"})
        assert "USD/CNY is 7.3000" in result

    def test_alias_resolution_gold(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools.execute("tracker", {"id": "gold"})
        assert "Gold is 2347.8000 USD/oz" in result

    def test_alias_resolution_food(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools.execute("tracker", {"id": "food"})
        assert "US CPI: Food is 305.0000" in result
        # CPI cadence is monthly, so 1m is the natural (and only) delta.
        assert "1 month ago" in result

    def test_invalid_id_returns_not_found(self, service_with_history):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_with_history):
            result = tools.execute("tracker", {"id": "not_a_tracker"})
        assert "not found" in result.lower()

    def test_stale_tracker_is_explicit(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools.execute("tracker", {"id": "gold"})
        assert "stale" in result.lower()

    def test_no_delta_when_single_reading(self, service_mixed):
        # usd_cny in service_mixed has only one reading -> no deltas possible.
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools.execute("tracker", {"id": "usd_cny"})
        assert "USD/CNY is 7.3000" in result
        # Guard against bogus synthesized delta text.
        for bogus in ("vs 1 month", "vs 1 week", "vs 1 day", "Up 0", "Down 0"):
            assert bogus not in result


# --- tests: tracker() (all enabled) + filter helper ----------------------

class TestTrackerMulti:
    def test_returns_all_when_no_ids(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools.execute("tracker", {})
        lines = result.splitlines()
        assert len(lines) == 3
        assert any("USD/CNY" in line for line in lines)
        assert any("Gold" in line for line in lines)
        assert any("US CPI: Food" in line for line in lines)

    # The consolidated `tracker` tool only exposes "all enabled" or
    # "single id" — the filter-by-list path is still alive in
    # bridge.tools._get_trackers, so the next four tests call that
    # helper directly. Same coverage; just one layer below the
    # dispatcher (which never sees ``tracker_ids`` arguments).

    def test_filters_by_explicit_ids(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools._get_trackers(["usd_cny", "gold_usd"])
        lines = result.splitlines()
        assert len(lines) == 2

    def test_filter_respects_aliases(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools._get_trackers(["gold", "food"])
        lines = result.splitlines()
        assert any("Gold" in line for line in lines)
        assert any("US CPI: Food" in line for line in lines)

    def test_all_invalid_ids_returns_clean_message(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools._get_trackers(["bad1", "bad2"])
        assert "no trackers found" in result.lower()
        # Lists the requested ids for clarity.
        assert "bad1" in result and "bad2" in result

    def test_stale_marker_in_multi(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed):
            result = tools._get_trackers(["gold"])
        # Compact line form: "Gold: <value> USD/oz (<date>, stale)"
        assert "stale" in result.lower()
        assert "USD/oz" in result


# --- path integrity --------------------------------------------------------

class TestPathIntegrity:
    def test_tool_path_calls_service_get_tracker(self, service_with_history):
        """The tool must go through TrackerService, not poke the store."""
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_with_history), \
             patch.object(service_with_history, "get_tracker",
                          wraps=service_with_history.get_tracker) as m:
            tools.execute("tracker", {"id": "usd_cny"})
        assert m.call_count == 1

    def test_tool_path_calls_service_get_trackers(self, service_mixed):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_mixed), \
             patch.object(service_mixed, "get_trackers",
                          wraps=service_mixed.get_trackers) as m:
            tools.execute("tracker", {})
        assert m.call_count == 1


# --- Phase 5.2: derived labels in tool output -----------------------------

def _backward_history(total_changes: int, build_pct: callable) -> list[TrackerReading]:
    """Build a history list ending at wall-clock `now`, with `total_changes`
    pct-changes leading up to it. build_pct(i) returns the pct change for
    step i (1..total_changes). Used by the shock / volatile fixtures.
    Timestamps are counted backward from now so TTL gate treats the fixture
    as fresh and doesn't try to hit the network.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=total_changes)
    v = 7.00
    readings = [TrackerReading(timestamp=start, value=v)]
    for i in range(1, total_changes + 1):
        v = v * (1 + build_pct(i) / 100.0)
        ts = start + timedelta(days=i)
        # Pin the last reading to exactly now so last_fetched_at=now
        # keeps the TTL gate happy.
        if i == total_changes:
            ts = now
        readings.append(TrackerReading(timestamp=ts, value=v))
    return readings


@pytest.fixture
def service_with_shock(tmp_path):
    """Daily FX tracker with 20 quiet baseline changes + 4 quiet + 1 surge.

    The 5% surge sits inside the _SHOCK_EXCLUDE_RECENT tail (5 latest
    changes) so it doesn't contaminate the baseline sigma estimate —
    the shock IS still detected because it's also the latest change_1d.
    """
    cfg = _cfg()

    def pct_for_step(i: int) -> float:
        if i <= 20:
            return 0.1 if i % 2 == 0 else -0.1
        if i <= 24:
            return 0.05
        return 5.0  # latest change: 5% surge

    history = _backward_history(25, pct_for_step)
    now = history[-1].timestamp
    store = TrackerStore(tmp_path)
    store.save({cfg.id: TrackerRecord(
        id=cfg.id, label=cfg.label, category=cfg.category,
        history=history, last_fetched_at=now,
    )})
    return TrackerService(store=store, configs=[cfg])


@pytest.fixture
def service_volatile(tmp_path):
    """Daily FX tracker: 20 baseline changes ~0.1% then 5 recent ~1.0%.

    Ratio ~10 -> movement label = 'volatile'. No shock (the 1% moves are
    below 3-sigma of the baseline which is ~0.1%... actually wait, 1% is
    ~10x, which would trigger shock. Calibrate so shock does NOT trigger:
    we want to isolate the 'volatile without shock' branch. Use recent
    moves at 0.5% — still 5x baseline (volatile), but below 3-sigma * some
    multiplier that still keeps it from being flagged as shock. Actually
    3 * 0.1% = 0.3%, and 0.5% > 0.3% so shock WOULD fire. Use recent
    moves of 0.25% instead: 2.5x baseline (volatile ratio >= 1.5 -> yes),
    and 0.25% < 0.3% shock threshold -> no shock. Clean isolation.
    """
    cfg = _cfg()

    def pct_for_step(i: int) -> float:
        if i <= 20:
            return 0.1 if i % 2 == 0 else -0.1  # baseline sigma ~0.1
        return 0.25 if i % 2 == 0 else -0.25    # recent sigma ~0.25 (2.5x)

    history = _backward_history(25, pct_for_step)
    now = history[-1].timestamp
    store = TrackerStore(tmp_path)
    store.save({cfg.id: TrackerRecord(
        id=cfg.id, label=cfg.label, category=cfg.category,
        history=history, last_fetched_at=now,
    )})
    return TrackerService(store=store, configs=[cfg])


class TestDerivedLabelsInOutput:
    def test_single_tracker_appends_surging_sentence(self, service_with_shock):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_with_shock):
            result = tools.execute("tracker", {"id": "usd_cny"})
        # Shock dominates; flag sentence appended after delta line.
        assert result.endswith("Surging.")

    def test_multi_tracker_tag_is_shock_when_shocked(self, service_with_shock):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_with_shock):
            result = tools.execute("tracker", {})
        # Single line, fresh, surging tag inside the parens.
        assert "surging" in result.lower()

    def test_single_tracker_appends_volatile_sentence_when_no_shock(
        self, service_volatile,
    ):
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_volatile):
            result = tools.execute("tracker", {"id": "usd_cny"})
        # Movement == volatile, no shock -> "Volatile." appended.
        assert result.endswith("Volatile.")
        # And NOT mistakenly marked as surging/plunging.
        assert "Surging" not in result and "Plunging" not in result

    def test_multi_tracker_omits_tag_on_quiet_series(self, service_with_history):
        # service_with_history has only 4 readings — no movement/shock
        # labels possible. Direction_1d may be populated, though.
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=service_with_history):
            result = tools.execute("tracker", {})
        # Line should NOT carry shock or volatile tags.
        assert "surging" not in result.lower()
        assert "plunging" not in result.lower()
        assert "volatile" not in result.lower()

    def test_single_tracker_cpi_output_has_no_derived_flag_sentence(self, tmp_path):
        # Build a monthly CPI service with plenty of history. Labels
        # must stay None; voice output must not gain a flag sentence.
        cfg = _cfg(
            id="us_cpi_food", label="US CPI: Food", category="food_index",
            source="bls", params={"series_id": "CUUR0000SAF1"},
            cadence="monthly", stale_after_hours=960, history_days=730,
        )
        base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        history = [
            TrackerReading(timestamp=base + timedelta(days=30 * i),
                           value=300.0 + i * 0.5)
            for i in range(30)
        ]
        store = TrackerStore(tmp_path)
        record = TrackerRecord(
            id=cfg.id, label=cfg.label, category=cfg.category,
            history=history,
            last_fetched_at=history[-1].timestamp,
        )
        store.save({cfg.id: record})
        svc = TrackerService(store=store, configs=[cfg])
        with patch("bridge.trackers.service.get_default_tracker_service",
                   return_value=svc):
            result = tools.execute("tracker", {"id": "us_cpi_food"})
        # No shock/volatile sentence should appear (monthly has None labels).
        for bogus in ("Surging.", "Plunging.", "Volatile."):
            assert bogus not in result
