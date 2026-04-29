"""Tests for bridge.trackers.service.

Uses tmp_path for the store and patches the fetcher dispatch so no
network calls happen. Covers:
  - snapshot math (deltas, percentage math, tolerance windows)
  - monthly cadence forces 1d delta to None
  - staleness based on latest reading timestamp
  - TTL gate: repeated get_tracker calls don't re-hit the network
  - error path: FetchError is recorded, not raised
  - refresh_all: one failure doesn't stop others
  - get_trackers with explicit ids list
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from bridge.trackers.fetch import FetchError
from bridge.trackers.models import (
    TrackerConfig,
    TrackerReading,
    TrackerRecord,
)
from bridge.trackers.service import (
    TrackerService,
    _compute_direction,
    _compute_movement,
    _compute_shock,
    _pct_changes,
    _stddev,
)
from bridge.trackers.store import TrackerStore, add_or_replace_reading


# --- fixtures --------------------------------------------------------------

def _cfg_fx(id="usd_cny", to="CNY", **kw) -> TrackerConfig:
    base = dict(
        id=id, label=f"USD/{to}", category="fx", source="frankfurter",
        params={"from": "USD", "to": to}, cadence="daily",
        stale_after_hours=72, history_days=90, enabled=True,
    )
    base.update(kw)
    return TrackerConfig(**base)


def _cfg_monthly(id="us_cpi_food", **kw) -> TrackerConfig:
    base = dict(
        id=id, label="US CPI: Food", category="food_index", source="bls",
        params={"series_id": "CUUR0000SAF1"}, cadence="monthly",
        stale_after_hours=960, history_days=730, enabled=True,
    )
    base.update(kw)
    return TrackerConfig(**base)


def _reading(ts: datetime, value: float) -> TrackerReading:
    return TrackerReading(timestamp=ts, value=value)


def _service_with_history(
    tmp_path, cfg: TrackerConfig, history: list[TrackerReading],
    *, last_fetched_at: datetime | None = None,
) -> TrackerService:
    """Build a service with pre-seeded history. last_fetched_at defaults
    to now() so TTL gate treats it as fresh (tests that need to fetch
    pass an older value or force=True)."""
    store = TrackerStore(tmp_path)
    record = TrackerRecord(
        id=cfg.id, label=cfg.label, category=cfg.category,
        history=list(history),
        last_fetched_at=last_fetched_at or datetime.now(timezone.utc),
    )
    store.save({cfg.id: record})
    return TrackerService(store=store, configs=[cfg])


# --- snapshot math --------------------------------------------------------

class TestSnapshotMath:
    def test_single_reading_gives_none_deltas(self, tmp_path):
        cfg = _cfg_fx()
        now = datetime.now(timezone.utc)
        svc = _service_with_history(tmp_path, cfg, [_reading(now, 7.2)])
        snap = svc.get_tracker("usd_cny")
        assert snap is not None
        assert snap.latest_value == 7.2
        assert snap.change_1d is None
        assert snap.change_1w is None
        assert snap.change_1m is None

    def test_computes_1d_1w_1m_deltas(self, tmp_path):
        cfg = _cfg_fx()
        now = datetime.now(timezone.utc)
        history = [
            _reading(now - timedelta(days=30), 7.00),  # baseline for 1m
            _reading(now - timedelta(days=7),  7.10),  # baseline for 1w
            _reading(now - timedelta(days=1),  7.20),  # baseline for 1d
            _reading(now,                      7.30),  # latest
        ]
        svc = _service_with_history(tmp_path, cfg, history)
        snap = svc.get_tracker("usd_cny")
        assert snap.change_1d == pytest.approx(0.10)
        assert snap.change_1w == pytest.approx(0.20)
        assert snap.change_1m == pytest.approx(0.30)
        # Percentages (2 d.p.)
        assert snap.change_1d_pct == pytest.approx(round((0.10 / 7.20) * 100, 2))
        assert snap.change_1w_pct == pytest.approx(round((0.20 / 7.10) * 100, 2))
        assert snap.change_1m_pct == pytest.approx(round((0.30 / 7.00) * 100, 2))

    def test_no_history_point_in_window_gives_none(self, tmp_path):
        cfg = _cfg_fx()
        now = datetime.now(timezone.utc)
        # Only a 1d-old reading in history, so 1w and 1m must be None.
        history = [
            _reading(now - timedelta(days=1), 7.20),
            _reading(now, 7.30),
        ]
        svc = _service_with_history(tmp_path, cfg, history)
        snap = svc.get_tracker("usd_cny")
        assert snap.change_1d == pytest.approx(0.10)
        assert snap.change_1w is None
        assert snap.change_1m is None

    def test_monthly_cadence_forces_1d_delta_none(self, tmp_path):
        cfg = _cfg_monthly()
        now = datetime.now(timezone.utc)
        # Simulate two monthly readings ~30 days apart.
        history = [
            _reading(now - timedelta(days=60), 300.0),
            _reading(now - timedelta(days=30), 305.0),
        ]
        svc = _service_with_history(
            tmp_path, cfg, history,
            last_fetched_at=now,  # keep TTL gate happy
        )
        snap = svc.get_tracker("us_cpi_food")
        # 1m delta is the standard month-over-month CPI change.
        assert snap.change_1m == pytest.approx(5.0)
        # 1d delta must be None regardless of whether a point is nearby.
        assert snap.change_1d is None
        assert snap.change_1d_pct is None


# --- staleness -----------------------------------------------------------

class TestStaleness:
    def test_recent_reading_not_stale(self, tmp_path):
        cfg = _cfg_fx(stale_after_hours=72)
        now = datetime.now(timezone.utc)
        svc = _service_with_history(
            tmp_path, cfg, [_reading(now - timedelta(hours=12), 7.2)],
        )
        snap = svc.get_tracker("usd_cny")
        assert snap.is_stale is False

    def test_old_reading_is_stale(self, tmp_path):
        cfg = _cfg_fx(stale_after_hours=72)
        now = datetime.now(timezone.utc)
        svc = _service_with_history(
            tmp_path, cfg, [_reading(now - timedelta(hours=100), 7.2)],
        )
        snap = svc.get_tracker("usd_cny")
        assert snap.is_stale is True


# --- TTL gate -------------------------------------------------------------

class TestTTLGate:
    def test_fresh_last_fetched_at_skips_network(self, tmp_path):
        cfg = _cfg_fx()
        now = datetime.now(timezone.utc)
        svc = _service_with_history(
            tmp_path, cfg, [_reading(now, 7.2)], last_fetched_at=now,
        )
        with patch("bridge.trackers.service.run_fetch") as m:
            svc.get_tracker("usd_cny")
        assert m.call_count == 0, "TTL gate should have skipped the fetch"

    def test_force_true_bypasses_ttl(self, tmp_path):
        cfg = _cfg_fx()
        now = datetime.now(timezone.utc)
        # Seed 2 readings so backfill doesn't trigger (needs_backfill is
        # True only when history has 0-1 readings). With 2+ readings the
        # service exercises the single-reading fetch path we're asserting.
        svc = _service_with_history(
            tmp_path, cfg,
            [_reading(now - timedelta(days=1), 7.1), _reading(now, 7.2)],
            last_fetched_at=now,
        )
        with patch("bridge.trackers.service.run_fetch",
                   return_value=_reading(now + timedelta(minutes=1), 7.25)) as m:
            svc.refresh_one("usd_cny", force=True)
        assert m.call_count == 1

    def test_get_tracker_fetch_false_skips_network(self, tmp_path):
        """Tools (and any caller passing fetch=False) must NEVER hit
        upstream — the background tracker poller is the sole origin
        of HTTP. Patching run_fetch to a sentinel proves the read-
        only contract holds even when the TTL window has elapsed."""
        cfg = _cfg_fx()
        now = datetime.now(timezone.utc)
        # last_fetched_at is set to the epoch so the TTL would
        # NORMALLY trigger a refresh. fetch=False must short-circuit
        # before that check.
        svc = _service_with_history(
            tmp_path, cfg, [_reading(now, 7.2)],
            last_fetched_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        with patch("bridge.trackers.service.run_fetch") as m:
            snap = svc.get_tracker("usd_cny", fetch=False)
        assert m.call_count == 0
        assert snap is not None
        assert snap.latest_value == 7.2

    def test_get_trackers_fetch_false_skips_network(self, tmp_path):
        """Same contract as get_tracker, but for the multi-snapshot
        path that the trackers panel uses."""
        cfg = _cfg_fx()
        now = datetime.now(timezone.utc)
        svc = _service_with_history(
            tmp_path, cfg, [_reading(now, 7.2)],
            last_fetched_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        with patch("bridge.trackers.service.run_fetch") as m:
            snaps = svc.get_trackers(fetch=False)
        assert m.call_count == 0
        assert len(snaps) == 1

    def test_stale_last_fetched_at_triggers_fetch(self, tmp_path):
        cfg = _cfg_fx()
        now = datetime.now(timezone.utc)
        svc = _service_with_history(
            tmp_path, cfg,
            [_reading(now - timedelta(hours=36), 7.1),
             _reading(now - timedelta(hours=15), 7.2)],
            last_fetched_at=now - timedelta(hours=15),   # beyond 12h daily TTL
        )
        with patch("bridge.trackers.service.run_fetch",
                   return_value=_reading(now, 7.25)) as m:
            svc.get_tracker("usd_cny")
        assert m.call_count == 1


# --- error handling ------------------------------------------------------

class TestErrorHandling:
    def test_fetch_error_recorded_not_raised(self, tmp_path):
        cfg = _cfg_fx()
        store = TrackerStore(tmp_path)
        svc = TrackerService(store=store, configs=[cfg])
        # An empty store triggers the backfill path first; patch that to
        # return [] so the service falls through to run_fetch (which we
        # assert raises a FetchError that gets persisted, not re-raised).
        with patch("bridge.trackers.service.run_fetch_history", return_value=[]), \
             patch("bridge.trackers.service.run_fetch",
                   side_effect=FetchError("frankfurter: status 503")):
            snap = svc.get_tracker("usd_cny")
        assert snap is not None
        assert snap.latest_value is None
        record = store.load()["usd_cny"]
        assert record.last_fetch_error == "frankfurter: status 503"
        assert "503" in snap.note

    def test_refresh_all_continues_past_one_failure(self, tmp_path):
        cfg_a = _cfg_fx(id="a", to="CNY")
        cfg_b = _cfg_fx(id="b", to="HKD")
        store = TrackerStore(tmp_path)
        svc = TrackerService(store=store, configs=[cfg_a, cfg_b])

        now = datetime.now(timezone.utc)

        def side_effect(cfg, client):
            if cfg.id == "a":
                raise FetchError("feed a failed")
            return _reading(now, 7.81)

        # Same pattern as above: short-circuit backfill so the mocked
        # single-reading fetcher is the one that actually runs.
        with patch("bridge.trackers.service.run_fetch_history", return_value=[]), \
             patch("bridge.trackers.service.run_fetch", side_effect=side_effect):
            result = svc.refresh_all(force=True)
        assert result == {"a": False, "b": True}
        records = store.load()
        assert records["a"].last_fetch_error == "feed a failed"
        assert records["b"].last_fetch_error is None
        assert len(records["b"].history) == 1


# --- unknown / disabled ---------------------------------------------------

class TestUnknownAndDisabled:
    def test_unknown_id_returns_none(self, tmp_path):
        store = TrackerStore(tmp_path)
        svc = TrackerService(store=store, configs=[_cfg_fx()])
        assert svc.get_tracker("not_a_real_id") is None

    def test_disabled_tracker_returns_none(self, tmp_path):
        cfg = _cfg_fx(enabled=False)
        store = TrackerStore(tmp_path)
        svc = TrackerService(store=store, configs=[cfg])
        assert svc.get_tracker("usd_cny") is None

    def test_get_trackers_preserves_caller_order(self, tmp_path):
        cfg_a = _cfg_fx(id="a", to="CNY")
        cfg_b = _cfg_fx(id="b", to="HKD")
        cfg_c = _cfg_fx(id="c", to="JPY")
        now = datetime.now(timezone.utc)
        store = TrackerStore(tmp_path)
        # Pre-seed all three so TTL gate skips fetch.
        for cfg in (cfg_a, cfg_b, cfg_c):
            record = TrackerRecord(id=cfg.id, label=cfg.label, category=cfg.category,
                                   history=[_reading(now, 1.0)], last_fetched_at=now)
            store.save({**store.load(), cfg.id: record})
        svc = TrackerService(store=store, configs=[cfg_a, cfg_b, cfg_c])
        snaps = svc.get_trackers(ids=["c", "a", "b"])
        assert [s.id for s in snaps] == ["c", "a", "b"]


# --- Phase 5.2: derived labels --------------------------------------------

class TestPctChangesAndStddev:
    def test_pct_changes_basic_math(self):
        now = datetime.now(timezone.utc)
        hist = [
            TrackerReading(timestamp=now - timedelta(days=2), value=100.0),
            TrackerReading(timestamp=now - timedelta(days=1), value=101.0),
            TrackerReading(timestamp=now, value=99.99),
        ]
        changes = _pct_changes(hist)
        assert changes[0] == pytest.approx(1.0)            # 100 -> 101
        assert changes[1] == pytest.approx(-1.0, rel=1e-3)  # 101 -> 99.99

    def test_pct_changes_skips_zero_baseline(self):
        now = datetime.now(timezone.utc)
        hist = [
            TrackerReading(timestamp=now - timedelta(days=2), value=0.0),
            TrackerReading(timestamp=now - timedelta(days=1), value=100.0),
            TrackerReading(timestamp=now, value=101.0),
        ]
        changes = _pct_changes(hist)
        # 0 -> 100 is skipped (div by zero); 100 -> 101 is 1.0%.
        assert len(changes) == 1
        assert changes[0] == pytest.approx(1.0)

    def test_stddev_sample_math(self):
        # sample std of {1, 2, 3, 4, 5} = sqrt(sum(sq_dev)/(n-1)) = sqrt(10/4)
        assert _stddev([1, 2, 3, 4, 5]) == pytest.approx((10 / 4) ** 0.5)

    def test_stddev_none_below_two(self):
        assert _stddev([]) is None
        assert _stddev([1.0]) is None


class TestDirectionLabel:
    def _changes(self, *, pct: float = 0.1, n: int = 20) -> list[float]:
        # Build a synthetic list of pct-changes with roughly constant
        # magnitude so sigma ≈ pct (sign-alternating gives exactly pct).
        return [pct * (1 if i % 2 == 0 else -1) for i in range(n)]

    def test_flat_when_change_within_sigma(self):
        # sigma of changes ≈ 0.1. A 0.05% change is below threshold -> flat.
        changes = self._changes(pct=0.1, n=20)
        assert _compute_direction(0.05, changes, lookback_days=1) == "flat"

    def test_up_when_change_exceeds_sigma(self):
        changes = self._changes(pct=0.1, n=20)
        assert _compute_direction(0.5, changes, lookback_days=1) == "up"

    def test_down_when_change_exceeds_sigma_negative(self):
        changes = self._changes(pct=0.1, n=20)
        assert _compute_direction(-0.5, changes, lookback_days=1) == "down"

    def test_none_when_change_is_none(self):
        changes = self._changes(pct=0.1, n=20)
        assert _compute_direction(None, changes, lookback_days=1) is None

    def test_fallback_threshold_when_history_too_short(self):
        # Only 5 observed changes -> below _DIRECTION_MIN_POINTS_FOR_SIGMA.
        # Falls back to fixed 0.25% threshold.
        short = self._changes(pct=5.0, n=5)  # high "sigma" but ignored
        # 0.1% < 0.25% fallback -> flat
        assert _compute_direction(0.1, short, lookback_days=1) == "flat"
        # 0.5% > 0.25% fallback -> up
        assert _compute_direction(0.5, short, lookback_days=1) == "up"

    def test_1w_threshold_scales_sqrt_7(self):
        # sigma ≈ 0.1. Daily threshold = 0.1; weekly threshold = 0.1 * sqrt(7) ≈ 0.265.
        changes = self._changes(pct=0.1, n=20)
        # 0.2% change: exceeds daily (0.1) but below weekly (0.265) -> up / flat
        assert _compute_direction(0.2, changes, lookback_days=1) == "up"
        assert _compute_direction(0.2, changes, lookback_days=7) == "flat"


class TestMovementLabel:
    def _series(self, sigmas_per_day: list[float]) -> list[TrackerReading]:
        """Build a history whose consecutive pct-changes match the given
        magnitudes (sign-alternating). Lets tests pin ratio behavior."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        readings = [TrackerReading(timestamp=base, value=100.0)]
        value = 100.0
        for i, mag in enumerate(sigmas_per_day, start=1):
            direction = 1 if i % 2 == 0 else -1
            value = value * (1 + direction * mag / 100.0)
            readings.append(TrackerReading(
                timestamp=base + timedelta(days=i), value=value,
            ))
        return readings

    def test_none_when_history_too_short(self):
        hist = self._series([0.1] * 10)  # only 10 changes; need 25+
        assert _compute_movement(hist) is None

    def test_stable_when_recent_much_quieter_than_baseline(self):
        # 20 changes at 0.4% sigma, then 5 changes at 0.05% sigma.
        # ratio ≈ 0.05/0.4 = 0.125 < 0.75 -> stable.
        hist = self._series([0.4] * 20 + [0.05] * 5)
        assert _compute_movement(hist) == "stable"

    def test_volatile_when_recent_much_louder_than_baseline(self):
        # 20 changes at 0.1%, then 5 at 1.0%. Ratio ≈ 10 -> volatile.
        hist = self._series([0.1] * 20 + [1.0] * 5)
        assert _compute_movement(hist) == "volatile"

    def test_moving_when_ratio_near_one(self):
        # Baseline 0.1% sigma, recent 0.1% sigma -> ratio ~1 -> moving.
        hist = self._series([0.1] * 25)
        assert _compute_movement(hist) == "moving"


class TestShockLabel:
    def _series_with_trailing(
        self, baseline_pct: float, trailing_pcts: list[float],
    ) -> list[TrackerReading]:
        """Build history with N baseline changes of given pct, then
        specific trailing changes (tail controls recent behavior)."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        readings = [TrackerReading(timestamp=base, value=100.0)]
        value = 100.0
        # 20 baseline changes so _SHOCK_MIN_BASELINE_POINTS (10) is comfortably met
        # after excluding the last 5.
        for i in range(20):
            direction = 1 if i % 2 == 0 else -1
            value = value * (1 + direction * baseline_pct / 100.0)
            readings.append(TrackerReading(
                timestamp=base + timedelta(days=len(readings)), value=value,
            ))
        # Trailing changes go as-specified, preserving their sign.
        for pct in trailing_pcts:
            value = value * (1 + pct / 100.0)
            readings.append(TrackerReading(
                timestamp=base + timedelta(days=len(readings)), value=value,
            ))
        return readings

    def test_surging_on_large_positive_move(self):
        hist = self._series_with_trailing(
            baseline_pct=0.1, trailing_pcts=[0, 0, 0, 0, 5.0],
        )
        # latest change_1d is +5%; baseline sigma ≈ 0.1%; 5% >> 3 * 0.1%.
        assert _compute_shock(5.0, hist) == "surging"

    def test_plunging_on_large_negative_move(self):
        hist = self._series_with_trailing(
            baseline_pct=0.1, trailing_pcts=[0, 0, 0, 0, -5.0],
        )
        assert _compute_shock(-5.0, hist) == "plunging"

    def test_none_on_normal_move(self):
        hist = self._series_with_trailing(
            baseline_pct=0.1, trailing_pcts=[0.1] * 5,
        )
        # 0.1% is the baseline norm; nowhere near 3-sigma.
        assert _compute_shock(0.1, hist) is None

    def test_lag_prevents_recent_shock_suppressing_next_detection(self):
        # Baseline 0.1% sigma. Then a 4% shock (in the excluded tail),
        # then a 5% shock right after. The 5% move should still be
        # flagged because the baseline calculation EXCLUDES the last 5
        # changes (which is where the prior shock sits).
        hist = self._series_with_trailing(
            baseline_pct=0.1, trailing_pcts=[0, 0, 0, 4.0, 5.0],
        )
        # change_1d_pct passed in is 5% (the latest).
        assert _compute_shock(5.0, hist) == "surging"


class TestCadenceGatesLabels:
    def test_monthly_cadence_skips_all_four_labels(self, tmp_path):
        # Build a monthly CPI tracker with 30 months of history.
        cfg = _cfg_monthly()
        base = datetime(2023, 1, 1, tzinfo=timezone.utc)
        history = [
            TrackerReading(
                timestamp=base + timedelta(days=30 * i), value=300.0 + i * 0.5,
            )
            for i in range(30)
        ]
        now = base + timedelta(days=30 * 29)  # match the latest reading
        svc = _service_with_history(tmp_path, cfg, history, last_fetched_at=now)
        snap = svc.get_tracker("us_cpi_food")
        assert snap is not None
        # All four derived labels must be None regardless of history length.
        assert snap.direction_1d is None
        assert snap.direction_1w is None
        assert snap.movement_label is None
        assert snap.shock_label is None
