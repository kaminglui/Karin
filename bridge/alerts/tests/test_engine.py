"""Tests for bridge.alerts.engine.AlertEngine.

The engine composes rules + cooldowns + store. Rules are tested in
isolation in test_rules.py; here we verify the composition itself:
cooldown suppression, logging both fires and suppressions, rule
crash tolerance.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bridge.alerts.cooldown import CooldownLedger
from bridge.alerts.engine import AlertEngine
from bridge.alerts.models import (
    Alert,
    AlertCategory,
    AlertLevel,
    Signal,
    SignalKind,
)
from bridge.alerts.rules import AlertRule, TrackerShockFX
from bridge.alerts.store import AlertStore


_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)


def _shock(**kw) -> Signal:
    payload = dict(
        tracker_id="usd_cny", label="USD/CNY", category="fx",
        direction="surging", change_pct=5.0, latest_value=7.0,
    )
    payload.update(kw)
    return Signal(
        kind=SignalKind.TRACKER_SHOCK,
        source=f"tracker:{payload['tracker_id']}",
        payload=payload,
        observed_at=_NOW,
    )


@pytest.fixture
def engine(tmp_path):
    store = AlertStore(tmp_path)
    ledger = CooldownLedger(tmp_path / "cooldowns.json")
    return AlertEngine(
        rules=[TrackerShockFX()],
        cooldown_ledger=ledger,
        store=store,
    ), store, ledger


class TestEngineRun:
    def test_fires_alert_and_marks_cooldown(self, engine):
        eng, store, ledger = engine
        result = eng.run([_shock()], _NOW)
        assert len(result.alerts) == 1
        assert len(result.suppressed) == 0
        assert ledger.is_in_cooldown("tracker_shock_fx", "usd_cny", 6, _NOW)

    def test_second_run_within_cooldown_is_suppressed(self, engine):
        eng, store, ledger = engine
        eng.run([_shock()], _NOW)
        later = _NOW + timedelta(minutes=5)
        result = eng.run([_shock()], later)
        assert len(result.alerts) == 0
        assert len(result.suppressed) == 1
        assert result.suppressed[0].rule_id == "tracker_shock_fx"
        assert result.suppressed[0].scope_key == "usd_cny"

    def test_run_after_cooldown_expires_fires_again(self, engine):
        eng, store, ledger = engine
        eng.run([_shock()], _NOW)
        later = _NOW + timedelta(hours=7)
        result = eng.run([_shock()], later)
        assert len(result.alerts) == 1

    def test_crashing_rule_does_not_halt_scan(self, tmp_path):
        class Broken(AlertRule):
            id = "broken"
            cooldown_hours = 1

            def evaluate(self, signals, now):
                raise RuntimeError("boom")

        store = AlertStore(tmp_path)
        ledger = CooldownLedger(tmp_path / "cooldowns.json")
        eng = AlertEngine(
            rules=[Broken(), TrackerShockFX()],
            cooldown_ledger=ledger,
            store=store,
        )
        # Broken rule crashes; TrackerShockFX still fires.
        result = eng.run([_shock()], _NOW)
        assert len(result.alerts) == 1
        assert result.alerts[0].rule_id == "tracker_shock_fx"
