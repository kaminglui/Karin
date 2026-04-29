"""Tests for bridge.alerts.cooldown."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bridge.alerts.cooldown import CooldownLedger


_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def ledger(tmp_path):
    return CooldownLedger(tmp_path / "cooldowns.json")


class TestCooldown:
    def test_fresh_scope_not_in_cooldown(self, ledger):
        assert ledger.is_in_cooldown("r1", "scope", 6, _NOW) is False

    def test_mark_fired_triggers_cooldown(self, ledger):
        ledger.mark_fired("r1", "scope", _NOW)
        assert ledger.is_in_cooldown("r1", "scope", 6, _NOW) is True

    def test_cooldown_expires_after_window(self, ledger):
        ledger.mark_fired("r1", "scope", _NOW)
        future = _NOW + timedelta(hours=7)
        assert ledger.is_in_cooldown("r1", "scope", 6, future) is False

    def test_different_scopes_independent(self, ledger):
        ledger.mark_fired("r1", "scope_a", _NOW)
        assert ledger.is_in_cooldown("r1", "scope_a", 6, _NOW) is True
        assert ledger.is_in_cooldown("r1", "scope_b", 6, _NOW) is False

    def test_different_rules_independent(self, ledger):
        ledger.mark_fired("r1", "scope", _NOW)
        assert ledger.is_in_cooldown("r1", "scope", 6, _NOW) is True
        assert ledger.is_in_cooldown("r2", "scope", 6, _NOW) is False

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "cooldowns.json"
        a = CooldownLedger(path)
        a.mark_fired("r1", "scope", _NOW)
        b = CooldownLedger(path)
        assert b.is_in_cooldown("r1", "scope", 6, _NOW) is True
