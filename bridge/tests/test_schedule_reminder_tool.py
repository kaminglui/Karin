"""Tests for the `schedule_reminder` LLM-facing tool + its
feature-flag gating in active_tool_schemas.

Covers:
* Valid trigger_at + message → reminder created + status string.
* Missing / empty args → clear error messages.
* Past / too-soon / too-far trigger_at → refused.
* Malformed ISO → refused with guidance.
* Feature flag OFF → tool hidden from active_tool_schemas.
* Feature flag ON → tool visible to the LLM.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bridge import features
from bridge import tools
from bridge.reminders import api as rem_api
from bridge.reminders.store import ReminderStore


@pytest.fixture(autouse=True)
def _isolated_reminders(tmp_path: Path, monkeypatch):
    """Every test gets a fresh SQLite-backed reminders store so a
    tool call in one test doesn't leave rows visible in another."""
    store = ReminderStore(tmp_path / "reminders.db")
    rem_api.reset_default_store()
    monkeypatch.setattr(rem_api, "_get_store", lambda: store)
    yield store
    rem_api.reset_default_store()


def _future_iso(minutes: int) -> str:
    dt = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Tool body behaviour
# ---------------------------------------------------------------------------
class TestScheduleReminderTool:
    def test_valid_args_creates_reminder(self, _isolated_reminders):
        store = _isolated_reminders
        result = tools._schedule_reminder(
            trigger_at=_future_iso(60), message="call mom",
        )
        assert "scheduled" in result.lower()
        # Row landed in the store.
        upcoming = store.list_upcoming()
        assert len(upcoming) == 1
        assert upcoming[0].message == "call mom"
        assert upcoming[0].source == "llm_tool"

    def test_z_suffix_accepted(self):
        # 'Z' is the standard ISO 8601 UTC designator; Python's
        # fromisoformat doesn't accept it pre-3.11, so the tool
        # rewrites it. This regression-tests that path.
        future = (datetime.now(timezone.utc) + timedelta(minutes=10))
        iso = future.strftime("%Y-%m-%dT%H:%M:%SZ")
        result = tools._schedule_reminder(trigger_at=iso, message="x")
        assert result.startswith("Reminder scheduled")

    def test_empty_trigger_at_rejected(self):
        result = tools._schedule_reminder(trigger_at="", message="x")
        assert result.lower().startswith("error")
        assert "trigger_at" in result

    def test_empty_message_rejected(self):
        result = tools._schedule_reminder(
            trigger_at=_future_iso(60), message="   ",
        )
        assert result.lower().startswith("error")
        assert "message" in result

    def test_malformed_iso_rejected(self):
        result = tools._schedule_reminder(
            trigger_at="tomorrow please", message="x",
        )
        assert result.lower().startswith("error")
        assert "iso" in result.lower() or "parse" in result.lower()

    def test_past_time_rejected(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        result = tools._schedule_reminder(trigger_at=past, message="x")
        assert result.lower().startswith("error")
        assert "past" in result.lower() or "too soon" in result.lower()

    def test_too_far_future_rejected(self):
        far = (datetime.now(timezone.utc) + timedelta(days=400)).isoformat()
        result = tools._schedule_reminder(trigger_at=far, message="x")
        assert result.lower().startswith("error")
        assert "year" in result.lower() or "far" in result.lower()

    def test_naive_datetime_assumed_utc(self, _isolated_reminders):
        """ISO string without tz offset is assumed UTC — the tool
        backfills timezone so the store never sees a naive dt
        (which would raise)."""
        naive = (
            datetime.now(timezone.utc) + timedelta(minutes=30)
        ).replace(tzinfo=None).isoformat()
        result = tools._schedule_reminder(trigger_at=naive, message="x")
        assert result.startswith("Reminder scheduled")


class TestScheduleReminderDedup:
    """When the regex detector creates a reminder pre-LLM and the LLM
    ALSO invokes schedule_reminder on the same prompt, the tool should
    NOT create a second row. Instead:
    * Identical message + close trigger → no-op ("already scheduled").
    * Slightly different details → update the existing row in place."""

    def test_identical_duplicate_is_noop(self, _isolated_reminders):
        store = _isolated_reminders
        trigger = datetime.now(timezone.utc) + timedelta(minutes=60)
        # Simulate the regex detector creating the first reminder.
        from bridge.reminders import create_reminder
        first = create_reminder(
            trigger_at=trigger, message="call mom", _store=store,
        )
        # LLM tool fires with the same time + message.
        result = tools._schedule_reminder(
            trigger_at=trigger.isoformat(), message="call mom",
        )
        assert "already scheduled" in result.lower()
        # No second row created.
        assert len(store.list_upcoming()) == 1

    def test_different_message_updates_existing(self, _isolated_reminders):
        store = _isolated_reminders
        trigger = datetime.now(timezone.utc) + timedelta(minutes=60)
        from bridge.reminders import create_reminder
        first = create_reminder(
            trigger_at=trigger, message="call mom", _store=store,
        )
        # LLM fires with a more specific message at the same time.
        result = tools._schedule_reminder(
            trigger_at=trigger.isoformat(),
            message="call mom about dinner plans",
        )
        assert "updated" in result.lower()
        # Still one row, but message changed.
        reminders = store.list_upcoming()
        assert len(reminders) == 1
        assert reminders[0].message == "call mom about dinner plans"

    def test_different_time_updates_existing(self, _isolated_reminders):
        store = _isolated_reminders
        trigger_old = datetime.now(timezone.utc) + timedelta(minutes=60)
        trigger_new = trigger_old + timedelta(minutes=3)
        from bridge.reminders import create_reminder
        create_reminder(
            trigger_at=trigger_old, message="call mom", _store=store,
        )
        result = tools._schedule_reminder(
            trigger_at=trigger_new.isoformat(), message="call mom",
        )
        assert "updated" in result.lower()
        reminders = store.list_upcoming()
        assert len(reminders) == 1
        # Time shifted by ~3 min.
        assert abs(
            (reminders[0].trigger_at - trigger_new).total_seconds()
        ) < 5

    def test_far_apart_times_create_new(self, _isolated_reminders):
        store = _isolated_reminders
        trigger1 = datetime.now(timezone.utc) + timedelta(minutes=60)
        trigger2 = datetime.now(timezone.utc) + timedelta(hours=3)
        from bridge.reminders import create_reminder
        create_reminder(
            trigger_at=trigger1, message="call mom", _store=store,
        )
        # Same message but 3h later — different event, should create.
        result = tools._schedule_reminder(
            trigger_at=trigger2.isoformat(), message="call mom",
        )
        assert "scheduled" in result.lower() and "updated" not in result.lower()
        assert len(store.list_upcoming()) == 2


# ---------------------------------------------------------------------------
# Feature-flag gating in active_tool_schemas
# ---------------------------------------------------------------------------
class TestFeatureFlagGating:
    def test_flag_off_hides_schedule_reminder(self, monkeypatch):
        monkeypatch.delenv("KARIN_REMINDERS_LLM_TOOL", raising=False)
        features.reload()
        names = [
            (s.get("function") or {}).get("name")
            for s in tools.active_tool_schemas()
        ]
        assert "schedule_reminder" not in names
        # Sanity: other always-on tools are still exposed.
        assert "get_weather" in names

    def test_flag_on_exposes_schedule_reminder(self, monkeypatch):
        monkeypatch.setenv("KARIN_REMINDERS_LLM_TOOL", "true")
        features.reload()
        names = [
            (s.get("function") or {}).get("name")
            for s in tools.active_tool_schemas()
        ]
        assert "schedule_reminder" in names

    def test_dispatcher_still_accepts_even_when_hidden(self, monkeypatch, _isolated_reminders):
        """Schema hidden ≠ tool disabled at dispatch. The
        `schedule_reminder` entry stays in `_DISPATCH` so a direct
        `tools.execute` call still works. Keeps the unit-tested
        behaviour independent of the opt-in UX toggle."""
        monkeypatch.delenv("KARIN_REMINDERS_LLM_TOOL", raising=False)
        features.reload()
        result = tools.execute(
            "schedule_reminder",
            {"trigger_at": _future_iso(30), "message": "via dispatch"},
        )
        assert "scheduled" in result.lower()
