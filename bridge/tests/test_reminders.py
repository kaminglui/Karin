"""Tests for bridge.reminders — store CRUD, scheduler dispatch, and
the notify integration handoff.

The store is constructed against ``tmp_path`` per test so cases stay
isolated. The notify dispatcher is exercised via its real public
function (``bridge.notify.notify``) with the underlying httpx.post
patched, so the integration path is the same one the production
poller takes — no shortcuts that would let the test pass while the
production wiring is broken.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from bridge.reminders.api import (
    cancel_reminder,
    create_reminder,
    fire_due_reminders,
    list_upcoming,
)
from bridge.reminders.models import Reminder
from bridge.reminders.store import ReminderStore


# ---------------------------------------------------------------------------
# Store CRUD
# ---------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path: Path) -> ReminderStore:
    return ReminderStore(tmp_path / "reminders.db")


def _utc_in(seconds: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


class TestStoreCRUD:
    def test_create_returns_persisted_row_with_id(self, store):
        rem = store.create(
            trigger_at=_utc_in(60),
            message="call mom",
            source="manual",
        )
        assert isinstance(rem, Reminder)
        assert rem.id  # store-assigned, non-empty
        assert rem.message == "call mom"
        assert rem.source == "manual"
        assert rem.delivered is False
        # Same row resolves on get().
        fetched = store.get(rem.id)
        assert fetched is not None
        assert fetched.id == rem.id

    def test_naive_trigger_at_rejected(self, store):
        with pytest.raises(ValueError):
            store.create(trigger_at=datetime.utcnow(), message="x")

    def test_empty_message_rejected(self, store):
        with pytest.raises(ValueError):
            store.create(trigger_at=_utc_in(60), message="   ")

    def test_message_is_trimmed(self, store):
        rem = store.create(trigger_at=_utc_in(60), message="  do thing  ")
        assert rem.message == "do thing"

    def test_get_unknown_returns_none(self, store):
        assert store.get("nope") is None

    def test_cancel_removes_row(self, store):
        rem = store.create(trigger_at=_utc_in(60), message="x")
        assert store.cancel(rem.id) is True
        assert store.get(rem.id) is None

    def test_cancel_unknown_returns_false(self, store):
        assert store.cancel("nope") is False

    def test_list_upcoming_orders_by_trigger_asc(self, store):
        late = store.create(trigger_at=_utc_in(120), message="late")
        soon = store.create(trigger_at=_utc_in(30), message="soon")
        mid = store.create(trigger_at=_utc_in(60), message="mid")
        ids = [r.id for r in store.list_upcoming()]
        assert ids == [soon.id, mid.id, late.id]

    def test_list_upcoming_excludes_delivered(self, store):
        a = store.create(trigger_at=_utc_in(60), message="a")
        b = store.create(trigger_at=_utc_in(120), message="b")
        store.mark_delivered(a.id)
        ids = [r.id for r in store.list_upcoming()]
        assert ids == [b.id]


# ---------------------------------------------------------------------------
# fetch_due + atomic mark_delivered
# ---------------------------------------------------------------------------
class TestFetchAndMark:
    def test_fetch_due_returns_only_past(self, store):
        past = store.create(trigger_at=_utc_in(-30), message="past")
        future = store.create(trigger_at=_utc_in(60), message="future")
        due = store.fetch_due()
        ids = [r.id for r in due]
        assert past.id in ids
        assert future.id not in ids

    def test_fetch_due_excludes_already_delivered(self, store):
        past = store.create(trigger_at=_utc_in(-30), message="past")
        store.mark_delivered(past.id)
        assert store.fetch_due() == []

    def test_mark_delivered_is_idempotent(self, store):
        rem = store.create(trigger_at=_utc_in(-30), message="x")
        assert store.mark_delivered(rem.id) is True
        # Second call returns False because the row's already flagged.
        assert store.mark_delivered(rem.id) is False

    def test_mark_delivered_unknown_returns_false(self, store):
        assert store.mark_delivered("nope") is False


# ---------------------------------------------------------------------------
# Public API + scheduler dispatch
# ---------------------------------------------------------------------------
class TestApiAndScheduler:
    def test_create_reminder_routes_through_store(self, store):
        rem = create_reminder(
            trigger_at=_utc_in(60), message="x", _store=store,
        )
        assert store.get(rem.id) is not None

    def test_cancel_reminder_routes_through_store(self, store):
        rem = create_reminder(
            trigger_at=_utc_in(60), message="x", _store=store,
        )
        assert cancel_reminder(rem.id, _store=store) is True
        assert store.get(rem.id) is None

    def test_list_upcoming_routes_through_store(self, store):
        a = create_reminder(trigger_at=_utc_in(120), message="a", _store=store)
        b = create_reminder(trigger_at=_utc_in(30), message="b", _store=store)
        ids = [r.id for r in list_upcoming(_store=store)]
        assert ids == [b.id, a.id]

    def test_fire_due_marks_delivered_and_returns_fired_list(self, store):
        past = create_reminder(
            trigger_at=_utc_in(-10), message="past", _store=store,
        )
        future = create_reminder(
            trigger_at=_utc_in(60), message="future", _store=store,
        )
        # Patch the notify call so we don't actually try to push.
        with patch("bridge.notify.notify") as mock_notify:
            fired = fire_due_reminders(_store=store)
        assert [r.id for r in fired] == [past.id]
        # mark_delivered actually flipped on disk.
        assert store.get(past.id).delivered is True
        assert store.get(future.id).delivered is False
        # notify was called exactly once with kind=reminders.fired.
        assert mock_notify.call_count == 1
        event = mock_notify.call_args.args[0]
        assert event.kind == "reminders.fired"
        assert event.payload["reminder_id"] == past.id

    def test_fire_due_does_not_double_fire_across_ticks(self, store):
        create_reminder(trigger_at=_utc_in(-10), message="x", _store=store)
        with patch("bridge.notify.notify"):
            first = fire_due_reminders(_store=store)
            second = fire_due_reminders(_store=store)
        assert len(first) == 1
        assert second == [], "second tick must not re-fire delivered reminder"

    def test_notify_dispatch_failure_does_not_unmark_delivery(self, store):
        """If the dispatcher raises (channel exception, missing config,
        etc.) the reminder STAYS delivered. Reasoning: retrying
        forever on a flaky webhook would spam the user; the audit
        log records the failure, and a manual replay can re-create
        the reminder if needed."""
        rem = create_reminder(trigger_at=_utc_in(-10), message="x", _store=store)
        with patch("bridge.notify.notify",
                   side_effect=RuntimeError("dispatcher broke")):
            fired = fire_due_reminders(_store=store)
        assert [r.id for r in fired] == [rem.id]
        # Marked delivered despite the exception.
        assert store.get(rem.id).delivered is True


# ---------------------------------------------------------------------------
# End-to-end: scheduler → real notify dispatcher → mocked HTTP post
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_full_pipeline_with_notifications_off_is_noop(
        self, store, monkeypatch,
    ):
        """With KARIN_NOTIFICATIONS unset, the dispatcher refuses to
        send anything. The reminder still fires, gets marked
        delivered, and no HTTP post happens."""
        from bridge import features
        from bridge.notify.dispatcher import reset_dispatcher
        monkeypatch.delenv("KARIN_NOTIFICATIONS", raising=False)
        features.reload()
        reset_dispatcher()
        rem = create_reminder(trigger_at=_utc_in(-10), message="x", _store=store)
        with patch("bridge.notify.channels.httpx.post") as post:
            fire_due_reminders(_store=store)
        assert post.call_count == 0
        assert store.get(rem.id).delivered is True

    def test_full_pipeline_pushes_to_configured_channel(
        self, store, monkeypatch,
    ):
        """With notifications on AND a configured channel, fire_due
        produces an HTTP POST to that channel."""
        from bridge import features
        from bridge.notify.dispatcher import reset_dispatcher
        monkeypatch.setenv("KARIN_NOTIFICATIONS", "true")
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://x.test/wh")
        monkeypatch.delenv("KARIN_NOTIFY_NTFY_TOPIC", raising=False)
        features.reload()
        reset_dispatcher()
        create_reminder(trigger_at=_utc_in(-10), message="ping me", _store=store)

        class _R:
            status_code = 204
            text = ""
        with patch("bridge.notify.channels.httpx.post", return_value=_R()) as post:
            fire_due_reminders(_store=store)
        assert post.call_count == 1
        # Sent to the discord webhook URL.
        called_url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs.get("url")
        assert called_url == "https://x.test/wh"
