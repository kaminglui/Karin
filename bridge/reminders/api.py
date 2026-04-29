"""Public reminder API + scheduler tick.

Thin wrapper over :class:`bridge.reminders.store.ReminderStore`. The
indirection exists because the chat-detection layer (Phase 3) and a
future ``/reminder`` tool will both call into here, and we want the
notify dispatch + delivery marking to happen exactly once per
reminder regardless of caller.

Singleton store: ``data/reminders/reminders.db`` lives next to the
other persistent state (`data/news/`, `data/digest/`, `data/alerts/`).
Tests can pass an explicit store via the ``_store`` argument to
override.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from bridge.reminders.models import Reminder
from bridge.reminders.store import ReminderStore

log = logging.getLogger("bridge.reminders.api")


_default_store: ReminderStore | None = None


def _get_store() -> ReminderStore:
    """Lazy singleton. Tests/REPL: call :func:`reset_default_store`
    after monkey-patching the path.

    Phase H: the DB lives inside the active profile's reminders dir so
    two profiles on the same box keep independent reminder sets."""
    global _default_store
    if _default_store is None:
        from bridge.profiles import active_profile
        profile = active_profile()
        path = profile.reminders_dir / "reminders.db"
        _default_store = ReminderStore(path)
    return _default_store


def reset_default_store() -> None:
    """Drop the cached store. Tests use this after relocating the DB."""
    global _default_store
    _default_store = None


def create_reminder(
    *,
    trigger_at: datetime,
    message: str,
    source: str = "manual",
    _store: ReminderStore | None = None,
) -> Reminder:
    """Schedule a reminder. ``trigger_at`` must be timezone-aware UTC.

    Returns the persisted Reminder (with ``id`` assigned). The
    scheduler poller picks it up on its next tick after trigger_at.
    """
    store = _store or _get_store()
    rem = store.create(trigger_at=trigger_at, message=message, source=source)
    log.info(
        "reminder created: id=%s trigger_at=%s source=%s",
        rem.id, rem.trigger_at.isoformat(), rem.source,
    )
    return rem


def cancel_reminder(
    reminder_id: str, *, _store: ReminderStore | None = None,
) -> bool:
    """Remove a reminder by id. Returns True iff a row was deleted."""
    store = _store or _get_store()
    ok = store.cancel(reminder_id)
    if ok:
        log.info("reminder cancelled: id=%s", reminder_id)
    return ok


def list_upcoming(
    *, limit: int = 50, _store: ReminderStore | None = None,
) -> list[Reminder]:
    """Pending reminders, soonest-first."""
    return (_store or _get_store()).list_upcoming(limit=limit)


def fire_due_reminders(
    *,
    now: datetime | None = None,
    _store: ReminderStore | None = None,
) -> list[Reminder]:
    """Scheduler entry point. Fetch every reminder whose trigger_at
    has passed, push each through :func:`bridge.notify.notify`, mark
    delivered atomically. Returns the list of reminders that fired
    (post-marking).

    Flow per reminder:

    1. Mark delivered first — this is the atomic guard against a
       second tick double-firing the same row. If the marking fails
       (someone else got there first), skip.
    2. Build + dispatch the NotifyEvent. Failure here is logged; the
       reminder STAYS marked as delivered (we don't want to retry
       indefinitely on a flaky webhook). The append-only event log
       in `bridge/notify` already records dispatch failures for
       audit / manual replay.
    """
    store = _store or _get_store()
    due = store.fetch_due(now=now)
    fired: list[Reminder] = []
    for rem in due:
        if not store.mark_delivered(rem.id):
            # Someone else fired this one between fetch_due and now.
            continue
        try:
            _dispatch(rem)
        except Exception as e:
            log.warning("notify dispatch raised on reminder %s: %s", rem.id, e)
        fired.append(rem)
    if fired:
        log.info("scheduler fired %d reminder(s)", len(fired))
    return fired


def _dispatch(rem: Reminder) -> None:
    """Build a NotifyEvent for a fired reminder + hand to the
    notify dispatcher. Severity=WARNING so it's visible without
    bypassing Do Not Disturb (CRITICAL would). dedupe_key=id keeps
    the dispatcher's cooldown ledger from collapsing distinct
    reminders that share text."""
    from bridge.notify import NotifyEvent, notify
    from bridge.notify.events import Severity
    notify(NotifyEvent(
        kind="reminders.fired",
        title=f"⏰ {rem.message[:180]}",
        body=rem.message,
        severity=Severity.WARNING,
        source="reminders",
        payload={
            "dedupe_key": rem.id,
            "reminder_id": rem.id,
            "reminder_source": rem.source,
        },
        timestamp=rem.trigger_at,
    ))
