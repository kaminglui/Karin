"""User reminders: persistent, scheduler-fired, push-notified.

Phase 1 scope:

* SQLite-backed persistence — survives restarts, resilient to the
  scheduler thread dying (poller respawns on next interval).
* Public API: create / cancel / list / fire_due.
* Background scheduler in :mod:`bridge.pollers` that calls fire_due
  every 60 s and pushes due reminders through
  :func:`bridge.notify.notify`.

Out of scope here (separate phases):

* Inline UI card with undo (chat-stream integration).
* Auto-detection from chat ("remind me at 5pm to ...") via regex
  or LLM fallback.
* Calendar (`.ics`) integration.

The store is intentionally minimal — id + trigger time + message +
delivered flag. Recurrence, snooze, attachments are deferred until
we have data on what users actually request.
"""
from __future__ import annotations

from .api import (
    cancel_reminder,
    create_reminder,
    fire_due_reminders,
    list_upcoming,
)
from .models import Reminder

__all__ = [
    "Reminder",
    "create_reminder",
    "cancel_reminder",
    "list_upcoming",
    "fire_due_reminders",
]
