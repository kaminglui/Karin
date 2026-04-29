"""Reminder scheduling tool."""
from __future__ import annotations

import logging

log = logging.getLogger("bridge.tools")


def _schedule_reminder(trigger_at: str, message: str) -> str:
    """LLM-invoked reminder scheduler. Mirrors the regex-detection
    path (bridge/reminders/detect.py) but driven by the model's own
    judgment of which prompts want a reminder. Kept behind the
    ``reminders_llm_tool`` feature flag so users who only want the
    regex path don't pay the ~100 tokens of tool-schema overhead.

    Returns a human-readable status string — the LLM reads it and
    typically paraphrases ("Got it, I'll ping you at 5pm").
    """
    from datetime import datetime, timedelta, timezone
    trigger_at = (trigger_at or "").strip()
    message = (message or "").strip()
    if not trigger_at:
        return "Error: trigger_at is required (ISO 8601 UTC)."
    if not message:
        return "Error: message is required."
    # Accept both "…Z" and "…+00:00" forms. datetime.fromisoformat
    # handles the latter natively; strip the Z for Python < 3.11.
    iso = trigger_at.replace("Z", "+00:00") if trigger_at.endswith("Z") else trigger_at
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return (
            f"Error: couldn't parse trigger_at={trigger_at!r}. "
            "Expected ISO 8601 UTC, e.g. '2026-04-16T22:00:00Z'."
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if dt <= now + timedelta(seconds=30):
        return (
            f"Error: trigger_at {dt.isoformat()} is in the past "
            f"or too soon (server time {now.isoformat()})."
        )
    if dt > now + timedelta(days=365):
        return (
            f"Error: trigger_at {dt.isoformat()} is more than a year "
            "out — refusing. Ask the user for confirmation."
        )
    try:
        from bridge.reminders import create_reminder
        from bridge.reminders.api import _get_store
        store = _get_store()
        # Check for a near-duplicate (e.g. the regex detector created
        # the same reminder seconds ago on this same chat turn). If
        # one exists and the new details match: no-op. If they differ:
        # update the existing row instead of creating a second one.
        existing = store.find_similar(message, dt)
        if existing is not None:
            old_trigger = existing.trigger_at
            old_msg = existing.message.strip().lower()
            new_msg = message.strip().lower()
            if old_msg == new_msg and abs((old_trigger - dt).total_seconds()) < 60:
                return (
                    f"Already scheduled: '{existing.message}' at "
                    f"{existing.trigger_at.isoformat()} (id={existing.id}). "
                    "No duplicate created."
                )
            # Details differ — update the existing row rather than dup.
            store.update(existing.id, trigger_at=dt, message=message)
            log.info(
                "schedule_reminder: updated existing %s (was %r@%s, now %r@%s)",
                existing.id, existing.message, old_trigger, message, dt,
            )
            return (
                f"Updated existing reminder {existing.id}: "
                f"'{message}' at {dt.isoformat()}."
            )
        rem = create_reminder(trigger_at=dt, message=message, source="llm_tool")
    except Exception as e:
        log.warning("schedule_reminder tool: create_reminder raised: %s", e)
        return f"Error: couldn't schedule reminder: {e}"
    return (
        f"Reminder scheduled: '{rem.message}' at {dt.isoformat()} "
        f"(id={rem.id})."
    )

