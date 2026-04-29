"""The poll tick: fetch configured calendars, fire notifications for
events crossing the lead-time window, update dedup ledger.

Called from :mod:`bridge.pollers` at the cadence set in
``config/calendar.yaml`` (default every 10 min).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bridge.calendar.fetch import CalendarEvent, fetch_events
from bridge.calendar.store import CalendarDedupeStore

log = logging.getLogger("bridge.calendar.poll")


@dataclass(frozen=True)
class CalendarConfig:
    """One calendar feed + how we handle it."""
    label: str
    url: str
    lead_time_minutes: int = 15


def _load_configs() -> list[CalendarConfig]:
    """Read ``config/calendar.yaml`` + resolve URL env vars.

    YAML shape (all fields optional)::

        lead_time_minutes: 15     # default for every calendar below
        calendars:
          - label: "work"
            url_env: "KARIN_CALENDAR_WORK_URL"
          - label: "personal"
            url_env: "KARIN_CALENDAR_PERSONAL_URL"
            lead_time_minutes: 30  # override per-calendar

    URLs live in env vars — never checked in — so a multi-calendar
    setup has one KARIN_CALENDAR_<NAME>_URL per feed. Missing env →
    that calendar is silently skipped at load time (same pattern
    as notify channels)."""
    from bridge.utils import REPO_ROOT, load_config
    path = REPO_ROOT / "config" / "calendar.yaml"
    if not path.exists():
        return []
    try:
        raw = load_config(path) or {}
    except Exception as e:
        log.warning("failed to parse %s: %s — calendar disabled", path, e)
        return []
    if not isinstance(raw, dict):
        log.warning("%s root must be a mapping — calendar disabled", path)
        return []
    default_lead = int(raw.get("lead_time_minutes", 15))
    out: list[CalendarConfig] = []
    for entry in raw.get("calendars") or []:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "calendar")
        env_key = str(entry.get("url_env") or "").strip()
        if not env_key:
            log.warning("calendar %r missing url_env — skipping", label)
            continue
        url = (os.environ.get(env_key) or "").strip()
        if not url:
            log.info(
                "calendar %r env var %s unset — skipping",
                label, env_key,
            )
            continue
        lead = int(entry.get("lead_time_minutes", default_lead))
        out.append(CalendarConfig(label=label, url=url, lead_time_minutes=lead))
    return out


_dedupe_store: CalendarDedupeStore | None = None


def _get_store() -> CalendarDedupeStore:
    global _dedupe_store
    if _dedupe_store is None:
        from bridge.profiles import active_profile
        _dedupe_store = CalendarDedupeStore(
            active_profile().root / "calendar" / "notified.db",
        )
    return _dedupe_store


def reset_default_store() -> None:
    """Tests only — drop the cached ledger so a tmp_path one binds."""
    global _dedupe_store
    _dedupe_store = None


def tick(
    *,
    now: datetime | None = None,
    configs: list[CalendarConfig] | None = None,
    store: CalendarDedupeStore | None = None,
) -> int:
    """One poll cycle. Returns the number of events that actually
    fired a notify (already-notified ones don't count).

    Tolerates partial failures: a broken feed logs a warning and the
    other calendars still run. Never raises — the poller wrapper
    only sees an int."""
    ref_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cfgs = configs if configs is not None else _load_configs()
    if not cfgs:
        return 0
    ledger = store if store is not None else _get_store()
    fired = 0
    for cfg in cfgs:
        try:
            events = fetch_events(cfg.url, label=cfg.label)
        except Exception as e:
            log.warning("calendar %r fetch crashed: %s", cfg.label, e)
            continue
        lead = timedelta(minutes=cfg.lead_time_minutes)
        # We want events starting within the next `lead` minutes (so
        # "in 15 min or less"). Past events and far-future events
        # are skipped here; the ledger dedups events already fired.
        window_end = ref_now + lead
        for ev in events:
            if ev.start_utc <= ref_now:
                continue
            if ev.start_utc > window_end:
                continue
            key = ev.dedupe_key()
            if ledger.already_notified(key):
                continue
            if ledger.mark_notified(key, ev.uid, ev.start_utc, ev.summary):
                _dispatch(ev, ref_now)
                fired += 1
    if fired:
        log.info("calendar poll: fired %d notification(s)", fired)
    return fired


def _dispatch(ev: CalendarEvent, ref_now: datetime) -> None:
    """Build + push a :class:`bridge.notify.NotifyEvent`. Severity
    is WARNING — important enough to surface without bypassing Do
    Not Disturb (CRITICAL would)."""
    try:
        from bridge.notify import NotifyEvent, notify
        from bridge.notify.events import Severity
    except Exception as e:
        log.warning("notify import failed: %s", e)
        return
    minutes_out = max(1, int((ev.start_utc - ref_now).total_seconds() // 60))
    label_part = f"[{ev.calendar_label}] " if ev.calendar_label else ""
    title = f"{label_part}{ev.summary} in {minutes_out} min"
    local_time = ev.start_utc.astimezone()   # browser-side relevant; server default
    body = f"{ev.summary} — starts {local_time.strftime('%H:%M %Z')}."
    try:
        notify(NotifyEvent(
            kind="calendar.upcoming",
            title=title[:200],
            body=body,
            severity=Severity.WARNING,
            source="calendar",
            payload={
                "dedupe_key": ev.dedupe_key(),
                "uid": ev.uid,
                "start_utc": ev.start_utc.isoformat(),
                "calendar_label": ev.calendar_label,
            },
            timestamp=ev.start_utc,
        ))
    except Exception as e:
        log.warning("calendar notify dispatch raised on %s: %s", ev.uid, e)
