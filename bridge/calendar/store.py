"""Dedup ledger: "did we already notify for this event start?"

Single-table SQLite, same pattern as :mod:`bridge.reminders.store`.
Key is ``(uid, start_iso)`` so a rescheduled meeting (same uid, new
start time) re-fires instead of being treated as a dup.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

log = logging.getLogger("bridge.calendar.store")


class CalendarDedupeStore:
    """Records which (uid, start) pairs have already been notified.

    Each row exists for about 30 days after the event fires (cleaned
    up on the next write so the table doesn't grow forever). The
    caller doesn't need to reason about this — ``mark_notified``
    returns False when it was a duplicate, True when it just
    inserted a fresh row.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS notified_events (
            key          TEXT PRIMARY KEY,
            uid          TEXT NOT NULL,
            start_iso    TEXT NOT NULL,
            summary      TEXT,
            notified_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_notified_start
            ON notified_events (start_iso);
    """

    _GC_WINDOW_DAYS = 30   # purge rows whose start is older than this

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # WAL + busy_timeout — the calendar poller writes here while
        # API reads may happen in parallel. See bridge/reminders/store.py
        # for the same rationale.
        conn = sqlite3.connect(self._path, isolation_level=None, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            yield conn
        finally:
            conn.close()

    def already_notified(self, key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM notified_events WHERE key = ?", (key,),
            ).fetchone()
        return row is not None

    def mark_notified(
        self,
        key: str, uid: str, start_utc: datetime, summary: str,
    ) -> bool:
        """Insert; return True on fresh insert, False on duplicate.
        Also GCs old rows opportunistically."""
        start_iso = start_utc.isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO notified_events "
                    "(key, uid, start_iso, summary, notified_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, uid, start_iso, summary, now_iso),
                )
            except sqlite3.IntegrityError:
                return False
            # Opportunistic GC. Cheap (indexed DELETE), runs at most
            # once per mark → worst case a handful of ops per poll.
            cutoff = _cutoff_iso(self._GC_WINDOW_DAYS)
            conn.execute(
                "DELETE FROM notified_events WHERE start_iso < ?",
                (cutoff,),
            )
        return True


def _cutoff_iso(days_ago: int) -> str:
    from datetime import timedelta
    return (
        datetime.now(timezone.utc) - timedelta(days=days_ago)
    ).isoformat()
