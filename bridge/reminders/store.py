"""SQLite-backed persistence for reminders.

Single-table schema, two indexes (one on `trigger_at` for the
scheduler's "what's due" scan, one on `delivered` to keep the scan
cheap as the table grows).

Connection lifecycle: each public method opens its own connection
(``sqlite3.connect(self._path)``) rather than sharing one across
threads. The scheduler poller and any future API endpoint may both
hit the store concurrently, and SQLite's per-connection threading
restriction makes "open per call" simpler than a thread-local pool.
At our scale (handful of reminders, 60 s tick) the open-cost is
microseconds.

UTC discipline: every datetime in/out is timezone-aware UTC. The
store rejects naive datetimes on insert (``ValueError``) so a
caller bug never persists ambiguous timestamps.
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from bridge.reminders.models import Reminder

log = logging.getLogger("bridge.reminders.store")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime, *, field: str) -> datetime:
    if dt.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware UTC, got naive {dt!r}")
    return dt.astimezone(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return _ensure_utc(dt, field="datetime").isoformat()


def _from_iso(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class ReminderStore:
    """Persistent home for all reminders. Thread-safe by virtue of
    each method opening its own connection."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS reminders (
            id            TEXT PRIMARY KEY,
            trigger_at    TEXT NOT NULL,
            message       TEXT NOT NULL,
            source        TEXT NOT NULL DEFAULT 'manual',
            created_at    TEXT NOT NULL,
            delivered     INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_reminders_trigger
            ON reminders (trigger_at);
        CREATE INDEX IF NOT EXISTS idx_reminders_pending
            ON reminders (delivered, trigger_at);
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # isolation_level=None puts us in autocommit mode; we wrap
        # multi-statement work in explicit BEGIN/COMMIT where it
        # matters (mark_delivered uses an atomic single-statement
        # update and doesn't need a transaction).
        #
        # WAL + busy_timeout: the reminders poller hits this every 60s
        # while the UI may concurrently read pending reminders. The
        # default rollback journal serializes all access, which on the
        # Jetson SD card surfaces as ``OperationalError: database is
        # locked``. WAL lets readers and the poller proceed in parallel;
        # the busy_timeout absorbs the brief moments they do contend.
        conn = sqlite3.connect(self._path, isolation_level=None, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            yield conn
        finally:
            conn.close()

    # --- CRUD --------------------------------------------------------------

    def create(
        self,
        *,
        trigger_at: datetime,
        message: str,
        source: str = "manual",
    ) -> Reminder:
        """Insert a new reminder. Returns the persisted row with its
        store-assigned id + timestamps. Naive trigger_at raises."""
        trigger_at = _ensure_utc(trigger_at, field="trigger_at")
        if not message or not message.strip():
            raise ValueError("reminder message cannot be empty")
        rid = secrets.token_urlsafe(8)
        now = _now_utc()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reminders "
                "(id, trigger_at, message, source, created_at, delivered) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (rid, _to_iso(trigger_at), message.strip(), source, _to_iso(now)),
            )
        return Reminder(
            id=rid,
            trigger_at=trigger_at,
            message=message.strip(),
            source=source,
            created_at=now,
            delivered=False,
        )

    def get(self, reminder_id: str) -> Reminder | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reminders WHERE id = ?", (reminder_id,),
            ).fetchone()
        return self._row_to_reminder(row) if row else None

    def cancel(self, reminder_id: str) -> bool:
        """Delete a reminder by id. Returns True if a row was removed.
        Cancelling a delivered reminder still removes it (we treat
        this as a "clear from history" action)."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM reminders WHERE id = ?", (reminder_id,),
            )
        return cur.rowcount > 0

    def list_upcoming(self, *, limit: int = 50) -> list[Reminder]:
        """All undelivered reminders ordered by trigger_at ASC."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders "
                "WHERE delivered = 0 "
                "ORDER BY trigger_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def fetch_due(self, *, now: datetime | None = None) -> list[Reminder]:
        """Undelivered reminders with trigger_at <= now. The scheduler
        calls this every tick, fires what comes back, then calls
        :meth:`mark_delivered` to flip the flag."""
        ref = _ensure_utc(now or _now_utc(), field="now")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders "
                "WHERE delivered = 0 AND trigger_at <= ? "
                "ORDER BY trigger_at ASC",
                (_to_iso(ref),),
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def mark_delivered(self, reminder_id: str) -> bool:
        """Atomic: only flips the flag if it was still undelivered.
        Guards against a double-fire if two scheduler ticks ever
        overlap (they shouldn't, but cheap to defend)."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE reminders SET delivered = 1 "
                "WHERE id = ? AND delivered = 0",
                (reminder_id,),
            )
        return cur.rowcount > 0

    def find_similar(
        self,
        message: str,
        trigger_at: datetime,
        *,
        window_s: int = 300,
    ) -> Reminder | None:
        """Find an undelivered reminder whose lowercased message matches
        and whose trigger_at is within ``window_s`` seconds of the
        given timestamp. Returns the best match (closest in time) or
        None.

        Used by the ``schedule_reminder`` tool to detect a near-
        duplicate created by the regex detector seconds earlier on
        the same user turn. Avoids double-firing without requiring
        callers to coordinate explicitly."""
        norm_msg = message.strip().lower()
        t_iso = _to_iso(trigger_at)
        dt_low = _to_iso(trigger_at - timedelta(seconds=window_s))
        dt_high = _to_iso(trigger_at + timedelta(seconds=window_s))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders "
                "WHERE delivered = 0 "
                "  AND trigger_at >= ? AND trigger_at <= ?",
                (dt_low, dt_high),
            ).fetchall()
        for row in rows:
            rem = self._row_to_reminder(row)
            old_norm = rem.message.strip().lower()
            # Match if either message is a prefix of the other. This
            # catches the "LLM adds detail" case: regex created "call
            # mom", LLM refines to "call mom about dinner plans" →
            # prefix hit, same intent, update instead of dup.
            if (
                old_norm == norm_msg
                or old_norm.startswith(norm_msg)
                or norm_msg.startswith(old_norm)
            ):
                return rem
        return None

    def update(
        self,
        reminder_id: str,
        *,
        trigger_at: datetime | None = None,
        message: str | None = None,
    ) -> bool:
        """Update one or both fields of an existing reminder. Returns
        True if the row was found + modified. Does NOT touch the
        ``delivered`` flag — that's the scheduler's job."""
        sets: list[str] = []
        vals: list = []
        if trigger_at is not None:
            trigger_at = _ensure_utc(trigger_at, field="trigger_at")
            sets.append("trigger_at = ?")
            vals.append(_to_iso(trigger_at))
        if message is not None:
            message = message.strip()
            if not message:
                raise ValueError("message cannot be empty")
            sets.append("message = ?")
            vals.append(message)
        if not sets:
            return False
        vals.append(reminder_id)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE reminders SET {', '.join(sets)} WHERE id = ?",
                vals,
            )
        return cur.rowcount > 0

    # --- helpers -----------------------------------------------------------

    def _row_to_reminder(self, row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=row["id"],
            trigger_at=_from_iso(row["trigger_at"]),  # type: ignore[arg-type]
            message=row["message"],
            source=row["source"] or "manual",
            created_at=_from_iso(row["created_at"]),
            delivered=bool(row["delivered"]),
        )
