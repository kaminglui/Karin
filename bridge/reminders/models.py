"""Typed Reminder dataclass. Mirrors the SQLite row shape exactly so
the store can hand back fully-typed objects without a translation
layer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Reminder:
    """One scheduled reminder.

    ``id`` is a short URL-safe token assigned by the store on create
    (so the API layer doesn't need to think about ID generation).

    ``trigger_at`` is timezone-aware UTC. The store enforces this
    invariant on insert; queries always return UTC.

    ``created_at`` is wall-clock UTC at insert time, useful for
    dedupe/audit even though we don't currently expose it.

    ``delivered`` flips to True the moment the scheduler successfully
    fires the reminder (regardless of whether downstream channels
    accepted it — the dispatcher's own logging tracks that). This
    prevents double-fires across scheduler ticks AND across container
    restarts.
    """

    id: str
    trigger_at: datetime
    message: str
    source: str = "manual"   # "manual" | "chat_detect" | "llm" | future
    created_at: datetime | None = None
    delivered: bool = False
