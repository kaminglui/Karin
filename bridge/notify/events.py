"""Typed event dataclass + severity enum for the notify module.

Kept in its own file so :mod:`bridge.notify.dispatcher` and
:mod:`bridge.notify.channels` can import it without each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    """Three-step ladder. Channel implementations may map this to
    their own native priority field (ntfy supports priority 1-5;
    Discord doesn't have severity but we color-code embeds).

    Higher value = more urgent. Compare with ``>=`` against the
    threshold from rules.
    """

    INFO = 0       # nice to know
    WARNING = 1    # active but not blocking
    CRITICAL = 2   # act now


@dataclass(frozen=True)
class NotifyEvent:
    """One push-worthy event from a subsystem.

    ``kind`` is the dotted event-type string ("alerts.fired",
    "trackers.shock", "news.confirmed", ...). Rules + channel
    routing decisions key off this string + the ``source`` /
    ``severity`` fields. The ``payload`` carries detector-specific
    data for downstream filtering (tracker id, alert category,
    cluster id, etc.) — the channels themselves don't read it.

    Built immutable so the dispatcher can fan out to multiple
    channels without copy-on-write concerns.
    """

    kind: str                # "alerts.fired" | "trackers.shock" | …
    title: str               # short headline ("CRITICAL: NWS Tornado Warning")
    body: str                # longer one-liner with concrete numbers
    severity: Severity = Severity.INFO
    source: str = ""         # subsystem id ("alerts" | "trackers" | "news")
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
