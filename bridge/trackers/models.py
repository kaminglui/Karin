"""Typed data models for bridge/trackers.

Four dataclasses:

  - TrackerConfig:    user-curated per-series metadata (from config file).
  - TrackerReading:   one observation in a series. Immutable.
  - TrackerRecord:    persisted state — identity + history + last fetch meta.
  - TrackerSnapshot:  user-facing view — latest + deltas + staleness. Not persisted.

All datetimes MUST be timezone-aware UTC. The store parses ISO-8601 and
coerces naive datetimes to UTC on load; fetchers stamp UTC explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class TrackerConfig:
    """One tracker's configuration. Loaded from trackers.json at startup."""

    id: str                         # unique, e.g. "usd_cny"
    label: str                      # display name, e.g. "USD/CNY"
    category: str                   # "fx" | "metal" | "food_index"
    source: str                     # dispatch key: "frankfurter" | "stooq" | "bls"
    params: dict[str, Any]          # source-specific args (e.g. {"from": "USD", "to": "CNY"})
    cadence: str                    # "daily" | "monthly" — drives TTL + 1d-delta semantics
    stale_after_hours: int          # read-side staleness threshold
    history_days: int               # retention window; older readings pruned on write
    enabled: bool = True


@dataclass(frozen=True)
class TrackerReading:
    """One observation in a series. Persisted inside TrackerRecord.history."""

    timestamp: datetime             # UTC, point in time of the observation
    value: float


@dataclass
class TrackerRecord:
    """Persisted state for one tracker. Mutable — history appends in place.

    last_fetch_error is set on failed fetches and cleared on success; lets
    readers surface "why is this tracker empty" without a separate channel.
    """

    id: str
    label: str
    category: str
    history: list[TrackerReading] = field(default_factory=list)
    last_fetched_at: datetime | None = None
    last_fetch_error: str | None = None


@dataclass(frozen=True)
class TrackerSnapshot:
    """User-facing snapshot. Computed at read time from a TrackerRecord.

    All delta fields are nullable: a brand-new tracker with only one
    reading has no history to diff against, and a monthly series has no
    meaningful 1d delta by construction.

    Phase 5.2 adds four derived labels. Like the delta fields, they are
    computed at read time (see service._build_snapshot) and are all
    nullable — None when history is too short to estimate volatility,
    or when the cadence doesn't support daily-style labels (monthly).
    """

    id: str
    label: str
    category: str
    latest_value: float | None
    latest_timestamp: datetime | None
    change_1d: float | None             # absolute delta vs. ~1 day ago
    change_1d_pct: float | None         # percentage delta (0-100 scale)
    change_1w: float | None
    change_1w_pct: float | None
    change_1m: float | None
    change_1m_pct: float | None
    is_stale: bool
    note: str = ""                      # "no history", last fetch error, etc.
    # Phase 5.2 derived labels. "up" | "down" | "flat" for directions;
    # "stable" | "moving" | "volatile" for movement; "surging" |
    # "plunging" for shock. All optional; see service helpers for rules.
    direction_1d: str | None = None
    direction_1w: str | None = None
    movement_label: str | None = None
    shock_label: str | None = None
