"""Digest data types.

These are DTOs, not LLM-facing. They carry pre-computed strings so
the API / UI / LLM tool all read from the same source of truth. Any
LLM involvement happens at a later layer (reading a ready-made
`voice_line`) — never in producing the digest itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class DigestNewsItem:
    """One confirmed (or provisionally confirmed) news brief."""

    cluster_id: str
    state: str                    # "confirmed" | "provisionally_confirmed"
    title: str                    # centroid display title
    voice_line: str               # the short paraphrasable sentence
    sources: list[str]            # display names (e.g. ["AP", "BBC", "NPR"])
    independent_confirmations: int
    latest_update_at: datetime


@dataclass(frozen=True)
class DigestAlertItem:
    """One currently-active alert from the alerts engine."""

    alert_id: str
    level: str                    # "INFO" | "WATCH" | "ADVISORY" | "CRITICAL"
    category: str
    title: str
    reasoning_bullets: list[str]
    created_at: datetime
    cooldown_until: datetime


@dataclass(frozen=True)
class DigestTrackerItem:
    """One tracker that moved meaningfully since yesterday."""

    tracker_id: str
    label: str
    latest_value: float
    change_1d_pct: float | None   # daily percent change (nullable by cadence)
    change_1w_pct: float | None
    direction_label: str | None   # "up" / "down" / "flat" / None
    shock_label: str | None       # Phase 5.2 shock indicator if fired


@dataclass(frozen=True)
class DigestSnapshot:
    """One day's digest — the whole payload the UI and tool render from.

    `generated_at` stamps when this snapshot was built. The same day
    can have multiple snapshots (the poller refreshes hourly); each
    write OVERWRITES `data/digest/YYYY-MM-DD.json` so the stored
    version is always the newest view.

    `headline` is a plain-English one-liner the LLM can paraphrase
    without summarizing further — e.g. "3 confirmed stories, 2 active
    alerts, and gold is up 2%."
    """

    generated_at: datetime
    date_key: str                       # "2026-04-15" — day this digest covers
    headline: str
    news: list[DigestNewsItem] = field(default_factory=list)
    alerts: list[DigestAlertItem] = field(default_factory=list)
    trackers: list[DigestTrackerItem] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.news or self.alerts or self.trackers)
