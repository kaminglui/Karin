"""Typed data models for bridge/alerts.

Five dataclasses + two enums:

  - SignalKind / Signal      pure observations produced by detectors
  - AlertLevel / AlertCategory / Alert   structured emission produced by rules
  - ScanResult               what AlertService.scan() returns to callers

Signals carry no severity. Rules decide severity. That keeps the
signal→rule interface narrow and inspectable: you can always tell which
signals were observed and which rule chose to elevate.

All datetimes UTC, timezone-aware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any


class SignalKind(str, Enum):
    """Typed observation categories. Extend when a new detector is added."""

    # From tracker snapshots (Phase 5.2 labels)
    TRACKER_SHOCK = "tracker_shock"
    TRACKER_VOLATILE = "tracker_volatile"
    TRACKER_DIRECTION_1W = "tracker_direction_1w"

    # From news subsystem (cluster + preferences intersection)
    NEWS_WATCHLIST_MATCH = "news_watchlist_match"

    # From external sources
    TRAVEL_ADVISORY_CHANGED = "travel_advisory_changed"
    NWS_WEATHER_ALERT = "nws_weather_alert"


@dataclass(frozen=True)
class Signal:
    """One observation. Pure; no severity, no interpretation.

    `source` is a stable identifier suitable for logging and
    source_attribution on emitted alerts. Examples:
      "tracker:usd_cny"
      "news:cluster:a1b2c3d4"
      "external:travel:EGY"

    `payload` is detector-specific data. Rules read it; the engine
    does not interpret it.
    """

    kind: SignalKind
    source: str
    payload: dict[str, Any]
    observed_at: datetime


class AlertLevel(IntEnum):
    """Four-level severity. IntEnum so rules can easily compute max()."""

    INFO = 0       # worth knowing
    WATCH = 1      # watch this space
    ADVISORY = 2   # practical impact likely
    CRITICAL = 3   # direct impact; act soon


class AlertCategory(str, Enum):
    ENERGY = "energy"
    TRAVEL = "travel"
    MACRO = "macro"
    GEOPOLITICAL = "geopolitical"
    WATCHLIST = "watchlist"
    MARKET_SHOCK = "market_shock"
    WEATHER = "weather"


@dataclass(frozen=True)
class Alert:
    """Structured emission from a rule. Persisted append-only.

    `is_active` is a DERIVED field. Freshly emitted alerts get True;
    when an Alert is rehydrated from the ledger, the loader computes it
    from `cooldown_until > now` so callers (and any future UI) can trust
    the value without having to re-derive it themselves. The on-disk
    value is informational only; the loader always recomputes.
    """

    alert_id: str                         # sha1(rule_id + scope_key + iso(created_at))[:16]
    level: AlertLevel
    category: AlertCategory
    title: str
    reasoning_bullets: list[str]
    triggered_by_signals: list[Signal]
    source_attribution: list[str]
    affected_domains: list[str]
    rule_id: str
    scope_key: str
    created_at: datetime
    cooldown_until: datetime
    is_active: bool = True   # derived; recomputed at deserialize time, see service._alert_from_dict


@dataclass(frozen=True)
class SuppressedAlertRecord:
    """What the engine logs when a rule would have fired but cooldown held it back.

    Stored as an `alert_suppressed` event in alerts.jsonl so the audit
    trail is complete — you can always answer "why didn't this fire?"
    """

    rule_id: str
    scope_key: str
    candidate_title: str
    last_fired_at: datetime
    suppressed_at: datetime


@dataclass(frozen=True)
class ScanResult:
    """Return type for AlertService.scan(). Structured, not a string."""

    alerts: list[Alert] = field(default_factory=list)
    suppressed: list[SuppressedAlertRecord] = field(default_factory=list)
    signals_considered: int = 0
    skipped_due_to_cache: bool = False
