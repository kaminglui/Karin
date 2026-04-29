"""Signal detectors: subsystem output -> typed Signal list.

Each detector is a pure function. No I/O, no side effects. Returns a
list of Signals (possibly empty). Callers decide what to do with them.

STALE-INPUT FILTERING (Phase 6 constraint):
Stale inputs (tracker.is_stale or cluster.is_stale) produce ZERO signals.
The filter lives here — in detectors — so rules stay simple and never
have to know about freshness semantics. If a stale input should start
producing signals again in some future version, it's a one-line change
in the detector, not a spread-out change across rules.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from bridge.alerts.models import Signal, SignalKind
from bridge.news.models import NormalizedArticle, StoryCluster
from bridge.news.preferences import Preferences, match_watchlist_items
from bridge.trackers.models import TrackerSnapshot


# --- tracker signals -------------------------------------------------------

def signals_from_trackers(
    snapshots: Iterable[TrackerSnapshot],
    now: datetime,
) -> list[Signal]:
    """Convert tracker snapshots into TRACKER_* signals.

    Stale trackers are skipped entirely — no signals emitted. A tracker
    with is_stale=True is presumed to have data too old to drive any
    alert, regardless of its derived labels.

    Emits up to 3 signals per snapshot (shock, volatile, direction_1w)
    when the respective Phase 5.2 labels are set.
    """
    signals: list[Signal] = []
    for snap in snapshots:
        if snap.is_stale:
            continue
        source = f"tracker:{snap.id}"
        if snap.shock_label is not None:
            signals.append(Signal(
                kind=SignalKind.TRACKER_SHOCK,
                source=source,
                payload={
                    "tracker_id": snap.id,
                    "label": snap.label,
                    "category": snap.category,
                    "direction": snap.shock_label,  # "surging" | "plunging"
                    "change_pct": snap.change_1d_pct,
                    "latest_value": snap.latest_value,
                },
                observed_at=now,
            ))
        if snap.movement_label == "volatile":
            signals.append(Signal(
                kind=SignalKind.TRACKER_VOLATILE,
                source=source,
                payload={
                    "tracker_id": snap.id,
                    "label": snap.label,
                    "category": snap.category,
                },
                observed_at=now,
            ))
        if snap.direction_1w in ("up", "down") and snap.change_1w_pct is not None:
            signals.append(Signal(
                kind=SignalKind.TRACKER_DIRECTION_1W,
                source=source,
                payload={
                    "tracker_id": snap.id,
                    "label": snap.label,
                    "category": snap.category,
                    "direction": snap.direction_1w,
                    "change_pct": snap.change_1w_pct,
                },
                observed_at=now,
            ))
    return signals


# --- news signals ----------------------------------------------------------

def signals_from_news(
    clusters: dict[str, StoryCluster],
    articles: dict[str, NormalizedArticle],
    preferences: Preferences,
    now: datetime,
) -> list[Signal]:
    """Convert news clusters + user watchlists into NEWS_WATCHLIST_MATCH signals.

    Stale clusters (cluster.is_stale=True) are skipped entirely. A stale
    cluster's last_checked_at is too old for us to trust the state engine's
    decision, so downstream rules should not act on it.

    If preferences are disabled or a cluster matches zero watchlist items,
    no signal is emitted for that cluster. Otherwise one signal per match
    (so a cluster matching both "region:us" and "event:ukraine_war"
    produces two signals, each scoped to its own watchlist).
    """
    if not preferences.enabled:
        return []
    signals: list[Signal] = []
    for cluster in clusters.values():
        if cluster.is_stale:
            continue
        matches = match_watchlist_items(cluster, articles, preferences)
        if not matches:
            continue
        for match in matches:
            signals.append(Signal(
                kind=SignalKind.NEWS_WATCHLIST_MATCH,
                source=f"news:cluster:{cluster.cluster_id}",
                payload={
                    "cluster_id": cluster.cluster_id,
                    "cluster_state": cluster.state.value,
                    "headline": cluster.centroid_display_title,
                    "watchlist_type": match.watchlist_type,
                    "watchlist_id": match.item_id,
                    "watchlist_label": match.item_label,
                    "watchlist_priority": match.priority,
                    "independent_confirmation_count": cluster.independent_confirmation_count,
                },
                observed_at=now,
            ))
    return signals


# --- external signals ------------------------------------------------------

def signals_from_advisory_changes(
    changes: list[dict],
    now: datetime,
) -> list[Signal]:
    """Convert travel-advisory change records into TRAVEL_ADVISORY_CHANGED signals.

    `changes` comes from advisory_fetch.diff_advisories(). Each dict:
        {"country": "EGY", "old_level": 2, "new_level": 3, "title": "Egypt — Level 3: Reconsider Travel"}

    This detector is trivially thin because advisory_fetch already does
    the heavy lifting (parsing RSS, diffing against prior state). It
    exists as a separate function purely so the signal-emission point
    is in detectors.py for consistency.
    """
    signals: list[Signal] = []
    for change in changes:
        country = change.get("country") or ""
        if not country:
            continue
        signals.append(Signal(
            kind=SignalKind.TRAVEL_ADVISORY_CHANGED,
            source=f"external:travel:{country}",
            payload={
                "country": country,
                "old_level": change.get("old_level"),
                "new_level": change.get("new_level"),
                "title": change.get("title", ""),
            },
            observed_at=now,
        ))
    return signals


def signals_from_nws(
    alerts: list[dict],
    now: datetime,
) -> list[Signal]:
    """Convert NWS active-alert dicts into NWS_WEATHER_ALERT signals.

    Already-filtered input: the fetcher drops Minor / Unknown severities,
    so everything reaching this function is Moderate or above. We still
    pass severity through the payload so the rule can choose AlertLevel.

    Each signal's `source` is keyed on the NWS alert id so the engine's
    per-rule cooldown deduplicates naturally: the same warning will
    produce the same signal every poll, but the cooldown prevents the
    same alert_id from firing twice.
    """
    signals: list[Signal] = []
    for a in alerts:
        alert_id = a.get("id") or ""
        event = a.get("event") or ""
        if not alert_id or not event:
            continue
        signals.append(Signal(
            kind=SignalKind.NWS_WEATHER_ALERT,
            source=f"external:nws:{alert_id}",
            payload={
                "id": alert_id,
                "event": event,
                "severity": a.get("severity") or "Unknown",
                "urgency": a.get("urgency") or "Unknown",
                "certainty": a.get("certainty") or "Unknown",
                "headline": a.get("headline") or "",
                "area_desc": a.get("area_desc") or "",
                "onset": a.get("onset") or "",
                "expires": a.get("expires") or "",
                "sender": a.get("sender") or "",
            },
            observed_at=now,
        ))
    return signals
