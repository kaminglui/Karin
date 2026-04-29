"""V1 alert rules.

Each rule is a small class. `evaluate(signals, now)` inspects the signal
batch and returns zero-or-more candidate Alerts (not yet cooldown-checked
— that's the engine's job). Rules are PURE: no I/O, no state, no LLM.

Rule catalog (V1):

  Independent channel:
    R1  tracker_shock_fx              shock on USD/CNY, USD/HKD, USD/JPY  -> ADVISORY, MACRO
    R2  tracker_shock_gold            shock on gold_usd                   -> WATCH, MARKET_SHOCK
    R3  tracker_shock_energy          shock on any category=energy        -> ADVISORY, ENERGY
    R4  news_confirmed_watchlist      CONFIRMED cluster + watchlist       -> ADVISORY, varies
    R5  news_provisional_watchlist    PROVISIONAL cluster + watchlist     -> WATCH, varies
    R6  travel_advisory_raised        advisory level >= 3                 -> ADVISORY/CRITICAL, TRAVEL

  Cross-channel (reserve CRITICAL for these):
    R7  shock_plus_geopolitical       any shock + CONFIRMED region/event cluster  -> CRITICAL
    R8  travel_plus_region_cluster    advisory change + CONFIRMED region cluster  -> CRITICAL

Thresholds are hardcoded here (Phase 6 choice). If they need to become
configurable later, that becomes Phase 6.1.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Iterable

from bridge.alerts.models import (
    Alert,
    AlertCategory,
    AlertLevel,
    Signal,
    SignalKind,
)


# --- base class -----------------------------------------------------------

class AlertRule(ABC):
    """Abstract rule. Subclasses define evaluate() and the class constants."""

    id: str = ""
    cooldown_hours: int = 0

    @abstractmethod
    def evaluate(self, signals: list[Signal], now: datetime) -> list[Alert]:
        ...

    # --- helpers for subclasses ------------------------------------------

    def _make_alert(
        self,
        *,
        level: AlertLevel,
        category: AlertCategory,
        title: str,
        reasoning_bullets: list[str],
        triggered_by_signals: list[Signal],
        source_attribution: list[str],
        affected_domains: list[str],
        scope_key: str,
        now: datetime,
    ) -> Alert:
        alert_id = hashlib.sha1(
            f"{self.id}::{scope_key}::{now.isoformat()}".encode("utf-8")
        ).hexdigest()[:16]
        return Alert(
            alert_id=alert_id,
            level=level,
            category=category,
            title=title,
            reasoning_bullets=reasoning_bullets,
            triggered_by_signals=triggered_by_signals,
            source_attribution=source_attribution,
            affected_domains=affected_domains,
            rule_id=self.id,
            scope_key=scope_key,
            created_at=now,
            cooldown_until=now + timedelta(hours=self.cooldown_hours),
            is_active=True,
        )


# --- helpers ---------------------------------------------------------------

_FX_TRACKERS = frozenset({"usd_cny", "usd_hkd", "usd_jpy"})
_ENERGY_CATEGORY = "energy"
_REGION_OR_EVENT = frozenset({"region", "event"})


def _shock_signals(signals: Iterable[Signal]) -> list[Signal]:
    return [s for s in signals if s.kind == SignalKind.TRACKER_SHOCK]


def _watchlist_signals(signals: Iterable[Signal]) -> list[Signal]:
    return [s for s in signals if s.kind == SignalKind.NEWS_WATCHLIST_MATCH]


def _advisory_signals(signals: Iterable[Signal]) -> list[Signal]:
    return [s for s in signals if s.kind == SignalKind.TRAVEL_ADVISORY_CHANGED]


def _nws_signals(signals: Iterable[Signal]) -> list[Signal]:
    return [s for s in signals if s.kind == SignalKind.NWS_WEATHER_ALERT]


def _is_confirmed(signal: Signal) -> bool:
    return signal.payload.get("cluster_state") == "confirmed"


def _is_region_or_event(signal: Signal) -> bool:
    return signal.payload.get("watchlist_type") in _REGION_OR_EVENT


def _fmt_pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.2f}%"


# --- R1: tracker shock FX --------------------------------------------------

class TrackerShockFX(AlertRule):
    id = "tracker_shock_fx"
    cooldown_hours = 6

    def evaluate(self, signals, now):
        out: list[Alert] = []
        for s in _shock_signals(signals):
            tid = s.payload.get("tracker_id", "")
            if tid not in _FX_TRACKERS:
                continue
            label = s.payload.get("label", tid)
            direction = s.payload.get("direction", "moving")
            change_pct = s.payload.get("change_pct")
            title = f"{label} {direction} ({_fmt_pct(change_pct)} in one day)"
            bullets = [
                f"1-day change: {_fmt_pct(change_pct)}",
                f"Detector: {direction} (>3 sigma vs lagged baseline)",
            ]
            out.append(self._make_alert(
                level=AlertLevel.ADVISORY,
                category=AlertCategory.MACRO,
                title=title,
                reasoning_bullets=bullets,
                triggered_by_signals=[s],
                source_attribution=[s.source],
                affected_domains=["macro"],
                scope_key=tid,
                now=now,
            ))
        return out


# --- R2: tracker shock gold -----------------------------------------------

class TrackerShockGold(AlertRule):
    id = "tracker_shock_gold"
    cooldown_hours = 6

    def evaluate(self, signals, now):
        out: list[Alert] = []
        for s in _shock_signals(signals):
            if s.payload.get("tracker_id") != "gold_usd":
                continue
            direction = s.payload.get("direction", "moving")
            change_pct = s.payload.get("change_pct")
            title = f"Gold {direction} ({_fmt_pct(change_pct)} in one day)"
            bullets = [
                f"Gold 1-day change: {_fmt_pct(change_pct)}",
                "Typical safe-haven flow signal; watch for accompanying news.",
            ]
            out.append(self._make_alert(
                level=AlertLevel.WATCH,
                category=AlertCategory.MARKET_SHOCK,
                title=title,
                reasoning_bullets=bullets,
                triggered_by_signals=[s],
                source_attribution=[s.source],
                affected_domains=["market_shock"],
                scope_key="gold_usd",
                now=now,
            ))
        return out


# --- R3: tracker shock energy ---------------------------------------------

class TrackerShockEnergy(AlertRule):
    id = "tracker_shock_energy"
    cooldown_hours = 6

    def evaluate(self, signals, now):
        out: list[Alert] = []
        for s in _shock_signals(signals):
            if s.payload.get("category") != _ENERGY_CATEGORY:
                continue
            tid = s.payload.get("tracker_id", "")
            label = s.payload.get("label", tid)
            direction = s.payload.get("direction", "moving")
            change_pct = s.payload.get("change_pct")
            title = f"{label} {direction} ({_fmt_pct(change_pct)} in one day)"
            bullets = [
                f"Energy 1-day change: {_fmt_pct(change_pct)}",
                "Expect knock-on effects on transportation and consumer prices.",
            ]
            out.append(self._make_alert(
                level=AlertLevel.ADVISORY,
                category=AlertCategory.ENERGY,
                title=title,
                reasoning_bullets=bullets,
                triggered_by_signals=[s],
                source_attribution=[s.source],
                affected_domains=["energy"],
                scope_key=tid,
                now=now,
            ))
        return out


# --- R4: news confirmed watchlist -----------------------------------------

def _watchlist_alert_category(wl_type: str) -> AlertCategory:
    """Map a watchlist section to an alert category.

    Regions and events -> GEOPOLITICAL (these are "something is happening
    in a place / tied to an event"). Topics -> WATCHLIST (catch-all for
    user's general interests like semiconductors or AI).
    """
    if wl_type in ("region", "event"):
        return AlertCategory.GEOPOLITICAL
    return AlertCategory.WATCHLIST


class NewsConfirmedWatchlist(AlertRule):
    id = "news_confirmed_watchlist"
    cooldown_hours = 12

    def evaluate(self, signals, now):
        out: list[Alert] = []
        seen_clusters: set[str] = set()
        for s in _watchlist_signals(signals):
            if not _is_confirmed(s):
                continue
            # Only EVENT-type watchlist matches fire as alerts.
            # Region + topic matches were filling the Alerts panel
            # with news-of-interest (ASML earnings, peace talks, etc.)
            # that's already surfaced on the News panel. Events are
            # the alert-worthy signal (elections, conflict, disasters,
            # summits) — keep this rule scoped to them.
            if s.payload.get("watchlist_type") != "event":
                continue
            # Phase G.a: additional threat gating. Signal must score
            # at or above 2 (watch) to fire. Events that happen far
            # from the user's location (score 0) are news-of-interest
            # but not personal threats — News panel territory.
            # Score field absent means the scorer hasn't run (legacy
            # test signals) — default to firing, preserving old tests.
            if s.payload.get("threat_score", 2) < 2:
                continue
            cluster_id = s.payload.get("cluster_id", "")
            # A cluster hitting multiple watchlists still only produces
            # one alert per rule per cluster. The most-significant
            # watchlist (first in our iteration) wins.
            if cluster_id in seen_clusters:
                continue
            seen_clusters.add(cluster_id)
            wl_type = s.payload.get("watchlist_type", "topic")
            wl_label = s.payload.get("watchlist_label", "watchlist")
            headline = s.payload.get("headline", "(no headline)")
            ic = s.payload.get("independent_confirmation_count", 0)
            title = f"Confirmed: {headline}"
            bullets = [
                f"Watchlist match: {wl_type}/{wl_label}",
                f"Independent confirmations: {ic}",
                "Multiple independent outlets reporting.",
            ]
            out.append(self._make_alert(
                level=AlertLevel.ADVISORY,
                category=_watchlist_alert_category(wl_type),
                title=title,
                reasoning_bullets=bullets,
                triggered_by_signals=[s],
                source_attribution=[s.source],
                affected_domains=[wl_type],
                scope_key=f"cluster:{cluster_id}",
                now=now,
            ))
        return out


# --- R5: news provisional watchlist ---------------------------------------

class NewsProvisionalWatchlist(AlertRule):
    id = "news_provisional_watchlist"
    cooldown_hours = 6

    def evaluate(self, signals, now):
        out: list[Alert] = []
        seen_clusters: set[str] = set()
        for s in _watchlist_signals(signals):
            if s.payload.get("cluster_state") != "provisionally_confirmed":
                continue
            # Mirror R4: only event-type watchlists fire here. See
            # NewsConfirmedWatchlist for the rationale.
            if s.payload.get("watchlist_type") != "event":
                continue
            # Provisional gets a tighter threshold (3+) than confirmed
            # (2+) — uncertain stories need a stronger threat signal
            # to justify an alert. Matches the general "confirmed
            # beats speculative" pattern in the news state machine.
            if s.payload.get("threat_score", 3) < 3:
                continue
            cluster_id = s.payload.get("cluster_id", "")
            if cluster_id in seen_clusters:
                continue
            seen_clusters.add(cluster_id)
            wl_type = s.payload.get("watchlist_type", "topic")
            wl_label = s.payload.get("watchlist_label", "watchlist")
            headline = s.payload.get("headline", "(no headline)")
            title = f"Developing: {headline}"
            bullets = [
                f"Watchlist match: {wl_type}/{wl_label}",
                "Provisionally confirmed; confidence may change.",
            ]
            out.append(self._make_alert(
                level=AlertLevel.WATCH,
                category=_watchlist_alert_category(wl_type),
                title=title,
                reasoning_bullets=bullets,
                triggered_by_signals=[s],
                source_attribution=[s.source],
                affected_domains=[wl_type],
                scope_key=f"cluster:{cluster_id}",
                now=now,
            ))
        return out


# --- R6: travel advisory raised -------------------------------------------

class TravelAdvisoryRaised(AlertRule):
    id = "travel_advisory_raised"
    cooldown_hours = 72

    def evaluate(self, signals, now):
        out: list[Alert] = []
        for s in _advisory_signals(signals):
            new_level = s.payload.get("new_level")
            if not isinstance(new_level, int) or new_level < 3:
                continue
            old_level = s.payload.get("old_level")
            country = s.payload.get("country", "")
            title = s.payload.get("title") or f"Travel advisory: {country} Level {new_level}"
            level = AlertLevel.CRITICAL if new_level == 4 else AlertLevel.ADVISORY
            bullets = [
                f"New advisory level: {new_level}",
                f"Previous level: {old_level if old_level is not None else 'unknown'}",
            ]
            out.append(self._make_alert(
                level=level,
                category=AlertCategory.TRAVEL,
                title=title,
                reasoning_bullets=bullets,
                triggered_by_signals=[s],
                source_attribution=[s.source],
                affected_domains=["travel"],
                scope_key=country,
                now=now,
            ))
        return out


# --- R7: cross-channel shock + geopolitical cluster -----------------------

class ShockPlusGeopolitical(AlertRule):
    """Elevates to CRITICAL when a market/energy shock coincides in the
    same scan with a confirmed cluster that matches a region or event
    watchlist. Two channels aligned is the whole point of CRITICAL.
    """

    id = "shock_plus_geopolitical"
    cooldown_hours = 24

    def evaluate(self, signals, now):
        shocks = _shock_signals(signals)
        geo_clusters = [
            s for s in _watchlist_signals(signals)
            if _is_confirmed(s) and _is_region_or_event(s)
        ]
        if not shocks or not geo_clusters:
            return []
        out: list[Alert] = []
        for shock in shocks:
            for geo in geo_clusters:
                tid = shock.payload.get("tracker_id", "")
                cluster_id = geo.payload.get("cluster_id", "")
                shock_label = shock.payload.get("label", tid)
                shock_dir = shock.payload.get("direction", "moving")
                shock_pct = shock.payload.get("change_pct")
                wl_label = geo.payload.get("watchlist_label", "watchlist")
                headline = geo.payload.get("headline", "(no headline)")
                title = f"{shock_label} {shock_dir} alongside confirmed {wl_label}"
                bullets = [
                    f"{shock_label} {shock_dir}: {_fmt_pct(shock_pct)}",
                    f"Confirmed news cluster: {headline}",
                    "Two independent channels agreeing elevates this to CRITICAL.",
                ]
                out.append(self._make_alert(
                    level=AlertLevel.CRITICAL,
                    category=AlertCategory.MARKET_SHOCK,
                    title=title,
                    reasoning_bullets=bullets,
                    triggered_by_signals=[shock, geo],
                    source_attribution=[shock.source, geo.source],
                    affected_domains=["market_shock", "geopolitical"],
                    scope_key=f"{tid}:cluster:{cluster_id}",
                    now=now,
                ))
        return out


# --- R8: cross-channel advisory change + region cluster -------------------

class TravelPlusRegionCluster(AlertRule):
    """Advisory level change for a country whose region also has a
    confirmed news cluster in the same scan. CRITICAL because two
    independent channels (State Dept + news) are aligned on the same
    region.
    """

    id = "travel_plus_region_cluster"
    cooldown_hours = 24

    def evaluate(self, signals, now):
        advisories = _advisory_signals(signals)
        region_clusters = [
            s for s in _watchlist_signals(signals)
            if _is_confirmed(s) and s.payload.get("watchlist_type") == "region"
        ]
        if not advisories or not region_clusters:
            return []
        out: list[Alert] = []
        for adv in advisories:
            for geo in region_clusters:
                country = adv.payload.get("country", "")
                cluster_id = geo.payload.get("cluster_id", "")
                region_label = geo.payload.get("watchlist_label", "region")
                new_level = adv.payload.get("new_level")
                headline = geo.payload.get("headline", "(no headline)")
                title = (
                    f"Travel advisory change for {country} alongside "
                    f"confirmed {region_label} cluster"
                )
                bullets = [
                    f"Advisory new level: {new_level}",
                    f"Confirmed news cluster: {headline}",
                    f"Region watchlist: {region_label}",
                ]
                out.append(self._make_alert(
                    level=AlertLevel.CRITICAL,
                    category=AlertCategory.TRAVEL,
                    title=title,
                    reasoning_bullets=bullets,
                    triggered_by_signals=[adv, geo],
                    source_attribution=[adv.source, geo.source],
                    affected_domains=["travel", "geopolitical"],
                    scope_key=f"{country}:cluster:{cluster_id}",
                    now=now,
                ))
        return out


# --- R9: NWS weather alert ------------------------------------------------

class NWSWeatherAlert(AlertRule):
    """Fires one alert per distinct NWS active-alert. Severity →
    AlertLevel mapping:
      Extreme → CRITICAL
      Severe  → ADVISORY
      Moderate→ WATCH
      (Minor/Unknown are already filtered out upstream in nws_fetch.)

    Cooldown is 6 hours per alert_id. NWS warnings typically last
    2-12 hours, so this avoids re-firing when we poll through the
    same active alert but still lets a replacement alert (new id)
    fire on the next poll if NWS reissues.
    """

    id = "nws_weather_alert"
    cooldown_hours = 6

    def evaluate(self, signals, now):
        out: list[Alert] = []
        for s in _nws_signals(signals):
            severity = (s.payload.get("severity") or "").strip()
            event = s.payload.get("event") or "Weather alert"
            area = s.payload.get("area_desc") or ""
            sender = s.payload.get("sender") or ""
            alert_id = s.payload.get("id") or ""
            headline = s.payload.get("headline") or ""
            if severity == "Extreme":
                level = AlertLevel.CRITICAL
            elif severity == "Severe":
                level = AlertLevel.ADVISORY
            elif severity == "Moderate":
                level = AlertLevel.WATCH
            else:
                continue  # unknown/minor — upstream should have filtered
            title = f"{event}" + (f" — {area}" if area else "")
            bullets: list[str] = []
            if severity:
                bullets.append(f"NWS severity: {severity}")
            if headline:
                # Trim headline — NWS headlines are often long and
                # already mostly said in the title.
                bullets.append(headline[:200])
            expires = s.payload.get("expires")
            if expires:
                bullets.append(f"Active until: {expires}")
            out.append(self._make_alert(
                level=level,
                category=AlertCategory.WEATHER,
                title=title,
                reasoning_bullets=bullets,
                triggered_by_signals=[s],
                source_attribution=[sender or "NWS"],
                affected_domains=["weather", "travel"],
                # scope_key: NWS alert id gets cooldown per-alert. When
                # NWS upgrades the same underlying storm to a new
                # alert id, we get a fresh fire.
                scope_key=alert_id or title,
                now=now,
            ))
        return out


# --- registry -------------------------------------------------------------

# Default rule set. AlertService accepts an override list for tests.
# Cross-channel rules come LAST so their cooldowns are registered after
# the independent rules they might shadow — but since cooldowns are
# per-rule (not cross-rule), the ordering is purely cosmetic.
DEFAULT_RULES: list[AlertRule] = [
    TrackerShockFX(),
    TrackerShockGold(),
    TrackerShockEnergy(),
    NewsConfirmedWatchlist(),
    NewsProvisionalWatchlist(),
    TravelAdvisoryRaised(),
    NWSWeatherAlert(),
    ShockPlusGeopolitical(),
    TravelPlusRegionCluster(),
]
