"""Tests for bridge.alerts.rules.

One or two tests per rule pinning:
  - positive fire with correct level, category, scope_key
  - negative (rule doesn't fire when signals absent / wrong shape)
"""
from __future__ import annotations

from datetime import datetime, timezone

from bridge.alerts.models import (
    AlertCategory,
    AlertLevel,
    Signal,
    SignalKind,
)
from bridge.alerts.rules import (
    NewsConfirmedWatchlist,
    NewsProvisionalWatchlist,
    ShockPlusGeopolitical,
    TrackerShockEnergy,
    TrackerShockFX,
    TrackerShockGold,
    TravelAdvisoryRaised,
    TravelPlusRegionCluster,
)


_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)


def _shock_signal(
    *, tracker_id="usd_cny", label="USD/CNY", category="fx",
    direction="surging", change_pct=5.0,
) -> Signal:
    return Signal(
        kind=SignalKind.TRACKER_SHOCK,
        source=f"tracker:{tracker_id}",
        payload={
            "tracker_id": tracker_id,
            "label": label,
            "category": category,
            "direction": direction,
            "change_pct": change_pct,
            "latest_value": 7.0,
        },
        observed_at=_NOW,
    )


def _news_signal(
    *, cluster_id="c1", cluster_state="confirmed",
    watchlist_type="event", watchlist_id="ukraine_war",
    watchlist_label="Ukraine", headline="Ukraine talks today",
    ic=3,
) -> Signal:
    return Signal(
        kind=SignalKind.NEWS_WATCHLIST_MATCH,
        source=f"news:cluster:{cluster_id}",
        payload={
            "cluster_id": cluster_id,
            "cluster_state": cluster_state,
            "headline": headline,
            "watchlist_type": watchlist_type,
            "watchlist_id": watchlist_id,
            "watchlist_label": watchlist_label,
            "watchlist_priority": "high",
            "independent_confirmation_count": ic,
        },
        observed_at=_NOW,
    )


def _advisory_signal(
    *, country="EGY", old_level=2, new_level=3, title="Egypt Level 3",
) -> Signal:
    return Signal(
        kind=SignalKind.TRAVEL_ADVISORY_CHANGED,
        source=f"external:travel:{country}",
        payload={
            "country": country,
            "old_level": old_level,
            "new_level": new_level,
            "title": title,
        },
        observed_at=_NOW,
    )


# --- R1: tracker_shock_fx --------------------------------------------------

class TestTrackerShockFX:
    def test_fires_on_fx_shock(self):
        alerts = TrackerShockFX().evaluate([_shock_signal()], _NOW)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.level == AlertLevel.ADVISORY
        assert a.category == AlertCategory.MACRO
        assert a.scope_key == "usd_cny"
        assert a.rule_id == "tracker_shock_fx"

    def test_does_not_fire_on_gold_shock(self):
        gold = _shock_signal(tracker_id="gold_usd", label="Gold", category="metal")
        assert TrackerShockFX().evaluate([gold], _NOW) == []

    def test_multiple_fx_shocks_produce_multiple_alerts(self):
        s1 = _shock_signal(tracker_id="usd_cny", label="USD/CNY")
        s2 = _shock_signal(tracker_id="usd_jpy", label="USD/JPY")
        alerts = TrackerShockFX().evaluate([s1, s2], _NOW)
        assert len(alerts) == 2
        assert {a.scope_key for a in alerts} == {"usd_cny", "usd_jpy"}


# --- R2: tracker_shock_gold -----------------------------------------------

class TestTrackerShockGold:
    def test_fires_on_gold_shock(self):
        s = _shock_signal(tracker_id="gold_usd", label="Gold", category="metal",
                          direction="surging", change_pct=4.0)
        alerts = TrackerShockGold().evaluate([s], _NOW)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.level == AlertLevel.WATCH
        assert a.category == AlertCategory.MARKET_SHOCK
        assert a.scope_key == "gold_usd"

    def test_does_not_fire_on_fx_shock(self):
        s = _shock_signal(tracker_id="usd_cny")
        assert TrackerShockGold().evaluate([s], _NOW) == []


# --- R3: tracker_shock_energy ---------------------------------------------

class TestTrackerShockEnergy:
    def test_fires_on_energy_shock(self):
        s = _shock_signal(
            tracker_id="rbob_gasoline", label="RBOB Gasoline",
            category="energy", direction="surging", change_pct=6.1,
        )
        alerts = TrackerShockEnergy().evaluate([s], _NOW)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.level == AlertLevel.ADVISORY
        assert a.category == AlertCategory.ENERGY
        assert a.scope_key == "rbob_gasoline"
        assert "energy" in a.affected_domains

    def test_does_not_fire_on_non_energy(self):
        s = _shock_signal(tracker_id="gold_usd", category="metal")
        assert TrackerShockEnergy().evaluate([s], _NOW) == []


# --- R4: news_confirmed_watchlist -----------------------------------------

class TestNewsConfirmedWatchlist:
    def test_fires_on_confirmed_match(self):
        alerts = NewsConfirmedWatchlist().evaluate([_news_signal()], _NOW)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.level == AlertLevel.ADVISORY
        assert a.category == AlertCategory.GEOPOLITICAL  # event watchlist type
        assert a.scope_key == "cluster:c1"

    def test_topic_watchlist_does_not_fire(self):
        # Post-narrowing: topic + region matches are news-of-interest,
        # not advisories. Only event-type watchlists still trigger R4.
        # The old test (now renamed) verified topic→WATCHLIST category;
        # that path is intentionally dead now.
        s = _news_signal(watchlist_type="topic", watchlist_id="ai",
                         watchlist_label="AI")
        assert NewsConfirmedWatchlist().evaluate([s], _NOW) == []

    def test_region_watchlist_does_not_fire(self):
        # Same as topic — regions are news, not alerts.
        s = _news_signal(watchlist_type="region", watchlist_id="us",
                         watchlist_label="United States")
        assert NewsConfirmedWatchlist().evaluate([s], _NOW) == []

    def test_does_not_fire_on_developing_cluster(self):
        s = _news_signal(cluster_state="developing")
        assert NewsConfirmedWatchlist().evaluate([s], _NOW) == []

    def test_event_match_still_fires_even_when_mixed_with_others(self):
        # Narrowing kept the rule useful for EVENT watchlists — so a
        # cluster matching (event, region, topic) should still emit one
        # alert via the event signal, and dedup across the other
        # matches on the same cluster.
        s_event = _news_signal(watchlist_id="ukraine_war", watchlist_label="Ukraine")
        s_region = _news_signal(watchlist_type="region", watchlist_id="us",
                                watchlist_label="United States")
        alerts = NewsConfirmedWatchlist().evaluate([s_event, s_region], _NOW)
        assert len(alerts) == 1


# --- R5: news_provisional_watchlist ---------------------------------------

class TestNewsProvisionalWatchlist:
    def test_fires_on_provisional_match(self):
        s = _news_signal(cluster_state="provisionally_confirmed")
        alerts = NewsProvisionalWatchlist().evaluate([s], _NOW)
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WATCH

    def test_does_not_fire_on_confirmed(self):
        # Confirmed is R4's job, not R5's.
        s = _news_signal(cluster_state="confirmed")
        assert NewsProvisionalWatchlist().evaluate([s], _NOW) == []


# --- R6: travel_advisory_raised -------------------------------------------

class TestTravelAdvisoryRaised:
    def test_fires_on_level_3(self):
        alerts = TravelAdvisoryRaised().evaluate([_advisory_signal(new_level=3)], _NOW)
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.ADVISORY
        assert alerts[0].category == AlertCategory.TRAVEL

    def test_fires_critical_on_level_4(self):
        alerts = TravelAdvisoryRaised().evaluate([_advisory_signal(new_level=4)], _NOW)
        assert alerts[0].level == AlertLevel.CRITICAL

    def test_does_not_fire_on_level_2(self):
        assert TravelAdvisoryRaised().evaluate([_advisory_signal(new_level=2)], _NOW) == []


# --- R7: cross-channel shock + geopolitical --------------------------------

class TestShockPlusGeopolitical:
    def test_fires_on_shock_plus_confirmed_event_cluster(self):
        signals = [_shock_signal(), _news_signal()]
        alerts = ShockPlusGeopolitical().evaluate(signals, _NOW)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.level == AlertLevel.CRITICAL
        assert a.category == AlertCategory.MARKET_SHOCK
        assert a.scope_key == "usd_cny:cluster:c1"

    def test_does_not_fire_without_news(self):
        assert ShockPlusGeopolitical().evaluate([_shock_signal()], _NOW) == []

    def test_does_not_fire_without_shock(self):
        assert ShockPlusGeopolitical().evaluate([_news_signal()], _NOW) == []

    def test_does_not_fire_on_topic_watchlist_only(self):
        # Topic watchlist isn't region/event -> no cross-channel fire.
        s = _news_signal(watchlist_type="topic", watchlist_id="ai")
        assert ShockPlusGeopolitical().evaluate([_shock_signal(), s], _NOW) == []

    def test_does_not_fire_on_developing_cluster(self):
        s = _news_signal(cluster_state="developing")
        assert ShockPlusGeopolitical().evaluate([_shock_signal(), s], _NOW) == []


# --- R8: cross-channel advisory + region ----------------------------------

class TestTravelPlusRegionCluster:
    def test_fires_on_advisory_plus_region_cluster(self):
        signals = [
            _advisory_signal(),
            _news_signal(watchlist_type="region", watchlist_id="egypt",
                         watchlist_label="Egypt"),
        ]
        alerts = TravelPlusRegionCluster().evaluate(signals, _NOW)
        assert len(alerts) == 1
        a = alerts[0]
        assert a.level == AlertLevel.CRITICAL
        assert a.category == AlertCategory.TRAVEL
        assert a.scope_key.startswith("EGY:cluster:")

    def test_does_not_fire_on_event_watchlist(self):
        # Spec: this rule requires REGION watchlist type specifically.
        signals = [_advisory_signal(), _news_signal(watchlist_type="event")]
        assert TravelPlusRegionCluster().evaluate(signals, _NOW) == []
