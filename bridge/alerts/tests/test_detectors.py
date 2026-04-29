"""Tests for bridge.alerts.detectors.

Covers:
  - tracker signal emission for each Phase 5.2 label
  - stale tracker filtering (spec requirement)
  - news signal emission per watchlist match
  - stale cluster filtering (spec requirement)
  - advisory changes -> signals
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bridge.alerts.detectors import (
    signals_from_advisory_changes,
    signals_from_news,
    signals_from_trackers,
)
from bridge.alerts.models import SignalKind
from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    StoryCluster,
)
from bridge.news.preferences import (
    Preferences,
    WatchlistItem,
)
from bridge.trackers.models import TrackerSnapshot


_NOW = datetime(2026, 4, 12, tzinfo=timezone.utc)


def _snap(
    *,
    id="usd_cny",
    label="USD/CNY",
    category="fx",
    latest_value=7.0,
    is_stale=False,
    shock_label=None,
    movement_label=None,
    direction_1d=None,
    direction_1w=None,
    change_1d_pct=None,
    change_1w_pct=None,
) -> TrackerSnapshot:
    return TrackerSnapshot(
        id=id, label=label, category=category,
        latest_value=latest_value, latest_timestamp=_NOW,
        change_1d=None, change_1d_pct=change_1d_pct,
        change_1w=None, change_1w_pct=change_1w_pct,
        change_1m=None, change_1m_pct=None,
        is_stale=is_stale, note="",
        direction_1d=direction_1d, direction_1w=direction_1w,
        movement_label=movement_label, shock_label=shock_label,
    )


def _cluster(
    *,
    cluster_id="c1",
    state=ConfidenceState.DEVELOPING,
    is_stale=False,
    centroid_display_title="Some news headline today now happening",
) -> StoryCluster:
    return StoryCluster(
        cluster_id=cluster_id,
        article_ids=["a1"],
        centroid_display_title=centroid_display_title,
        centroid_normalized_title=centroid_display_title.lower(),
        first_seen_at=_NOW,
        latest_update_at=_NOW,
        last_checked_at=_NOW,
        last_state_change_at=_NOW,
        state=state,
        is_stale=is_stale,
        independent_confirmation_count=2,
        article_count=2,
        syndicated_article_count=0,
    )


def _article(*, article_id="a1", summary="") -> NormalizedArticle:
    return NormalizedArticle(
        article_id=article_id,
        source_id="bbc",
        url=f"https://x.test/{article_id}",
        display_title="headline",
        normalized_title="headline",
        summary=summary.lower(),
        fingerprint=f"fp_{article_id}",
        wire_attribution=None,
        published_at=_NOW,
        fetched_at=_NOW,
    )


def _prefs(*, events=(), regions=(), topics=(), enabled=True) -> Preferences:
    return Preferences(
        enabled=enabled, regions=regions, topics=topics, events=events,
    )


# --- tracker signals -------------------------------------------------------

class TestTrackerSignals:
    def test_shock_emits_tracker_shock(self):
        snap = _snap(shock_label="surging", change_1d_pct=5.2)
        signals = signals_from_trackers([snap], now=_NOW)
        assert len(signals) == 1
        assert signals[0].kind == SignalKind.TRACKER_SHOCK
        assert signals[0].payload["direction"] == "surging"
        assert signals[0].payload["change_pct"] == 5.2

    def test_volatile_emits_tracker_volatile(self):
        snap = _snap(movement_label="volatile")
        signals = signals_from_trackers([snap], now=_NOW)
        assert len(signals) == 1
        assert signals[0].kind == SignalKind.TRACKER_VOLATILE

    def test_direction_1w_emits_tracker_direction_1w(self):
        snap = _snap(direction_1w="up", change_1w_pct=2.5)
        signals = signals_from_trackers([snap], now=_NOW)
        assert len(signals) == 1
        assert signals[0].kind == SignalKind.TRACKER_DIRECTION_1W
        assert signals[0].payload["direction"] == "up"

    def test_multiple_labels_multiple_signals(self):
        # Shock + volatile + direction_1w all true -> 3 signals.
        snap = _snap(
            shock_label="plunging", change_1d_pct=-4.1,
            movement_label="volatile",
            direction_1w="down", change_1w_pct=-3.2,
        )
        signals = signals_from_trackers([snap], now=_NOW)
        kinds = {s.kind for s in signals}
        assert kinds == {
            SignalKind.TRACKER_SHOCK,
            SignalKind.TRACKER_VOLATILE,
            SignalKind.TRACKER_DIRECTION_1W,
        }

    def test_no_labels_no_signals(self):
        snap = _snap()
        assert signals_from_trackers([snap], now=_NOW) == []

    def test_direction_1w_flat_is_not_emitted(self):
        snap = _snap(direction_1w="flat", change_1w_pct=0.0)
        assert signals_from_trackers([snap], now=_NOW) == []

    def test_stale_tracker_emits_no_signals(self):
        # Spec requirement: stale inputs are filtered in the detector.
        snap = _snap(
            is_stale=True,
            shock_label="surging", change_1d_pct=5.2,
            movement_label="volatile",
            direction_1w="up", change_1w_pct=3.0,
        )
        assert signals_from_trackers([snap], now=_NOW) == []


# --- news signals ---------------------------------------------------------

class TestNewsSignals:
    def _setup_ukraine_event_prefs(self) -> Preferences:
        return _prefs(events=(
            WatchlistItem(id="ukraine_war", label="Russia-Ukraine War",
                          keywords=("ukraine",), priority="high"),
        ))

    def test_cluster_matching_watchlist_emits_signal(self):
        prefs = self._setup_ukraine_event_prefs()
        cluster = _cluster(
            cluster_id="u1",
            state=ConfidenceState.CONFIRMED,
            centroid_display_title="Ukraine talks continue in Brussels today",
        )
        articles = {"a1": _article()}
        signals = signals_from_news(
            {"u1": cluster}, articles, prefs, now=_NOW,
        )
        assert len(signals) == 1
        sig = signals[0]
        assert sig.kind == SignalKind.NEWS_WATCHLIST_MATCH
        assert sig.payload["cluster_state"] == "confirmed"
        assert sig.payload["watchlist_id"] == "ukraine_war"
        assert sig.payload["watchlist_type"] == "event"

    def test_cluster_not_matching_emits_nothing(self):
        prefs = self._setup_ukraine_event_prefs()
        cluster = _cluster(
            centroid_display_title="Election results in a distant country tonight",
        )
        articles = {"a1": _article()}
        signals = signals_from_news(
            {"c1": cluster}, articles, prefs, now=_NOW,
        )
        assert signals == []

    def test_stale_cluster_emits_no_signals(self):
        # Spec requirement: stale clusters are filtered in the detector.
        prefs = self._setup_ukraine_event_prefs()
        cluster = _cluster(
            cluster_id="u1",
            state=ConfidenceState.CONFIRMED,
            is_stale=True,
            centroid_display_title="Ukraine talks continue in Brussels today",
        )
        articles = {"a1": _article()}
        signals = signals_from_news(
            {"u1": cluster}, articles, prefs, now=_NOW,
        )
        assert signals == []

    def test_disabled_preferences_emits_nothing(self):
        prefs = _prefs(enabled=False, events=(
            WatchlistItem(id="x", label="x", keywords=("ukraine",)),
        ))
        cluster = _cluster(centroid_display_title="Ukraine talks today now")
        articles = {"a1": _article()}
        assert signals_from_news({"c1": cluster}, articles, prefs, now=_NOW) == []

    def test_cluster_matching_multiple_watchlists_emits_one_signal_per_match(self):
        prefs = _prefs(
            events=(WatchlistItem(id="ukraine_war", label="Ukraine",
                                  keywords=("ukraine",), priority="high"),),
            regions=(WatchlistItem(id="us", label="United States",
                                   keywords=("washington",), priority="high"),),
        )
        cluster = _cluster(
            cluster_id="u1",
            state=ConfidenceState.CONFIRMED,
            centroid_display_title="Ukraine talks in Washington today",
        )
        articles = {"a1": _article()}
        signals = signals_from_news(
            {"u1": cluster}, articles, prefs, now=_NOW,
        )
        assert len(signals) == 2
        ids = {s.payload["watchlist_id"] for s in signals}
        assert ids == {"ukraine_war", "us"}


# --- advisory signals -----------------------------------------------------

class TestAdvisorySignals:
    def test_changes_become_signals(self):
        changes = [
            {"country": "EGY", "old_level": 2, "new_level": 3, "title": "Egypt Level 3"},
            {"country": "JPN", "old_level": 1, "new_level": 2, "title": "Japan Level 2"},
        ]
        signals = signals_from_advisory_changes(changes, now=_NOW)
        assert len(signals) == 2
        assert signals[0].kind == SignalKind.TRAVEL_ADVISORY_CHANGED
        assert signals[0].payload["country"] == "EGY"
        assert signals[0].payload["new_level"] == 3

    def test_empty_changes_empty_signals(self):
        assert signals_from_advisory_changes([], now=_NOW) == []

    def test_missing_country_skipped(self):
        changes = [{"country": "", "old_level": 2, "new_level": 3}]
        assert signals_from_advisory_changes(changes, now=_NOW) == []
