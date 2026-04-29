"""Tests for bridge.alerts.service.AlertService.

Focused on orchestration: TTL gate, advisory baseline seeding on first
poll, and end-to-end scan with mocked subsystems.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from bridge.alerts.cooldown import CooldownLedger
from bridge.alerts.models import AlertLevel
from bridge.alerts.service import AlertService
from bridge.alerts.store import AlertStore
from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    StoryCluster,
)
from bridge.news.preferences import Preferences, WatchlistItem
from bridge.trackers.models import TrackerSnapshot


_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)


def _snap(**kw) -> TrackerSnapshot:
    base = dict(
        id="usd_cny", label="USD/CNY", category="fx",
        latest_value=7.0, latest_timestamp=_NOW,
        change_1d=None, change_1d_pct=5.0,
        change_1w=None, change_1w_pct=None,
        change_1m=None, change_1m_pct=None,
        is_stale=False, note="",
        direction_1d=None, direction_1w=None,
        movement_label=None, shock_label="surging",
    )
    base.update(kw)
    return TrackerSnapshot(**base)


def _cluster(*, cluster_id="c1",
             state=ConfidenceState.CONFIRMED,
             title="Ukraine talks happening in Brussels today") -> StoryCluster:
    return StoryCluster(
        cluster_id=cluster_id,
        article_ids=["a1"],
        centroid_display_title=title,
        centroid_normalized_title=title.lower(),
        first_seen_at=_NOW,
        latest_update_at=_NOW,
        last_checked_at=_NOW,
        last_state_change_at=_NOW,
        state=state,
        is_stale=False,
        independent_confirmation_count=3,
        article_count=3,
        syndicated_article_count=0,
    )


def _article(*, article_id="a1") -> NormalizedArticle:
    return NormalizedArticle(
        article_id=article_id, source_id="bbc",
        url=f"https://x.test/{article_id}",
        display_title="Ukraine talks in Brussels",
        normalized_title="ukraine talks in brussels",
        summary="", fingerprint=f"fp_{article_id}",
        wire_attribution=None, published_at=_NOW, fetched_at=_NOW,
    )


def _fake_tracker_service(snapshots):
    svc = MagicMock()
    svc.get_trackers.return_value = snapshots
    return svc


def _fake_news_service(clusters, articles, prefs):
    svc = MagicMock()
    svc.load_all_articles.return_value = articles
    svc.load_all_clusters.return_value = clusters
    svc.get_preferences.return_value = prefs
    return svc


@pytest.fixture
def store(tmp_path):
    return AlertStore(tmp_path)


@pytest.fixture
def ledger(tmp_path):
    return CooldownLedger(tmp_path / "cooldowns.json")


class TestScanOrchestration:
    def test_end_to_end_fires_cross_channel_critical(self, store, ledger):
        # Shock + confirmed ukraine event cluster -> R1, R4, R7 all fire.
        # Phase G.a: R4 only fires when threat_score >= 2, which
        # requires proximity to the user's location. Inject a
        # UserContext whose country matches Ukraine so the news
        # signal clears that threshold.
        from bridge.alerts.user_context import UserContext
        prefs = Preferences(
            enabled=True,
            events=(WatchlistItem(id="ukraine_war", label="Ukraine",
                                  keywords=("ukraine",), priority="high"),),
        )
        svc = AlertService(
            store=store,
            cooldown_ledger=ledger,
            tracker_service=_fake_tracker_service([_snap()]),
            news_service=_fake_news_service(
                {"c1": _cluster()}, {"a1": _article()}, prefs,
            ),
            advisory_fetcher=lambda: {},  # no advisories
            user_context=UserContext(country="Ukraine", region="Ukraine"),
        )
        result = svc.scan(force=True)
        rule_ids = {a.rule_id for a in result.alerts}
        assert "tracker_shock_fx" in rule_ids
        assert "news_confirmed_watchlist" in rule_ids
        assert "shock_plus_geopolitical" in rule_ids
        critical = [a for a in result.alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) == 1
        assert critical[0].rule_id == "shock_plus_geopolitical"

    def test_ttl_gate_skips_rescan(self, store, ledger):
        svc = AlertService(
            store=store, cooldown_ledger=ledger,
            tracker_service=_fake_tracker_service([]),
            news_service=None,
            advisory_fetcher=lambda: {},
        )
        svc.scan(force=True)  # populates scan_ok event
        result = svc.scan(force=False)
        assert result.skipped_due_to_cache is True

    @pytest.fixture
    def _travel_on(self, monkeypatch):
        """Force the alerts_travel_advisory flag on for tests in this
        class that exercise the advisory pipeline. Required since the
        default is now off (the user toggled it off after deciding
        per-country advisory levels weren't actionable)."""
        from bridge import features
        monkeypatch.setenv("KARIN_ALERTS_TRAVEL_ADVISORY", "true")
        features.reload()
        yield
        monkeypatch.delenv("KARIN_ALERTS_TRAVEL_ADVISORY", raising=False)
        features.reload()

    def test_advisory_baseline_seeded_silently_on_first_run(
        self, store, ledger, _travel_on,
    ):
        # On first poll, advisory_state.json is empty. The fetcher returns
        # a non-empty set. Service must NOT emit any advisory alerts and
        # must persist the state for next time.
        fake_fetch = lambda: {
            "EGY": {"level": 3, "title": "Egypt Level 3"},
            "JPN": {"level": 1, "title": "Japan Level 1"},
        }
        svc = AlertService(
            store=store, cooldown_ledger=ledger,
            tracker_service=None, news_service=None,
            advisory_fetcher=fake_fetch,
        )
        result = svc.scan(force=True)
        rule_ids = {a.rule_id for a in result.alerts}
        assert "travel_advisory_raised" not in rule_ids
        # Baseline is persisted.
        assert store.load_advisory_state() == {"EGY": 3, "JPN": 1}

    def test_advisory_change_after_baseline_fires(
        self, store, ledger, _travel_on,
    ):
        # Pre-seed state as if a prior baseline ran.
        store.save_advisory_state({"EGY": 2})
        fake_fetch = lambda: {"EGY": {"level": 3, "title": "Egypt raised to Level 3"}}
        svc = AlertService(
            store=store, cooldown_ledger=ledger,
            tracker_service=None, news_service=None,
            advisory_fetcher=fake_fetch,
        )
        result = svc.scan(force=True)
        rule_ids = {a.rule_id for a in result.alerts}
        assert "travel_advisory_raised" in rule_ids

    def test_get_active_alerts_filters_by_cooldown(self, store, ledger):
        # Fire one alert, then query. It should be active (cooldown hasn't expired).
        svc = AlertService(
            store=store, cooldown_ledger=ledger,
            tracker_service=_fake_tracker_service([_snap()]),
            news_service=None,
            advisory_fetcher=lambda: {},
        )
        svc.scan(force=True)
        active = svc.get_active_alerts()
        assert len(active) >= 1
        assert all(a.is_active for a in active)


class TestTravelAdvisoryFeatureGate:
    """The `alerts_travel_advisory` feature flag controls both the
    write path (collection during scan) AND the read path (filtering
    out previously-fired travel rows from get_active_alerts). When the
    user disables travel feeds we should see no NEW advisories AND no
    OLD ones lingering on the panel."""

    @pytest.fixture
    def features_off(self, monkeypatch):
        """Force `alerts_travel_advisory` off via the feature registry,
        regardless of what the YAML on disk says. Reload guarantees a
        clean state per test."""
        from bridge import features
        features.reload()
        monkeypatch.setenv("KARIN_ALERTS_TRAVEL_ADVISORY", "false")
        features.reload()
        yield
        monkeypatch.delenv("KARIN_ALERTS_TRAVEL_ADVISORY", raising=False)
        features.reload()

    @pytest.fixture
    def features_on(self, monkeypatch):
        from bridge import features
        features.reload()
        monkeypatch.setenv("KARIN_ALERTS_TRAVEL_ADVISORY", "true")
        features.reload()
        yield
        monkeypatch.delenv("KARIN_ALERTS_TRAVEL_ADVISORY", raising=False)
        features.reload()

    def test_scan_skips_advisory_collection_when_off(
        self, store, ledger, features_off,
    ):
        """When the flag is off, the advisory_fetcher must NEVER be
        invoked during scan — even in force=True mode. Proves we're
        not just hiding the cards; we're also not making the upstream
        HTTP call."""
        called = []
        def boom_fetcher():
            called.append("invoked")
            return {"EGY": {"level": 4, "title": "Egypt Level 4"}}
        svc = AlertService(
            store=store, cooldown_ledger=ledger,
            tracker_service=None, news_service=None,
            advisory_fetcher=boom_fetcher,
        )
        result = svc.scan(force=True)
        assert called == []   # fetcher never called
        rule_ids = {a.rule_id for a in result.alerts}
        assert "travel_advisory_raised" not in rule_ids

    def test_scan_collects_advisories_when_on(
        self, store, ledger, features_on,
    ):
        called = []
        def fetcher():
            called.append("invoked")
            return {"EGY": {"level": 4, "title": "Egypt Level 4"}}
        # Pre-seed baseline so the change actually fires.
        store.save_advisory_state({"EGY": 2})
        svc = AlertService(
            store=store, cooldown_ledger=ledger,
            tracker_service=None, news_service=None,
            advisory_fetcher=fetcher,
        )
        result = svc.scan(force=True)
        assert called == ["invoked"]
        rule_ids = {a.rule_id for a in result.alerts}
        assert "travel_advisory_raised" in rule_ids

    def test_get_active_alerts_filters_existing_travel_when_off(
        self, store, ledger, features_on, monkeypatch,
    ):
        """Fire a travel-advisory alert with the flag on, then flip
        the flag off and confirm the alert disappears from the panel
        view even though it's still in the append-only log."""
        store.save_advisory_state({"EGY": 2})
        svc = AlertService(
            store=store, cooldown_ledger=ledger,
            tracker_service=None, news_service=None,
            advisory_fetcher=lambda: {"EGY": {"level": 4, "title": "Egypt Level 4"}},
        )
        svc.scan(force=True)
        # Sanity: the alert IS visible while the flag is on.
        active_on = svc.get_active_alerts()
        from bridge.alerts.models import AlertCategory
        travel_on = [a for a in active_on if a.category == AlertCategory.TRAVEL]
        assert len(travel_on) >= 1, "travel alert should be visible when flag is on"

        # Flip the flag off + reload registry so get_active_alerts
        # sees the new state.
        from bridge import features
        monkeypatch.setenv("KARIN_ALERTS_TRAVEL_ADVISORY", "false")
        features.reload()

        active_off = svc.get_active_alerts()
        travel_off = [a for a in active_off if a.category == AlertCategory.TRAVEL]
        assert travel_off == [], "travel alert should be hidden when flag is off"
