"""Tests for bridge.news.service.

Uses tmp_path for the ledger and patches fetch_all so no network is
involved. Covers:
  - TTL cache gate (skip within window, refetch outside, force bypass)
  - ingest merges new articles by article_id; repeat ingests are no-ops
  - get_news ranking (confirmed > provisional > developing; recency tiebreak)
  - get_news topic filter (keyword match; recent-fallback when no match)
  - get_cluster: returns brief or None
  - refresh_and_rebuild: re-clusters without refetching
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from bridge.news.ledger import Ledger
from bridge.news.models import (
    ConfidenceState,
    Feed,
    RawArticle,
    Source,
    Tier,
)
from bridge.news.service import NewsService


# --- fixtures --------------------------------------------------------------

def _sources() -> dict[str, Source]:
    return {
        "bbc": Source(id="bbc", name="BBC", domain="bbc.com",
                      tier=Tier.REPUTABLE, ownership_group="bbc",
                      is_wire_service=False),
        "npr": Source(id="npr", name="NPR", domain="npr.org",
                      tier=Tier.REPUTABLE, ownership_group="npr",
                      is_wire_service=False),
        "guardian": Source(id="guardian", name="Guardian",
                           domain="theguardian.com", tier=Tier.REPUTABLE,
                           ownership_group="guardian_media_group",
                           is_wire_service=False),
    }


def _raw(
    *,
    title: str,
    source_id: str,
    url: str,
    hours_ago: float = 1.0,
) -> RawArticle:
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return RawArticle(
        source_id=source_id, url=url, title=title, summary="",
        published_at=ts, fetched_at=ts,
    )


@pytest.fixture
def service(tmp_path):
    ledger = Ledger(tmp_path)
    feeds = [Feed(source_id="bbc", url="https://fake.test/bbc")]
    return NewsService(
        ledger=ledger, sources=_sources(), feeds=feeds,
        fetch_ttl_minutes=15, stale_threshold_hours=24,
    )


# --- TTL gate --------------------------------------------------------------

class TestTTLGate:
    def test_first_call_fetches(self, service):
        with patch("bridge.news.service.fetch_all", return_value=([], 1, 0)):
            result = service.ingest_latest()
        assert result.skipped_due_to_cache is False
        assert result.fetched_feeds == 1

    def test_within_ttl_is_skipped(self, service):
        with patch("bridge.news.service.fetch_all", return_value=([], 1, 0)) as m:
            service.ingest_latest()            # populates ingest_ok event
            result = service.ingest_latest()    # second call within TTL
        assert result.skipped_due_to_cache is True
        # fetch_all should have been called exactly once — second call gated.
        assert m.call_count == 1

    def test_force_bypasses_ttl(self, service):
        with patch("bridge.news.service.fetch_all", return_value=([], 1, 0)) as m:
            service.ingest_latest()
            result = service.ingest_latest(force=True)
        assert result.skipped_due_to_cache is False
        assert m.call_count == 2

    def test_get_news_fetch_false_skips_ingest(self, service):
        """`fetch=False` is the contract for tool / panel callers: they
        must NEVER trigger an upstream fetch (the background poller is
        the sole source). Patching fetch_all to a sentinel proves the
        tool path doesn't even reach it."""
        with patch("bridge.news.service.fetch_all") as m:
            service.get_news(fetch=False)
        assert m.call_count == 0, (
            "get_news(fetch=False) must not call fetch_all — the tool "
            "layer relies on this to stay under the NewsAPI quota"
        )

    def test_get_news_fetch_true_default_does_attempt_ingest(self, service):
        """The opposite contract: panel pages and admin scripts that
        DON'T pass fetch=False should still go through ingest_latest()
        (which is itself TTL-gated). Confirms the default isn't
        accidentally flipped to read-only."""
        with patch("bridge.news.service.fetch_all", return_value=([], 1, 0)) as m:
            service.get_news()
        assert m.call_count == 1


# --- ingest merge behaviour ------------------------------------------------

class TestIngestMerge:
    def test_new_articles_are_added(self, service):
        raw = [_raw(title="Tokyo mayor wins election decisively today",
                    source_id="bbc", url="https://x.test/1")]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            result = service.ingest_latest()
        assert result.new_articles == 1
        assert len(result.new_article_ids) == 1
        # Articles persisted.
        assert len(service._ledger.load_articles()) == 1

    def test_same_article_not_re_added(self, service):
        raw = [_raw(title="Tokyo mayor wins election decisively today",
                    source_id="bbc", url="https://x.test/1")]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            service.ingest_latest()
            # Force bypass TTL so fetch_all runs again.
            result = service.ingest_latest(force=True)
        assert result.new_articles == 0, (
            "second ingest with same URL must not create a duplicate"
        )
        assert len(service._ledger.load_articles()) == 1


# --- get_news ranking ------------------------------------------------------

class TestRanking:
    def test_confirmed_beats_provisional_beats_developing(self, service):
        # 3 reputable non-wire outlets on one story -> confirmed
        # 2 reputable non-wire outlets on another  -> provisional
        # 1 reputable outlet alone                 -> developing
        raw = [
            _raw(title="Confirmed story about economic policy news today",
                 source_id="bbc",      url="https://x.test/c1", hours_ago=5),
            _raw(title="Confirmed story about economic policy news today",
                 source_id="npr",      url="https://x.test/c2", hours_ago=5),
            _raw(title="Confirmed story about economic policy news today",
                 source_id="guardian", url="https://x.test/c3", hours_ago=5),
            _raw(title="Provisional story on climate summit progress today",
                 source_id="bbc", url="https://x.test/p1", hours_ago=2),
            _raw(title="Provisional story on climate summit progress today",
                 source_id="npr", url="https://x.test/p2", hours_ago=2),
            _raw(title="Solo developing story with single outlet reporting",
                 source_id="bbc", url="https://x.test/d1", hours_ago=1),
        ]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            briefs = service.get_news(max_results=5)
        assert len(briefs) == 3
        assert briefs[0].state == ConfidenceState.CONFIRMED
        assert briefs[1].state == ConfidenceState.PROVISIONALLY_CONFIRMED
        assert briefs[2].state == ConfidenceState.DEVELOPING

    def test_recency_tiebreaks_within_same_state(self, service):
        # Two DEVELOPING clusters (each a solo BBC story). Newer wins.
        raw = [
            _raw(title="Older developing story about something today",
                 source_id="bbc", url="https://x.test/old", hours_ago=10),
            _raw(title="Newer developing story about other topic now",
                 source_id="bbc", url="https://x.test/new", hours_ago=1),
        ]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            briefs = service.get_news(max_results=2)
        assert briefs[0].headline.startswith("Newer"), (
            "more recent developing story should rank first"
        )


# --- topic filter ----------------------------------------------------------

class TestTopicFilter:
    def test_keyword_match(self, service):
        raw = [
            _raw(title="Tokyo mayor wins re-election in historic vote",
                 source_id="bbc", url="https://x.test/tokyo", hours_ago=3),
            _raw(title="Osaka earthquake causes widespread damage today",
                 source_id="bbc", url="https://x.test/osaka", hours_ago=2),
        ]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            briefs = service.get_news(topic="tokyo", max_results=5)
        assert len(briefs) == 1
        assert "Tokyo" in briefs[0].headline

    def test_no_match_falls_back_to_recent(self, service):
        raw = [
            _raw(title="Tokyo mayor wins re-election in historic vote",
                 source_id="bbc", url="https://x.test/tokyo", hours_ago=3),
            _raw(title="Osaka earthquake causes widespread damage today",
                 source_id="bbc", url="https://x.test/osaka", hours_ago=2),
        ]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            briefs = service.get_news(topic="nonexistent_keyword", max_results=5)
        # Fallback: returns all clusters ranked by recency.
        assert len(briefs) == 2


# --- get_cluster -----------------------------------------------------------

class TestGetCluster:
    def test_returns_brief_for_existing(self, service):
        raw = [_raw(title="Some story with plenty of tokens for clustering",
                    source_id="bbc", url="https://x.test/1", hours_ago=1)]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            service.ingest_latest()
        clusters = service._ledger.load_clusters()
        cid = next(iter(clusters))
        brief = service.get_cluster(cid)
        assert brief is not None
        assert brief.cluster_id == cid

    def test_returns_none_for_unknown(self, service):
        assert service.get_cluster("nonexistent_id") is None


# --- refresh_and_rebuild ---------------------------------------------------

class TestRefreshAndRebuild:
    def test_rebuilds_without_refetching(self, service):
        raw = [_raw(title="A story that will be clustered and rescored later",
                    source_id="bbc", url="https://x.test/1", hours_ago=1)]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            service.ingest_latest()

        before_articles = service._ledger.load_articles()
        before_clusters = service._ledger.load_clusters()
        assert len(before_articles) == 1
        assert len(before_clusters) == 1

        # Rebuild without patching fetch_all — if it tried to fetch it
        # would hit the real URL and likely fail; test asserts it did not.
        service.refresh_and_rebuild()

        after_articles = service._ledger.load_articles()
        after_clusters = service._ledger.load_clusters()
        assert len(after_articles) == len(before_articles)
        assert len(after_clusters) == len(before_clusters)


# --- caching + save-skip ---------------------------------------------------

class TestCaching:
    """The hot path (get_news back-to-back) must hit memory, not disk,
    and must not re-write clusters.json when rescoring produces
    identical state. These two behaviors drove most of the 650ms →
    cached-hit latency win on the Jetson."""

    def test_second_get_news_reuses_cached_articles_and_clusters(self, service):
        # Prime: ingest + first read populate the caches.
        with patch("bridge.news.service.fetch_all", return_value=(
            [_raw(title="Big story today", source_id="bbc", url="https://x.test/1")],
            1, 0,
        )):
            service.ingest_latest(force=True)
        service.get_news(fetch=False)

        # Now wrap the ledger loaders with counters; a second get_news
        # must not hit either.
        articles_calls = {"n": 0}
        clusters_calls = {"n": 0}
        real_load_articles = service._ledger.load_articles
        real_load_clusters = service._ledger.load_clusters

        def tracked_articles():
            articles_calls["n"] += 1
            return real_load_articles()

        def tracked_clusters():
            clusters_calls["n"] += 1
            return real_load_clusters()

        service._ledger.load_articles = tracked_articles
        service._ledger.load_clusters = tracked_clusters

        service.get_news(fetch=False)
        assert articles_calls["n"] == 0
        assert clusters_calls["n"] == 0

    def test_get_news_skips_cluster_save_when_state_unchanged(self, service):
        # Ingest produces a cluster + a save. Subsequent reads within
        # the stale window should not rewrite the 1MB clusters.json.
        with patch("bridge.news.service.fetch_all", return_value=(
            [_raw(title="Static story today", source_id="bbc", url="https://x.test/1")],
            1, 0,
        )):
            service.ingest_latest(force=True)

        save_calls = {"n": 0}
        real_save = service._ledger.save_clusters

        def tracked_save(*a, **kw):
            save_calls["n"] += 1
            return real_save(*a, **kw)

        service._ledger.save_clusters = tracked_save
        service.get_news(fetch=False)
        service.get_news(fetch=False)
        # Zero saves — rescore produced identical state, so the write
        # was correctly elided. Pre-optimization this was 2.
        assert save_calls["n"] == 0

    def test_prune_drops_old_clusters_and_orphan_articles(self, service):
        # Seed with two clusters, one fresh, one 45 days old. ingest's
        # built-in prune hook will drop the old one on the way through —
        # which is exactly what we want to verify, so don't re-invoke
        # prune_old explicitly afterward.
        fresh = _raw(title="Fresh story today", source_id="bbc",
                     url="https://x.test/f", hours_ago=1)
        old = _raw(title="Old story ancient history here", source_id="npr",
                   url="https://x.test/o", hours_ago=45 * 24)
        with patch("bridge.news.service.fetch_all", return_value=([fresh, old], 1, 0)):
            service.ingest_latest(force=True)

        remaining_c = service._ledger.load_clusters()
        remaining_a = service._ledger.load_articles()
        # The 45-day-old cluster should be gone; the fresh one kept.
        assert any(
            c.centroid_display_title.startswith("Fresh") for c in remaining_c.values()
        )
        assert not any(
            c.centroid_display_title.startswith("Old story") for c in remaining_c.values()
        )
        # Article from the old cluster is orphaned and dropped too.
        assert not any(
            a.display_title.startswith("Old story") for a in remaining_a.values()
        )

    def test_prune_disabled_when_retention_zero(self, service):
        fresh = _raw(title="Story", source_id="bbc",
                     url="https://x.test/f", hours_ago=1)
        with patch("bridge.news.service.fetch_all", return_value=([fresh], 1, 0)):
            service.ingest_latest(force=True)
        out = service.prune_old(retention_days=0)
        assert out["skipped"] is True
        assert out["clusters_dropped"] == 0

    def test_ingest_invalidates_clusters_cache(self, service):
        # First read populates cache. Then a fresh ingest adds a new
        # article → cluster gets rebuilt → next read must see the new
        # cluster, not the stale cached one.
        with patch("bridge.news.service.fetch_all", return_value=(
            [_raw(title="First story today", source_id="bbc", url="https://x.test/1")],
            1, 0,
        )):
            service.ingest_latest(force=True)
        first = service.get_news(fetch=False)
        assert len(first) == 1

        with patch("bridge.news.service.fetch_all", return_value=(
            [_raw(title="Second unrelated headline here", source_id="npr", url="https://x.test/2")],
            1, 0,
        )):
            service.ingest_latest(force=True)
        second = service.get_news(fetch=False)
        assert len(second) == 2   # cache saw the new cluster
