"""Tests for bridge.news.graph — the Phase F.a relation-graph builder.

Fixtures construct a small cluster set with known watchlist matches,
then we assert node/edge weights + cluster-membership arrays directly.
No network, no I/O — graph is a pure computation.
"""
from __future__ import annotations

from datetime import datetime, timezone

from bridge.news.graph import build_news_graph
from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    Source,
    StoryCluster,
    Tier,
)
from bridge.news.preferences import Preferences, WatchlistItem, RankingConfig


_EPOCH = datetime(2026, 4, 15, tzinfo=timezone.utc)


def _article(
    *, aid: str, source_id: str = "bbc",
    display_title: str, display_summary: str = "",
) -> NormalizedArticle:
    return NormalizedArticle(
        article_id=aid,
        source_id=source_id,
        url=f"https://x.test/{aid}",
        display_title=display_title,
        normalized_title=display_title.lower(),
        summary=display_summary.lower(),
        fingerprint=f"fp_{aid}",
        wire_attribution=None,
        published_at=_EPOCH,
        fetched_at=_EPOCH,
        display_summary=display_summary,
        language="en",
    )


def _cluster(
    *, cluster_id: str, article_ids: list[str],
    centroid_display_title: str,
) -> StoryCluster:
    return StoryCluster(
        cluster_id=cluster_id,
        article_ids=list(article_ids),
        centroid_display_title=centroid_display_title,
        centroid_normalized_title=centroid_display_title.lower(),
        first_seen_at=_EPOCH,
        latest_update_at=_EPOCH,
        last_checked_at=_EPOCH,
        last_state_change_at=_EPOCH,
        state=ConfidenceState.CONFIRMED,
        is_stale=False,
        independent_confirmation_count=3,
        article_count=3,
        syndicated_article_count=0,
    )


def _prefs() -> Preferences:
    return Preferences(
        enabled=True,
        regions=(
            WatchlistItem(id="us", label="US", keywords=("united states", "washington"), priority="high"),
            WatchlistItem(id="china", label="China", keywords=("china", "beijing"), priority="high"),
        ),
        topics=(
            WatchlistItem(id="ai", label="AI / Tech", keywords=("openai", "chipmaker", "ai"), priority="medium"),
        ),
        events=(
            WatchlistItem(id="elections", label="Elections", keywords=("election", "primary"), priority="low"),
        ),
        ranking=RankingConfig(),
    )


# ---------------------------------------------------------------------------
# Empty / disabled paths
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_clusters_returns_empty_graph(self):
        out = build_news_graph({}, {}, _prefs())
        assert out["nodes"] == []
        assert out["edges"] == []
        assert out["counts"] == {"clusters": 0, "nodes": 0, "edges": 0}

    def test_preferences_disabled_returns_empty(self):
        # Prefs off -> match_watchlist_items returns [] -> no nodes.
        a = _article(aid="a1", display_title="United States announces plan")
        c = _cluster(
            cluster_id="c1", article_ids=["a1"],
            centroid_display_title="United States announces plan",
        )
        prefs = Preferences(enabled=False)
        out = build_news_graph({"c1": c}, {"a1": a}, prefs)
        assert out["nodes"] == []
        assert out["edges"] == []


# ---------------------------------------------------------------------------
# Node aggregation
# ---------------------------------------------------------------------------

class TestNodeWeights:
    def test_single_cluster_single_match(self):
        a = _article(aid="a1", display_title="United States announces plan")
        c = _cluster(
            cluster_id="c1", article_ids=["a1"],
            centroid_display_title="United States announces plan",
        )
        out = build_news_graph({"c1": c}, {"a1": a}, _prefs())
        assert len(out["nodes"]) == 1
        n = out["nodes"][0]
        assert n["id"] == "region:us"
        assert n["kind"] == "region"
        assert n["label"] == "US"
        assert n["weight"] == 1
        assert n["cluster_ids"] == ["c1"]

    def test_multiple_clusters_aggregate_weight(self):
        # Two clusters both match US -> US node weight == 2.
        articles = {
            "a1": _article(aid="a1", display_title="United States announces plan"),
            "a2": _article(aid="a2", display_title="Washington budget talks"),
        }
        clusters = {
            "c1": _cluster(cluster_id="c1", article_ids=["a1"],
                           centroid_display_title="United States announces plan"),
            "c2": _cluster(cluster_id="c2", article_ids=["a2"],
                           centroid_display_title="Washington budget talks"),
        }
        out = build_news_graph(clusters, articles, _prefs())
        us = next(n for n in out["nodes"] if n["id"] == "region:us")
        assert us["weight"] == 2
        assert us["cluster_ids"] == ["c1", "c2"]

    def test_non_matching_clusters_skipped(self):
        # Cluster doesn't match any watchlist -> no nodes, no edges.
        a = _article(aid="a1", display_title="Local parade tomorrow")
        c = _cluster(
            cluster_id="c1", article_ids=["a1"],
            centroid_display_title="Local parade tomorrow",
        )
        out = build_news_graph({"c1": c}, {"a1": a}, _prefs())
        assert out["counts"]["clusters"] == 0
        assert out["nodes"] == []


# ---------------------------------------------------------------------------
# Edge aggregation
# ---------------------------------------------------------------------------

class TestEdges:
    def test_co_occurring_matches_produce_edge(self):
        # One cluster mentions BOTH US and China -> edge between them.
        a = _article(
            aid="a1",
            display_title="United States and China meet",
            display_summary="beijing and washington announce talks",
        )
        c = _cluster(
            cluster_id="c1", article_ids=["a1"],
            centroid_display_title="United States and China meet",
        )
        out = build_news_graph({"c1": c}, {"a1": a}, _prefs())
        assert len(out["edges"]) == 1
        e = out["edges"][0]
        assert {e["source"], e["target"]} == {"region:us", "region:china"}
        assert e["weight"] == 1
        assert e["cluster_ids"] == ["c1"]

    def test_repeated_co_occurrence_increments_weight(self):
        articles = {
            "a1": _article(
                aid="a1",
                display_title="United States and China meet",
                display_summary="beijing and washington announce talks",
            ),
            "a2": _article(
                aid="a2",
                display_title="China warns United States about trade",
            ),
        }
        clusters = {
            "c1": _cluster(cluster_id="c1", article_ids=["a1"],
                           centroid_display_title="United States and China meet"),
            "c2": _cluster(cluster_id="c2", article_ids=["a2"],
                           centroid_display_title="China warns United States about trade"),
        }
        out = build_news_graph(clusters, articles, _prefs())
        us_china = next(
            e for e in out["edges"]
            if {e["source"], e["target"]} == {"region:us", "region:china"}
        )
        assert us_china["weight"] == 2
        assert us_china["cluster_ids"] == ["c1", "c2"]

    def test_cross_kind_edge_region_to_topic(self):
        a = _article(
            aid="a1",
            display_title="OpenAI chipmaker news from Washington",
            display_summary="united states regulators weigh ai oversight",
        )
        c = _cluster(
            cluster_id="c1", article_ids=["a1"],
            centroid_display_title="OpenAI chipmaker news from Washington",
        )
        out = build_news_graph({"c1": c}, {"a1": a}, _prefs())
        # Should produce nodes for US (region) and AI / Tech (topic),
        # plus one edge between them.
        node_ids = {n["id"] for n in out["nodes"]}
        assert node_ids == {"region:us", "topic:ai"}
        assert len(out["edges"]) == 1
        e = out["edges"][0]
        assert {e["source"], e["target"]} == {"region:us", "topic:ai"}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterministicOrder:
    def test_node_order_is_stable(self):
        # Regions first (kind rank 0), sorted by label within kind.
        articles = {
            "a1": _article(
                aid="a1",
                display_title="United States and China meet",
                display_summary="beijing washington",
            ),
            "a2": _article(
                aid="a2",
                display_title="OpenAI chipmaker",
                display_summary="ai regulators washington",
            ),
        }
        clusters = {
            "c1": _cluster(cluster_id="c1", article_ids=["a1"],
                           centroid_display_title="United States and China meet"),
            "c2": _cluster(cluster_id="c2", article_ids=["a2"],
                           centroid_display_title="OpenAI chipmaker"),
        }
        out = build_news_graph(clusters, articles, _prefs())
        kinds = [n["kind"] for n in out["nodes"]]
        # All regions come before all topics.
        assert kinds == sorted(kinds, key=lambda k: {"region": 0, "topic": 1, "event": 2}[k])
        # Alphabetical within region block: China (C) before US (U).
        region_labels = [n["label"] for n in out["nodes"] if n["kind"] == "region"]
        assert region_labels == sorted(region_labels, key=str.lower)

    def test_edge_order_is_by_weight_desc(self):
        # Heavier edge should sort before lighter ones.
        articles = {
            "a1": _article(aid="a1", display_title="United States and China",
                           display_summary="beijing washington"),
            "a2": _article(aid="a2", display_title="United States and China again",
                           display_summary="beijing washington"),
            "a3": _article(aid="a3", display_title="OpenAI chipmaker Washington",
                           display_summary="ai united states"),
        }
        clusters = {
            "c1": _cluster(cluster_id="c1", article_ids=["a1"],
                           centroid_display_title="United States and China"),
            "c2": _cluster(cluster_id="c2", article_ids=["a2"],
                           centroid_display_title="United States and China again"),
            "c3": _cluster(cluster_id="c3", article_ids=["a3"],
                           centroid_display_title="OpenAI chipmaker Washington"),
        }
        out = build_news_graph(clusters, articles, _prefs())
        weights = [e["weight"] for e in out["edges"]]
        assert weights == sorted(weights, reverse=True)
