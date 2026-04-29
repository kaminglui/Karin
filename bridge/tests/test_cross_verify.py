"""Tests for bridge.news.cross_verify.

Mocked LLM, pure functions. Verifies:
  - candidate-pair selection applies all gates (size, freshness, band)
  - LLM 'SAME' merges, 'DIFFERENT' doesn't, errors don't
  - rate cap caps calls AND no pair consumes a cluster twice
  - audit events fire one-per-ask
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bridge.news.cross_verify import (
    BORDERLINE_JACCARD_HIGH,
    BORDERLINE_JACCARD_LOW,
    MAX_LLM_CALLS_PER_CYCLE,
    VerificationEvent,
    cross_verify_clusters,
    find_candidate_pairs,
)
from bridge.news.models import NormalizedArticle, StoryCluster, ConfidenceState


# ---- fixtures ------------------------------------------------------------


def _art(aid: str, title: str, published_at: datetime) -> NormalizedArticle:
    return NormalizedArticle(
        article_id=aid, source_id="src1", url=f"https://x/{aid}",
        display_title=title, normalized_title=title.lower(),
        summary="", fingerprint=f"fp-{aid}",
        wire_attribution=None,
        published_at=published_at,
        fetched_at=published_at,
    )


def _cluster(cid: str, article_ids: list[str], title: str,
             created_at: datetime, latest_update: datetime) -> StoryCluster:
    n = len(article_ids)
    return StoryCluster(
        cluster_id=cid,
        article_ids=list(article_ids),
        centroid_display_title=title,
        centroid_normalized_title=title.lower(),
        first_seen_at=created_at,
        latest_update_at=latest_update,
        last_checked_at=created_at,
        last_state_change_at=created_at,
        state=ConfidenceState.DEVELOPING,
        is_stale=False,
        independent_confirmation_count=1,
        article_count=n,
        syndicated_article_count=max(0, n - 1),
    )


def _two_borderline_clusters(now: datetime):
    """Build a fixture where two clusters have borderline Jaccard.

    Titles share {major, earthquake, turkey, southern} = 4 tokens.
    Unique union = {major, earthquake, strikes, turkey, southern,
    coastal, region, devastates, zone} = 9 tokens. Jaccard 4/9 ≈ 0.44
    which sits inside the [0.30, 0.55) borderline band.
    """
    articles = {
        "a1": _art("a1", "major earthquake strikes turkey southern coastal region", now),
        "a2": _art("a2", "major earthquake strikes turkey southern coastal region", now),
        "b1": _art("b1", "major earthquake devastates turkey southern zone", now),
        "b2": _art("b2", "major earthquake devastates turkey southern zone", now),
    }
    clusters = {
        "c_a": _cluster("c_a", ["a1", "a2"],
                        "major earthquake strikes turkey southern coastal region",
                        now, now),
        "c_b": _cluster("c_b", ["b1", "b2"],
                        "major earthquake devastates turkey southern zone",
                        now, now),
    }
    return clusters, articles


# ---- candidate selection ------------------------------------------------


class TestFindCandidatePairs:
    def test_borderline_pair_is_selected(self):
        now = datetime.now(timezone.utc)
        clusters, articles = _two_borderline_clusters(now)
        pairs = find_candidate_pairs(clusters, articles, now=now)
        assert len(pairs) == 1
        a, b, score = pairs[0]
        assert {a, b} == {"c_a", "c_b"}
        assert BORDERLINE_JACCARD_LOW <= score < BORDERLINE_JACCARD_HIGH

    def test_identical_titles_not_selected(self):
        """Jaccard >= HIGH means lexical matcher already merged these —
        not a borderline case, skip."""
        now = datetime.now(timezone.utc)
        articles = {
            "a1": _art("a1", "earthquake strikes turkey", now),
            "a2": _art("a2", "earthquake strikes turkey", now),
            "b1": _art("b1", "earthquake strikes turkey", now),
            "b2": _art("b2", "earthquake strikes turkey", now),
        }
        clusters = {
            "c_a": _cluster("c_a", ["a1", "a2"], "earthquake strikes turkey", now, now),
            "c_b": _cluster("c_b", ["b1", "b2"], "earthquake strikes turkey", now, now),
        }
        assert find_candidate_pairs(clusters, articles, now=now) == []

    def test_completely_unrelated_not_selected(self):
        """Below LOW means no semantic overlap — not worth asking."""
        now = datetime.now(timezone.utc)
        articles = {
            "a1": _art("a1", "earthquake strikes turkey coast today", now),
            "a2": _art("a2", "earthquake strikes turkey coast today", now),
            "b1": _art("b1", "chocolate cake recipe baking tutorial", now),
            "b2": _art("b2", "chocolate cake recipe baking tutorial", now),
        }
        clusters = {
            "c_a": _cluster("c_a", ["a1", "a2"],
                            "earthquake strikes turkey coast today", now, now),
            "c_b": _cluster("c_b", ["b1", "b2"],
                            "chocolate cake recipe baking tutorial", now, now),
        }
        assert find_candidate_pairs(clusters, articles, now=now) == []

    def test_singleton_clusters_ARE_eligible(self):
        """Singletons are the PRIMARY target — they're the cases the
        lexical matcher missed. Cross-verify should ask about them when
        Jaccard is borderline, relying on the rate cap for control."""
        now = datetime.now(timezone.utc)
        articles = {
            "a1": _art("a1", "major earthquake strikes turkey southern coastal region", now),
            "b1": _art("b1", "major earthquake devastates turkey southern zone", now),
        }
        clusters = {
            "c_a": _cluster("c_a", ["a1"],
                            "major earthquake strikes turkey southern coastal region",
                            now, now),
            "c_b": _cluster("c_b", ["b1"],
                            "major earthquake devastates turkey southern zone",
                            now, now),
        }
        pairs = find_candidate_pairs(clusters, articles, now=now)
        assert len(pairs) == 1

    def test_stale_clusters_skipped(self):
        """Both clusters need ≥1 fresh article to qualify."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=3)
        articles = {
            "a1": _art("a1", "earthquake strikes turkey southern region", old),
            "a2": _art("a2", "earthquake strikes turkey southern region", old),
            "b1": _art("b1", "seismic event devastates turkey southern zone", old),
            "b2": _art("b2", "seismic event devastates turkey southern zone", old),
        }
        clusters = {
            "c_a": _cluster("c_a", ["a1", "a2"],
                            "earthquake strikes turkey southern region", old, old),
            "c_b": _cluster("c_b", ["b1", "b2"],
                            "seismic event devastates turkey southern zone", old, old),
        }
        assert find_candidate_pairs(clusters, articles, now=now) == []


# ---- cross_verify_clusters ---------------------------------------------


class _FakeVerifier:
    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[tuple[str, str]] = []

    def same_event(self, title_a: str, title_b: str) -> bool:
        self.calls.append((title_a, title_b))
        return self.verdicts.pop(0) if self.verdicts else False


class _ExplodingVerifier:
    def same_event(self, title_a: str, title_b: str) -> bool:
        raise RuntimeError("llm is down")


class TestCrossVerifyClusters:
    def test_merge_on_same(self):
        now = datetime.now(timezone.utc)
        clusters, articles = _two_borderline_clusters(now)
        verifier = _FakeVerifier([True])
        events: list[VerificationEvent] = []
        result = cross_verify_clusters(
            clusters, articles, verifier,
            now=now, event_sink=events.append,
        )
        # One cluster absorbed into the other → 1 cluster left.
        assert len(result) == 1
        remaining = next(iter(result.values()))
        # Merged cluster has all 4 articles.
        assert len(remaining.article_ids) == 4
        # Audit event fired
        assert len(events) == 1
        assert events[0].merged is True

    def test_no_merge_on_different(self):
        now = datetime.now(timezone.utc)
        clusters, articles = _two_borderline_clusters(now)
        verifier = _FakeVerifier([False])
        events: list[VerificationEvent] = []
        result = cross_verify_clusters(
            clusters, articles, verifier,
            now=now, event_sink=events.append,
        )
        assert len(result) == 2
        assert events[0].merged is False
        assert events[0].llm_said_same is False

    def test_llm_error_treated_as_different(self):
        now = datetime.now(timezone.utc)
        clusters, articles = _two_borderline_clusters(now)
        verifier = _ExplodingVerifier()
        result = cross_verify_clusters(
            clusters, articles, verifier, now=now,
        )
        # Fail-safe: no merge when the LLM throws.
        assert len(result) == 2

    def test_rate_cap_is_respected(self):
        """With MAX_LLM_CALLS_PER_CYCLE+1 eligible pairs, at most
        MAX_LLM_CALLS_PER_CYCLE round-trips happen."""
        now = datetime.now(timezone.utc)
        # Build many borderline pairs by instantiating many clusters
        # with nearly-but-not-quite-identical centroids.
        articles: dict[str, NormalizedArticle] = {}
        clusters: dict[str, StoryCluster] = {}
        n = MAX_LLM_CALLS_PER_CYCLE + 3
        base_words = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON",
                      "ZETA", "ETA", "THETA", "IOTA", "KAPPA"]
        for i in range(n):
            w = base_words[i % len(base_words)]
            a_id, b_id = f"a{i}_1", f"a{i}_2"
            articles[a_id] = _art(
                a_id, f"{w} quake strikes coastal region today", now,
            )
            articles[b_id] = _art(
                b_id, f"{w} quake strikes coastal region today", now,
            )
            clusters[f"c{i}_a"] = _cluster(
                f"c{i}_a", [a_id, b_id],
                f"{w} quake strikes coastal region today", now, now,
            )
        # Add shadow clusters whose titles overlap with originals enough
        # to hit the borderline band. (Easier: copy + swap one word.)
        for i in range(n):
            w = base_words[i % len(base_words)]
            a_id, b_id = f"s{i}_1", f"s{i}_2"
            articles[a_id] = _art(
                a_id, f"{w} earthquake coastal devastated region morning", now,
            )
            articles[b_id] = _art(
                b_id, f"{w} earthquake coastal devastated region morning", now,
            )
            clusters[f"c{i}_b"] = _cluster(
                f"c{i}_b", [a_id, b_id],
                f"{w} earthquake coastal devastated region morning", now, now,
            )
        # Should detect many pairs but cap at the limit.
        verifier = _FakeVerifier([False] * 100)
        cross_verify_clusters(clusters, articles, verifier, now=now)
        assert len(verifier.calls) <= MAX_LLM_CALLS_PER_CYCLE

    def test_absorbed_cluster_not_reconsulted(self):
        """If c_a absorbs c_b, and a later candidate pairs c_b with c_c,
        we should skip — c_b no longer exists in the result."""
        now = datetime.now(timezone.utc)
        articles = {
            "a1": _art("a1", "one borderline title term alpha", now),
            "a2": _art("a2", "one borderline title term alpha", now),
            "b1": _art("b1", "one borderline headline phrase alpha", now),
            "b2": _art("b2", "one borderline headline phrase alpha", now),
            "c1": _art("c1", "one borderline announcement bulletin alpha", now),
            "c2": _art("c2", "one borderline announcement bulletin alpha", now),
        }
        clusters = {
            "c_a": _cluster("c_a", ["a1", "a2"],
                            "one borderline title term alpha", now, now),
            "c_b": _cluster("c_b", ["b1", "b2"],
                            "one borderline headline phrase alpha", now, now),
            "c_c": _cluster("c_c", ["c1", "c2"],
                            "one borderline announcement bulletin alpha", now, now),
        }
        # Verifier merges first pair, rejects any subsequent.
        verifier = _FakeVerifier([True, False, False])
        result = cross_verify_clusters(clusters, articles, verifier, now=now)
        # Exactly one merge happened. Absorbed cluster is gone from result.
        # Merged cluster owns 4 articles (2 + 2).
        assert len(result) == 2
        sizes = sorted(len(c.article_ids) for c in result.values())
        assert sizes == [2, 4]
