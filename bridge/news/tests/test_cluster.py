"""Tests for bridge.news.cluster.cluster_articles.

Pins the four scenarios the spec called out explicitly, plus three
additional cases that bracket the algorithm's edges:

  1. Lexically-varied same-event merge (recall @ 0.55 threshold)
  2. Same-topic different-event no-merge (precision)
  3. Same wire story across outlets -> one independent confirmation
  4. Distinct non-wire outlets counted independently
  5. Mixed wire + independent arithmetic
  6. Time-bucket violation -> separate clusters
  7. Fingerprint fast-path bypasses time bucket

Plus:
  - single article -> one cluster
  - articles already assigned to an existing cluster are not re-processed
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bridge.news.cluster import cluster_articles
from bridge.news.models import NormalizedArticle, Source, Tier


# --- fixtures --------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _t(hours: float) -> datetime:
    return _EPOCH + timedelta(hours=hours)


def _article(
    *,
    article_id: str,
    source_id: str,
    display_title: str,
    published_at: datetime,
    wire_attribution: str | None = None,
    fingerprint: str | None = None,
) -> NormalizedArticle:
    return NormalizedArticle(
        article_id=article_id,
        source_id=source_id,
        url=f"https://x.test/{article_id}",
        display_title=display_title,
        normalized_title=display_title.lower(),
        summary="",
        fingerprint=fingerprint if fingerprint is not None else f"fp_{article_id}",
        wire_attribution=wire_attribution,
        published_at=published_at,
        fetched_at=published_at,
    )


def _sources() -> dict[str, Source]:
    return {
        "ap": Source(id="ap", name="AP", domain="apnews.com",
                     tier=Tier.WIRE, ownership_group="ap", is_wire_service=True),
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
        "nyt": Source(id="nyt", name="NYT", domain="nytimes.com",
                      tier=Tier.REPUTABLE, ownership_group="nyt",
                      is_wire_service=False),
        "washpost": Source(id="washpost", name="Washington Post",
                           domain="washingtonpost.com", tier=Tier.REPUTABLE,
                           ownership_group="washpost", is_wire_service=False),
    }


def _as_dict(articles: list[NormalizedArticle]) -> dict[str, NormalizedArticle]:
    return {a.article_id: a for a in articles}


# --- tests -----------------------------------------------------------------

class TestBasicAssignment:
    def test_single_article_creates_one_cluster(self):
        a = _article(article_id="1", source_id="bbc",
                     display_title="Tokyo mayor wins re-election",
                     published_at=_t(0))
        clusters, affected = cluster_articles(_as_dict([a]), {}, _sources())
        assert len(clusters) == 1
        assert len(affected) == 1
        c = next(iter(clusters.values()))
        assert c.article_ids == ["1"]
        assert c.article_count == 1
        assert c.independent_confirmation_count == 1
        assert c.syndicated_article_count == 0
        assert c.centroid_display_title == "Tokyo mayor wins re-election"


class TestMergeSameEvent:
    def test_lexically_varied_same_event_merges(self):
        # Same event, differently worded. Designed to cross the 0.55
        # threshold: token sets share {tokyo, mayor, sato, re} — 4 of 7
        # unioned tokens = 0.571, just over the threshold.
        a1 = _article(
            article_id="1", source_id="bbc",
            display_title="Tokyo Mayor Sato Wins Re-Election",
            published_at=_t(0),
        )
        a2 = _article(
            article_id="2", source_id="npr",
            display_title="Sato Re-Elected as Tokyo Mayor",
            published_at=_t(6),  # 6h later, well within bucket
        )
        clusters, affected = cluster_articles(_as_dict([a1, a2]), {}, _sources())
        assert len(clusters) == 1, "varied same-event headlines should merge"
        c = next(iter(clusters.values()))
        assert set(c.article_ids) == {"1", "2"}
        assert c.article_count == 2
        assert c.independent_confirmation_count == 2  # bbc + npr
        assert c.syndicated_article_count == 0


class TestNoMergeDifferentEvent:
    def test_same_topic_different_event_do_not_merge(self):
        # Same topic (Tokyo mayor Sato) but opposite event. Token overlap
        # is {tokyo, mayor, sato} — 3 of 9 union = 0.333 < 0.55.
        a1 = _article(
            article_id="1", source_id="bbc",
            display_title="Tokyo Mayor Sato Wins Re-Election",
            published_at=_t(0),
        )
        a2 = _article(
            article_id="2", source_id="npr",
            display_title="Tokyo Mayor Sato Resigns Amid Scandal",
            published_at=_t(6),  # within time bucket — precision test
        )
        clusters, _ = cluster_articles(_as_dict([a1, a2]), {}, _sources())
        assert len(clusters) == 2, "different events on same topic must not merge"


class TestWireSyndication:
    def test_same_wire_across_three_outlets_one_cluster_one_confirmation(self):
        # Canonical syndication case. Three outlets carry the same AP wire
        # story with identical fingerprints — fingerprint fast path merges
        # them, and the wire-attribution rule collapses to ONE independent
        # confirmation.
        arts = [
            _article(article_id="1", source_id="ap",
                     display_title="Breaking news from Washington",
                     published_at=_t(0), wire_attribution="AP",
                     fingerprint="wire_fp_1"),
            _article(article_id="2", source_id="washpost",
                     display_title="Breaking news from Washington",
                     published_at=_t(1), wire_attribution="AP",
                     fingerprint="wire_fp_1"),
            _article(article_id="3", source_id="nyt",
                     display_title="Breaking news from Washington",
                     published_at=_t(2), wire_attribution="AP",
                     fingerprint="wire_fp_1"),
        ]
        clusters, _ = cluster_articles(_as_dict(arts), {}, _sources())
        assert len(clusters) == 1
        c = next(iter(clusters.values()))
        assert c.article_count == 3
        assert c.independent_confirmation_count == 1
        assert c.syndicated_article_count == 2


class TestDistinctNonWireOutlets:
    def test_three_independent_newsrooms_count_separately(self):
        # Three independent outlets each reporting the same event with
        # their own headline. Pairs of headlines must cross the 0.55
        # threshold against the seed article for all three to land in
        # one cluster.
        #
        # Token sets (stopwords "by"/"a"/"the" removed):
        #   bbc:      {osaka, earthquake, causes, major, damage}
        #   npr:      {major, earthquake, strikes, osaka, causing, damage}
        #   guardian: {osaka, region, hit, major, earthquake, damage}
        #
        # Each pair vs bbc:
        #   npr vs bbc:      inter={osaka,earthquake,major,damage}=4,
        #                    union=7, j=4/7=0.571 PASS
        #   guardian vs bbc: inter={osaka,earthquake,major,damage}=4,
        #                    union=7, j=4/7=0.571 PASS
        arts = [
            _article(article_id="1", source_id="bbc",
                     display_title="Osaka Earthquake Causes Major Damage",
                     published_at=_t(0)),
            _article(article_id="2", source_id="npr",
                     display_title="Major Earthquake Strikes Osaka Causing Damage",
                     published_at=_t(2)),
            _article(article_id="3", source_id="guardian",
                     display_title="Osaka Region Hit by Major Earthquake Damage",
                     published_at=_t(4)),
        ]
        clusters, _ = cluster_articles(_as_dict(arts), {}, _sources())
        assert len(clusters) == 1, "three varied but same-event headlines should merge"
        c = next(iter(clusters.values()))
        assert c.article_count == 3
        assert c.independent_confirmation_count == 3  # bbc + npr + guardian
        assert c.syndicated_article_count == 0


class TestMixedWireAndIndependent:
    def test_wire_plus_two_non_wire_counts_three(self):
        # Mix: AP wire article + two independent newsroom reports that
        # lexically cluster with the AP headline. Expected:
        #   independent_confirmation_count = 1 (AP) + 2 (bbc, npr) = 3
        #   article_count = 3, syndicated_article_count = 0
        arts = [
            _article(article_id="1", source_id="ap",
                     display_title="Osaka Earthquake Causes Major Damage",
                     published_at=_t(0), wire_attribution="AP"),
            _article(article_id="2", source_id="bbc",
                     display_title="Major Earthquake Strikes Osaka Causing Damage",
                     published_at=_t(2)),
            _article(article_id="3", source_id="npr",
                     display_title="Osaka Region Hit by Major Earthquake Damage",
                     published_at=_t(4)),
        ]
        clusters, _ = cluster_articles(_as_dict(arts), {}, _sources())
        assert len(clusters) == 1
        c = next(iter(clusters.values()))
        assert c.article_count == 3
        assert c.independent_confirmation_count == 3
        assert c.syndicated_article_count == 0


class TestTimeBucket:
    def test_identical_headlines_outside_bucket_do_not_merge(self):
        # Same headline, 72h apart (> 48h bucket), different fingerprints
        # so fingerprint fast path does not trigger. Must split into two
        # clusters — the older story is a distinct event.
        a1 = _article(article_id="1", source_id="bbc",
                      display_title="Tokyo Mayor Wins Re-Election",
                      published_at=_t(0), fingerprint="fp_unique_1")
        a2 = _article(article_id="2", source_id="npr",
                      display_title="Tokyo Mayor Wins Re-Election",
                      published_at=_t(72), fingerprint="fp_unique_2")
        clusters, _ = cluster_articles(_as_dict([a1, a2]), {}, _sources())
        assert len(clusters) == 2, "same headline 72h apart must not merge"


class TestFingerprintFastPath:
    def test_identical_fingerprint_merges_across_time_bucket(self):
        # Two articles with identical fingerprints but published 72h apart.
        # Lexical path is blocked by the time bucket; fingerprint fast path
        # merges them anyway. Models wire-syndicated content republished
        # days later.
        a1 = _article(article_id="1", source_id="ap",
                      display_title="Some verbatim wire headline here",
                      published_at=_t(0), wire_attribution="AP",
                      fingerprint="shared_fp")
        a2 = _article(article_id="2", source_id="washpost",
                      display_title="Some verbatim wire headline here",
                      published_at=_t(72), wire_attribution="AP",
                      fingerprint="shared_fp")
        clusters, _ = cluster_articles(_as_dict([a1, a2]), {}, _sources())
        assert len(clusters) == 1
        c = next(iter(clusters.values()))
        assert c.article_count == 2
        assert c.independent_confirmation_count == 1  # one wire bucket
        assert c.syndicated_article_count == 1


class TestLowSignalHeadlineGuard:
    def test_generic_headline_does_not_cause_false_merge(self):
        # "Here's the latest" is the NYT stock headline used across
        # multiple unrelated ongoing stories. Tokens after cleanup are
        # {here, latest} = 2, below MIN_INFORMATIVE_TOKENS=3, so the
        # lexical path is suppressed entirely and these articles must
        # not merge even though their headlines are identical.
        # (Their fingerprints differ because summaries differ.)
        a1 = _article(article_id="1", source_id="nyt",
                      display_title="Here's the latest.",
                      published_at=_t(0))
        a2 = _article(article_id="2", source_id="nyt",
                      display_title="Here's the latest.",
                      published_at=_t(1))
        clusters, _ = cluster_articles(_as_dict([a1, a2]), {}, _sources())
        assert len(clusters) == 2, "stock headlines must not lexically merge"

    def test_low_signal_headline_still_merges_on_fingerprint(self):
        # Same stock headline, but identical fingerprints (e.g. verbatim
        # syndicated wire republication). Fingerprint fast path bypasses
        # the token guard — these SHOULD merge.
        a1 = _article(article_id="1", source_id="ap",
                      display_title="Here's the latest.",
                      published_at=_t(0), wire_attribution="AP",
                      fingerprint="shared_fp")
        a2 = _article(article_id="2", source_id="nyt",
                      display_title="Here's the latest.",
                      published_at=_t(1), wire_attribution="AP",
                      fingerprint="shared_fp")
        clusters, _ = cluster_articles(_as_dict([a1, a2]), {}, _sources())
        assert len(clusters) == 1
        c = next(iter(clusters.values()))
        assert c.article_count == 2


class TestExistingClusterIntegration:
    def test_article_already_in_cluster_not_reprocessed(self):
        # article_1 is already assigned to C1; a new unrelated article_2
        # arrives. C1 must NOT appear in affected (nothing changed); a new
        # cluster for article_2 must be created.
        a1 = _article(article_id="1", source_id="bbc",
                      display_title="First story headline",
                      published_at=_t(0))
        a2 = _article(article_id="2", source_id="npr",
                      display_title="Completely unrelated other story",
                      published_at=_t(1))
        # Pre-build an existing cluster for a1.
        existing, _ = cluster_articles(_as_dict([a1]), {}, _sources(), now=_t(0))
        assert len(existing) == 1
        existing_cid = next(iter(existing.keys()))

        # Now add a2.
        clusters, affected = cluster_articles(
            _as_dict([a1, a2]), existing, _sources(), now=_t(1),
        )
        assert len(clusters) == 2
        assert existing_cid in clusters
        assert existing_cid not in affected, (
            "pre-existing cluster that gained no new articles must not be affected"
        )
        # Exactly one affected cluster — the new one for a2.
        assert len(affected) == 1

    def test_new_article_joins_existing_cluster(self):
        # a1 pre-clustered; a2 arrives and lexically matches -> joins C1.
        # C1 must appear in affected.
        a1 = _article(article_id="1", source_id="bbc",
                      display_title="Tokyo Mayor Sato Wins Re-Election",
                      published_at=_t(0))
        existing, _ = cluster_articles(_as_dict([a1]), {}, _sources(), now=_t(0))
        existing_cid = next(iter(existing.keys()))

        a2 = _article(article_id="2", source_id="npr",
                      display_title="Sato Re-Elected as Tokyo Mayor",
                      published_at=_t(6))
        clusters, affected = cluster_articles(
            _as_dict([a1, a2]), existing, _sources(), now=_t(6),
        )
        assert len(clusters) == 1
        assert existing_cid in affected
        c = clusters[existing_cid]
        assert c.article_count == 2
        assert set(c.article_ids) == {"1", "2"}
