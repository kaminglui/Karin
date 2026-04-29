"""Tests for bridge.news.cluster.count_independent_confirmations.

Pins the spec-delta #4 rule precisely:
  - Articles with non-null wire_attribution collapse to one bucket per wire
  - Articles without wire_attribution contribute one bucket per unique
    ownership_group
  - Articles with unknown source_id contribute nothing
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bridge.news.cluster import count_independent_confirmations
from bridge.news.models import NormalizedArticle, Source, Tier


# --- fixtures --------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _t(hours: float) -> datetime:
    return _EPOCH + timedelta(hours=hours)


def _article(
    *,
    article_id: str,
    source_id: str,
    wire_attribution: str | None = None,
    display_title: str = "x",
    published_at: datetime | None = None,
) -> NormalizedArticle:
    if published_at is None:
        published_at = _t(0)
    return NormalizedArticle(
        article_id=article_id,
        source_id=source_id,
        url=f"https://x.test/{article_id}",
        display_title=display_title,
        normalized_title=display_title.lower(),
        summary="",
        fingerprint=f"fp_{article_id}",
        wire_attribution=wire_attribution,
        published_at=published_at,
        fetched_at=published_at,
    )


def _sources() -> dict[str, Source]:
    """Minimal source registry for tests — just enough distinct
    ownership_groups to exercise the counting rule."""
    return {
        "ap": Source(id="ap", name="AP", domain="apnews.com",
                     tier=Tier.WIRE, ownership_group="ap", is_wire_service=True),
        "reuters": Source(id="reuters", name="Reuters", domain="reuters.com",
                          tier=Tier.WIRE, ownership_group="thomson_reuters",
                          is_wire_service=True),
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


# --- tests -----------------------------------------------------------------

class TestIndependence:
    def test_empty_list_returns_zero(self):
        assert count_independent_confirmations([], _sources()) == 0

    def test_single_non_wire_article_counts_one(self):
        a = _article(article_id="1", source_id="bbc")
        assert count_independent_confirmations([a], _sources()) == 1

    def test_same_wire_across_three_outlets_counts_one(self):
        # Classic syndication case: one AP wire story surfaced via three
        # outlets. The wire byline is identical -> ONE independent bucket.
        arts = [
            _article(article_id="1", source_id="ap", wire_attribution="AP"),
            _article(article_id="2", source_id="washpost", wire_attribution="AP"),
            _article(article_id="3", source_id="nyt", wire_attribution="AP"),
        ]
        assert count_independent_confirmations(arts, _sources()) == 1

    def test_three_distinct_non_wire_outlets_count_three(self):
        # Three independent newsrooms reporting separately -> 3 buckets.
        arts = [
            _article(article_id="1", source_id="bbc"),
            _article(article_id="2", source_id="npr"),
            _article(article_id="3", source_id="guardian"),
        ]
        assert count_independent_confirmations(arts, _sources()) == 3

    def test_same_ownership_group_non_wire_counts_one(self):
        # Two outlets in the same ownership_group reporting independently
        # (no wire attribution) -> ONE bucket, because they're not
        # editorially independent.
        srcs = _sources()
        srcs["nyt_sibling"] = Source(
            id="nyt_sibling", name="NYT Sibling", domain="nytsib.com",
            tier=Tier.REPUTABLE, ownership_group="nyt", is_wire_service=False,
        )
        arts = [
            _article(article_id="1", source_id="nyt"),
            _article(article_id="2", source_id="nyt_sibling"),
        ]
        assert count_independent_confirmations(arts, srcs) == 1

    def test_mixed_wire_and_non_wire(self):
        # 1 AP wire + 2 independent non-wire (BBC, NPR)
        # -> 1 wire bucket + 2 ownership buckets = 3
        arts = [
            _article(article_id="1", source_id="ap", wire_attribution="AP"),
            _article(article_id="2", source_id="bbc"),
            _article(article_id="3", source_id="npr"),
        ]
        assert count_independent_confirmations(arts, _sources()) == 3

    def test_two_different_wires_count_two(self):
        # AP + Reuters on the same story -> two wire buckets.
        arts = [
            _article(article_id="1", source_id="ap", wire_attribution="AP"),
            _article(article_id="2", source_id="reuters", wire_attribution="Reuters"),
        ]
        assert count_independent_confirmations(arts, _sources()) == 2

    def test_unknown_source_contributes_nothing(self):
        # Article from an outlet not in the registry -> contributes no bucket.
        arts = [
            _article(article_id="1", source_id="unknown_outlet"),
            _article(article_id="2", source_id="bbc"),
        ]
        assert count_independent_confirmations(arts, _sources()) == 1

    def test_wire_attribution_takes_precedence_over_ownership(self):
        # An article carries wire_attribution="AP" but its surfacing outlet
        # is BBC. It counts as an AP wire bucket (NOT a bbc ownership bucket).
        # Then a plain BBC article adds the bbc bucket separately.
        arts = [
            _article(article_id="1", source_id="bbc", wire_attribution="AP"),
            _article(article_id="2", source_id="bbc"),
        ]
        # 1 wire (AP) + 1 ownership (bbc) = 2
        assert count_independent_confirmations(arts, _sources()) == 2
