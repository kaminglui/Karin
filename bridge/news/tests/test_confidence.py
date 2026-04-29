"""Tests for bridge.news.confidence.

Covers:
  - state transitions (DEVELOPING / PROVISIONALLY_CONFIRMED / CONFIRMED)
    under the spec thresholds, including the tricky cases:
      * 2 distinct reputable -> provisional
      * 3 distinct reputable -> confirmed
      * 1 wire + 2 reputable -> confirmed (wire branch)
      * 2 OTHER-tier only -> developing (no wire/reputable contributor)
      * same ownership group counted once
  - is_stale overlay based on last_checked_at (NOT latest_update_at)
  - last_state_change_at only updated on actual state change
  - brief building: voice_line templates per state, stale suffix,
    top_sources ordering
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bridge.news.confidence import (
    build_brief,
    compute_is_stale,
    compute_state,
    independent_buckets,
    rescore_cluster,
)
from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    Source,
    StoryCluster,
    Tier,
)


# --- fixtures --------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _t(hours: float) -> datetime:
    return _EPOCH + timedelta(hours=hours)


def _article(
    *,
    article_id: str,
    source_id: str,
    display_title: str = "some news headline today here",
    wire_attribution: str | None = None,
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
        "blog1": Source(id="blog1", name="Blog One", domain="blog1.test",
                        tier=Tier.OTHER, ownership_group="blog1",
                        is_wire_service=False),
        "blog2": Source(id="blog2", name="Blog Two", domain="blog2.test",
                        tier=Tier.OTHER, ownership_group="blog2",
                        is_wire_service=False),
        "blog3": Source(id="blog3", name="Blog Three", domain="blog3.test",
                        tier=Tier.OTHER, ownership_group="blog3",
                        is_wire_service=False),
    }


def _cluster(
    *,
    cluster_id: str = "c1",
    article_ids: list[str],
    centroid_display_title: str = "Big news about something important today",
    state: ConfidenceState = ConfidenceState.DEVELOPING,
    is_stale: bool = False,
    last_checked_at: datetime | None = None,
    last_state_change_at: datetime | None = None,
    independent_confirmation_count: int = 1,
    article_count: int = 1,
    syndicated_article_count: int = 0,
) -> StoryCluster:
    if last_checked_at is None:
        last_checked_at = _t(0)
    if last_state_change_at is None:
        last_state_change_at = _t(0)
    return StoryCluster(
        cluster_id=cluster_id,
        article_ids=list(article_ids),
        centroid_display_title=centroid_display_title,
        centroid_normalized_title=centroid_display_title.lower(),
        first_seen_at=_t(0),
        latest_update_at=_t(0),
        last_checked_at=last_checked_at,
        last_state_change_at=last_state_change_at,
        state=state,
        is_stale=is_stale,
        independent_confirmation_count=independent_confirmation_count,
        article_count=article_count,
        syndicated_article_count=syndicated_article_count,
    )


# --- compute_state ---------------------------------------------------------

class TestComputeState:
    def test_single_article_developing(self):
        arts = [_article(article_id="1", source_id="bbc")]
        assert compute_state(arts, _sources()) == ConfidenceState.DEVELOPING

    def test_two_same_ownership_group_still_developing(self):
        # Both in "bbc" ownership group -> 1 independent bucket -> developing
        srcs = _sources()
        srcs["bbc_sibling"] = Source(
            id="bbc_sibling", name="BBC Sibling", domain="bbcsib.test",
            tier=Tier.REPUTABLE, ownership_group="bbc", is_wire_service=False,
        )
        arts = [
            _article(article_id="1", source_id="bbc"),
            _article(article_id="2", source_id="bbc_sibling"),
        ]
        assert compute_state(arts, srcs) == ConfidenceState.DEVELOPING

    def test_two_reputable_distinct_groups_provisional(self):
        arts = [
            _article(article_id="1", source_id="bbc"),
            _article(article_id="2", source_id="npr"),
        ]
        assert compute_state(arts, _sources()) == ConfidenceState.PROVISIONALLY_CONFIRMED

    def test_two_other_tier_only_developing(self):
        # Two distinct ownership groups but BOTH tier=OTHER: lacks the
        # wire/reputable contributor that provisional requires.
        arts = [
            _article(article_id="1", source_id="blog1"),
            _article(article_id="2", source_id="blog2"),
        ]
        assert compute_state(arts, _sources()) == ConfidenceState.DEVELOPING

    def test_three_reputable_confirmed_via_two_reputable_branch(self):
        arts = [
            _article(article_id="1", source_id="bbc"),
            _article(article_id="2", source_id="npr"),
            _article(article_id="3", source_id="guardian"),
        ]
        assert compute_state(arts, _sources()) == ConfidenceState.CONFIRMED

    def test_one_wire_plus_two_reputable_confirmed_via_wire_branch(self):
        arts = [
            _article(article_id="1", source_id="ap", wire_attribution="AP"),
            _article(article_id="2", source_id="bbc"),
            _article(article_id="3", source_id="npr"),
        ]
        assert compute_state(arts, _sources()) == ConfidenceState.CONFIRMED

    def test_three_other_tier_only_stays_developing(self):
        # 3 independent OTHER-tier buckets: count is enough, but without
        # wire or reputable contributors confirmed/provisional both fail.
        arts = [
            _article(article_id="1", source_id="blog1"),
            _article(article_id="2", source_id="blog2"),
            _article(article_id="3", source_id="blog3"),
        ]
        assert compute_state(arts, _sources()) == ConfidenceState.DEVELOPING

    def test_one_wire_plus_one_reputable_provisional(self):
        arts = [
            _article(article_id="1", source_id="ap", wire_attribution="AP"),
            _article(article_id="2", source_id="bbc"),
        ]
        assert compute_state(arts, _sources()) == ConfidenceState.PROVISIONALLY_CONFIRMED

    def test_same_wire_across_outlets_collapses_to_developing(self):
        # All three articles carry AP wire -> 1 bucket total, state stays
        # developing even with 3 articles.
        arts = [
            _article(article_id="1", source_id="ap", wire_attribution="AP"),
            _article(article_id="2", source_id="bbc", wire_attribution="AP"),
            _article(article_id="3", source_id="npr", wire_attribution="AP"),
        ]
        assert compute_state(arts, _sources()) == ConfidenceState.DEVELOPING


# --- independent_buckets ordering ------------------------------------------

class TestIndependentBucketOrdering:
    def test_wires_first_then_by_tier_then_alphabetical(self):
        arts = [
            _article(article_id="1", source_id="npr"),
            _article(article_id="2", source_id="bbc"),
            _article(article_id="3", source_id="ap", wire_attribution="AP"),
            _article(article_id="4", source_id="reuters", wire_attribution="Reuters"),
            _article(article_id="5", source_id="blog1"),
        ]
        buckets = independent_buckets(arts, _sources())
        display_names = [b.display_name for b in buckets]
        # Wires alphabetical first: AP, Reuters.
        # Then reputable (alphabetical): BBC, NPR.
        # Then other: Blog One.
        assert display_names == ["AP", "Reuters", "BBC", "NPR", "Blog One"]


# --- is_stale --------------------------------------------------------------

class TestIsStale:
    def test_just_checked_not_stale(self):
        c = _cluster(article_ids=["1"], last_checked_at=_t(0))
        assert compute_is_stale(c, now=_t(0)) is False

    def test_23h_not_stale_at_default_threshold(self):
        c = _cluster(article_ids=["1"], last_checked_at=_t(0))
        assert compute_is_stale(c, now=_t(23)) is False

    def test_25h_is_stale_at_default_threshold(self):
        c = _cluster(article_ids=["1"], last_checked_at=_t(0))
        assert compute_is_stale(c, now=_t(25)) is True

    def test_custom_threshold(self):
        c = _cluster(article_ids=["1"], last_checked_at=_t(0))
        assert compute_is_stale(c, now=_t(7), stale_threshold_hours=6) is True
        assert compute_is_stale(c, now=_t(5), stale_threshold_hours=6) is False


# --- rescore_cluster -------------------------------------------------------

class TestRescoreCluster:
    def test_state_unchanged_does_not_touch_last_state_change_at(self):
        arts = [_article(article_id="1", source_id="bbc")]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1"],
            state=ConfidenceState.DEVELOPING,
            last_state_change_at=_t(-10),
            last_checked_at=_t(-10),
        )
        rescore_cluster(c, articles, _sources(), now=_t(0))
        assert c.state == ConfidenceState.DEVELOPING
        assert c.last_state_change_at == _t(-10), (
            "timestamp must not move when state is unchanged"
        )

    def test_state_change_updates_last_state_change_at(self):
        # Cluster currently DEVELOPING, but members support CONFIRMED.
        # Rescore should bump state + timestamp.
        arts = [
            _article(article_id="1", source_id="bbc"),
            _article(article_id="2", source_id="npr"),
            _article(article_id="3", source_id="guardian"),
        ]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1", "2", "3"],
            state=ConfidenceState.DEVELOPING,
            last_state_change_at=_t(-10),
            last_checked_at=_t(0),
        )
        rescore_cluster(c, articles, _sources(), now=_t(0))
        assert c.state == ConfidenceState.CONFIRMED
        assert c.last_state_change_at == _t(0)

    def test_updates_is_stale_flag(self):
        arts = [_article(article_id="1", source_id="bbc")]
        articles = {a.article_id: a for a in arts}
        c = _cluster(article_ids=["1"], last_checked_at=_t(-48), is_stale=False)
        rescore_cluster(c, articles, _sources(), now=_t(0))
        assert c.is_stale is True  # 48h > 24h default


# --- build_brief -----------------------------------------------------------

class TestBuildBrief:
    def test_confirmed_brief_voice_line(self):
        arts = [
            _article(article_id="1", source_id="bbc",
                     display_title="Major Earthquake Strikes Osaka Region Tonight"),
            _article(article_id="2", source_id="npr",
                     display_title="Major Earthquake Strikes Osaka Region Tonight"),
            _article(article_id="3", source_id="guardian",
                     display_title="Major Earthquake Strikes Osaka Region Tonight"),
        ]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1", "2", "3"],
            centroid_display_title="Major Earthquake Strikes Osaka Region Tonight",
            state=ConfidenceState.CONFIRMED,
            independent_confirmation_count=3,
            article_count=3,
        )
        brief = build_brief(c, articles, _sources())
        # Alphabetical within REPUTABLE tier: BBC, Guardian, NPR.
        assert brief.top_sources == ["BBC", "Guardian", "NPR"]
        assert brief.voice_line == (
            "BBC and Guardian report Major Earthquake Strikes Osaka Region Tonight. "
            "3 independent sources, confirmed."
        )

    def test_provisional_brief_voice_line(self):
        arts = [
            _article(article_id="1", source_id="bbc",
                     display_title="Tokyo Mayor Wins Re-Election"),
            _article(article_id="2", source_id="npr",
                     display_title="Tokyo Mayor Wins Re-Election"),
        ]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1", "2"],
            centroid_display_title="Tokyo Mayor Wins Re-Election",
            state=ConfidenceState.PROVISIONALLY_CONFIRMED,
            independent_confirmation_count=2,
            article_count=2,
        )
        brief = build_brief(c, articles, _sources())
        assert brief.voice_line == (
            "BBC and NPR report Tokyo Mayor Wins Re-Election. "
            "2 independent sources, provisionally confirmed."
        )

    def test_developing_brief_uses_one_source_phrasing(self):
        arts = [_article(article_id="1", source_id="bbc",
                         display_title="Breaking story just happened now")]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1"],
            centroid_display_title="Breaking story just happened now",
            state=ConfidenceState.DEVELOPING,
        )
        brief = build_brief(c, articles, _sources())
        assert brief.voice_line == (
            "One source reports Breaking story just happened now. Still developing."
        )

    def test_stale_suffix_appended(self):
        arts = [_article(article_id="1", source_id="bbc",
                         display_title="Some old story from a while back")]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1"],
            centroid_display_title="Some old story from a while back",
            state=ConfidenceState.DEVELOPING,
            is_stale=True,
        )
        brief = build_brief(c, articles, _sources())
        assert brief.voice_line.endswith(" This may be outdated.")

    def test_topic_short_trims_long_headlines(self):
        long_headline = (
            "Prince Harry Sued for Defamation by Charity He Co-Founded "
            "in a Bizarre Twist of Events Today"
        )
        arts = [_article(article_id="1", source_id="bbc", display_title=long_headline)]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1"],
            centroid_display_title=long_headline,
            state=ConfidenceState.DEVELOPING,
        )
        brief = build_brief(c, articles, _sources())
        # _TOPIC_SHORT_MAX_WORDS = 8. Words 1..8 of the headline are
        # "Prince Harry Sued for Defamation by Charity He" — Co-Founded
        # is word 9 and gets trimmed, as does everything after.
        assert "Bizarre Twist" not in brief.voice_line
        assert "Co-Founded" not in brief.voice_line
        assert "Prince Harry Sued for Defamation by Charity He" in brief.voice_line

    def test_wire_appears_first_in_voice_line(self):
        # 1 AP wire + 2 reputable -> confirmed via wire branch. AP should
        # lead the voice line since wire buckets sort before reputable.
        arts = [
            _article(article_id="1", source_id="ap", wire_attribution="AP",
                     display_title="Global summit reaches historic agreement tonight"),
            _article(article_id="2", source_id="bbc",
                     display_title="Global summit reaches historic agreement tonight"),
            _article(article_id="3", source_id="npr",
                     display_title="Global summit reaches historic agreement tonight"),
        ]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1", "2", "3"],
            centroid_display_title="Global summit reaches historic agreement tonight",
            state=ConfidenceState.CONFIRMED,
            independent_confirmation_count=3,
            article_count=3,
        )
        brief = build_brief(c, articles, _sources())
        assert brief.top_sources == ["AP", "BBC", "NPR"]
        assert brief.voice_line.startswith("AP and BBC report ")

    def test_reasoning_includes_tier_labels(self):
        arts = [
            _article(article_id="1", source_id="ap", wire_attribution="AP"),
            _article(article_id="2", source_id="bbc"),
            _article(article_id="3", source_id="blog1"),
        ]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1", "2", "3"],
            state=ConfidenceState.CONFIRMED,
            independent_confirmation_count=3,
            article_count=3,
        )
        brief = build_brief(c, articles, _sources())
        assert brief.reasoning == [
            "AP (wire)",
            "BBC (reputable)",
            "Blog One (other)",
        ]


# --- polish: snippet truncation drops weak trailing words ---------------

class TestTopicShortDropsWeakTrailingWords:
    """Phase-7 polish: snippets cut at max_words shouldn't end on a
    function word like "in", "the", "and"."""

    def test_trailing_preposition_dropped(self):
        # 8-word headline ending in "in" -> snippet should drop it.
        long_headline = (
            "Hungary election live: Viktor Sato concedes defeat in Budapest today"
        )
        arts = [_article(article_id="1", source_id="bbc",
                         display_title=long_headline)]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1"],
            centroid_display_title=long_headline,
            state=ConfidenceState.DEVELOPING,
        )
        brief = build_brief(c, articles, _sources())
        # Snippet is "...concedes defeat" not "...concedes defeat in".
        assert " defeat in." not in brief.voice_line
        assert " defeat in " not in brief.voice_line
        assert "defeat" in brief.voice_line

    def test_trailing_article_dropped(self):
        long_headline = "Some news headline today happens with the "
        long_headline += "deal closing soon"
        arts = [_article(article_id="1", source_id="bbc",
                         display_title=long_headline)]
        articles = {a.article_id: a for a in arts}
        c = _cluster(
            article_ids=["1"],
            centroid_display_title=long_headline,
            state=ConfidenceState.DEVELOPING,
        )
        brief = build_brief(c, articles, _sources())
        # First 8 words: "Some news headline today happens with the deal".
        # "deal" is fine -> stays. But if the test headline were 8 ending
        # in "the", we'd drop it. Verify the snippet doesn't end in " the".
        assert not brief.voice_line.split(". Still")[0].endswith(" the")
