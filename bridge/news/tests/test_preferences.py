"""Tests for bridge.news.preferences.

Covers:
  - JSON loading: valid file, missing file (disabled default), malformed JSON
  - compute_score matching: regions, topics, events
  - priority multipliers (high > medium > low for same base match)
  - stale penalty (subtracts)
  - state bonuses (confirmed > provisional > developing, observable in score)
  - no-double-boost from multiple keywords in one watchlist item
  - integration: explicit topic= filters before preference reranking
  - integration: no preferences file -> identical order to Phase 4
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from bridge.news.ledger import Ledger
from bridge.news.models import (
    ConfidenceState,
    Feed,
    NormalizedArticle,
    RawArticle,
    Source,
    StoryCluster,
    Tier,
)
from bridge.news.preferences import (
    Preferences,
    RankingConfig,
    WatchlistItem,
    compute_score,
    load_preferences,
)
from bridge.news.service import NewsService


# --- fixtures --------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _t(hours: float) -> datetime:
    return _EPOCH + timedelta(hours=hours)


def _article(
    *,
    article_id: str,
    source_id: str = "bbc",
    display_title: str = "Some generic placeholder story headline today now",
    summary: str = "",
    wire_attribution: str | None = None,
) -> NormalizedArticle:
    return NormalizedArticle(
        article_id=article_id,
        source_id=source_id,
        url=f"https://x.test/{article_id}",
        display_title=display_title,
        normalized_title=display_title.lower(),
        summary=summary.lower(),
        fingerprint=f"fp_{article_id}",
        wire_attribution=wire_attribution,
        published_at=_t(0),
        fetched_at=_t(0),
    )


def _cluster(
    *,
    article_ids: list[str],
    centroid_display_title: str = "Some generic placeholder story headline today now",
    state: ConfidenceState = ConfidenceState.DEVELOPING,
    is_stale: bool = False,
) -> StoryCluster:
    return StoryCluster(
        cluster_id="c1",
        article_ids=list(article_ids),
        centroid_display_title=centroid_display_title,
        centroid_normalized_title=centroid_display_title.lower(),
        first_seen_at=_t(0),
        latest_update_at=_t(0),
        last_checked_at=_t(0),
        last_state_change_at=_t(0),
        state=state,
        is_stale=is_stale,
        independent_confirmation_count=1,
        article_count=len(article_ids),
        syndicated_article_count=0,
    )


def _prefs_with(**kwargs) -> Preferences:
    """Minimal Preferences with all sections empty unless overridden."""
    base = dict(enabled=True, regions=(), topics=(), events=())
    base.update(kwargs)
    return Preferences(**base)


# --- loading ---------------------------------------------------------------

class TestLoading:
    def test_valid_json_loads(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({
            "enabled": True,
            "watchlists": {
                "regions": [{
                    "id": "us", "label": "US",
                    "keywords": ["united states", "washington"],
                    "priority": "high", "enabled": True,
                }],
                "topics": [],
                "events": [{
                    "id": "ukraine", "label": "Ukraine",
                    "keywords": ["ukraine", "kyiv"],
                    "priority": "medium", "enabled": True,
                }],
            },
            "ranking": {
                "region_match_boost": 3.0,
                "event_match_boost": 4.0,
            },
        }))
        prefs = load_preferences(path)
        assert prefs.enabled is True
        assert len(prefs.regions) == 1
        assert prefs.regions[0].id == "us"
        assert prefs.regions[0].keywords == ("united states", "washington")
        assert prefs.regions[0].priority == "high"
        assert len(prefs.events) == 1
        # Custom ranking overrides survived.
        assert prefs.ranking.region_match_boost == 3.0
        assert prefs.ranking.event_match_boost == 4.0
        # Un-overridden ranking fields took the defaults.
        assert prefs.ranking.topic_match_boost == 1.5

    def test_missing_file_returns_disabled(self, tmp_path):
        prefs = load_preferences(tmp_path / "does_not_exist.json")
        assert prefs.enabled is False
        assert prefs.regions == ()
        assert prefs.topics == ()
        assert prefs.events == ()

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_preferences(path)

    def test_keywords_lowercased_at_load(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({
            "watchlists": {
                "topics": [{
                    "id": "ai", "label": "AI",
                    "keywords": ["OpenAI", "ANTHROPIC", "Artificial Intelligence"],
                }],
            },
        }))
        prefs = load_preferences(path)
        assert prefs.topics[0].keywords == ("openai", "anthropic", "artificial intelligence")

    def test_missing_priority_defaults_to_medium(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({
            "watchlists": {"topics": [{"id": "x", "keywords": ["x"]}]},
        }))
        prefs = load_preferences(path)
        assert prefs.topics[0].priority == "medium"

    def test_target_language_missing_defaults_to_en(self, tmp_path):
        # No target_language key + no file both collapse to "en".
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({"watchlists": {}}))
        assert load_preferences(path).target_language == "en"
        assert load_preferences(tmp_path / "nope.json").target_language == "en"

    def test_target_language_supported_values_round_trip(self, tmp_path):
        for lang in ("en", "zh", "ja", "ko"):
            path = tmp_path / f"prefs_{lang}.json"
            path.write_text(json.dumps({"target_language": lang}))
            assert load_preferences(path).target_language == lang

    def test_target_language_uppercase_coerced(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({"target_language": "ZH"}))
        assert load_preferences(path).target_language == "zh"

    def test_target_language_unsupported_defaults_to_en(self, tmp_path, caplog):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({"target_language": "klingon"}))
        with caplog.at_level("WARNING"):
            prefs = load_preferences(path)
        assert prefs.target_language == "en"
        # Loud on typo — don't silently swallow.
        assert any("klingon" in r.getMessage() for r in caplog.records)


# --- compute_score: matching ----------------------------------------------

class TestMatching:
    def test_region_match_increments_score(self):
        prefs = _prefs_with(regions=(
            WatchlistItem(id="us", label="US",
                          keywords=("washington",), priority="medium"),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Washington lawmakers pass new bill today")
        articles = {"1": _article(article_id="1", summary="")}
        score = compute_score(c, articles, prefs)
        # medium mult (1.2) * region boost (2.0) = 2.4
        assert score == pytest.approx(2.4)

    def test_topic_match_increments_score(self):
        prefs = _prefs_with(topics=(
            WatchlistItem(id="ai", label="AI",
                          keywords=("artificial intelligence",), priority="medium"),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="New artificial intelligence breakthrough announced")
        articles = {"1": _article(article_id="1")}
        score = compute_score(c, articles, prefs)
        # medium mult (1.2) * topic boost (1.5) = 1.8
        assert score == pytest.approx(1.8)

    def test_event_match_increments_score(self):
        prefs = _prefs_with(events=(
            WatchlistItem(id="ukraine", label="Ukraine",
                          keywords=("kyiv",), priority="medium"),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Breaking news from Kyiv this morning")
        articles = {"1": _article(article_id="1")}
        score = compute_score(c, articles, prefs)
        # medium mult (1.2) * event boost (2.5) = 3.0
        assert score == pytest.approx(3.0)

    def test_match_can_come_from_member_summary_not_title(self):
        prefs = _prefs_with(events=(
            WatchlistItem(id="ukraine", label="Ukraine",
                          keywords=("zelensky",), priority="medium"),
        ))
        # Centroid title says nothing about zelensky; but a member summary
        # does. Should still match (spec: haystack includes summaries).
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Diplomatic meeting in Brussels concludes")
        articles = {"1": _article(
            article_id="1",
            display_title="Diplomatic meeting in Brussels concludes",
            summary="President Zelensky spoke at the session about ongoing war.",
        )}
        score = compute_score(c, articles, prefs)
        assert score > 0

    def test_empty_summaries_are_safely_ignored(self):
        prefs = _prefs_with(topics=(
            WatchlistItem(id="ai", label="AI", keywords=("ai",),
                          priority="medium"),
        ))
        c = _cluster(article_ids=["1", "2"],
                     centroid_display_title="An AI-driven breakthrough story unveiled")
        articles = {
            "1": _article(article_id="1", summary=""),
            "2": _article(article_id="2", summary=""),
        }
        # Should not raise; should still find the match in the centroid title.
        score = compute_score(c, articles, prefs)
        assert score > 0


# --- priority + stale + state ---------------------------------------------

class TestScoreArithmetic:
    def test_higher_priority_beats_lower_same_base_match(self):
        kw = ("chip",)
        high = _prefs_with(topics=(
            WatchlistItem(id="sc", label="SC", keywords=kw, priority="high"),
        ))
        medium = _prefs_with(topics=(
            WatchlistItem(id="sc", label="SC", keywords=kw, priority="medium"),
        ))
        low = _prefs_with(topics=(
            WatchlistItem(id="sc", label="SC", keywords=kw, priority="low"),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Major chip manufacturer announces new fab")
        articles = {"1": _article(article_id="1")}
        score_high = compute_score(c, articles, high)
        score_medium = compute_score(c, articles, medium)
        score_low = compute_score(c, articles, low)
        assert score_high > score_medium > score_low > 0

    def test_stale_penalty_subtracts(self):
        prefs = _prefs_with(topics=(
            WatchlistItem(id="ai", label="AI", keywords=("ai",),
                          priority="medium"),
        ))
        fresh = _cluster(article_ids=["1"],
                         centroid_display_title="An AI model was released today",
                         is_stale=False)
        stale = _cluster(article_ids=["1"],
                         centroid_display_title="An AI model was released today",
                         is_stale=True)
        articles = {"1": _article(article_id="1")}
        # Difference should be exactly stale_penalty (1.0).
        assert compute_score(fresh, articles, prefs) - compute_score(stale, articles, prefs) \
            == pytest.approx(prefs.ranking.stale_penalty)

    def test_confirmed_bonus_greater_than_provisional_greater_than_developing(self):
        # No watchlist matches at all; score differences come purely from
        # the state bonuses and are observable via compute_score even
        # though the service-level sort key keeps state categorical.
        prefs = _prefs_with()  # all watchlists empty, enabled=True
        c_confirmed = _cluster(
            article_ids=["1"],
            centroid_display_title="Some unrelated placeholder title here today",
            state=ConfidenceState.CONFIRMED,
        )
        c_provisional = _cluster(
            article_ids=["1"],
            centroid_display_title="Some unrelated placeholder title here today",
            state=ConfidenceState.PROVISIONALLY_CONFIRMED,
        )
        c_developing = _cluster(
            article_ids=["1"],
            centroid_display_title="Some unrelated placeholder title here today",
            state=ConfidenceState.DEVELOPING,
        )
        articles = {"1": _article(article_id="1")}
        s_c = compute_score(c_confirmed, articles, prefs)
        s_p = compute_score(c_provisional, articles, prefs)
        s_d = compute_score(c_developing, articles, prefs)
        assert s_c > s_p > s_d
        # Pin the exact gap so future ranking-config changes are visible.
        assert s_c == pytest.approx(prefs.ranking.confirmed_bonus)
        assert s_p == pytest.approx(prefs.ranking.provisionally_confirmed_bonus)
        assert s_d == 0.0


class TestNoDuplicateBoost:
    def test_multiple_keywords_in_one_item_score_once(self):
        # A single watchlist item with several keywords. The haystack
        # contains multiple of its keywords — the item must still
        # contribute exactly one match's worth of score.
        prefs = _prefs_with(events=(
            WatchlistItem(
                id="ukraine", label="Ukraine",
                keywords=("ukraine", "kyiv", "zelensky"),  # three keywords
                priority="medium",
            ),
        ))
        # Title mentions two of the three keywords.
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Zelensky visits Kyiv after negotiations")
        articles = {"1": _article(article_id="1")}
        score = compute_score(c, articles, prefs)
        # Should equal ONE event-match at medium priority, not two or three.
        expected = prefs.ranking.event_match_boost * prefs.ranking.medium_priority_multiplier
        assert score == pytest.approx(expected)


class TestDisabled:
    def test_disabled_preferences_always_score_zero(self):
        prefs = Preferences(enabled=False, events=(
            WatchlistItem(id="x", label="x", keywords=("ukraine",)),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Ukraine ceasefire talks stall again today")
        articles = {"1": _article(article_id="1")}
        assert compute_score(c, articles, prefs) == 0.0

    def test_disabled_item_within_enabled_prefs_is_skipped(self):
        prefs = _prefs_with(events=(
            WatchlistItem(id="x", label="x", keywords=("ukraine",),
                          priority="medium", enabled=False),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Ukraine ceasefire talks stall again today")
        articles = {"1": _article(article_id="1")}
        assert compute_score(c, articles, prefs) == 0.0


# --- service integration ---------------------------------------------------

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


class TestServiceIntegration:
    def test_explicit_topic_filter_runs_before_preference_rerank(self, tmp_path):
        # Preferences say "ukraine" is a high-priority event watch.
        # But user asks for topic="osaka". Only osaka cluster should
        # come back — preferences must not override the explicit filter.
        prefs = _prefs_with(events=(
            WatchlistItem(id="ukraine", label="Ukraine",
                          keywords=("ukraine",), priority="high"),
        ))
        service = NewsService(
            ledger=Ledger(tmp_path),
            sources=_sources(),
            feeds=[Feed(source_id="bbc", url="https://fake.test/bbc")],
            preferences=prefs,
            fetch_ttl_minutes=15,
        )
        raw = [
            _raw(title="Ukraine ceasefire talks continue through the night",
                 source_id="bbc", url="https://x.test/uk", hours_ago=3),
            _raw(title="Osaka earthquake causes widespread damage tonight",
                 source_id="bbc", url="https://x.test/os", hours_ago=2),
        ]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            briefs = service.get_news(topic="osaka", max_results=5)
        assert len(briefs) == 1
        assert "Osaka" in briefs[0].headline, (
            "preferences must not bypass an explicit topic filter"
        )

    def test_preferences_rerank_within_same_state_tier(self, tmp_path):
        # Two developing clusters, same recency (close enough). One matches
        # an event watch; preferences should lift it above the non-matching one.
        prefs = _prefs_with(events=(
            WatchlistItem(id="ukraine", label="Ukraine",
                          keywords=("ukraine",), priority="high"),
        ))
        service = NewsService(
            ledger=Ledger(tmp_path),
            sources=_sources(),
            feeds=[Feed(source_id="bbc", url="https://fake.test/bbc")],
            preferences=prefs,
            fetch_ttl_minutes=15,
        )
        raw = [
            # Make the non-matching cluster NEWER so recency alone would
            # put it first. Preference boost must flip that ordering.
            _raw(title="Some unrelated miscellaneous developing story today",
                 source_id="bbc", url="https://x.test/a", hours_ago=1),
            _raw(title="Ukraine ceasefire talks resume in new round today",
                 source_id="bbc", url="https://x.test/b", hours_ago=3),
        ]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            briefs = service.get_news(max_results=2)
        assert len(briefs) == 2
        assert "Ukraine" in briefs[0].headline, (
            "preference-matching developing cluster should outrank "
            "newer non-matching developing cluster"
        )

    def test_state_primacy_preference_cannot_promote_across_tier(self, tmp_path):
        # A CONFIRMED cluster with no preference match should still
        # outrank a DEVELOPING cluster with a max-boost preference match.
        prefs = _prefs_with(events=(
            WatchlistItem(id="ukraine", label="Ukraine",
                          keywords=("ukraine",), priority="high"),
        ))
        service = NewsService(
            ledger=Ledger(tmp_path),
            sources=_sources(),
            feeds=[Feed(source_id="bbc", url="https://fake.test/bbc")],
            preferences=prefs,
            fetch_ttl_minutes=15,
        )
        raw = [
            # 3 distinct reputable outlets on one story -> CONFIRMED.
            # Content deliberately unrelated to any preference.
            _raw(title="Some miscellaneous topic reaches completion today soon",
                 source_id="bbc",      url="https://x.test/c1", hours_ago=5),
            _raw(title="Some miscellaneous topic reaches completion today soon",
                 source_id="npr",      url="https://x.test/c2", hours_ago=5),
            _raw(title="Some miscellaneous topic reaches completion today soon",
                 source_id="guardian", url="https://x.test/c3", hours_ago=5),
            # Solo developing cluster that matches the watchlist.
            _raw(title="Ukraine ceasefire talks resume in new round today",
                 source_id="bbc", url="https://x.test/d1", hours_ago=1),
        ]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            briefs = service.get_news(max_results=5)
        assert briefs[0].state == ConfidenceState.CONFIRMED, (
            "state primacy: confirmed no-match must outrank developing max-match"
        )
        assert briefs[1].state == ConfidenceState.DEVELOPING

    def test_no_preferences_ordering_identical_to_phase4(self, tmp_path):
        # With no preferences, ordering must be exactly state+recency.
        # Two DEVELOPING clusters: newer one wins, older one loses.
        service_no_prefs = NewsService(
            ledger=Ledger(tmp_path),
            sources=_sources(),
            feeds=[Feed(source_id="bbc", url="https://fake.test/bbc")],
            preferences=None,  # disabled default
            fetch_ttl_minutes=15,
        )
        raw = [
            _raw(title="Ukraine developing story about something today happening",
                 source_id="bbc", url="https://x.test/u", hours_ago=5),
            _raw(title="Osaka developing story about something else today happening",
                 source_id="bbc", url="https://x.test/o", hours_ago=1),
        ]
        with patch("bridge.news.service.fetch_all", return_value=(raw, 1, 0)):
            briefs = service_no_prefs.get_news(max_results=2)
        # Recency breaks the tie: Osaka (1h ago) ahead of Ukraine (5h ago).
        assert "Osaka" in briefs[0].headline, (
            "no preferences should mean pure state+recency order"
        )


# --- polish: short-keyword whole-word matching ---------------------------

class TestShortKeywordWholeWord:
    """Phase-7 polish: keywords <= 3 chars require word-boundary matches
    so noisy short tokens like "ai" don't match inside "pakistan", "said",
    "claim", etc."""

    def test_short_keyword_does_not_match_inside_word(self):
        # "ai" must NOT match "pakistan" (contains a-i substring).
        prefs = _prefs_with(topics=(
            WatchlistItem(id="ai", label="AI", keywords=("ai",),
                          priority="medium"),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Pakistan signs trade deal today")
        articles = {"1": _article(article_id="1")}
        assert compute_score(c, articles, prefs) == 0.0

    def test_short_keyword_matches_when_standalone(self):
        # "ai" SHOULD match a standalone occurrence.
        prefs = _prefs_with(topics=(
            WatchlistItem(id="ai", label="AI", keywords=("ai",),
                          priority="medium"),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="New ai system released today now")
        articles = {"1": _article(article_id="1")}
        assert compute_score(c, articles, prefs) > 0

    def test_long_keyword_substring_still_works(self):
        # Multi-word "artificial intelligence" still matches substring-style.
        prefs = _prefs_with(topics=(
            WatchlistItem(id="ai", label="AI",
                          keywords=("artificial intelligence",),
                          priority="medium"),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="The artificial intelligence boom continues today")
        articles = {"1": _article(article_id="1")}
        assert compute_score(c, articles, prefs) > 0

    def test_us_alias_does_not_match_inside_word(self):
        # "us" (2 chars) must NOT match "discuss" or "industry".
        prefs = _prefs_with(regions=(
            WatchlistItem(id="us", label="United States",
                          keywords=("us",), priority="high"),
        ))
        c = _cluster(article_ids=["1"],
                     centroid_display_title="Industry leaders discuss policy today now")
        articles = {"1": _article(article_id="1")}
        assert compute_score(c, articles, prefs) == 0.0
