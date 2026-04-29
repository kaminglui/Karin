"""User preferences + ranking boost for bridge/news.

Loads a `preferences.json` file curated by the user, exposes a
`compute_score(cluster, articles, preferences)` pure function used by the
service to rank clusters within a confidence tier.

Design rules (per Phase 4.5 spec):

  - Preferences are ADDITIVE BOOSTS, never filters. They cannot cause a
    cluster to be dropped from results — only reordered.

  - State priority is a hard categorical tier over preference score:
    CONFIRMED > PROVISIONALLY_CONFIRMED > DEVELOPING. The sort key in
    service.py is (state_priority, score, latest_update_at). Preferences
    reorder within a tier; they cannot promote across tiers.

  - Region preferences are TEXT-BASED, not outlet-based. BBC, NYT, DW
    etc. cover many regions outside their editorial home, so matching
    on outlet region would bias wrongly. We match against the story
    text (centroid title + member summaries) instead.

  - Score math: for each enabled watchlist item whose keywords match
    the haystack, add (base_boost_for_type * priority_multiplier) once,
    regardless of how many keywords in that item matched. Then add a
    state bonus (confirmed=2.0, provisional=1.0, developing=0) and
    subtract the stale penalty if is_stale. All magnitudes come from
    RankingConfig, which is loaded from the config file with defaults
    filled in for any missing keys.

  - Missing preferences file OR `enabled: false` at top level OR a
    `Preferences(enabled=False)` instance all collapse compute_score to
    0.0 across the board — ranking behavior reverts to Phase 4
    identically (state then recency).

  - Deterministic, pure, no side effects. No NLP, no embeddings, no LLM.
    Pure substring matching against a lowercased haystack.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bridge.news.detect import SUPPORTED_LANGUAGES
from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    StoryCluster,
)

# Day-one scope: EN + ZH. SUPPORTED_LANGUAGES covers {en, zh, ja, ko} so
# the backend will accept ja/ko if the UI ever exposes them — no schema
# change needed, just a picker update. Anything outside this set coerces
# to "en" with a warning, matching the "loud config bugs" pattern used
# elsewhere in this file.
_DEFAULT_TARGET_LANGUAGE = "en"

log = logging.getLogger("bridge.news.preferences")

# Keywords this short or shorter use whole-word matching (\bkw\b) instead
# of plain substring. Stops "ai" from matching "pakistan", "claim", "said";
# "us" from matching "discuss"; "hk" from matching "checkpoint", etc.
# Longer keywords still use substring so multi-word phrases like
# "artificial intelligence" continue to work without escaping.
_SHORT_KEYWORD_THRESHOLD = 3


# --- dataclasses ----------------------------------------------------------

@dataclass(frozen=True)
class WatchlistItem:
    """One named watchlist entry. Keywords are lowercased at load time."""

    id: str
    label: str
    keywords: tuple[str, ...]
    priority: str = "medium"          # "high" | "medium" | "low"
    enabled: bool = True


@dataclass(frozen=True)
class RankingConfig:
    """Numeric knobs for the score. Defaults match the Phase 4.5 spec."""

    region_match_boost: float = 2.0
    topic_match_boost: float = 1.5
    event_match_boost: float = 2.5
    high_priority_multiplier: float = 1.5
    medium_priority_multiplier: float = 1.2
    low_priority_multiplier: float = 1.0
    provisionally_confirmed_bonus: float = 1.0
    confirmed_bonus: float = 2.0
    stale_penalty: float = 1.0


@dataclass(frozen=True)
class Preferences:
    """Loaded user preferences. `enabled=False` short-circuits scoring.

    `target_language` is independent of `enabled` — translation is a
    display concern, not a ranking one. Even with preferences disabled,
    the UI can still request Chinese headlines.
    """

    enabled: bool = False
    regions: tuple[WatchlistItem, ...] = ()
    topics: tuple[WatchlistItem, ...] = ()
    events: tuple[WatchlistItem, ...] = ()
    ranking: RankingConfig = field(default_factory=RankingConfig)
    target_language: str = _DEFAULT_TARGET_LANGUAGE


# --- loading --------------------------------------------------------------

_VALID_PRIORITIES = frozenset({"high", "medium", "low"})


def _parse_items(raw_items: Any) -> tuple[WatchlistItem, ...]:
    """Parse a list of watchlist-item dicts into WatchlistItem tuples.

    Unknown priority values coerce to "medium" (safe default). Missing
    `enabled` defaults to True, missing `label` defaults to `id`.
    Keywords are lowercased eagerly so the matching hot path avoids
    repeating the work.
    """
    if not isinstance(raw_items, list):
        return ()
    parsed: list[WatchlistItem] = []
    for it in raw_items:
        if not isinstance(it, dict) or "id" not in it or "keywords" not in it:
            log.warning("skipping malformed watchlist item: %r", it)
            continue
        priority = str(it.get("priority", "medium")).lower()
        if priority not in _VALID_PRIORITIES:
            log.warning(
                "watchlist item %r has unknown priority %r; defaulting to medium",
                it["id"], priority,
            )
            priority = "medium"
        parsed.append(WatchlistItem(
            id=str(it["id"]),
            label=str(it.get("label", it["id"])),
            # NFKC normalize so full-width / compatibility chars in
            # keywords match the same way they appear in normalized
            # haystacks. Lowercase for case-insensitive matching.
            keywords=tuple(
                unicodedata.normalize("NFKC", str(k)).lower()
                for k in it["keywords"]
            ),
            priority=priority,
            enabled=bool(it.get("enabled", True)),
        ))
    return tuple(parsed)


def _parse_ranking(raw: Any) -> RankingConfig:
    """Build a RankingConfig with defaults filled in for missing keys."""
    defaults = RankingConfig()
    if not isinstance(raw, dict):
        return defaults
    return RankingConfig(
        region_match_boost=float(raw.get("region_match_boost", defaults.region_match_boost)),
        topic_match_boost=float(raw.get("topic_match_boost", defaults.topic_match_boost)),
        event_match_boost=float(raw.get("event_match_boost", defaults.event_match_boost)),
        high_priority_multiplier=float(raw.get("high_priority_multiplier", defaults.high_priority_multiplier)),
        medium_priority_multiplier=float(raw.get("medium_priority_multiplier", defaults.medium_priority_multiplier)),
        low_priority_multiplier=float(raw.get("low_priority_multiplier", defaults.low_priority_multiplier)),
        provisionally_confirmed_bonus=float(raw.get("provisionally_confirmed_bonus", defaults.provisionally_confirmed_bonus)),
        confirmed_bonus=float(raw.get("confirmed_bonus", defaults.confirmed_bonus)),
        stale_penalty=float(raw.get("stale_penalty", defaults.stale_penalty)),
    )


def load_preferences(path: Path) -> Preferences:
    """Load preferences from a JSON file.

    Missing file -> Preferences(enabled=False). This is the "user hasn't
    set anything up" path; service behavior should match Phase 4 exactly.

    Malformed JSON -> json.JSONDecodeError propagates. Config bugs should
    be loud, not silent — a typo shouldn't silently disable all the
    user's watchlists.
    """
    if not path.exists():
        log.info("no preferences.json at %s; running with preferences disabled", path)
        return Preferences(enabled=False)

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"preferences.json must be a JSON object, got {type(data).__name__}")

    enabled = bool(data.get("enabled", True))
    watchlists = data.get("watchlists", {}) or {}

    regions = _parse_items(watchlists.get("regions", []))
    topics = _parse_items(watchlists.get("topics", []))
    events = _parse_items(watchlists.get("events", []))
    ranking = _parse_ranking(data.get("ranking", {}))
    target_language = _parse_target_language(data.get("target_language"))

    log.info(
        "loaded preferences: enabled=%s regions=%d topics=%d events=%d target_language=%s",
        enabled, len(regions), len(topics), len(events), target_language,
    )
    return Preferences(
        enabled=enabled,
        regions=regions,
        topics=topics,
        events=events,
        ranking=ranking,
        target_language=target_language,
    )


def _parse_target_language(raw: Any) -> str:
    """Validate a target_language value from prefs JSON.

    Missing / None -> default ("en"). Unsupported -> warn + default.
    Writing this tolerantly so a typo in preferences.json doesn't
    explode the whole news path; the ranking side already follows
    this pattern. The UI Settings panel POSTs the full prefs blob
    through /api/preferences/news, so this is the single validation
    point for both config-file and UI-written values.
    """
    if raw is None:
        return _DEFAULT_TARGET_LANGUAGE
    value = str(raw).lower().strip()
    if value not in SUPPORTED_LANGUAGES:
        log.warning(
            "preferences.target_language %r not in %s; defaulting to %s",
            value, sorted(SUPPORTED_LANGUAGES), _DEFAULT_TARGET_LANGUAGE,
        )
        return _DEFAULT_TARGET_LANGUAGE
    return value


# --- scoring --------------------------------------------------------------

# Priority string -> RankingConfig field name that holds its multiplier.
_PRIORITY_FIELD: dict[str, str] = {
    "high": "high_priority_multiplier",
    "medium": "medium_priority_multiplier",
    "low": "low_priority_multiplier",
}


def _build_haystack(
    cluster: StoryCluster,
    all_articles: dict[str, NormalizedArticle],
) -> str:
    """Lowercase concat of centroid title + all member summaries.

    `centroid_normalized_title` and `NormalizedArticle.summary` are
    already lowercased by normalize.py. The explicit .lower() at the
    end is defensive against callers who hand-construct clusters in
    tests; it's effectively a no-op on ledger-loaded data.

    Missing article_ids (e.g. after a ledger prune) are skipped silently.
    Empty summaries contribute nothing to the haystack but don't break
    matching — concatenation with empty strings is safe.
    """
    parts: list[str] = [cluster.centroid_normalized_title]
    for aid in cluster.article_ids:
        a = all_articles.get(aid)
        if a is None or not a.summary:
            continue
        parts.append(a.summary)
    return " ".join(parts).lower()


def _item_matches(item: WatchlistItem, haystack: str) -> bool:
    """True if ANY keyword in the item matches the haystack.

    Short keywords (len <= _SHORT_KEYWORD_THRESHOLD) require whole-word
    matching via \\b{kw}\\b regex. Longer keywords use plain substring
    matching, which naturally handles multi-word phrases like
    "artificial intelligence" without escaping.

    Keywords are pre-lowercased and NFKC-normalized at load time;
    haystacks come in already lowercased from clean_display_title +
    normalize_summary.
    """
    for kw in item.keywords:
        if not kw:
            continue
        if len(kw) <= _SHORT_KEYWORD_THRESHOLD:
            # Word-boundary match. re caches compiled patterns internally,
            # so re-calls with the same kw don't re-compile.
            if re.search(rf"\b{re.escape(kw)}\b", haystack):
                return True
        elif kw in haystack:
            return True
    return False


def _priority_multiplier(item: WatchlistItem, ranking: RankingConfig) -> float:
    """Resolve an item's priority to its numeric multiplier."""
    field_name = _PRIORITY_FIELD.get(item.priority, "medium_priority_multiplier")
    return getattr(ranking, field_name)


@dataclass(frozen=True)
class WatchlistMatch:
    """One watchlist item that a cluster matched, plus which section it's from.

    Additive helper type introduced in Phase 6 so the alerts subsystem can
    know WHICH watchlists a cluster hit (not just that its score was
    positive). Does not affect compute_score() or any existing ranking
    behavior — computed independently from the same haystack.
    """

    watchlist_type: str     # "region" | "topic" | "event"
    item_id: str
    item_label: str
    priority: str


def match_watchlist_items(
    cluster: StoryCluster,
    all_articles: dict[str, NormalizedArticle],
    preferences: Preferences,
) -> list[WatchlistMatch]:
    """Return the watchlist items this cluster's haystack matches.

    Phase 6 helper. Reuses the existing haystack construction and
    keyword-matching logic from compute_score() so a cluster that scores
    positive and a cluster that matches will never disagree on which
    items were hits.

    Returns an empty list when preferences are disabled. Per-item
    `enabled=False` items are skipped, same as in compute_score.
    """
    if not preferences.enabled:
        return []
    haystack = _build_haystack(cluster, all_articles)
    matches: list[WatchlistMatch] = []
    for item in preferences.regions:
        if item.enabled and _item_matches(item, haystack):
            matches.append(WatchlistMatch(
                watchlist_type="region", item_id=item.id,
                item_label=item.label, priority=item.priority,
            ))
    for item in preferences.topics:
        if item.enabled and _item_matches(item, haystack):
            matches.append(WatchlistMatch(
                watchlist_type="topic", item_id=item.id,
                item_label=item.label, priority=item.priority,
            ))
    for item in preferences.events:
        if item.enabled and _item_matches(item, haystack):
            matches.append(WatchlistMatch(
                watchlist_type="event", item_id=item.id,
                item_label=item.label, priority=item.priority,
            ))
    return matches


def compute_score(
    cluster: StoryCluster,
    all_articles: dict[str, NormalizedArticle],
    preferences: Preferences,
) -> float:
    """Score a cluster against preferences. Pure; no side effects.

    Returns 0.0 when preferences.enabled is False — this makes the sort
    key in service.py collapse to (state_priority, 0, latest_update_at),
    which is order-equivalent to the Phase 4 two-tuple.

    Each watchlist item contributes AT MOST ONCE to the score, regardless
    of how many of its keywords hit the haystack (spec-delta item 5).
    State bonus and stale penalty are applied once per call.
    """
    if not preferences.enabled:
        return 0.0

    haystack = _build_haystack(cluster, all_articles)
    ranking = preferences.ranking
    score = 0.0

    for item in preferences.regions:
        if not item.enabled:
            continue
        if _item_matches(item, haystack):
            score += ranking.region_match_boost * _priority_multiplier(item, ranking)

    for item in preferences.topics:
        if not item.enabled:
            continue
        if _item_matches(item, haystack):
            score += ranking.topic_match_boost * _priority_multiplier(item, ranking)

    for item in preferences.events:
        if not item.enabled:
            continue
        if _item_matches(item, haystack):
            score += ranking.event_match_boost * _priority_multiplier(item, ranking)

    # State bonuses. Applied to the secondary sort key — they're visible
    # in compute_score output and testable, but cannot cross the
    # categorical state tier in service-level ranking (by design).
    if cluster.state == ConfidenceState.CONFIRMED:
        score += ranking.confirmed_bonus
    elif cluster.state == ConfidenceState.PROVISIONALLY_CONFIRMED:
        score += ranking.provisionally_confirmed_bonus

    if cluster.is_stale:
        score -= ranking.stale_penalty

    return score
