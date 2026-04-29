"""Typed data models for bridge/news.

Contracts between pipeline stages: fetch -> normalize -> cluster ->
confidence -> service. Everything persisted to the JSON ledger
serializes from these dataclasses.

All datetimes MUST be timezone-aware (UTC). The ledger enforces this
on load; the fetcher stamps UTC on fetched_at. Naive datetimes are a bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Tier(str, Enum):
    """Source reputation tier. Curated, slow-moving."""

    WIRE = "wire"             # AP, Reuters, AFP — origin of syndication
    REPUTABLE = "reputable"   # major outlets with editorial standards
    OTHER = "other"           # everything else we ingest but don't weight heavily


@dataclass(frozen=True)
class Source:
    """Curated outlet metadata. Loaded from sources.json at startup.

    Fields deferred to V2 (see feedback): editorial_lean, reputation_score,
    primary_reporting_strength.
    """

    id: str
    name: str
    domain: str
    tier: Tier
    ownership_group: str
    is_wire_service: bool
    notes: str = ""


@dataclass(frozen=True)
class Feed:
    """RSS feed URL bound to a source_id. Loaded from feeds.json."""

    source_id: str
    url: str


@dataclass
class RawArticle:
    """Pre-normalize form. Lives only in memory between fetch and normalize.
    Not persisted."""

    source_id: str
    url: str
    title: str
    summary: str
    published_at: datetime
    fetched_at: datetime


@dataclass
class NormalizedArticle:
    """Post-normalize, ledger-persistable form.

    `display_title` preserves case/punctuation for human-facing surfaces
    (briefs, voice_line). `normalized_title` is lowercased/stripped for
    lexical matching inside the clusterer. `display_summary` is the
    case-preserved short blurb that the brief/tool path exposes;
    `summary` stays lowercased because watchlist matching depends on it.
    `language` is the BCP-47-ish code ("en", "zh", "ja", "ko", or "und"
    when we can't tell) detected from the combined title + summary text.
    """

    article_id: str                  # sha1(url)[:16]
    source_id: str
    url: str
    display_title: str               # preserves case; drives voice + briefs
    normalized_title: str            # lowercased, stripped; drives clustering/matching
    summary: str                     # normalized (lowercased, whitespace-collapsed)
    fingerprint: str                 # 5-gram shingle hash for near-dup detection
    wire_attribution: str | None     # "AP" / "Reuters" / "AFP" / None
    published_at: datetime
    fetched_at: datetime
    display_summary: str = ""        # case-preserved, HTML-stripped; drives translated detail
    language: str = "en"             # detected script/language; "und" if ambiguous


class ConfidenceState(str, Enum):
    """Cluster-level story confidence. Deterministic; no LLM involvement.

    V1 implements only DEVELOPING / PROVISIONALLY_CONFIRMED / CONFIRMED.
    CONTESTED and RETRACTED are reserved — they require contradiction
    detection and correction parsing, deferred to V2.
    """

    DEVELOPING = "developing"
    PROVISIONALLY_CONFIRMED = "provisionally_confirmed"
    CONFIRMED = "confirmed"
    # Reserved for V2:
    # CONTESTED = "contested"
    # RETRACTED = "retracted"


@dataclass
class StoryCluster:
    """A group of articles about the same story.

    Time fields distinguish content freshness from ledger freshness:
      - latest_update_at: newest article's published_at (story recency)
      - last_checked_at: last time we re-scored this cluster (ledger recency)
    `is_stale` is tied to last_checked_at (see confidence.py).
    """

    cluster_id: str
    article_ids: list[str]
    centroid_display_title: str      # feeds headline + voice_line
    centroid_normalized_title: str   # used for topic matching
    first_seen_at: datetime
    latest_update_at: datetime       # content freshness
    last_checked_at: datetime        # ledger freshness — drives is_stale
    last_state_change_at: datetime
    state: ConfidenceState
    is_stale: bool
    independent_confirmation_count: int
    article_count: int               # total articles in cluster
    syndicated_article_count: int    # = article_count - independent_confirmation_count


@dataclass
class StoryBrief:
    """User-facing cluster summary.

    `voice_line` is the single short sentence the tool returns to Karin.
    The richer fields (reasoning, top_sources) exist for a future UI panel.
    """

    cluster_id: str
    headline: str                    # display-safe title
    state: ConfidenceState
    is_stale: bool
    independent_confirmation_count: int
    article_count: int
    syndicated_article_count: int
    top_sources: list[str]           # source display names, ordered by tier
    reasoning: list[str]             # bullets for a future UI panel
    voice_line: str                  # ONE short sentence for Karin
    first_seen_at: datetime | None = None   # when the cluster was first created
    latest_update_at: datetime | None = None # most recent article in the cluster
    detail: str = ""                 # 1-3 sentence blurb; extracted body or RSS summary, translated under the "fits" rule
    language: str = "en"             # language of headline + detail (post-translation); voice_line matches


@dataclass
class IngestResult:
    """Outcome of a single ingest_latest() call."""

    fetched_feeds: int
    failed_feeds: int
    new_articles: int
    new_article_ids: list[str]
    skipped_due_to_cache: bool
