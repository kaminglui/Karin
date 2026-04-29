"""Clustering: assign articles to story clusters via lexical similarity.

Pure and deterministic. Does not persist, does not fetch. Callers pass in
all articles + current cluster state + the source registry, and receive
an updated cluster dict plus the set of cluster_ids touched by this call.

Two articles merge into the same cluster if either:
  - fingerprint fast path: identical fingerprint (near-verbatim text —
    typically a wire story reposted by multiple outlets); OR
  - lexical path: token Jaccard over normalized_title (punctuation and
    stopwords removed) >= title_jaccard_threshold AND
    |published_at delta| <= time_bucket_hours.

Fingerprint match intentionally bypasses the time bucket — wire stories
with staggered republication days later should still merge as one story.

Independence counting follows spec-delta #4: articles with a non-null
wire_attribution collapse to one bucket per wire; non-wire articles
contribute one bucket per ownership_group.
"""
from __future__ import annotations

import hashlib
import string
from datetime import datetime, timedelta, timezone
from typing import Iterable

from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    Source,
    StoryCluster,
)

# --- tunable defaults ------------------------------------------------------
# Live values come from config/tuning.yaml when present (under
# `news.cluster.*`); the literals below are the fallback defaults.

from bridge import tuning as _tuning

DEFAULT_TITLE_JACCARD_THRESHOLD = _tuning.get(
    "news.cluster.title_jaccard", 0.55,
)
DEFAULT_TIME_BUCKET_HOURS = _tuning.get(
    "news.cluster.time_bucket_hours", 48,
)

# Minimum informative (non-stopword, non-single-char) tokens required on
# BOTH sides before a lexical match is even considered. Guards against
# false merges on stock headlines like "Here's the latest" (2 tokens
# after cleanup) or "Live updates" (0 tokens). Fingerprint fast path is
# unaffected — verbatim content still merges regardless of token count.
MIN_INFORMATIVE_TOKENS = _tuning.get(
    "news.cluster.min_informative_tokens", 3,
)

# Minimal English stopword list. Kept tight (~30 words) so we don't
# over-strip short headlines — "Tokyo Mayor Elected" would lose too much
# signal with a bigger list. No NLTK dep: 30 strings don't justify it.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "or", "but", "the", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "you",
})

# Map every ASCII punctuation character to a space, so tokenizing on
# whitespace splits "tokyo's" into {tokyo, s}. Stopwords catch the "s".
_PUNCT_TRANS = str.maketrans({c: " " for c in string.punctuation})


# --- tokenization + similarity --------------------------------------------

def _tokenize(normalized_title: str) -> set[str]:
    """Tokenize a normalized title for Jaccard similarity.

    Strips ASCII punctuation, splits on whitespace, drops stopwords,
    empty tokens, and single-character fragments (e.g. the "s" left
    behind when "tokyo's" is split on the apostrophe). Returns a set
    since Jaccard is set-based.
    """
    cleaned = normalized_title.translate(_PUNCT_TRANS)
    return {
        t for t in cleaned.split()
        if t and len(t) > 1 and t not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity on token sets. Empty either side -> 0.0."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- independence counting (exposed for tests + service) -------------------

def count_independent_confirmations(
    articles: Iterable[NormalizedArticle],
    sources: dict[str, Source],
) -> int:
    """Count distinct wire attributions + distinct ownership_groups.

    Rule (spec-delta #4):
      - Articles with a non-null wire_attribution collapse into one
        bucket per wire name (regardless of which outlet surfaced them).
      - Articles without wire_attribution contribute one bucket per
        distinct ownership_group.
      - Articles whose source_id is not in the registry are ignored
        (they contribute no bucket either way).
    """
    wires: set[str] = set()
    ownership_groups: set[str] = set()
    for a in articles:
        if a.wire_attribution is not None:
            wires.add(a.wire_attribution)
            continue
        src = sources.get(a.source_id)
        if src is not None:
            ownership_groups.add(src.ownership_group)
    return len(wires) + len(ownership_groups)


# --- cluster matching ------------------------------------------------------

def _match_cluster(
    article: NormalizedArticle,
    cluster: StoryCluster,
    all_articles: dict[str, NormalizedArticle],
    *,
    title_jaccard_threshold: float,
    time_bucket: timedelta,
) -> tuple[bool, float]:
    """Does `article` belong in `cluster`? Returns (match, best_score).

    The score is only meaningful when match is True; callers use it to
    pick the best candidate when multiple clusters match.

    Compares against every existing member of the cluster (not just the
    centroid) so a cluster that's drifted across several sub-variants
    of a headline can still attract a new article that matches any one
    of them.
    """
    article_tokens = _tokenize(article.normalized_title)
    # Low-signal headline guard: if this article's own title tokenizes
    # to too few informative tokens (e.g. "Here's the latest"), suppress
    # lexical matching entirely — it carries no topical signal, so any
    # match would be a coincidence. Fingerprint fast path still runs.
    article_lexically_matchable = len(article_tokens) >= MIN_INFORMATIVE_TOKENS

    best_jaccard = 0.0
    for member_id in cluster.article_ids:
        member = all_articles.get(member_id)
        if member is None:
            continue
        # Fingerprint fast path — bypasses the time bucket. Wire stories
        # republished days later should still merge.
        #
        # TODO(V2): tighten to also require wire_attribution on at least
        # one side. Rationale: the unconditional bypass handles verbatim
        # wire republication cleanly, but in principle a hash collision
        # on a long-gap story update could cross-merge unrelated content.
        # Gating on wire_attribution preserves the syndication case while
        # preventing that. Revisit once we have real fetch data to inspect.
        if member.fingerprint == article.fingerprint:
            return True, 1.0
        # Lexical path gated by time proximity AND by both sides having
        # enough informative tokens for a meaningful Jaccard score.
        if not article_lexically_matchable:
            continue
        if abs(article.published_at - member.published_at) > time_bucket:
            continue
        member_tokens = _tokenize(member.normalized_title)
        if len(member_tokens) < MIN_INFORMATIVE_TOKENS:
            continue
        score = _jaccard(article_tokens, member_tokens)
        if score > best_jaccard:
            best_jaccard = score
    if best_jaccard >= title_jaccard_threshold:
        return True, best_jaccard
    return False, 0.0


# --- cluster creation + bookkeeping ---------------------------------------

def _new_cluster_id(article_id: str) -> str:
    """Derive a deterministic 16-hex cluster id from its seed article.

    Same seed -> same cluster_id, which keeps ledger rebuilds idempotent
    when a fresh state is reconstructed from articles.json alone.
    """
    return hashlib.sha1(f"cluster:{article_id}".encode("utf-8")).hexdigest()[:16]


def _rebuild_cluster_bookkeeping(
    cluster: StoryCluster,
    all_articles: dict[str, NormalizedArticle],
    sources: dict[str, Source],
    now: datetime,
) -> None:
    """Recompute centroid + counts + timestamps from current article_ids.

    Mutates the cluster in place. Does NOT touch `state`, `is_stale`, or
    `last_state_change_at` — those are confidence.py's concern (Phase 3).
    """
    members = [all_articles[aid] for aid in cluster.article_ids if aid in all_articles]
    if not members:
        return  # defensive: shouldn't happen in normal flow

    # Centroid = longest display_title. Cheap heuristic: wire-syndicated
    # headlines are typically trimmer; fuller outlet headlines carry
    # more signal for the voice line.
    centroid = max(members, key=lambda a: len(a.display_title))

    indep = count_independent_confirmations(members, sources)
    article_count = len(members)

    cluster.centroid_display_title = centroid.display_title
    cluster.centroid_normalized_title = centroid.normalized_title
    cluster.first_seen_at = min(a.published_at for a in members)
    cluster.latest_update_at = max(a.published_at for a in members)
    cluster.last_checked_at = now
    cluster.independent_confirmation_count = indep
    cluster.article_count = article_count
    cluster.syndicated_article_count = article_count - indep


# --- main entry point -----------------------------------------------------

def cluster_articles(
    all_articles: dict[str, NormalizedArticle],
    existing_clusters: dict[str, StoryCluster],
    sources: dict[str, Source],
    *,
    title_jaccard_threshold: float = DEFAULT_TITLE_JACCARD_THRESHOLD,
    time_bucket_hours: int = DEFAULT_TIME_BUCKET_HOURS,
    now: datetime | None = None,
) -> tuple[dict[str, StoryCluster], set[str]]:
    """Assign unassigned articles to clusters. Pure.

    Args:
        all_articles: article_id -> NormalizedArticle. Includes both
            articles already in existing_clusters and new ones.
        existing_clusters: current cluster_id -> StoryCluster mapping.
            Not mutated.
        sources: source_id -> Source, for ownership_group lookup.
        title_jaccard_threshold: minimum token Jaccard for a lexical
            match. Default 0.55 (see spec delta).
        time_bucket_hours: max |published_at delta| for a lexical match.
            Fingerprint matches bypass this. Default 48.
        now: clock override for deterministic tests. Defaults to utcnow().

    Returns:
        (updated_clusters, affected_cluster_ids)
          - updated_clusters: all clusters post-assignment. Unaffected
            ones are identical-by-identity to the input.
          - affected_cluster_ids: set of clusters that were newly
            created or gained at least one article this call.

    Unassigned articles are processed in published_at ascending order so
    the older article always seeds a cluster and later articles merge
    in. When an article matches multiple clusters, the highest-scoring
    cluster wins; ties break by insertion order.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    time_bucket = timedelta(hours=time_bucket_hours)

    # Which articles are already assigned to some existing cluster?
    assigned: set[str] = set()
    for c in existing_clusters.values():
        assigned.update(c.article_ids)

    unassigned = sorted(
        (all_articles[aid] for aid in all_articles if aid not in assigned),
        key=lambda a: a.published_at,
    )

    # Shallow copy the mapping; cluster objects may be mutated in place
    # (StoryCluster is a plain @dataclass, not frozen).
    clusters: dict[str, StoryCluster] = dict(existing_clusters)
    affected: set[str] = set()

    for article in unassigned:
        best_cluster_id: str | None = None
        best_score = 0.0
        for cid, cluster in clusters.items():
            matched, score = _match_cluster(
                article, cluster, all_articles,
                title_jaccard_threshold=title_jaccard_threshold,
                time_bucket=time_bucket,
            )
            if matched and score > best_score:
                best_cluster_id = cid
                best_score = score

        if best_cluster_id is not None:
            clusters[best_cluster_id].article_ids.append(article.article_id)
            affected.add(best_cluster_id)
        else:
            cid = _new_cluster_id(article.article_id)
            clusters[cid] = StoryCluster(
                cluster_id=cid,
                article_ids=[article.article_id],
                centroid_display_title=article.display_title,
                centroid_normalized_title=article.normalized_title,
                first_seen_at=article.published_at,
                latest_update_at=article.published_at,
                last_checked_at=now,
                last_state_change_at=now,
                state=ConfidenceState.DEVELOPING,
                is_stale=False,
                independent_confirmation_count=0,   # filled below
                article_count=0,                    # filled below
                syndicated_article_count=0,         # filled below
            )
            affected.add(cid)

    # Recompute centroid + counts + timestamps for every touched cluster.
    for cid in affected:
        _rebuild_cluster_bookkeeping(clusters[cid], all_articles, sources, now)

    return clusters, affected
