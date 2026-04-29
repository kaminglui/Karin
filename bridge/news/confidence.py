"""Confidence state engine + story brief builder.

Pure and deterministic. Reads cluster state + member articles + source
registry, produces an updated ConfidenceState, is_stale flag, and a
user-facing StoryBrief with a one-sentence voice_line.

State rules (V1, per spec — CONTESTED / RETRACTED deferred to V2):

  DEVELOPING
      Default. Holds until one of the thresholds below is met.

  PROVISIONALLY_CONFIRMED
      independent_confirmation_count >= 2 AND
      at least one contributing bucket has tier in {WIRE, REPUTABLE}.

  CONFIRMED
      independent_confirmation_count >= 3 AND (
          at least one contributing bucket has tier == WIRE
          OR at least two have tier == REPUTABLE
      ).

Three-bucket-of-OTHER scenarios stay DEVELOPING by design: without a
reputable or wire contributor, count alone isn't enough for V1 to
promote a story. Revisit in V2 if it ever matters.

`is_stale` is an overlay tied to `last_checked_at`, not
`latest_update_at` — per spec-delta #1. A cluster is stale if we haven't
re-scored it in more than stale_threshold_hours (default 24h). Content
freshness (`latest_update_at`) is separate and not surfaced as a flag
in V1.

Voice lines are deterministic string templates. No LLM in the trust
path — the LLM paraphrases voice_line at the tool boundary, never
computes it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Iterable

from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    Source,
    StoryBrief,
    StoryCluster,
    Tier,
)

if TYPE_CHECKING:
    # Import-only for typing; avoids a circular import at runtime
    # (translate -> (nothing news-specific), but confidence is on the
    # hot path and doesn't need to pay for httpx import on startup).
    from bridge.news.extract import ExtractedArticle
    from bridge.news.translate import Translator

# --- defaults --------------------------------------------------------------

DEFAULT_STALE_THRESHOLD_HOURS = 24
_TOPIC_SHORT_MAX_WORDS = 8

# Words a short snippet should not END on. Snippets that get cut at
# max_words sometimes land on a function word ("...concedes defeat in"),
# which reads awkwardly when spoken. Conservative list — only obviously
# weak endings; pronouns NOT included so e.g. "...by Charity He" still
# carries useful signal.
_WEAK_TRAILING_WORDS: frozenset[str] = frozenset({
    # prepositions
    "in", "on", "at", "by", "for", "to", "of", "with", "from",
    "as", "into", "onto", "upon", "against", "about", "over",
    "under", "through", "during", "before", "after",
    # articles
    "a", "an", "the",
    # coordinating + common subordinating conjunctions
    "and", "or", "but", "nor", "yet", "so", "if", "while", "when",
    # linking verbs (often dangle)
    "is", "are", "was", "were", "be", "been", "being",
})

# Tier-priority for sorting ownership buckets in voice lines + briefs.
# Wire buckets are sorted separately (alphabetical by wire name); this
# ordering only applies to the non-wire ownership buckets.
_TIER_RANK: dict[Tier, int] = {Tier.WIRE: 0, Tier.REPUTABLE: 1, Tier.OTHER: 2}

# Detail rendering. The extracted body can be a full article; we only
# want a 1-3 sentence "what happened" blurb for the brief. Cap length
# and sentence count so voice output + tool responses stay bounded.
_DETAIL_MAX_CHARS = 320
_DETAIL_MAX_SENTENCES = 3

# Sentence splitter — ASCII stops require trailing whitespace (so
# "U.S." doesn't trip a split mid-abbreviation), but CJK full-width
# stops split on the stop alone because Chinese / Japanese typography
# doesn't use inter-sentence spaces. Splitting on a zero-width match
# is supported by re.split in Python 3.7+.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])")


# --- bucket model ---------------------------------------------------------

@dataclass(frozen=True)
class IndependentBucket:
    """One independent-confirmation bucket. Either a wire byline or an
    ownership_group. The bucket structure is what state rules and brief
    text both read from — single source of truth for `independence`.
    """

    kind: str            # "wire" | "ownership"
    display_name: str    # "AP" for wire, Source.name for ownership
    tier: Tier


def independent_buckets(
    articles: Iterable[NormalizedArticle],
    sources: dict[str, Source],
) -> list[IndependentBucket]:
    """Collapse articles into independent confirmation buckets.

    Mirrors the rule in cluster.count_independent_confirmations but
    returns the bucket metadata instead of just a count, so callers can
    drive both the state engine and the voice line from one pass.

    Returned order: wire buckets alphabetical-by-name first, then
    ownership buckets ordered by (tier rank, display_name). This gives
    deterministic `top_source` / `second_source` text in the voice line.
    """
    wires: dict[str, IndependentBucket] = {}
    owns: dict[str, IndependentBucket] = {}
    for a in articles:
        if a.wire_attribution is not None:
            if a.wire_attribution not in wires:
                wires[a.wire_attribution] = IndependentBucket(
                    kind="wire",
                    display_name=a.wire_attribution,
                    tier=Tier.WIRE,
                )
            continue
        src = sources.get(a.source_id)
        if src is None:
            continue
        if src.ownership_group not in owns:
            owns[src.ownership_group] = IndependentBucket(
                kind="ownership",
                display_name=src.name,
                tier=src.tier,
            )

    result = sorted(wires.values(), key=lambda b: b.display_name)
    result += sorted(
        owns.values(),
        key=lambda b: (_TIER_RANK[b.tier], b.display_name),
    )
    return result


# --- state computation ----------------------------------------------------

def compute_state(
    articles: Iterable[NormalizedArticle],
    sources: dict[str, Source],
) -> ConfidenceState:
    """Derive a cluster's confidence state from its members.

    Evaluates CONFIRMED first, then PROVISIONALLY_CONFIRMED, then falls
    back to DEVELOPING. Monotone within a turn — a cluster never gets
    evaluated as "both provisional and confirmed."
    """
    buckets = independent_buckets(articles, sources)
    n_indep = len(buckets)
    n_wire = sum(1 for b in buckets if b.tier == Tier.WIRE)
    n_reputable = sum(1 for b in buckets if b.tier == Tier.REPUTABLE)

    if n_indep >= 3 and (n_wire >= 1 or n_reputable >= 2):
        return ConfidenceState.CONFIRMED
    if n_indep >= 2 and (n_wire + n_reputable) >= 1:
        return ConfidenceState.PROVISIONALLY_CONFIRMED
    return ConfidenceState.DEVELOPING


# --- staleness ------------------------------------------------------------

def compute_is_stale(
    cluster: StoryCluster,
    now: datetime,
    stale_threshold_hours: int = DEFAULT_STALE_THRESHOLD_HOURS,
) -> bool:
    """Ledger-staleness: have we re-scored this cluster recently enough?

    Returns True when (now - last_checked_at) exceeds the threshold.
    Intentionally does NOT consult latest_update_at — content freshness
    is separate and not exposed as a flag in V1 (spec-delta #1).
    """
    return (now - cluster.last_checked_at) > timedelta(hours=stale_threshold_hours)


# --- cluster rescoring ----------------------------------------------------

def rescore_cluster(
    cluster: StoryCluster,
    all_articles: dict[str, NormalizedArticle],
    sources: dict[str, Source],
    now: datetime | None = None,
    stale_threshold_hours: int = DEFAULT_STALE_THRESHOLD_HOURS,
) -> None:
    """Update cluster.state + cluster.is_stale in place.

    - state: recomputed from current members via compute_state().
    - last_state_change_at: only updated when state actually changes —
      lets readers tell "confirmed 2 days ago" from "just confirmed."
    - is_stale: computed from last_checked_at (which cluster.py owns).
      The service layer is expected to call cluster.py's
      _rebuild_cluster_bookkeeping before rescore_cluster so
      last_checked_at is fresh.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    members = [all_articles[aid] for aid in cluster.article_ids if aid in all_articles]
    new_state = compute_state(members, sources)
    if new_state != cluster.state:
        cluster.state = new_state
        cluster.last_state_change_at = now
    cluster.is_stale = compute_is_stale(cluster, now, stale_threshold_hours)


# --- brief building -------------------------------------------------------

def _topic_short(display_title: str, max_words: int = _TOPIC_SHORT_MAX_WORDS) -> str:
    """Trim a centroid headline to a voice-friendly fragment.

    Takes the first max_words tokens, then drops any trailing weak
    function words (prepositions, articles, conjunctions, linking verbs)
    so snippets don't end on "...defeat in" or "...the deal and".
    Strips trailing punctuation last so sentence templates don't
    double up.

    Conservative weak-word list — pronouns NOT included, so an ending
    like "...by Charity He" still passes. Worst case: an extra weak
    word stays; never worse than no truncation polish at all.
    """
    words = display_title.split()
    if len(words) <= max_words:
        trimmed_words = list(words)
    else:
        trimmed_words = list(words[:max_words])
    # Drop trailing weak words. Strip per-token punctuation (so "in." or
    # "in," still trip the check) before comparing.
    while trimmed_words:
        last_clean = trimmed_words[-1].lower().strip(".,:;!?\"')]")
        if last_clean in _WEAK_TRAILING_WORDS:
            trimmed_words.pop()
            continue
        break
    if not trimmed_words:
        # Pathological input where every leading word was weak — fall
        # back to the original first word so we never return "".
        trimmed_words = [words[0]] if words else []
    return " ".join(trimmed_words).rstrip(".,:;!?").strip()


def _build_voice_line(
    state: ConfidenceState,
    is_stale: bool,
    buckets: list[IndependentBucket],
    topic_short: str,
) -> str:
    """Deterministic one-sentence voice line per state.

    Templates (spec-delta #2):
      confirmed:               "{top} and {second} report {topic}. {n} independent sources, confirmed."
      provisionally_confirmed: "{top} and {second} report {topic}. {n} independent sources, provisionally confirmed."
      developing:              "One source reports {topic}. Still developing."
      + is_stale suffix:       " This may be outdated."

    If a confirmed/provisional state somehow has <2 buckets (shouldn't
    happen given compute_state thresholds), falls back to the developing
    template — defensive.
    """
    n = len(buckets)
    fallback = f"One source reports {topic_short}. Still developing."

    if state == ConfidenceState.CONFIRMED and n >= 2:
        top = buckets[0].display_name
        second = buckets[1].display_name
        line = (
            f"{top} and {second} report {topic_short}. "
            f"{n} independent sources, confirmed."
        )
    elif state == ConfidenceState.PROVISIONALLY_CONFIRMED and n >= 2:
        top = buckets[0].display_name
        second = buckets[1].display_name
        line = (
            f"{top} and {second} report {topic_short}. "
            f"{n} independent sources, provisionally confirmed."
        )
    else:
        line = fallback

    if is_stale:
        line += " This may be outdated."
    return line


def _cluster_language(members: list[NormalizedArticle]) -> str:
    """Pick the language label we attribute to the cluster as a whole.

    Rule: mode over non-"und" member languages, tie-broken by the
    centroid's member (first one in the list). Falls back to "en" when
    every member is undetected (legacy ledger rows, pathological
    titles). This is a display concern only — clustering never runs on
    language.
    """
    counts: dict[str, int] = {}
    for m in members:
        lang = m.language or "en"
        if lang == "und":
            continue
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "en"
    return max(counts, key=lambda k: counts[k])


def _pick_detail_source(
    cluster: StoryCluster,
    members: list[NormalizedArticle],
    extracted: "dict[str, ExtractedArticle] | None",
) -> str:
    """Return the best case-preserved blurb available for a cluster.

    Preference order:
      1. Extracted body text of the centroid-matching member (best —
         that's what the modal shows). Trimmed to a few sentences.
      2. display_summary of any member (RSS-provided, usually 1
         sentence).
      3. Empty string. Caller renders without a detail line.

    Centroid match is best-effort: we pick whichever member has the
    same display_title as the cluster centroid, else the first member.
    """
    if not members:
        return ""

    # Walk members by centroid match first, then stable list order.
    ordered = sorted(
        members,
        key=lambda m: 0 if m.display_title == cluster.centroid_display_title else 1,
    )

    if extracted:
        for m in ordered:
            rec = extracted.get(m.article_id)
            if rec is not None and rec.text:
                return _trim_detail(rec.text)

    for m in ordered:
        if m.display_summary:
            return _trim_detail(m.display_summary)
    return ""


def _trim_detail(text: str) -> str:
    """Clip a long body to <= _DETAIL_MAX_SENTENCES sentences and
    _DETAIL_MAX_CHARS characters. Whichever limit trips first.

    Splits on ASCII + CJK sentence punctuation so translated Chinese
    or Japanese detail survives trimming; the regex keeps punctuation
    on the preceding sentence, so "她说。" comes out whole.
    """
    text = text.strip()
    if not text:
        return ""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    picked = sentences[:_DETAIL_MAX_SENTENCES]
    joined = " ".join(s.strip() for s in picked if s.strip())
    if len(joined) > _DETAIL_MAX_CHARS:
        # Prefer a sentence boundary near the limit; fall back to hard
        # char cut + ellipsis so we never emit a half-word.
        truncated = joined[:_DETAIL_MAX_CHARS]
        cut = max(
            truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "),
            truncated.rfind("\u3002"), truncated.rfind("\uff01"), truncated.rfind("\uff1f"),
        )
        if cut > _DETAIL_MAX_CHARS // 2:
            joined = truncated[:cut + 1].strip()
        else:
            joined = truncated.rstrip() + "\u2026"
    return joined


def _maybe_translate(
    text: str,
    source_lang: str,
    target_lang: str,
    translator: "Translator | None",
) -> tuple[str, str]:
    """Apply the "fits" rule and return (text, effective_language).

    Rule:
      - Empty text -> passthrough, language = target (nothing to render).
      - source_lang == target_lang -> original verbatim (always fits;
        headlines are publisher-bounded and don't need translation).
      - source_lang == "und" -> treat as target: when the detector
        can't decide, don't try to translate something we can't name.
        Matches current behavior on legacy / thin input.
      - translator is None -> original + its own language. Caller is
        opting out of translation (feature flag off, or brief built
        from a non-translating path like the alerts subsystem).
      - Else -> translator.translate(); if it fails, fall back to
        original + original language. The translator is already
        fail-soft internally, so we just plumb the result through.
    """
    if not text:
        return text, target_lang
    if source_lang == target_lang or source_lang == "und":
        return text, target_lang if source_lang == "und" else source_lang
    if translator is None:
        return text, source_lang
    result = translator.translate(text, target_lang=target_lang, source_lang=source_lang)
    if result.translated:
        return result.text, target_lang
    return result.text, source_lang


def build_brief(
    cluster: StoryCluster,
    all_articles: dict[str, NormalizedArticle],
    sources: dict[str, Source],
    *,
    target_language: str = "en",
    translator: "Translator | None" = None,
    extracted: "dict[str, ExtractedArticle] | None" = None,
) -> StoryBrief:
    """Construct a user-facing brief for one cluster.

    Reads the cluster's CURRENT state + is_stale fields — so the service
    layer should call rescore_cluster first if it wants fresh values.
    Pure: does not mutate cluster.

    Translation path:
      - `target_language` is the language the UI / Karin wants.
      - The cluster's effective source language comes from a mode over
        member article languages (set by the detector at ingest).
      - If the source matches target, we short-circuit — the
        translator is never called, so a flaky Ollama can't poison
        a brief we don't need it for.
      - Otherwise each of (headline, detail, topic_short) is
        translated independently. Independent calls mean the cache
        hits when the same detail appears under two different
        headlines — common with wire stories.

    Backwards compatible: with default args (`target_language="en"`,
    `translator=None`, `extracted=None`) behavior matches Phase 4.5 —
    same headline, same voice_line, empty detail. Existing callers need
    no changes.
    """
    members = [all_articles[aid] for aid in cluster.article_ids if aid in all_articles]
    buckets = independent_buckets(members, sources)

    top_sources = [b.display_name for b in buckets]
    reasoning = [f"{b.display_name} ({b.tier.value})" for b in buckets]

    source_lang = _cluster_language(members)
    detail_raw = _pick_detail_source(cluster, members, extracted)
    topic_short_raw = _topic_short(cluster.centroid_display_title)

    headline, headline_lang = _maybe_translate(
        cluster.centroid_display_title, source_lang, target_language, translator,
    )
    detail, _detail_lang = _maybe_translate(
        detail_raw, source_lang, target_language, translator,
    )
    topic_short, _topic_lang = _maybe_translate(
        topic_short_raw, source_lang, target_language, translator,
    )

    voice_line = _build_voice_line(
        state=cluster.state,
        is_stale=cluster.is_stale,
        buckets=buckets,
        topic_short=topic_short,
    )

    return StoryBrief(
        cluster_id=cluster.cluster_id,
        headline=headline,
        state=cluster.state,
        is_stale=cluster.is_stale,
        independent_confirmation_count=cluster.independent_confirmation_count,
        article_count=cluster.article_count,
        syndicated_article_count=cluster.syndicated_article_count,
        top_sources=top_sources,
        reasoning=reasoning,
        voice_line=voice_line,
        first_seen_at=cluster.first_seen_at,
        latest_update_at=cluster.latest_update_at,
        detail=detail,
        language=headline_lang,
    )
