"""LLM-assisted semantic-duplicate merge for borderline news clusters.

Principle: the code path (lexical Jaccard + fingerprint) is still the
trust layer. The LLM acts ONLY as a second opinion on cases that fall
in a narrow borderline band, and it can only AGREE to a merge — never
split a code-merged cluster, never force a merge code has rejected
too far (Jaccard < 0.3 stays apart regardless of what the LLM thinks).

Design invariants:
  - Deterministic layer stays authoritative. LLM vote never SPLITS
    clusters — it can only MERGE two code-kept-apart clusters when
    Jaccard was borderline (below the auto-merge threshold but not
    implausibly far below).
  - Every LLM decision is logged as an event in events.jsonl with the
    input pair + the answer + the action taken, for audit / later RL
    signal extraction.
  - Strict rate cap per ingest cycle so a noisy day doesn't burn the
    LLM's attention budget or exhaust free-tier API quotas.
  - Fail-safe: any LLM error is treated as "no merge" and logged. The
    news subsystem keeps working exactly as before when the LLM is
    unavailable.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from bridge.news.cluster import MIN_INFORMATIVE_TOKENS, _jaccard, _tokenize
from bridge.news.models import NormalizedArticle, StoryCluster

log = logging.getLogger("bridge.news.cross_verify")

# --- tunables (all conservative) --------------------------------------------

# Jaccard similarity band in which the LLM is allowed to cast a vote.
# Below LOW: pairs are obviously different (stays apart, no ask).
# At/above HIGH (which is the cluster.py merge threshold): pairs already
# merged by code; no ask.
# Between LOW and HIGH: borderline — the LLM might catch a semantic
# duplicate that lexical matching missed (different words, same event).
# All thresholds below are tunable via config/tuning.yaml under
# `news.cross_verify.*`; the literals are the fallback defaults.
from bridge import tuning as _tuning

BORDERLINE_JACCARD_LOW = _tuning.get(
    "news.cross_verify.borderline_jaccard_low", 0.30,
)
BORDERLINE_JACCARD_HIGH = _tuning.get(
    "news.cross_verify.borderline_jaccard_high", 0.55,
)

# Only consider pairs where both clusters have at least one fresh
# article. "Fresh" = published within this window. Picks up reasonable
# news-day rhythm without chasing week-old stories.
FRESH_WINDOW_HOURS = _tuning.get(
    "news.cross_verify.fresh_window_hours", 24,
)

# Hard cap on LLM round-trips per ingest cycle. Keeps cost predictable
# even on a high-volume day, and keeps the supplementary layer from
# dominating the latency budget.
MAX_LLM_CALLS_PER_CYCLE = _tuning.get(
    "news.cross_verify.max_llm_calls_per_cycle", 5,
)

# Minimum cluster size before we'll spend LLM budget on it. Singletons
# (1-article clusters) ARE the primary target — if the lexical matcher
# had found a match they wouldn't be singletons — so we accept them.
# The rate cap (MAX_LLM_CALLS_PER_CYCLE) is what actually prevents
# explosion; we sort by Jaccard DESC so the most plausible borderline
# pairs get the budget first.
MIN_CLUSTER_ARTICLES = _tuning.get(
    "news.cross_verify.min_cluster_articles", 1,
)


# --- ask-the-LLM protocol ---------------------------------------------------


class LLMVerifier(Protocol):
    """The LLM adapter we call to verify a borderline pair.

    Keep this tiny on purpose — we pass two strings, get a bool.
    Implementations can wrap any LLM; tests inject a fake to avoid
    network.
    """

    def same_event(self, title_a: str, title_b: str) -> bool: ...


@dataclass(frozen=True)
class VerificationEvent:
    """One audit record per LLM ask. Serialized into events.jsonl.

    Useful for:
      - Ops: "did we over-merge anything?"
      - RL later: this is exactly the (prompt, response, decision)
        triple a reward model would train on.
    """

    cluster_a_id: str
    cluster_b_id: str
    title_a: str
    title_b: str
    jaccard: float
    llm_said_same: bool
    merged: bool
    when: datetime


# --- candidate selection ----------------------------------------------------


def find_candidate_pairs(
    clusters: dict[str, StoryCluster],
    articles: dict[str, NormalizedArticle],
    *,
    now: datetime | None = None,
) -> list[tuple[str, str, float]]:
    """Return ``[(cluster_a_id, cluster_b_id, jaccard)]`` for pairs
    worth asking the LLM about.

    Selection rules (all must hold):
      - Both clusters have ≥ MIN_CLUSTER_ARTICLES articles.
      - Both have an article within FRESH_WINDOW_HOURS of ``now``.
      - Jaccard between centroid tokens is in
        [BORDERLINE_JACCARD_LOW, BORDERLINE_JACCARD_HIGH).
      - Both centroids have ≥ MIN_INFORMATIVE_TOKENS after tokenization
        (same guard the lexical matcher uses).

    Sorted highest-Jaccard-first so when we rate-cap, we spend the
    budget on the pairs most likely to benefit. Pure function —
    no LLM, no network.
    """
    now = now or datetime.now(timezone.utc)
    fresh_cutoff = now - timedelta(hours=FRESH_WINDOW_HOURS)

    qualifying: list[tuple[str, StoryCluster, set[str]]] = []
    for cid, c in clusters.items():
        if len(c.article_ids) < MIN_CLUSTER_ARTICLES:
            continue
        members = [articles[aid] for aid in c.article_ids if aid in articles]
        if not members:
            continue
        if not any(m.published_at >= fresh_cutoff for m in members):
            continue
        tokens = _tokenize(c.centroid_normalized_title)
        if len(tokens) < MIN_INFORMATIVE_TOKENS:
            continue
        qualifying.append((cid, c, tokens))

    candidates: list[tuple[str, str, float]] = []
    for i in range(len(qualifying)):
        a_id, _a, a_tokens = qualifying[i]
        for j in range(i + 1, len(qualifying)):
            b_id, _b, b_tokens = qualifying[j]
            score = _jaccard(a_tokens, b_tokens)
            if BORDERLINE_JACCARD_LOW <= score < BORDERLINE_JACCARD_HIGH:
                candidates.append((a_id, b_id, score))
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates


# --- merge execution --------------------------------------------------------


def _merge_clusters(
    target: StoryCluster, absorbed: StoryCluster,
) -> StoryCluster:
    """Produce a merged cluster. Keeps ``target``'s id; absorbed
    cluster goes away.

    The merge is naive: union of article_ids, take the more-recent
    ``latest_update_at``, keep target's centroid + state. The caller
    re-runs ``rescore_cluster()`` on the result so state + counts
    reflect the combined membership. Bookkeeping fields (counts, etc.)
    are left to the rescore pass; we seed approximate values here so
    the object is valid between merge and rescore.
    """
    combined_article_ids = list(dict.fromkeys(
        list(target.article_ids) + list(absorbed.article_ids)
    ))
    latest = max(target.latest_update_at, absorbed.latest_update_at)
    earliest = min(target.first_seen_at, absorbed.first_seen_at)
    return StoryCluster(
        cluster_id=target.cluster_id,
        article_ids=combined_article_ids,
        centroid_display_title=target.centroid_display_title,
        centroid_normalized_title=target.centroid_normalized_title,
        first_seen_at=earliest,
        latest_update_at=latest,
        last_checked_at=target.last_checked_at,
        last_state_change_at=target.last_state_change_at,
        state=target.state,                  # will be rescored by caller
        is_stale=False,                       # recompute at read time
        # Counts are approximate — rescore will recompute.
        independent_confirmation_count=(
            target.independent_confirmation_count
            + absorbed.independent_confirmation_count
        ),
        article_count=len(combined_article_ids),
        syndicated_article_count=max(
            0,
            len(combined_article_ids) - (
                target.independent_confirmation_count
                + absorbed.independent_confirmation_count
            ),
        ),
    )


# --- the public entry point -------------------------------------------------


def cross_verify_clusters(
    clusters: dict[str, StoryCluster],
    articles: dict[str, NormalizedArticle],
    verifier: LLMVerifier,
    *,
    now: datetime | None = None,
    max_calls: int = MAX_LLM_CALLS_PER_CYCLE,
    event_sink: Callable[[VerificationEvent], None] | None = None,
) -> dict[str, StoryCluster]:
    """Apply LLM-assisted merges to a cluster set.

    Returns a NEW cluster dict. Never mutates the input. Absorbed
    clusters are removed from the returned dict; targets are replaced
    with their merged form. Non-candidate clusters pass through
    untouched.

    ``event_sink``, if given, receives one :class:`VerificationEvent`
    per LLM call (merge or not). Service layer wires it to
    `ledger.append_event` so every cross-verify decision is auditable.
    """
    now = now or datetime.now(timezone.utc)
    candidates = find_candidate_pairs(clusters, articles, now=now)
    if not candidates:
        return dict(clusters)

    result = dict(clusters)
    absorbed_ids: set[str] = set()
    calls_made = 0

    for a_id, b_id, score in candidates:
        if calls_made >= max_calls:
            break
        # If either side was already absorbed by an earlier merge in
        # this pass, skip — the decision was already made.
        if a_id in absorbed_ids or b_id in absorbed_ids:
            continue
        if a_id not in result or b_id not in result:
            continue
        a_cluster = result[a_id]
        b_cluster = result[b_id]

        title_a = a_cluster.centroid_display_title
        title_b = b_cluster.centroid_display_title

        calls_made += 1
        try:
            same = bool(verifier.same_event(title_a, title_b))
        except Exception as e:
            log.warning(
                "cross_verify: LLM call failed (treating as 'different'): %s",
                e,
            )
            same = False

        merged_now = False
        if same:
            merged = _merge_clusters(a_cluster, b_cluster)
            result[a_id] = merged
            del result[b_id]
            absorbed_ids.add(b_id)
            merged_now = True
            log.info(
                "cross_verify: merged %s + %s (jaccard=%.2f) via LLM yes",
                a_id, b_id, score,
            )
        else:
            log.debug(
                "cross_verify: LLM said different for %s / %s (jaccard=%.2f)",
                a_id, b_id, score,
            )

        if event_sink is not None:
            try:
                event_sink(VerificationEvent(
                    cluster_a_id=a_id, cluster_b_id=b_id,
                    title_a=title_a, title_b=title_b,
                    jaccard=score, llm_said_same=same,
                    merged=merged_now, when=now,
                ))
            except Exception as e:
                # Never let an audit hook break the merge path.
                log.debug("cross_verify event sink raised: %s", e)

    log.info(
        "cross_verify: %d candidates, %d LLM calls, %d merges",
        len(candidates), calls_made, len(absorbed_ids),
    )
    return result


# --- a default OllamaLLM-backed verifier ------------------------------------


_VERIFY_PROMPT = (
    "You are a deduplication checker for a news system. You will be "
    "given two article headlines. Decide if they report the SAME "
    "real-world event (same incident, same decision, same announcement) "
    "or two DIFFERENT stories that merely share keywords.\n\n"
    "Answer with ONE word only: SAME or DIFFERENT. No explanation, no "
    "punctuation, nothing else."
)


class OllamaLLMVerifier:
    """LLMVerifier backed by an OllamaLLM client.

    Uses the same model the assistant is already running — no extra
    model load, no extra VRAM. One short, tool-less chat per pair:
    <system prompt>: instruction, <user>: "A: ...\\nB: ...".

    The LLM's response is normalized: strip, uppercase, take first
    word. Anything other than 'SAME' is treated as 'different' —
    fail-safe.
    """

    def __init__(self, llm, *, request_timeout: float | None = None) -> None:
        self._llm = llm
        # Optional override; default uses whatever the llm was configured with.
        self._request_timeout = request_timeout

    def same_event(self, title_a: str, title_b: str) -> bool:
        # We intentionally call at a LOWER level than llm.chat() — we
        # don't want tool schemas, history, or the Karin persona
        # contaminating this classification. Build a one-shot chat
        # body by hand.
        import json as _json
        messages = [
            {"role": "system", "content": _VERIFY_PROMPT},
            {"role": "user", "content": f"A: {title_a}\nB: {title_b}"},
        ]
        body = {
            "model": self._llm.model,
            "messages": messages,
            "stream": False,
            "options": {
                # Near-deterministic: we want the same input to give
                # the same answer on retry.
                "temperature": 0.0,
                # Short cap; 'SAME' / 'DIFFERENT' is one token.
                "num_predict": 4,
                "num_ctx": 512,
            },
            "keep_alive": -1,
        }
        if getattr(self._llm, "_think", None) is not None:
            body["think"] = self._llm._think
        try:
            resp = self._llm._post_chat(body)
            data = resp.json()
        except Exception as e:
            log.warning("cross_verify LLM call failed: %s", e)
            return False
        content = (data.get("message") or {}).get("content") or ""
        token = content.strip().upper().split(None, 1)
        if not token:
            return False
        return token[0] == "SAME"
