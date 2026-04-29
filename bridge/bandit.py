"""Retrieval-based preference bandit for tool-routing nudges.

Given a prompt embedding for the CURRENT turn, look up the most
similar past turns in the feedback store, aggregate their per-tool
ratings, and produce a short system-prompt block the LLM can read.
Never blocks, never fails noisily — insufficient data just returns
an empty string.

This is explicitly NOT:
- a model-weight update (no PPO / DPO / GRPO)
- a hard constraint on tool selection (the suffix is a nudge, not a rule)

It's a k-NN over a local JSONL log with a weighted average per tool.
Cheap enough to run on every turn (~1 ms for <10 k entries), simple
enough to reason about, and trivially disabled by clearing the log.
"""
from __future__ import annotations

import logging
from typing import Callable

from bridge.feedback import FeedbackStore

log = logging.getLogger("bridge.bandit")


# Thresholds picked for conservatism: we'd rather say nothing than
# inject a misleading hint into the system prompt. Tuned empirically —
# tighten if hints start firing from noisy signal, loosen if they
# never fire.
_MIN_NEIGHBOURS_FOR_HINT: int = 2        # need ≥2 similar past turns
_MIN_SAMPLES_PER_TOOL: int = 1           # a tool needs this many data points
_TOOL_AVG_MAGNITUDE_THRESHOLD: float = 0.4  # |avg rating| must exceed this
_DEFAULT_K: int = 5


def preference_hint(
    embedding: list[float] | None,
    store: FeedbackStore,
    *,
    k: int = _DEFAULT_K,
    min_similarity: float = 0.5,
) -> str:
    """Build a system-prompt nudge from past feedback on similar prompts.

    Returns a short block describing which tools tended to get thumbs-up
    vs thumbs-down on prompts similar to this one. Returns ``""`` (no
    hint) when:
      - no embedding was supplied (embedder was down)
      - the store has <MIN_NEIGHBOURS_FOR_HINT rated similar entries
      - no tool's weighted average clears the magnitude threshold

    The LLM sees the hint in its system prompt; the existing Karin
    routing rules still apply — this only biases the margin.
    """
    if not embedding:
        return ""

    neighbours = store.knn_similar(embedding, k=k, min_similarity=min_similarity)
    if len(neighbours) < _MIN_NEIGHBOURS_FOR_HINT:
        return ""

    # Weighted-average per tool: each past turn's rating counts with
    # weight == its cosine similarity to the current prompt. That way
    # the closest neighbours dominate and mildly-related turns don't
    # derail the signal.
    tool_sum: dict[str, float] = {}
    tool_weight: dict[str, float] = {}
    for entry, sim in neighbours:
        r = entry.rating
        if r is None:
            continue
        for tc in entry.tool_chain:
            name = tc.get("name") or ""
            if not name:
                continue
            tool_sum[name] = tool_sum.get(name, 0.0) + float(r) * sim
            tool_weight[name] = tool_weight.get(name, 0.0) + sim

    liked: list[tuple[str, float]] = []
    disliked: list[tuple[str, float]] = []
    for name, total_weight in tool_weight.items():
        if total_weight <= 0.0:
            continue
        avg = tool_sum[name] / total_weight
        if abs(avg) < _TOOL_AVG_MAGNITUDE_THRESHOLD:
            continue
        # Also require at least N samples for this specific tool, so a
        # single strongly-rated similar turn can't dominate the hint.
        samples = sum(
            1 for entry, _s in neighbours
            if any(tc.get("name") == name for tc in entry.tool_chain)
        )
        if samples < _MIN_SAMPLES_PER_TOOL:
            continue
        if avg > 0:
            liked.append((name, avg))
        else:
            disliked.append((name, avg))

    if not liked and not disliked:
        return ""

    # Sort strongest signal first for each bucket.
    liked.sort(key=lambda t: t[1], reverse=True)
    disliked.sort(key=lambda t: t[1])

    lines: list[str] = ["[preference hint from past feedback]"]
    if liked:
        parts = ", ".join(f"{n} ({v:+.2f})" for n, v in liked[:3])
        lines.append(f"For prompts like this, users preferred: {parts}.")
    if disliked:
        parts = ", ".join(f"{n} ({v:+.2f})" for n, v in disliked[:3])
        lines.append(f"For prompts like this, users disliked: {parts}.")
    lines.append(
        "Treat this as a nudge, not a rule — follow the routing "
        "rules above unless the hint clearly applies."
    )
    return "\n".join(lines)


def retry_hint(prev_tool_chain: list[dict]) -> str:
    """System-prompt nudge used when the user clicks thumbs-down and
    we're re-running the same user message.

    Adds a short note listing the tools used on the unsatisfactory
    attempt so the model knows what to avoid repeating. Paired with
    the preference_hint suffix, this steers the retry toward the next
    best option without hardcoding any fallback tool.
    """
    if not prev_tool_chain:
        return (
            "[retry] The previous answer was rated negative. Try a "
            "different approach — if you used a tool, consider whether "
            "answering directly from your own knowledge would be "
            "better, or pick a more specific tool."
        )
    names = [tc.get("name") or "" for tc in prev_tool_chain if tc.get("name")]
    names_str = ", ".join(names) if names else "(none)"
    return (
        f"[retry] The previous answer was rated negative. On that "
        f"attempt you called: {names_str}. Take a different path this "
        f"time — either a different tool, or answer directly from "
        f"your own knowledge."
    )
