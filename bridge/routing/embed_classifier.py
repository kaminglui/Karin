"""Embedding-based fallback classifier for the routing pipeline.

Runs *after* the regex classifier abstains (returns None). Encodes the
user prompt with a small sentence-transformer (MiniLM-L6-v2 via
fastembed / onnxruntime, CPU-only so it doesn't contend with Ollama
for the Jetson's 8 GB VRAM), then picks the tool whose anchor set has
the highest cosine similarity — provided the score clears a threshold
AND beats the runner-up by a margin.

Why this layer exists:

The regex classifier is precise but brittle — it only fires on the
exact lexical shapes we put in ``routing_patterns``. Paraphrases
("Where's a decent ramen spot?", "Give me the rundown on X",
"DIY bookshelf plans") sail past it and land on the LLM, which often
fails to route them without a hint. Embeddings catch the *semantic*
similarity the regex misses.

Design choices:

- **CPU-only** (``providers=["CPUExecutionProvider"]``). The Jetson's
  GPU is fully committed to mannix's ~5 GB KV cache; adding a second
  ONNX session on CUDA would either evict the LLM or OOM.
- **Anchor corpus is hand-curated** in ``_ANCHORS`` below. Each tool
  has 6-10 short phrases covering its intent space. Keeping them
  inline (not a YAML file) avoids a parse step on import and makes
  diffs cheap.
- **Cosine max over anchors, not mean.** A query only needs to match
  ONE anchor well for the tool to be a plausible fit.
- **Threshold + margin.** Score alone over-routes on ambiguous prompts;
  the margin requirement suppresses hints when two tools score close.
- **Lazy load**. Model + anchor embeddings are computed on first call,
  cached. Pays the ~1 s startup cost only if anyone routes through.
- **Failure-safe**. Any import/embed error => silent no-op. The regex
  + LLM path still works.

See ``bridge/routing/classifier.py::routing_hint`` for integration.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("bridge.routing.embed_classifier")

# Lazy state: (embedder_model, {tool: np.ndarray of shape (n_anchors, dim)})
_state: object | None = None  # None=not tried; False=tried/failed; tuple=loaded


# Per-tool anchor phrases. Each phrase is a short paraphrase of a
# canonical prompt that should route to this tool. Aim for diversity
# in lexical form (slangy, formal, terse, verbose) — the embedding
# will smooth over surface variance; we just need representative
# points in semantic space. Do NOT add anchors that overlap another
# tool's intent — that hurts the margin check.
_ANCHORS: dict[str, list[str]] = {
    "get_weather": [
        "what's the weather like today",
        "is it raining right now",
        "will it snow tomorrow",
        "temperature outside",
        "forecast for this weekend",
        "chilly enough for a jacket",
        "good weather for a barbecue",
    ],
    "get_news": [
        "any news today",
        "what's going on in the world",
        "top headlines right now",
        "current events update",
        "latest news in tech",
        "anything new in politics",
    ],
    "get_alerts": [
        "any severe weather warnings",
        "tornado watch in effect",
        "emergency alerts nearby",
        "hurricane warnings",
        "something dangerous going on",
        "is there a storm warning",
    ],
    "get_digest": [
        "what did I miss while I was away",
        "morning summary",
        "catch me up on today",
        "briefing of what happened",
        "recap of the day",
    ],
    "get_time": [
        "what time is it",
        "current date",
        "is it morning or evening there",
        "what's the hour now",
        "today's date",
    ],
    "tracker": [
        "gold price right now",
        "how's bitcoin doing",
        "current stock price of apple",
        "silver spot price",
        "tesla stock today",
        "crypto prices",
    ],
    "find_places": [
        "where's a good ramen spot",
        "coffee shop near me",
        "any decent bars around here",
        "restaurant recommendations nearby",
        "places with outdoor seating",
        "closest pharmacy",
    ],
    "wiki": [
        "what's the capital of france",
        "how far is mars from earth",
        "tell me about the manhattan project",
        "who was albert einstein",
        "encyclopedia entry on the roman empire",
        "history of world war two",
        "explain kirchhoff's voltage law",
        "when was napoleon born",
    ],
    "web_search": [
        "how do I fix a leaky faucet",
        "best budget 4k tvs of 2026",
        "top rated hiking boots",
        "diy bookshelf plans",
        "tutorial for git rebase",
        "reviews of the latest iphone",
        "how to make sourdough bread",
    ],
    "math": [
        "what's 15 times 23",
        "calculate 30 percent of 120",
        "solve x squared minus 4",
        "inverse of this matrix",
        "derivative of x cubed",
        "integrate sin x from 0 to pi",
    ],
    "convert": [
        "how many meters in a mile",
        "convert 72 fahrenheit to celsius",
        "5 pounds in kilograms",
        "how long is 3 hours in seconds",
        "convert 100 dollars to yen",
    ],
    "graph": [
        "plot y equals x squared",
        "show me e to the x",
        "visualize sine from 0 to 2 pi",
        "draw the curve x cubed",
        "graph this function",
    ],
    "circuit": [
        "series resistance of 100 and 200 ohms",
        "ohm's law for 5 volts and 100 ohms",
        "rc time constant with these values",
        "parallel resistor calculation",
        "truth table for this logic gate",
    ],
    "schedule_reminder": [
        "remind me to call mom at 5",
        "set an alarm for 7 am",
        "wake me up in an hour",
        "alert me at noon",
        "nudge me about the meeting",
    ],
    "update_memory": [
        "my name is alex",
        "I'm allergic to peanuts",
        "I live in seattle",
        "call me by my nickname from now on",
        "I was born in 1990",
    ],
    "inflation": [
        "how much was a dollar in 1970",
        "what would 50 bucks in 1985 be today",
        "purchasing power of 100 dollars from 1965",
        "how much did gas cost in 1980",
        "how have wages kept up with inflation",
        "what was a loaf of bread in 1990",
        "1 yuan in 1990 versus today",
        "Hong Kong dollar in 1990",
        "yen in 1970 worth today",
    ],
    "population": [
        "how many people lived in japan in 1970",
        "what was the world population in 1985",
        "population of hong kong in 1990",
        "how big is china's population",
        "how many people are there on earth",
        "us population growth from 1980 to 2020",
        "global population today",
    ],
    "facts": [
        "tell me about 1985",
        "what was 1965 like",
        "fun facts about 1990",
        "give me a snapshot of 1972",
        "1985 in numbers",
        "summary of the year 1995",
        "what happened in 1980",
    ],
}


def _load() -> object:
    """Lazy-init the embedder and per-tool anchor matrices. Returns
    ``(embedder, per_tool_embeddings)`` or False on failure."""
    global _state
    if _state is not None:
        return _state
    try:
        import numpy as np
        from fastembed import TextEmbedding

        # MiniLM-L6-v2: 22M params, 384-dim, CPU-fast (~30 ms/query on
        # Jetson Orin Nano), downloads once to ~/.cache on first use.
        model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            providers=["CPUExecutionProvider"],
        )

        tool_vecs: dict[str, "np.ndarray"] = {}
        for tool, phrases in _ANCHORS.items():
            embs = np.array(list(model.embed(phrases)), dtype=np.float32)
            # Normalize rows so cosine = dot product
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            tool_vecs[tool] = embs / norms

        _state = (model, tool_vecs)
        log.info(
            "embed_classifier: loaded MiniLM-L6-v2, %d tools, %d anchors",
            len(tool_vecs),
            sum(len(a) for a in _ANCHORS.values()),
        )
    except Exception as e:
        log.warning("embed_classifier disabled — %s", e)
        _state = False
    return _state


# Tuned against the v4 216-case eval. Start conservative; raise these
# if decoys over-route, lower them if paraphrases still slip through.
SCORE_THRESHOLD = 0.55  # min cosine for the top tool
MARGIN_THRESHOLD = 0.05  # top tool must beat #2 by this much


def embed_classify(text: str | None) -> Optional[tuple[str, float, float]]:
    """Return ``(tool, score, margin)`` if a tool's anchors semantically
    match the prompt well enough; ``None`` otherwise.

    Abstains when:
      * input is empty;
      * embedder failed to load;
      * top score is below :data:`SCORE_THRESHOLD`;
      * margin over runner-up is below :data:`MARGIN_THRESHOLD`.

    Returned ``score`` and ``margin`` are useful for logging and later
    threshold calibration.
    """
    if not text or not text.strip():
        return None
    state = _load()
    if not state:
        return None
    try:
        import numpy as np
        model, tool_vecs = state  # type: ignore[misc]
        q = np.array(list(model.embed([text])), dtype=np.float32)[0]
        qnorm = np.linalg.norm(q)
        if qnorm == 0:
            return None
        q = q / qnorm

        scores: list[tuple[str, float]] = []
        for tool, vecs in tool_vecs.items():
            # cosine max = max dot product (rows are pre-normalized)
            score = float((vecs @ q).max())
            scores.append((tool, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_tool, top_score = scores[0]
        runner_score = scores[1][1] if len(scores) > 1 else 0.0
        margin = top_score - runner_score

        if top_score < SCORE_THRESHOLD:
            return None
        if margin < MARGIN_THRESHOLD:
            return None
        return (top_tool, top_score, margin)
    except Exception as e:
        log.debug("embed_classify failed on %r: %s", (text or "")[:50], e)
        return None


def reload() -> None:
    """Force re-load of model + anchor embeddings. Call when ``_ANCHORS``
    changes at runtime or when tuning thresholds manually."""
    global _state
    _state = None
