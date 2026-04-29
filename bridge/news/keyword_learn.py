"""LLM-assisted keyword extraction for watchlist-aware news.

Given a batch of recent headlines + summaries tagged with a watchlist
bucket (region / topic / event), ask Qwen-via-Ollama to name the
entities worth watching — people, organizations, places, events.
Stores them in `learned_store` so future calls can merge with existing
state + age them out via TTL.

Design rules:
  - Deterministic: temperature=0, strict prompt, JSON-only response.
  - Bounded: caller caps how many LLM calls happen per cycle, and the
    prompt imposes a 5-10 entity limit per request.
  - Fail-soft: any Ollama error returns an empty entity list; the
    news pipeline never blocks on a flaky LLM.
  - Qwen-tuned: uses names not BCP-47 codes ("Mainland China", not
    "zh"), and `think: False` at the request level to dodge Qwen3's
    verbose thinking mode (see memory: Ollama `think` is top-level).

Public API:
    extract_entities(texts, bucket_label, *, base_url, model,
                     request_timeout=60.0) -> list[LearnedEntity]
    LearnedEntity is a frozen dataclass with label + kind + source
    article count.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable

import httpx

log = logging.getLogger("bridge.news.keyword_learn")


# --- data types -----------------------------------------------------------

@dataclass(frozen=True)
class LearnedEntity:
    """One entity extracted from a batch of articles.

    `kind` is one of "person" / "organization" / "place" / "event" /
    "other". The bucket (regions / topics / events) is tracked by the
    caller — entity-level kind is orthogonal and helps the UI group
    results ("who", "what", "where") inside a single bucket view.
    """
    label: str      # canonical display form, e.g. "ASML"
    kind: str       # "person" | "organization" | "place" | "event" | "other"
    confidence: int = 0   # 0-100, from the LLM's own estimate; 0 if absent


# --- prompt ---------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You extract recurring named entities from news text. "
    "You output ONLY a JSON array — no prose, no markdown fences, no "
    "commentary. Each array item is an object with exactly three keys: "
    "\"label\" (the entity name as it should be shown), \"kind\" (one "
    "of \"person\", \"organization\", \"place\", \"event\", \"other\"), "
    "and \"confidence\" (integer 0-100). Return between 3 and 10 items. "
    "Prefer specific entities (people, companies, places) over generic "
    "nouns (\"economy\", \"government\"). Deduplicate case-insensitively."
)

_USER_TEMPLATE = (
    "Bucket: {bucket}\n\n"
    "News fragments (headline + summary per story, one per line):\n\n"
    "{corpus}\n\n"
    "List the most recurring entities from these fragments as a JSON "
    "array. Output only the array."
)

# Cap the corpus we send to the LLM. Qwen 3.5 4B has a 4k context,
# shared with the system prompt + options. 8000 chars ~ 2000 tokens,
# leaving headroom for the prompt + response. Empirically enough to
# capture recurrence without hitting context limits on long summaries.
_MAX_CORPUS_CHARS = 8000


def _build_corpus(texts: Iterable[str]) -> str:
    """Join fragments one-per-line, clipped to _MAX_CORPUS_CHARS.

    Frontloads the input since the most recent / relevant articles
    should be at the top of the caller's list. Dropping the tail when
    we run over is safer than truncating mid-item — a half-article
    confuses the model more than a missing one.
    """
    out: list[str] = []
    used = 0
    for t in texts:
        clean = (t or "").strip().replace("\n", " ")
        if not clean:
            continue
        # +1 for the newline separator. If adding this line would
        # exceed the budget, stop — better to send 40 complete stories
        # than 41 with the last one cut.
        needed = len(clean) + 1
        if used + needed > _MAX_CORPUS_CHARS:
            break
        out.append(clean)
        used += needed
    return "\n".join(out)


def _build_messages(corpus: str, bucket_label: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(
            bucket=bucket_label, corpus=corpus,
        )},
    ]


# --- JSON parsing (defensive) ---------------------------------------------

# Strip common wrapper noise Qwen occasionally emits despite the
# "ONLY a JSON array" rule: code fences, prose preamble ("Here's the
# list:"), trailing commentary.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


def _extract_json_array(raw: str) -> list[dict]:
    """Pull a JSON array out of raw LLM output.

    Tries strict parse first; if that fails, greedily locate the first
    ``[`` and last ``]`` and parse the slice between. Unknown / missing
    keys are tolerated — `_coerce_entity` handles the per-item shape.
    """
    text = _CODE_FENCE_RE.sub("", raw.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"no JSON array in output: {text[:120]!r}")
        parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}")
    # Only keep dict items. The LLM occasionally emits bare strings
    # (["ASML", "Nvidia"]) — rescue those with label-only coercion.
    out: list[dict] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"label": item, "kind": "other", "confidence": 50})
    return out


_VALID_KINDS = frozenset({"person", "organization", "place", "event", "other"})


def _coerce_entity(raw: dict) -> LearnedEntity | None:
    """Normalize one entity dict. Returns None on missing / empty label.

    Unknown kinds coerce to "other"; non-int confidence clamps to 0.
    Labels get whitespace-collapsed + NFKC-free basic cleanup so the
    dedup in learned_store doesn't see "ASML " and "ASML" as
    different.
    """
    label = str(raw.get("label", "")).strip()
    if not label:
        return None
    label = re.sub(r"\s+", " ", label)
    kind = str(raw.get("kind", "other")).strip().lower()
    if kind not in _VALID_KINDS:
        kind = "other"
    try:
        conf = int(raw.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0
    if conf < 0:
        conf = 0
    if conf > 100:
        conf = 100
    return LearnedEntity(label=label, kind=kind, confidence=conf)


# --- Ollama call ----------------------------------------------------------

def _call_ollama(
    messages: list[dict],
    *,
    base_url: str,
    model: str,
    request_timeout: float,
) -> str:
    """POST to /api/chat and return the assistant content.

    Mirrors the pattern in bridge.news.translate._call_ollama so both
    LLM entry points share the same `think: False` + temperature=0
    guarantees (see memory: Ollama `think` is a top-level field, not
    an options key).
    """
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 3072,
            "num_predict": 512,
        },
    }
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=request_timeout) as client:
        resp = client.post("/api/chat", json=body)
        resp.raise_for_status()
        payload = resp.json()
    msg = payload.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"empty Ollama response: {payload!r}")
    return content


# --- public API -----------------------------------------------------------

def extract_entities(
    texts: list[str],
    bucket_label: str,
    *,
    base_url: str,
    model: str,
    request_timeout: float = 60.0,
) -> list[LearnedEntity]:
    """Run the LLM on a batch of fragments and return named entities.

    Args:
        texts: Fragments (usually "headline — summary") for stories in
            one watchlist bucket. Order matters — earlier fragments
            are prioritised when trimming to the context cap.
        bucket_label: The watchlist bucket name ("US", "Mainland China",
            "AI / Tech", etc.). Injected into the prompt so the LLM
            knows the scope. Not used for anything else.
        base_url: Ollama HTTP base, e.g. http://localhost:11434.
        model: Ollama model tag. Qwen-family recommended — entity
            naming is a strong Qwen suit.
        request_timeout: HTTP timeout per call. Extraction is a little
            heavier than translation, so 60 s default is conservative.

    Returns:
        List of LearnedEntity. Empty on any error path — the news
        pipeline upstream should treat an empty list as "skip this
        bucket this cycle", not as a hard failure.
    """
    corpus = _build_corpus(texts)
    if not corpus:
        return []
    try:
        raw = _call_ollama(
            _build_messages(corpus, bucket_label),
            base_url=base_url, model=model,
            request_timeout=request_timeout,
        )
    except (httpx.HTTPError, ValueError) as e:
        log.warning("keyword_learn: LLM call failed for bucket %r: %s", bucket_label, e)
        return []

    try:
        items = _extract_json_array(raw)
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(
            "keyword_learn: failed to parse LLM output for bucket %r: %s",
            bucket_label, e,
        )
        return []

    out: list[LearnedEntity] = []
    seen: set[str] = set()
    for item in items:
        ent = _coerce_entity(item)
        if ent is None:
            continue
        key = ent.label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
    return out
