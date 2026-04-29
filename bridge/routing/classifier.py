"""Data-driven pre-classifier for LLM tool routing.

Reads ``routing_patterns`` from each tool's schema in
``bridge.tools.TOOL_SCHEMAS``, compiles them once at first use, and
runs a single pass to find the unique winning tool. Adding a new tool
= adding ``routing_patterns`` to its schema entry. No separate file
to keep in sync.

Conventions:

* Each tool MAY have a ``routing_patterns`` list of regex source strings
  inside its ``function`` dict (next to ``name`` and ``description``).
  Tools without patterns are invisible to the classifier.
* Patterns are compiled at first call (lazy init) so import-time cost
  is zero. Case-insensitive.
* If two *different* tools both match, we abstain (return None) rather
  than guess. The LLM still has the full tool schema to pick from.
* Chitchat ("hi", "thanks", ...) is NOT a routing target. The chitchat
  guard in :mod:`bridge.llm` handles that before we get here.
"""
from __future__ import annotations

import re
from typing import Optional

_compiled: list[tuple[str, "re.Pattern[str]"]] | None = None


def _ensure_compiled() -> list[tuple[str, "re.Pattern[str]"]]:
    """Lazily compile patterns from TOOL_SCHEMAS on first call."""
    global _compiled
    if _compiled is not None:
        return _compiled

    from bridge.tools import TOOL_SCHEMAS

    pairs: list[tuple[str, "re.Pattern[str]"]] = []
    for schema in TOOL_SCHEMAS:
        fn = schema.get("function", {})
        name = fn.get("name", "")
        patterns = fn.get("routing_patterns", [])
        if not name or not patterns:
            continue
        for src in patterns:
            try:
                pairs.append((name, re.compile(src, re.IGNORECASE)))
            except re.error:
                pass
    _compiled = pairs
    return _compiled


def classify(user_text: str | None) -> str | None:
    """Return the single tool name a prompt clearly maps to, or None.

    Abstains when:
      * the prompt is empty;
      * no pattern matches;
      * patterns for *two different tools* match (ambiguous intent).

    When only one tool's patterns match (possibly multiple of its own
    patterns — that still counts as a unique winner), return that name.
    """
    if not user_text:
        return None
    text = user_text.strip()
    if not text:
        return None
    compiled = _ensure_compiled()
    hits: set[str] = set()
    for tool, pat in compiled:
        if pat.search(text):
            hits.add(tool)
    if not hits:
        return None
    # Explicit-action tools win over passive-mention conflicts: an
    # utterance like "remind me to check the gold price" mentions
    # both ``remind me`` (schedule_reminder) and ``gold price``
    # (tracker), which would otherwise abstain. The user's intent
    # is clearly the reminder — the gold reference is just the
    # REMINDER CONTENT. Same reasoning for explicit alarm/timer
    # verbs.
    _EXPLICIT_PRIORITY = ("schedule_reminder",)
    for name in _EXPLICIT_PRIORITY:
        if name in hits:
            return name
    # Facts priority: "tell me about 1985" hits both ``wiki`` (catch-
    # all subject pattern) and ``facts`` (year-card pattern). When the
    # subject is a bare 4-digit year, the user usually wants the
    # curated year-card, not the Wikipedia article. Runs first so the
    # population/inflation/money rules don't claim it either.
    if "facts" in hits and re.search(
        r"\btell\s+me\s+about\s+(?:the\s+year\s+)?\d{4}\b|"
        r"\b(?:fun|interesting|cool)\s+facts?\s+(?:about|from|of)\s+(?:the\s+year\s+)?\d{4}\b|"
        r"\b\d{4}\s+in\s+numbers\b|"
        r"\bsummary\s+of\s+(?:the\s+year\s+)?\d{4}\b|"
        r"\bwhat\s+was\s+\d{4}\s+like\b",
        text, flags=re.IGNORECASE,
    ):
        return "facts"
    # Population priority: "how many people lived in Japan in 1970",
    # "population of Hong Kong in 1990" hit both ``inflation``
    # (because the region-keyword inflation patterns are broad) and
    # ``population``. When the query has explicit demographic
    # keywords, population wins. Runs BEFORE the money+year rule.
    if "population" in hits and re.search(
        r"\bpopulation\b|\bpeople\s+(?:are|were|live[ds]?|lived|alive|in|of)\b|"
        r"\bhow\s+many\s+(?:people|inhabitants|residents|citizens)\b|"
        r"\binhabitants\b|\bdemograph(?:y|ic|ics)\b",
        text, flags=re.IGNORECASE,
    ):
        return "population"
    # FX-pair priority: "USD/JPY", "eur/usd" hit both ``tracker`` (FX
    # tickers) and ``inflation`` (bare currency tokens like "jpy").
    # The slash-separated shape is unambiguous — that's an FX quote
    # request, not a purchasing-power query. Runs before the inflation
    # priority so the bare-token rule can't claim it.
    if "tracker" in hits and re.search(
        r"\b(?:usd|eur|jpy|gbp|cny|hkd|aud|cad|chf|krw)\s*/\s*"
        r"(?:usd|eur|jpy|gbp|cny|hkd|aud|cad|chf|krw)\b",
        text, flags=re.IGNORECASE,
    ):
        return "tracker"
    # Currency-convert priority: "100 usd to jpy", "convert 50 eur to
    # gbp" hit both ``convert`` (numeric + currency-code-to-code) and
    # ``inflation`` (bare currency tokens). The number-then-currency-
    # then-`to`-then-currency shape is unambiguously a conversion.
    if "convert" in hits and re.search(
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:usd|eur|jpy|gbp|cny|hkd|aud|cad|chf|krw|twd)"
        r"\s+(?:to|in)\s+"
        r"(?:usd|eur|jpy|gbp|cny|hkd|aud|cad|chf|krw|twd)\b",
        text, flags=re.IGNORECASE,
    ):
        return "convert"
    # Money + year priority: "how much was a dollar in 1865 worth
    # today" hits both ``wiki`` (historical-cost shape) and
    # ``inflation`` (purchasing-power shape). The user wants the
    # actual number, not a Wikipedia article — inflation has the
    # data. Runs BEFORE the year-anchored wiki rule below so it
    # captures these cases first. Region names (Hong Kong / Japan /
    # South Korea) are strong inflation cues by themselves; foreign
    # currency tokens (yuan/yen/won/HKD/JPY) require a year or
    # inflation cue alongside, otherwise FX pairs like "USD/JPY" and
    # currency-convert prompts like "100 usd to jpy" wrongly route
    # here instead of to ``tracker`` / ``convert``.
    if "inflation" in hits and re.search(
        r"\$|\bdollars?\b|\bbucks?\b|"
        r"\bpurchasing\s+power\b|\binflation\b|\b(?:wage|salary|paycheck|earnings)s?\b|"
        r"\bhong\s*kong\b|\bmainland\s+china\b|\bjapanese\b|\bsouth\s+korea(?:n)?\b",
        text, flags=re.IGNORECASE,
    ):
        return "inflation"
    if "inflation" in hits and re.search(
        r"\b(?:yuan|yen|won|hkd|cny|jpy|krw|hk\$|rmb|renminbi)\b",
        text, flags=re.IGNORECASE,
    ) and re.search(
        r"\b\d{4}\b|\binflation\b|\bworth\s+(?:today|now)\b|\bpurchasing\s+power\b",
        text, flags=re.IGNORECASE,
    ):
        return "inflation"
    # Year-anchored priority: "what was the price of gas in 1980"
    # matches both ``tracker`` (commodity keyword) and ``wiki``
    # (historical-year pattern). When a 4-digit year is present
    # alongside a wiki match, the user wants historical context;
    # tracker's current-price view doesn't satisfy that. Same for
    # "in 1960s" decade form.
    if "wiki" in hits and re.search(r"\b\d{4}s?\b", text):
        return "wiki"
    # Alert-priority: "any severe weather warnings in my area" hits
    # both get_weather (on "weather") and get_alerts (on "warnings"
    # / "severe weather"). Alerts win when the prompt is clearly
    # asking about warnings/advisories, otherwise weather would
    # swallow every alert query.
    if "get_alerts" in hits and "get_weather" in hits and re.search(
        r"\b(warning|warnings|alert|alerts|advisory|advisories|hazard|hazards)\b",
        text, flags=re.IGNORECASE,
    ):
        return "get_alerts"
    # Compound "A and B" resolver: when two different tools both match
    # and the prompt connects them with "and", fire the tool whose
    # pattern hits EARLIEST in the prompt (treated as the primary
    # intent). Catches "weather and news" → get_weather, "convert X
    # and the news" → convert, "weather and any alerts" → get_weather.
    # Legitimate single-tool "A and B" phrasings (e.g. "weather in
    # Paris and London" — both are get_weather) only match one tool,
    # so this branch never runs on them.
    if len(hits) >= 2 and re.search(r"\band\b", text, flags=re.IGNORECASE):
        earliest_pos = None
        earliest_tool = None
        for tool, pat in compiled:
            if tool not in hits:
                continue
            m = pat.search(text)
            if m is None:
                continue
            if earliest_pos is None or m.start() < earliest_pos:
                earliest_pos = m.start()
                earliest_tool = tool
        if earliest_tool is not None:
            return earliest_tool
    if len(hits) == 1:
        return next(iter(hits))
    return None


# Continuation prompts — short follow-ups without enough lexical
# content for the regex classifier to fire, but which clearly depend
# on the prior turn. When we spot one AND the prior turn called a
# tool, the hint falls back to that tool.
_CONTINUATION_PATTERNS = [
    re.compile(r"^\s*(how\s+about|what\s+about|and)\b", re.IGNORECASE),
    re.compile(r"^\s*(no\s+wait|actually|cancel\s+that|wait)\b", re.IGNORECASE),
    # Short pronoun-led fragments: "there", "it", "they" as subject
    re.compile(r"^\s*(there|it|they|he|she)\b", re.IGNORECASE),
    # Topic-switch follow-ups like "Anything on tech?", "More about Japan".
    # Requires the on/about/for anchor so bare "More" / "Anything" doesn't
    # over-trigger on fresh prompts.
    re.compile(r"^\s*(anything|anymore|more)\s+(on|about|for)\b", re.IGNORECASE),
    # Very short prompts (3 words or fewer, no subject-verb)
]


def _looks_like_continuation(text: str) -> bool:
    """True if the prompt looks like a follow-up riding on prior context."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Short prompts are usually follow-ups
    word_count = len(stripped.split())
    if word_count <= 3:
        return True
    for pat in _CONTINUATION_PATTERNS:
        if pat.match(stripped):
            return True
    return False


# Patterns that indicate the prior assistant turn asked for clarification
# scoped to a specific tool. Each tuple is (tool_name, pattern). When the
# prior assistant reply matches AND the current user message is short
# (likely the answer), the turn routes to that tool — even if the
# classifier doesn't hit on the current message in isolation.
#
# Why this exists: if the assistant asks "what's your household
# composition?" and the user replies "Single, 1 person", the classifier
# sees "Single, 1 person" with no match and the LoRA may misroute (e.g.
# to get_time on "Single, 1 person"). This layer reads the prior reply
# to recover the lost tool intent.
#
# Patterns are conservative — only match phrasings that ONLY appear in
# tool-clarification questions, not generic prose. False positives just
# nudge the routing toward a specific tool; the rescue's per-tool arg
# extractor still has the final say (declines on bad/missing args).
_CLARIFICATION_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    (
        "alice",
        re.compile(
            r"(?:household\s+composition|"
            r"single\s+adult\s+or\s+(?:family|couple)|"
            r"how\s+many\s+(?:adults?|kids?|people|persons?|in\s+the\s+household)|"
            r"(?:specify|provide|tell\s+me)\s+(?:how\s+many|the\s+composition)|"
            r"compute\s+ALICE)",
            re.IGNORECASE,
        ),
    ),
    (
        "get_weather",
        re.compile(
            r"(?:which\s+(?:city|location)|where\s+are\s+you|"
            r"what\s+(?:city|location)|need\s+(?:a\s+)?location)",
            re.IGNORECASE,
        ),
    ),
    (
        "get_time",
        re.compile(
            r"(?:which\s+(?:time\s*zone|timezone)|"
            r"what\s+time\s*zone|need\s+(?:a\s+)?time\s*zone)",
            re.IGNORECASE,
        ),
    ),
    (
        "wiki",
        re.compile(
            r"(?:more\s+specific|do\s+you\s+mean|narrow\s+down|which\s+\w+\s+do\s+you\s+mean)",
            re.IGNORECASE,
        ),
    ),
    (
        "inflation",
        re.compile(
            r"(?:from\s+(?:which|what)\s+year|need\s+(?:a\s+)?(?:base\s+)?year|"
            r"how\s+much\s+(?:in|was)\s+\$?\d|amount\s+to\s+convert)",
            re.IGNORECASE,
        ),
    ),
]


def detect_clarification_followup(prior_assistant_reply: str | None) -> str | None:
    """If the most recent assistant turn asked for tool-scoped clarification,
    return the tool name. Used as a routing layer between the explicit
    classifier hit and the continuation fallback — when the prior turn
    asked for ALICE composition / weather location / etc., the next short
    user message is the answer to that clarification.

    Returns ``None`` when no clarification pattern matches.
    """
    if not prior_assistant_reply:
        return None
    for tool, pattern in _CLARIFICATION_PATTERNS:
        if pattern.search(prior_assistant_reply):
            return tool
    return None


def _last_assistant_text(history: list[dict] | None) -> str:
    """Return the most recent assistant reply text, or empty string."""
    if not history:
        return ""
    snapshot = list(history)
    for msg in reversed(snapshot):
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def _last_assistant_tool_call(history: list[dict] | None) -> str | None:
    """Scan chat history backwards for the most recent assistant
    ``tool_calls`` entry and return that tool's name. Walks past
    prose-only assistant turns — a typical tool-using turn has both
    a tool-call message AND a final prose message, and we want the
    underlying tool from the prior turn, not the prose wrapper.

    Returns None if no prior tool call is found in recent history.
    """
    if not history:
        return None
    # Materialize once so we don't re-acquire the HistorySession lock
    # for each indexed access during reversed() iteration. Plain lists
    # round-trip as themselves; HistorySession returns its snapshot.
    snapshot = list(history)
    for msg in reversed(snapshot):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # Prose-only assistant turn — skip past it and keep looking
            # for a tool-calling turn further back.
            continue
        fn = tool_calls[0].get("function") or {}
        name = fn.get("name")
        if name:
            return str(name)
    return None


def routing_hint(user_text: str | None, history: list[dict] | None = None) -> str:
    """Return a one-line hint to append to the system prompt, or ``""``.

    When the classifier has a confident answer, the hint tells the LLM
    which tool to prefer. When ambiguous or no match, returns empty so
    the prompt doesn't contain a misleading suggestion.

    Wording note: the original "most likely maps to" phrasing was too
    soft for some de-aligned / abliterated fine-tunes (observed: they'd
    read the hint and still answer from memory without calling the
    tool). The escape clause "unless the prompt is clearly unrelated"
    keeps stock, instruction-tuned models from over-routing on edge
    cases while nudging the less-compliant tunes to actually call.

    spaCy veto: before emitting the hint, we run a structural check
    on the prompt. If the dep parse shows a negation directly on the
    tool's trigger verb ("don't remind me"), we suppress the hint so
    the LLM isn't pushed toward a tool the user is explicitly
    declining. Failure-safe: if spaCy is unavailable, the veto is a
    no-op.

    Multi-turn context (``history`` param): when the regex classifier
    abstains AND the prompt looks like a continuation ("And silver?",
    "How about San Francisco?"), we carry forward whichever tool was
    called on the most recent tool-calling assistant turn. This fixes
    the pronoun/follow-up multi-turn failures without needing the LLM
    to reason about state.
    """
    tool = classify(user_text)

    if not tool:
        # Continuation fallback — the regex didn't fire, but this
        # prompt is clearly a follow-up. Offer the prior turn's tool
        # as a gentle suggestion. Runs BEFORE the embedding fallback
        # so short pronoun-led prompts ("And silver?") still ride on
        # history rather than getting misrouted by a thin anchor match.
        if _looks_like_continuation(user_text or "") and history:
            prior = _last_assistant_tool_call(history)
            if prior:
                return (
                    f"[routing continuation hint: the prior turn called "
                    f"`{prior}`; this short follow-up most likely wants "
                    f"the same tool again with new arguments.]"
                )

        # Embedding fallback — catches paraphrased prompts that the
        # regex doesn't cover ("Where's a decent ramen spot?",
        # "DIY bookshelf plans"). Abstains when no tool clears the
        # score/margin thresholds; silent no-op if the embedder is
        # unavailable.
        try:
            from bridge.routing.embed_classifier import embed_classify
            hit = embed_classify(user_text or "")
            if hit:
                embed_tool, _score, _margin = hit
                return (
                    f"[routing hint (semantic): this prompt looks like "
                    f"a `{embed_tool}` query — prefer calling it unless "
                    f"the prompt clearly wants something else.]"
                )
        except Exception:
            pass
        return ""

    # spaCy structural veto: catches "don't remind me" / "don't worry
    # about the weather" style prompts where the regex matched but
    # the user's intent is the opposite.
    try:
        from bridge.routing.spacy_filter import should_veto
        if should_veto(user_text or "", tool):
            return ""
    except Exception:
        # Filter is a best-effort layer; never let it break routing.
        pass
    return (
        f"[routing rule: this prompt strongly matches the `{tool}` "
        f"tool — call it unless the prompt is clearly unrelated.]"
    )


def reload() -> None:
    """Force recompilation of patterns. Called when TOOL_SCHEMAS changes
    at runtime (e.g. feature flags toggling tools on/off)."""
    global _compiled
    _compiled = None
