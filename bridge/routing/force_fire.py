"""Under-fire rescue: heuristic argument inference for force-called tools.

When the LLM responds to a user turn without emitting a tool call, but the
regex classifier is confident about which tool SHOULD have fired, the chat
loop in :mod:`bridge.llm` asks this module for default arguments to feed
into that tool. We then execute the tool, inject the result as a
``role=tool`` message, and let the LLM produce a final reply using the
data it refused to fetch on its own.

Design constraints:

* Only safe when the tool tolerates noisy args (``wiki`` searches
  keywords; ``web_search`` takes any string). Tools that need precise
  args (``convert`` value+units, ``schedule_reminder`` trigger_at,
  ``update_memory`` fact) are NOT rescuable here — guessing their
  args wrong is worse than under-firing.
* Every extractor must fail gracefully: return ``None`` from
  :func:`default_args` when the tool isn't rescuable or the prompt is
  too ambiguous. ``None`` tells the chat loop to skip rescue for this
  turn.
* Keep dependencies minimal — just ``re`` — so this is cheap to call
  on every under-fire and never introduces an import cycle.

Integration: see :meth:`bridge.llm.OllamaLLM._maybe_force_fire_rescue`.
"""
from __future__ import annotations

import re


# Tools where force-firing with best-effort args is acceptable. Any tool
# NOT in this set returns None from default_args() and is skipped.
FORCE_RESCUE_TOOLS: frozenset[str] = frozenset({
    "wiki",
    "get_weather",
    "get_time",
    "get_news",
    "get_alerts",
    "get_digest",
    "find_places",
    "web_search",
    # math is only rescuable for the narrow closed-form distribution ops
    # (kl_dist / entropy_dist). General math is handled by the LoRA; if
    # _extract_math_args can't find an unambiguous distribution shorthand
    # it returns None and this rescue is declined.
    "math",
    # inflation: iter-3 was trained without this tool, so it under-fires
    # consistently on "dollar in YYYY" / "Hong Kong CPI" / wage queries.
    # extract_inflation_args() produces a deterministic arg set from the
    # user text; if it returns nothing useful (no from_year), the rescue
    # is declined.
    "inflation",
    # population: also new (post iter-3), same under-fire pattern.
    # extract_population_args() builds the args; if it can't extract
    # at least region or year, the rescue declines.
    "population",
    # facts: year-card aggregator, also post-iter-3. extract_facts_args
    # requires a year; if absent, decline.
    "facts",
    # alice: post-iter-3 too. Both args are optional (year defaults to
    # latest, household_size defaults to 4) so an empty extraction is
    # still a valid call. Force-rescue declines if there's no
    # ALICE-related keyword in the text — see default_args() below.
    "alice",
    # say: passthrough echo. Force-rescue when the user explicitly asks
    # to repeat a phrase but the LoRA picks "no tool" — extractor
    # produces the verbatim text from the same anchored regex set the
    # bridge uses post-fire. If extraction returns None, decline.
    "say",
})


# Subject extractor for wiki: strip the interrogative prefix (`who was`,
# `what is`, `when did X happen`, `how much did X cost`...) and use the
# remainder as the search query. Wiki's server-side search tolerates
# noise, so we don't need a perfect parse — just a reasonable keyword
# phrase.
_WIKI_PREFIX = re.compile(
    r"^\s*(?:"
    r"who\s+(?:is|was|were)\s+|"
    r"what\s+(?:is|was|are|were|year|about)\s+|"
    r"when\s+(?:is|was|did|were)\s+|"
    r"how\s+(?:much|many|old|long)\s+(?:is|was|were|did|ago|do|does)?\s*|"
    r"tell\s+me\s+about\s+|"
    r"look\s+up\s+|"
    r"wikipedia\s+(?:on|for)?\s*|"
    r"history\s+of\s+|"
    r"origin\s+of\s+|"
    r"meaning\s+of\s+"
    r")",
    flags=re.IGNORECASE,
)

# Matches "in 1980", "in 1960s" trailing clauses — strip them so the
# wiki search is on the subject, not the year.
_TRAILING_YEAR = re.compile(r"\bin\s+\d{4}s?\b[\s,.!?]*$", flags=re.IGNORECASE)

# Matches "in <Location>" at end of a weather/time prompt.
_LOCATION_TRAILING = re.compile(
    r"\b(?:in|at|for)\s+([A-Z][\w\s,.'-]*?)\s*[.?!]?\s*$",
)

# News topic extractor: "news on tech", "headlines about Japan".
_NEWS_TOPIC = re.compile(
    r"\b(?:news|headlines?|stories)\s+(?:on|about|for|regarding)\s+([\w\s]+?)\s*[.?!]?\s*$",
    flags=re.IGNORECASE,
)

# Find-places: "coffee shop near me", "pharmacy nearby". Capture the
# noun phrase before the near/nearby marker.
_FIND_PLACES_QUERY = re.compile(
    r"(?:best\s+|nearest\s+|find\s+(?:a\s+|an\s+|me\s+(?:a\s+|an\s+)?)?|where\s+(?:can\s+i\s+(?:find|get|buy)\s+))?"
    r"(?:(?:a|an|the)\s+)?"
    r"([\w\s-]+?)\s*"
    r"(?:(?:near\s+me|nearby|close\s+by|around\s+here)|$)",
    flags=re.IGNORECASE,
)


def _clip(s: str, limit: int = 120) -> str:
    """Trim and cap whitespace/length. Noisy args still work for the
    server-side-search tools, but an enormous blob is wasteful."""
    s = (s or "").strip().rstrip("?.!,;")
    return s[:limit].strip()


def _extract_wiki_subject(user_text: str) -> str:
    """Best-effort extraction of the thing to search on wiki.

    Strips interrogative prefix and trailing "in YYYY" anchors, returns
    the surviving subject phrase. Falls back to the full prompt (clipped)
    if the regex doesn't find a recognizable prefix — wiki's search is
    forgiving enough that even "How much was a house in 1960s America?"
    → "a house in 1960s America" → wiki search returns useful results.
    """
    text = (user_text or "").strip()
    # Drop interrogative prefix
    stripped = _WIKI_PREFIX.sub("", text, count=1)
    # Drop trailing year
    stripped = _TRAILING_YEAR.sub("", stripped)
    # Final cleanup of stopwords at the start
    stripped = re.sub(r"^(a|an|the)\s+", "", stripped, flags=re.IGNORECASE)
    return _clip(stripped) or _clip(text)


def _extract_location(user_text: str) -> str:
    """Pull a location out of a weather/time prompt. Returns "" when
    no location phrase is found — tool callers use the user's default."""
    m = _LOCATION_TRAILING.search(user_text or "")
    return _clip(m.group(1)) if m else ""


def _extract_news_topic(user_text: str) -> str | None:
    """Topic for get_news, or None for the catch-all headlines feed."""
    m = _NEWS_TOPIC.search(user_text or "")
    return _clip(m.group(1)) if m else None


# Closed-form distribution names accepted by math's kl_dist/entropy_dist
# (kept in sync with bridge.tools._math._DIST_ALIASES). Matching is
# case-insensitive via the (?i:…) atomic group.
_DIST_NAMES_RE = (
    r"(?i:N|Normal|Gaussian|Exp|Exponential|Beta|Bern|Bernoulli|Uniform|U)"
)
# Single distribution call, e.g. ``N(0, 1)`` or ``Beta(2,5)``.
_DIST_CALL_RE = re.compile(rf"{_DIST_NAMES_RE}\s*\(\s*[^)]*\s*\)")
# Two distribution calls joined by ``||`` / ``vs`` / ``against``, which is
# the canonical KL-divergence shorthand the math tool accepts.
_KL_PAIR_SEP_RE = re.compile(
    rf"({_DIST_NAMES_RE}\s*\(\s*[^)]*\s*\))"
    r"\s*(?:\|\||vs\.?|against)\s*"
    rf"({_DIST_NAMES_RE}\s*\(\s*[^)]*\s*\))"
)
# "KL divergence between <dist> and <dist>" long form.
_KL_BETWEEN_RE = re.compile(
    r"\bkl\s+(?:divergence\s+)?(?:between|of)\s+"
    rf"({_DIST_NAMES_RE}\s*\(\s*[^)]*\s*\))"
    r"\s*(?:,|and)\s*"
    rf"({_DIST_NAMES_RE}\s*\(\s*[^)]*\s*\))",
    flags=re.IGNORECASE,
)
# Entropy long form: "entropy of <dist>" or "H(<dist>)".
_ENTROPY_OF_RE = re.compile(
    r"\bentropy\s+of\s+(?:a\s+|the\s+)?"
    rf"({_DIST_NAMES_RE}\s*\(\s*[^)]*\s*\))",
    flags=re.IGNORECASE,
)
_ENTROPY_H_RE = re.compile(
    rf"\bH\s*\(\s*({_DIST_NAMES_RE}\s*\(\s*[^)]*\s*\))\s*\)"
)
# Vector brackets anywhere in the prompt disqualify distribution rescue —
# "KL between N(0,1) and [0.9, 0.1]" is a cross-type request math can't
# compute with kl_dist. Let the LoRA respond conversationally.
_HAS_VECTOR_RE = re.compile(r"\[[^\[\]]*\]")


def _extract_math_args(user_text: str) -> dict | None:
    """Return args for math's kl_dist/entropy_dist ops when the prompt
    unambiguously uses distribution shorthand, else None.

    We're intentionally narrow: math is otherwise the LoRA's turf and
    over-forcing would override competent symbolic work. Only fire on
    shorthand like ``N(0,1) || N(1,2)`` / ``entropy of Beta(2,5)`` /
    ``H(N(0,1))`` / ``KL divergence between X and Y``.
    """
    text = (user_text or "").strip()
    if not text:
        return None
    # Mixed continuous+discrete (vector + distribution) is out of scope —
    # kl_dist can't handle cross-type pairs and kl_divergence wants two
    # vectors. Return None and let the LoRA explain.
    if _HAS_VECTOR_RE.search(text):
        return None

    m = _KL_BETWEEN_RE.search(text)
    if m:
        a, b = m.group(1), m.group(2)
        return {"op": "kl_dist", "expression": f"{a} || {b}"}

    m = _KL_PAIR_SEP_RE.search(text)
    if m:
        a, b = m.group(1), m.group(2)
        return {"op": "kl_dist", "expression": f"{a} || {b}"}

    m = _ENTROPY_H_RE.search(text)
    if m:
        return {"op": "entropy_dist", "expression": m.group(1)}

    m = _ENTROPY_OF_RE.search(text)
    if m:
        return {"op": "entropy_dist", "expression": m.group(1)}

    return None


def _extract_find_places_query(user_text: str) -> str:
    """What to look for; falls back to the trimmed prompt."""
    m = _FIND_PLACES_QUERY.search(user_text or "")
    if m:
        q = _clip(m.group(1))
        if q:
            return q
    return _clip(user_text, limit=60)


# Continuation prefix: the wrapper around a follow-up's payload. We
# strip these before feeding the remainder to tool args, so
# ``How about Chicago?`` → ``Chicago``, ``Anything on tech?`` → ``tech``.
# Order matters — longer/more-specific forms first so we don't greedily
# eat "actually" from "actually, try X instead".
_CONT_PREFIX = re.compile(
    r"^\s*(?:"
    r"how\s+about\s+|"
    r"what\s+about\s+|"
    r"and\s+(?:in\s+|also\s+to\s+|to\s+|for\s+|on\s+)?|"
    r"also\s+(?:in\s+|for\s+|on\s+)?|"
    r"actually,?\s+(?:i\s+meant\s+)?(?:try\s+)?|"
    r"wait,?\s+(?:make\s+it\s+)?|"
    r"scratch\s+that,?\s+|"
    r"try\s+|"
    r"(?:anything|anymore|more)\s+(?:on|about|for)\s+"
    r")",
    flags=re.IGNORECASE,
)

# Continuation suffix noise — drop trailing hedges so they don't
# pollute the payload. "tech specifically" → "tech".
_CONT_SUFFIX = re.compile(
    r"\s+(?:instead|specifically|too|also|please)\s*[.?!]*\s*$",
    flags=re.IGNORECASE,
)


def _continuation_payload(user_text: str) -> str:
    """Strip the continuation wrapper and return the core noun phrase."""
    text = (user_text or "").strip()
    if not text:
        return ""
    text = _CONT_PREFIX.sub("", text, count=1)
    text = _CONT_SUFFIX.sub("", text)
    return _clip(text)


def default_args_continuation(tool: str, user_text: str) -> dict | None:
    """Arg extractor for short continuation prompts re-using the prior tool.

    Unlike :func:`default_args`, this treats the user text as a follow-up
    whose payload (after stripping "how about" / "and" / "actually" etc.)
    is the sole substantive content. That payload drops directly into the
    tool's primary arg — no need for "in/at/for" anchoring.

    Also covers the clarification-followup path (when the prior assistant
    turn asked for tool-scoped info like "what's your household
    composition?" and the user is answering): tools that gate their
    main `default_args` behind a keyword guard re-route here so the
    guard doesn't reject the answer ("Single, 1 person" has no ALICE
    keyword but IS a valid composition).

    Returns ``None`` when the tool isn't force-safe or the payload is empty
    (e.g. a bare "how about?" with no content).
    """
    if tool not in FORCE_RESCUE_TOOLS:
        return None
    payload = _continuation_payload(user_text)
    if not payload:
        return None
    if tool == "get_weather":
        return {"location": payload}
    if tool == "get_time":
        return {"timezone": payload}
    if tool == "get_news":
        return {"topic": payload}
    if tool == "wiki":
        return {"query": payload}
    if tool == "find_places":
        return {"query": payload}
    if tool == "web_search":
        return {"query": payload}
    if tool in ("get_alerts", "get_digest"):
        return {}
    if tool == "alice":
        # Clarification-followup path: user is answering "what's your
        # household composition?". Extract directly from the original
        # user_text (not the stripped payload — composition phrases
        # like "Single, 1 person" survive without continuation
        # markers). Empty extraction is still a valid call (alice has
        # sensible defaults), so don't decline on no match.
        from bridge.tools._alice import extract_alice_args
        return extract_alice_args(user_text) or {}
    if tool == "inflation":
        from bridge.tools._inflation import extract_inflation_args
        ext = extract_inflation_args(user_text)
        # Inflation needs a from_year to be meaningful; decline if the
        # follow-up answer didn't include one.
        if not ext.get("from_year"):
            return None
        return ext
    if tool == "facts":
        from bridge.tools._facts import extract_facts_args
        ext = extract_facts_args(user_text)
        if not ext.get("year"):
            return None
        return ext
    if tool == "population":
        from bridge.tools._population import extract_population_args
        return extract_population_args(user_text) or {}
    return None


def default_args(
    tool: str,
    user_text: str,
    history: list[dict] | None = None,   # reserved for future multi-turn use
) -> dict | None:
    """Return best-effort arguments for a force-called tool, or None to
    skip the rescue.

    ``None`` means "this tool is not force-safe, or we have no reasonable
    argument to supply". The caller should fall back to returning the
    LLM's original no-tool reply.
    """
    if tool not in FORCE_RESCUE_TOOLS:
        return None

    text = (user_text or "").strip()
    if not text:
        return None

    if tool == "wiki":
        return {"query": _extract_wiki_subject(text)}
    if tool == "get_weather":
        return {"location": _extract_location(text)}
    if tool == "get_time":
        loc = _extract_location(text)
        return {"timezone": loc} if loc else {}
    if tool == "get_news":
        topic = _extract_news_topic(text)
        return {"topic": topic} if topic else {}
    if tool in ("get_alerts", "get_digest"):
        return {}
    if tool == "find_places":
        return {"query": _extract_find_places_query(text)}
    if tool == "web_search":
        return {"query": _clip(text)}
    if tool == "math":
        return _extract_math_args(text)
    if tool == "inflation":
        # Use the inflation tool's own extractor — it produces the
        # canonical args (amount/from_year/to_year/measure/item/region)
        # the tool expects. If it can't find a from_year, decline (the
        # LoRA's no-tool reply is more honest than calling with junk).
        from bridge.tools._inflation import extract_inflation_args
        ext = extract_inflation_args(text)
        if not ext.get("from_year"):
            return None
        return ext
    if tool == "population":
        from bridge.tools._population import extract_population_args
        ext = extract_population_args(text)
        # Population tool defaults region="world" and year=latest, so
        # an empty extraction is still valid ("world population today").
        # But if there's NO population/people/world keyword in the text,
        # better to decline than fire on an unrelated query.
        if not ext and not re.search(
            r"\b(?:people|population|inhabitants|residents|alive|live[ds]?)\b",
            text, re.IGNORECASE,
        ):
            return None
        return ext
    if tool == "facts":
        from bridge.tools._facts import extract_facts_args
        ext = extract_facts_args(text)
        # facts requires a year — decline without one.
        if not ext.get("year"):
            return None
        return ext
    if tool == "alice":
        from bridge.tools._alice import extract_alice_args
        ext = extract_alice_args(text)
        # alice's args are all optional, but firing it on a query
        # that doesn't mention ALICE / survival budget / working
        # poor / poverty would be a misroute. Require at least one
        # signal keyword in the text before rescuing.
        if not re.search(
            r"\b(?:ALICE|survival\s+budget|working\s+poor|"
            r"asset[-\s]limited|income[-\s]constrained|"
            r"can'?t\s+afford\s+(?:to\s+live|basics?|rent|food)|"
            r"poverty\s+line|too\s+(?:rich|poor)\s+(?:for|to))\b",
            text,
            re.IGNORECASE,
        ):
            return None
        return ext
    if tool == "say":
        # The same extractor the bridge uses post-fire to override the
        # LoRA's text arg. Returns None (= decline rescue) when no
        # anchored pattern matches — e.g. "do you say hello in
        # Japanese" doesn't trigger the rescue, but "say hello world"
        # does.
        from bridge.tools._say import extract_verbatim_phrase
        phrase = extract_verbatim_phrase(text)
        if not phrase:
            return None
        return {"text": phrase}
    return None
