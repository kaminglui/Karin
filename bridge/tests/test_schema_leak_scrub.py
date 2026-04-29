"""Regression guard for the tool-schema narration leak + the
bare-form ``say <text>`` routing/extraction fix.

The LoRA occasionally reads its own tool catalog out of context and
emits a description of the available tools instead of routing to one
or replying. Observed 2026-04-28 on `say` prompts producing replies
like "The input text is a JSON object representing a set of functions
... `say`: No description provided ...". This surfaces are tool count
grew (21 schemas) and the description-only routing for `say` became
ambiguous to the LoRA.

The runtime fix lives in ``bridge.llm.OllamaLLM._PROMPT_LEAK_MARKERS``;
this file pins the regex against representative leak shapes so a
future markers-edit can't silently drop coverage.

Tests are split into:

* **``leak`` cases** — replies that match the published bug report or
  a near-paraphrase. ``_clean_reply`` MUST swap to a fallback string.
* **``benign`` cases** — replies that contain similar tokens (e.g. the
  word "function") in legitimate ways. ``_clean_reply`` MUST leave
  these alone (modulo cosmetic em-dash / bullet collapse, which is
  upstream of the leak scrub).
"""
from __future__ import annotations

import pytest

from bridge.llm import OllamaLLM


# ---------------------------------------------------------------------------
# Leak shapes that MUST be scrubbed
# ---------------------------------------------------------------------------

_LEAK_CASES: list[tuple[str, str]] = [
    (
        # The verbatim shape from the user's bug report.
        "exact-bug-report",
        "The input text is a JSON object representing a set of functions, "
        "each with its own description and parameters. The functions are:\n\n"
        "`say`: No description provided.\n"
        "`alice`: Estimate the share of US households that are Asset Limited.",
    ),
    (
        # Lowercase + minor rewording, which the LoRA also emits.
        "lowercase-paraphrase",
        "the input text is a json object describing a set of functions, "
        "each with parameters. the functions are:\n- alice\n- inflation",
    ),
    (
        # "No description provided" is the only place this exact phrase
        # appears in the system — the schema's empty-description placeholder.
        "stray-no-description-line",
        "Here are the tools:\n- `say`: No description provided.\n- `wiki`: lookup",
    ),
    (
        # Just the "set of functions, each with" fragment.
        "fragment-set-of-functions",
        "I see a set of functions, each with their own arguments. "
        "I'll pick one.",
    ),
    (
        # "the functions are:" header with a bulleted list.
        "functions-are-header",
        "Sure. The functions are:\n- get_time\n- get_weather\n- wiki",
    ),
]


@pytest.mark.parametrize("label,leak", _LEAK_CASES)
def test_schema_leak_scrubbed_to_fallback(label: str, leak: str) -> None:
    """Each leak shape must end up replaced by an entry from one of the
    fallback pools, never returned to the user verbatim."""
    out = OllamaLLM._clean_reply(
        leak,
        tools_were_offered=True,
        user_text="say hello",
        tools_fired=False,
    )
    expected_pool = OllamaLLM._TOOL_FALLBACKS  # tools_were_offered=True
    assert out in expected_pool, (
        f"[{label}] _clean_reply did not scrub the leak.\n"
        f"  in:  {leak[:120]!r}\n"
        f"  out: {out[:120]!r}"
    )


def test_schema_leak_uses_chitchat_pool_when_no_tools_offered() -> None:
    """When tools_were_offered=False the scrub falls into the chitchat
    pool. Same scrub trigger — pool selection is the only difference."""
    out = OllamaLLM._clean_reply(
        "the input text is a JSON object listing the functions",
        tools_were_offered=False,
        user_text="hi",
        tools_fired=False,
    )
    assert out in OllamaLLM._CHITCHAT_FALLBACKS


# ---------------------------------------------------------------------------
# Benign replies that MUST pass through (modulo upstream cosmetic edits)
# ---------------------------------------------------------------------------

_BENIGN_REPLIES: list[tuple[str, str]] = [
    (
        # Real data-tool reply — uses "function" in a math context.
        "math-function-reply",
        "The function f(x) = x^2 has its minimum at x=0.",
    ),
    (
        # Legitimate explanation about a feature.
        "feature-explanation",
        "Karin supports a few text-only commands you can type directly.",
    ),
    (
        # Real wiki-style reply that mentions "set" and "function" but
        # not the leak fragment.
        "wiki-set-function",
        "A function is a relation that maps each element of a set to "
        "exactly one element of another set.",
    ),
    (
        # Forecast-style reply that happens to mention "function".
        "math-explanation",
        "The derivative of that function is 2x; the slope at x=3 is 6.",
    ),
]


@pytest.mark.parametrize("label,reply", _BENIGN_REPLIES)
def test_benign_reply_not_scrubbed(label: str, reply: str) -> None:
    """A real reply that mentions individual leak-adjacent words but
    not the full schema-narration shape must NOT be replaced by a
    fallback. (Cosmetic em-dash / bullet collapse is allowed — we
    only assert the reply isn't *replaced* with one of the fallback
    pool strings.)"""
    out = OllamaLLM._clean_reply(
        reply,
        tools_were_offered=True,
        user_text="some prompt",
        tools_fired=True,
    )
    fallback_pools = (
        OllamaLLM._TOOL_FALLBACKS
        + OllamaLLM._CHITCHAT_FALLBACKS
        + OllamaLLM._BOOKKEEPING_FALLBACKS
    )
    assert out not in fallback_pools, (
        f"[{label}] benign reply was scrubbed to a fallback pool entry.\n"
        f"  in:  {reply!r}\n"
        f"  out: {out!r}"
    )


# ---------------------------------------------------------------------------
# Sanity: the regex pattern is actually present (catches accidental
# rollback when someone deletes lines from _PROMPT_LEAK_MARKERS).
# ---------------------------------------------------------------------------

def test_leak_marker_pattern_includes_new_shapes() -> None:
    """If a future edit removes the new patterns by accident, this test
    fails before the scrub does. Cheaper to debug than the parametric
    cases above."""
    pat = OllamaLLM._PROMPT_LEAK_MARKERS.pattern.lower()
    # Word-anchored fragments (whitespace becomes `\s+` in the compiled
    # pattern, so split each phrase and check token presence).
    for required in (
        "input",
        "json",
        "representing",
        "functions",
        "description",
        "provided",
    ):
        assert required in pat, (
            f"_PROMPT_LEAK_MARKERS no longer contains {required!r}; "
            f"schema-leak coverage may have regressed."
        )


# ---------------------------------------------------------------------------
# Bare-form ``say <text>`` — the routing-pattern + extractor fix shipped
# alongside the leak scrub. The schema leak surfaces when plain "say
# hello world" misroutes; once routing fires correctly the LoRA never
# gets a chance to narrate the catalog. Belt and braces.
# ---------------------------------------------------------------------------

def test_say_extractor_handles_bare_imperative() -> None:
    """``say hello world`` (no quotes, no "after me" framing) used to
    return None from the extractor and the LoRA latched onto stale
    history phrases. The new last-priority pattern catches it."""
    from bridge.tools._say import extract_verbatim_phrase
    assert extract_verbatim_phrase("say hello world") == "hello world"
    assert extract_verbatim_phrase("say goodnight") == "goodnight"
    assert extract_verbatim_phrase("Say cheese!") == "cheese!"


def test_say_extractor_specific_patterns_still_win() -> None:
    """The new bare pattern is registered LAST so the more-specific
    forms keep extracting cleanly (without the framing words)."""
    from bridge.tools._say import extract_verbatim_phrase
    # "after me" framing is still stripped, not preserved
    assert extract_verbatim_phrase("say after me hello") == "hello"
    # quoted form still extracts the inner content
    assert extract_verbatim_phrase('say "foo bar"') == "foo bar"


def test_say_extractor_rejects_non_imperative_say() -> None:
    """Mid-sentence "say" or meta-questions about saying must NOT
    extract — that's what the ^-anchor on the new pattern guards."""
    from bridge.tools._say import extract_verbatim_phrase
    for non_imperative in (
        "how do you say hello in spanish",
        "did you say something",
        "I'd say it's time",
        "when did you say that",
    ):
        assert extract_verbatim_phrase(non_imperative) is None, (
            f"extractor false-positively matched: {non_imperative!r}"
        )


def test_say_routing_pattern_anchors_to_message_start() -> None:
    """The schema's new routing_pattern requires `say` to be the first
    word. Confirms the regex behaviour matches the docstring claim."""
    import re
    from bridge.tools._schemas import TOOL_SCHEMAS
    # TOOL_SCHEMAS is OpenAI-style: list of {"type": "function",
    # "function": {...}}. Reach inside the nested function block.
    say_fn = next(
        s["function"] for s in TOOL_SCHEMAS
        if s.get("function", {}).get("name") == "say"
    )
    patterns = [re.compile(p, re.IGNORECASE) for p in say_fn.get("routing_patterns", [])]

    def any_match(text: str) -> bool:
        return any(p.search(text) for p in patterns)

    # MUST fire (the bug-report case + adjacent forms)
    for fire in (
        "say hello world",
        "say goodnight",
        '  say cheese',  # leading whitespace tolerated
    ):
        assert any_match(fire), f"missing fire: {fire!r}"

    # MUST NOT fire (false-positive guards)
    for skip in (
        "how do you say hello",
        "did you say something",
        "I'd say it's time",
        "what's the weather",
    ):
        assert not any_match(skip), f"false positive: {skip!r}"
