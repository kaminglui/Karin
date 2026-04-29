"""Tests for the clarification-followup routing layer.

Background: when the assistant asks for tool-scoped info ("what's your
household composition?", "where are you?"), the user's next short
message is the answer. Without this layer, the classifier sees a bare
"Single, 1 person" with no match and the LoRA can misroute (observed
2026-04-29: routed to ``get_time`` with ``timezone="Single, 1 person"``).

Two pieces under test:
  * ``detect_clarification_followup`` reads the prior assistant reply
    and returns a tool name when it matched a clarification pattern.
  * ``default_args_continuation`` handles alice / inflation / facts /
    population payloads that the original ``default_args`` would have
    declined (because of keyword guards).
"""
from __future__ import annotations

import pytest

from bridge.routing.classifier import (
    _CLARIFICATION_PATTERNS,
    _last_assistant_text,
    detect_clarification_followup,
)
from bridge.routing.force_fire import default_args_continuation


# --- detect_clarification_followup ----------------------------------------

@pytest.mark.parametrize("expected_tool,prior_reply", [
    (
        "alice",
        "I need more context. Please provide your household composition: "
        "single adult or family with kids?",
    ),
    (
        "alice",
        "To compute ALICE, please specify how many people are in the household.",
    ),
    (
        "alice",
        "Are you a single adult or a couple? How many adults / kids?",
    ),
    (
        "get_weather",
        "Sure, I can check. Which city are you in?",
    ),
    (
        "get_weather",
        "Where are you? I need a location.",
    ),
    (
        "get_time",
        "Which time zone do you want? I need a timezone.",
    ),
    (
        "wiki",
        "Which Lincoln do you mean — Abraham or Bedford?",
    ),
    (
        "inflation",
        "From which year? I need a base year.",
    ),
])
def test_clarification_pattern_matches(expected_tool: str, prior_reply: str) -> None:
    assert detect_clarification_followup(prior_reply) == expected_tool


@pytest.mark.parametrize("benign", [
    "Hey! How's your day going?",
    "It's currently 22 C and sunny in Tokyo.",
    "I think you're looking for the python documentation.",
    "Got it - I'll remember that.",
    "",
    None,
])
def test_clarification_returns_none_on_benign(benign) -> None:
    assert detect_clarification_followup(benign) is None


def test_clarification_pattern_count() -> None:
    """Sanity: every entry in _CLARIFICATION_PATTERNS is (str, regex)."""
    for entry in _CLARIFICATION_PATTERNS:
        assert isinstance(entry, tuple) and len(entry) == 2
        tool, pat = entry
        assert isinstance(tool, str) and tool, "tool name must be non-empty"
        # regex objects have .search; sanity-check the pattern compiled.
        assert hasattr(pat, "search")


# --- _last_assistant_text -------------------------------------------------

def test_last_assistant_text_walks_back_to_most_recent() -> None:
    history = [
        {"role": "user", "content": "tell me about ALICE"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "second reply"},
    ]
    assert _last_assistant_text(history) == "second reply"


def test_last_assistant_text_empty_history() -> None:
    assert _last_assistant_text([]) == ""
    assert _last_assistant_text(None) == ""


def test_last_assistant_text_no_assistant_yet() -> None:
    """Pre-first-reply history (only user message) returns empty."""
    history = [{"role": "user", "content": "hi"}]
    assert _last_assistant_text(history) == ""


# --- default_args_continuation: alice path --------------------------------

def test_alice_continuation_extracts_size_from_short_answer() -> None:
    """User reply 'Single, 1 person.' has no ALICE keyword (the regular
    default_args would decline), but as a clarification follow-up the
    extractor produces household_size=1 directly."""
    args = default_args_continuation("alice", "Single, 1 person.")
    assert args == {"household_size": 1}


def test_alice_continuation_recognizes_couple_phrasings() -> None:
    args = default_args_continuation("alice", "couple with 2 kids")
    assert args == {"composition": "2A2K"}


def test_alice_continuation_empty_dict_is_acceptable() -> None:
    """Alice has sensible defaults — even an extraction that returned
    nothing is a valid call (4-person family, latest year). Continuation
    path returns ``{}`` rather than declining the rescue."""
    args = default_args_continuation("alice", "uh, default I guess")
    assert args == {}


# --- default_args_continuation: inflation/facts/population paths ----------

def test_inflation_continuation_requires_from_year() -> None:
    """Inflation without a year would compute against today, which is
    nonsense — decline so the LoRA can ask again rather than fire."""
    assert default_args_continuation("inflation", "100 dollars") is None
    args = default_args_continuation("inflation", "100 dollars in 1985")
    assert args.get("from_year") == 1985


def test_facts_continuation_requires_year() -> None:
    assert default_args_continuation("facts", "tell me about food") is None
    args = default_args_continuation("facts", "1985")
    assert args.get("year") == 1985


def test_population_continuation_accepts_empty() -> None:
    """Population defaults to world + latest year, so any payload is
    valid — continuation path just returns whatever the extractor
    found, possibly empty."""
    args = default_args_continuation("population", "Japan")
    # extract_population_args may pick up region=japan, but even ``{}``
    # is acceptable here.
    assert isinstance(args, dict)


# --- regression: tool not in FORCE_RESCUE_TOOLS returns None --------------

def test_continuation_declines_for_unrescuable_tool() -> None:
    assert default_args_continuation("circuit", "hello") is None
    assert default_args_continuation("graph", "1 + 2") is None
