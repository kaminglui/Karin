"""Unit tests for the bridge.routing pre-classifier.

The classifier's contract is "quiet unless confident": abstaining is
the safe default, so most tests verify confident-match cases and a
representative set of abstain cases (chitchat, ambiguity, noise).
"""
from __future__ import annotations

import pytest

from bridge.routing import classify, routing_hint


# ---------------------------------------------------------------------------
# Confident-match cases: a prompt should land on exactly one tool.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        # time
        ("what time is it", "get_time"),
        ("what's the date", "get_time"),
        ("today's date?", "get_time"),
        # weather
        ("what's the weather in tokyo", "get_weather"),
        ("is it raining outside", "get_weather"),
        ("what should i wear today", "get_weather"),
        # news
        ("any news today", "get_news"),
        ("latest headlines", "get_news"),
        ("what's happening in the world", "get_news"),
        # alerts
        ("any alerts", "get_alerts"),
        ("any storm warnings", "get_alerts"),
        ("anything i should know", "get_alerts"),
        # digest
        ("give me the digest", "get_digest"),
        ("daily brief", "get_digest"),
        ("what do i need to know today", "get_digest"),
        # tracker
        ("gold price", "tracker"),
        ("price of silver", "tracker"),
        ("local gas price", "tracker"),
        ("usd/jpy", "tracker"),
        ("exchange rate", "tracker"),
        # wiki
        ("who is marie curie", "wiki"),
        ("tell me about quantum entanglement", "wiki"),
        ("random fact", "wiki"),
        # find_places
        ("coffee shops near me", "find_places"),
        ("where can i buy batteries", "find_places"),
        # web_search
        ("pasta recipe", "web_search"),
        ("how to fix a leaky faucet", "web_search"),
        # convert
        ("5 miles to km", "convert"),
        ("convert 100 usd to eur", "convert"),
        # graph
        ("plot sin(x)", "graph"),
        ("y = x^2 + 3", "graph"),
        # math
        ("integrate x^2", "math"),
        ("eigenvalues of [[1,2],[3,4]]", "math"),
        ("laplace transform of sin(t)", "math"),
        # circuit
        ("voltage divider 5v 1k 2k", "circuit"),
        ("ohm's law calculator", "circuit"),
    ],
)
def test_classify_confident(prompt: str, expected: str) -> None:
    assert classify(prompt) == expected


# ---------------------------------------------------------------------------
# Abstain cases: no hint should be emitted.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt",
    [
        None,
        "",
        "   ",
        # chitchat — handled upstream, but classifier should still abstain
        "hi",
        "hello",
        "thanks",
        "how are you",
        "ok",
        # noise that shouldn't match anything
        "can you write me a haiku about frogs",
        "i'm feeling a bit down today",
        # keyword-looks-like-trigger but not really
        "silver lining",  # no price/quote verb, should abstain
        "rainbow colors",  # shouldn't trigger the weather "rain" word alone
    ],
)
def test_classify_abstain(prompt) -> None:
    assert classify(prompt) is None


# ---------------------------------------------------------------------------
# Compound "A and B" prompts: when two different tools both match, the
# classifier picks the EARLIEST-mentioned one (treated as the primary
# intent) rather than abstaining. The LLM can chain the second tool
# itself if it's still relevant. Previously these abstained — that left
# the LoRA to pick from the full tool set, which fragmented queries.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("what's the weather and any news", "get_weather"),
        ("gold price and the latest headlines", "tracker"),
    ],
)
def test_classify_compound_picks_earliest(prompt, expected) -> None:
    assert classify(prompt) == expected


# ---------------------------------------------------------------------------
# routing_hint formatter
# ---------------------------------------------------------------------------
def test_routing_hint_formats_confident_match() -> None:
    hint = routing_hint("what time is it")
    # Regex classifier emits "[routing rule: ...`get_time` tool ...]";
    # embedding classifier emits "[routing hint (semantic): ...]". Both
    # carry the tool name and the bracketed-prefix marker.
    assert hint.startswith("[routing ")
    assert "get_time" in hint


def test_routing_hint_empty_on_abstain() -> None:
    assert routing_hint("hi") == ""
    assert routing_hint(None) == ""
    assert routing_hint("") == ""


# ---------------------------------------------------------------------------
# Stability: the emitted hint format is grep-friendly and won't change
# accidentally. Guards against silent prompt-format drift.
# ---------------------------------------------------------------------------
def test_routing_hint_format_is_stable() -> None:
    assert routing_hint("latest news") == (
        "[routing rule: this prompt strongly matches the `get_news` "
        "tool — call it unless the prompt is clearly unrelated.]"
    )


# ---------------------------------------------------------------------------
# Robustness: age / biography queries → wiki
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("how old is Obama", "wiki"),
        ("how old was Einstein", "wiki"),
        ("when was Einstein born", "wiki"),
        ("what year was Lincoln born", "wiki"),
        ("what year did Napoleon die", "wiki"),
        ("birthday of Marie Curie", "wiki"),
    ],
)
def test_classify_age_biography(prompt: str, expected: str) -> None:
    assert classify(prompt) == expected


# ---------------------------------------------------------------------------
# Robustness: knowledge / fact queries → wiki
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("who invented the lightbulb", "wiki"),
        ("who discovered penicillin", "wiki"),
        ("who wrote Romeo and Juliet", "wiki"),
        ("who founded Apple", "wiki"),
        ("where is the Eiffel Tower", "wiki"),
        ("where is Tokyo", "wiki"),
        ("when did World War 2 end", "wiki"),
        ("when did the Berlin Wall fall", "wiki"),
        ("history of the Roman Empire", "wiki"),
        ("meaning of democracy", "wiki"),
        ("what is quantum computing", "wiki"),
    ],
)
def test_classify_knowledge_queries(prompt: str, expected: str) -> None:
    assert classify(prompt) == expected


# ---------------------------------------------------------------------------
# Robustness: weather-related queries → get_weather
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("do I need an umbrella", "get_weather"),
        ("will it rain tomorrow", "get_weather"),
        ("will it be cold tonight", "get_weather"),
        ("bring a jacket", "get_weather"),
        ("need sunscreen", "get_weather"),
    ],
)
def test_classify_weather_implicit(prompt: str, expected: str) -> None:
    assert classify(prompt) == expected


# ---------------------------------------------------------------------------
# Robustness: tracker / market queries → tracker
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("gold worth today", "tracker"),
        ("what is gold trading at", "tracker"),
        ("bitcoin trading at", "tracker"),
        ("stock market today", "tracker"),
    ],
)
def test_classify_tracker_extended(prompt: str, expected: str) -> None:
    assert classify(prompt) == expected


# ---------------------------------------------------------------------------
# Robustness: must NOT route (ambiguous, personal, creative)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt",
    [
        # Creative — no tool needed
        "write me a poem about the sea",
        "tell me a joke",
        "make up a bedtime story",
        # Personal opinion — no tool
        "what's your favorite color",
        "do you like sushi",
        # Too vague for any specific tool
        "interesting",
        "hm",
        "lol",
        # Looks like a keyword match but semantically wrong
        "I need to convert my thinking",  # not a unit conversion
    ],
)
def test_classify_robustness_abstain(prompt) -> None:
    assert classify(prompt) is None


# ---------------------------------------------------------------------------
# Robustness: queries that look like one tool but should route to another
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        # "how to" → web_search, not wiki
        ("how to bake sourdough bread", "web_search"),
        # "recipe" → web_search, not wiki
        ("chicken tikka masala recipe", "web_search"),
        # "plot" → graph, not wiki
        ("plot x^2 + 1", "graph"),
        # pure arithmetic → math, not web_search
        ("what's 15 times 23", "math"),
        ("50% of 200", "math"),
        # reminder, not time
        ("remind me to call mom at 5pm", "schedule_reminder"),
        ("set a timer for 10 minutes", "schedule_reminder"),
    ],
)
def test_classify_tricky_routing(prompt: str, expected: str) -> None:
    assert classify(prompt) == expected
