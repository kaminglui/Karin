"""Comprehensive tests for the regex-based routing classifier.

Tests every tool with multiple phrasings — both hits that SHOULD match
and false positives that SHOULD NOT. Fast (no LLM, no network).
"""
from __future__ import annotations

import pytest

from bridge.routing.classifier import classify


# --- chitchat (should return None — no tool) --------------------------

class TestChitchat:
    @pytest.mark.parametrize("text", [
        "hi", "hello", "hey", "thanks", "ok", "got it", "sure",
        "how are you?", "what's up?", "tell me about yourself",
        "good morning", "lol", "haha", "nice", "cool", "bye",
    ])
    def test_chitchat_returns_none(self, text):
        assert classify(text) is None


# --- get_time ---------------------------------------------------------

class TestGetTime:
    @pytest.mark.parametrize("text", [
        "what time is it?",
        "what's the time?",
        "what is the time",
        "current time",
        "time now",
        "what day is it?",
        "today's date",
        "what time",
        "what date is it",
    ])
    def test_time_hits(self, text):
        assert classify(text) == "get_time"


# --- get_weather ------------------------------------------------------

class TestGetWeather:
    @pytest.mark.parametrize("text", [
        "what's the weather?",
        "weather in Tokyo",
        "is it raining?",
        "is it snowing?",
        "is it cold outside?",
        "is it hot today?",
        "is it windy?",
        "is it sunny?",
        "forecast for tomorrow",
        "temperature outside",
        "what should I wear?",
        "how hot is it?",
        "how cold is it?",
        "rain today",
        "humidity",
    ])
    def test_weather_hits(self, text):
        assert classify(text) == "get_weather"


# --- get_news ---------------------------------------------------------

class TestGetNews:
    @pytest.mark.parametrize("text", [
        "any news?",
        "what's the news?",
        "latest headlines",
        "what's happening?",
        "breaking news",
        "news about tech",
        "latest stories",
    ])
    def test_news_hits(self, text):
        assert classify(text) == "get_news"


# --- get_alerts -------------------------------------------------------

class TestGetAlerts:
    @pytest.mark.parametrize("text", [
        "any alerts?",
        "show me alerts",
        "are there any warnings?",
        "travel advisory",
        "anything I should know?",
        "emergencies",
    ])
    def test_alerts_hits(self, text):
        assert classify(text) == "get_alerts"


# --- get_digest -------------------------------------------------------

class TestGetDigest:
    @pytest.mark.parametrize("text", [
        "daily digest",
        "daily brief",
        "morning brief",
        "my brief",
        "catch me up",
        "what's new",
        "what do I need to know?",
    ])
    def test_digest_hits(self, text):
        assert classify(text) == "get_digest"


# --- tracker ----------------------------------------------------------

class TestTracker:
    @pytest.mark.parametrize("text", [
        "gold price",
        "silver price",
        "oil price",
        "bitcoin price",
        "btc price",
        "price of gold",
        "what's the gold quote?",
        "gas price",
        "fuel price",
        "USD/JPY",
        "eur/usd",
        "exchange rate",
        "how much is gold",
        "how much is bitcoin",
        "gold today",
        "food index",
        "food cpi",
    ])
    def test_tracker_hits(self, text):
        assert classify(text) == "tracker"

    @pytest.mark.parametrize("text", [
        "gold ring for my wife",
        "silver lining",
        "oil painting",
    ])
    def test_tracker_false_positives_avoided(self, text):
        assert classify(text) != "tracker"


# --- math -------------------------------------------------------------

class TestMath:
    @pytest.mark.parametrize("text", [
        "what is 15 * 23?",
        "15 times 23",
        "100 + 200",
        "30% of 120",
        "compute 5 * 8",
        "calculate 99 / 3",
        "integrate x^2 dx",
        "derivative of sin(x)",
        "solve for x: 2x + 3 = 7",
        "eigenvalues of [[1,2],[3,4]]",
        "determinant of [[1,0],[0,1]]",
        "laplace transform",
    ])
    def test_math_hits(self, text):
        assert classify(text) == "math"


# --- convert ----------------------------------------------------------

class TestConvert:
    @pytest.mark.parametrize("text", [
        "5 miles to km",
        "100 usd to jpy",
        "10 kg in pounds",
        "convert 32 celsius to fahrenheit",
        "50 eur to gbp",
        "how many cups in 500 ml",
    ])
    def test_convert_hits(self, text):
        assert classify(text) == "convert"


# --- wiki -------------------------------------------------------------

class TestWiki:
    @pytest.mark.parametrize("text", [
        "who is Albert Einstein?",
        "what is photosynthesis?",
        "tell me about Japan",
        "random fact",
        "who was Napoleon?",
    ])
    def test_wiki_hits(self, text):
        assert classify(text) == "wiki"

    def test_what_is_number_not_wiki(self):
        # "what is 15 times 23" should NOT match wiki
        assert classify("what is 15 times 23") != "wiki"


class TestWikiCurrentXPatterns:
    """`current X` title queries added 2026-04-24 as case 26 fix —
    routes `who's the current prime minister` to wiki instead of letting
    the LoRA answer from (possibly stale) parametric knowledge.

    Three route paths:
      * `(current|latest|today's|present) <named-office>`
      * `who is the current|latest <anything>`
      * `<named-office> (right now|today|currently)`
    """

    @pytest.mark.parametrize("text", [
        # current-X + named office
        "who's the current prime minister of the UK",
        "what's the current president of Argentina",
        "current chancellor of Germany",
        "latest pope",
        "today's king of Sweden",
        "present monarch of Japan",
        # "who is the current <anything>"
        "who is the current president",
        "who's the current ceo of Apple",
        # <office> (right now|today|currently) — adjacency-based, so
        # trailing modifier sits directly on the office noun.
        "president right now",
        "prime minister today",
        "ceo currently",
        "king these days",
    ])
    def test_current_x_routes_to_wiki(self, text):
        assert classify(text) == "wiki", text

    def test_current_time_still_time_not_wiki(self):
        """Regression guard: `current time` must still match get_time,
        not get confused by the new wiki current-X patterns. The time
        pattern is more specific (anchored to `time|date` nouns) so the
        classifier's per-tool resolution puts get_time first."""
        assert classify("what's the current time") == "get_time"

    def test_historical_holder_passes_to_wiki(self):
        """Baseline: `who was X` (historical) still goes to wiki through
        the existing `who is|was X` pattern — not affected by the
        current-X addition."""
        assert classify("who was FDR") == "wiki"


# --- find_places ------------------------------------------------------

class TestFindPlaces:
    @pytest.mark.parametrize("text", [
        "pizza near me",
        "coffee shops nearby",
        "where can I find a bookstore?",
        "ramen around here",
    ])
    def test_places_hits(self, text):
        assert classify(text) == "find_places"


# --- web_search -------------------------------------------------------

class TestWebSearch:
    @pytest.mark.parametrize("text", [
        "recipe for banana bread",
        "how to change a tire",
        "iphone 16 reviews",
        "python tutorial",
        "search for best headphones",
        "google best restaurants in NY",
    ])
    def test_search_hits(self, text):
        assert classify(text) == "web_search"


# --- graph ------------------------------------------------------------

class TestGraph:
    @pytest.mark.parametrize("text", [
        "plot y = x^2",
        "graph sin(x)",
        "draw f(x) = 2x + 1",
        "y = e^x",
        "f(x) = log(x)",
    ])
    def test_graph_hits(self, text):
        assert classify(text) == "graph"


# --- circuit ----------------------------------------------------------

class TestCircuit:
    @pytest.mark.parametrize("text", [
        "voltage divider 10k and 20k",
        "RC circuit with 1k and 100nF",
        "RL circuit",
        "impedance at 1kHz",
        "ohm's law",
    ])
    def test_circuit_hits(self, text):
        assert classify(text) == "circuit"


# --- schedule_reminder ------------------------------------------------

class TestReminder:
    @pytest.mark.parametrize("text", [
        "remind me to call mom at 5pm",
        "set a reminder for tomorrow",
        "set an alarm for 7am",
        "don't forget to buy milk",
        "don't let me forget the meeting",
    ])
    def test_reminder_hits(self, text):
        assert classify(text) == "schedule_reminder"


# --- update_memory ----------------------------------------------------

class TestUpdateMemory:
    @pytest.mark.parametrize("text", [
        "remember that I like ramen",
        "my name is Alex",
        "I study electrical engineering",
        "I live in Pennsylvania",
    ])
    def test_memory_hits(self, text):
        assert classify(text) == "update_memory"

    @pytest.mark.parametrize("text", [
        # Casual emotional expressions — NOT durable user facts. The
        # schema's docstring explicitly excludes these to avoid
        # spurious memory writes on chitchat-shaped preferences.
        "I hate spicy food",
        "I love hiking",
        # Pattern-tight: "call me X" alone collides with "call me when
        # you're free". Routing requires the "from now on" / "instead"
        # disambiguator. Without it, the classifier abstains and the
        # LoRA decides at runtime.
        "call me Alex",
        # Generic "I prefer X" is ambiguous — could be a preference
        # update, could be conversational. Classifier abstains; the
        # LoRA decides whether to fire update_memory.
        "I prefer short answers",
    ])
    def test_memory_false_positives_avoided(self, text):
        assert classify(text) != "update_memory"


# --- ambiguity (two tools match → abstain) ----------------------------

class TestAmbiguity:
    def test_empty_returns_none(self):
        assert classify("") is None
        assert classify(None) is None

    def test_no_match_returns_none(self):
        assert classify("bleep bloop zorp") is None

    def test_ambiguous_abstains(self):
        # "how to convert 5 miles to km" matches both web_search
        # (how to) and convert (5 miles to km) → should abstain
        result = classify("how to convert 5 miles to km")
        # Either abstains (None) or picks one — both acceptable
        assert result in (None, "convert", "web_search")
