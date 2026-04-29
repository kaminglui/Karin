"""Unit tests for the new reply-cleanup + guard extensions in bridge.llm.

Covers:

  * _clean_reply() JSON-stub substitution
  * _clean_reply() forbidden-prefix scrubbing
  * _clean_reply() markdown bullet / blank-line collapse
  * _looks_like_chitchat() now catches identity prompts
  * (basic-arithmetic guard removed: classifier now hints math for those)
"""
from __future__ import annotations

from bridge.llm import OllamaLLM


# ---------------------------------------------------------------------------
# _clean_reply — JSON-stub substitution
# ---------------------------------------------------------------------------
def test_clean_reply_scrubs_no_think_prefix() -> None:
    """The LoRA was trained with `/no_think` at the top of the system
    prompt (a Qwen3-family directive). Llama3.1-based karin-tuned echoes
    it as literal prefix. Scrub to a persona fallback. Regression guard
    for the 2026-04-24 prompt-leak fix — removing it from the live
    system prompt can't retrain what the LoRA weights learned."""
    out = OllamaLLM._clean_reply(
        "/no_think Hello! How can I help?",
        tools_were_offered=False,
        user_text="hello",
    )
    assert out in OllamaLLM._CHITCHAT_FALLBACKS


def test_clean_reply_scrubs_no_think_variants() -> None:
    """Covers the -think / _think dash/underscore variants and leading-
    whitespace forms observed in LoRA output."""
    for bad in (
        "/no_think hi there",
        "/no-think hi there",
        "  /no_think hi there",
        "I'm good /no_think thanks for asking",
    ):
        out = OllamaLLM._clean_reply(
            bad, tools_were_offered=False, user_text="hello",
        )
        assert out in OllamaLLM._CHITCHAT_FALLBACKS, f"did not scrub: {bad!r}"


def test_clean_reply_leaves_innocent_think_phrases_alone() -> None:
    """Don't false-positive on legitimate uses of `think` without the
    leading slash + no_think token shape."""
    out = OllamaLLM._clean_reply(
        "I don't think the weather will change today.",
        tools_were_offered=True,
        user_text="what's the weather",
    )
    assert "don't think the weather" in out


def test_clean_reply_substitutes_on_json_stub_reply() -> None:
    out = OllamaLLM._clean_reply(
        '{"name": "None", "parameters": {}}',
        tools_were_offered=False,
        user_text="how are you",
    )
    assert out in OllamaLLM._CHITCHAT_FALLBACKS


def test_clean_reply_uses_tool_fallback_when_tools_were_offered() -> None:
    """When tools were on offer and the model only emitted a JSON stub,
    prefer the tool-context fallback ("check above") over the chitchat
    pool ("pretty chill day") — the widget above has the real data."""
    out = OllamaLLM._clean_reply(
        '{"name": "None", "parameters": {}}',
        tools_were_offered=True,
        user_text="weather in tokyo",
    )
    assert out in OllamaLLM._TOOL_FALLBACKS


def test_clean_reply_substitutes_on_tool_call_stub() -> None:
    out = OllamaLLM._clean_reply(
        '{"function": "get_weather", "arguments": {}}',
        tools_were_offered=False,
        user_text="hi",
    )
    assert out in OllamaLLM._CHITCHAT_FALLBACKS


def test_clean_reply_fallback_is_stable_for_same_prompt() -> None:
    # Hash-based rotation: same prompt → same fallback across runs.
    a = OllamaLLM._clean_reply('{"x": 1}', False, user_text="how are you")
    b = OllamaLLM._clean_reply('{"x": 1}', False, user_text="how are you")
    assert a == b


def test_clean_reply_leaves_json_like_prose_alone() -> None:
    # Prose that merely contains "{" isn't a JSON stub.
    reply = "Sure! here's how you'd write {name: 'karin'} in python"
    out = OllamaLLM._clean_reply(reply, tools_were_offered=False)
    assert out == reply


# ---------------------------------------------------------------------------
# _clean_reply — forbidden prefix scrub
# ---------------------------------------------------------------------------
def test_clean_reply_strips_note_prefix_line() -> None:
    reply = "Note: The tool returned the weather.\nIt's sunny and 22°C."
    out = OllamaLLM._clean_reply(reply, tools_were_offered=True)
    assert out == "It's sunny and 22°C."


def test_clean_reply_strips_parenthesized_note_prefix() -> None:
    reply = "(Note: The news is from the inline widget, I just added a comment)"
    out = OllamaLLM._clean_reply(reply, tools_were_offered=True, user_text="news")
    # All lines were forbidden prefixes → tool-context fallback kicks in
    # because tools were offered.
    assert out in OllamaLLM._TOOL_FALLBACKS


def test_clean_reply_strips_output_is_prefix() -> None:
    reply = "The output is: 345.\n345"
    out = OllamaLLM._clean_reply(reply, tools_were_offered=True)
    assert out == "345"


def test_clean_reply_preserves_good_reply() -> None:
    reply = "It's 18 degrees out, so jacket weather."
    out = OllamaLLM._clean_reply(reply, tools_were_offered=True)
    assert out == reply


# ---------------------------------------------------------------------------
# _clean_reply — markdown cleanup
# ---------------------------------------------------------------------------
def test_clean_reply_strips_bullet_markers() -> None:
    reply = "- first point\n- second point"
    out = OllamaLLM._clean_reply(reply, tools_were_offered=True)
    assert out == "first point\nsecond point"


def test_clean_reply_strips_numbered_list_markers() -> None:
    reply = "1. alpha\n2. beta"
    out = OllamaLLM._clean_reply(reply, tools_were_offered=True)
    assert out == "alpha\nbeta"


def test_clean_reply_collapses_multiple_blank_lines() -> None:
    reply = "line one\n\n\n\nline two"
    out = OllamaLLM._clean_reply(reply, tools_were_offered=True)
    assert out == "line one\n\nline two"


# ---------------------------------------------------------------------------
# _looks_like_chitchat — identity prompts
# ---------------------------------------------------------------------------
def _mk_llm() -> OllamaLLM:
    # We only need the instance methods; a real HTTP client won't be touched.
    llm = OllamaLLM(
        base_url="http://mock",
        model="t",
        system_prompt="",
        temperature=0.0,
        num_ctx=512,
        options={},
        request_timeout=1.0,
    )
    llm.close()
    return llm


def test_chitchat_identity_questions() -> None:
    llm = _mk_llm()
    for q in ("what's your name", "What's your name?", "What is your name?", "your name"):
        assert llm._looks_like_chitchat(q), q
    for q in ("who are you", "tell me about yourself", "what do you do"):
        assert llm._looks_like_chitchat(q), q


def test_chitchat_startswith_short_extensions() -> None:
    """Short prompts that extend a chitchat pattern by a word or two
    should still be recognized — "How are you today?" etc. — without
    letting long specific questions slip through."""
    llm = _mk_llm()
    for q in ("how are you today", "how are you doing", "hi karin how's it going"):
        assert llm._looks_like_chitchat(q), q


def test_chitchat_still_lets_through_data_questions() -> None:
    llm = _mk_llm()
    assert not llm._looks_like_chitchat("what's the weather in tokyo")
    assert not llm._looks_like_chitchat("what's the price of gold")
    assert not llm._looks_like_chitchat("integrate x squared")
    # Long question that happens to begin with a chitchat-ish phrase
    # must NOT short-circuit tool routing.
    assert not llm._looks_like_chitchat(
        "how are you going to solve this rlc circuit problem for me"
    )


# ---------------------------------------------------------------------------
# Arithmetic routing — the pre-classifier now hints `math` for basic
# arithmetic so the user sees the calculation widget. The old guard
# that stripped tools for these prompts was removed on request.
# ---------------------------------------------------------------------------
def test_classifier_routes_basic_arithmetic_to_math() -> None:
    from bridge.routing import classify
    for q in (
        "what's 15 times 23",
        "what is 15 times 23",
        "15 x 23",
        "30% of 120",
        "15 percent of 120",
        "compute 47 plus 88",
        "calculate 9 minus 3",
    ):
        assert classify(q) == "math", q


# ---------------------------------------------------------------------------
# _clean_reply — L8 capability-fabrication scrub (shipped 2026-04-24)
# ---------------------------------------------------------------------------
# Two-step detection: user prompt names a capability domain Karin has no
# tool for AND reply claims first-person success AND no tool fired.
# Failure case: "Set the thermostat to 72" → "I've set it to 72 degrees"
# with zero tools run — the LoRA is lying to the user.

_CAPABILITY_DECLINE = (
    "That one's outside what I can actually do — "
    "you'll have to handle it on your end."
)


class TestCapabilityScrubFires:
    """Positive cases: capability request in user_text + success confirmation
    in reply + no tool fired → canned decline substituted."""

    def _run(self, user_text: str, reply: str) -> str:
        return OllamaLLM._clean_reply(
            reply,
            tools_were_offered=False,
            user_text=user_text,
            tools_fired=False,
        )

    def test_smart_home_thermostat(self):
        out = self._run(
            "Set the thermostat to 72",
            "I've set the thermostat to 72 degrees. Hope that feels comfortable!",
        )
        assert out == _CAPABILITY_DECLINE

    def test_smart_home_with_modifier_words(self):
        """Fix for iter-7 regression: `the front door` has a modifier word
        between article and noun. The user-intent pattern allows up to 2
        modifier words so this still matches."""
        out = self._run(
            "Lock the front door",
            "Done! Locked.",
        )
        assert out == _CAPABILITY_DECLINE

    def test_media_playback(self):
        out = self._run(
            "Play some music",
            "Playing your favorite playlist now.",
        )
        assert out == _CAPABILITY_DECLINE

    def test_ordering_pizza_with_adjectives(self):
        """Adjective padding between verb and noun must not break the match.
        Flat single-regex approaches miss this; two-step detection catches
        it because the request pattern allows up-to-20-char intermediates."""
        out = self._run(
            "Order me a pizza",
            "I've ordered a large pepperoni pizza for you.",
        )
        assert out == _CAPABILITY_DECLINE

    def test_ordering_uber_on_the_way(self):
        """`on the way` is one of the en-route success phrases."""
        out = self._run(
            "Call an Uber",
            "Alright! Your Uber is on the way.",
        )
        assert out == _CAPABILITY_DECLINE

    def test_messaging_email(self):
        out = self._run(
            "Send an email to mom",
            "Sent! Email delivered.",
        )
        assert out == _CAPABILITY_DECLINE

    def test_system_control_restart(self):
        """-ing forms like `restarting` are in the success-phrase list."""
        out = self._run(
            "Restart my computer",
            "Restarting now...",
        )
        assert out == _CAPABILITY_DECLINE

    def test_opener_acknowledgement(self):
        """Imperative openers like `Done!` / `Alright!` count as success
        even without an explicit verb later."""
        out = self._run(
            "Turn on the lights",
            "Done! Lights are on.",
        )
        assert out == _CAPABILITY_DECLINE


class TestCapabilityScrubAllowsPass:
    """Negative cases: scrub must NOT fire."""

    def _run(self, user_text, reply, tools_fired=False):
        return OllamaLLM._clean_reply(
            reply,
            tools_were_offered=False,
            user_text=user_text,
            tools_fired=tools_fired,
        )

    def test_capability_request_but_tool_fired(self):
        """If a tool actually ran, the model is grounded — don't scrub."""
        out = self._run(
            "Set the thermostat to 72",
            "I've set the thermostat to 72 degrees.",
            tools_fired=True,
        )
        assert "thermostat" in out  # reply passes through

    def test_honest_decline_passes_through(self):
        """Model already declined — reply has no success confirmation,
        so the SUCCESS pattern doesn't match and scrub doesn't fire."""
        out = self._run(
            "Set the thermostat to 72",
            "I can't touch your thermostat — but if you tell me what feels off, I'll help you pick a number.",
        )
        assert "thermostat" in out

    def test_no_capability_request_success_passes_through(self):
        """Reply mentions a success verb, but user didn't ask for a
        Karin-incapable action (math queries are normal)."""
        out = self._run(
            "What's 2+2",
            "It's 4. I've done the math for you.",
        )
        assert "4" in out

    def test_missing_user_text_no_scrub(self):
        """Scrub requires both user_text AND reply signals. Missing
        user_text → no scrub."""
        out = self._run(
            None,
            "I've sent the message.",
        )
        # Not scrubbed. Exact string preserved (maybe with markdown cleanup).
        assert "sent" in out.lower()


# ---------------------------------------------------------------------------
# _clean_reply — L8 market-fabrication scrub
# ---------------------------------------------------------------------------
# Pattern: chitchat prompt (greeting) + zero tools fired + reply mentions
# a commodity / crypto / dollar figure / percent move → LoRA invented the
# numbers. Scrub to fallback so user doesn't see fake market data.


class TestMarketScrubFires:
    """Positive: reply confabulates market data on a chitchat turn."""

    def _run(self, user_text: str, reply: str) -> str:
        return OllamaLLM._clean_reply(
            reply,
            tools_were_offered=False,
            user_text=user_text,
            tools_fired=False,
            is_chitchat=True,
        )

    def test_gold_price_on_greeting(self):
        out = self._run(
            "how are you",
            "I'm great! Gold's at 3200 bucks and holding steady.",
        )
        assert out in OllamaLLM._CHITCHAT_FALLBACKS

    def test_crypto_tickers(self):
        out = self._run(
            "hi",
            "Doing well. BTC is up 4% and ETH is up 2.5%.",
        )
        assert out in OllamaLLM._CHITCHAT_FALLBACKS

    def test_dollar_figure(self):
        out = self._run(
            "hey",
            "Pretty good, S&P is hovering around $4500 today.",
        )
        assert out in OllamaLLM._CHITCHAT_FALLBACKS

    def test_percent_move_phrasing(self):
        out = self._run(
            "what's up",
            "Not much, stocks gained 2% in the last hour.",
        )
        assert out in OllamaLLM._CHITCHAT_FALLBACKS


class TestMarketScrubAllowsPass:
    """Negative: market-looking text when the trigger conditions aren't met."""

    def test_not_chitchat_tool_output_passes_through(self):
        """If user explicitly asked about prices and a tool fired, the
        reply's numbers are real — don't scrub. is_chitchat=False.
        """
        out = OllamaLLM._clean_reply(
            "Gold's at $3200 per ounce today.",
            tools_were_offered=True,
            user_text="what's the gold price",
            tools_fired=True,
            is_chitchat=False,
        )
        assert "3200" in out

    def test_chitchat_but_tool_fired_passes_through(self):
        """Edge case: trivial chitchat prompt triggered a tool somehow
        (e.g. user added a price question after). If tool fired, keep
        the reply."""
        out = OllamaLLM._clean_reply(
            "Silver's at $28.50.",
            tools_were_offered=True,
            user_text="how are you",
            tools_fired=True,
            is_chitchat=True,
        )
        assert "28.50" in out

    def test_chitchat_generic_numbers_pass(self):
        """Reply has numbers but no market keywords or dollar/percent
        phrasing. Should pass through."""
        out = OllamaLLM._clean_reply(
            "Einstein died at 76 years old. He was born in 1879.",
            tools_were_offered=False,
            user_text="how old was Einstein when he died",
            tools_fired=False,
            is_chitchat=False,  # this isn't a greeting
        )
        assert "76" in out
