"""Tests for the OllamaLLM runtime flags: under_fire_rescue,
two_phase_compose, hint_in_user_msg.

Covers constructor propagation, defaults, runtime toggles, and the
system-prompt stable/effective split that underpins hint_in_user_msg.
Pure unit tests — no live Ollama server required.
"""
from __future__ import annotations

import pytest

from bridge.llm import OllamaLLM


def _make_llm(**kw) -> OllamaLLM:
    """Build an OllamaLLM with defaults — no network call fires until
    someone invokes chat/chat_stream, so we can instantiate cheaply."""
    defaults = dict(
        base_url="http://127.0.0.1:11434",
        model="karin-tuned:latest",
        system_prompt="You are Karin.",
        temperature=0.3,
        num_ctx=2048,
        options={},
    )
    defaults.update(kw)
    return OllamaLLM(**defaults)


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------
class TestFlagDefaults:
    """Match the constructor signature in bridge/llm.py:
        under_fire_rescue: bool = True
        two_phase_compose: bool = False
        hint_in_user_msg: bool = False
    """

    def test_under_fire_rescue_default_on(self):
        assert _make_llm().under_fire_rescue is True

    def test_two_phase_compose_default_off(self):
        assert _make_llm().two_phase_compose is False

    def test_hint_in_user_msg_default_off(self):
        assert _make_llm().hint_in_user_msg is False


# ---------------------------------------------------------------------------
# Flag propagation (constructor args → instance attributes)
# ---------------------------------------------------------------------------
class TestFlagPropagation:
    def test_all_flags_turned_on(self):
        llm = _make_llm(
            under_fire_rescue=True,
            two_phase_compose=True,
            hint_in_user_msg=True,
        )
        assert llm.under_fire_rescue is True
        assert llm.two_phase_compose is True
        assert llm.hint_in_user_msg is True

    def test_all_flags_turned_off(self):
        llm = _make_llm(
            under_fire_rescue=False,
            two_phase_compose=False,
            hint_in_user_msg=False,
        )
        assert llm.under_fire_rescue is False
        assert llm.two_phase_compose is False
        assert llm.hint_in_user_msg is False

    @pytest.mark.parametrize("truthy", [True, 1, "yes", object()])
    def test_truthy_values_coerce_to_true(self, truthy):
        llm = _make_llm(two_phase_compose=truthy)
        assert llm.two_phase_compose is True

    @pytest.mark.parametrize("falsy", [False, 0, "", None])
    def test_falsy_values_coerce_to_false(self, falsy):
        llm = _make_llm(under_fire_rescue=falsy)
        assert llm.under_fire_rescue is False


# ---------------------------------------------------------------------------
# Runtime toggle — two_phase_compose is flippable without restart
# ---------------------------------------------------------------------------
class TestTwoPhaseComposeRuntimeToggle:
    def test_set_two_phase_compose_to_true(self):
        llm = _make_llm(two_phase_compose=False)
        llm.set_two_phase_compose(True)
        assert llm.two_phase_compose is True

    def test_set_two_phase_compose_to_false(self):
        llm = _make_llm(two_phase_compose=True)
        llm.set_two_phase_compose(False)
        assert llm.two_phase_compose is False

    def test_set_two_phase_compose_coerces_truthy(self):
        llm = _make_llm()
        llm.set_two_phase_compose(1)
        assert llm.two_phase_compose is True
        llm.set_two_phase_compose(0)
        assert llm.two_phase_compose is False


# ---------------------------------------------------------------------------
# System-prompt stable / effective split
# ---------------------------------------------------------------------------
# hint_in_user_msg=True implies the system prompt stays byte-identical across
# turns so Ollama's KV cache can reuse the full [system] + [history] prefix.
# Implementation: _stable_system_prompt() omits the per-turn routing hint;
# _effective_system_prompt() appends it (the legacy / hint_in_user_msg=off
# path). Verifies the split is honored.


class TestStableVsEffectiveSystemPrompt:
    def test_stable_prompt_contains_base_system(self):
        llm = _make_llm(system_prompt="You are Karin, a voice assistant.")
        stable = llm._stable_system_prompt(user_text="hi")
        assert "You are Karin" in stable

    def test_stable_prompt_invariant_across_user_texts(self):
        """Core invariant — the whole point of the hint_in_user_msg
        refactor is that _stable_system_prompt returns the same bytes
        regardless of user_text. Prefix-cache-friendly."""
        llm = _make_llm()
        a = llm._stable_system_prompt(user_text="hi")
        b = llm._stable_system_prompt(user_text="what's the weather in Tokyo")
        c = llm._stable_system_prompt(user_text=None)
        assert a == b == c

    def test_effective_prompt_starts_with_stable(self):
        """The effective (legacy) prompt is the stable prefix + routing
        hint (if any). It must be a superset starting with stable."""
        llm = _make_llm()
        stable = llm._stable_system_prompt(user_text="hi")
        effective = llm._effective_system_prompt(user_text="hi")
        assert effective.startswith(stable)

    def test_stable_prompt_never_contains_routing_hint_marker(self):
        """`routing hint:` / `routing continuation hint` are signature
        tokens of the hint text. They must never end up in the stable
        prompt regardless of user_text, because that's what makes it
        byte-stable."""
        llm = _make_llm()
        for t in ("what's the weather", "remind me to call mom",
                  "PM of Canada right now", "set the thermostat to 72"):
            stable = llm._stable_system_prompt(user_text=t)
            assert "routing hint" not in stable.lower(), t


# ---------------------------------------------------------------------------
# Suffix provider — used by _stable_system_prompt; prove it's honored
# ---------------------------------------------------------------------------
class TestSuffixProvider:
    def test_zero_arg_provider_is_called(self):
        calls = []
        def provider():
            calls.append(True)
            return "[memory: user loves tea]"
        llm = _make_llm(system_prompt_suffix_provider=provider)
        out = llm._stable_system_prompt(user_text="hi")
        assert calls, "provider should have been called"
        assert "[memory: user loves tea]" in out

    def test_user_text_arg_provider_receives_text(self):
        seen = []
        def provider(user_text):
            seen.append(user_text)
            return f"[ctx: {user_text}]"
        llm = _make_llm(system_prompt_suffix_provider=provider)
        out = llm._stable_system_prompt(user_text="what's new")
        assert "what's new" in (seen[0] if seen else "")
        assert "[ctx: what's new]" in out

    def test_provider_exception_falls_back_gracefully(self):
        """Provider crashes must NOT take down the chat — we log and
        skip the suffix. Prevents a memory-layer bug from bricking
        production replies."""
        def provider():
            raise RuntimeError("memory subsystem is down")
        llm = _make_llm(system_prompt_suffix_provider=provider)
        out = llm._stable_system_prompt(user_text="hi")
        assert "You are Karin" in out
