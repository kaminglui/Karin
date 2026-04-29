"""Unit tests for bridge.llm.OllamaLLM — mocked HTTP, no live Ollama.

Exercises the tool-call loop, streaming path, history-commit rules,
suffix-provider injection, and summarize(). A real Ollama is never
contacted; responses are canned through ``httpx.MockTransport`` so
these run in CI on any platform. Model-quality / tool-routing
accuracy is out of scope — see ``scripts/eval_routing.py`` (if added)
for a live-model eval harness.
"""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from bridge.llm import OllamaLLM


# ---- helpers --------------------------------------------------------------


def _make_llm(
    responder: Callable[[httpx.Request], httpx.Response],
    *,
    system_prompt: str = "SYS",
    suffix_provider: Callable[[], str] | None = None,
    options: dict | None = None,
) -> OllamaLLM:
    """Build an OllamaLLM whose internal httpx.Client routes through
    a MockTransport driven by ``responder``.
    """
    llm = OllamaLLM(
        base_url="http://mock",
        model="test-model",
        system_prompt=system_prompt,
        temperature=0.7,
        num_ctx=2048,
        options=options or {},
        request_timeout=5.0,
        system_prompt_suffix_provider=suffix_provider,
    )
    # Replace the auto-built client with one wired to MockTransport.
    llm.close()
    llm._client = httpx.Client(
        base_url="http://mock",
        transport=httpx.MockTransport(responder),
        timeout=5.0,
    )
    return llm


def _chat_response(content: str = "", tool_calls: list | None = None) -> dict:
    """Shape an /api/chat non-streaming JSON body."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"model": "test-model", "message": msg, "done": True}


def _tool_call(name: str, args: dict) -> dict:
    return {"function": {"name": name, "arguments": args}}


# ---- chat(): basic non-tool path ------------------------------------------


class TestChatBasic:
    def test_returns_content_and_commits_history(self):
        posted_bodies: list[dict] = []

        def responder(req: httpx.Request) -> httpx.Response:
            posted_bodies.append(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("hello there"))

        llm = _make_llm(responder)
        reply = llm.chat("tell me something")

        assert reply == "hello there"
        # System prompt sent, history now has user+assistant
        assert posted_bodies[0]["messages"][0] == {"role": "system", "content": "SYS"}
        assert len(llm._history) == 2
        assert llm._history[0] == {"role": "user", "content": "tell me something"}
        assert llm._history[1]["role"] == "assistant"

    def test_failed_post_does_not_mutate_history(self, monkeypatch):
        def responder(req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "nope"})

        from bridge.llm import OllamaLLM
        # Tighten the retry delay so the test doesn't sleep 3s.
        monkeypatch.setattr(OllamaLLM, "_RETRY_DELAY_S", 0.0)
        llm = _make_llm(responder)
        with pytest.raises(httpx.HTTPStatusError):
            llm.chat("tell me something")
        assert llm._history == []

    def test_transient_500_is_retried_and_succeeds(self, monkeypatch):
        """A 500 followed by a 200 should yield a normal reply, not raise.

        Real-world cause: on Jetson the LLM runner crashes mid-load when
        sovits is restarting, then recovers. The retry hides that flap.
        """
        from bridge.llm import OllamaLLM
        monkeypatch.setattr(OllamaLLM, "_RETRY_DELAY_S", 0.0)

        state = {"n": 0}

        def responder(req: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(500, json={"error": "runner crashed"})
            return httpx.Response(200, json=_chat_response("recovered"))

        llm = _make_llm(responder)
        reply = llm.chat("tell me something")
        assert reply == "recovered"
        assert state["n"] == 2   # one failed call + one successful retry
        assert llm._history[-1]["content"] == "recovered"

    def test_empty_reply_does_not_commit(self):
        def responder(req):
            return httpx.Response(200, json=_chat_response("   "))

        llm = _make_llm(responder)
        llm.chat("tell me something")
        # Whitespace-only reply → history unchanged
        assert llm._history == []

    def test_commit_history_false_skips_mutation(self):
        def responder(req):
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder)
        llm.chat("hi", commit_history=False)
        assert llm._history == []

    def test_options_forwarded_verbatim(self):
        seen: dict = {}

        def responder(req):
            seen.update(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder, options={"think": "off", "temperature": 0.1})
        llm.chat("tell me something")
        opts = seen["options"]
        # `think` is extracted from options and sent as a top-level field
        # (Ollama rejects it inside options with an "invalid option" warning).
        assert "think" not in opts
        assert seen["think"] is False   # "off" normalized to bool
        assert opts["temperature"] == 0.1   # explicit options win over ctor fallback
        assert opts["num_ctx"] == 2048      # ctor fallback filled in
        assert seen["keep_alive"] == -1

    def test_think_true_top_level(self):
        seen: dict = {}

        def responder(req):
            seen.update(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder, options={"think": True})
        llm.chat("tell me something")
        assert seen["think"] is True

    def test_think_absent_when_not_configured(self):
        seen: dict = {}

        def responder(req):
            seen.update(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder, options={})
        llm.chat("tell me something")
        assert "think" not in seen   # no think field → Ollama uses model default


# ---- chat(): tool-call loop -----------------------------------------------


class TestToolCallLoop:
    def test_single_tool_call_then_final(self, monkeypatch):
        from bridge import tools as tools_mod
        monkeypatch.setattr(tools_mod, "execute", lambda n, a: f"tool:{n}:{a}")

        calls: list[dict] = []

        def responder(req):
            body = json.loads(req.content)
            calls.append(body)
            if len(calls) == 1:
                return httpx.Response(200, json=_chat_response(
                    tool_calls=[_tool_call("get_time", {"tz": "UTC"})],
                ))
            return httpx.Response(200, json=_chat_response("the time is noon"))

        llm = _make_llm(responder)
        observed: list[tuple] = []
        reply = llm.chat(
            "what time is it",
            tools=[{"type": "function", "function": {"name": "get_time"}}],
            on_tool_call=lambda n, a, r: observed.append((n, a, r)),
        )

        assert reply == "the time is noon"
        assert observed == [("get_time", {"tz": "UTC"}, "tool:get_time:{'tz': 'UTC'}")]
        # 2nd POST should have the tool result appended
        second_messages = calls[1]["messages"]
        assert any(m.get("role") == "tool" and m.get("name") == "get_time"
                   for m in second_messages)
        # History commits: user + assistant(tool_call) + tool + assistant(final)
        roles = [m["role"] for m in llm._history]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_duplicate_tool_call_suppressed(self, monkeypatch):
        """LLM retrying the same (name, args) twice should NOT re-execute
        the tool; a synthetic "already tried" note gets fed back instead."""
        from bridge import tools as tools_mod
        exec_calls: list[tuple[str, dict]] = []

        def fake_execute(n, a):
            exec_calls.append((n, dict(a) if isinstance(a, dict) else a))
            return f"article-{len(exec_calls)}"

        monkeypatch.setattr(tools_mod, "execute", fake_execute)

        state = {"n": 0}

        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(200, json=_chat_response(
                    tool_calls=[_tool_call("wiki_random", {})],
                ))
            if state["n"] == 2:
                # LLM retries the same call — should be suppressed
                return httpx.Response(200, json=_chat_response(
                    tool_calls=[_tool_call("wiki_random", {})],
                ))
            return httpx.Response(200, json=_chat_response("OK, here's what I know."))

        llm = _make_llm(responder)
        observed: list = []
        reply = llm.chat(
            "tell me a story",
            tools=[{"type": "function", "function": {"name": "wiki_random"}}],
            on_tool_call=lambda n, a, r: observed.append((n, a, r)),
        )

        assert reply == "OK, here's what I know."
        # tools.execute ran ONCE, not twice
        assert len(exec_calls) == 1
        # on_tool_call observer also fired only once (the suppressed call
        # shouldn't be surfaced to the UI either)
        assert len(observed) == 1

    def test_max_tool_iters_cap_forces_final_reply(self, monkeypatch):
        """When the tool loop caps out, we issue one more sans-tools call
        so the user gets a real answer instead of a cryptic "gave up" stub.
        """
        from bridge import tools as tools_mod
        monkeypatch.setattr(tools_mod, "execute", lambda n, a: "x")

        bodies: list[dict] = []

        def responder(req):
            body = json.loads(req.content)
            bodies.append(body)
            # Requests that include a `tools` key → keep asking for tool
            # calls. The forced-final request has NO tools key → return a
            # plain content reply.
            if "tools" in body:
                return httpx.Response(200, json=_chat_response(
                    tool_calls=[_tool_call("loop", {})],
                ))
            return httpx.Response(200, json=_chat_response("here's my best answer from what I have."))

        llm = _make_llm(responder)
        reply = llm.chat("call the loop tool repeatedly", tools=[{"type": "function", "function": {"name": "loop"}}])
        # We looped the cap, then made a final sans-tools request.
        assert any("tools" not in b for b in bodies), "expected a final sans-tools request"
        assert reply == "here's my best answer from what I have."
        # History IS committed for the forced-final reply — user should
        # see this turn in their scrollback.
        assert any(m.get("role") == "assistant" and m.get("content") == reply for m in llm._history)

    def test_per_tool_cap_suppresses_second_call(self, monkeypatch):
        """After MAX_PER_TOOL (=1) calls of the same tool with ANY args,
        further calls of that tool get suppressed without executing.
        Why 1: small models often repeat the same tool with slightly
        different args (empty vs explicit location), wasting a
        roundtrip and then paraphrasing the wrong result. First answer
        wins."""
        from bridge import tools as tools_mod
        exec_calls: list[tuple[str, dict]] = []

        def fake_execute(n, a):
            exec_calls.append((n, dict(a) if isinstance(a, dict) else a))
            return f"r{len(exec_calls)}"

        monkeypatch.setattr(tools_mod, "execute", fake_execute)

        state = {"n": 0}

        def responder(req):
            state["n"] += 1
            body = json.loads(req.content)
            if "tools" not in body:
                # Forced-final call — return plain content.
                return httpx.Response(200, json=_chat_response("final answer"))
            # Each tool-requesting turn varies args; only the FIRST executes.
            return httpx.Response(200, json=_chat_response(
                tool_calls=[_tool_call("wiki_search", {"query": f"q{state['n']}"})],
            ))

        llm = _make_llm(responder)
        llm.chat("x", tools=[{"type": "function", "function": {"name": "wiki_search"}}])
        # Only 1 real execution despite the model requesting multiple
        assert len(exec_calls) == 1

    def test_on_tool_call_exception_is_swallowed(self, monkeypatch):
        from bridge import tools as tools_mod
        monkeypatch.setattr(tools_mod, "execute", lambda n, a: "ok")

        state = {"n": 0}

        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(200, json=_chat_response(
                    tool_calls=[_tool_call("x", {})],
                ))
            return httpx.Response(200, json=_chat_response("done"))

        def bad_cb(n, a, r):
            raise RuntimeError("observer exploded")

        llm = _make_llm(responder)
        reply = llm.chat("run the x tool",
                         tools=[{"type": "function", "function": {"name": "x"}}],
                         on_tool_call=bad_cb)
        assert reply == "done"  # buggy callback didn't break the turn


class TestLeakedToolCallRecovery:
    """Llama 3.1 sometimes emits tool-args JSON as content instead of
    using the tool_calls field. The recovery path reconstructs a
    synthetic tool_call when the content's keys match a known schema."""

    def _tool_schemas_for(self, name: str, param_keys: list[str]) -> list:
        """Build a minimal Ollama-format tool schema."""
        return [{
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in param_keys},
                },
            },
        }]

    def test_recovers_tell_story_from_content_json(self, monkeypatch):
        """Exact reproduction of the Llama 3.1 leak we saw live:
        content='{"kind":"story","topic":"yourself"}', tool_calls=[]."""
        # Patch tools.execute so no real tool runs.
        from bridge import tools as _tools
        monkeypatch.setitem(
            _tools._DISPATCH, "tell_story",
            lambda topic=None, kind=None: "ok-story-result",
        )

        calls: list[int] = []
        def responder(req):
            calls.append(1)
            if len(calls) == 1:
                # First turn: model leaks args into content, tool_calls empty.
                return httpx.Response(200, json=_chat_response(
                    content='{"kind": "story", "topic": "yourself"}',
                ))
            # Second turn: model is prompted with the tool result and
            # now produces a real user-facing reply.
            return httpx.Response(200, json=_chat_response("final words"))

        llm = _make_llm(responder)
        reply = llm.chat(
            "tell me a joke",
            tools=self._tool_schemas_for("tell_story", ["topic", "kind"]),
        )
        # Two chat calls: one that leaked, one that wrapped up
        assert len(calls) == 2
        # Final reply is the second-turn content, not the leaked JSON
        assert reply == "final words"

    def test_leaked_json_routes_on_tool_call_callback(self, monkeypatch):
        """The recovered call should surface through on_tool_call
        exactly like a natively-structured one, so panels still mount."""
        from bridge import tools as _tools
        monkeypatch.setitem(
            _tools._DISPATCH, "tell_story",
            lambda topic=None, kind=None: "ok",
        )

        state = {"n": 0}
        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(200, json=_chat_response(
                    content='{"topic": "PA", "kind": "joke"}',
                ))
            return httpx.Response(200, json=_chat_response("ha"))

        observed: list[tuple[str, dict]] = []
        def on_tool(name: str, args: dict, result: str) -> None:
            observed.append((name, args))

        llm = _make_llm(responder)
        llm.chat(
            "tell me a joke about PA",
            tools=self._tool_schemas_for("tell_story", ["topic", "kind"]),
            on_tool_call=on_tool,
        )
        assert observed == [("tell_story", {"topic": "PA", "kind": "joke"})]

    def test_non_json_content_is_left_alone(self):
        """Plain text content must NOT be mistaken for a tool call."""
        def responder(req):
            return httpx.Response(200, json=_chat_response("hi there"))

        llm = _make_llm(responder)
        reply = llm.chat(
            "tell me a story",
            tools=self._tool_schemas_for("tell_story", ["topic", "kind"]),
        )
        assert reply == "hi there"

    def test_json_that_matches_no_tool_is_swapped_for_fallback(self):
        """If the leaked JSON's keys don't match any tool, leak-recovery
        abstains. The reply-cleanup layer then catches the bare-JSON
        content and substitutes a persona fallback so the user never
        sees raw ``{"foo": 1}`` in the chat window."""
        def responder(req):
            return httpx.Response(200, json=_chat_response(
                content='{"foo": 1, "bar": 2}',
            ))

        llm = _make_llm(responder)
        reply = llm.chat(
            "hi",
            tools=self._tool_schemas_for("tell_story", ["topic", "kind"]),
        )
        # The prompt "hi" is chitchat → the chitchat guard stripped
        # tools to None before the loop ran, so cleanup sees
        # tools_were_offered=False and substitutes a chitchat fallback.
        assert reply in OllamaLLM._CHITCHAT_FALLBACKS

    def test_recovers_named_wrapper_shape(self, monkeypatch):
        """Llama 3.1 also leaks the ``{"name": "<tool>", "params": {...}}``
        shape — especially for no-arg tools where the raw-args recovery
        can't match (no property keys to compare against)."""
        from bridge import tools as _tools
        called: list[str] = []
        monkeypatch.setitem(_tools._DISPATCH, "__noarg",
                            lambda: called.append("ran") or "ok")
        state = {"n": 0}
        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(200, json=_chat_response(
                    content='{"name": "__noarg", "params": {}}',
                ))
            return httpx.Response(200, json=_chat_response("done"))
        llm = _make_llm(responder)
        reply = llm.chat(
            "catch me up",
            tools=[{"type": "function", "function": {
                "name": "__noarg",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )
        assert called == ["ran"]
        assert reply == "done"

    def test_recovers_function_key_wrapper(self, monkeypatch):
        """Same as above but with ``function`` instead of ``name``."""
        from bridge import tools as _tools
        seen_args: list[dict] = []
        monkeypatch.setitem(
            _tools._DISPATCH, "__withargs",
            lambda **kw: seen_args.append(dict(kw)) or "ok",
        )
        state = {"n": 0}
        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(200, json=_chat_response(
                    content='{"function": "__withargs", "arguments": {"x": 1}}',
                ))
            return httpx.Response(200, json=_chat_response("done"))
        llm = _make_llm(responder)
        llm.chat(
            "x",
            tools=[{"type": "function", "function": {
                "name": "__withargs",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
            }}],
        )
        assert seen_args == [{"x": 1}]

    def test_chitchat_blocks_data_fetch_recovery(self, monkeypatch):
        """Rule Zero guard: for clearly-chitchat user prompts, the
        leak-recovery MUST refuse to auto-dispatch a data-fetch tool
        even when the model leaks its call-shape JSON. Prevents the
        LLM from bypassing the 'reactive not proactive' persona rule
        via stray placeholders."""
        # Track whether dispatch happened.
        dispatched: list[str] = []
        from bridge import tools as _tools
        monkeypatch.setitem(_tools._DISPATCH, "get_digest",
                            lambda: dispatched.append("dispatched") or "d")

        state = {"n": 0}
        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                # First turn: model emits the leaked wrapper shape.
                return httpx.Response(200, json=_chat_response(
                    content='{"function": "get_digest", "params": {}}',
                ))
            # Second turn (if reached): plain chitchat reply.
            return httpx.Response(200, json=_chat_response("hey!"))

        llm = _make_llm(responder)
        reply = llm.chat(
            "hi karin",
            tools=[{"type": "function", "function": {
                "name": "get_digest",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )
        # Recovery was refused → no dispatch happened → content passes
        # through verbatim (which is also strange but preferred over
        # firing the tool).
        assert dispatched == []
        # Leak-recovery refused AND the cleanup layer scrubbed the raw
        # JSON. Because the user prompt was chitchat ("hi karin") the
        # chitchat guard set tools=None before this leak path, so from
        # cleanup's point of view no tools were offered → chitchat pool.
        assert reply in OllamaLLM._CHITCHAT_FALLBACKS

    def test_explicit_data_request_still_recovers(self, monkeypatch):
        """Counterpart to the above: a real news request should still
        auto-recover the leaked wrapper — the chitchat guard is scoped
        to chitchat only."""
        dispatched: list[str] = []
        from bridge import tools as _tools
        monkeypatch.setitem(
            _tools._DISPATCH, "get_digest",
            lambda: dispatched.append("dispatched") or "digest-result",
        )

        state = {"n": 0}
        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(200, json=_chat_response(
                    content='{"function": "get_digest", "params": {}}',
                ))
            return httpx.Response(200, json=_chat_response("here's today's digest"))

        llm = _make_llm(responder)
        reply = llm.chat(
            "catch me up on today's news",
            tools=[{"type": "function", "function": {
                "name": "get_digest",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )
        assert dispatched == ["dispatched"]
        assert reply == "here's today's digest"

    def test_wrapper_with_unknown_tool_name_is_not_recovered(self, monkeypatch):
        """A ``{"name": "nope", ...}`` referring to a tool that isn't
        in the schema list shouldn't be synthesized — otherwise we'd
        dispatch an unknown tool and pollute history."""
        def responder(req):
            return httpx.Response(200, json=_chat_response(
                content='{"name": "not_in_schemas", "params": {}}',
            ))
        llm = _make_llm(responder)
        reply = llm.chat(
            "x",
            tools=[{"type": "function", "function": {
                "name": "real_tool",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )
        # Leak-recovery refused (unknown tool name), and the cleanup
        # layer substitutes a fallback for the raw JSON content so the
        # user never sees it. Tools were on offer here, so the
        # tool-context fallback pool is used (not the chitchat pool).
        assert reply in OllamaLLM._TOOL_FALLBACKS

    def test_picks_narrowest_tool_match(self, monkeypatch):
        """When leaked keys could match several tools, prefer the one
        whose schema is the tightest fit (fewest extra optional params)."""
        from bridge import tools as _tools
        monkeypatch.setitem(_tools._DISPATCH, "narrow",
                            lambda topic=None: "n")
        monkeypatch.setitem(_tools._DISPATCH, "wide",
                            lambda topic=None, a=None, b=None, c=None: "w")

        state = {"n": 0}
        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(200, json=_chat_response(
                    content='{"topic": "x"}',
                ))
            return httpx.Response(200, json=_chat_response("done"))

        observed: list[str] = []
        def on_tool(name, args, result):
            observed.append(name)

        tools_list = [
            {"type": "function", "function": {
                "name": "narrow",
                "parameters": {"type": "object",
                               "properties": {"topic": {"type": "string"}}},
            }},
            {"type": "function", "function": {
                "name": "wide",
                "parameters": {"type": "object",
                               "properties": {k: {"type": "string"}
                                              for k in ("topic", "a", "b", "c")}},
            }},
        ]
        llm = _make_llm(responder)
        llm.chat("test", tools=tools_list, on_tool_call=on_tool)
        assert observed == ["narrow"]  # narrower schema wins


# ---- system prompt + suffix provider --------------------------------------


class TestSystemPrompt:
    def test_suffix_provider_appended(self):
        captured: dict = {}

        def responder(req):
            captured.update(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder,
                        system_prompt="BASE",
                        suffix_provider=lambda: "user likes dogs")
        llm.chat("tell me something")
        sys_msg = captured["messages"][0]
        assert sys_msg["role"] == "system"
        assert sys_msg["content"].startswith("BASE")
        assert "user likes dogs" in sys_msg["content"]

    def test_empty_suffix_leaves_prompt_unchanged(self):
        captured: dict = {}

        def responder(req):
            captured.update(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder, system_prompt="BASE", suffix_provider=lambda: "  ")
        llm.chat("tell me something")
        assert captured["messages"][0]["content"] == "BASE"

    def test_suffix_provider_exception_falls_back(self):
        captured: dict = {}

        def boom() -> str:
            raise RuntimeError("memory unreadable")

        def responder(req):
            captured.update(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder, system_prompt="BASE", suffix_provider=boom)
        # Should not raise — suffix failure is non-fatal.
        llm.chat("tell me something")
        assert captured["messages"][0]["content"] == "BASE"


# ---- history management ---------------------------------------------------


class TestHistoryManagement:
    def test_reset_clears_history(self):
        def responder(req):
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder)
        llm.chat("first")
        llm.chat("second")
        assert len(llm._history) == 4
        llm.reset()
        assert llm._history == []

    def test_set_history_replaces_wholesale(self):
        def responder(req):
            return httpx.Response(200, json=_chat_response("ok"))

        llm = _make_llm(responder)
        llm.set_history([
            {"role": "user", "content": "old-u"},
            {"role": "assistant", "content": "old-a"},
        ])
        assert len(llm._history) == 2
        # Subsequent chat builds on the replaced history
        llm.chat("new")
        roles = [m["role"] for m in llm._history]
        assert roles == ["user", "assistant", "user", "assistant"]


class TestHistoryScoping:
    """history_pairs=N keeps only the last N user+assistant pairs in
    the prompt, preventing stale context from bleeding across turns.
    Full _history is preserved (UI widget replay still works)."""

    def _chatty_llm(self, responder) -> OllamaLLM:
        llm = OllamaLLM(
            base_url="http://mock",
            model="test-model",
            system_prompt="SYS",
            temperature=0.7,
            num_ctx=2048,
            options={},
            request_timeout=5.0,
            history_pairs=2,
        )
        llm.close()
        llm._client = httpx.Client(
            base_url="http://mock",
            transport=httpx.MockTransport(responder),
            timeout=5.0,
        )
        return llm

    def test_history_pairs_zero_is_unlimited(self):
        """The default behavior (history_pairs=0) keeps every pair."""
        llm = OllamaLLM(
            base_url="http://mock", model="t", system_prompt="SYS",
            temperature=0.7, num_ctx=2048, options={},
            history_pairs=0,
        )
        llm._history = [
            {"role": "user", "content": f"u{i}"} for i in range(10)
        ]
        # Interleave assistant messages
        llm._history = []
        for i in range(10):
            llm._history.append({"role": "user", "content": f"u{i}"})
            llm._history.append({"role": "assistant", "content": f"a{i}"})
        visible = llm._llm_visible_history()
        assert len(visible) == 20   # nothing trimmed

    def test_history_pairs_caps_visible_but_leaves_storage(self):
        """Prompt context is trimmed; the underlying _history remains
        full so widget replay + compaction still see everything."""
        captured: list[dict] = []
        def responder(req):
            captured.append(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("done"))

        llm = self._chatty_llm(responder)
        # Seed 5 prior pairs
        for i in range(5):
            llm._history.append({"role": "user", "content": f"u{i}"})
            llm._history.append({"role": "assistant", "content": f"a{i}"})

        llm.chat("new-turn")
        # Full history retained server-side for UI replay etc.
        assert len(llm._history) == 12  # 10 prior + current user + assistant

        # Prompt sent to Ollama: system + last 2 pairs (4 msgs) + current user
        msgs = captured[0]["messages"]
        roles_contents = [(m["role"], m.get("content")) for m in msgs]
        assert roles_contents[0][0] == "system"
        # Expect only u3,a3,u4,a4, then the new user turn — not u0..u2
        tail_users = [c for r, c in roles_contents if r == "user"]
        assert "u0" not in tail_users
        assert "u1" not in tail_users
        assert "u2" not in tail_users
        assert "u3" in tail_users
        assert "u4" in tail_users
        assert "new-turn" in tail_users

    def test_compaction_summaries_survive_trim(self):
        """Compaction summaries are labeled with '[compaction]' as the
        first line of a system message, and must not be trimmed even
        when history_pairs would otherwise drop them."""
        captured: list[dict] = []
        def responder(req):
            captured.append(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("ok"))

        llm = self._chatty_llm(responder)
        llm._history.append({"role": "system", "content": "[compaction] old stuff"})
        for i in range(5):
            llm._history.append({"role": "user", "content": f"u{i}"})
            llm._history.append({"role": "assistant", "content": f"a{i}"})

        llm.chat("new")
        msgs = captured[0]["messages"]
        # The compaction message must appear somewhere in the prompt
        contents = [m.get("content") for m in msgs]
        assert any("[compaction]" in (c or "") for c in contents)

    def test_tool_messages_are_always_stripped(self):
        """history_pairs doesn't change the rule that role=tool messages
        from prior turns are filtered out — they're stale noise."""
        llm = self._chatty_llm(lambda req: httpx.Response(200,
            json=_chat_response("x")))
        llm._history = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "tool", "name": "get_weather", "content": "stale"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        visible = llm._llm_visible_history()
        roles = [m["role"] for m in visible]
        assert "tool" not in roles


# ---- chat_stream() --------------------------------------------------------


def _ndjson_stream_body(deltas: list[str]) -> bytes:
    """Build an ndjson body matching Ollama's /api/chat stream=true shape."""
    lines = []
    for i, d in enumerate(deltas):
        lines.append(json.dumps({
            "message": {"role": "assistant", "content": d},
            "done": i == len(deltas) - 1,
        }))
    return ("\n".join(lines) + "\n").encode("utf-8")


class TestChatStream:
    def test_streams_tokens_and_returns_full_reply(self):
        state = {"n": 0}

        def responder(req):
            state["n"] += 1
            body = json.loads(req.content)
            if not body.get("stream"):
                # Probe call: no tool calls, so the stream call will run.
                return httpx.Response(200, json=_chat_response("ignored-probe"))
            return httpx.Response(200, content=_ndjson_stream_body(["Hel", "lo", "!"]))

        llm = _make_llm(responder)
        tokens: list[str] = []
        reply = llm.chat_stream("hi", on_token=tokens.append)

        assert reply == "Hello!"
        # Reply is shorter than the pre-stream buffer window (32 chars),
        # so the whole thing is buffered then flushed as one emission
        # when the stream ends. Still delivered in full.
        assert "".join(tokens) == "Hello!"
        # History committed once with the streamed reply, not the probe text
        assert llm._history[-1] == {"role": "assistant", "content": "Hello!"}

    def test_stream_tool_loop_then_streams_final(self, monkeypatch):
        from bridge import tools as tools_mod
        monkeypatch.setattr(tools_mod, "execute", lambda n, a: "43F")

        state = {"n": 0}

        def responder(req):
            state["n"] += 1
            body = json.loads(req.content)
            # Call 1 (probe): request a tool
            if state["n"] == 1:
                return httpx.Response(200, json=_chat_response(
                    tool_calls=[_tool_call("get_weather", {"loc": "SF"})],
                ))
            # Call 2 (probe after tool): no tool, trigger stream
            if state["n"] == 2:
                assert body["stream"] is False
                return httpx.Response(200, json=_chat_response("probe"))
            # Call 3 (stream): emit tokens
            assert body["stream"] is True
            return httpx.Response(200, content=_ndjson_stream_body(["it's ", "43F"]))

        llm = _make_llm(responder)
        tool_events: list = []
        tokens: list[str] = []
        reply = llm.chat_stream(
            "weather?",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            on_tool_call=lambda n, a, r: tool_events.append((n, a, r)),
            on_token=tokens.append,
        )

        assert reply == "it's 43F"
        # Short reply → buffered then flushed as one emission at end of stream.
        assert "".join(tokens) == "it's 43F"
        assert tool_events == [("get_weather", {"loc": "SF"}, "43F")]

    def test_stream_buffers_and_scrubs_json_stub(self):
        """When the model streams a JSON-stub reply like
        ``{"name": "no_tool", ...}``, the raw text must NEVER reach
        ``on_token`` — otherwise the UI shows the garbage before the
        end-of-stream cleanup can swap it out.
        """
        def responder(req):
            body = json.loads(req.content)
            if not body.get("stream"):
                return httpx.Response(200, json=_chat_response("probe"))
            return httpx.Response(200, content=_ndjson_stream_body([
                '{"name": ', '"no_tool", ', '"parameters": {}}',
            ]))

        llm = _make_llm(responder)
        tokens: list[str] = []
        reply = llm.chat_stream("random non-chitchat prompt that isnt short",
                                on_token=tokens.append)

        # The raw JSON stub must not leak to the UI stream.
        streamed = "".join(tokens)
        assert '{"name"' not in streamed
        assert "no_tool" not in streamed
        # The returned reply (and what the UI ultimately sees) is a fallback.
        assert reply in OllamaLLM._CHITCHAT_FALLBACKS + OllamaLLM._TOOL_FALLBACKS

    def test_stream_buffers_and_scrubs_note_prefix(self):
        """Same principle for "Note:" prefixes — the cleanup layer strips
        them, so they must not hit the UI either."""
        def responder(req):
            body = json.loads(req.content)
            if not body.get("stream"):
                return httpx.Response(200, json=_chat_response("probe"))
            return httpx.Response(200, content=_ndjson_stream_body([
                "(Note: The ", "output is cool)",
            ]))

        llm = _make_llm(responder)
        tokens: list[str] = []
        reply = llm.chat_stream("random non-chitchat prompt that isnt short",
                                on_token=tokens.append)

        streamed = "".join(tokens)
        assert "(Note:" not in streamed
        assert "The output is" not in streamed

    def test_stream_flushes_normal_short_reply(self):
        """Short non-suspicious replies get buffered and then emitted
        as one chunk at stream end. The user still sees the full reply
        and the return value matches."""
        def responder(req):
            body = json.loads(req.content)
            if not body.get("stream"):
                return httpx.Response(200, json=_chat_response("probe"))
            return httpx.Response(200, content=_ndjson_stream_body([
                "hey", " there", "!",
            ]))

        llm = _make_llm(responder)
        tokens: list[str] = []
        reply = llm.chat_stream("anything", on_token=tokens.append)
        assert reply == "hey there!"
        assert "".join(tokens) == "hey there!"

    def test_stream_live_after_buffer_flush_for_long_normal_reply(self):
        """Once buffered text passes the window without looking suspicious,
        the rest of the stream must flow live token-by-token — not
        stay held until done."""
        deltas = ["This is a ", "long enough reply ", "that exits buffer ",
                  "mode and streams ", "live for the rest."]
        assert sum(len(d) for d in deltas) > 32

        def responder(req):
            body = json.loads(req.content)
            if not body.get("stream"):
                return httpx.Response(200, json=_chat_response("probe"))
            return httpx.Response(200, content=_ndjson_stream_body(deltas))

        llm = _make_llm(responder)
        tokens: list[str] = []
        reply = llm.chat_stream("any", on_token=tokens.append)
        assert reply == "".join(deltas)
        # At least two separate emissions: one flush + ≥1 live delta.
        assert len(tokens) >= 2

    def test_on_token_exception_is_swallowed(self):
        def responder(req):
            body = json.loads(req.content)
            if not body.get("stream"):
                return httpx.Response(200, json=_chat_response("probe"))
            return httpx.Response(200, content=_ndjson_stream_body(["a", "b"]))

        llm = _make_llm(responder)
        reply = llm.chat_stream("hi", on_token=lambda _t: (_ for _ in ()).throw(RuntimeError("x")))
        # Turn still completes with the full reply despite buggy callback
        assert reply == "ab"


# ---- summarize() ----------------------------------------------------------


class TestSummarize:
    def test_flattens_tool_messages(self):
        captured: dict = {}

        def responder(req):
            captured.update(json.loads(req.content))
            return httpx.Response(200, json=_chat_response("SUMMARY"))

        llm = _make_llm(responder)
        out = llm.summarize(
            [
                {"role": "user", "content": "ask"},
                {"role": "assistant", "content": "use tool"},
                {"role": "tool", "name": "get_time", "content": "noon"},
                {"role": "assistant", "content": "it's noon"},
            ],
            system_prompt="SUMMARIZER",
        )
        assert out == "SUMMARY"
        msgs = captured["messages"]
        assert msgs[0] == {"role": "system", "content": "SUMMARIZER"}
        # Tool messages get flattened to role=user with a marker
        tool_flattened = [m for m in msgs if "[tool get_time returned]" in (m.get("content") or "")]
        assert len(tool_flattened) == 1
        assert tool_flattened[0]["role"] == "user"
        # summarize uses a lower temperature than the main chat
        assert captured["options"]["temperature"] == 0.2

    def test_summarize_does_not_touch_history(self):
        def responder(req):
            return httpx.Response(200, json=_chat_response("SUM"))

        llm = _make_llm(responder)
        llm.set_history([{"role": "user", "content": "orig"}])
        llm.summarize([{"role": "user", "content": "x"}], "SYS")
        # summarize() must not mutate rolling history
        assert llm._history == [{"role": "user", "content": "orig"}]
