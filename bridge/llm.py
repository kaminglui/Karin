"""Ollama chat client with rolling conversation history.

Holds one rolling history (system prompt + alternating user/assistant
turns) and forwards an ``options`` dict verbatim on every request, so
Qwen-specific flags like ``think: "off"`` stay where the config puts them.

History is committed on successful response only — if the POST fails, the
user turn is NOT appended. Otherwise a retry would double-count the turn.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

import httpx

from bridge import compose as _compose
from bridge import llm_backends as _backends
from bridge import reply_scrub as _reply_scrub
from bridge import tool_call_recovery as _tool_call_recovery

log = logging.getLogger("bridge.llm")


# Re-exported for callers that imported these as private module-level
# names. The canonical homes are now in bridge.tool_call_recovery.
_INJECTION_PATTERNS = _tool_call_recovery.INJECTION_PATTERNS
_sanitize_tool_result = _tool_call_recovery.sanitize_tool_result


class OllamaLLM:
    """Ollama ``/api/chat`` client with rolling conversation history.

    Holds one rolling history (system prompt + alternating user/assistant
    turns) and forwards an ``options`` dict verbatim on every request, so
    model-specific flags like Qwen's ``think: "off"`` stay where the
    config puts them. History is committed on successful response only —
    a failed POST does not mutate state.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        temperature: float,
        num_ctx: int,
        options: dict,
        request_timeout: float = 120.0,
        system_prompt_suffix_provider: "Callable[[], str] | None" = None,
        max_per_tool: int = 1,
        max_tool_iters: int = 5,
        history_pairs: int = 0,
        backend: str = "ollama",
        under_fire_rescue: bool = True,
        two_phase_compose: bool = False,
        hint_in_user_msg: bool = False,
    ) -> None:
        """Open an HTTP client and seed the conversation history empty.

        Args:
            base_url: Ollama server base URL (e.g.
                ``http://localhost:11434``).
            model: Ollama model tag.
            system_prompt: System message prepended to every chat request.
            temperature: Fallback temperature if not in ``options``.
            num_ctx: Fallback context window if not in ``options``.
            options: Dict forwarded verbatim to Ollama's ``options`` field.
                Overrides the top-level temperature/num_ctx fallbacks above.
            request_timeout: Per-request HTTP timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        # "ollama" (default) → POST /api/chat with Ollama's body/response shape.
        # "llamacpp" → POST /v1/chat/completions (llama-server's OpenAI-compat
        # endpoint) and translate response back into Ollama shape at the wire
        # layer so the rest of this class is agnostic.
        if backend not in ("ollama", "llamacpp", "mlc"):
            raise ValueError(f"backend must be 'ollama', 'llamacpp', or 'mlc', got {backend!r}")
        self.backend = backend
        # When the LLM under-fires on a tool-worthy prompt (e.g. a
        # historical-fact question the wiki tool should answer), the
        # classifier usually has a confident hint. If this flag is on,
        # we force-execute that hinted tool with best-effort args and
        # let the LLM produce a final reply grounded in the result.
        # See bridge/routing/force_fire.py for the arg-inference and
        # _maybe_force_fire_rescue below for the integration point.
        self.under_fire_rescue = bool(under_fire_rescue)
        # Two-phase compose: when a tool has fired this turn, skip the
        # default "final reply" LLM call (which sees the full tool schema
        # + coaching in context and often produces schema_leak /
        # refuses_own_tool / disavowal / meta-commentary failures) and
        # instead ask the LLM to write a user-facing reply from scratch
        # using ONLY the user's prompt + scrubbed tool outputs. Off by
        # default while we A/B test it against the stock path. See
        # _compose_reply_from_tools below.
        self.two_phase_compose = bool(two_phase_compose)
        # Prefix-cache optimization: move the per-turn routing hint out
        # of the system prompt and into the user message prefix. The
        # system prompt then stays byte-identical across turns, so
        # Ollama/llama.cpp's KV cache can reuse the full system + history
        # prefix on multi-turn sessions. Routing quality can shift because
        # the iter-3 LoRA was trained seeing the hint at end-of-system —
        # flag must be validated by the 135-case eval before shipping.
        self.hint_in_user_msg = bool(hint_in_user_msg)
        # Merge config options with temperature/num_ctx, without clobbering
        # anything the user put in the options dict explicitly.
        self._options = dict(options)
        self._options.setdefault("temperature", temperature)
        self._options.setdefault("num_ctx", num_ctx)
        # Ollama's chat API takes `think` as a TOP-LEVEL request field
        # (not an option). If the config put it in options, pop it out
        # here and normalize to bool so every request includes it at
        # the right spot. Qwen3/Qwen3.5 otherwise generates hundreds of
        # thinking tokens per turn — turning this off is the difference
        # between a 1s and a 40s reply.
        think_raw = self._options.pop("think", None)
        if isinstance(think_raw, bool):
            self._think: bool | None = think_raw
        elif isinstance(think_raw, str):
            self._think = think_raw.strip().lower() not in ("off", "false", "no", "0", "")
        else:
            self._think = None
        self._client = httpx.Client(base_url=self.base_url, timeout=request_timeout)
        # Rolling chat history. Lives in a HistorySession so the lock
        # that guards mutations is shared by every code path that
        # touches it — async routes (via run_in_threadpool) and the
        # worker-thread chat loop both acquire the same RLock.
        from bridge.history import HistorySession
        self._history: "HistorySession" = HistorySession()
        # Optional provider that returns extra text (user + agent memory)
        # to append to the system prompt on every turn. Queried fresh
        # each call so edits take effect without a server restart.
        self._suffix_provider = system_prompt_suffix_provider
        # Per-model tool-loop caps. Resolved by model_config.resolve_for()
        # at startup; different quant/size combos need different caps
        # (3B repeats tools, 7B handles chains).
        self.max_per_tool = int(max_per_tool)
        self.max_tool_iters = int(max_tool_iters)
        # How many prior user+assistant turn pairs to include in the
        # prompt. 0 = unlimited (legacy behavior). Prevents cross-turn
        # context bleed on small models.
        self.history_pairs = int(history_pairs)

    def _stable_system_prompt(self, user_text: str | None = None) -> str:
        """Return base system prompt + suffix-provider output, WITHOUT
        the per-turn routing hint. Byte-stable across turns provided
        the suffix provider itself is stable (memory + location are;
        the bandit preference-hint is not, but bandit is off in prod).
        """
        parts: list[str] = [self.system_prompt]
        if self._suffix_provider is not None:
            try:
                import inspect
                sig = inspect.signature(self._suffix_provider)
                if len(sig.parameters) == 0:
                    suffix = (self._suffix_provider() or "").strip()  # type: ignore[call-arg]
                else:
                    suffix = (self._suffix_provider(user_text) or "").strip()  # type: ignore[call-arg]
            except Exception as e:
                log.warning("system prompt suffix provider raised: %s", e)
                suffix = ""
            if suffix:
                parts.append(suffix)
        return "\n\n".join(parts)

    def _routing_hint_text(self, user_text: str | None) -> str:
        """Return the per-turn routing hint string, or empty if the
        classifier abstains. Split out so callers can choose whether
        to embed the hint in the system prompt (legacy) or prefix it
        onto the user message (prefix-cache optimised).
        """
        try:
            from bridge.routing import routing_hint
            return routing_hint(user_text, history=self._history)
        except Exception as e:
            log.warning("routing classifier raised: %s", e)
            return ""

    def _effective_system_prompt(self, user_text: str | None = None) -> str:
        """Legacy composite: stable prefix + routing hint in one blob.

        Preserved for paths that don't yet split the hint off (e.g. the
        summarizer, older tests). New call sites in ``chat`` and
        ``chat_stream`` honour ``self.hint_in_user_msg`` and build the
        two halves separately.
        """
        base = self._stable_system_prompt(user_text)
        hint = self._routing_hint_text(user_text)
        if hint:
            return f"{base}\n\n{hint}"
        return base

    # Backend transport (Ollama / llama.cpp / MLC-LLM) lives in
    # bridge.llm_backends. Re-exposed here as class attributes so any
    # code that pokes at the retry knobs continues to work; the
    # implementations themselves are pure functions in that module.
    _RETRY_STATUS_CODES = _backends.DEFAULT_RETRY.status_codes
    _RETRY_DELAY_S = _backends.DEFAULT_RETRY.delay_s
    _RETRY_MAX = _backends.DEFAULT_RETRY.max_retries

    def _post_chat(self, body: dict):
        """Dispatch to the backend-specific transport. Returns
        something that quacks like httpx.Response for the
        caller's .json() and .raise_for_status(). The actual
        body translation, retry loop, and error salvage live in
        bridge.llm_backends; this method only forwards."""
        retry = _backends.Retry(
            status_codes=self._RETRY_STATUS_CODES,
            delay_s=self._RETRY_DELAY_S,
            max_retries=self._RETRY_MAX,
        )
        return _backends.post_chat(self.backend, self._client, body, retry)

    # Tools that fetch world-state data. We intentionally refuse to
    # auto-recover leaked tool-calls for these if the user prompt is
    # short / chitchat-looking — the Karin persona's Rule Zero forbids
    # calling them as greeting filler, and rescuing a stray JSON here
    # would silently violate that rule.
    _DATA_FETCH_TOOLS: frozenset[str] = frozenset({
        "get_news", "get_alerts", "get_digest",
        "tracker", "get_weather", "get_time",
        "find_places", "web_search", "wiki",
    })

    # Tools whose output is fire-and-forget bookkeeping — a confirmation
    # string the reply doesn't need to quote ("memory updated", "reminder
    # set for 7pm"). When these are the *only* tools fired this turn, the
    # phase-1 LLM reply is already a fine casual reply with the side
    # effect done invisibly; running L7a _compose_reply_from_tools adds
    # ~5-10 s on Orin Nano for no quality lift. Mixed turns (bookkeeping
    # + a data tool) still go through L7a so the data tool's output gets
    # composed into the reply.
    _BOOKKEEPING_TOOLS: frozenset[str] = frozenset({
        "update_memory",
        "schedule_reminder",
    })

    # Tools whose output IS the reply, verbatim. The LLM's compose pass
    # would re-write or moralize ("I cannot repeat that") instead of
    # echoing, defeating the whole point. When only passthrough tools
    # fire, skip both phases of LLM composition and emit the tool
    # output directly. Mixed with data tools is undefined for now —
    # treat as passthrough wins (the user asked for an echo).
    _PASSTHROUGH_TOOLS: frozenset[str] = frozenset({
        "say",
    })

    @staticmethod
    def _all_tools_in(record: list[dict], allowed: frozenset[str]) -> bool:
        """True iff `record` is non-empty AND every entry's name is in `allowed`."""
        return bool(record) and all(
            rec.get("name") in allowed for rec in record
        )

    # Prompts that should never trigger a data-fetch tool regardless of
    # what the model leaks into content. Lowercased, substring match.
    _CHITCHAT_PATTERNS: tuple[str, ...] = (
        "hi", "hello", "hey", "yo ", "sup",
        "thanks", "thank you", "thx", "ty",
        "ok", "okay", "cool", "nice", "got it", "sounds good",
        "bye", "later", "see you", "goodnight", "good night",
        "how are you", "how's it going", "what's up",
        "tell me about yourself", "who are you",
        # Identity questions — the persona answers these, no tool.
        # Without these the model sometimes picks wiki for "what's your name".
        "what's your name", "what is your name", "your name",
        "what do you do", "what are you",
    )

    # Trivial single-word greetings. When the user sends any of these
    # (optionally followed by Karin's name and/or punctuation), we bypass
    # the LoRA entirely and return a chitchat fallback — short prompts
    # are the failure mode that triggers the iter-3 system-prompt leak
    # ("The system background-pings ..." regurgitated in place of a
    # reply). Keep this list conservative: only clearly-greeting tokens,
    # nothing that could be a legitimate question.
    _TRIVIAL_GREETING = re.compile(
        r"^\s*(?:"
        r"hi|hello|hey|yo|sup|heya|howdy|hiya|ok|okay|hola"
        r"|good\s+(?:morning|afternoon|evening|night)"
        r"|morning|evening"
        r")(?:\s+karin)?[\s.,!?]*$",
        re.IGNORECASE,
    )

    # Substrings that only appear when the LoRA leaks the system prompt
    # (or its own training-time paraphrase of it) into the reply. Scrub
    # matches at clean-reply time → fall through to the chitchat
    # fallback. Keep this list narrow: every marker here is a phrase a
    # real user reply would never contain by accident.
    #   `═══` — section-heading character from karin.yaml, unmistakable.
    #   `background-ping(s)` / `refreshing memory` — specific to an
    #       older version of the system prompt that lives in iter-3 SFT.
    #   `turn record` — same provenance.
    #   `routing hint` + `:` — only appears when the appended
    #       classifier hint bleeds through instead of being acted on.
    # Reply-scrub regexes + fallback pools live in bridge.reply_scrub
    # (L8 in the routing pipeline). Re-exported here as class attributes
    # so existing internal call sites (`cls._PROMPT_LEAK_MARKERS`,
    # `self._FORBIDDEN_PREFIXES`) keep working without rewriting every
    # site. Mutating the originals would also update these references.
    _PROMPT_LEAK_MARKERS = _reply_scrub.PROMPT_LEAK_MARKERS
    _CAPABILITY_REQUEST_PATTERNS = _reply_scrub.CAPABILITY_REQUEST_PATTERNS
    _CAPABILITY_SUCCESS_PATTERNS = _reply_scrub.CAPABILITY_SUCCESS_PATTERNS
    _MARKET_FABRICATION_PATTERNS = _reply_scrub.MARKET_FABRICATION_PATTERNS

    def _looks_like_chitchat(self, user_text: str | None) -> bool:
        """Heuristic: is this prompt one we should NEVER turn into a
        data-fetch call?

        Rules, in order:
          * Exact-match a chitchat pattern after stripping trailing
            punctuation — catches "hi", "how are you", "thanks".
          * ``startswith(pattern + " ")`` on a short prompt — catches
            "how are you today", "hi karin!", "tell me about yourself
            already" without letting long, specific prompts like
            "how are you going to solve the rlc circuit?" through.

        Deliberately conservative: longer / more specific prompts fall
        through so legitimate data-fetch questions still route normally.
        """
        if not user_text:
            return False
        norm = user_text.strip().lower().rstrip("!?.,")
        # Exact match — cheap and catches the base cases.
        if len(norm) <= 35 and norm in {p.rstrip() for p in self._CHITCHAT_PATTERNS}:
            return True
        # Startswith match for short prompts that extend a chitchat
        # pattern with a word or two ("how are you today", "hi karin").
        # Cap at 6 words so we don't over-trigger on long questions
        # that happen to start with a chitchat-ish phrase.
        word_count = len(norm.split())
        if word_count <= 6:
            for pattern in self._CHITCHAT_PATTERNS:
                p = pattern.rstrip()
                if not p:
                    continue
                if norm == p or norm.startswith(p + " "):
                    return True
        return False

    def _recover_leaked_tool_call(
        self,
        content: str,
        tools: list | None,
        user_text: str | None = None,
    ) -> dict | None:
        """Thin wrapper around ``tool_call_recovery.recover_leaked_tool_call``
        that wires in OllamaLLM's chitchat heuristic and the
        ``_DATA_FETCH_TOOLS`` allowlist. The recovery logic itself
        (JSON parsing, named-vs-bare-args shapes, narrowest-schema
        match) lives in ``bridge.tool_call_recovery``."""
        return _tool_call_recovery.recover_leaked_tool_call(
            content,
            tools,
            data_fetch_tools=self._DATA_FETCH_TOOLS,
            looks_like_chitchat=self._looks_like_chitchat,
            user_text=user_text,
        )

    def _llm_visible_history(self) -> list[dict]:
        """Return ``_history`` with PRIOR-TURN tool residue stripped and
        optionally capped to the last ``history_pairs`` user+assistant
        pairs.

        `_history` is persistent (used for UI widget replay across reloads)
        so it keeps every role=tool message and every intermediate
        assistant-with-tool-calls message. The LLM doesn't need that
        scrollback on *new* turns — yesterday's weather result is stale
        noise, and re-prompting with it bloats context + confuses routing.
        Tool messages within the CURRENT turn's loop are appended to the
        local ``messages`` list and are unaffected by this filter.

        When ``history_pairs > 0``, keep only the last N user/assistant
        message pairs. Prevents cross-turn context bleed (e.g. the model
        latching onto "Hong Kong" from turn N-1 when turn N asks about
        something unrelated). Compaction summaries (if any) are always
        kept — they carry condensed context for the full prior session.

        """
        out: list[dict] = []
        for m in self._history:
            role = m.get("role")
            if role == "tool":
                continue
            if role == "assistant":
                content = m.get("content") or ""
                # Skip the intermediate tool-call carrier messages — they
                # have no user-visible content, only a tool_calls field.
                if not content.strip():
                    continue
            out.append(m)
        if self.history_pairs <= 0:
            return out
        # Separate compaction summaries (system-role, content starts with
        # a compaction marker) from the user/assistant dialog. We never
        # trim summaries — they are the condensed context for turns we
        # would otherwise forget.
        summaries: list[dict] = []
        dialog: list[dict] = []
        for m in out:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role == "system" and content.startswith("[compaction]"):
                summaries.append(m)
            elif role in ("user", "assistant"):
                dialog.append(m)
            else:
                # Other roles (unusual but possible) — keep them in place.
                dialog.append(m)
        # Keep only the last 2*N messages of dialog (one user + one
        # assistant per pair). Works even if the history ends on a
        # dangling user message (e.g. current turn being processed).
        keep = self.history_pairs * 2
        trimmed_dialog = dialog[-keep:] if keep < len(dialog) else dialog
        if len(dialog) != len(trimmed_dialog):
            log.debug(
                "history trim: %d → %d messages (history_pairs=%d)",
                len(dialog), len(trimmed_dialog), self.history_pairs,
            )
        return summaries + trimmed_dialog

    def chat(
        self,
        user_text: str,
        tools: list | None = None,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
        commit_history: bool = True,
    ) -> str:
        """Send a user turn and return the assistant's reply.

        If ``tools`` is provided, handles the tool-call loop: the LLM may
        respond with tool calls, which are executed locally and fed back
        into the conversation until the LLM produces a final answer
        (no more tool calls).

        Args:
            user_text: The new user message. Appended to history only on
                successful response.
            tools: Optional list of tool JSON schemas to offer the LLM.
            on_tool_call: Optional callback ``(name, arguments, result)``
                invoked after each tool the LLM requests executes. Lets
                external observers (e.g. UI-5's chat streaming endpoint)
                surface tool invocations without modifying this method's
                return type. Exceptions in the callback are logged and
                swallowed — a buggy observer won't corrupt the chat turn.

        Returns:
            The assistant's final reply text (no tool-call messages,
            just the final ``message.content`` the LLM produces).

        Raises:
            httpx.HTTPError: If the POST fails or the server returns a
                non-2xx status.
        """
        # Trivial-greeting bypass: single-word "hi"/"hello"/"hey"/etc
        # prompts are the exact shape that triggers iter-3's system-
        # prompt regurgitation bug. Short user input gives the LoRA
        # too little to anchor on, and it sometimes completes the
        # SFT-remembered system prompt instead of replying. Skip the
        # LoRA entirely on these — a canned chitchat line is both
        # faster AND leak-proof. The _PROMPT_LEAK_MARKERS scrub in
        # _clean_reply catches longer inputs that slip through.
        if self._TRIVIAL_GREETING.match(user_text or ""):
            reply = self._pick_fallback(user_text, tools_were_offered=False)
            log.info(
                "trivial-greeting bypass: user=%r → %r",
                (user_text or "")[:40], reply[:60],
            )
            if commit_history:
                self._history.append({"role": "user", "content": user_text or ""})
                self._history.append({"role": "assistant", "content": reply})
            return reply

        # Chitchat guard (Rule Zero): when the user's prompt is clearly
        # a greeting / ack / "how are you", strip tools entirely. With
        # no tools on offer the model can't emit tool_calls and can't
        # leak placeholder call JSON — it has to respond with plain
        # text, which is exactly what the persona calls for on chitchat.
        # Stripping JUST data-fetch tools doesn't work reliably: some
        # models then emit synthetic "no_tool" placeholders as a way
        # to signal "I didn't pick any." Clearing tools is cleaner.
        if tools and self._looks_like_chitchat(user_text):
            log.info(
                "chitchat guard: withholding all %d tools for prompt %r",
                len(tools), (user_text or "")[:40],
            )
            tools = None

        # Capture the pre-classifier's guess for this turn so we can log
        # (hint, picked) at turn end. Computed once here and reused by
        # _effective_system_prompt to avoid double classification.
        hint_tool, hint_is_continuation = self._resolve_hint_tool(
            user_text, tools is not None,
        )
        picked_tools: list[str] = []

        # Tool narrowing: when the regex classifier is confident (single
        # unambiguous match), restrict the tool list to ONLY that tool
        # plus always-available passive tools (update_memory). This
        # prevents the LLM from picking web_search when get_weather is
        # the right call. When the classifier abstains, the full list
        # is preserved and the LLM decides freely. Continuation hints
        # (prior-turn fallback for pronoun-led follow-ups) skip
        # narrowing — they're a softer signal, and if the LoRA picks
        # a different tool on the follow-up, we should let it.
        _ALWAYS_AVAILABLE = {"update_memory"}
        # Narrowing is skipped for llamacpp: the LoRA reliably emits
        # the correct tool from the full catalog, and llama-server
        # strict-rejects (HTTP 500) any tool_call whose name isn't in
        # the request's `tools` array — so a too-aggressive narrowing
        # kills otherwise-correct routing.
        if (
            hint_tool
            and not hint_is_continuation
            and tools
            and self.backend == "ollama"
        ):
            narrowed = [
                t for t in tools
                if (t.get("function", {}).get("name") == hint_tool
                    or t.get("function", {}).get("name") in _ALWAYS_AVAILABLE)
            ]
            if narrowed:
                log.info(
                    "tool narrowing: classifier=%s, %d->%d tools for %r",
                    hint_tool, len(tools), len(narrowed),
                    (user_text or "")[:40],
                )
                tools = narrowed

        # Build the message list for this turn. We work on a local copy
        # so intermediate tool-call messages don't touch self._history
        # until we have a final clean reply.
        #
        # hint_in_user_msg: when on, the per-turn routing hint is moved
        # from the end of the system prompt to the start of the user
        # message. That keeps the system prompt byte-stable across turns
        # so Ollama's KV cache can reuse the entire system + history
        # prefix instead of re-prefilling from the hint onward.
        if self.hint_in_user_msg:
            sys_content = self._stable_system_prompt(user_text)
            hint = self._routing_hint_text(user_text)
            user_content_with_hint = (
                f"{hint}\n\n{user_text}" if hint else user_text
            )
        else:
            sys_content = self._effective_system_prompt(user_text)
            user_content_with_hint = user_text
        # Clean user message for history commit — the per-turn hint is
        # stale next turn, so we store only the user's actual text so
        # the history we resend on future turns is byte-stable.
        user_msg_clean = {"role": "user", "content": user_text}
        user_msg_sent = {"role": "user", "content": user_content_with_hint}
        messages = [
            {"role": "system", "content": sys_content},
            *self._llm_visible_history(),
            user_msg_sent,
        ]
        new_turn_messages = [user_msg_clean]

        # Tool-routing safety nets for smaller models:
        #   seen_calls  — suppress exact duplicates (same tool + same args)
        #   tool_counts — cap any tool at MAX_PER_TOOL uses per turn even
        #                 with different args, so a model that keeps
        #                 re-querying wiki_search with "PA" → "Pennsylvania"
        #                 → "PA state" doesn't burn the whole iteration
        #                 budget on near-synonym probes.
        import json as _json
        seen_calls: set[tuple[str, str]] = set()
        tool_counts: dict[str, int] = {}
        # Records every tool that fired during this turn, in order.
        # Used by the two-phase compose path (see
        # _compose_reply_from_tools) to build a focused reply prompt.
        # Updated both by the normal tool-execution loop below AND by
        # the under-fire rescue pipeline (see _try_force_fire_rescue,
        # which accepts this list and appends to it).
        compose_tool_record: list[dict] = []
        MAX_PER_TOOL = self.max_per_tool
        MAX_TOOL_ITERS = self.max_tool_iters
        for _ in range(MAX_TOOL_ITERS):
            body = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": self._options,
                # keep_alive: -1 = stay loaded forever (until ollama
                # restart). On Jetson the model load from disk is ~10s,
                # so paying that cost ONCE per server start instead of
                # every 4 minutes is a huge latency win.
                "keep_alive": -1,
            }
            if self._think is not None:
                body["think"] = self._think
            if tools:
                body["tools"] = tools

            resp = self._post_chat(body)
            data = resp.json()
            msg = data["message"]

            tool_calls = msg.get("tool_calls") or []
            # Leak recovery: some tunes (notably Llama 3.1) emit tool-args
            # JSON into content instead of tool_calls. When we detect that,
            # rebuild a synthetic call so the loop keeps going.
            if not tool_calls:
                leaked = self._recover_leaked_tool_call(
                    msg.get("content", ""), tools, user_text=user_text,
                )
                if leaked is not None:
                    tool_calls = [leaked]
                    msg = {**msg, "tool_calls": tool_calls, "content": ""}
            if not tool_calls:
                # Under-fire rescue: only applies when no tool has fired
                # YET THIS TURN (picked_tools empty) and we haven't
                # already rescued once. If the classifier gave a
                # confident hint and the LLM chose no tool anyway,
                # force the hinted tool with heuristic args and loop
                # back through this same code path. The rescue emits a
                # synthetic assistant+tool message pair; the next
                # iteration's LLM call will see the tool result and
                # produce a grounded reply. Budget: 1 rescue per turn.
                if (
                    self.under_fire_rescue
                    and hint_tool
                    and not picked_tools
                    and tools is not None
                ):
                    rescued = self._try_force_fire_rescue(
                        hint_tool, user_text, messages, new_turn_messages,
                        picked_tools, seen_calls, tool_counts, on_tool_call,
                        is_continuation=hint_is_continuation,
                        compose_tool_record=compose_tool_record,
                    )
                    if rescued:
                        continue   # loop back; next iteration runs the LLM on the new history
                # Final answer — no more tool calls.
                #
                # Two-phase compose: the stock reply from ``msg`` comes
                # from an LLM call where the full tool schema + coaching
                # are still in context. That's what drives most of the
                # reply-quality failures (schema_leak, refuses_own_tool,
                # disavowal, you_meta_commentary, widget_template_leak).
                # When two_phase is on, re-invoke the LLM with a focused
                # no-schema prompt:
                #   * tools fired this turn → compose using those outputs
                #   * no tools fired → casual-reply compose (chitchat /
                #     idiom / direct-knowledge), still tool-free
                # Sanity check: if the compose call also returns a schema
                # leak we fall back to the stock reply.
                reply = ""
                # Passthrough short-circuit: when only ``say``-style
                # echo tools fired, the tool output IS the reply. Skip
                # both L7a compose AND the LoRA's stock reply (which
                # may have refused to echo). Runs regardless of the
                # two_phase_compose flag.
                if self._all_tools_in(compose_tool_record, self._PASSTHROUGH_TOOLS):
                    log.info(
                        "L7a skipped — passthrough-only tools fired: %s",
                        [r.get("name") for r in compose_tool_record],
                    )
                    reply = "\n".join(
                        str(rec.get("result") or "")
                        for rec in compose_tool_record
                    ).strip()
                elif self.two_phase_compose:
                    # Skip L7a when every fired tool is bookkeeping — the
                    # phase-1 reply already absorbed the side effects, no
                    # data needs to be re-grounded.
                    only_bookkeeping = self._all_tools_in(
                        compose_tool_record, self._BOOKKEEPING_TOOLS,
                    )
                    if compose_tool_record and not only_bookkeeping:
                        candidate = self._compose_reply_from_tools(
                            user_text, compose_tool_record,
                        )
                    elif not compose_tool_record:
                        candidate = self._compose_reply_no_tool(
                            user_text, self._history,
                        )
                    else:
                        log.info(
                            "L7a skipped — bookkeeping-only tools fired: %s",
                            [r.get("name") for r in compose_tool_record],
                        )
                        candidate = ""
                    if candidate and self._compose_output_ok(candidate):
                        reply = candidate
                if not reply:
                    reply = msg.get("content", "")
                # Bookkeeping turns earn a casual side-effect ack ("got
                # it") rather than the data-tool fallback ("check above"
                # / "details in the widget") which makes no sense when
                # the only side effect was a memory write.
                _bookkeeping_only_for_clean = self._all_tools_in(
                    compose_tool_record, self._BOOKKEEPING_TOOLS,
                )
                reply = self._clean_reply(
                    reply,
                    tools_were_offered=tools is not None,
                    user_text=user_text,
                    tools_fired=bool(picked_tools),
                    is_chitchat=self._looks_like_chitchat(user_text),
                    bookkeeping_only=_bookkeeping_only_for_clean,
                )
                if reply.strip() and commit_history:
                    self._history.extend(new_turn_messages)
                    self._history.append({"role": "assistant", "content": reply})
                self._log_routing_decision(user_text, hint_tool, picked_tools)
                return reply

            # Tool calls were requested — append the assistant's tool-call
            # message, execute each tool, append role=tool results, loop.
            messages.append(msg)
            new_turn_messages.append(msg)

            from bridge import tools as tools_mod  # lazy import to avoid cycle
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                if name and name not in picked_tools:
                    picked_tools.append(name)
                args = fn.get("arguments", {})
                # Argument override for `say`: the iter-3 LoRA tends to
                # latch onto a phrase from history (the previous reply
                # is often the most-recent string-shaped span in
                # context). Pre-extract from the CURRENT user message
                # using anchored patterns; only override when extraction
                # succeeds. Falls through silently otherwise.
                if name == "say" and isinstance(args, dict):
                    from bridge.tools._say import extract_verbatim_phrase
                    _phrase = extract_verbatim_phrase(user_text)
                    if _phrase and _phrase != args.get("text"):
                        log.info(
                            "say arg override: LoRA=%r → extracted=%r",
                            args.get("text"), _phrase,
                        )
                        args = {**args, "text": _phrase}
                # Wiki → inflation redirect: iter-3 was trained without
                # the inflation tool, so it tends to pick wiki for
                # "dollar in YYYY worth today" / "how much was X in
                # YYYY worth" phrasings. When the user_text has clear
                # inflation cues (a from_year extractable AND wiki's
                # query echoes those cues), swap the call to inflation.
                # Conservative: requires BOTH a successful extract AND
                # the LoRA's wiki query to overlap with money/year/
                # worth/dollar tokens — so legitimate "Berlin Wall in
                # 1989" wiki queries are unaffected.
                if name == "wiki" and isinstance(args, dict):
                    from bridge.tools._inflation import (
                        _WIKI_TO_INFLATION_REDIRECT_RE,
                        extract_inflation_args,
                    )
                    _wiki_q = str(args.get("query") or "")
                    if _WIKI_TO_INFLATION_REDIRECT_RE.search(_wiki_q):
                        _ext = extract_inflation_args(user_text)
                        if _ext.get("from_year"):
                            log.info(
                                "wiki→inflation redirect: wiki(query=%r) "
                                "user=%r → inflation(%s)",
                                _wiki_q, user_text, _ext,
                            )
                            name = "inflation"
                            args = _ext
                            if "inflation" not in picked_tools:
                                picked_tools.append("inflation")
                # Argument fill for `inflation`: the LoRA frequently
                # omits `amount` on phrasings like "How much is a dollar
                # in 1970 money?" (errors out on amount=0) or misreads
                # the source year. Pre-extract from the user message
                # and merge — fills missing/empty args. measure is
                # special-cased: extracted 'wages'/'both' overrides
                # LoRA's default 'cpi' because explicit wage keywords
                # in the user text outrank the LoRA's lazy default.
                if name == "inflation" and isinstance(args, dict):
                    from bridge.tools._inflation import extract_inflation_args
                    _extracted = extract_inflation_args(user_text)
                    if _extracted:
                        merged = dict(args)
                        for k, v in _extracted.items():
                            if k == "measure" and v in ("wages", "both"):
                                merged[k] = v
                            elif k == "item" and v:
                                # Always override LoRA's item — the
                                # extractor matches against an exact
                                # known-keys list while the LoRA tends
                                # to invent free-text item names.
                                merged[k] = v
                            elif k == "region" and v and v != "us":
                                # Always override when extractor finds
                                # a non-US region keyword. iter-3
                                # doesn't know `region` exists and
                                # defaults to "us"; explicit "in Hong
                                # Kong" / "yen in 1970" must win.
                                merged[k] = v
                            elif k == "regions" and v:
                                # Multi-region comparison — extractor
                                # found at least 2 regions plus a
                                # comparison cue. Always override; the
                                # tool's own logic switches to
                                # comparison mode when this is set.
                                merged[k] = v
                            elif not merged.get(k):
                                merged[k] = v
                        if merged != args:
                            log.info(
                                "inflation arg fill: LoRA=%s → merged=%s",
                                args, merged,
                            )
                            args = merged
                if name == "population" and isinstance(args, dict):
                    # Same pattern as inflation: extractor produces
                    # canonical region+year args; merge over LoRA's
                    # output. iter-3 doesn't know `population` exists.
                    from bridge.tools._population import extract_population_args
                    _extracted = extract_population_args(user_text)
                    if _extracted:
                        merged = dict(args)
                        for k, v in _extracted.items():
                            if k == "region" and v and v != "world":
                                merged[k] = v  # explicit region wins
                            elif not merged.get(k):
                                merged[k] = v
                        if merged != args:
                            log.info(
                                "population arg fill: LoRA=%s → merged=%s",
                                args, merged,
                            )
                            args = merged
                if name == "facts" and isinstance(args, dict):
                    from bridge.tools._facts import extract_facts_args
                    _extracted = extract_facts_args(user_text)
                    if _extracted:
                        merged = dict(args)
                        for k, v in _extracted.items():
                            if not merged.get(k):
                                merged[k] = v
                        if merged != args:
                            log.info(
                                "facts arg fill: LoRA=%s → merged=%s",
                                args, merged,
                            )
                            args = merged
                if name == "alice" and isinstance(args, dict):
                    # ALICE accepts everything optional, but extract
                    # household_size + year so phrasings like
                    # "ALICE for a 3-person family in 2020" route
                    # correctly even when the LoRA emits {}.
                    from bridge.tools._alice import extract_alice_args
                    _extracted = extract_alice_args(user_text)
                    if _extracted:
                        merged = dict(args)
                        for k, v in _extracted.items():
                            if not merged.get(k):
                                merged[k] = v
                        if merged != args:
                            log.info(
                                "alice arg fill: LoRA=%s → merged=%s",
                                args, merged,
                            )
                            args = merged
                args_key = _json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)
                if (name, args_key) in seen_calls:
                    log.info("duplicate tool call suppressed: %s(%s)", name, args_key)
                    result = (
                        f"[tool call suppressed — {name} was already called with "
                        "these arguments this turn. Answer from your own knowledge, "
                        "or try a different tool.]"
                    )
                elif tool_counts.get(name, 0) >= MAX_PER_TOOL:
                    log.info("per-tool cap hit for %s (args=%s)", name, args_key)
                    result = (
                        f"[tool call suppressed — {name} has been called "
                        f"{MAX_PER_TOOL} times already this turn and isn't "
                        "helping. Answer from your own knowledge, or use a "
                        "different tool.]"
                    )
                else:
                    seen_calls.add((name, args_key))
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    result = tools_mod.execute(name, args)
                    compose_tool_record.append({
                        "name": name,
                        "args": args if isinstance(args, dict) else {},
                        "result": result,
                    })
                    if on_tool_call is not None:
                        try:
                            on_tool_call(name, args if isinstance(args, dict) else {}, result)
                        except Exception as e:
                            log.warning("on_tool_call callback raised: %s", e)
                tool_msg = {
                    "role": "tool",
                    "name": name,
                    "content": _sanitize_tool_result(result),
                }
                messages.append(tool_msg)
                new_turn_messages.append(tool_msg)

        # Fell off the loop — model kept requesting tools past the cap.
        # Force a final sans-tools reply so the user gets something
        # useful, not the cryptic "gave up" stub.
        self._log_routing_decision(user_text, hint_tool, picked_tools)
        return self._forced_final_reply(messages, new_turn_messages, commit_history, user_text)

    def _resolve_hint_tool(
        self, user_text: str | None, has_tools: bool,
    ) -> tuple[str | None, bool]:
        """Pick the rescue hint tool for this turn.

        Returns ``(hint_tool, is_continuation)``:

          * ``hint_tool`` — non-None when either the regex classifier
            fired with confidence, OR the prompt is a continuation
            (``How about Chicago?``, ``Anything on tech?``) AND the
            most recent tool-calling assistant turn used a force-safe
            tool. The continuation fallback lets multi-turn follow-ups
            inherit the prior turn's tool when the LoRA under-fires.
          * ``is_continuation`` — True iff the hint came from the
            continuation fallback (signals the rescue to use
            ``default_args_continuation`` for short-form arg extraction
            and tells chat() to skip tool narrowing).

        Short-circuits to ``(None, False)`` when tools aren't offered
        this turn (chitchat guard stripped them) — no rescue is possible
        without tools.
        """
        if not has_tools:
            return None, False
        from bridge.routing import classify as _routing_classify
        hit = _routing_classify(user_text)
        if hit:
            return hit, False
        # Classifier abstained. Try fallbacks — both require the rescue
        # path to be enabled (otherwise we'd hint a tool the bridge
        # can't force-fire).
        if not self.under_fire_rescue:
            return None, False
        from bridge.routing.classifier import (
            _last_assistant_text,
            _last_assistant_tool_call,
            _looks_like_continuation,
            detect_clarification_followup,
        )
        from bridge.routing.force_fire import FORCE_RESCUE_TOOLS

        # Layer 2: clarification-followup. If the prior assistant reply
        # asked for tool-scoped info (e.g. "what's your household
        # composition?") and the current user reply is short, treat
        # the message as the answer to that clarification. Catches the
        # "Tell me about ALICE" → "Single, 1 person" misroute where
        # the LoRA picked get_time without context awareness.
        last_assistant_text = _last_assistant_text(self._history)
        clarify_tool = detect_clarification_followup(last_assistant_text)
        if clarify_tool and clarify_tool in FORCE_RESCUE_TOOLS:
            short = len((user_text or "").split()) <= 12
            if short:
                return clarify_tool, True

        # Layer 3: continuation fallback — re-use the prior turn's
        # tool when the new prompt is a short follow-up phrasing
        # ("how about X", "and Y"). Only fires when a tool actually
        # ran in the prior turn (we have something to inherit).
        if not _looks_like_continuation(user_text or ""):
            return None, False
        prior = _last_assistant_tool_call(self._history)
        if prior and prior in FORCE_RESCUE_TOOLS:
            return prior, True
        return None, False

    def set_two_phase_compose(self, enabled: bool) -> None:
        """Flip two-phase compose on/off at runtime.

        Exposed so the web layer can toggle it per-session without a
        restart (e.g. a "quality" vs "speed" setting in the UI). The
        flag is read on every turn; the next chat() call will use the
        new value.
        """
        self.two_phase_compose = bool(enabled)
        log.info("two_phase_compose set to %s", self.two_phase_compose)

    # Regexes for the compose output sanity check. If the focused compose
    # call STILL produces a schema dump, blatant meta-commentary opener,
    # or an obvious fabrication we shouldn't trust it over the stock
    # reply. Falling back lets _clean_reply + the rest of the pipeline
    # have a shot. Patterns fall into 3 buckets:
    #   (a) schema/meta leaks  — compose should never produce these
    #   (b) placeholder leaks  — "X.XX" etc slipping through
    #   (c) compound fabrication — specific patterns observed on Step 2
    #       where compose invented the missing half of an A+B query
    # Compose-output sanity-check patterns + the helper live in
    # bridge.compose. Re-exported as class attributes so the chat loop
    # can keep calling `self._compose_output_ok(...)` without rewrites.
    _COMPOSE_REJECT_PATTERNS = _compose.COMPOSE_REJECT_PATTERNS

    @staticmethod
    def _compose_output_ok(composed: str) -> bool:
        return _compose.output_ok(composed)

    def _compose_reply_no_tool(
        self,
        user_text: str | None,
        history: list[dict],
    ) -> str:
        """Thin wrapper around ``compose.compose_no_tool``. The body
        building + system prompt + post-call lives in ``bridge.compose``;
        this passes ``self`` so the compose path can read ``model`` /
        ``_options`` / ``_think`` and call ``_post_chat``."""
        return _compose.compose_no_tool(self, user_text, history)

    # Compose-pass per-tool suffixes + nonce scrubber live in
    # bridge.compose. Re-exported so existing inspect / monkeypatch
    # call sites keep working.
    _TOOL_COMPOSE_SUFFIX = _compose.TOOL_COMPOSE_SUFFIX
    _NONCE_SCRUB = _compose.NONCE_SCRUB

    def _compose_reply_from_tools(
        self,
        user_text: str | None,
        tool_record: list[dict],
    ) -> str:
        """Thin wrapper around compose.compose_from_tools. The body
        building + system prompt + post-call lives in bridge.compose;
        this passes self so the compose path can read model /
        _options / _think and call _post_chat."""
        return _compose.compose_from_tools(self, user_text, tool_record)

    def _try_force_fire_rescue(
        self,
        hint_tool: str,
        user_text: str | None,
        messages: list[dict],
        new_turn_messages: list[dict],
        picked_tools: list[str],
        seen_calls: set[tuple[str, str]],
        tool_counts: dict[str, int],
        on_tool_call: "Callable[[str, dict, str], None] | None",
        is_continuation: bool = False,
        compose_tool_record: list[dict] | None = None,
    ) -> bool:
        """Execute the classifier's hinted tool with heuristic args and
        inject the result into ``messages`` so the next chat() iteration
        produces a grounded reply.

        Returns True iff the rescue fired (tool executed, messages
        mutated). False means the rescue was declined (tool not in the
        force-rescue allowlist, no reasonable args, or tool raised) and
        the caller should proceed with the original no-tool reply.

        ``is_continuation``: when True, use
        :func:`default_args_continuation` to strip the continuation
        wrapper ("how about", "and", "actually") before extracting args.
        The regular :func:`default_args` extractors assume full prompts
        with "in/at/for" anchors and don't fire on short follow-ups.

        Side effects on success:
          * Executes ``hint_tool`` via ``tools_mod.execute``.
          * Appends a synthetic assistant(tool_call) + tool(result) pair
            to both ``messages`` and ``new_turn_messages`` (so the
            turn's history is consistent if ``commit_history=True``).
          * Records the rescue in ``picked_tools``, ``seen_calls``,
            ``tool_counts`` so the normal budget machinery treats it
            just like a model-emitted call.
          * Invokes ``on_tool_call`` so Karin's UI shows a tool fired.
        """
        from bridge.routing.force_fire import (
            default_args as _rescue_args,
            default_args_continuation as _rescue_args_cont,
        )
        from bridge import tools as tools_mod
        import json as _json

        if is_continuation:
            args = _rescue_args_cont(hint_tool, user_text or "")
        else:
            args = _rescue_args(hint_tool, user_text or "")
        if args is None:
            # Tool isn't in the force-safe set, or prompt was unusable.
            return False

        log.info(
            "under-fire rescue%s: hint=%s, no tool emitted by LLM — "
            "force-calling %s(%s) on prompt %r",
            " (continuation)" if is_continuation else "",
            hint_tool, hint_tool, args, (user_text or "")[:60],
        )
        try:
            result = tools_mod.execute(hint_tool, args)
        except Exception as e:
            log.warning("under-fire rescue: %s raised %s — skipping", hint_tool, e)
            return False

        # Record in the same bookkeeping the normal loop uses so the
        # next iteration's cap/dedupe logic knows we already called this.
        args_key = _json.dumps(args, sort_keys=True)
        seen_calls.add((hint_tool, args_key))
        tool_counts[hint_tool] = tool_counts.get(hint_tool, 0) + 1
        if hint_tool not in picked_tools:
            picked_tools.append(hint_tool)
        # Two-phase compose tracks rescues alongside model-emitted calls.
        if compose_tool_record is not None:
            compose_tool_record.append({
                "name": hint_tool,
                "args": args,
                "result": result,
            })

        # Surface the fire to the caller (UI streams "tool_call" events).
        if on_tool_call is not None:
            try:
                on_tool_call(hint_tool, args, result)
            except Exception as e:
                log.warning("on_tool_call callback (rescue) raised: %s", e)

        # Synthesize the assistant(tool_call) + tool(result) pair so the
        # LLM's next call sees a natural continuation. Ollama rejects
        # some shapes on re-submission: ``content: None`` with empty
        # string and ``arguments`` as a *dict* (not a JSON-dumped
        # string) is the shape the server-emitted tool_calls use, and
        # the one Ollama accepts when we echo it back.
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "rescue_0",
                "type": "function",
                "function": {
                    "name": hint_tool,
                    "arguments": args,
                },
            }],
        }
        tool_msg = {
            "role": "tool",
            "name": hint_tool,
            "content": _sanitize_tool_result(result),
        }
        messages.append(assistant_msg)
        new_turn_messages.append(assistant_msg)
        messages.append(tool_msg)
        new_turn_messages.append(tool_msg)
        return True

    def _forced_final_reply(
        self,
        messages: list[dict],
        new_turn_messages: list[dict],
        commit_history: bool,
        user_text: str | None = None,
    ) -> str:
        """One more /api/chat call WITHOUT tools so the model has to answer
        from what it's already collected. Used as the fail-soft when the
        tool-call loop hits its iteration cap.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options,
            "keep_alive": -1,
        }
        if self._think is not None:
            body["think"] = self._think
        # Deliberately no "tools" key — model can't request more tools.
        try:
            resp = self._post_chat(body)
            reply = (resp.json().get("message", {}) or {}).get("content", "") or ""
        except Exception as e:
            log.warning("forced-final-reply failed: %s", e)
            return "Sorry, I couldn't put together a clear answer for that."
        reply = self._clean_reply(
            reply,
            tools_were_offered=False,
            user_text=user_text,
            tools_fired=False,
            is_chitchat=self._looks_like_chitchat(user_text),
        ).strip()
        if not reply:
            return "Sorry, I couldn't put together a clear answer for that."
        if commit_history:
            self._history.extend(new_turn_messages)
            self._history.append({"role": "assistant", "content": reply})
        return reply

    # Reply-scrub fallback pools and forbidden-prefix list live in
    # bridge.reply_scrub. Re-exported as class attributes so existing
    # call sites (`cls._BOOKKEEPING_FALLBACKS`, etc.) keep working.
    _CHITCHAT_FALLBACKS = _reply_scrub.CHITCHAT_FALLBACKS
    _TOOL_FALLBACKS = _reply_scrub.TOOL_FALLBACKS
    _BOOKKEEPING_FALLBACKS = _reply_scrub.BOOKKEEPING_FALLBACKS
    _FORBIDDEN_PREFIXES = _reply_scrub.FORBIDDEN_PREFIXES

    # Reply-scrub helper functions live in bridge.reply_scrub.
    # Re-exported so `cls._is_json_stub(...)` etc. still work.
    _is_json_stub = staticmethod(_reply_scrub.is_json_stub)
    _scrub_forbidden_prefixes = staticmethod(_reply_scrub._scrub_forbidden_prefixes)
    _strip_bullet_markers = staticmethod(_reply_scrub._strip_bullet_markers)
    _normalize_dashes = staticmethod(_reply_scrub._normalize_dashes)
    _collapse_blank_runs = staticmethod(_reply_scrub._collapse_blank_runs)
    _clean_reply = staticmethod(_reply_scrub.clean_reply)
    _pick_fallback = staticmethod(_reply_scrub.pick_fallback)

    @staticmethod
    def _log_routing_decision(
        user_text: str | None,
        hint_tool: str | None,
        picked_tools: list[str],
    ) -> None:
        """Thin wrapper so the chat/chat_stream paths can log in one
        line without importing the routing module at each call site.
        Never raises — event logging is best-effort."""
        try:
            from bridge.routing import log_decision
            log_decision(user_text, hint_tool, picked_tools)
        except Exception as e:
            log.warning("routing decision log failed: %s", e)

    def chat_stream(
        self,
        user_text: str,
        tools: list | None = None,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        commit_history: bool = True,
        extra_system_suffix: str = "",
    ) -> str:
        """Like :meth:`chat`, but streams the FINAL assistant reply
        token-by-token through ``on_token``. Tool-call roundtrips still
        run synchronously (their JSON output isn't streamable in any
        useful way) — only the last, user-facing response is streamed.

        ``extra_system_suffix`` is appended to the system prompt for
        this turn only. Used by the retry-on-thumbs-down flow to pass
        a "don't repeat what you did last time" nudge without polluting
        future turns' prompts.

        Returns the full final reply (also passed token-by-token to the
        callback). History semantics match ``chat``.
        """
        # Chitchat guard (see chat() for rationale). Strip tools entirely
        # — partial stripping (data-fetch only) made some models emit
        # synthetic "no_tool" placeholders. Keep behavior identical to
        # chat() so the two paths can't drift again.
        if tools and self._looks_like_chitchat(user_text):
            log.info(
                "chitchat guard: withholding all %d tools for prompt %r",
                len(tools), (user_text or "")[:40],
            )
            tools = None

        # Pre-classifier hint + outcome tracking for the routing event log.
        hint_tool, hint_is_continuation = self._resolve_hint_tool(
            user_text, tools is not None,
        )
        picked_tools: list[str] = []

        _ALWAYS_AVAILABLE = {"update_memory"}
        # Narrowing is skipped for llamacpp: the LoRA reliably emits
        # the correct tool from the full catalog, and llama-server
        # strict-rejects (HTTP 500) any tool_call whose name isn't in
        # the request's `tools` array — so a too-aggressive narrowing
        # kills otherwise-correct routing. Continuation hints (see
        # chat() for rationale) also skip narrowing.
        if (
            hint_tool
            and not hint_is_continuation
            and tools
            and self.backend == "ollama"
        ):
            narrowed = [
                t for t in tools
                if (t.get("function", {}).get("name") == hint_tool
                    or t.get("function", {}).get("name") in _ALWAYS_AVAILABLE)
            ]
            if narrowed:
                log.info(
                    "tool narrowing: classifier=%s, %d->%d tools for %r",
                    hint_tool, len(tools), len(narrowed),
                    (user_text or "")[:40],
                )
                tools = narrowed

        import json as _json
        # Mirror of chat()'s hint-in-user-msg split. When the flag is on,
        # the routing hint moves to the start of the user message and the
        # system prompt stays byte-stable across turns (prefix-cache
        # friendly). History commit still gets the clean user text so
        # future-turn history prefixes are stable.
        if self.hint_in_user_msg:
            base_sys = self._stable_system_prompt(user_text)
            hint = self._routing_hint_text(user_text)
            user_content_with_hint = (
                f"{hint}\n\n{user_text}" if hint else user_text
            )
        else:
            base_sys = self._effective_system_prompt(user_text)
            user_content_with_hint = user_text
        sys_content = (
            f"{base_sys}\n\n{extra_system_suffix.strip()}"
            if extra_system_suffix and extra_system_suffix.strip()
            else base_sys
        )
        user_msg_clean = {"role": "user", "content": user_text}
        user_msg_sent = {"role": "user", "content": user_content_with_hint}
        messages = [
            {"role": "system", "content": sys_content},
            *self._llm_visible_history(),
            user_msg_sent,
        ]
        new_turn_messages = [user_msg_clean]
        # See chat() for the rationale — same dedup + per-tool cap here.
        seen_calls: set[tuple[str, str]] = set()
        tool_counts: dict[str, int] = {}
        MAX_PER_TOOL = self.max_per_tool
        MAX_TOOL_ITERS = self.max_tool_iters
        for iteration in range(MAX_TOOL_ITERS):
            # Probe non-streaming first to see if the model wants a tool;
            # only the iteration that produces the FINAL text gets streamed.
            body = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": self._options,
                "keep_alive": -1,
            }
            if self._think is not None:
                body["think"] = self._think
            if tools:
                body["tools"] = tools
            resp = self._post_chat(body)
            data = resp.json()
            msg = data["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                leaked = self._recover_leaked_tool_call(
                    msg.get("content", ""), tools, user_text=user_text,
                )
                if leaked is not None:
                    tool_calls = [leaked]
                    msg = {**msg, "tool_calls": tool_calls, "content": ""}

            if tool_calls:
                # Same as chat(): append assistant tool-call msg, execute
                # each tool, append role=tool results, loop.
                messages.append(msg)
                new_turn_messages.append(msg)
                from bridge import tools as tools_mod
                for call in tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name", "")
                    if name and name not in picked_tools:
                        picked_tools.append(name)
                    args = fn.get("arguments", {})
                    # Same say-arg override as the chat() path. See note
                    # at the chat() call site for rationale.
                    if name == "say" and isinstance(args, dict):
                        from bridge.tools._say import extract_verbatim_phrase
                        _phrase = extract_verbatim_phrase(user_text)
                        if _phrase and _phrase != args.get("text"):
                            log.info(
                                "say arg override (stream): LoRA=%r → extracted=%r",
                                args.get("text"), _phrase,
                            )
                            args = {**args, "text": _phrase}
                    # Wiki→inflation redirect (stream path). See chat()
                    # call site for rationale.
                    if name == "wiki" and isinstance(args, dict):
                        from bridge.tools._inflation import (
                            _WIKI_TO_INFLATION_REDIRECT_RE,
                            extract_inflation_args,
                        )
                        _wiki_q = str(args.get("query") or "")
                        if _WIKI_TO_INFLATION_REDIRECT_RE.search(_wiki_q):
                            _ext = extract_inflation_args(user_text)
                            if _ext.get("from_year"):
                                log.info(
                                    "wiki→inflation redirect (stream): "
                                    "wiki(query=%r) user=%r → inflation(%s)",
                                    _wiki_q, user_text, _ext,
                                )
                                name = "inflation"
                                args = _ext
                                if "inflation" not in picked_tools:
                                    picked_tools.append("inflation")
                    if name == "inflation" and isinstance(args, dict):
                        from bridge.tools._inflation import extract_inflation_args
                        _extracted = extract_inflation_args(user_text)
                        if _extracted:
                            merged = dict(args)
                            for k, v in _extracted.items():
                                if k == "measure" and v in ("wages", "both"):
                                    merged[k] = v
                                elif k == "item" and v:
                                    # Always override LoRA's item — the
                                    # extractor matches against an
                                    # exact known-keys list, while the
                                    # LoRA tends to invent free-text
                                    # item names ("gas, gallon" vs
                                    # canonical "gasoline").
                                    merged[k] = v
                                elif k == "region" and v and v != "us":
                                    # Always override when extractor
                                    # finds a non-US region keyword.
                                    # iter-3 doesn't know `region`
                                    # exists and defaults to omitting
                                    # it (→ "us"); explicit "in Hong
                                    # Kong" must win.
                                    merged[k] = v
                                elif not merged.get(k):
                                    merged[k] = v
                            if merged != args:
                                log.info(
                                    "inflation arg fill (stream): LoRA=%s → merged=%s",
                                    args, merged,
                                )
                                args = merged
                    if name == "population" and isinstance(args, dict):
                        from bridge.tools._population import extract_population_args
                        _extracted = extract_population_args(user_text)
                        if _extracted:
                            merged = dict(args)
                            for k, v in _extracted.items():
                                if k == "region" and v and v != "world":
                                    merged[k] = v
                                elif not merged.get(k):
                                    merged[k] = v
                            if merged != args:
                                log.info(
                                    "population arg fill (stream): LoRA=%s → merged=%s",
                                    args, merged,
                                )
                                args = merged
                    if name == "facts" and isinstance(args, dict):
                        from bridge.tools._facts import extract_facts_args
                        _extracted = extract_facts_args(user_text)
                        if _extracted:
                            merged = dict(args)
                            for k, v in _extracted.items():
                                if not merged.get(k):
                                    merged[k] = v
                            if merged != args:
                                log.info(
                                    "facts arg fill (stream): LoRA=%s → merged=%s",
                                    args, merged,
                                )
                                args = merged
                    if name == "alice" and isinstance(args, dict):
                        from bridge.tools._alice import extract_alice_args
                        _extracted = extract_alice_args(user_text)
                        if _extracted:
                            merged = dict(args)
                            for k, v in _extracted.items():
                                if not merged.get(k):
                                    merged[k] = v
                            if merged != args:
                                log.info(
                                    "alice arg fill (stream): LoRA=%s → merged=%s",
                                    args, merged,
                                )
                                args = merged
                    args_key = _json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)
                    if (name, args_key) in seen_calls:
                        log.info("duplicate tool call suppressed: %s(%s)", name, args_key)
                        result = (
                            f"[tool call suppressed — {name} was already called with "
                            "these arguments this turn. Answer from your own knowledge, "
                            "or try a different tool.]"
                        )
                    elif tool_counts.get(name, 0) >= MAX_PER_TOOL:
                        log.info("per-tool cap hit for %s (args=%s)", name, args_key)
                        result = (
                            f"[tool call suppressed — {name} has been called "
                            f"{MAX_PER_TOOL} times already this turn and isn't "
                            "helping. Answer from your own knowledge, or use a "
                            "different tool.]"
                        )
                    else:
                        seen_calls.add((name, args_key))
                        tool_counts[name] = tool_counts.get(name, 0) + 1
                        result = tools_mod.execute(name, args)
                        if on_tool_call is not None:
                            try:
                                on_tool_call(name, args if isinstance(args, dict) else {}, result)
                            except Exception as e:
                                log.warning("on_tool_call callback raised: %s", e)
                    tool_msg = {"role": "tool", "name": name, "content": _sanitize_tool_result(result)}
                    messages.append(tool_msg)
                    new_turn_messages.append(tool_msg)
                continue

            # Under-fire rescue: no tool call was emitted this iteration,
            # but the classifier has a confident hint and we haven't
            # rescued yet this turn. Force-execute the hinted tool and
            # loop back so the next iteration produces a grounded reply.
            # See chat() for the mirror logic + rationale.
            if (
                self.under_fire_rescue
                and hint_tool
                and not picked_tools
                and tools is not None
            ):
                rescued = self._try_force_fire_rescue(
                    hint_tool, user_text, messages, new_turn_messages,
                    picked_tools, seen_calls, tool_counts, on_tool_call,
                    is_continuation=hint_is_continuation,
                )
                if rescued:
                    continue

            # No tool calls — RE-RUN this same prompt with stream=True to
            # get the user-facing response token-by-token. We use the
            # already-cached probe result if it's a complete reply, or
            # restream for chunked delivery. Streaming the same context
            # twice is wasteful but Ollama caches the prefix, so the
            # second call is fast (just emits tokens).
            stream_body = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": self._options,
                "keep_alive": -1,
            }
            if self._think is not None:
                stream_body["think"] = self._think
            if tools:
                stream_body["tools"] = tools
            # Stream-time scrub: the model sometimes emits a reply that
            # _clean_reply would normally substitute (raw JSON stub like
            # {"name": "no_tool", ...}, or a forbidden "Note:" prefix).
            # If we stream tokens live as they arrive, the UI sees the
            # raw garbage BEFORE the end-of-stream cleanup can swap it
            # out. Fix: buffer the first chunk of the reply, peek at
            # its shape, and only start live-streaming once we're
            # confident it isn't a pattern cleanup would rewrite. If it
            # IS suspicious, hold the whole stream and emit the cleaned
            # reply in one shot at the end.
            BUFFER_UNTIL_CHARS = 32
            SUSPICIOUS_STARTS = (
                "{",  # JSON-stub leak
                "(note:", "note:", "[note:",
                "the output is", "the tool returned",
                "i called ",
                "here is the summary",
                "per the guidelines",
                "according to the tool",
            )

            with self._client.stream("POST", "/api/chat", json=stream_body) as sresp:
                sresp.raise_for_status()
                full: list[str] = []
                buffered: list[str] = []
                buffer_mode = True
                for line in sresp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = _json.loads(line)
                    except Exception as e:
                        # Ollama occasionally hands us a partial line
                        # mid-chunk; skipping is correct, but a debug
                        # line surfaces protocol-corruption issues if
                        # the rate ever spikes.
                        log.debug("malformed JSON in stream chunk: %s — %r",
                                  e, line[:80])
                        continue
                    delta = (chunk.get("message") or {}).get("content", "")
                    done_flag = bool(chunk.get("done"))
                    if delta:
                        full.append(delta)
                        if buffer_mode:
                            buffered.append(delta)
                            probe = "".join(buffered).lstrip().lower()
                            # Decide once we have enough text OR the
                            # stream ended (short replies).
                            if len(probe) >= BUFFER_UNTIL_CHARS or done_flag:
                                suspicious = any(
                                    probe.startswith(s) for s in SUSPICIOUS_STARTS
                                )
                                if not suspicious:
                                    # Reply looks fine — flush buffer
                                    # and resume live streaming.
                                    flush = "".join(buffered)
                                    buffered = []
                                    buffer_mode = False
                                    if on_token is not None and flush:
                                        try:
                                            on_token(flush)
                                        except Exception as e:
                                            log.warning(
                                                "on_token callback raised: %s", e,
                                            )
                                # else: stay in buffer_mode — cleanup
                                # will substitute a fallback at end.
                        else:
                            if on_token is not None:
                                try:
                                    on_token(delta)
                                except Exception as e:
                                    log.warning("on_token callback raised: %s", e)
                    if done_flag:
                        break
            reply = "".join(full)
            reply = self._clean_reply(
                reply,
                tools_were_offered=tools is not None,
                user_text=user_text,
                tools_fired=bool(picked_tools),
                is_chitchat=self._looks_like_chitchat(user_text),
            )
            # If we held the whole stream (suspicious start), emit the
            # cleaned reply now. This is the one-shot fallback path —
            # the UI sees a single emission instead of token-by-token,
            # but the content is correct.
            if buffer_mode and on_token is not None:
                try:
                    on_token(reply)
                except Exception as e:
                    log.warning("on_token callback raised on held stream: %s", e)
            if reply.strip() and commit_history:
                self._history.extend(new_turn_messages)
                self._history.append({"role": "assistant", "content": reply})
            self._log_routing_decision(user_text, hint_tool, picked_tools)
            return reply

        # Iteration cap exhausted with tools still being requested — fall
        # back to a sans-tools reply so the user gets something real. We
        # can't stream this one (we're past the chosen streaming moment),
        # but at least the returned string is a useful answer.
        self._log_routing_decision(user_text, hint_tool, picked_tools)
        reply = self._forced_final_reply(messages, new_turn_messages, commit_history, user_text)
        if on_token is not None:
            try:
                on_token(reply)
            except Exception as e:
                log.warning("on_token callback raised during forced-final: %s", e)
        return reply

    def reset(self) -> None:
        """Clear conversation history but keep the system prompt."""
        self._history.clear()

    def set_history(self, history: list[dict]) -> None:
        """Replace the rolling history wholesale.

        Used by the persistence layer on server boot (to restore the last
        conversation) and by the compactor (to swap in a summarized
        version). The system prompt is not stored in ``_history``, so
        callers should pass the same list shape ``chat()`` produces.
        """
        self._history.replace(history)

    def summarize(self, messages: list[dict], system_prompt: str) -> str:
        """One-shot summarization call, independent of the rolling history.

        Used by ``bridge.history.maybe_compact`` so compaction can shell
        out to the same Ollama server without touching ``_history`` or
        the main system prompt. Tool calls / results inside ``messages``
        are flattened to role=user / role=assistant since the summarizer
        doesn't need to replay them.
        """
        flat: list[dict] = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = m.get("role", "")
            content = m.get("content") or ""
            if role in ("user", "assistant") and isinstance(content, str):
                flat.append({"role": role, "content": content})
            elif role == "tool":
                flat.append({
                    "role": "user",
                    "content": f"[tool {m.get('name','?')} returned] {content}",
                })
        body = {
            "model": self.model,
            "messages": flat,
            "stream": False,
            "options": {**self._options, "temperature": 0.2},
        }
        if self._think is not None:
            body["think"] = self._think
        resp = self._post_chat(body)
        return (resp.json().get("message", {}) or {}).get("content", "") or ""

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "OllamaLLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
