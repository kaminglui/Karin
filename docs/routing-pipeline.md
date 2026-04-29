# The routing & reply pipeline — how, why, what we tried

How Karin gets from a user text to a user-facing reply, and why the pipeline has the specific shape it does. Complements [architecture.md](architecture.md) (audio / STT / TTS data flow) and [design.md](design.md) (project philosophy). Scope here is **strictly the LLM side**: tool routing, rescue layers, reply composition, and the experimental journey that led to the current production stack.

If you want to know "why is there both a regex classifier AND an under-fire rescue AND a two-phase compose step?", read this doc.

---

## TL;DR — current production pipeline

```
 user text
     │
     ▼
┌───────────────────────┐
│ L1 chitchat guard     │  Rule Zero: "hi"/"thanks"/"cool" → strip tools
└──────────┬────────────┘    → short plain-text reply, no tool call possible
           │ (tools preserved if not chitchat)
           ▼
┌──────────────────────────────────────────────────┐
│ L2 classifier stack (bridge/routing/)            │
│    L2a regex classifier        — routing_patterns│
│    L2b spaCy negation veto     — neg → trigger   │
│    L2c continuation fallback   — walks _history  │
│    L2d embedding fallback      — MiniLM-L6-v2    │
│    → returns (hint_tool, is_continuation)        │
└──────────┬───────────────────────────────────────┘
           │
           │─── hint_tool? ──────────────────┐
           │                                 ▼
           │                    ┌────────────────────────┐
           │                    │ L3 tool narrowing      │
           │                    │ restrict tool list to  │  only when L2a hit
           │                    │ hint + always-avail    │  & backend=ollama &
           │                    └────────────┬───────────┘  not continuation
           │                                 │
           ▼                                 ▼
┌───────────────────────────────────────────────────────┐
│ L4 LoRA call #1                                       │
│    Ollama /api/chat with tools=[narrowed or full]     │
│    LoRA picks: tool_calls=[…]  OR  plain content      │
└──────────┬────────────────────────────────────────────┘
           │
           ├─ emitted tool_calls?  ────────── yes ──┐
           │                                        ▼
           │                        execute tool, append
           │                        role=tool result, loop → L4
           │
           ├─ content looks like leaked JSON?       │
           │   L5 leak recovery                    yes
           │   _recover_leaked_tool_call       ────┘
           │   rebuilds structured tool_call
           │
           ▼ no tool emitted, no leak
┌───────────────────────────────────────────────────────┐
│ L6 under-fire rescue (bridge/routing/force_fire.py +  │
│    OllamaLLM._try_force_fire_rescue)                  │
│    fires only when:                                   │
│    • L2 produced a hint                               │
│    • hint is in the force-safe allowlist (wiki /      │
│      get_weather / get_time / get_news / get_alerts / │
│      get_digest / find_places / web_search / math*)   │
│      * math rescue only fires on unambiguous KL/      │
│        entropy distribution shorthand (N(0,1) ||      │
│        N(1,2), H(Beta(2,5))); all other math is LoRA  │
│        territory.                                     │
│    • picked_tools is still empty (one rescue per      │
│      turn — rescue appends to picked_tools, so the    │
│      next iter's check gates it out implicitly)       │
│    • args extracted heuristically (default_args or    │
│      default_args_continuation for pronoun-led turns) │
│    → injects synthetic (assistant+tool_calls, tool)   │
│      pair; loop returns to L4                         │
└──────────┬────────────────────────────────────────────┘
           │
           ▼ final answer
┌───────────────────────────────────────────────────────┐
│ L7 two-phase compose (default ON via config)          │
│    if ONLY passthrough tools fired (say):             │
│      reply = tool output verbatim — skip both the     │
│      LoRA stock reply AND L7a (the LLM may moralize); │
│      runs regardless of two_phase_compose flag        │
│    elif llm.two_phase_compose AND ONLY bookkeeping    │
│         tools fired (update_memory, schedule_reminder)│
│      skip L7a — phase-1 reply already absorbed the    │
│      side effect, no data needs grounding             │
│    elif llm.two_phase_compose AND tools fired:        │
│      L7a _compose_reply_from_tools() — fresh LLM call │
│          with ONLY user prompt + scrubbed tool output │
│          (no schema, no coaching)                     │
│    elif llm.two_phase_compose AND no tools fired:     │
│      L7b _compose_reply_no_tool() — casual-reply      │
│          compose; schema stripped, Karin identity pin │
│    sanity check: if compose output still schema-leaks │
│    → fall back to stock reply                         │
└──────────┬────────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ L8 reply post-processing (_clean_reply)               │
│    six scrubs applied to EVERY final reply, regardless│
│    of whether tools fired or the two-phase path ran:  │
│      · JSON-stub leak        · prompt-leak markers    │
│      · forbidden-prefix leak · market-fabrication     │
│      · capability-decline    · markdown/dash noise    │
│    see "L8 — reply post-processing" below for fire    │
│    conditions + class attributes                      │
└──────────┬────────────────────────────────────────────┘
           │
           ▼
     final reply → UI / TTS
```

Eight layers numbered L1–L8 (L2 is a composite of four sub-classifiers; L7 has two sub-compose paths; L8 scrubs run on every reply). None replace the LoRA — they all sit around it. Each layer was added in response to a specific failure class the layer above couldn't fix.

### Why layered instead of monolithic

Small on-device LLMs fail in characteristic ways that are cheap to detect and cheap to correct outside the model. Retraining for each failure class is expensive, slow, and risks regressing what works.

| Layer | Fixes | Cost | Toggleable? |
|---|---|---|---|
| L1 Chitchat guard | LoRA hallucinating tool calls on greetings / thanks | 0 (regex match, no LLM call) | No — hardcoded. Edit `_CHITCHAT_PATTERNS` / `_looks_like_chitchat` in `bridge/llm.py`. |
| L2a Regex classifier | Classifier abstains, LoRA trusts keyword cues too much | 0 (regex), +0 LLM latency | Partial — edit `routing_patterns` in `bridge/tools/_schemas.py` per tool. Always-on by default. |
| L2b spaCy veto | False-positive hints on negated triggers ("don't remind me") | ~10-30 ms when L2a hits | Auto — silent no-op if spaCy isn't installed. Remove `en_core_web_sm` to disable. |
| L2c Continuation fallback | Short pronoun-led follow-ups ("how about Chicago?") | ~0 (regex) | Coupled to L6 — only meaningful when rescue is enabled. |
| L2d Embedding fallback | Paraphrased prompts regex misses | ~30-80 ms when L2a abstains | Auto — silent no-op if fastembed / ONNX model isn't installed. |
| L3 Tool narrowing | LoRA picking the wrong tool even with a confident hint | 0 | Not toggleable — fires automatically when L2a is confident + backend=ollama + hint isn't a continuation. |
| L4 LoRA call | Core tool selection + args emission | 5-25 s | Not toggleable — this IS the LLM. `max_tool_iters` and `max_per_tool` in `config/models.yaml` cap it. |
| L5 Leak recovery | LoRA emitting JSON in `content` instead of structured `tool_calls` | 0 (post-process) | No — always on. Handles mlabonne-style tune output. |
| L6 Under-fire rescue | LoRA abstaining on clearly-tool-worthy prompts | +3-5 s per rescue (one extra tool execute + next-iter L4) | **YES — config flag.** `llm.under_fire_rescue: true` in `config/assistant.yaml`. Default true. |
| L7 Two-phase compose | schema_leak, refuses_own_tool, disavowal, you_meta, persona slip at compose time | +5-10 s per data-tool turn (one extra L4-sized call, same model). **0 s when only `update_memory` / `schedule_reminder` fire** (bookkeeping skip — `_BOOKKEEPING_TOOLS`). **0 s when only `say` fires** AND tool output replaces both LoRA + L7a (passthrough — `_PASSTHROUGH_TOOLS`). | **YES — config default + runtime override.** `llm.two_phase_compose: true` in `config/assistant.yaml` (default on since 2026-04-22). Runtime override via `POST /api/settings {"two_phase_compose": bool}` or sidebar "✨ Quality replies (slower)" — does NOT persist across restart. The two short-circuit sets in `bridge/llm.py` are not config-toggleable; edit the frozensets to add tools to either bucket. |
| L8 Reply scrubs (`_clean_reply`) | JSON-stub leak, forbidden-prefix leak, prompt-leak markers, market-fabrication, capability-decline, markdown/dash noise | 0 (post-process regex; ~ms) | No — always on. Add/remove a scrub by editing the class attributes in `bridge/llm.py` (`_PROMPT_LEAK_MARKERS`, `_MARKET_FABRICATION_PATTERNS`, `_CAPABILITY_REQUEST_PATTERNS`, `_CAPABILITY_SUCCESS_PATTERNS`). See "L8 — reply post-processing" below. |
| Hint placement | cache breakage from per-turn routing hint sitting at end of system prompt (forces re-prefill of history on every turn) | 0 (a message-construction detail, not a call) | **YES — config flag.** `llm.hint_in_user_msg: true` in `config/assistant.yaml` (default on since 2026-04-22). When off, hint goes to end of system prompt (legacy behavior). |

The alternative — one giant retrain that gets all of these right — is what iter-4 tried. It regressed (details below).

### Runtime toggle map — what to flip, where, and when

If you want to A/B a configuration or build a minimal pipeline, here's the cheat sheet:

| What | Where | Effect | Persists? |
|---|---|---|---|
| **L6 under-fire rescue off** | Edit `config/assistant.yaml`: `llm.under_fire_rescue: false`; restart bridge. | LoRA under-fires pass through as no-tool replies. ~-10 pp routing on 135-case eval (back to iter-3 level before Phase-0). | Yes, config-persistent. |
| **L7 two-phase compose — current default is ON** | `config/assistant.yaml: llm.two_phase_compose: true` (shipped 2026-04-22). | Adds 1 extra LLM call per tool turn. +19.2 pp manual reply-pass. +5-10 s/turn. | Yes, config-persistent. Runtime override via `/api/settings` or sidebar; runtime change does NOT persist across restart. |
| **Hint-in-user-msg — current default is ON** | `config/assistant.yaml: llm.hint_in_user_msg: true` (shipped 2026-04-22). | Routing hint moves from end of system prompt to start of user message. System prompt becomes byte-stable → KV prefix cache reusable. +~7 pp reply-pass, +10 pp tool-output usage in A/B. | Yes, config-persistent. |
| **L2b spaCy veto off** | Uninstall `spacy` OR remove `en_core_web_sm` model. | Negated triggers ("don't remind me") get routed to schedule_reminder. Lower precision. | Yes (dependency removal). |
| **L2d embedding fallback off** | Uninstall `fastembed` OR delete the MiniLM ONNX cache. | Paraphrased prompts regex misses fall through uncaught. -1-2 pp routing on 135-case eval (edge cases only). | Yes (dependency removal). |
| **L4 tool-loop caps** | Per-model tuning in `config/models.yaml`: `max_per_tool`, `max_tool_iters`. | Higher caps let 7B+ models chain tools; lower caps prevent 3B models from re-querying the same tool. | Yes, config-persistent. |
| **Classifier hints off entirely** | Comment out `routing_hint()` call in `OllamaLLM._effective_system_prompt()`. | LoRA sees no hint → reverts to iter-3's raw routing (71.1% on 135-case). Not a supported knob; for debugging only. | Manual code edit. |

**What's NOT toggleable** (and why): L1 chitchat guard (stops tool hallucination on "hi" — removing it breaks Rule Zero), L3 tool narrowing (automatic when L2a is confident — no reason to disable), L4 LoRA call (core), L5 leak recovery (cheap, no downside), L8 reply scrubs (cheap, the safety net that catches everything upstream missed).

**Toggle ordering for debugging.** When a reply looks wrong, flip layers off top-down until the bug disappears: (1) comment out individual L8 scrubs in `bridge/llm.py::_clean_reply` → is the scrubber mangling a legitimate reply? (2) turn L7 off → is the issue in compose? (3) set `llm.under_fire_rescue: false` → is the issue from a force-fired rescue? (4) check Ollama logs for what tool the LoRA actually emitted → is the issue in L4 routing vs L2 hint?

---

## Evolution summary

The detailed training and runtime-evolution narrative lives in [history/routing-evolution.md](history/routing-evolution.md). Short version:

1. Prompt-only routing hit a ceiling on small local models.
2. Iter-3 LoRA became the production base model.
3. llama.cpp, MLC, and later retraining attempts regressed or did not fit the Jetson constraints.
4. The shipped gains came from runtime layers: classifier hints, under-fire rescue, continuation handling, two-phase compose, and moving per-turn hints into the user message for better cache reuse.

The rest of this file describes the current mechanics those experiments produced.

## Cross-cutting concerns — stuff that runs around every layer

L8 is in the main flow diagram above; the other items below aren't — but they still shape every turn's behavior.

### L8 — reply post-processing (`_clean_reply`)
Every reply the pipeline returns goes through `OllamaLLM._clean_reply(reply, tools_were_offered, tools_fired=False, is_chitchat=False, user_text="")` before reaching the caller. Six failure modes, checked in order:
- **JSON-stub leak**: reply is a bare `{...}` parseable JSON. Substitute a persona fallback (one of `_CHITCHAT_FALLBACKS` or `_TOOL_FALLBACKS` depending on whether tools were on offer).
- **Forbidden-prefix leak**: scrub lines starting with banned phrases (`Note:`, `I called X`, `The output is`, `according to the tool`, `per the guidelines`).
- **Prompt-leak markers** (`_PROMPT_LEAK_MARKERS`): partial prompt-injection echoes like "my actual rules are", "stay in character as Karin", "no tool lookup for". Replace with persona fallback. Defense against a LoRA that's been nudged by an adversarial or noisy tool output into echoing its own system rules.
- **Market-fabrication scrub** (`_MARKET_FABRICATION_PATTERNS`): fires only when `is_chitchat=True AND tools_fired=False`. If a no-tool chitchat reply mentions prices, tickers, crypto, or "up/down X percent"-style claims, replace with persona fallback. Catches the old "how are you?" → "Bitcoin's at 67k, gold's up 2%" failure class where the LoRA confabulates market data on a greeting.
- **Capability-decline scrub** (`_CAPABILITY_REQUEST_PATTERNS` + `_CAPABILITY_SUCCESS_PATTERNS`, shipped 2026-04-24): two-step check. If (a) the user's prompt names a capability domain Karin doesn't have — smart-home/messaging/ordering/system-control/media — AND (b) the reply contains a first-person success confirmation ("I've sent it", "ordering now", "done!", "on the way") AND (c) no tool actually fired, replace with `"That one's outside what I can actually do — you'll have to handle it on your end."` Split into user-intent regex + reply-signal regex because a flat single-regex for verb+noun adjacency was too brittle (missed "front door", "large pizza", "-ing" forms, "on the way" openers).
- **Markdown / dash noise**: strip bullet prefixes (`- `, `* `, `1. `), replace em/en dashes with commas (dashes cause GPT-SoVITS to pause oddly), collapse runs of blank lines.

If cleanup empties the reply, substitute a rotating fallback keyed by `hash(user_text) % len(pool)` so the same prompt gets the same fallback and retries can see "this reply is broken" rather than random chitchat each time.

**Why the scrubs aren't one big regex.** Each scrub has a different fire condition: the JSON-stub scrub fires on any reply shape, the market-fabrication scrub only on no-tool chitchat, the capability-decline scrub needs both user-side AND reply-side signals. Merging them would force every call to pay every regex's cost and would couple their fire conditions. Keeping them separate means each can be tuned, disabled, or added without touching the others — and a miss in one doesn't blind the next.

### Routing-hint injection (system prompt vs user message)

Where the classifier stack's (L2) hint gets placed matters for prefix caching. Two placement modes:

**Legacy (`llm.hint_in_user_msg: false`)** — hint goes at the end of the system prompt via `_effective_system_prompt(user_text)`. Breaks prefix cache at the hint position on every turn because the hint is per-turn and not byte-stable; all subsequent tokens (history + new user turn) need re-prefilling.

**Current (`llm.hint_in_user_msg: true`, shipped 2026-04-22)** — system prompt is only the stable base + memory + location suffixes (`_stable_system_prompt()`); the per-turn hint is returned separately by `_routing_hint_text()` and prepended to the user message content. History commits still store the clean user text (hint stripped) so committed history is byte-stable too. Net: Ollama's KV cache can now reuse `[system] + [all prior history]` across turns. Side effect observed in A/B: +2-7 pp reply-pass + ~10 pp tool-output usage — the hint sitting right next to the user turn also helps the compose step ground in tool output.

The hint line itself is the same in either mode:

```
[routing rule: this prompt strongly matches the `get_weather`
tool — call it unless the prompt is clearly unrelated.]
```
Or on continuations:
```
[routing continuation hint: the prior turn called `get_weather`;
this short follow-up most likely wants the same tool again with
new arguments.]
```
Or on embedding fallback:
```
[routing hint (semantic): this prompt looks like a `find_places`
query — prefer calling it unless the prompt clearly wants
something else.]
```
The LoRA reads these lines of the system prompt just like it reads persona rules. Wording was iterated: the original "most likely maps to" was too soft for abliterated tunes (they'd read the hint and answer from memory anyway). The "call it unless the prompt is clearly unrelated" phrasing nudges compliance.

### Tool-loop safety (`chat` inner loop)
Three guards on every tool execution:
- **`seen_calls` dedup**: `(tool_name, args_json)` tuples are tracked; a repeat call gets a synthetic suppressed-result message (`[tool call suppressed — X was already called with these arguments this turn]`) instead of actually firing.
- **`tool_counts` per-tool cap**: `max_per_tool` (default 1) limits how many times a given tool can fire per turn regardless of args. 3B models sometimes re-query `wiki_search` with slight phrasing variants ("PA" → "Pennsylvania") burning the iteration budget; the cap kills this.
- **`_sanitize_tool_result`**: before appending a tool's output to `messages`, strip prompt-injection preambles (`ignore previous instructions`, `system: you are`, `override system/prompt`, etc.) via `_INJECTION_PATTERNS` regex. External sources (wiki, web_search) can return adversarial text; the sanitizer is the choke point. Also truncates to 4000 chars.

### Backend dispatch (`_post_chat`)
`OllamaLLM.chat()` and `chat_stream()` both call `_post_chat(body)` which dispatches on `self.backend`:
- **`ollama`** (default): POST `/api/chat`, direct response shape. One retry on 5xx (3 s delay) before raising.
- **`llamacpp`** (`_post_chat_llamacpp`): POST `/v1/chat/completions` (OpenAI-compat), translate body + response. Injects a synthetic `no_tool` shim function so llama-server's strict tool-call parser accepts the LoRA's trained "no tool needed" signal. On 500 "Failed to parse input", salvages the leaked blob into `message.content` so `_recover_leaked_tool_call` can have a go.
- **`mlc`** (`_post_chat_mlc`): POST `/v1/chat/completions` to `mlc_llm serve`. Flattens assistant `tool_calls` into content-style JSON stubs (MLC rejects `role=assistant` with `tool_calls`). Strips `tool_call_id` on `role=tool` messages (another MLC reject). Injects Ollama-style coaching at the last user message so the LoRA sees the nudge it was trained under.

The rest of the class is backend-agnostic once `_post_chat` returns — it expects a `{"message": {...}}` shape and the translators give it that.

### Streaming path (`chat_stream`)
The UI calls `chat_stream()`, not `chat()`. The tool-call loop runs identically; the divergence is the final-reply step. When the LoRA produces `tool_calls=[]` and the turn is ready to compose:
1. Run the same compose branch (two-phase or stock).
2. RE-run the same body with `stream=True`. Ollama caches the prefix so the second call is fast.
3. Token-by-token delivery via `on_token` callback, BUT:
4. **First-chunk buffering**: hold tokens in `buffered[]` until we accumulate 32 chars OR the stream ends. Peek at the prefix — if it starts with `{` (JSON stub) or a forbidden phrase (`Note:`, `The output is`, `here is the summary`, `according to the tool`), DON'T live-stream. Hold the whole stream, then emit the cleaned fallback at end in one shot.
5. If the prefix looks fine, flush the buffer to `on_token` and resume live streaming.

This buffering exists because a streaming UI would otherwise paint a schema leak to the user BEFORE `_clean_reply` can swap it out for a fallback. The 32-char probe + suspicious-starts heuristic is the compromise.

### Forced-final-reply safety net
If the tool loop exhausts `max_tool_iters` (default 5) without the LoRA producing a final content reply, `_forced_final_reply()` does one more call with NO tools on offer. The LoRA must reply from the data it has. Returns a persona fallback if even that call errors out. Prevents the cryptic "gave up" stub from reaching the user.

### Conversation history hygiene (`_llm_visible_history`)
The persistent `_history` keeps every `role=tool` message + every intermediate tool-call carrier (for UI widget replay across reloads). But the LoRA doesn't need yesterday's weather result in context for today's prompt. On each new turn, `_llm_visible_history()` filters `_history` down:
- Strips `role=tool` entries from PRIOR turns (within-turn tool messages are in the local `messages` list, unaffected).
- Strips intermediate `assistant` carrier messages (the ones with `tool_calls` but empty content).
- If `history_pairs > 0`, keeps only the last N user/assistant pairs (prevents cross-turn context bleed where the model latches onto a prior turn's topic).
- Compaction summaries (prefix `[compaction]`, role=system) are NEVER trimmed — they're the condensed context for older turns.

Tuning knob: `history_pairs` in `config/models.yaml`, per-model. 3B models use smaller windows to avoid topic-bleed; 7B+ can handle more.

---

## Current pipeline at the call level

```
 user text (from voice or UI chat)
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ OllamaLLM.chat(user_text, tools=TOOL_SCHEMAS, …)             │
│                                                              │
│  1. Chitchat guard                                           │
│     _looks_like_chitchat(user_text) → strip tools if match   │
│     ("hi"/"thanks"/"cool"/"how are you"/…)                   │
│                                                              │
│  2. Resolve hint_tool                                        │
│     _resolve_hint_tool(user_text, has_tools) returns         │
│     (hint, is_continuation)                                  │
│     Branches:                                                │
│       a) classify() returns a hit → (hit, False)             │
│       b) classify abstains, prompt looks like continuation,  │
│          _history has a force-safe prior tool                │
│          → (prior, True)                                     │
│       c) otherwise → (None, False)                           │
│                                                              │
│  3. Tool narrowing                                           │
│     if hint AND not continuation AND backend==ollama:        │
│       tools = [hint tool] + always-available (update_memory) │
│                                                              │
│  4. Tool loop (up to MAX_TOOL_ITERS = 5 iters, 1/tool cap)   │
│     while iter < cap:                                        │
│       body = {messages, tools, options, keep_alive=-1}       │
│       msg = _post_chat(body).message                         │
│       tool_calls = msg.tool_calls or []                      │
│       if not tool_calls:                                     │
│           # try to recover leaked JSON from content          │
│           leaked = _recover_leaked_tool_call(                │
│               msg.content, tools, user_text                  │
│           )                                                  │
│           if leaked: tool_calls = [leaked]                   │
│       if not tool_calls:                                     │
│           # under-fire rescue                                │
│           if hint AND in FORCE_RESCUE_TOOLS AND not already:  │
│               args = default_args(                           │
│                   hint,                                      │
│                   continuation=is_continuation               │
│               )                                              │
│               if args is not None:                           │
│                   execute tool; inject synthetic             │
│                   (assistant+tool_calls, role=tool) pair;    │
│                   continue loop                              │
│           # final answer branch                              │
│           reply = (compose OR msg.content) → clean → return  │
│       else:                                                  │
│           for call in tool_calls:                            │
│               if duplicate → suppressed result               │
│               elif over per-tool cap → suppressed result     │
│               else: execute; append role=tool msg            │
│                                                              │
│  5. Final answer: two-phase compose branch (final iter, no   │
│     more tool calls, exiting the loop)                       │
│     if self.two_phase_compose:                               │
│         if compose_tool_record (tools fired):                │
│             candidate = _compose_reply_from_tools(…)         │
│         else:                                                │
│             candidate = _compose_reply_no_tool(user_text,    │
│                 self._history)                               │
│         if _compose_output_ok(candidate):                    │
│             reply = candidate                                │
│     if not reply:                                            │
│         reply = msg.content                                  │
│     reply = _clean_reply(reply, tools_were_offered, …)       │
│     commit history; return reply                             │
└──────────────────────────────────────────────────────────────┘
```

Two tuning knobs that matter per model (`config/models.yaml`):
- `max_per_tool` — how often a given tool can fire in one turn (default 1). 3B models repeat tools; 7B chains them.
- `max_tool_iters` — how many LoRA/tool round-trips before we force a sans-tools final reply (default 5).

Rescue pipeline uses one rescue per turn regardless of iter budget.

---

## Design decisions — what we tried, what stuck

### Kept

1. **Regex classifier over semantic/embedding classifier.** Embedding fallback exists (`bridge/routing/embed_classifier.py`) and fires only when regex abstains. Regex is fast, deterministic, auditable, and cheap to extend.
2. **Force-safe allowlist.** Restricts rescue to 8 tools whose args can be extracted heuristically without catastrophic results. Wrong args on `convert` or `schedule_reminder` cause concrete harm; wrong args on `wiki` just return generic results.
3. **One rescue per turn.** Stops cascades. If the rescue tool call doesn't help, the next iteration gets the LLM's final answer without another rescue attempt.
4. **Synthetic `{content:"", arguments:dict}` shape.** Ollama's own tool-call response shape, echoed back verbatim. Using `content=null` or JSON-dumped `arguments` string triggers 400s on re-submission.
5. **Two-phase compose as toggle, default off.** Quality vs speed is a real tradeoff. Users on voice UX care more about responsiveness (~3-5 s total turn); users on text chat can afford the extra 5-10 s for polish.
6. **Per-model tuning.** `config/models.yaml` knobs (num_ctx, temperature, max_per_tool, think) let the same bridge code serve mannix Q4_K_M, qwen3.5-4B, llama3.1-8B with model-appropriate caps.

### Tried and discarded

1. **llama.cpp migration** — memory-blocked on Orin Nano. Infra kept for future retry.
2. **MLC-LLM migration** — the LoRA is template-coupled to Ollama's exact prompt shape. Any alternative backend needs re-training with that backend's template, which would require rebuilding the dataset pipeline.
3. **Iter-4 DPO single-turn flatten** — training distribution doesn't match serve-time distribution. Iter-5 plan explicitly uses `continue_final_message=True` with a trailing empty-content assistant turn to preserve multi-turn shape.
4. **GRPO as the iter-5 path** — "GRPO only refines patterns the model is uncertain on; it can't teach patterns missing from the training distribution." Our remaining failures are coverage gaps, not optimization problems.
5. **Heavy-handed prompting for edge cases** — past ~40 lines in `karin.yaml`, additional coaching regressed routing. Examples get copied verbatim into user-facing replies.

### Not tried (explicitly)

- **MCP** — a single-host personal assistant doesn't benefit from MCP's cross-process / cross-language model. Adds latency and a subprocess boundary for no unlock. See design principle #9 in [design.md](design.md).
- **Rewriting hot paths in C++** — our workloads are network-bound (external APIs) or already-C++ (llama.cpp under Ollama). Python's not the bottleneck.

---

## The full classifier stack

The "regex classifier" box in the pipeline diagram is actually three layered components. `bridge/routing/classifier.py::routing_hint()` is the integration point; it calls the others in order.

### 1. Regex classifier — `bridge/routing/classifier.py::classify()`
Per-tool `routing_patterns` from `bridge/tools/_schemas.py` compiled once at first call. Single pass over the text returns `{tool_name}` per matched pattern. Precedence rules applied in order:
1. **Explicit-verb priority** (`schedule_reminder` wins over passive-mention conflicts).
2. **Year-anchored wiki** (wiki wins over tracker when `\b\d{4}s?\b` is present).
3. **Alert priority** (`get_alerts` wins over `get_weather` on warning/advisory/hazard keywords).
4. **Compound "A and B" picker** (earliest-matching tool wins when ≥2 tools hit + `and` present).
5. Single-hit → return that tool; else abstain.

Microsecond-scale, deterministic, auditable. Catches ~80% of routing-worthy prompts.

### 2. spaCy structural veto — `bridge/routing/spacy_filter.py::should_veto()`
Runs *after* regex returns a hit. Parses the prompt with spaCy's dependency parser and suppresses the hint if the tool's trigger verb is directly negated. Example:

```
"Don't remind me, I already know."
   regex  → schedule_reminder (matches 'remind me')
   spaCy  → dep parse sees neg → remind (the root verb 'remind'
            carries a 'not' negation marker)
   result → hint suppressed; LLM sees no hint; replies in prose
```

Per-tool negation targets (`_TRIGGER_LEMMAS` in the file):
- `schedule_reminder` → remind/alarm/wake/nudge/set
- `update_memory` → remember/call
- `get_weather` → worry ("don't worry about the weather")
- `get_news` → tell ("don't tell me the news")
- `find_places` → find/buy

Lazy-loaded on first call (~200 ms spaCy import cost paid once), parser+tagger only (NER and textcat disabled for latency). Graceful degradation: if spaCy fails to load the filter becomes a no-op — routing keeps working. Adds ~10-30 ms per call after warm-up.

### 3. Embedding fallback — `bridge/routing/embed_classifier.py::embed_classify()`
Runs when regex abstains AND no continuation fallback resolves. Encodes the user prompt with MiniLM-L6-v2 via fastembed/onnxruntime (CPU-only, so it doesn't contend with Ollama for the Jetson's GPU), scored against a hand-curated anchor corpus in `_ANCHORS` (6-10 phrases per tool covering intent space). Picks the tool with max cosine similarity if the score clears a threshold AND beats the runner-up by a margin.

Catches paraphrased prompts the regex misses ("Where's a decent ramen spot?" → find_places, "DIY bookshelf plans" → web_search). Abstains silently when nothing clears thresholds. Adds ~30-80 ms per call.

**Order of fallbacks in `routing_hint()`:** regex → (spaCy veto) → if regex abstained: continuation fallback from `_history` → embedding fallback → give up (no hint). First hit wins; later layers are skipped.

---

## Pipeline performance breakdown

Actual per-case latencies from the 135-case eval on the Jetson Orin Nano (karin-tuned:latest in Ollama alongside STT + TTS models loaded). Numbers are from [evals/eval_phase0_round2_step2.json](evals/eval_phase0_round2_step2.json) and [evals/eval_phase0_round2_step2_twophase.json](evals/eval_phase0_round2_step2_twophase.json).

### Per-turn latency (observed on eval)

| Turn type | No two-phase | + two-phase | Delta |
|---|---|---|---|
| no-tool turn (median) | 16.1 s | 16.5 s | +0.4 s |
| no-tool turn (max) | 18.0 s | 38.2 s | +20 s |
| tool-fire turn (median) | 24.8 s | 32.3 s | +7.5 s |
| tool-fire turn (max) | 69.1 s | 101.0 s | +32 s |
| all turns (mean) | 22.3 s | 27.3 s | +5.0 s |

No-tool turns mostly unaffected by two-phase (the chitchat guard strips tools → no casual-compose fires on matched chitchat; short pronoun-less prompts that don't look like continuations skip compose). Tool-fire turns pay the full +5-10 s compose cost predictably. Tail latency (max) gets worse by ~30 s with two-phase — long stub outputs produce long compose calls.

### Per-component costs — latency + resource accounting

Full breakdown of what each pipeline step costs, in each of the four constraint dimensions that matter on a Jetson Orin Nano (7.4 GiB usable unified memory — spec says 8 GB, userspace sees 7607 MiB; integrated GPU, 6-core Cortex-A78AE CPU at 1.7 GHz, shared RAM between CPU and GPU).

| Step | RAM (CPU-side) | VRAM / unified | CPU load | GPU load | Network | Latency |
|---|---|---|---|---|---|---|
| L1 Chitchat guard | ~1 KB compiled patterns | — | brief string match | — | — | <1 ms |
| L2a Regex classifier | ~10 KB compiled patterns | — | single-pass scan + priority rules | — | — | <1 ms |
| L2b spaCy negation veto | ~20 MB (en_core_web_sm parser + tagger; NER / textcat disabled) | — | dep parse ~1-5 K words/s | — | — | 10-30 ms, only when regex hit |
| L2c Continuation fallback | — | — | history walk + regex match | — | — | <1 ms |
| L2d Embedding fallback | ~90 MB (MiniLM-L6-v2 ONNX + sentence-transformer tokenizer) | — | ONNX inference on CPUExecutionProvider | — | — | 30-80 ms, only when regex abstains |
| L3 Tool narrowing | — | — | list comprehension | — | — | <1 ms |
| **L4 LoRA call #1** (pick tool + args) | ~50 MB (httpx client + message buffer) | **~4.5 GB** (mannix Q4_K_M + iter-3 LoRA adapter, `keep_alive: -1`) | idle during inference | **saturated** | localhost loopback to Ollama | **5-25 s** |
| L5 Leak recovery | — | — | JSON parse + regex | — | — | <1 ms |
| Tool execution (real) | varies | — | tool module code | — | **external API** (weather / wiki / news) or local SQLite | 100 ms – 3 s |
| Tool execution (eval stub) | — | — | negligible | — | — | <10 ms |
| L4 LoRA call #2 (default compose, schema still in context) | ~50 MB | same 4.5 GB (model stays loaded) | idle | **saturated** | loopback | **5-15 s** |
| L6 Under-fire rescue | ~10 KB (regex extractors in force_fire.py) | same 4.5 GB | arg-extraction regex | — (execute only — L4 inference is next iteration) | external APIs (when a real tool fires) | +3-5 s total |
| L7 Two-phase compose call (if toggled on) | ~20 MB (separate message list + NONCE-scrub regex) | same 4.5 GB (reuses loaded LoRA) | idle | **saturated** | loopback | **+5-10 s median, +30 s tail** |

### Why this shape, given the Orin Nano constraints

**1. Unified memory is the hardest ceiling.** The Orin Nano shares a single 7.4 GiB usable pool between CPU and GPU (see [architecture.md § Memory budget](architecture.md#memory-budget) for the full measured table). Default text-only deploy is already ~6.8 GiB resident at idle, with karin-tuned alone taking ~5 GB. Enabling STT + local sovits on top would project to ~9-10 GiB — over the cap, so sovits lives on a PC via Tailscale offload.

That directly shapes several routing-pipeline decisions:

- **Two-phase compose reuses the SAME Ollama model**, not a separate smaller one. Loading a second model (even a 4B judge) would evict LLM weights or OOM the system. The compose path is a second `POST /api/chat` to the same warm model with different messages — zero incremental VRAM, at the cost of one extra forward pass.
- **spaCy + embedding models live in CPU RAM, not on the GPU.** MiniLM-L6-v2 would fit in ~90 MB of VRAM, but running a second ONNX session on CUDA would either evict the LLM's KV cache or trigger OOM. CPU is "slow" (80 ms per call) but free in the VRAM accounting.
- **spaCy ships without NER/textcat.** Parser + tagger only (~20 MB). Full pipeline is ~50 MB; the extras aren't needed for dep-based negation detection.
- **`keep_alive: -1` on every chat call.** Forces Ollama to keep the model resident between turns. A cold load takes ~10 s. Paying that once per service start vs once per turn is the difference between a working assistant and a 15-second warm-up for every greeting.

**2. GPU is the latency dominant.** Every step marked "saturated" above is an Ollama inference call that fully occupies the integrated GPU. The classifier stack (regex + spaCy + embedding) is deliberately CPU-only so it can run concurrently with Ollama's warm-up (first request after idle is still faster than without the warm path) and so the GPU isn't contended.

**3. CPU is ample, so we don't optimize it aggressively.** The 6-core A78AE is idle during LLM inference (inference is GPU-bound). That's why we can afford 80 ms of ONNX embedding work — it's on a resource that would otherwise be wasted. Same for spaCy's dep parse.

**4. Network is loopback-or-external.** The LLM, STT, and TTS all live on `127.0.0.1` (no external LLM API). The "external" network cost only shows up in real tool execution (`get_weather`, `get_news`, `wiki`, `web_search`, `tracker`). Those can be flaky and slow in ways Ollama isn't — `fail-soft` principle (returns `[]` / `None` / friendly-string on failure, see [design.md](design.md) § 4) keeps individual tool outages from cascading.

### Latency breakdown — observed on 135-case eval

Actual per-case latencies, Jetson Orin Nano, karin-tuned:latest loaded:

| Turn type | No two-phase | + two-phase | Delta |
|---|---|---|---|
| no-tool turn (median) | 16.1 s | 16.5 s | +0.4 s |
| no-tool turn (max) | 18.0 s | 38.2 s | +20 s |
| tool-fire turn (median) | 24.8 s | 32.3 s | +7.5 s |
| tool-fire turn (max) | 69.1 s | 101.0 s | +32 s |
| all turns (mean) | 22.3 s | 27.3 s | +5.0 s |

No-tool medians unchanged because the chitchat guard short-circuits most common no-tool prompts before the compose path. Tool-fire medians pay the full +5-10 s compose cost. Tail max grows ~30 s with two-phase — long tool-output text → long compose prompt → long compose response.

Sources: [evals/eval_phase0_round2_step2.json](evals/eval_phase0_round2_step2.json) and [evals/eval_phase0_round2_step2_twophase.json](evals/eval_phase0_round2_step2_twophase.json).

### End-to-end voice turn

Adding STT and TTS from [architecture.md](architecture.md):

| Stage | Cost | Where |
|---|---|---|
| STT (faster-whisper, default `tiny.en` CPU int8) | 0.3-1.0 s | ~400 MB RAM. `base.en` adds ~300 MB and ~3 pp WER improvement; only fits on Orin NX 16 GB+. Remote (PC-STT offload) = 0 on Jetson, +50-150 ms Tailscale round-trip. |
| Bridge pipeline (this doc) | 16-32 s typical, +5-10 s with two-phase | ~5.0 GB unified (karin-tuned weights + KV) + ~110 MB RAM (classifier stack) |
| TTS (GPT-SoVITS v2Pro) | 2-8 s depending on reply length | ~1.5-2 GB unified if `LOCAL_SOVITS=yes`. Remote (PC-TTS) = 0 on Jetson. |
| **End-to-end voice turn** | **~20-40 s** default, **+5-10 s with two-phase** | ~6.8 GB / 7.4 GiB with text-only defaults; ~9+ GB if both STT+TTS are on-device → OOM on Orin Nano, use PC offload. |

Sub-optimal for a voice assistant but acceptable for a Jetson Orin Nano class device. Text-chat UX without TTS is ~15-30 s end-to-end.

### What optimizations we explicitly rejected

- **Running a second Ollama model (e.g. qwen3:4b) as the two-phase "judge".** Would nearly double VRAM and either evict the primary LoRA between turns (10 s warm-up penalty) or OOM. Reusing the loaded LoRA at the cost of one extra forward pass is strictly cheaper.
- **GPU-accelerated embedding fallback.** ONNX + MiniLM would fit in ~90 MB of VRAM in isolation but contends with LLM KV cache. CPU at 80 ms is fine when the next step is a 10-second LLM call anyway.
- **Rewriting hot paths in C++** (see [design.md](design.md) § 9 and § 10). LLM inference is already C++ via llama.cpp / Ollama. Tool execution is network-bound, not CPU-bound. Our Python overhead in the bridge process is microseconds and invisible next to the LLM round-trips.

---

## Evaluation methodology

### The eval set
`sft/eval_cases_novel.yaml` — 135 cases, held-out from training (`sft/scripts/assert_disjoint.py` enforces zero verbatim overlap with `sft/phrase_library/train/`).

Cases cover:
- Direct tool triggers (weather, time, math, convert, wiki, tracker, news, alerts, digest, graph, circuit, find_places, web_search, update_memory, schedule_reminder)
- Decoys and idioms ("call me when you're free", "gold standard of debugging", "schedule conflicts are the worst")
- Compound queries ("weather and news", "convert X and the news")
- Multi-turn follow-ups (with `history` field: "Weather in Tokyo." → "How about Chicago?")
- Chitchat + greetings + farewells
- Gap-tool probes (graph, circuit, update_memory — tools with less training coverage)

### Stub-based eval (`scripts/eval_routing.py::_stub_execute`)
Real tool execution would hit external APIs (weather services, wiki, search). For 135 cases × eval re-runs, that's not feasible. So tools are stubbed with sentinel-containing placeholder text:

```
get_weather → "42 degrees, partly cloudy, humidity 83%, wind 17 mph [NONCE_xxx]"
get_time    → "13:37:42 EST on Sunday, November 23, 2099 [NONCE_xxx]"
wiki        → "Summary: notable 20th-century figure, born 1842 [NONCE_xxx]"
math        → "Result: 42 [NONCE_xxx]"
```

The NONCE is a per-case random sentinel. Downstream checks whether the NONCE (or any distinctive number from the stub) appears in the final reply — if yes, the LoRA used the tool output; if not, it ignored the tool and answered from memory.

### The stub-artifact trap
Generic stubs produce specific artifacts. The wiki stub "notable 20th-century figure, born 1842" associates (for this LoRA) with Leon Trotsky — every wiki call's reply starts with "Leon Trotsky was born in 1879, not 1842…" regardless of whether the user asked about Mount Fuji or Marie Curie or the Berlin Wall. The math/convert/web_search stubs all contain "42" which triggers Hitchhiker's Guide jokes.

**These are eval artifacts, not production bugs.** Real tool data in production wouldn't reproduce them. The `reply_flags` scanner classifies `trotsky_artifact` and `hitchhiker_joke` as "stub_artifact" (surfaced but NOT counted against reply-pass). Manual Claude judge applies the same discounting.

Future improvement (in iter-5 plan): topic-appropriate stubs (`wiki(query="Berlin Wall")` returns Berlin-Wall-flavored placeholder) OR LLM-as-judge via `scripts/eval_reply_quality.py`.

### Three metrics, not one

| Metric | What it measures | How computed |
|---|---|---|
| `routing_pass` | Right tool with acceptable args | `scripts/eval_routing.py::grade` |
| `reply_pass (regex)` | Routing + no phrase-pattern reply bugs | `scan_reply_flags()` in same file |
| `reply_pass (manual)` | Routing + faithful to tool + in-character + actually addresses user | Claude reads every reply per `feedback_eval_read_replies.md` |

All three always reported together. Single-number reporting ("92.6% pass rate!") hides the gap.

---

## Historical next-step notes

The old iter-5 planning notes are historical now and live in [history/routing-evolution.md#former-iter-5-plan](history/routing-evolution.md#former-iter-5-plan). Current work should be based on the production pipeline and the latest eval results, not that archived plan.

## Related reading

- [architecture.md](architecture.md) — audio / STT / TTS data flow, VRAM budget, half-duplex design.
- [design.md](design.md) — 12 project-level design principles (active vs passive, code-is-truth, fail-soft, etc.).
- [eval_135_manual_review.md](eval_135_manual_review.md) — per-iteration manual review of replies + the bifurcation section tracking all score history.
- [routing_eval_comparison.md](routing_eval_comparison.md) — cross-backend + cross-iteration metric table.
- [tools.md](tools.md) — 21-tool catalog (20 active by default), widget rendering, recipe for adding a tool.
- [api-reference.md](api-reference.md) — `assistant.yaml` schema + every HTTP route including the new `/api/settings`.

Per-iteration memory notes (in `~/.claude/projects/.../memory/`): `project_sft_iter1_regression.md`, `project_routing_ceiling.md`, `project_llamacpp_path1_inflight.md`, `project_llamacpp_retry_plan.md`, `project_iter4_postmortem.md`, `project_phase0_rescue.md`, `project_phase0_round2.md`, `project_iter5_plan.md`, `feedback_eval_read_replies.md`, `feedback_qwen3_rejected.md`.
