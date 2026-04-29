# Routing evolution history

This file holds the historical journey split out of [../routing-pipeline.md](../routing-pipeline.md): the training iterations, backend experiments, runtime rescue layers, and reply-quality work that led to the current pipeline.

Read [../routing-pipeline.md](../routing-pipeline.md) first if you only need the current production mechanics.

## The problem

Karin is a local personal voice assistant running on a Jetson Orin Nano 8 GB. Routing a user's spoken intent to one of 16 tools (weather, news, wiki, math, schedule_reminder, say, etc.) needs to work well with a 4-8 B-parameter quantized LLM running alongside STT + TTS in ~7.4 GiB of usable unified memory. Cloud LLMs aren't an option (privacy + offline requirement).

Constraints that shape the pipeline:
- **4-8 B LLMs over-fit on keyword cues.** "Call me when you're free" triggers `schedule_reminder` on the word "remind". "The weather has been weird lately" triggers `get_weather` on the word "weather". Pure LoRA SFT can only partially counter this.
- **Token budget is tight.** Tool schemas + coaching + user system prompt eats ~2100 tokens before the user even speaks. Every extra coaching phrase for edge cases pushes us toward `num_ctx`.
- **Retraining is expensive.** A Colab SFT run is ~30 min wallclock + manual eval after. Iteration cost is hours. Runtime fixes iterate in seconds.
- **Replies must be voice-friendly.** The LoRA's output goes to GPT-SoVITS. Long schema dumps, nonce tokens, and meta-commentary produce terrible audio.

---

## Starting point — mannix 79.2% (prompt-engineering ceiling)

The base model is `mannix/llama3.1-8b-abliterated:tools-iq4_xs` — a Llama 3.1 8B abliteration with `tools` mode enabled. Before any LoRA, we tried to get routing quality out of pure prompt engineering.

**7 eval runs on a 38-case novel set.** Best pass rate: **30/38 = 79.2%**. Each run added more coaching to `karin.yaml`: decoys, examples, phrase-to-tool mappings, priority rules.

**What we learned.** Past ~40 lines of coaching, additional rules *regressed* routing. The LoRA copied example phrasings verbatim into user-facing replies ("Based on your prompt matching the `schedule_reminder` pattern…"). Heavy prompts also bloated context and made responses slower. **79.2% was a prompt-engineering ceiling**, not a LoRA ceiling.

Decision: move to LoRA SFT. Memory `project_routing_ceiling.md` captures this.

---

## Iter-1 through iter-3 — LoRA SFT evolution

### Iter-1 (`run_d94ae3e5`) — 60.5% (regressed)
- 294 SFT + 40 DPO examples.
- HPs: r=16, epochs=3, lr=3e-4, dropout=0.05, no weight decay.
- **Bug**: mlabonne's chat template silently drops `tool_calls=[...]` and renders `content=None` as the literal string "None". Trained the LoRA to output "None." on tool turns.
- Score on 38 cases: 23/38 = 60.5%. **Worse than mannix baseline.**

### Iter-2 (`run_47b09b19`) — 71.1%
- Same data, but `sft/scripts/build_dataset.py` now flattens `tool_calls` into `content` as plain JSON.
- Tool calls emit correctly but LoRA still overfit keyword-to-tool associations.

### Iter-3 (`run_0ac17bc7`) — 89.5% ← became production
- Same 294 SFT + 40 DPO.
- **Anti-overfit HPs**: r=8, epochs=2, lr=1e-4, dropout=0.1, weight_decay=0.01.
- 21 M trainable params (half of iter-1). Forced the adapter toward smaller updates.

**Result: 34/38 = 89.5% on the 38-case set, +10 pp over mannix.** Shipped as `karin-tuned:latest`.

### Anti-overfit lesson
The checkpoint picker in `sft/colab_sft.ipynb` cell 5b scores each checkpoint on 40 held-out prompts and promotes by routing accuracy. For iter-1/iter-2 (permissive HPs), the picker correlated positively with held-out evals — pick the highest-scoring checkpoint, get the best production. For iter-3 (regularized HPs), the picker **anti-correlated**: it promoted the more-overfit checkpoint, which generalized WORSE than `load_best_model_at_end`'s eval-loss pick.

Captured in `project_sft_iter1_regression.md`. Directly informed the picker-caveat section in `colab_sft.ipynb`.

---

## Serving-backend experiments (all failed)

Before building runtime layers on top of Ollama, we tried alternative serving paths. None worked.

### llama.cpp — 76.3%, blocked on Jetson memory
- Plan: use `llama-server` with a custom jinja chat template (copied from mannix's Modelfile TEMPLATE) for more flexible prompt shaping + lower per-request overhead than Ollama.
- **Outcome**: 29/38 = 76.3% on 38-case set (-13 pp vs Ollama). Template shape worked in isolation but tool-schema injection pushes context above 4 KB. Raising `--ctx-size` exhausts the Orin Nano's 7.4 GiB of usable unified memory during the prompt cache save — llama-server OOMs and restarts.
- Infrastructure kept (`deploy/systemd/karin-llama.service` + jinja template file) for future retry when tool schema condenses or hardware upgrades.
- Full V1/V2/V3 template-variant plan + phased rollback lives in `project_llamacpp_path1_inflight.md` + `project_llamacpp_retry_plan.md`.

### MLC-LLM — 57.9%, distribution mismatch
- Plan: `mlc_llm serve` with merged LoRA (fp16 bake-in) for tighter inference than Ollama's LoRA-on-base.
- **Outcome**: 22/38 = 57.9%. MLC's `llama-3_1` conv_template defaults to `use_function_calling=false` and doesn't inject a JSON-format coaching block. The LoRA emits natural-language replies on tool-worthy prompts because the trained "emit a JSON stub" signal isn't in context.
- Fixed via a hard-coach template injection (prepends the coaching sentence + tool-name list to every user message). Score unchanged: still 57.9%.
- **Root cause**: the LoRA was trained against Ollama's exact TEMPLATE shape. Any prompt-shape drift degrades routing below baseline.
- Abandoned. `project_llamacpp_path1_inflight.md` has the per-backend failure breakdown.

### Iter-4 retraining — 74.1% routing, 48.2% tool-output (rolled back)
Faced with Ollama + iter-3 scoring only 71.1% on the larger 135-case set (not the 89.5% from the easier 38-case set), the response was to retrain.

**Additions**: +80 SFT (compound, historical, multi-turn, anti-widget, gap-tools) + 40 DPO (tool-output-usage, force-fire). Same HPs as iter-3. Run = `run_52227b4c`.

**Outcome**:
- Routing 100/135 = 74.1% (+3.0 pp).
- **Tool-output usage 27/56 = 48.2% (-8.7 pp)**.
- Net: not a win. Rolled back to iter-3.

**Root cause 1 — DPO single-turn flatten.** TRL rejected multi-turn DPO pairs with `role: tool` as the last prompt message. The workaround was to flatten pairs into single-turn prompts of shape `"User asked: X. The Y tool returned Z. Reply to the user."`. This trains on a distribution that doesn't exist at serve time: in production, the LoRA sees a proper multi-turn history with `role=tool`, not a flattened description. The preference signal didn't transfer. **Tool-output usage regressed 8.7 pp.**

**Root cause 2 — over-generalization of SFT additions.** The `schedule_reminder` gap-fill examples taught the LoRA to fire `schedule_reminder` on any prompt containing "remind", "call", "schedule", "when". Two regressions: "Call me when you're free" → schedule_reminder (idiom); "Tell me about my schedule today" → update_memory (unrelated).

**Root cause 3 — classifier work did most of the lifting.** 5 of the 6 iter-4 case-level gains came from classifier-pattern additions in `bridge/` that landed concurrently — zero training time, would apply to iter-3 just as well.

Detailed in `project_iter4_postmortem.md`. The takeaway: retraining is risky AND classifier/runtime work is high-leverage. Led directly to Phase-0.

---

## Phase-0 — runtime layer (shipped 2026-04-19)

After iter-4 rolled back, the decision was to stop retraining and double down on runtime. Two pieces:

### Classifier patch set (`bridge/routing/classifier.py` + `bridge/tools/_schemas.py`)
7 new regex patterns covering failure categories from the iter-3 review:

| Pattern | Tool | Example prompt |
|---|---|---|
| `\bin\s+\d{4}s?\s+(money\|dollars?\|value\|terms\|america\|europe)?\b` | wiki | "How much is a dollar in 1970 money?" |
| `(when\|once\|after\|as\s+soon\s+as\|if)\s+.+,?\s+(remind\|ping\|nudge\|wake\|alert\|tell)\s+(me\|us)\b` | schedule_reminder | "Once I finish dinner, nudge me about my pills" |
| `\bwhich\s+(costs?\|is)\s+(more\|higher\|bigger\|worth\s+more)\b.*(gold\|silver\|btc\|bitcoin)` | tracker | "Which costs more, gold or silver?" |
| `(can\|should\|will)\s+i\s+(bbq\|grill\|picnic\|hike\|run)\s+(outside\|today\|tomorrow)` | get_weather | "Can I BBQ outside today?" |
| `\bsolve\b[^.\n]*\bequals?\s+\d` | math | "Solve 3x²+5x-2 equals 0" |
| `chart\s+(the\s+)?(last\|next\|past)\s+\d` | graph | "Chart the last 7 days of temperatures" |
| `\bagenda\s+(today\|this\s+morning)` | get_digest | "Run through what's on the agenda today" |
| `\b(current\|latest\|today's\|present)\s+(prime minister\|president\|ceo\|king\|queen\|…)\b` † | wiki | "Who's the current prime minister of the UK?" |
| `\bwho(?:'s\|\s+is)\s+(?:the\s+)?(?:current\|latest\|…)\s+\w+` † | wiki | "Who is the current president of Argentina?" |
| `(president\|prime minister\|ceo\|…)\s+(right now\|today\|currently\|these days)\b` † | wiki | "PM of Canada right now?" |

† Added 2026-04-24 to fix case 26 (stale parametric "Rishi Sunak" on 2026-era queries). Prevents the LoRA from answering from its training cutoff when the user asks for a current title-holder. Routes to wiki, which sees fresh intro sentences.

Plus priority rules:
- **Explicit-verb priority**: `schedule_reminder` wins when it matches AND another tool matches (e.g. "remind me to check the gold price" matches both `schedule_reminder` and `tracker`; the reminder verb wins, the gold is the reminder content).
- **Year-anchored wiki**: when `\b\d{4}s?\b` is present and both `wiki` and `tracker` match, wiki wins (historical question, not current price).

### Under-fire rescue pipeline (`bridge/routing/force_fire.py` + integration in `OllamaLLM.chat`)

The observation: when the classifier is confident and the LoRA emits no tool call, the LoRA is under-firing. Instead of accepting the LoRA's silence, the bridge force-executes the classifier's hinted tool.

```
   LoRA call #1 completes
          │
          ▼
   tool_calls = [] AND _recover_leaked_tool_call returns None
          │
          ▼
   under_fire_rescue enabled?   no ─→ return LoRA's content as final reply
          │
          │ yes
          ▼
   hint_tool from classifier?   no ─→ return LoRA's content
          │
          │ yes
          ▼
   already rescued this turn?  yes ─→ return LoRA's content (budget = 1/turn)
          │
          │ no
          ▼
   hint_tool ∈ FORCE_RESCUE_TOOLS?  no ─→ return LoRA's content
          │  (wiki, get_weather, get_time, get_news, get_alerts,
          │   get_digest, find_places, web_search, math* — tools
          │   with forgiving args. NOT convert/tracker/update_memory/
          │   schedule_reminder/circuit/graph where bad args cause harm)
          │  * math rescue is narrow: only unambiguous KL/entropy
          │    distribution shorthand (N(0,1) || N(1,2), H(Beta(2,5)))
          │    goes through force-fire. All other math stays LoRA-routed.
          │
          │ yes
          ▼
   extract args heuristically (default_args or default_args_continuation)
          │
          ▼
   execute tool → build synthetic (assistant_with_tool_call, tool_result)
   pair → inject into messages → loop back (LoRA call #2 sees it)
          │
          ▼
   next iteration composes the user-facing reply grounded in the tool result
```

The synthetic messages use the shape Ollama emits itself (content=`""`, `arguments` as a dict — not a JSON-dumped string), otherwise Ollama 400s on re-submission.

### Result
- Eval JSON: [../evals/eval_phase0_rescue.json](../evals/eval_phase0_rescue.json).
- **112/135 = 83.0% routing** (+11.9 pp vs iter-3 alone).
- **38/64 = 59.4% tool-output usage** (+2.5 pp).
- **16 F→P flips, zero regressions**.
- Zero retrain.

Captured in `project_phase0_rescue.md`.

---

## Phase-0 round 2 — Step 1 (shipped 2026-04-20 AM)

Re-reading the Phase-0 eval JSON revealed 3 cases marked FAIL for stale-yaml reasons (earlier fixes for `fact: emerald` / `fact: peanut` / `to_unit: g` hadn't propagated to the Jetson's copy of `eval_cases_novel.yaml`). After re-pushing the yaml fix, two new classifier priority rules were stacked on top:

### Alert priority
When both `get_alerts` and `get_weather` pattern-hit AND the prompt contains `warning/warnings/alert/alerts/advisory/hazard`, `get_alerts` wins. Fixes "Any severe weather warnings in my area?" which was routing to get_weather.

### Compound "A and B" picker
When ≥2 different tools match AND the prompt contains `and`, return the tool whose pattern matches EARLIEST in the prompt.

```
"What's the weather and the news?"
   get_weather pattern: matches "weather" at pos 11
   get_news    pattern: matches "news"    at pos 26
   → get_weather wins (earliest)

"Convert 50 euros to dollars and give me the news."
   convert  pattern: matches "Convert"     at pos 0
   get_news pattern: matches "news"        at pos 46
   → convert wins

"Weather in NYC and is there any severe weather alert?"
   get_weather pattern: matches "Weather"  at pos 0
   get_alerts  pattern: matches "alert"    at pos 46
   → get_weather wins (earliest; either accepted by eval)
```

Single-tool "A and B" phrasings ("weather in Paris and London") only match one tool, so the branch never fires on them.

### Result
- Eval JSON: [../evals/eval_phase0_round2.json](../evals/eval_phase0_round2.json).
- **119/135 = 88.1% routing** (+5.1 pp vs Phase-0).
- **45/70 = 64.3% tool-output** (+4.9 pp).
- **7 F→P flips, zero regressions** — BUT reply-pass (regex) only +0.8 pp; 4 of 7 flips had mediocre replies (empty-location weather, nonce leak on convert, widget leak on yaml-fix update_memory cases).

Two takeaways that compound (pun intended): routing-layer wins don't always translate to reply-quality wins, and the regex reply-pass scanner we had at that point was too lenient.

---

## Phase-0 round 2 — Step 2: continuation rescue (shipped 2026-04-20 PM)

Short pronoun-led follow-ups like "How about Chicago?" / "And in Tokyo?" / "Anything on tech specifically?" — the LoRA abstains because the prompt has no explicit tool keyword; the classifier abstains for the same reason. Neither can fire, rescue can't trigger without a hint, and the LoRA's fallback is a schema dump or meta-commentary.

Fix: extend the rescue to also fire on continuations.

### `_resolve_hint_tool` (bridge/llm.py)

```python
def _resolve_hint_tool(self, user_text, has_tools):
    if not has_tools:
        return (None, False)
    hit = _routing_classify(user_text)
    if hit:
        return (hit, False)           # regex classifier hit
    if not self.under_fire_rescue:
        return (None, False)
    if not _looks_like_continuation(user_text):
        return (None, False)          # not a short follow-up
    prior = _last_assistant_tool_call(self._history)
    if prior and prior in FORCE_RESCUE_TOOLS:
        return (prior, True)          # carry prior tool forward
    return (None, False)
```

`_looks_like_continuation` matches the short prompt shapes (`how about`, `what about`, `and in`, `actually`, `anything on`, `try`, plus ≤3-word fragments). `_last_assistant_tool_call` walks `_history` backwards past prose-only assistant turns and returns the most recent structured tool call's name.

### `default_args_continuation` (bridge/routing/force_fire.py)

Strips the continuation wrapper (`_CONT_PREFIX`: `how about`/`what about`/`and (in|on|to|for)`/`also`/`actually (i meant)? (try)?`/`wait (make it)?`/`scratch that`/`try`/`anything|anymore|more (on|about|for)`) and any suffix noise (`instead`/`specifically`/`too`/`also`/`please`), leaves the payload, drops it into the tool's primary arg. Note: this is a SEPARATE prefix set from classifier.py's `_CONTINUATION_PATTERNS` which only needs to DETECT continuations — the classifier set is narrower (`how about`/`what about`/`and`/`no wait`/`actually`/`wait`/`cancel that`/`there`/`it`/`they`/`he`/`she`/`anything|anymore|more (on|about|for)`), plus any ≤3-word prompt. `"Actually, try bolognese"` is detected via the `actually` pattern in the classifier AND stripped via the `actually,?\s+(?:try\s+)?` branch in force_fire.

```
"How about Chicago?"       → payload "Chicago"  → get_weather(location=Chicago)
"And in Tokyo?"            → payload "Tokyo"    → get_time(timezone=Tokyo)
"What about Newton?"       → payload "Newton"   → wiki(query=Newton)
"Anything on tech today?"  → payload "tech"     → get_news(topic=tech)
"Actually I meant Osaka"   → payload "Osaka"    → get_weather(location=Osaka)
"Actually, try bolognese"  → payload "bolognese"→ web_search(query=bolognese)
```

Tool narrowing is SKIPPED for continuation hints (softer signal than a direct classifier hit) so if the LoRA DOES want to fire a different tool on the follow-up, it can.

### Result
- Eval JSON: [../evals/eval_phase0_round2_step2.json](../evals/eval_phase0_round2_step2.json).
- **125/135 = 92.6% routing** (+4.5 pp vs Step 1). 6 of 7 target continuations flipped; one miss (`What about Newton?` — LoRA emitted a natural-language response before rescue could catch it).

Cases that routing can NOT fix at runtime (remaining 10, iter-5 territory): 3 continuations with non-force-safe prior tool (convert / schedule_reminder), 3 gap-tool under-fires (graph / circuit / find_places directions), 1 update_memory under-fire, 1 over-fire, 1 digest→alerts misroute, 1 Newton miss.

---

## Honest reading: routing ≠ reply quality

Two things happened on 2026-04-20 that changed the priority map:

### 1. Regex reply-scanner added (false comfort)
`scripts/eval_routing.py::scan_reply_flags` pattern-matches for schema_leak, widget leak, nonce leak, disavowal phrasings, topic_drift_coffee, bus_hallucination. Gives a second metric next to routing-pass. On the Step 2 run it reported **89/135 = 65.9% reply-pass**.

### 2. Manual Claude read of all 135 replies (reality check)
The regex is a floor, not a ceiling. It misses semantic hallucinations: invented commodity prices on get_news ("Gold prices are up today, at 1,933.50 per troy ounce" with the stub saying nothing about gold), wrong-commodity substitutions on tracker ("What's the USD/JPY rate?" → reply says "gold at $42/oz"), compound-query fabrication ("Set an alarm for 7am and tell me the weather" → invents specific weather averages), "converted to gold" on a Fahrenheit→Celsius prompt. Novel phrasings of refuse-own-tool ("it looks like this prompt is actually a tool call") slipped through too.

Manual Claude read: **36/135 = 26.7% reply-pass**. The regex scanner was under-reporting by 53 cases — the gap between routing (92.6%) and reply-pass (26.7%) was **65.9 pp**, not the 30 pp the scanner suggested.

**Routing was largely solved for the runtime-lever budget. Reply quality was the real problem.**

`feedback_eval_read_replies.md` codified the lesson: every eval run gets a two-stage manual review (Stage 1 routing soft-grade, Stage 2 reply read) before writeup. `scripts/eval_routing.py` was tightened to emit `reply_flags` + `reply_pass` columns, plus a `--score-json` mode for retroactive scoring.

---

## Two-phase compose — shipped 2026-04-20 PM

Reading the Step 2 failures showed a clear pattern: the default reply generation has the **tool schema + coaching still in context** when the LLM is supposed to be writing prose. That context drives most of the reply bugs:

- `schema_leak` (23 cases) — LoRA sees the JSON tool catalog and describes it back at the user
- `refuses_own_tool` (8) — "No tool needed here, we already called convert" (LoRA disavows the call it just made)
- `disavowal` (7) — "I'm just a conversational AI, not a calculator" (after firing math)
- `you_meta_commentary` (8) — "You've asked me to evaluate a JSON schema" (third-person narration instead of reply)

None of these get a second chance in the default pipeline — the LoRA call that writes the reply IS the one that sees the schema.

### Fix: fork the reply composition out to a focused call

```
                Default pipeline (two-phase OFF)           Two-phase pipeline (ON)
                ─────────────────────────────────          ───────────────────────────
tools fired     LoRA call #2 sees:                         LoRA compose call sees:
then compose     • system prompt (persona + rules)          • compose-system prompt
                 • tool schemas (all 14)                      (focused, no schema)
                 • coaching ("call tool if needed")         • user's prompt
                 • multi-turn history                       • scrubbed tool outputs
                 • tool results                             (NONCE_ stripped)
                 → "writes reply"                           → "writes reply"

                ↓                                           ↓
                schema_leak, disavowal,                    short grounded reply
                refuses_own_tool, etc.                     using the data

no tool fired   LoRA's initial call returned               casual-compose call:
then compose    plain content already.                      • no schema
                                                            • Karin identity pinned
                The content often contains                  • forbid widget phrases
                schema_leak, you_meta,                      • last 2-4 message pairs
                wrong_persona (on "what's                   for continuation context
                your name").                               → clean chitchat reply
```

Implementation: [bridge/llm.py](../../bridge/llm.py) gains two methods + a sanity check:
- `_compose_reply_from_tools(user_text, tool_record)` — tool-fired compose.
- `_compose_reply_no_tool(user_text, history)` — chitchat/idiom compose with 2-4 message history for continuations.
- `_compose_output_ok(reply)` — regex sanity check; if compose output still schema-leaks we fall back to the stock reply (belt + suspenders).

NONCE scrubbing happens before handing the tool output to the compose LLM; otherwise the LoRA faithfully echoes `NONCE_xxxxxx` into the user-facing text.

### Runtime toggle

Added 2026-04-20 evening:
- `OllamaLLM.set_two_phase_compose(bool)` — runtime setter.
- `POST /api/settings` with `{"two_phase_compose": bool}` in web/server.py — mutates the running LLM singleton.
- `GET /api/features` extended to include the current `two_phase_compose` state so the UI can sync.
- Sidebar toggle "✨ Quality replies (slower)" in web/static/index.html + app.js — fetches state on load, POSTs on change, optimistic UI with rollback on failure.

**Default off.** Extra LLM call = +5-10 s per tool-firing turn on Jetson. Users flip it on when they want quality, leave it off for chat-speed UX.

### Result
Eval JSON: [../evals/eval_phase0_round2_step2_twophase.json](../evals/eval_phase0_round2_step2_twophase.json). Manual Claude judge of all 135 replies:

| Metric | Step 2 (no compose) | Step 2 + two-phase | Δ |
|---|---|---|---|
| Routing | 125/135 = 92.6% | 125/135 = 92.6% | 0 |
| Regex reply-pass | 89/135 = 65.9% | 100/135 = 74.1% | +11 |
| **Manual reply-pass** | **36/135 = 26.7%** | **62/135 = 45.9%** | **+26 (+19.2 pp)** |

32 F→P flips, 6 P→F regressions.

**Fixed at runtime (no retrain):** all 3 convert refusals, all 3 math refusals, both update_memory meta-failures (now confirm "emerald green" / "peanut"), 6 of 7 nonce leaks, 12 of 14 Trotsky cases, tracker commodity confusion, graph self-disavowal.

**New P→F class (compound-fabrication):** when the user asks A+B and tool only covers A, the compose is too helpful and invents B. 6 cases: weather+time invents time "12:45 PM"; USD→EUR+GBP invents "85 GBP"; digest invents specific items; alarm+weather invents weather averages. Compose system prompt tightened this session ("if tools didn't cover part of the question, say so honestly; never fabricate the missing part"); iter-5 SFT reinforcement planned for the residual.

---

## Hint-in-user-msg — shipped 2026-04-22

Research into LLM inference caching (KV quantization, prefix caching, semantic caching) surfaced that Ollama's prompt KV cache can't reuse prefix tokens past the first diverging byte across requests. Karin's per-turn classifier hint, sitting at the END of the system prompt, was breaking the cache on every turn — even though everything before the hint (base prompt + memory + location) was byte-stable.

### Fix: split the prompt, move the hint to the user message

`OllamaLLM._effective_system_prompt(user_text)` was doing two jobs at once: building the stable prefix AND appending the per-turn hint. Refactored into two methods:
- `_stable_system_prompt(user_text)` — base + suffix-provider output only. Byte-stable across turns (given bandit is off in prod so `preference_hint` never adds).
- `_routing_hint_text(user_text)` — per-turn classifier hint, returned as a separate string.

When `llm.hint_in_user_msg: true`, `chat()` and `chat_stream()` assemble messages as:
- `system` → `_stable_system_prompt(...)`
- `user` → `f"{hint}\n\n{user_text}"` (hint prepended to user content)
- history commit → clean user text only (hint stripped), so next turn's history prefix is also byte-stable.

`_effective_system_prompt(...)` is preserved as a thin wrapper that concatenates the two halves for the non-chat code paths (summarizer, some tests).

### Result

A/B on 135-case eval with iter-3 LoRA + Phase-0 + two-phase compose, only the hint placement changed:

| Metric | Hint in system (baseline, 2026-04-21) | Hint in user msg (shipped, 2026-04-22) | Δ |
|---|---|---|---|
| Routing | 123/135 = 91.1% | **126/135 = 93.3%** | +3 cases (+2.2 pp, within run-to-run noise envelope) |
| Reply-pass | 115/135 = 85.2% | **124/135 = 91.9%** | +9 cases (+6.7 pp) |
| Tool-output usage | 40/75 = 53.3% | **45/76 = 59.2%** | +5.9 pp |

Caveat: the A/B is cross-day (temperature 0.3 has some stochasticity; no same-day baseline was rerun). Reply-pass and tool-output-usage gaps are too large for pure noise; routing gap is within noise envelope. Honest summary: **at worst routing-neutral, likely a reply-quality win**, and the prefix-cache benefit is real independent of eval scores.

All 9 remaining routing fails on the 2026-04-22 run are chronic failure classes (find_places directions, multi-turn continuations, update_memory silent pretend, Ohm's law under-fire). No new regressions from the hint move.

### Same ship: persisting two_phase_compose into config

Two-phase compose was previously runtime-only via `/api/settings`, which silently reset to False on every web restart (easy footgun if the operator assumed it was sticky). Added `llm.two_phase_compose: true` to `config/assistant.yaml` and wired the constructor in `web/server.py` to read it. Runtime `/api/settings` override still works for per-session experimentation but no longer carries the persistent state.

Current prod flag set (`config/assistant.yaml`):
```yaml
llm:
  under_fire_rescue: true
  two_phase_compose: true
  hint_in_user_msg: true
```

---

---

## Former iter-5 plan

Two-phase compose absorbed most of the failure classes the original iter-5 plan targeted (`project_iter5_plan.md`, revised top section). Remaining scope (~65 SFT + ~20 DPO):

1. **Structural routing fails** (~10 cases): 3 continuations with non-force-safe prior tools (schedule_reminder / convert), 3 gap-tool under-fires (graph / circuit / find_places directions), 1 update_memory under-fire, 1 over-fire ("weather has been weird lately"), 1 digest → alerts misroute, 1 Newton continuation miss.
2. **Compound-honesty** (~15 examples): teach LoRA to say "got A but didn't look up B" rather than invent B when only half the tools fire.
3. **update_memory args-shape reinforcement** (~10 examples).
4. **"Speed-mode baseline" SFT** (~10 examples): proper tool-grounded replies so the raw LoRA is still decent when users disable two-phase.
5. **~20 DPO pairs** in multi-turn format (`continue_final_message=True`) covering the over-fire decoy + tool-output-usage reinforcement.

Gated on **eval-stub fix** before iter-5 ships: either topic-appropriate per-tool stubs OR LLM-as-judge integration. Without this, iter-5 ship decisions would still be polluted by Trotsky/Hitchhiker artifacts.

---
