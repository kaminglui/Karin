# Routing eval: cross-backend and cross-iteration comparison

Two tables below:
- The original cross-backend comparison on the 38-case set (all using iter-3 LoRA, varying the serving backend).
- An iteration comparison on the 135-case set (all on Ollama, varying the LoRA).

## 38-case cross-backend summary (iter-3 LoRA, five backends)

| Backend / variant | Score | % | Δ vs baseline | Notes |
|---|---|---|---|---|
| Ollama baseline | 35/38 | 92.1% | — | iter-3 LoRA, no Karin-level fixes. |
| Ollama + cls fixes | **36/38** | **94.7%** | **+2.6 pp** | iter-3 LoRA + `schedule_reminder` priority override in the classifier + wiki pattern for `in YYYY money` historical questions. |
| llama.cpp | 29/38 | 76.3% | −15.8 pp | llama-server with a custom jinja chat template (JSON-format coaching + terse tool-name list) and a `no_tool` shim in the bridge to absorb the LoRA's "no tool needed" signal. |
| MLC hard coach | 22/38 | 57.9% | −34.2 pp | `mlc_llm serve` with merged LoRA (fp16 bake-in) + Ollama-style "please respond with a JSON for a function call" coaching prefixed to every user turn. |
| MLC soft coach | 22/38 | 57.9% | −34.2 pp | Same setup as MLC hard, but softened "if a tool applies, respond with JSON; else reply normally" wording. Trade: fewer decoy false-fires, more under-fires on genuine tool cases. |

_Scores reflect the cleaned ground truth: case #38 (`What's the time zone in Tokyo?`) accepts either `get_time` or `wiki`; case #12 does a substring check on the reminder `message` arg. Previous-version scores were 89.5% / 92.1% / 76.3% / 57.9% / 55.3% respectively — the two flagged expectations shifted three outcomes in the Ollama and MLC-soft rows._

## 135-case iteration comparison (Ollama, iter-3 vs iter-4)

Same backend (Ollama + mannix base + classifier+bridge patches). Varying the LoRA adapter on top.

**Metric columns.** Routing = right tool fires with acceptable args. Reply-pass split into two:
- **regex** — heuristic scan from `scan_reply_flags()` in `scripts/eval_routing.py`. Catches widget leaks, nonce leaks, disavowal phrasings, schema leaks. Misses semantic hallucinations.
- **manual** — Claude read every `final_reply`. Catches what regex misses (fabricated commodities, wrong-subject conversions, compound-query invention).

Stub-artifact flags (Trotsky on wiki, Hitchhiker-42 on web_search) surfaced but NOT counted against reply-pass — eval-stub side-effects, not production bugs.

| LoRA + runtime layer | Routing | Reply-pass (regex) | Reply-pass (manual) | Tool-out | Tools fired |
|---|---|---|---|---|---|
| iter-3 baseline (`run_0ac17bc7`) | 96/135 = 71.1% | — | — | 29/51 = 56.9% | 51 |
| iter-4 (`run_52227b4c`, rolled back) | 100/135 = 74.1% | — | — | 27/56 = 48.2% | 56 |
| + Phase-0 classifier + rescue | 112/135 = 83.0% | 77/135 = 57.0% | — | 38/64 = 59.4% | 64 |
| + Phase-0 round 2 Step 1 | 119/135 = 88.1% | 78/135 = 57.8% | — | 45/70 = 64.3% | 70 |
| + Step 2 (continuation rescue) | 125/135 = 92.6% | 89/135 = 65.9% | **36/135 = 26.7%** | 48/75 = 64.0% | 75 |
| + two-phase compose (generic stubs) | 125/135 = 92.6% | 100/135 = 74.1% | 62/135 = 45.9% | 48/75 = 64.0% | 75 |
| + two-phase compose + topic-aware stubs (prev prod, honest eval, 2026-04-21) | 123/135 = 91.1% | 120/135 = 88.9% | 115/135 = 85.2% | 40/75 = 53.3% | 75 |
| **+ hint-in-user-msg** (current prod, shipped 2026-04-22) | **126/135 = 93.3%** | — | **124/135 = 91.9%** | **45/76 = 59.2%** | **76** |
| iter-5 (`run_7db68f92`, not shipped) | 110/135 = 81.5% | — | 107/135 = 79.3% | 10/64 = 15.6% | 64 |
| iter-6 (`run_d6ae0966`, not shipped) | 111/135 = 82.2% | — | 104/135 = 77.0% | 26/90 = 28.9% | 90 |

### iter-4 retrospective (rolled back)
- +3 pp routing driven by classifier-pattern additions (wiki historical, `how do I X` web_search) — 5 of 6 case flips came from `bridge/` changes, not the LoRA.
- One clean LoRA flip (`Turn on the lights` → no_tool); two regressions (`Call me when you're free`, `Tell me about my schedule today` — gap-fill SFT over-generalized).
- DPO single-turn flatten regressed tool-output usage −8.7 pp. Flattened `"User asked: X. The Y tool returned Z."` is a distribution that doesn't exist at serve time.
- Reverted to iter-3. Details in `project_iter4_postmortem.md`.

### Phase-0 + rescue (shipped 2026-04-19)
- 7 new classifier regex patterns: `in YYYY money/dollars`, `(when\|once\|after\|if)… (remind\|ping\|nudge) me`, `which (costs\|is) more (gold\|silver\|btc…)`, `(should\|can\|will) i (bbq\|grill\|run…)`, `solve … equals N`, `chart the last N`, `agenda today`.
- Under-fire rescue in `bridge/routing/force_fire.py` + `OllamaLLM.chat`. Fires when (a) classifier has a single-pattern hint, (b) LLM emitted no tool call AND no recoverable leak, (c) hinted tool in force-safe allowlist: wiki / get_weather / get_time / get_news / get_alerts / get_digest / find_places / web_search. NOT convert / math / tracker / schedule_reminder / circuit / graph / update_memory (wrong args cause concrete harm).
- One rescue per turn. Config: `llm.under_fire_rescue: true`.
- **+16 F→P routing, 0 regressions, +11.9 pp routing, +2.5 pp tool-output** — zero retrain.

### Phase-0 round 2 Step 1 (shipped 2026-04-20 AM)
- Yaml re-push: case #57 `to_unit: g`, #98 `fact: emerald`, #100 `fact: peanut` (stale on Jetson, was bumping 3 cases to FAIL for grading reasons).
- Alert priority rule: `get_alerts` wins over `get_weather` when prompt has `warning/warnings/alert/alerts/advisory/hazard`.
- Compound "A and B" picker: when ≥2 different tools match + `and` present, return the one whose pattern matches earliest.
- **+7 F→P routing, 0 regressions, +5.1 pp routing, +4.9 pp tool-output, +0.8 pp reply-pass (regex).** Routing gain real, reply-quality barely moved: 4 of 7 flips had mediocre replies (empty-location weather ×2, nonce leak on convert, widget leak on 2 update_memory yaml-fix cases).

### Phase-0 round 2 Step 2 (shipped 2026-04-20 PM, continuation rescue)
- `default_args_continuation()` strips continuation prefix (`how about`/`what about`/`and in`/`actually`/`anything on`/`try`) and drops payload into tool's primary arg.
- `_resolve_hint_tool()` falls back to `_last_assistant_tool_call()` on short pronoun-led prompts. Only fires when prior tool is force-safe.
- **+6 F→P routing, 0 regressions, +4.5 pp routing, +8.1 pp reply-pass (regex).** Targets hit: Chicago / Tokyo / Osaka weather + `lbs→grams` convert yaml fix + tech-news + bolognese-web-search. One miss: `What about Newton?` — LoRA emitted no-tool before rescue could catch it.
- **Manual Claude judge exposed the regex blind spot**: honest reply-pass 36/135 = 26.7%, not 65.9%. Regex under-reported bad replies by 53 cases — misses semantic hallucinations (fabricated gold/silver prices on get_news, wrong-commodity substitutions on tracker, invented digest items, compound-query fabrication, novel phrasings of refuse-own-tool).
- Top model-bug reply-fail buckets (ex 18 stub-driven): schema_leak ×23, hallucinated_data ×10, you_meta_commentary ×8, refuses_own_tool ×8, nonce_leak ×6, disavowal ×4, topic_drift ×3.

### Two-phase compose (shipped 2026-04-20 PM, runtime toggle, default off)
- `_compose_reply_from_tools()`: after tools fire, re-ask the LLM with ONLY user prompt + scrubbed tool outputs (no tool schema, no coaching). Eliminates the "tool fires but LoRA then refuses/disavows/narrates schema" class.
- `_compose_reply_no_tool()`: parallel path for chitchat/idiom/greeting turns. Strips schema from context, pins Karin identity (not "KaraoKe-chan" / "vocaloid"), forbids widget phrases.
- `_compose_output_ok()` sanity check: if compose output still schema-leaks, fall back to stock reply.
- UI: sidebar toggle "✨ Quality replies (slower)". `POST /api/settings {two_phase_compose: bool}` + `GET /api/features` exposes current state. Default off (+5-10s/turn with on).
- **+26 reply-pass cases, +19.2 pp manual reply-pass absolute (26.7% → 45.9%)**. 32 F→P flips, 6 P→F regressions. Zero retrain. Routing unchanged.
- P→F regressions dominated by new class: **compound-fabrication** — when user asks A+B and tool only covers A, compose invents B (cases: weather+time → invented "12:45 PM"; USD→EUR+GBP → invented "85 GBP"; digest "3 items" → invented specific items). Compose-prompt tightened this session; may need SFT reinforcement in iter-5.

### Topic-aware stub fix (shipped 2026-04-21, `scripts/eval_routing.py::_stub_execute` rewrite)
- Old generic stubs ("Summary: notable 20th-century figure, born 1842" on every wiki call) triggered specific LoRA hallucinations (Trotsky substitution on wiki, Hitchhiker "42" jokes on web_search, bus-system rambling on get_news, gold substitution on tracker) that **didn't exist in production** — real tool data doesn't look generic like that.
- Topic-aware stubs echo the user's args into the stub text (`"Wikipedia summary for 'Mount Fuji': ..."`, `"Weather in Chicago: ..."`, `"Top headline on tech: ..."`). The LoRA can't substitute Trotsky because "Mount Fuji" is literally in the stub it's trying to paraphrase.
- Also rolls off the 42 / 1842 specific trigger numbers in favor of 67 / 54 / 156.78 / 19:12 / etc. — same number-match heuristic for `used_tool_output` detection, no specific-value hallucination triggers.

### The stub-pollution finding
Before the topic-stub fix, the manual reply-pass was 62/135 = 45.9% with a reported 46.7 pp gap vs routing. After the fix, the SAME prod stack scored **115/135 = 85.2%** with a 6.7 pp gap. The 39.3 pp jump is NOT a prod improvement — it's the eval becoming honest about what the LoRA was actually producing in prod. The "hidden reply-quality gap" we spent sessions analyzing was mostly our own fake data triggering artifacts the LoRA wouldn't emit against real tool output. See `docs/evals/eval_prod_twophase_topicstubs.json` + the Claude manual-judge fields on each case.

### Remaining 9 reply fails (the real iter-5 target)
- **Compound fabrication (3):** #121 Paris vs London (fabricated per-city differences), #124 coffee+pharmacy (fabricated "both on-site"), #125 USD to EUR + GBP (invented GBP value). All match the iter-5 `iter5_compound_honesty.jsonl` SFT class.
- **Wiki gap-filling (2):** #26 "dollar in 1970 money" (invented "smoke" barter system), #52 "house in 1960s America" (invented "American Dream theme" as article content). LoRA fills in when stub is vague.
- **Wrong math (2):** #96 RC time constant (23.5ms vs correct 10ms), #105 solve quadratic (called 156.78 "imaginary number"). LoRA does math itself instead of quoting stub + gets it wrong.
- **Capability misrepresentation (2):** #84 "text my mom" (drafts message without acknowledging it can't send), #115 "wake me if a storm" (misleadingly promises "I'll wake you up").

### Remaining 11 routing fails (unchanged, iter-5 training territory)
3 continuations with non-force-safe prior tool (schedule_reminder / convert), 3 gap-tool under-fires (graph / circuit / find_places directions), 1 update_memory under-fire, 1 over-fire (`the weather has been weird lately`), 1 digest → alerts misroute, 1 Newton continuation miss, 1 flight-book over-fire. See `project_iter5_plan.md` for training recipe.

## Per-case results (all 38)

Columns = runs. Cell = ✅ PASS / ❌ FAIL with latency in seconds. `expected` column shows the correct tool routing.

| # | Prompt | Expected tool | Ollama baseline | Ollama + cls fixes | llama.cpp | MLC hard coach | MLC soft coach |
|---|---|---|---|---|---|---|---|
| 1 | `What is Mount Fuji famous for?` | wiki | ✅ 30.7s | ✅ 25.5s | ✅ 4.7s | ✅ 5.4s | ✅ 8.5s |
| 2 | `Hello!` | *no_tool* | ✅ 8.7s | ✅ 8.8s | ✅ 0.5s | ✅ 1.1s | ✅ 1.0s |
| 3 | `Hi Karin.` | *no_tool* | ✅ 9.2s | ✅ 8.4s | ✅ 0.4s | ✅ 0.8s | ✅ 0.5s |
| 4 | `Thanks, that's all I needed.` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ✅ 1.8s | ✅ 2.2s | ✅ 2.5s |
| 5 | `What's your name?` | *no_tool* | ✅ 16.0s | ✅ 13.4s | ✅ 0.8s | ✅ 1.3s | ✅ 1.3s |
| 6 | `Look up quantum computing on Wikipedia.` | wiki | ✅ 25.5s | ✅ 20.4s | ✅ 5.0s | ✅ 5.3s | ✅ 10.9s |
| 7 | `Remember that time we went hiking last summer?` | *no_tool* | ✅ 16.1s | ✅ 16.0s | ✅ 1.8s | ✅ 1.4s | ✅ 3.5s |
| 8 | `Call me when you're free.` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ❌ 11.9s | ❌ 9.8s | ❌ 5.2s |
| 9 | `Forget what I said about batteries.` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ✅ 6.0s | ✅ 1.7s | ✅ 2.0s |
| 10 | `The gold standard of debugging is print statements.` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ❌ 10.0s | ❌ 12.3s | ❌ 5.1s |
| 11 | `What's the weather and the news?` | get_weather | ❌ 16.0s | ❌ 15.9s | ✅ 5.6s | ❌ 6.2s | ❌ 2.4s |
| 12 | `Remind me to check the gold price tomorrow at 9am.` | schedule_reminder | ❌ 16.0s | ✅ 26.1s | ✅ 11.3s | ✅ 19.7s | ✅ 5.5s |
| 13 | `Convert 100 usd to jpy and 5 miles to km.` | convert | ✅ 29.1s | ✅ 28.1s | ✅ 6.2s | ❌ 4.8s | ❌ 5.3s |
| 14 | `History of the gold standard.` | wiki | ✅ 22.1s | ✅ 23.7s | ✅ 10.7s | ✅ 12.7s | ❌ 4.0s |
| 15 | `How many miles is it from NYC to LA?` | *no_tool* | ✅ 16.0s | ✅ 12.5s | ❌ 5.4s | ❌ 15.3s | ❌ 6.7s |
| 16 | `What's the weather in San Francisco?` | get_weather | ✅ 25.7s | ✅ 20.8s | ✅ 10.5s | ✅ 19.0s | ✅ 11.6s |
| 17 | `Convert 100 usd to jpy.` | convert | ✅ 26.9s | ✅ 24.8s | ❌ 11.7s | ❌ 16.8s | ❌ 9.5s |
| 18 | `Thanks!` | *no_tool* | ✅ 7.9s | ✅ 8.4s | ✅ 0.7s | ✅ 0.7s | ✅ 0.7s |
| 19 | `My favorite memory is our camping trip.` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ✅ 1.9s | ❌ 6.7s | ✅ 2.9s |
| 20 | `Schedule conflicts are the worst.` | *no_tool* | ✅ 16.0s | ✅ 15.9s | ❌ 13.6s | ❌ 14.7s | ❌ 10.9s |
| 21 | `Set aside some time for yourself.` | *no_tool* | ✅ 16.1s | ✅ 16.0s | ✅ 5.9s | ❌ 14.3s | ❌ 5.0s |
| 22 | `How's everybody doing this morning?` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ✅ 5.7s | ❌ 8.1s | ✅ 1.5s |
| 23 | `Tell me a joke.` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ✅ 6.2s | ✅ 1.5s | ✅ 3.2s |
| 24 | `Cancel that.` | *no_tool* | ✅ 16.0s | ✅ 15.9s | ❌ 10.3s | ❌ 7.0s | ❌ 4.1s |
| 25 | `Tell me about my schedule today.` | *no_tool* | ✅ 9.6s | ✅ 9.6s | ✅ 8.4s | ✅ 2.3s | ✅ 4.1s |
| 26 | `How much is a dollar in 1970 money?` | wiki | ❌ 16.0s | ❌ 10.1s | ❌ 5.8s | ❌ 15.8s | ❌ 10.2s |
| 27 | `What's the exchange rate history for USD/JPY?` | tracker | ✅ 22.2s | ✅ 24.2s | ✅ 8.9s | ✅ 19.0s | ✅ 12.3s |
| 28 | `Remember Mount Everest?` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ❌ 9.7s | ❌ 5.2s | ❌ 5.0s |
| 29 | `What's the truth about resistors?` | *no_tool* | ✅ 16.0s | ✅ 16.1s | ❌ 9.7s | ❌ 5.7s | ✅ 2.8s |
| 30 | `Good morning.` | *no_tool* | ✅ 15.9s | ✅ 16.0s | ✅ 5.9s | ❌ 4.5s | ❌ 6.1s |
| 31 | `Hey there.` | *no_tool* | ✅ 8.4s | ✅ 8.4s | ✅ 0.4s | ✅ 0.9s | ✅ 0.9s |
| 32 | `What's up?` | *no_tool* | ✅ 10.3s | ✅ 8.9s | ✅ 1.2s | ✅ 1.7s | ✅ 2.1s |
| 33 | `Have a good one.` | *no_tool* | ✅ 16.0s | ✅ 16.1s | ✅ 1.8s | ✅ 1.2s | ❌ 11.6s |
| 34 | `Cool, thanks.` | *no_tool* | ✅ 16.1s | ✅ 16.0s | ✅ 5.7s | ✅ 1.2s | ❌ 10.6s |
| 35 | `Got it.` | *no_tool* | ✅ 9.4s | ✅ 14.6s | ✅ 0.8s | ✅ 2.1s | ✅ 2.0s |
| 36 | `I forgot what I was going to say.` | *no_tool* | ✅ 16.1s | ✅ 16.0s | ✅ 1.8s | ✅ 1.4s | ❌ 3.6s |
| 37 | `Let's convert this argument into a conversation.` | *no_tool* | ✅ 16.0s | ✅ 16.0s | ✅ 1.4s | ❌ 7.1s | ✅ 1.8s |
| 38 | `What's the time zone in Tokyo?` | get_time \| wiki | ✅ 19.0s | ✅ 18.2s | ✅ 10.1s | ✅ 6.9s | ✅ 4.8s |

## Failure breakdown (consolidated)

One row per failing case. **Failed on** uses short backend codes:
- **Ob** = Ollama baseline · **Of** = Ollama + classifier fixes · **LC** = llama.cpp · **MH** = MLC hard coach · **MS** = MLC soft coach

"Typical actual" is the wrong tool the LoRA picked (or `—` when it emitted no tool call at all). When different backends emitted different wrong tools, they're listed alongside their codes.

| # | Prompt | Expected | Failed on → actual | Why |
|---|---|---|---|---|
| 8 | `Call me when you're free.` | _(no tool)_ | LC, MH, MS → schedule_reminder | Figurative/idiom decoy. "Call me when X" keyword-matches the reminder tool without the full tool description to contextualize it as a concrete future-time request. Ollama's template includes the schema+description which suppresses this. |
| 10 | `The gold standard of debugging is print statements.` | _(no tool)_ | LC, MH, MS → web_search | Idiom decoy. "gold standard" tripped the search tool on the terse llama.cpp/MLC templates. Ollama's full-schema prompt gives enough context to recognize the idiomatic use. |
| 11 | `What's the weather and the news?` | get_weather | Ob, Of → _(no tool)_; MH → get_digest; MS → _(no tool)_ | Compound "A and B" query that the LoRA's training data never covered. Model either freezes (Ollama/MS) or picks a non-answer tool (MH's `get_digest`). |
| 12 | `Remind me to check the gold price tomorrow at 9am.` | schedule_reminder | Ob → _(no tool)_ | Pre-classifier abstained because BOTH `schedule_reminder` (matched "remind me") AND `tracker` (matched "gold price") hit. With no hint reaching the LoRA, it played safe. Fixed in `Of` by adding a priority override so explicit-verb tools (`schedule_reminder`) win over passive-mention conflicts. |
| 13 | `Convert 100 usd to jpy and 5 miles to km.` | convert | MH, MS → _(no tool)_ | Compound conversion confused the MLC-merged LoRA into emitting nothing. Same training-distribution gap as #11. |
| 14 | `History of the gold standard.` | wiki | MS → tracker | "soft coach" wording weakened the JSON bias just enough that the LoRA fell back to a keyword-based tool pick. "gold" + history re-read as a tracker query. |
| 15 | `How many miles is it from NYC to LA?` | _(no tool)_ | LC → math; MH → convert; MS → math | "how many miles" keyword-matched the units tools even though the prompt asks for a geographical fact. Under Ollama's full schema the LoRA reads the convert/math descriptions as "needs a unit pair" and abstains. |
| 17 | `Convert 100 usd to jpy.` | convert | LC, MH, MS → convert with wrong args | Correct tool, wrong arg names. The LoRA emitted `{amount, from_currency, to_currency}` (or `{unit_from, unit_to, value}` under MH) instead of Karin's `{value, from_unit, to_unit}`. Ollama's template injects the full arg schema; the terse non-Ollama templates only list tool names, so the LoRA fell back to "natural" arg naming. |
| 19 | `My favorite memory is our camping trip.` | _(no tool)_ | MH → update_memory | Word "memory" keyword-fired the memory tool despite idiomatic usage. MLC's merged-fp16 LoRA has a stronger tool bias than Ollama's runtime-applied LoRA. |
| 20 | `Schedule conflicts are the worst.` | _(no tool)_ | LC, MS → schedule_reminder; MH → tracker | "schedule" / "worst" keyword-fires. Same category as #8. |
| 21 | `Set aside some time for yourself.` | _(no tool)_ | MH → get_digest; MS → schedule_reminder | Advice phrase misread as an action request. |
| 22 | `How's everybody doing this morning?` | _(no tool)_ | MH → get_digest | Pleasantry read as a morning digest request by the merged LoRA's stronger tool bias. |
| 24 | `Cancel that.` | _(no tool)_ | LC, MH, MS → schedule_reminder | Short prompt with no concrete time-object — non-Ollama backends assume it refers to a prior reminder. |
| 26 | `How much is a dollar in 1970 money?` | wiki | Ob, Of → _(no tool)_; LC → web_search; MH, MS → convert | Historical-value phrasing. `Ob` abstained because no pattern matched. `Of` added a wiki hint for "in YYYY money" but the narrowed tool list `{wiki, update_memory}` still let the LoRA choose nothing. LC/MH/MS over-fired on other info tools. |
| 28 | `Remember Mount Everest?` | _(no tool)_ | LC, MH, MS → wiki | "Remember" + named entity keyword-fires the wiki lookup. Conversational "remember" vs action-request "remember X" isn't distinguishable without the tool description context. |
| 29 | `What's the truth about resistors?` | _(no tool)_ | LC, MH → wiki | Rhetorical/reflective question misread as a fact lookup. Soft coaching (MS) actually helped here — it reined in the over-fire. |
| 30 | `Good morning.` | _(no tool)_ | MH, MS → get_digest | Plain greeting read as a morning-digest request. Merged-LoRA-only failure; Ollama handles this correctly. |
| 33 | `Have a good one.` | _(no tool)_ | MS → update_memory | Soft-coaching side effect: the softened wording let the LoRA fall through to a passive memory-save tool on a parting phrase. |
| 34 | `Cool, thanks.` | _(no tool)_ | MS → update_memory | Same pattern as #33. Soft coaching over-relaxed the JSON-emit pressure, but the LoRA still wanted a tool name. |
| 36 | `I forgot what I was going to say.` | _(no tool)_ | MS → get_digest | "forgot" read as a memory/digest signal. |
| 37 | `Let's convert this argument into a conversation.` | _(no tool)_ | MH → web_search | "convert" keyword-fires despite the figurative phrasing. |
_(#38 "time zone in Tokyo" was previously a fail for 3 backends that picked `get_time` instead of the expected `wiki`. After the audit, the ground truth now accepts either tool — see methodology notes — so this case is no longer in the fail list.)_

### Failure categories (TL;DR)

- **Figurative/idiom decoys** (#8, 10, 15, 19, 20, 22, 28, 29, 30, 33, 34, 37) — keyword matches an action verb/entity but the prompt is figurative. **Root cause**: without the tool description in context, the model over-weights keyword hits. Mostly hits non-Ollama backends because their templates omit descriptions.
- **Compound queries** (#11, 13) — "A and B" patterns the LoRA's training data never covered.
- **Classifier hint gaps** (#12, 26) — two tools match (ambiguity → abstain), or no pattern matches. Fixed partially for #12; #26 needs a stronger hint mechanism.
- **Arg-name mismatches** (#17) — correct tool, wrong arg names. Non-Ollama templates don't inject arg schemas.
- **Judgment-call mis-routes (resolved)** — #38 was a debatable wiki-vs-get_time call. After auditing, the ground truth now accepts either tool, so this category is eliminated.

## Latency stats (seconds per case)

| Backend | Min | Median | p95 | Max | Total |
|---|---|---|---|---|---|
| Ollama baseline | 7.9 | 16.0 | 26.9 | 30.7 | 633 |
| Ollama + cls fixes | 8.4 | 16.0 | 25.5 | 28.1 | 619 |
| llama.cpp | 0.4 | 5.8 | 11.7 | 13.6 | 220 |
| MLC hard coach | 0.7 | 5.3 | 19.0 | 19.7 | 264 |
| MLC soft coach | 0.5 | 4.1 | 11.6 | 12.3 | 192 |

## Methodology notes
- Single-turn mode (`commit_history=False`): the full conversation is NOT persisted across cases.
- Eval script: `scripts/eval_routing.py --cases sft/eval_cases_novel.yaml`.
- Karin serves through `/api/chat/stream`; each prompt is a fresh session.
- LoRA: `run_0ac17bc7` (SFT-only). DPO adapter NOT tested here.
- Timestamps of runs: 2026-04-19 / 2026-04-20 (same day).
- 'Ollama + cls fixes' = the classifier improvements made after the MLC experiments: schedule_reminder priority override on ambiguity, plus a wiki routing pattern for `in YYYY money` historical prices.
- **Ground-truth audit (2026-04-20)**: reviewed all 38 expectations. Two changes:
  - **#38 `What's the time zone in Tokyo?`** — previously `expected_tool: wiki`, but `get_time(timezone=Asia/Tokyo)` is a defensible reading (and arguably the more useful one). Now accepts either. `eval_routing.py::grade` was extended to accept a list for `expected_tool`.
  - **#12 `Remind me to check the gold price tomorrow at 9am`** — added `expected_args.message: gold` (substring check) so we at least catch the LoRA hallucinating an unrelated reminder body. `trigger_at` is not pinned (hard to validate cross-date), so a garbage date still passes — a known gap.
- Remaining debatable expectations left unchanged: #11 (compound A+B), #15 (distance), #26 (historical CPI), #27 (exchange rate history), #28 (rhetorical), #29 (reflective). Could be relaxed further but each is more-or-less defensible as labeled.
