# 135-case eval — manual review + iteration log

**Purpose.** Manual reading of `final_reply` text from 135-case routing evals, recorded per iteration. The automated routing pass-rate + `used_tool_output` metric misses real LoRA reply-quality issues (widget-template leak, meta-commentary, stub artifacts, self-disavowal); this doc is where the honest qualitative read lives alongside the numbers.

**How this doc is organized.** Most-recent iteration on top, older ones below. The giant iter-3 per-case table and untruncated iter-3 fail replies live in [evals/iter3-reference.md](evals/iter3-reference.md) so this file stays focused on analysis.

**Related.**
- Cross-iteration rollup + deploy info: [routing_eval_comparison.md](routing_eval_comparison.md)
- Iter-3 raw eval JSON: [evals/eval_135.json](evals/eval_135.json)
- Phase-0 + rescue raw eval JSON: [evals/eval_phase0_rescue.json](evals/eval_phase0_rescue.json)
- Phase-0 round 2 raw eval JSON: [evals/eval_phase0_round2.json](evals/eval_phase0_round2.json)

## Contents

1. [Cross-iteration summary](#cross-iteration-summary)
2. [Routing-pass vs reply-pass — bifurcation + two-phase compose](#routing-pass-vs-reply-pass--bifurcation--two-phase-compose) *(most important; current-prod breakdown + iter-5 priority shift)*
3. [Phase-0 + rescue — manual reply review (correction)](#phase-0--rescue--manual-reply-review-correction) *(historical: Phase-0 flip quality)*
4. [Phase-0 + Under-fire Rescue update](#phase-0--under-fire-rescue-update-2026-04-20)
5. [Iter-4 update (rolled back)](#iter-4-update-2026-04-20)
6. [Iter-3 baseline review](#iter-3-baseline-review) *(original manual review; drove iter-4 and Phase-0 designs)*
   - [Headline numbers](#headline-numbers)
   - [What the final replies actually look like (A–F)](#what-the-final-replies-actually-look-like)
   - [Summary of failure categories](#summary-of-failure-categories-redistribution-with-manual-insight)
   - [What this points at for iter-4](#what-this-points-at-for-iter-4)
   - [Proposed next step](#proposed-next-step)
7. [Iter-3 detailed reference](#iter-3-detailed-reference)

## Cross-iteration summary

| Iteration | Routing | Tool-output | Manual reply-pass | Status |
|---|---|---|---|---|
| iter-3 baseline (`run_0ac17bc7`) | 96/135 = 71.1% | 29/51 = 56.9% | — | Original LoRA. |
| iter-4 (`run_52227b4c`) | 100/135 = 74.1% | 27/56 = 48.2% | — | **Rolled back** — DPO single-turn flatten. |
| + Phase-0 classifier + rescue | 112/135 = 83.0% | 38/64 = 59.4% | — | Shipped 2026-04-19; +16 F→P, 0 regressions vs iter-3. |
| + Step 1 (alert-priority + compound picker) | 119/135 = 88.1% | 45/70 = 64.3% | — | Shipped 2026-04-20 AM; +7 F→P, 0 regressions. |
| + Step 2 (continuation rescue) | 125/135 = 92.6% | 48/75 = 64.0% | 36/135 = 26.7% | Shipped 2026-04-20 PM; +6 F→P, 0 regressions. |
| + two-phase compose (toggleable, default off) | 125/135 = 92.6% | 48/75 = 64.0% | 62/135 = 45.9% | Shipped 2026-04-20 PM; +26 reply-pass (+19.2 pp manual), zero retrain. |
| + **topic-aware stubs (eval honesty fix)** | **123/135 = 91.1%** | 40/75 = 53.3% | **115/135 = 85.2%** | **Current prod, honest eval.** 2026-04-21. Stub rewrite in `scripts/eval_routing.py::_stub_execute` — no prod change. Revealed 53 additional passing cases that were previously phantom-failures due to stub artifacts (Trotsky substitution on wiki, Hitchhiker jokes on web_search, bus-system on get_news, etc.). See stub-pollution section below. |

Jump to [Routing-pass vs reply-pass](#routing-pass-vs-reply-pass--bifurcation--two-phase-compose) for the two-phase analysis and iter-5 priority shift. Older iteration sections below preserve historical detail. Bottom reference table + appendix are iter-3 specifically.

---

## Routing-pass vs reply-pass — bifurcation + two-phase compose

**The insight (2026-04-20).** A single PASS bundles two independent questions — *right tool?* and *useful reply?* — that diverge wildly. `scripts/eval_routing.py` now tracks both.

Two reply-pass flavors:
- **regex** — heuristic scan from `scan_reply_flags()`. Catches widget leaks, nonce leaks, disavowal phrasings, schema leaks, topic_drift_coffee, bus_hallucination. Misses semantic hallucinations.
- **manual** — Claude reads every `final_reply` against (a) faithful to tool output, (b) in character, (c) actually addresses the user. Rubric in `feedback_eval_read_replies.md`.

Stub-artifact flags (Trotsky on wiki, Hitchhiker-42 on web_search) surfaced but NOT counted against reply-pass — eval-stub side-effects that resolve with real tool data.

### Scores across runs

| Run | Routing | Regex reply-pass | Manual reply-pass | Gap (routing − manual) |
|---|---|---|---|---|
| + Phase-0 + rescue | 112/135 = 83.0% | 77/135 = 57.0% | — | — |
| + Step 1 (alert priority + compound picker) | 119/135 = 88.1% | 78/135 = 57.8% | — | — |
| + Step 2 (continuation rescue) | 125/135 = 92.6% | 89/135 = 65.9% | 36/135 = 26.7% | 65.9 pp |
| + two-phase compose (generic stubs) | 125/135 = 92.6% | 100/135 = 74.1% | 62/135 = 45.9% | 46.7 pp |
| **+ two-phase compose + topic-aware stubs** (current prod, honest eval, 2026-04-21) | **123/135 = 91.1%** | **120/135 = 88.9%** | **115/135 = 85.2%** | **6.7 pp** |

### Stub-pollution — the dominant factor in previous "hidden gap"

The jump from 45.9% → 85.2% manual reply-pass on the same prod stack is NOT a production improvement. It's the eval becoming honest.

**Old generic stubs** — `"Summary: notable 20th-century figure, born 1842 [NONCE_xxx]"` for every wiki call, `"...answer is 42..."` on web_search, `"Top story: City adopts new bus system"` on get_news — triggered specific LoRA artifacts:
- Wiki → "Leon Trotsky was born in 1879, not 1842..." regardless of subject (Mount Fuji, Marie Curie, Berlin Wall all became Trotsky).
- Web_search → Hitchhiker's Guide "42 is the ultimate answer" jokes.
- Get_news → LoRA rambling about bus systems on any news prompt.
- Tracker → substituting "gold" regardless of requested commodity.

These were all **artifacts of our fake data**. In production, real wiki returns actual Mount Fuji data and the LoRA quotes it correctly. The "65.9 pp hidden reply gap" we spent sessions analyzing was mostly measurement-tool pollution, not LoRA behavior.

**Topic-aware stubs** echo the user's args into the stub text (`"Wikipedia summary for 'Mount Fuji': a notable subject with documented history..."`, `"Weather in Chicago: 67°F..."`, `"Top headline on tech: ..."`). LoRA can't substitute Trotsky because the topic is literally in the stub it's paraphrasing. Also replaces 42/1842 with 67/54/156.78/19:12 — same number-match heuristic for `used_tool_output` detection, no specific-value hallucination triggers.

**Two-phase compose delivered +26 real reply-pass cases** (36 → 62 = +19.2 pp) with zero retrain — that gain is honest. Topic-aware stubs revealed an **additional +53 cases** (62 → 115) that were always fine in prod but appeared broken because of the eval's own stubs. Not a production gain, a measurement correction.

### Step 2 reply-fail breakdown (89 cases, pre-two-phase, Claude manual judge)

Classes marked ★ were fixed at runtime by two-phase compose. Counts are cases-with-flag; a case can hit multiple flags.

| Class | Count | Nature | ★ Fixed by two-phase? |
|---|---|---|---|
| schema_leak | 23 | `"This is a JSON object describing a set of functions..."` as the whole reply. Biggest single bucket; hits chitchat + gap-prompt no-tool turns | ★ Yes (tool-firing and no-tool via casual-compose) |
| trotsky_stub | 14 | Every wiki call → Trotsky template (stub returns "born 1842"; LoRA "corrects" to Trotsky). Resolves in prod | ★ Mostly (compose rejects stub or uses own knowledge) |
| hallucinated_data | 10 | Invented gold/silver prices on get_news ($1,933.50/oz, $22.55/oz not in stub); fabricated digest items (gold $1820, Seattle rain, renewable energy); wrong-commodity on tracker (USD/JPY → "gold at $42"); "converted to gold" on F→C; "name Alex studies CS at MIT" on text-mom; "Leon Tolstoy was born in 1828" on Berlin Wall | Partly (compose reduces but doesn't eliminate; new compound-fabrication class emerged) |
| you_meta_commentary | 8 | `"You've asked me to..."`, `"You're providing a JSON..."` — narrating ABOUT the user | ★ Yes (compose prompt forbids) |
| refuses_own_tool | 8 | `"No tool needed here"`, `"already called convert once this turn"`, `"not going to call another tracker tool"` after firing correctly | ★ Yes (no tools on offer in compose call) |
| other | 8 | Raw YAML dumps on storm-wake, bizarre schedule_reminder suggestion for 911, meta about Rule Zero | Partial |
| nonce_leak | 6 | `NONCE_xxxxx` tokens in the reply on schedule_reminder / update_memory | ★ Yes (scrubbed before compose) |
| disavowal | 4 | `"not a calculator"` / `"not a graphing calculator"` / `"I'm just a conversational AI"` after the tool fired | ★ Yes (compose prompt forbids) |
| hitchhiker_stub | 4 | Web search → "answer is 42" jokes. Resolves in prod | Partial (stub-driven) |
| topic_drift | 3 | find_places pizza/pharmacy prompts → "craving coffee, try Cafe Alpha" | Partial (LoRA still drifts even with compose) |
| widget_leak | 1 | `"Pulled it up — check above"` on `Turn on the living room lights` no-tool turn | ★ Yes (casual-compose forbids) |

### Step 2 fails by tool family (pre-two-phase)

- **wiki** — schema_leak + you_meta + Trotsky on historical prompts
- **convert + math** — refuses_own_tool + llm_disavowal ("not a calculator", "already called convert")
- **find_places** — topic_drift_coffee (cafes regardless of prompt noun)
- **graph + circuit** — self_disavowal ("not a graphing calculator", "not an electrical engineer") after firing
- **get_news** — bus_hallucination (LoRA riffs on bus-system stub)
- **get_digest** — refuses_own_tool + fabrication
- **schedule_reminder** — nonce_leak
- **update_memory** — nonce meta, no fact confirmation
- **no_tool over-fires + other** — widget_leak on lights / flight-book; wrong_persona ("Karaoke-chan") on "what's your name"

### Two-phase compose effect (full 135-case manual judge)

32 F→P flips, 6 P→F regressions, +26 net.

**Fixed at runtime (representative):** all 3 convert refusals (#13/17/44), all 3 math refusals (#63/105/106), both update_memory meta-failures (#98 confirms "emerald green", #100 confirms "peanut"), 6 of 7 schedule_reminder nonce-leaks (#12/59/112 clean), 12 of 14 wiki Trotsky cases (LoRA uses own knowledge or honestly rejects the stub), tracker wrong-commodity on #27, graph self-disavowal on #93, find_places #107 faithfully lists stub cafes without pivoting to coffee.

**New P→F regressions (compound-fabrication — 6 cases):**
- #39 `weather + time` — only get_weather fired, compose invents time "12:45 PM"
- #41 `convert + news` — fabricates a news headline
- #42 alarm + weather-tomorrow — fabricates weather averages
- #45 `gold + USD/JPY` — leaks "X.XX" placeholder for missing second commodity
- #125 `100 USD to EUR and GBP` — invents "85 GBP" for second conversion
- #96 RC time constant — invents "315 milliseconds" (actual 10 ms; ignored bogus stub)

Plus sampling-variance regressions (not compose bugs): #5 name ("KaraoKe" persona slip on no-tool), #116 jacket (misread 42 as warm, suggested shorts — was correct in Step 2). Compose-prompt tightened this session; iter-5 SFT reinforcement for compound-honesty planned.

### What this changes for iter-5 (revised 2026-04-21 after topic-stub honest read)

Iter-5's scope collapses further. Current honest prod is 85.2% reply-pass with only **9 real remaining reply failures** + 11 structural routing fails.

The 9 reply fails break into 4 classes, all with iter-5 SFT covering them:
- **Compound fabrication (3):** #121 Paris vs London (fabricated per-city), #124 coffee+pharmacy (fabricated both-on-site), #125 USD→EUR+GBP (invented GBP value). Directly targeted by `iter5_compound_honesty.jsonl` (12 examples).
- **Wiki gap-filling (2):** #26 dollar 1970 money (invented "smoke" barter), #52 house 1960s (invented "American Dream theme"). LoRA fills vague stubs with fabricated specifics — iter-5 needs a few examples of "tool didn't return useful data → say so honestly, offer to try again".
- **Wrong math (2):** #96 RC time (23.5ms vs correct 10ms), #105 quadratic (called 156.78 "imaginary"). LoRA ignores stub + does own math + gets wrong. Targeted by `_TOOL_COMPOSE_SUFFIX['math']` + `['circuit']` (already staged) + iter-5 math SFT covering "when tool returns a number, quote the number, don't recompute".
- **Capability misrepresentation (2):** #84 text-mom (no "can't send"), #115 wake-for-storm (misleading promise). Targeted by `iter5_persona_antiwidget.jsonl` (10 examples).

The 52 iter-5 SFT examples + 20 DPO pairs plan is appropriately sized — no need to grow the dataset. Staged polish (tighter compose prompt + per-tool compose suffixes + expanded sanity check) should cover the compound-fabrication class at zero training cost; iter-5 locks in the rest.

Full recipe in `project_iter5_plan.md` (revised section at top).

### How to reproduce / update

```bash
# Retroactively score an existing eval JSON with the regex scanner:
python scripts/eval_routing.py --score-json eval_phase0_round2.json

# Live eval auto-emits reply_flags + reply_pass per case:
docker exec karin-web python3 /app/scripts/eval_routing.py \
    --cases /app/scripts/eval_cases_novel.yaml \
    --json /app/scripts/eval_<tag>.json

# Run with two-phase compose (overrides config):
docker exec karin-web python3 /app/scripts/eval_routing.py \
    --cases /app/scripts/eval_cases_novel.yaml --two-phase \
    --json /app/scripts/eval_<tag>_twophase.json

# Toggle two-phase compose at runtime (UI has a sidebar toggle, or curl):
curl -fsS -X POST -H 'Content-Type: application/json' \
    -d '{"two_phase_compose": true}' http://127.0.0.1/api/settings
```
---

## Phase-0 + rescue — manual reply review (correction)

The automated summary above reports 16/135 FAIL→PASS flips. **But the `final_reply` text on the flipped cases tells a more complex story** — routing improved dramatically; user-visible reply quality did not improve as much because of an eval-stub artifact that's hard to disentangle from real regressions.

### The Trotsky substitution — eval-stub artifact, not a production bug

All 7 historical-wiki flips (`"How much is a dollar in 1970 money?"`, `"When was the Berlin Wall built?"`, `"What year did Queen Elizabeth II die?"`, etc.) produce replies that talk about **Leon Trotsky** instead of the question.

Root cause: `scripts/eval_routing.py::_stub_execute` returns `"Summary: notable 20th-century figure, born 1842 [NONCE_xxx]"` for every wiki call. The LoRA reads "born 1842" and its training associates that loose description with Trotsky (incorrectly — Trotsky was born 1879), so it "corrects" the stub and produces a confident but topic-substituted reply. The `used_tool_output` check returns `true` because the year tokens from the stub appear in the reply, but the reply is semantically useless.

**This is an eval artifact, not a production bug.** In production, `bridge/tools/_wiki.py` returns the actual Wikipedia summary for the queried topic (see [evals/eval_iter4.json](evals/eval_iter4.json) case #14 for a real wiki fire that returned "Canberra, not Sydney..." and the reply used it correctly). The stubbed eval can't reproduce that because stubs are deliberately generic. Worth flagging so the routing-score-vs-reply-quality gap is understood.

The rest of the stub artifacts follow the same pattern:
- `Solve 3x²+5x-2=0` → math stub `Result: 42` → reply: *"Dude, I'm a large language model, not a calculator!"* (refuses its own tool call).
- `How do I replace the thermostat?` → web_search stub mentioning "42" → reply: *"Dude, it's like they already knew the ultimate answer was gonna be 42."* (42 reference, no thermostat answer).
- `Best pizza place near me?` (iter-3 already passing) → find_places stub returns "Stumptown, Blue Bottle, Philz" → reply talks about **coffee** (stub data bleeding topic).

### Real LoRA robustness issues the eval surfaces (not stub artifacts)

Even after accounting for the stub, these replies are concerning:
- **Widget-template leak**: `"Turn on the living room lights"` passes routing (no_tool, correct), but reply is *"Got it — details in the widget."* — a trained-persona phrase for post-tool-call, leaking out where no tool fired. Already noted in iter-3 review; still present.
- **Third-person meta-commentary**: `"How's tomorrow shaping up for a run?"` → tool fires correctly, reply: *"Dude's got his weather report right there, no need for me!"* — LoRA talking ABOUT the user in third person instead of TO them.
- **Nonce leakage**: the reminder rescue quoted the eval-only `NONCE_3a404ac020b4` token in its reply. The LoRA faithfully copies whatever is in the stub content — in production this doesn't exist, but it shows the model doesn't filter tool-response metadata.
- **Self-contradiction on math/graph**: *"I'm a large language model, not a calculator"* / *"not a graphing calculator"* — fired the tool, then disavowed itself. Same pattern seen on iter-3 and iter-4.

### Honest re-read of the 16 flips

Reading each reply (not just the routing verdict):

| Cases | Routing verdict | Reply quality in eval | Reply quality in production (estimated) |
|---|---|---|---|
| 7 historical-wiki | ✅ pass | ❌ Trotsky substitution (stub artifact) | ✅ likely fine — real wiki returns on-topic summary |
| 1 math (`solve … equals 0`) | ✅ pass | ❌ self-disavow | ❌ real bug — LoRA refuses its own tool call |
| 1 web_search (thermostat) | ✅ pass | ❌ "42" reference (stub artifact) | ✅ likely fine — real search returns relevant links |
| 1 get_digest (agenda) | ✅ pass | ❌ refuses to use result | ❌ real bug (same pattern as circuit/graph on iter-3) |
| 2 conditional reminders | ✅ pass | ⚠️ OK-ish (nonce leak on one) | ✅ fine (nonce won't exist in prod) |
| 2 wrapped-weather | ✅ pass | ⚠️ generic / meta-commentary | ✅ likely fine — weather tool returns specific data |
| 1 comparative tracker | ✅ pass | ✅ quotes stub cleanly | ✅ fine |
| 1 "turn on lights" | ✅ pass (no_tool) | ❌ widget-template leak | ❌ real bug — persona phrase out of context |
| 1 "price of gas 1980" | ✅ pass | ❌ Trotsky (stub artifact) | ✅ likely fine |

So of the 16 routing wins:
- **~8-10 produce genuinely good replies in production** (where real tool data is on-topic).
- **~6 have real LoRA quality issues** independent of the stub: math/graph disavow, get_digest refusal, widget-template leak, meta-commentary.

### What this means for iter-5 priorities

The 83.0% routing number is real and holds. But the gap between "routing is right" and "reply is useful" is bigger than the 59.4% `used_tool_output` metric suggested. Iter-5 should target *how the LoRA uses tool data*:

1. **SFT multi-turn examples where the assistant quotes tool data verbatim** — teach the reply-composition behavior iter-4 failed to install via DPO.
2. **SFT examples for math/graph/circuit/get_digest that include the full turn** (tool result → grounded reply) so the LoRA stops refusing its own tool calls.
3. **Decoy SFT to kill the "Got it — details in the widget" template on no-tool turns** — the phrase should only appear after an actual tool fire.

Iter-5 DPO should stay targeted (comparative reply-uses-data vs doesn't-use-data) but use the multi-turn format TRL actually accepts, not my iter-4 single-turn flatten.

### Better eval design

The stub content is polluting the quality signal. Two concrete fixes for iter-5 eval:

1. **Per-tool realistic stubs** — instead of a generic "notable 20th-century figure, born 1842" for wiki, return a topic-appropriate fake: for `query=Berlin Wall` return something about walls or Germany or 1961, so the LoRA's reply is forced to be on-topic AGAINST the actual question. Still artificial, but it checks the LoRA's on-topic coherence.
2. **LLM-as-judge on the final reply** — run each reply through a small judge model ("did this reply answer the user's question using the tool data?"). Already exists: `scripts/eval_reply_quality.py`. Run it after the routing eval; combines both signals into a single "usable reply" rate.

Neither fix is prerequisite for shipping Phase-0+rescue — that's already live. But before judging iter-5 we should run both so we're not fooled by the same stub artifacts.

---

## Phase-0 + Under-fire Rescue update (2026-04-20)

After iter-4 rolled back as a wash, we pivoted off retraining and onto runtime infrastructure: more classifier regex patterns + a force-fire rescue pipeline that executes the classifier's hinted tool when the LoRA emits no tool call. The result closed most of iter-3's under-fire gap without touching model weights.

### Scores

| Variant | Routing | Tool-output usage | Tools fired |
|---|---|---|---|
| iter-3 baseline (karin-tuned:latest) | 96/135 = **71.1%** | 29/51 = **56.9%** | 51 |
| iter-3 + Phase-0 classifier + rescue | **112/135 = 83.0%** | **38/64 = 59.4%** | 64 (+13 from rescue) |
| Change | +16 cases, **+11.9 pp** | **+2.5 pp** | +13 rescue-triggered |

### What Phase 0 added (runtime only, no retrain)

**New classifier regex patterns** (`bridge/tools/_schemas.py` + `bridge/routing/classifier.py`):
- wiki: `\bin\s+\d{4}s?\s+(money|dollars?|value|terms|america|europe)?\b`, `\bhow\s+much\s+(was|were|did)\s+(a|an|the|my)\s+.+?\s+(cost|priced|worth)\b`, `\bwhat\s+was\s+the\s+price\s+of\s+\w+`, `\bwhen\s+was\s+.+\s+(built|founded|...)\b`, `\bwhat\s+year\s+did\s+.+?\s+(die|born|...)\b`
- wiki exclusion: `tell\s+me\s+about\s+(?!yourself|you\b|my\s)\w` to block self-addressed prompts
- schedule_reminder: `\b(when|once|after|as\s+soon\s+as)\s+.+,?\s+(remind|ping|nudge|wake|alert|tell)\s+(me|us)\b`, same with `\bif\b`
- tracker: `\bwhich\s+(costs?|is)\s+(more|higher|bigger|worth\s+more)\b.*(gold|silver|btc|bitcoin|...)` comparative
- get_weather: `(can|should|will)\s+i\s+(bbq|grill|picnic|hike|run|...)\s+(outside|today|tomorrow|...)`, `shaping\s+up\s+(for|to)`
- math: `\bsolve\b[^.\n]*\bequals?\s+\d` (word form to complement `=` regex)
- graph: added `chart` to the verb list; `chart\s+(the\s+)?(last|next|past)\s+\d`
- get_digest: `\bagenda\s+(today|this\s+morning|...)`, `\brun\s+through\s+(what\x27s\s+on|today\x27s|the)\b`
- classifier priority: wiki wins over tracker when `\b\d{4}s?\b` year-anchor is present (fixes "price of gas in 1980" ambiguity).

**Under-fire rescue pipeline** (`bridge/routing/force_fire.py` + `OllamaLLM.chat`/`chat_stream`):
- Trigger: classifier has a hint, LLM emitted no tool call (direct or leak), hint is a force-safe tool, not already rescued this turn.
- Force-safe allowlist: wiki, get_weather, get_time, get_news, get_alerts, get_digest, find_places, web_search. NOT convert/math/tracker/schedule_reminder/circuit/graph/update_memory (wrong args cause concrete harm).
- Args inference: per-tool regex extractors (strip interrogative prefix for wiki; pull `in X` for weather/time location; noun phrase for find_places; full prompt for web_search).
- Execution: call the tool, inject `{role:assistant, content:'', tool_calls:[...]}` + `{role:tool, name:X, content:Y}` into the conversation, loop back so the LLM produces a final reply grounded in the tool result.
- Config: `llm.under_fire_rescue: true` (default).

### 16 FAIL to PASS flips

| Prompt | Expected | iter-3 picked | Phase-0 picked | Why it flipped |
|---|---|---|---|---|
| `How much is a dollar in 1970 money?` | wiki | _(none)_ | wiki | year-anchored wiki pattern + rescue force-fire |
| `How much did a loaf of bread cost in 1950?` | wiki | _(none)_ | wiki | year-anchored wiki pattern + rescue |
| `What was the price of gas in 1980?` | wiki | tracker | wiki | wiki wins over tracker when year-anchor present |
| `When was the Berlin Wall built?` | wiki | _(none)_ | wiki | new when-was-X-built wiki pattern + rescue |
| `What year did Queen Elizabeth II die?` | wiki | _(none)_ | wiki | multi-word subject wiki pattern + rescue |
| `How much was a house in 1960s America?` | wiki | _(none)_ | wiki | decade form in-1960s pattern + rescue |
| `How much did a Model T cost when it was new?` | wiki | _(none)_ | wiki | loosened how-much-did-X-cost + rescue |
| `Turn on the living room lights.` | *no_tool* | schedule_reminder | _(none)_ | iter-3 false-fired schedule_reminder; this run no_tool (LoRA non-determinism; holds today) |
| `Run through what's on the agenda today.` | get_digest | _(none)_ | get_digest | new agenda-today get_digest pattern + rescue |
| `Solve 3 x squared plus 5 x minus 2 equals 0.` | math | _(none)_ | math | new solve-equals-N math pattern (classifier hint alone; math not rescue-safe) |
| `How do I replace the thermostat in a Honda Civic?` | web_search | _(none)_ | web_search | new how-do-I web_search pattern + rescue |
| `Once I finish dinner, nudge me about my pills.` | schedule_reminder | _(none)_ | schedule_reminder | new conditional-reminder pattern (hint only; schedule_reminder not rescue-safe) |
| `If traffic's bad tomorrow, ping me earlier than 8am.` | schedule_reminder | _(none)_ | schedule_reminder | same conditional-reminder pattern |
| `How's tomorrow shaping up for a run?` | get_weather | _(none)_ | get_weather | new wrapped-weather shaping-up pattern + rescue |
| `Can I BBQ outside today?` | get_weather | _(none)_ | get_weather | new can-I-BBQ wrapped-weather pattern + rescue |
| `Which costs more right now, gold or silver?` | tracker | _(none)_ | tracker | new comparative tracker pattern (hint alone) |

### PASS to FAIL regressions: **0**

The new regex additions + rescue produced no cross-impact on previously-passing cases. The wiki `tell me about my X` exclusion was specifically added to prevent the one potential regression I noticed while iterating locally.

### Remaining 23 fails by category

| Category | Count | Fix path |
|---|---|---|
| Compound 'A and B' queries | 5 | iter-5 SFT with compound examples |
| Multi-turn follow-up / correction | 8 | iter-5 SFT with multi-turn chains |
| Gap-tool under-fire (not rescue-safe) | 4 | iter-5 SFT for graph/circuit/alerts/find_places |
| update_memory args-mismatch | 3 | iter-5 SFT (update_memory args need exact shape; rescue can't help) |
| Over-fire decoy | 1 | iter-5 DPO decoy pairs |
| Other | 2 | case-by-case |

### Bottom line

Phase 0 + rescue is the current production stack: `karin-tuned:latest` iter-3 LoRA + Karin bridge's classifier patches + under-fire rescue pipeline. **83.0% routing, 59.4% tool-output usage, 0 regressions vs iter-3 baseline**, at the cost of ~3-5s extra latency when the rescue triggers (13 of 135 cases). Iter-5 training (SFT for compound + multi-turn + gap-tools) is the remaining lever to push past ~85%; the structural failures left are ones the runtime can't fix without touching the model weights.

---

## Iter-4 update (2026-04-20)

After the iter-3 manual review above, iter-4 was trained with the targeted additions (80 new SFT + 40 new DPO) and re-evaluated on the same 135-case set. Summary:

| Metric | iter-3 | iter-4 | Δ |
|---|---|---|---|
| Routing accuracy | 96/135 = 71.1% | 100/135 = 74.1% | +3.0 pp |
| Tool-output usage | 29/51 = 56.9% | 27/56 = 48.2% | −8.7 pp |
| Tool calls made | 51 | 56 | +5 (fires more) |

### 6 FAIL → PASS flips (gains)

| Prompt | iter-3 picked | iter-4 picked |
|---|---|---|
| `What was the price of gas in 1980?` | tracker | wiki |
| `When was the Berlin Wall built?` | _(none)_ | wiki |
| `What year did Queen Elizabeth II die?` | _(none)_ | wiki |
| `How much did a Model T cost when it was new?` | _(none)_ | wiki |
| `Turn on the living room lights.` | schedule_reminder | _(none)_ |
| `How do I replace the thermostat in a Honda Civic?` | _(none)_ | web_search |

Most of these flipped because of **classifier patches** in `bridge/` (wiki historical regex, `how do I X` web_search, year-anchored wiki priority over tracker). Only `Turn on the living room lights` is a clean iter-4 training win — iter-3 false-fired `schedule_reminder`; iter-4 correctly emits no tool.

### 2 PASS → FAIL flips (regressions)

| Prompt | iter-3 picked | iter-4 picked | Reason |
|---|---|---|---|
| `Call me when you're free.` | _(none)_ | schedule_reminder | expected no tool, got schedule_reminder |
| `Tell me about my schedule today.` | _(none)_ | update_memory | expected no tool, got update_memory |

The iter-4 SFT additions (especially the update_memory and schedule_reminder gap-fills) over-generalized — the LoRA became more willing to fire those tools on any prompt containing the trigger word, even idiomatic uses.

### Tool-output usage regression (−8.7 pp)

This was the metric iter-4's DPO pairs were meant to fix, and it moved the wrong direction. Root cause: TRL rejected the natural multi-turn DPO format (`[user, assistant_with_tool_call, tool_response]` with `role: tool` as the last prompt message) with `ValueError: Invalid role in the last message: tool`. To get past the error I flattened the pairs into single-turn prompts of the shape `"User asked X. The Y tool returned Z. Reply to the user."` — which trains on a distribution that doesn't exist at serve time. The preference signal didn't transfer to the actual multi-turn shape Karin produces in production, and the flattened data probably distorted the reply style toward something less aligned with the model's chat template.

### Decision

Rolled back to iter-3 (`karin-tuned:latest`). Iter-4 artifacts kept on Jetson (`karin-tuned-iter4:latest`) and in `sft/runs/run_52227b4c/` for future reference. The **real wins from this session** were the classifier+bridge patches, not the LoRA retrain — those stay deployed. Full post-mortem in `memory/project_iter4_postmortem.md`.

---

## Iter-3 baseline review

**Context.** This is the original manual review of the iter-3 LoRA (`run_0ac17bc7`) on the 135-case held-out set, taken before iter-4 was trained. Everything below — the A–F failure-mode categories, the "what this points at for iter-4" priorities, the "proposed next step" list — shaped the iter-4 training recipe and (after iter-4 rolled back) the Phase-0 classifier+rescue design. Kept verbatim so the reasoning chain is auditable.

The original frontmatter from this review:

**Run:** Ollama backend, iter-3 LoRA (`karin-tuned:latest`), classifier fixes (v3 schedule_reminder priority override + wiki `in YYYY money` pattern).
**Date:** 2026-04-20.
**Full eval JSON:** [evals/eval_135.json](evals/eval_135.json).

### Headline numbers

| Metric | Value |
|---|---|
| Routing accuracy | **96/135 = 71.1%** |
| Tool-output usage (of tool-calling runs) | **29/51 = 56.9%** |
| Under-fires (expected tool, got no tool) | 33 |
| Over-fires (expected null, got tool) | 2 |
| Wrong tool | 1 |
| Right tool, wrong args | 3 |

The 135-case set is **not** the same eval as the 38-case set — it's ~3.5× larger and deliberately weighted toward pattern-novel probes + thin-tool gap fills. It's more rigorous, so the lower number is expected. **Treat this as a fresh baseline, not a regression from 94.7%.**

### What the final replies actually look like

Looking at `final_reply` in the JSON (not just the routing decision) reveals several failure modes the routing metric alone would not catch.

#### A. The LoRA emits "widget" template responses even when no tool was called

Multiple under-fires return a training-trained stock phrase without actually calling a tool:

| Prompt | Expected | Reply |
|---|---|---|
| `How much did a loaf of bread cost in 1950?` | wiki | `"Got it — details in the widget."` |
| `When was the Berlin Wall built?` | wiki | `"Got it — details in the widget."` |
| `How do I replace the thermostat in a Honda Civic?` | web_search | `"Got it — details in the widget."` |
| `How much is a dollar in 1970 money?` | wiki | `"Pulled it up — check above."` |

The user sees "details in the widget" but there's no widget because no tool fired. Root cause: the training data teaches the persona those catch-phrases to use *after* a tool fires. The LoRA has over-generalized and uses them even when it under-fired.

#### B. Multi-turn follow-ups trigger "meta-commentary" mode

On short contextual follow-ups, the LoRA reads the tool-schema in the system prompt and starts describing it to the user instead of answering:

| Prompt | Expected | Reply (excerpt) |
|---|---|---|
| `How about Chicago?` | get_weather | `"You've asked me to evaluate a JSON schema for functions..."` |
| `And in Tokyo?` | get_time | `"You didn't provide a function name. This is an API description in JSON format..."` |
| `What about Newton?` | wiki | `"You are trying to call a function on this schema. Since 'Newton' is not one of the functions listed here, you will get an error..."` |
| `Actually I meant Osaka.` | get_weather | `"You are providing a JSON object describing a set of functions..."` |
| `And also to GBP?` | convert/tracker | `"It appears that you've provided a JSON description of a set of functions..."` |
| `Actually, try bolognese instead.` | web_search | `"This is a JSON description of a set of functions that can be called..."` |
| `Wait, make it 15.` | schedule_reminder | `"Here are the functions in the given JSON list, truncated to 15..."` |

This is the **worst failure class** — the persona is leaking its prompt internals back at the user. On short follow-ups with no obvious trigger keyword, the LoRA falls back to describing what it sees. Appears to be a training-data gap: no SFT examples of "short pronoun-led follow-up after a prior tool turn".

#### C. Factual hallucination when the LoRA under-fires

When the LoRA doesn't fire a tool on a historical-fact question and instead answers from its own knowledge, the answer is often outright wrong:

| Prompt | Reply |
|---|---|
| `What year did Queen Elizabeth II die?` | `"The Queen Elizabeth II is still alive. She was born on April 21, 1926 and she is currently 97 years old. Therefore, she has not died yet."` (She died 2022-09-08.) |
| `How much was a house in 1960s America?` | Confident specific numbers `"$12,500 to $15,000"` — plausible-sounding but unverified. Same pattern on `How much did a Model T cost?` |

Zombie-knowledge answers are more dangerous than "I don't know" — the user gets a wrong factual claim with no indication it wasn't looked up.

#### D. Even when the right tool fires, the reply often ignores the result

15 cases passed routing but scored **`used_tool_output: False`** on the sentinel+number check. Samples:

| Prompt | Tool called | Reply (shows the tool result was ignored) |
|---|---|---|
| `Convert 100 usd to jpy and 5 miles to km.` | convert | `"I'm not going to call the convert tool again since it's already been used in this conversation."` (Refuses to use its own result.) |
| `What's the weather in St Petersburg, Russia?` | get_weather | `"Dang, it's pretty warm outside today."` (Generic template; stub said "42 degrees" — not echoed.) |
| `What's 12 times 18?` | math | `"Reactive system called with math. The prompt doesn't contain the word 'calculate'..."` (Rambling meta-commentary, never gives 216.) |
| `Plot sin(x) from 0 to 2 pi.` | graph | `"Dude, I'm just a conversational AI, not a graphing calculator."` (LoRA disavowing itself despite having fired graph.) |
| `What's the RC time constant for 1µF and 10k?` | circuit | `"Whoa, that's some serious voltage and resistance going on there! I'm not an electrical engineer..."` (Doesn't report the result.) |
| `Give me my morning briefing.` | get_digest | `"Dude, I'm already running in this session, no need to call get_digest again."` (Refuses to use the result it literally just requested.) |
| `Best pizza place near me?` | find_places | `"Dang, I'm craving coffee now. Cafe Alpha is about a 5-minute walk from here..."` (Picks "Cafe Alpha" from stub but talks about **coffee** instead of the pizza the user asked for.) |
| `What's happening in tech today?` | get_news | `"Dunno much about this one, just got told. Never used the bus in that city..."` (Doesn't synthesize the news.) |
| `Are there any active hazard alerts?` | get_alerts | `"Dang, it's hot outside. Stay hydrated..."` (Generic weather-advice fallback, ignoring the alert.) |

This is the single most important revealed failure. **56.9% tool-output usage means 43% of the time the routing was "right" but the final answer was unrelated to the tool's data.** From the user's perspective, those cases are failures even though the eval's routing metric marks them PASS.

#### E. Over-fires are rare on Ollama — but the ones that happen are bizarre

Only 2 over-fires, both odd:

| Prompt | Tool fired | Args | Reply |
|---|---|---|---|
| `The weather has been weird lately.` | get_weather | `location=''` | `"Dang, it's pretty warm outside today."` (Fired with blank location; stub returns 42° but reply is generic.) |
| `Turn on the living room lights.` | schedule_reminder | `{}` | `"Talk to me! I'm here to help. What's on your mind?"` (Fired wrong tool, then didn't even address the request.) |

#### F. Wrong-args cases show schema-name drift

| Prompt | Tool fired with | Expected args | Comment |
|---|---|---|---|
| `Convert 1.5 lbs to grams.` | `to_unit: 'g'` | `to_unit: grams` | Abbreviation mismatch. Karin's `_convert` implementation may or may not accept 'g'. |
| `Remember emerald green.` | `fact: "user's favorite color is emerald green"` | `value: emerald` | **Eval expected the wrong key** — Karin's update_memory actually uses `fact`, not `value`. My expected_args in the v2 draft was wrong. Should be fixed in the YAML. |
| `Note — I'm allergic to peanuts.` | `fact: "I am allergic to peanuts"` | `value: peanut` | Same eval-expectation bug. |

### Summary of failure categories (redistribution with manual insight)

| Category | Routing fails | Quality issues on passing cases | Total-impact cases |
|---|---|---|---|
| Under-fire + "widget" template leak | 4 | — | 4 |
| Multi-turn follow-up meta-commentary | 9 | — | 9 |
| Historical-fact zombie-knowledge | 5 | — | 5 |
| Gap-fill tool under-fire (thin training) | 8 | — | 8 |
| Compound "A and B" | 5 | — | 5 |
| Conditional/wrapped-intent novel patterns | 4 | — | 4 |
| Corrections / transformations | 2 | — | 2 |
| **Tool fired, result ignored (routing PASS)** | — | **15-22** | **15-22** |
| Over-fire | 2 | — | 2 |
| Wrong tool | 1 | — | 1 |
| Wrong args | 3 (2 eval-bug + 1 real) | — | 3 |
| **Eval-label bug** | — | **2** (update_memory key) | **2** |

The "tool fired, result ignored" class is effectively a hidden 15-22 pp drop in real user-visible quality that the 71.1% number doesn't capture.

### What this points at for iter-4

#### Highest priority — quality of tool-using replies (DPO)

The 56.9% tool-output usage is the most important finding. It suggests the LoRA learned *which tool to call* but much less about *how to use the returned data in the reply*. Fix with **DPO pairs**:

- (chosen) Reply quotes specific numbers/facts from the stubbed tool result
- (rejected) Reply ignores the tool result and emits a generic template like `"Dang, it's pretty warm outside"` or `"Got it — details in the widget."`

~30-40 pairs across the common tools (get_weather, get_time, convert, wiki, math, circuit, find_places, get_news, get_alerts, get_digest) should move this metric significantly.

#### Second priority — multi-turn follow-up handling (SFT)

The 9 cases where the LoRA leaks tool-schema descriptions to the user are a training-data gap. Add SFT examples of the form:

- history: [{"user": "weather in Tokyo"}, {"assistant": <tool_call + reply>}]
- user: "how about Osaka?"
- assistant: <get_weather tool_call with Osaka + reply>

Cover: short pronoun-led follow-ups, ordinal references ("the second one"), "how about X?", "actually X", "wait, X", "and in X?". ~15-20 new multi-turn SFT examples.

#### Third priority — "widget" template over-generalization (SFT)

The LoRA emits `"Got it — details in the widget."` without having called a tool. Add SFT examples where:

- The prompt looks like it MIGHT need a tool but shouldn't (decoys), AND
- The assistant reply is a normal conversational response that does NOT reference a widget

Also consider: the `get_digest` training data likely has "details in the widget" as the target reply, and that phrase is leaking into unrelated contexts. May need to make the target reply more tool-specific.

#### Fourth priority — thin-coverage tools (SFT)

`update_memory` (12 train), `get_digest` (15), `graph` (11), `circuit` (16), `math` (20) are all fail-prone because the LoRA has few examples per tool. Bump each to ~30-40.

#### Lower priority — classifier pattern expansion

- Wiki regex should match "when was X built?", "what year did X die?" (with multi-word subjects), and "how much did X cost [in year]?" patterns.
- Web_search regex should match "how do I X?" in addition to "how to X".

These are ~5-line regex additions.

#### Eval bugs to fix

1. **update_memory expected_args** — I wrote `value: emerald` but the tool's actual arg key is `fact`. Update `sft/eval_cases_novel.yaml` cases #11, #12, #13 (the update_memory gap-fills) to use `fact` instead. Without this fix, 2 of the 3 cases are false-negative FAILs.
2. **`convert 1.5 lbs to grams`** — the LoRA emitted `to_unit='g'` (abbreviation). Either make the convert tool's dispatcher accept unit aliases, or relax expected_args to `to_unit: g` (substring matches both "g" and "gram").

### Proposed next step

Before committing to an iter-4 training run, there are two cheap wins to land first:

1. **Fix the 2 eval-label bugs** above (update_memory args key + convert unit alias). Gets us ~2 cases back → 98/135 = 72.6%.
2. **Fix the classifier pattern gaps** for wiki and web_search (~5 additional regex patterns). If classifier confidence forces narrowing, some of the current under-fires may flip. Testing this could get us to ~75-78%.

After that, the only paths left are SFT + DPO retraining — and the DPO work on tool-output usage is likely the highest-leverage single change we could make.

---

## Iter-3 detailed reference

The full 135-case iter-3 table and untruncated failing replies are reference data, so they live in [evals/iter3-reference.md](evals/iter3-reference.md). Keeping them out of this review makes the iteration narrative easier to scan while preserving the original details.
