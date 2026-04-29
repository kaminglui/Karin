# Iter-5 post-mortem — 2026-04-22

> **Context note (added 2026-04-23).** This post-mortem compares iter-5 to the *then*-prod baseline of 91.1% routing / 85.2% reply. Prod has since shifted to **93.3% routing / 91.9% reply** via the `hint_in_user_msg` refactor (same iter-3 LoRA, different prompt structure). iter-5 remains not shipped. See `docs/routing-pipeline.md` → "Hint-in-user-msg" and `CHANGES.md` 2026-04-22 entry.

## TL;DR

- **Bare iter-5 LoRA routing: 106/135 = 78.5 %.**
- **Current prod (iter-3 + Phase-0 + rescue): 123/135 = 91.1 %.**
- Net delta: **−17 cases** (20 regressions, 3 gains). Iter-5 was **not shipped**.
- Rollback is a no-op: prod Jetson `.env` still points at `karin-tuned:latest` (iter-3). The iter-5 model lives as `karin-tuned-iter5:latest` on the Jetson Ollama store for diagnostic access but is not wired in.
- The **masking fix itself worked**: zero `═══`/`background-pings` prompt leaks on iter-5 outputs, and the held-out validation holdout hit 92 %. The regression was caused by **dataset imbalance amplified by the new loss mask**, not by the mask.

---

## What was in the release

Compared with iter-3 SFT data (374 rows):

1. **Training-time loss mask** — `DataCollatorForCompletionOnlyLM(instruction_template="<|start_header_id|>user<|end_header_id|>", response_template="<|start_header_id|>assistant<|end_header_id|>", tokenizer=tokenizer)`. Primary design goal: stop the LoRA from memorising the 8 KB system prompt (root cause of the `Hello → prompt paraphrase` leak).
2. **+52 new SFT rows**:
   - `iter5_structural.jsonl` — 20 rows (continuations, graph gap-tool, circuit gap-tool, find-places directions, update_memory, weather-decoy no-tool, get_digest)
   - `iter5_compound_honesty.jsonl` — 12 rows (acknowledge the missing half of an A+B query instead of fabricating)
   - `iter5_update_memory_args.jsonl` — 10 rows (short-fact phrasings)
   - `iter5_persona_antiwidget.jsonl` — 10 rows (Karin identity; capability honesty — "I can't turn on lights" style)
3. **DPO skipped** (`run_dpo=False`). Multi-turn DPO rebuild deferred.

Training config: `seed=42`, 2 epochs, r=8 / α=32 / dropout=0.1, `assistant_only_loss` via collator. Probe passed at 1.8 % unmasked (healthy; within the 0.5-5 % band for this dataset shape).

Adapter: `sft/runs/run_7db68f92/karin-lora.gguf`. Installed on Jetson as `karin-tuned-iter5:latest`.

---

## The regressions — two clear patterns

### Pattern A: under-fire (16 of 20)

LoRA emits **no tool** on prompts that should have triggered one:

```
Remind me to check the gold price tomorrow at 9am.    → None  (want schedule_reminder)
Convert 1.5 lbs to grams.                             → None  (want convert)
Calculate the RC time constant for 1μF and 10kΩ.      → None  (want circuit)
Voltage divider output for 12V across 10k and 4.7k.   → None  (want circuit)
When I get home, remind me to feed the cat.           → None  (want schedule_reminder)
Remind me after my 3pm meeting to send the report.    → None  (want schedule_reminder)
If traffic's bad tomorrow, ping me earlier than 8am.  → None  (want schedule_reminder)
Wake me if a storm comes through tonight.             → None  (want schedule_reminder)
Set an alarm for 7am and tell me the weather.         → None  (want schedule_reminder)

How about Chicago? / And in Tokyo? / Actually Osaka?  → None  (continuation, want previous tool)
Anything on tech specifically?                        → None  (continuation, want get_news)
Actually, try bolognese instead.                      → None  (continuation, want web_search)

What's the time zone in Tokyo?                        → get_time only (want get_time OR wiki)
What's the weather in NY and any alerts?              → get_alerts only (want both)
Tell me the weather and what time it is.              → get_weather only (want both)
```

### Pattern B: over-fire on newly-introduced tools (2 of 20)

```
Cancel that.                     → graph              (should be None)
Tell me about my schedule today. → schedule_reminder  (should be None)
```

### Actual gains (3)

```
Convert 50 euros to dollars and give me the news.  → compound handled (was iter-3 miss)
Book me a flight to Tokyo.                         → honest no-capability reply (was fail)
Chart the last 7 days of temperatures.             → graph tool now exists & routes
```

The gains prove the new-data intent was sound. The failure is in **data balance**, not in what the new rows try to teach.

---

## Root-cause analysis

The 10 `iter5_persona_antiwidget.jsonl` rows plus the 12 `iter5_compound_honesty.jsonl` rows concentrate on "emit no-tool reply + honest-decline text". That's 22 rows out of 426 — 5 % of the SFT set — all pushing the `no_tool` prior.

Three multipliers made this worse than it looks:

1. **Under-fire was iter-3's already-largest residual failure class.** Phase-0 + rescue exists specifically to compensate for iter-3's under-fire. Adding *any* more no-tool pressure bumps directly into that sore spot.
2. **Completion-only masking amplifies the per-row signal.** With only ~1.8 % of tokens carrying loss (system prompt + user turn are masked), each assistant turn's gradient has a much larger effective weight than in iter-3 (which trained on all tokens). The 22 no-tool rows punch above their count.
3. **Ambiguous-input prompts bias toward no-tool.** Short prompts like `How about Chicago?` or `And in Tokyo?` were already borderline in iter-3; any added no-tool pressure tips them over.

The over-fire on `Cancel that. → graph` is the flip-side: the iter5_structural graph rows taught the pattern "short imperative → graph" too broadly without paired "`cancel that.` → None" negatives.

---

## What worked (keep for iter-6)

1. **DataCollatorForCompletionOnlyLM with both-templates mode** (`instruction_template` + `response_template`). Probe lands consistently in the 0.5-5 % band. No prompt-leak outputs observed on any iter-5 reply. The mask is not the problem.
2. **Validation run methodology** — run the new mask on the OLD data first, confirm the probe and eval_loss behave sensibly, *then* change the data. Caught the TRL 0.15.2 flag-name issue before a real experiment was wasted.
3. **Notebook guardrails** — banner print + pre-train probe that raises on `<0.3 %` or `>40 %` unmasked. Already saved one dead-training run.
4. **Separate `dataset_version` per experiment** — each iter gets its own run-dir on Drive, zero risk of overwriting prior adapters.
5. **Skipping DPO** — correctly avoided replaying iter-4's single-turn-DPO corruption.

---

## What to change for iter-6

1. **Rebalance no-tool rows.** For every row that teaches "reply with no tool", add 2-3 rows with *similar-surface-form* tool-call prompts that must route correctly. Particularly:
   - For each `persona_antiwidget` "can't turn on lights" row, add a paired "turn on the living room lights" → `update_memory` / no-tool distinction with clear user-intent markers.
   - For each `compound_honesty` "A+B but only B possible" row, add a matched "A+B both possible" positive.
2. **Prune over-generalizing graph rows.** Two or three `iter5_structural.jsonl` graph rows taught `short imperative → graph`. Either narrow those to explicit graph-triggering phrasings (`chart`, `plot`, `graph`) or delete and retry.
3. **Add explicit negatives for short imperatives.** `cancel that.` / `never mind.` / `scratch that.` → None. iter-3 already handles these via the `_CANCEL_ACK_PATTERNS` classifier rule, but SFT should reinforce, not contradict.
4. **Continuation-over-followup bias.** 4 of 16 under-fires are pronoun-led continuations (`How about Chicago?`, `And in Tokyo?`, `Anything on tech?`, `Actually Osaka`). These are solved at runtime by the Phase-0 continuation rescue — but iter-6 SFT should still learn them directly so the rescue becomes unused on the happy path.
5. **Keep the mask. Keep `seed=42`. Keep the probe thresholds (0.3-40 %).** These are baseline.
6. **DPO rebuild — start when ~20 multi-turn pairs are ready.** This is orthogonal to the SFT fix above and should be its own iter.

---

## Artifacts preserved

- `docs/evals/eval_iter5.json` — raw 135-case eval output (106/135 pass)
- `sft/runs/run_7db68f92/karin-lora.gguf` — iter-5 adapter GGUF
- `karin-tuned-iter5:latest` — Ollama model on Jetson, not wired into prod
- `sft/sft_dataset.arrow` / `sft_dataset.tar` — iter-5 build (426 rows)
- `sft/sft_dataset.arrow.iter4` / `sft_dataset.tar.iter4` — iter-4 build, preserved for rollback

`sft/colab_sft.ipynb` currently reflects the iter-5 attempt (`dataset_version='iter5_sft_production'`, `run_dpo=False`). A warning block has been added to cell 5 flagging that these values were from a regressed run and should be re-examined for iter-6, not copy-pasted.

---

## Lesson

Data-balance bugs are invisible in training metrics. Both the held-out routing holdout (87.5 %) and the probe ratio (1.8 %) looked healthy. Only the production-equivalent 135-case eval caught the regression.

**Process change:** before declaring a training run a success, the 135-case eval is mandatory — the notebook's own held-out accuracy is a weak signal because the holdout is drawn from the same distribution as the skew we're adding.
