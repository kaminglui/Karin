# Iter-7 post-mortem — 2026-04-23

## TL;DR

- **Iter-7 scored 118/135 = 87.4% routing and 83/135 = 61.5% reply-pass** on the 135-case eval. Prod (iter-3 + Phase-0 + two-phase + hint-in-user-msg) is 93.3% / 91.9%.
- **Net: −5.9 pp routing, −30.4 pp reply, −48.2 pp tool-output-usage.** Reply-quality collapse is worse than iter-4 / iter-5 / iter-6.
- **Targeted bugs didn't get fixed.** Case 8 (thermostat fabrication) regressed from a soft "I've passed on your request" to a hard raw-JSON schema leak. Case 26 (stale PM) unchanged.
- **Not shipped.** Prod unchanged; `karin-tuned-iter7:latest` on Jetson Ollama for diagnostic access only. This is the fourth consecutive training attempt (iter-4 DPO, iter-5 SFT-rebalance, iter-6 SFT-rebalance, iter-7 DPO) to regress vs prod.

## What shipped (run `run_f7950a38`)

Configuration per `run_config.json`:

```json
{
  "dataset_version": "iter7_capability_current",
  "sft_epochs": 2,
  "sft_lr": 0.0001,
  "lora_r": 8, "lora_alpha": 32, "lora_dropout": 0.1,
  "weight_decay": 0.01,
  "dpo_lr": 5e-06, "dpo_beta": 0.1,
  "run_sft": true, "run_dpo": true, "run_grpo": false
}
```

- **SFT data**: 294 rows (iter-3-era only — iter-4/5/6 files moved to `sft/phrase_library/train/archive/`). Meant to reproduce iter-3's SFT base cleanly.
- **DPO data**: 110 pairs = 80 iter-4 (`iter4_force_fire.jsonl` + `iter4_tool_output_usage.jsonl`) + 30 new iter-7 (`iter7_capability_decline.jsonl` 20 pairs, `iter7_current_x_hedge.jsonl` 10 pairs).
- **SFT trainer_state.json**: healthy. Loss 22.5 → 2.18, eval loss 0.458 → 0.194 monotonic, picker landed on checkpoint-25 at 86.67% on 30-case proxy. Early stopping at step 25 of 32.
- **DPO trainer_state.json**: MISSING. Notebook cell 23 didn't persist TRL's trainer state. So loss / reward-margin / reward-accuracy curves are unobservable. **Flagged for iter-8 notebook fix: add `trainer.save_state()` to the DPO cell.**

## What regressed and how badly

### 135-case eval (`sft/eval_results/iter7_135_2026-04-23.json`)

| Metric | Prod (iter-3 + hint) | Iter-7 | Δ |
|---|---|---|---|
| Routing pass | 126/135 = 93.3% | 118/135 = 87.4% | **−5.9 pp** |
| Reply pass (regex) | 124/135 = 91.9% | 83/135 = 61.5% | **−30.4 pp** |
| Tool-output usage | 45/76 = 59.2% | 9/82 = 11.0% | **−48.2 pp** |

The tool-output usage crash is the smoking gun: the LoRA is firing tools almost as often as prod (82 attempts vs 76), but only 9 of 82 replies actually use the tool output. In prod 45 of 76 did.

### 40-case reply-quality eval (`sft/eval_results/iter7_replyq_2026-04-23.json`)

| Metric | Prod (iter-3 + hint, post-wins) | Iter-7 | Δ |
|---|---|---|---|
| Routing pass | 34/40 = 85.0% | 32/40 = 80.0% | −5.0 pp |
| Reply pass (regex) | 33/40 = 82.5% | 26/40 = 65.0% | **−17.5 pp** |
| Tool-output usage | ~50% band | 2/21 = 9.5% | − |

### Failure modes observed in the manual read

**1. Schema-leak (raw tool-call JSON in reply text).** Five+ cases emit the tool-call JSON as user-facing content instead of a natural-language reply. Examples:
- Case 8 (thermostat): `{"name": "set_thermostat", "arguments": {"temperature": 72}}}` — a **phantom tool** the model invented. Karin has no `set_thermostat` tool.
- Case 9: `{"name": "send_email", "arguments": {"to": "boss@example.com", "subject": "Will be Late", ...}}}`
- Case 10, 11, 14, 22, 25, 31, 33: similar — inventing `order`, `search`, `calculate_derivative`, etc.

Prior to iter-7, the Phase-0 runtime layer + `_recover_leaked_tool_call` scrub caught most of these. Iter-7 produces them at volume.

**2. Stock-fallback replacing compose output.** Replies like "Pretty chill day. You?" / "Same as always, just hanging out." / "Nothing wild — what's up?" appear after legitimate tool fires. These are `_pick_fallback` chitchat canned responses triggered when `_clean_reply` scrubs the model output. Cases 12, 15, 16, 17, 28, 29 all show this pattern. In prod, these same prompts produced grounded summaries.

**3. Phantom `update_memory` calls.** Cases 16, 20, 21, 22, 23, 28, 29, 30 all fire `update_memory` as a second tool after the primary tool's turn. None of my iter-7 DPO pairs involved `update_memory`. The LoRA seems to have learned a spurious "after any tool turn, save a fact" pattern. Saved facts are also frequently garbage: after a WWII wiki lookup, the saved fact is `"User asked about gold price."`.

**4. Malformed math args.** Case 19 (2+2) calls math with `expression="[0,1]+[0,1]"`. Case 29 (KL N(0,1)‖N(1,2)) passes `[0,1] * [0.3333, 0.6667]` — vector shorthand on a distribution-shorthand prompt. Case 31 passes `sin(x)` as a bare string. Math tool returns errors; compose step can't ground on error text.

### DPO target cases

- **Case 8 (thermostat)** — REGRESSED. Was "I've passed on your request" (soft fabrication). Now: `{"name": "set_thermostat", "arguments": {"temperature": 72}}}`. DPO somehow made the target bug worse by encouraging the model to invent a nonexistent tool instead of decline.
- **Case 26 (current PM)** — UNCHANGED. "That would be Rishi Sunak." Still stale. DPO's 10 hedging pairs didn't shift routing on this exact prompt.

Zero of two targets fixed. Both targets also came with broader regression.

## Root cause hypothesis

**DPO over-generalized the "reject" surface features.** The rejected examples for capability-decline contained confident action-confirmation strings:

- `"Alright! I've adjusted the temperature to 72 degrees."` (rejected)
- `"Email sent! Team has been notified."` (rejected)
- `"Ordered! Your large pepperoni from Domino's is on the way."` (rejected)

The DPO loss (contrastive on chosen vs rejected) pushed the model's distribution away from those strings. But those strings share surface features with legitimate tool-grounded replies:

- `"Gold's up to 3200 bucks."` (would be a PREFERRED reply after a tracker call in prod)
- `"It's 72 degrees and partly cloudy."` (weather reply)
- `"The KL divergence is 1.457 nats."` (math reply)

All of these start with confident action/state verbs that look like the *rejected* class to a DPO gradient operating on a small preference set. With only 110 pairs and a strong prior for confident tool-grounded replies, DPO appears to have learned "confident action language → lower reward" at a population level instead of "confident action language WITHOUT a tool fire → lower reward" specifically.

Downstream effects:
- Model prefers either raw tool-call JSON (doesn't look like the reject class — it's JSON, not prose) OR a vague stock fallback (avoids any "I did X" phrasing entirely).
- `update_memory` firing looks like a way to technically claim "I did something" without using the confident action vocabulary that got punished.

**Secondary factor**: the `adapter_dpo/trainer_state.json` is missing, so we can't verify reward margin / accuracy curves. It's plausible the training was degenerate (rewards collapsed, accuracy stuck near 50%). Without instrumentation we can't prove the loss was healthy.

## Why three targeted interventions failed in a row

This is the fourth consecutive training run (iter-4 DPO + RL, iter-5 SFT rebalance, iter-6 SFT rebalance, iter-7 DPO-only) to regress vs iter-3 + Phase-0. Pattern:

| Iteration | Approach | Outcome | Diagnosis |
|---|---|---|---|
| iter-4 | Full SFT + DPO with single-turn flatten | −17 pp reply | DPO single-turn flatten broke tool-output-usage |
| iter-5 | SFT + completion-only mask + 52 rows | 78.5% routing | Concentrated no-tool rows amplified under-fire |
| iter-6 | SFT rebalance with paired positives | 78.5% routing | Paired positives over-generalized to decoys |
| iter-7 | SFT (iter-3 data) + DPO on 110 pairs | 87.4% routing / 61.5% reply | DPO reject surface generalized past intended target |

**Common thread: every targeted training fix introduces a bigger secondary failure than the one it intended to address.** Small-data DPO / SFT on a LoRA over a 4-8 B abliterated base is an underdetermined problem — the gradient has too few constraints to shift one pattern without pulling ten adjacent ones.

Runtime layers (Phase-0 classifier patches, under-fire rescue, two-phase compose, hint-in-user-msg, `_clean_reply` scrubs, market-fabrication scrub) have shipped *13 successive wins* over the same period without regressions.

## Decision

**Keep iter-7 parked on Jetson as `karin-tuned-iter7:latest` for diagnostic only.** Prod remains `karin-tuned:latest` (iter-3) + Phase-0 + two-phase + hint-in-user-msg.

**Don't retry DPO with more pairs.** The pattern isn't data-volume-bounded — it's a surface-feature-collision problem that more pairs can't fix without also paring chosen examples more carefully. The engineering cost to do that rigorously exceeds the expected value of the two bugs it would address.

**Don't GRPO.** GRPO is a reasoning-task optimizer and Karin's bugs are distribution/coverage problems. Lever-mismatch.

**Keep the runtime-layer strategy.** Working answer for the two unfixed bugs:
- Case 8 (thermostat): already partially mitigated by the persona-decline rule shipped 2026-04-23. If it gets worse, add a classifier pattern that force-fires a canned decline for smart-home imperatives.
- Case 26 (stale PM): add a `\b(current|latest|today'?s)\s+(prime minister|president|king|queen|CEO)` routing pattern to `wiki` so it fires.

Neither needs training.

## Artifacts preserved

- `run_f7950a38/` — full Colab run dir (adapter, adapter_dpo, GGUF, configs, SFT trainer_state, picker result).
- `sft/eval_results/iter7_135_2026-04-23.json` + `.log` — raw 135-case eval output.
- `sft/eval_results/iter7_replyq_2026-04-23.json` + `.log` — raw 40-case eval output.
- `sft/eval_results/iter7_135_fails_annotated.txt` — 292 lines, all 17 failing cases with reply text + tool calls.
- `sft/eval_results/iter7_replyq_annotated.txt` — 201 lines, all 40 cases with reply text for manual comparison against prior runs.
- `sft/phrase_library/dpo_pairs/iter7_capability_decline.jsonl` + `iter7_current_x_hedge.jsonl` — the 30 new pairs. Kept for historical reference; **do not reuse in iter-8 without significant chosen-side rewrites that avoid the surface-feature collision**.
- `karin-tuned-iter7:latest` on Jetson Ollama store — diagnostic only.

## Lessons for any future iter-8

Only try training again when all three of these are true:

1. **The data budget is materially larger** (≥ 2–3× current ~450 SFT rows, ≥ 4× current 110 DPO pairs), not just "a few more targeted pairs." Small targeted nudges have now failed three times.
2. **Chosen-side surface features are audited to differ from rejected-side features only in the variable we want to change**, not in confidence / verb tense / action language in general. That means drafting preference pairs where the chosen sentence has the *same* sentence structure as the rejected, differing only at the capability word. DPO contrast is stronger there and spreads less.
3. **Notebook cell 23 persists DPO trainer_state.json** so we can see reward margins / accuracies without re-running.

Until all three are true, the highest-ROI path is more runtime layers (classifier patches, persona rules, reply scrubs) rather than weights.
