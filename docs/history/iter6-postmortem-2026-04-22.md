# Iter-6 post-mortem — 2026-04-22

> **Context note (added 2026-04-23).** This post-mortem compares iter-6 to the *then*-prod baseline of 91.1% routing. Prod has since shifted to **93.3% routing / 91.9% reply** via the `hint_in_user_msg` refactor (same iter-3 LoRA, different prompt structure). iter-6 remains not shipped — the 14.8 pp gap vs current prod is wider than the 12.6 pp noted below. See `docs/routing-pipeline.md` → "Hint-in-user-msg" and `CHANGES.md` 2026-04-22 entry.

## TL;DR

- **Iter-6 routing: 106/135 = 78.5 %.** Numerically identical to iter-5.
- **Prod (iter-3 + Phase-0 + rescue): 123/135 = 91.1 %.** Still 12.6 pp ahead.
- **Net iter-5 → iter-6 churn: +10 fixed, −10 broken.** Zero aggregate improvement.
- **Not shipped.** Prod unchanged; `karin-tuned-iter6:latest` parked on Jetson Ollama for diagnostic access only.
- The rebalance worked as a targeted fix (schedule_reminder + circuit under-fires disappeared) but the paired-positive rows over-generalized into new over-fires on decoy prompts.

---

## What iter-6 actually fixed (+10 vs iter-5)

Our iter-6 additions worked exactly as designed on the cases we trained them for:

| Case | How iter-6 fixed it |
|---|---|
| `Remind me to check the gold price tomorrow at 9am.` → schedule_reminder | `iter6_antiwidget_pairs.jsonl` paired positive |
| `Set an alarm for 7am and tell me the weather tomorrow.` → schedule_reminder | paired positive + compound training |
| `When I get home, remind me to feed the cat.` → schedule_reminder | paired positive |
| `Remind me after my 3pm meeting to send the report.` → schedule_reminder | paired positive |
| `If traffic's bad tomorrow, ping me earlier than 8am.` → schedule_reminder | paired positive |
| `What's the current through a 220 ohm resistor at 9 volts?` → circuit | mask + existing circuit rows |
| `Calculate the RC time constant for 1 microfarad and 10 kilohms.` → circuit | same |
| `Voltage divider output for 12V across 10k and 4.7k resistors.` → circuit | same |
| `Cancel that.` → None | `iter6_short_imperative_negatives.jsonl` explicit negative |
| `Actually make it 6pm instead.` → schedule_reminder (continuation) | `iter6_continuation_positives.jsonl` |

Eight of 10 fixes are schedule_reminder / circuit / continuation — exactly the patterns the rebalance targeted.

---

## What iter-6 broke (−10 vs iter-5)

Every new failure is an **over-fire on a decoy**. The paired positives taught the LoRA to fire tools on broader surface patterns than intended.

| Case | Wrong tool emitted | What happened |
|---|---|---|
| `Call me when you're free.` | update_memory | "call me" ≠ "save that about me" — over-generalized from update_memory rows |
| `The gold standard of debugging is print statements.` | graph | "gold" triggered; graph fired as a fallback |
| `How many miles is it from NYC to LA?` | convert | informational distance, not a unit conversion; convert over-applied |
| `Schedule conflicts are the worst.` | schedule_reminder | figurative "schedule"; new schedule_reminder rows over-generalized |
| `Have a good one.` | graph | pure chitchat; `graph` over-fire |
| `Cool, thanks.` | graph | pure chitchat; `graph` over-fire |
| `Make a note to self — don't panic.` | update_memory | figurative "note to self" |
| `The gold in your eyes is blinding.` | convert | figurative "gold"; convert over-applied |
| `Text my mom that I'll be late.` | schedule_reminder | can't actually text; should be no-tool + honest decline |
| `Scratch that, never mind.` | convert | we added `Scratch that.` as a negative, but `Scratch that, never mind.` didn't match |

The four repeating over-fire culprits are `update_memory`, `schedule_reminder`, `convert`, and `graph` — all tools we added paired positives for in iter-6. The paired rows taught the LoRA the wrong generalization: "surface pattern X → tool Y" applied too broadly.

---

## The mechanism

iter-6 was an A/B against iter-5 holding everything constant except **+29 new SFT rows** across four files:

1. `iter6_antiwidget_pairs.jsonl` (10 rows) — tool-fire positives for `Remind me to …` / `Save that …` / `Find a …`
2. `iter6_short_imperative_negatives.jsonl` (8 rows) — no-tool replies to `Never mind.`/`Scratch that.`/…
3. `iter6_schedule_query_negatives.jsonl` (5 rows) — no-tool to `Walk me through my schedule today.`
4. `iter6_continuation_positives.jsonl` (6 rows) — multi-turn continuations

The completion-only mask means each row's assistant tokens carry disproportionate weight. 10 antiwidget_pairs rows teaching "Remind me → schedule_reminder" were enough to bias the model toward firing schedule_reminder on ANY prompt containing "schedule" — including the decoy `Schedule conflicts are the worst.`.

The short_imperative_negatives helped (`Cancel that.` PASSED), but only on the **exact surface forms** we added. `Scratch that, never mind.` extended `Scratch that.` and didn't match. Training a concept needs ~3-5 variants to generalize reliably.

Same lesson as iter-5: **SFT pushes distribution weight in direct proportion to the row count**, and the completion-only mask amplifies that further. Small concentrations of any single pattern reshape the routing prior.

---

## Why 91.1 % looks like the SFT ceiling

Iter-3 without Phase-0 was ~89 % on this eval. Iter-4 (rolled back) was 74 %. Iter-5 and iter-6 both landed at 78.5 %. Three SFT iterations against the same ~450-row budget cycled failure modes:

- iter-4: DPO single-turn flatten broke tool-output-usage → regressed 74.1 %
- iter-5: concentrated no-tool rows broke under-fire discrimination → 78.5 %
- iter-6: concentrated paired positives broke over-fire discrimination → 78.5 %

The bare-LoRA numbers are bounded by the ambiguity in the eval cases. Most of the failing prompts are genuinely ambiguous ("Text my mom that I'll be late." — no-tool honest reply? schedule_reminder? update_memory? All three are defensible). A 450-row training set can't teach enough disambiguation to beat ~80 %.

**Phase-0 + rescue gets prod to 91.1 %** because it's exactly the right layer for ambiguity handling:
- Classifier regex rules encode hard priorities (alert > weather, compound-A-and-B parsing).
- Under-fire rescue force-fires specific tools when the LoRA abstains on clear tool-call prompts.
- Continuation rescue looks up prior tool and re-fires on pronoun-led followups.

These operate on the *type* of ambiguity the LoRA systematically mishandles, which is orthogonal to what SFT can teach with ~450 rows of training data.

---

## What to do next

**Stop SFT iteration on this rig.** Three passes (iter-4, iter-5, iter-6) cycling failure modes is the signal. The 91.1 % Phase-0 stack is the practical ceiling without a fundamentally different training approach.

**Orthogonal wins that still matter:**

1. **Tool-side robustness** — the real production UX improvements.
   - *Graph*: fixed `^` → `**` parse (sympy `convert_xor`); added preset dictionary for `gaussian`/`sigmoid`/`tanh`/`relu`/`softmax`/`exponential`/`laplace`/`cauchy`/`logistic`/`uniform`/`sine`/`cosine`/`parabola`/`cubic` with canonical formulas and sensible x-ranges. LoRA no longer has to invent the math for standard distributions.
   - *Math*: added `dot`, `norm` (L1/L2/L∞), `softmax`, `sigmoid`/`tanh`/`relu` element-wise, `mean`/`variance`/`std`, `mse`/`mae`/`cross_entropy`. Covers the basic ML-math surface.
   - *Graph widget*: scroll-zoom, drag-pan, double-click-reset, max/min markers with annotations, crosshair spike-lines. Plotly features that were disabled are now on.
2. **Runtime guards** (already shipped previously) — `_TRIVIAL_GREETING` bypass + `_PROMPT_LEAK_MARKERS` scrub keep prod leak-free.
3. **Data-balance audit** (`sft/scripts/audit_dataset_balance.py`) — flags concentrated no-tool files before training.

**What iter-7 would need if someone did try again:**

- 2-3× more training data (~1000+ rows) so small concentrations don't dominate.
- Either keep the mask OR go back to full-sequence loss with heavier weight decay — hard to tell which.
- DPO multi-turn rebuild (still deferred — 80 existing pairs are single-turn).
- New classes of ambiguity-aware data: paraphrase triples, contrastive pairs, figurative-use negatives.
- Mandatory: dataset-balance lint + 135-case eval gate.

Today: none of that. Prod stays on iter-3 + Phase-0. Tool improvements are shipped.

---

## Artifacts preserved

- `docs/evals/eval_iter6.json` — raw 135-case output (106/135 passed).
- `sft/runs/run_d6ae0966/karin-lora.gguf` — iter-6 adapter.
- `karin-tuned-iter6:latest` on Jetson Ollama — not wired into prod.
- `sft/sft_dataset.arrow.iter5` + `.iter4` + current iter-6 build — all three prior datasets preserved.
- `sft/phrase_library/dpo_pairs/iter5_format_template.jsonl.bak` — kept for future DPO rebuild.
- `sft/colab_sft.ipynb` — still reflects iter-6 config; iter-5 warning block in cell 5 is still relevant (same lesson applies).

Notebook-level notes for any iter-7 attempt are in `docs/history/lora-training-audit-2026-04-21.md`; the iter-5 warning block in cell 5 remains the correct guidance.

---

## Attribution

- Iter-6 trained: 2026-04-22 (Colab, `run_d6ae0966`)
- Iter-6 evaluated: 2026-04-22 on the 135-case held-out eval (`eval_cases_novel.yaml`)
- Iter-6 data authors: automated rebalance from iter-5 post-mortem lessons
- Base model: `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`
- TRL version: 0.15.2, `DataCollatorForCompletionOnlyLM` both-templates mode
