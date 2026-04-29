# LoRA training method audit — 2026-04-21

## What triggered this audit

In a live chat session on 2026-04-21, user input `Hello` produced an
assistant reply that was a verbatim fragment of an older version of
the system prompt:

> `"how are you"). Just reply like you'd text a friend. No emoji, no
> markdown, one casual short line. The system background-pings latest
> news / alerts / trackers as part of refreshing memory, these calls
> are silent, do not show up in the turn record. When you do`

The fragment is not present in the current `config/characters/karin.yaml`
or anywhere in the repo — it only exists as reply text in the stored
conversation transcript. The LoRA (iter-3, `karin-tuned:latest`) had
memorised a paraphrase of an earlier prompt draft during SFT and was
regurgitating it when user input was too short to anchor on.

This doc records (a) how training currently works, (b) the root cause
of the leak, (c) the changes being made before iter-5, and (d) the
verification plan.

---

## 1. How the current method works

### 1.1 Training pipeline (iter-3 / iter-4)

Notebook: `sft/colab_sft.ipynb`, run on Colab against Drive-mounted
storage. The relevant stages:

| Cell | Stage | Summary |
|------|-------|---------|
| 3 | Env + Drive | Creates `Karin_SFT/{input,runs,hf_cache}/` on Drive, persists HF cache so the 16 GB base model isn't redownloaded each session. |
| 9 | Pip pins | `transformers==4.47.0`, `peft==0.13.2`, `trl>=0.13,<0.16`, `bitsandbytes>=0.46`, `datasets>=3.6,<5`, `accelerate==1.1.1`. |
| 15 | Base-model load | `AutoModelForCausalLM.from_pretrained(BASE_MODEL)` where `BASE_MODEL = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"`. 4-bit NF4 bnb, bf16 compute. `tokenizer.pad_token = tokenizer.eos_token`. |
| 17 | PEFT wrap | `prepare_model_for_kbit_training` + `LoraConfig(r=8, alpha=32, dropout=0.1, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], task_type='CAUSAL_LM')`. |
| 19 | **SFT training** | `SFTTrainer` with `SFTConfig(max_seq_length=3072, packing=False, num_train_epochs=2, lr=1e-4, weight_decay=0.01, bf16=True)`. Early-stopping path splits 10 % off train, `EVAL_STEPS=5`, `EarlyStoppingCallback(patience=3)`, `metric_for_best_model='eval_loss'`. |
| 21 | Checkpoint picker | Ranks saved checkpoints by tool-routing accuracy on a held-out set (not `eval_loss`, since loss-best and routing-best don't coincide). |
| 23 | **DPO training** | `DPOTrainer` with `DPOConfig(lr=5e-6, beta=0.1, num_train_epochs=1, max_length=4096, max_prompt_length=3072)`. |
| 25 | GRPO (optional) | Alt to DPO, reward = tool-name regex match. Currently off in CONFIG. |

### 1.2 Dataset format (current `sft/sft_dataset.arrow`)

Verified via `load_from_disk` inspection:

**`train_sft` — 374 rows**

- Column: `messages: list[{role, content, ...}]`
- Role distribution: 374 system / 406 user / 438 assistant / 32 tool
- All rows well-formed (no missing roles, no malformed entries)
- System prompt begins with `/no_think` then `═══ RULE ZERO — NO TOOL CALLS FOR CHITCHAT ═══` and runs ~2 KB; the same system text is duplicated across every row
- Assistant turns that represent tool calls store the call as inline
  JSON in `content` (e.g. `{"name": "circuit", "arguments": {...}}`),
  not as structured `tool_calls`

**`train_dpo` — 80 rows**

- Columns: `prompt`, `chosen`, `rejected` — each a list of messages
- All 80 rows have `prompt_len = 2` (system + user) and
  `has_assistant_in_prompt = False`
- **100 % single-turn flattened** — matches the iter-4 post-mortem
  note that DPO was flattened to single-turn, destroying multi-turn
  tool-output-usage signal

### 1.3 Base-model tokenizer

`mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` ships the standard
Llama 3.1 chat template with `<|start_header_id|>role<|end_header_id|>
\n\n…<|eot_id|>` delimiters. TRL's `SFTTrainer` uses this template to
render chat-format rows into token sequences. Assistant role markers
are unambiguous — which matters for the mask-loss fix below.

---

## 2. Root cause of the Hello → prompt leak

### 2.1 What the trainer was actually doing

`SFTConfig` was instantiated **without** `assistant_only_loss=True`.
In TRL's `SFTTrainer`, when that flag is `False` (the default), the
cross-entropy loss is computed across **every token** in the chat-
rendered sequence — including the system prompt and user turns. Every
system-prompt token contributes gradient signal to the LoRA.

Concretely, for a row like

```
<|start_header_id|>system<|end_header_id|>
/no_think
═══ RULE ZERO — NO TOOL CALLS FOR CHITCHAT ═══
…(~2000 more system tokens)…
<|eot_id|><|start_header_id|>user<|end_header_id|>
Hello
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Hey, what's up?<|eot_id|>
```

the LoRA receives loss on:

- ~2000 system-prompt tokens (identical across all 374 rows) — strong
  memorisation pressure
- ~3 user tokens
- ~5 assistant tokens

The system-prompt tokens dominate the loss by two orders of magnitude,
and since they appear verbatim in all 374 rows, the LoRA memorises them
effectively after a few epochs. At inference, when user input is short
and provides weak steering (e.g. `Hello`), the most-likely continuation
from the model's perspective is … the system prompt itself. Regurgitation
follows.

### 2.2 Why this matches the observed leak

The leaked string uses phrasings (`background-pings`, `turn record`,
`one casual short line`) that don't appear in the current
`config/characters/karin.yaml`, but are recognisable as a paraphrase of
an older draft. Iter-3 was trained on a `sft_dataset.arrow` built
against that older system-prompt text. The current prod `karin.yaml`
was rewritten later, but the LoRA still has the old one baked in as a
high-probability completion path.

### 2.3 Why the runtime scrub works but isn't enough

The runtime `_PROMPT_LEAK_MARKERS` regex scrub added today
(`bridge/llm.py`) catches the leak post-hoc and substitutes a canned
fallback — fine as a safety net. But every time it fires is a wasted
LoRA call and a reply the LoRA was capable of generating correctly if
training had masked properly. Root cause must be fixed at training time.

---

## 3. Changes applied for iter-5

### 3.1 Primary fix — `assistant_only_loss=True`

Add one kwarg to `common_kwargs` in `sft/colab_sft.ipynb` cell 19:

```python
common_kwargs = dict(
    …
    packing=False,
    assistant_only_loss=True,   # NEW 2026-04-21
    report_to='none',
)
```

**What this does.** TRL builds a loss mask that zeros out every token
outside assistant turns. Loss is computed only on tokens between
`<|start_header_id|>assistant<|end_header_id|>` and the matching
`<|eot_id|>`. System, user, and tool tokens contribute zero gradient
to the LoRA.

**Why it's safe.**

- Requires TRL ≥ 0.13 — already pinned in cell 9 (`trl>=0.13,<0.16`)
- Requires chat-format dataset with role-tagged `messages` — confirmed
  in §1.2
- Requires the chat template to use consistent role headers — Llama
  3.1's template qualifies (§1.3)
- `eval_loss` still works (only the reduction changes) so early
  stopping keeps functioning

**Expected consequences.**

- Loss magnitude will be **higher** in absolute terms (averaged over
  fewer tokens). Early-stopping patience and `metric_for_best_model`
  don't care about absolute scale, so no retuning needed
- System-prompt memorisation drops to near zero
- Tool-routing signal strengthens in relative terms (assistant turns
  are where the tool-call JSON lives)
- Training will be marginally faster — the backward pass over masked
  tokens is still computed but contributes nothing to the gradient

### 3.2 Secondary fix — DPO multi-turn format (iter-5 dataset rebuild)

The existing DPO set is 100 % single-turn flattened, which the iter-4
post-mortem identified as the cause of the tool-output-usage regression
(56.9 % → 48.2 %). Iter-5's dataset-build step must produce DPO rows
with a multi-turn `prompt` (system + user + assistant-with-tool-call +
tool-output) and `chosen` / `rejected` being alternative final-assistant
continuations. TRL `DPOTrainer` accepts this shape via
`continue_final_message=True`.

This is a dataset-generation change (`sft/scripts/build_dataset.py`),
not a notebook change. The Colab smoke-test script
`sft/scripts/smoke_test_dpo_format.py` already exists to validate
the resulting format against TRL before the full training run.

### 3.3 Runtime guards remaining in place

- `_TRIVIAL_GREETING` in `bridge/llm.py` — short-greeting bypass that
  skips the LoRA entirely for inputs like `hi` / `hello` / `hey`.
  Zero-latency, leak-proof by construction.
- `_PROMPT_LEAK_MARKERS` in `bridge/llm.py` — post-reply regex scrub
  on `═══`, `background-pings`, `turn record`, etc. Catches any
  residual leak from longer inputs.

After iter-5 with `assistant_only_loss=True` ships, both guards should
become vestigial. Keep them on — defense in depth, near-zero cost.

---

## 4. Verification plan

### 4.1 Automated checks wired into the Run-All flow

Three things now print or enforce themselves during a fresh Run-All
so silent failures can't sneak past:

- **Config banner at the top of cell 19** prints the critical settings
  (base model, dataset version, epochs, max_seq_length, LoRA
  r/α/dropout, `assistant_only_loss=True`, `packing=False`, `seed=42`,
  dataset system-prompt length + first 48 chars) **before** the
  trainer is built. User eyeballs it once and confirms nothing drifted.
- **Pre-train label-mask probe (still in cell 19, right before
  `trainer.train()`)** pulls a single batch from the real train
  dataloader, counts `(labels != -100)` positions, and:
  - prints the unmasked-token ratio (expected 5-15 %)
  - raises `RuntimeError` if > 40 % are unmasked (flag silently not
    applied → training would memorise the prompt again → abort before
    burning GPU hours)
  - decodes the first 80 unmasked tokens of row 0 so the user can
    visually confirm the model will be learning on *assistant* text,
    not system text
- **`seed=42`** on both `SFTConfig` and `DPOConfig` — reproducible
  dataset shuffle, dropout, early-stopping split across Colab sessions.

### 4.2 Manual post-train checks

After the full run:

1. **Training curve** — train_loss starts higher in absolute terms
   than iter-3/4 (smaller unmasked denominator) but decreases cleanly.
   If it's flat or explodes, abort.
2. **Post-train leak probe** — feed the trained LoRA `Hello` and
   `Hi` on 100 different random seeds. Expected: zero outputs
   matching `_PROMPT_LEAK_MARKERS`. Iter-3 under the same test emits
   the `═══` marker at a measurable rate — this delta is the headline
   proof the mask worked.
3. **Held-out routing eval** — the 135-case
   `sft/eval_cases_novel.yaml` suite. Routing pass-rate should match
   or exceed iter-3's 91.1 %; reply-pass should improve thanks to the
   removed system-prompt memorisation and the new iter-5 SFT rows.

Rollback remains trivial: revert `assistant_only_loss=True` and
re-run. No dataset rebuild required because the fix is loss-side only.

---

## 5. Other pre-run observations (and what we are/aren't doing)

Issues reviewed against the current dataset + notebook before
greenlighting iter-5:

| Observation | Status | Action |
|---|---|---|
| `assistant_only_loss` missing → system-prompt memorisation | **fixed** | added to cell 19 common_kwargs |
| `packing=False` — required alongside assistant_only_loss | already correct | banner asserts on it |
| DPO dataset is 100 % single-turn flattened (80/80 rows) | **deferred** — dataset-side, not notebook | iter-5 build_dataset.py change per iter-5 plan |
| System prompt duplicated ×374 in SFT rows; drift risk vs live karin.yaml | partial | banner prints dataset prompt's length + first 48 chars so a drift from current `config/characters/karin.yaml` is eyeball-visible at train time |
| Non-deterministic runs across Colab sessions | **fixed** | `seed=42` on both `SFTConfig` and `DPOConfig` |
| No runtime verification that loss-mask is behaving as expected | **fixed** | label-mask probe (raises on >40 % unmasked) |
| `/no_think` prefix wasted on Llama 3.1 (qwen legacy) | deferred | ~4 tokens per row; no correctness impact |
| LoRA targets all 7 projections (q/k/v/o/gate/up/down) — broad memorisation surface | acceptable with mask fix | leave as-is; revisit if post-run leak probe still trips |
| Early stopping at `EVAL_STEPS=5`, `patience=3`, ~46 total steps | acceptable for this dataset size | comments already note bumping `patience=5` once library >500 rows |
| Base-model mismatch training (mlabonne) vs inference (mannix on Jetson) | known — `feedback_lora_base_mismatch.md` | not addressed here; smoke-test step before deploy catches it |
| `prediction_loss_only=True` + `eval_accumulation_steps=1` | still correct | unchanged — these are memory-management settings orthogonal to masking |
| Checkpoint picker (cell 21) ranks by routing accuracy not eval_loss | good | unchanged |
| DPO `continue_final_message=True` multi-turn format | **deferred** — dataset-side | iter-5 `build_dataset.py` + `smoke_test_dpo_format.py` already staged for this |

**Non-goals for this audit:** anything that would slow the iter-5
iteration loop. Specifically we are NOT:

- Reworking the classifier / rescue runtime layer (already shipped,
  working)
- Changing the LoRA rank/α (iter-3's r=8/α=32 beat baseline)
- Touching the 135-case eval set (held out, stable)
- Rebuilding the SFT dataset from scratch (only the DPO half needs
  the multi-turn rebuild)

---

## 6. Open items not addressed here

- **`/no_think` prefix in every system prompt.** Legacy from the
  mannix/qwen thinking-control scheme; wastes ~4 tokens per row on a
  Llama 3.1 tokenizer. Removing it is a separate dataset-generation
  change worth doing but not blocking iter-5.
- **Persona-fine-tuning rows.** The current SFT set is routing-heavy.
  Iter-5's planned 52 new examples (structural, compound-honesty,
  update-memory args, persona-antiwidget) will partially address this;
  further persona coverage remains an open improvement.
- **DPO β tuning.** Current β=0.1. With the new multi-turn format, β
  may want revisiting; out of scope for this audit.

---

## 7. Attribution

- Audit performed: 2026-04-21
- Leak first observed: user session earlier same day
- Training method reviewed: `sft/colab_sft.ipynb` + `sft/sft_dataset.arrow`
  + `sft/runs/run_52227b4c/run_config.json`
- Base model: `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`
- TRL version pinned: `>=0.13,<0.16`
