# SFT pipeline for Karin routing

LoRA fine-tunes `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` (HF equivalent of the Jetson's `mannix/llama3.1-8b-abliterated:tools-iq4_xs`) on routing-specific failure patterns.

**Status** (as of 2026-04-22):

| Configuration | Serving | 135-case routing | 135-case reply | Notes |
|---|---|---|---|---|
| v4 mannix (baseline, no LoRA) | Ollama | — | — | historical ceiling was 79.2% prompt-engineering-only |
| iter 1–2 (`run_d94ae3e5`, `run_47b09b19`) | Ollama | — | — | diagnostic only — regressions from training bugs |
| iter 3 (`run_0ac17bc7`) | Ollama | 96/135 = 71.1% | — | baseline LoRA; same weights still in prod |
| **iter 3 + Phase-0 + rescue + two-phase + hint-in-user-msg** | **Ollama** | **126/135 = 93.3%** | **124/135 = 91.9%** | **current prod (`karin-tuned:latest`)**, shipped 2026-04-22. See `docs/routing-pipeline.md`. |
| iter 4 (`run_52227b4c`) | Ollama | — | — | rolled back — DPO single-turn flatten broke tool-output-usage |
| iter 5 (`run_7db68f92`) | Ollama | 106/135 = 78.5% | — | **not shipped.** Concentrated no-tool rows + completion-only mask amplified under-fire. Diagnostic only (`karin-tuned-iter5:latest`). |
| iter 6 (`run_d6ae0966`) | Ollama | 106/135 = 78.5% | — | **not shipped.** Paired-positive rebalance over-generalized to decoys. Diagnostic only (`karin-tuned-iter6:latest`). |

**Three iterations on the ~450-row budget plateaued at 78.5%.** Iter-4/5/6 each traded one failure class for another without net progress. Stop iterating unless the data budget goes 2-3× or the training approach changes materially. See [`docs/history/iter5-postmortem-2026-04-22.md`](../docs/history/iter5-postmortem-2026-04-22.md) and [`docs/history/iter6-postmortem-2026-04-22.md`](../docs/history/iter6-postmortem-2026-04-22.md).

**Current prod LoRA = iter-3.** All runtime improvements since ship date (Phase-0 classifier patches, under-fire rescue, two-phase compose, hint-in-user-msg) are bridge-layer — the weights haven't changed since `run_0ac17bc7`.

### Training data layout

```
sft/phrase_library/
├── train/
│   ├── <tool>.jsonl              — base phrase library (per-tool)
│   ├── iter4_*.jsonl             — iter-4 additions (compound, historical wiki, multi-turn, anti-widget, gap-tools)
│   ├── iter5_*.jsonl             — iter-5 additions (structural, compound-honesty, persona-antiwidget, update_memory-args)
│   └── iter6_*.jsonl             — iter-6 additions (antiwidget pairs, short-imperative/schedule-query negatives, continuation positives)
├── test/                         — held-out splits for training-time eval (not the ship gate)
└── dpo_pairs/                    — DPO preference pairs per iteration
```

**Treat iter5_*.jsonl and iter6_*.jsonl as diagnostic history, not as starting material for iter-7.** Both sets cycled plateau failure modes and shouldn't be copied into the next training run without rebalancing against the specific over/under-fire patterns they introduced. The iter5/6 post-mortems enumerate the bad cases per file.

The **ship gate** for any future LoRA iteration is `sft/eval_cases_novel.yaml` run via `scripts/eval_routing.py --two-phase --hint-in-user-msg --json <out>.json --verbose`. Routing ≥ 93% AND reply-pass ≥ 90% to match current prod; anything less is a regression.

---

## Historical context (pre-plateau)

| Configuration | Serving | 38-case | 135-case | Notes |
|---|---|---|---|---|
| v4 mannix | Ollama | 33/38 = 86.8% | — | pre-LoRA baseline |
| iter 1 | Ollama | 23/38 = 60.5% | — | training on literal "None" strings |
| iter 2 | Ollama | 27/38 = 71.1% | — | flattened tool_calls, still overfit |
| **iter 3** | **Ollama** | **34/38 = 89.5%** | 96/135 = 71.1% | landed anti-overfit HPs — became prod baseline |
| iter 3 + classifier fixes | Ollama | 36/38 = 94.7% | ~104/135 ≈ 77% | Fix A (schedule_reminder priority) + Fix B (wiki historical patterns) |
| iter 4 | Ollama | — | — | _target ≥ 85%_ — regressed; rolled back |

The 38-case eval is a historical reference point; the 135-case eval (expanded 2026-04-20 with pattern-novel generalization probes + gap-tool coverage) is the one to watch going forward — see [docs/eval_135_manual_review.md](../docs/eval_135_manual_review.md) for the failure analysis that drove iter-4.

**Iter 3 beats baseline on Ollama.** Anti-overfit hyperparameters (`lora_r=8`, `sft_epochs=2`, `sft_lr=1e-4`, `lora_dropout=0.1`, `weight_decay=0.01`) turned the regression into a +2.7 pp lead.

**Counter-intuitive picker behavior:** internal proxy dropped (93%→63%) while real held-out eval shot up (71%→89.5%). The proxy uses the training eval split which favors memorization-prone models; regularized models score lower on the proxy precisely because they generalize better. **Don't trust the picker number** — run the 38-case novel eval.

Multi-turn verified on Ollama — 4/4 prompts emit correct tool_calls including across turns 2-4 with full history.

`eval_cases_novel.yaml` is the held-out routing eval (88 cases as of 2026-04-20, disjoint from training by `sft/scripts/assert_disjoint.py`). The earlier 216-case `scripts/eval_cases.yaml` has been removed: it had 82% verbatim overlap with the phrase-library training data, so the score on it was memorization-inflated and not comparable to tuned-model runs. Historical v1-v7 prompt-engineering baselines (which topped at 79.2%) were measured on that old set and are no longer apples-to-apples with the current numbers — see [docs/routing_eval_comparison.md](../docs/routing_eval_comparison.md) for the current cross-backend comparison on `eval_cases_novel.yaml`.

### llama.cpp migration: attempted, multi-turn regression, reverted (2026-04-19)

Status: **reverted to `backend: ollama`.** Infrastructure in place for a future retry; current production runs on Ollama.

Single-turn eval on llama.cpp was clean (31/38 = 81.6%, +10.5 pp over Ollama's 71.1% for the same LoRA) after fixing three bridge bugs:
- Four caller sites missing `backend=` param (panels_api, web/server.py, news/service.py, scripts/eval_routing.py).
- `_post_chat_llamacpp` missing `type`/`id` on llama-server's tool_calls.
- `_recover_leaked_tool_call` missing `type`/`id` on synthetic tool_calls.

But a **multi-turn regression surfaced during endpoint testing** that the single-turn eval couldn't see. On the same LoRA, same Karin config, same `history_pairs`:

| Turn | Ollama | llama.cpp |
|------|--------|-----------|
| 1 | ✅ tool_call | ✅ tool_call |
| 2 | ✅ tool_call | ❌ hallucinated natural text |
| 3 | ✅ tool_call | ❌ hallucinated natural text |

Root cause: Ollama's Modelfile `TEMPLATE` directive instructs the model how to render prior tool_calls in history; llama-server's `--jinja` uses the model's embedded chat template which doesn't have that coaching. After turn 1, the model answers from its own knowledge instead of re-calling tools. Eval_routing.py uses `commit_history=False` so it never hit this — real users on `/api/chat/stream` (which defaults to `commit_history=True`) would hit it every turn after the first.

**Production reverted to Ollama** until the chat-template handling is resolved. Two viable paths for a future llama.cpp retry:
1. Author a jinja `.chat-template` file matching mannix's Modelfile `TEMPLATE`, pass via `--chat-template-file`.
2. Add a `backend=="llamacpp"` branch in `_llm_visible_history()` that strips all prior assistant content (kills conversation continuity but restores tool routing).

Bridge+systemd scaffolding is kept in the tree for the retry. llama-server is stopped; `sudo systemctl disable karin-llama` recommended so it doesn't auto-start on reboot.

## Layout

```
sft/
├── phrase_library/
│   ├── train/                          positive examples (prompt → tool call), one JSONL per tool
│   └── dpo_pairs/                      contrastive pairs for hard negatives (decoys)
├── eval_cases_novel.yaml               38-case held-out test set (disjoint from train)
├── scripts/
│   ├── build_dataset.py                JSONL → HuggingFace dataset; flattens tool_calls; enforces disjointness
│   └── assert_disjoint.py              hard guard: fail the build if any train prompt duplicates a test prompt
├── runs/<hash>/                        gitignored; per-run artifacts from colab_sft.ipynb
│   ├── run_config.json                 settings snapshot + hash
│   ├── adapter/ adapter_dpo/           PEFT adapters
│   ├── checkpoint_picker.json          per-checkpoint routing scores
│   └── karin-lora.gguf                 ~150 MB deploy artifact
├── sft_dataset.tar                     packed arrow dataset for Colab upload
└── colab_sft.ipynb                     end-to-end training notebook (hash-based run identity, resume-aware)
```

## Data format

### Train (SFT) — JSONL, one example per line

```json
{"messages": [
  {"role": "system", "content": "{{SYSTEM}}"},
  {"role": "user", "content": "How do I fix a leaky faucet?"},
  {"role": "assistant", "content": null, "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "web_search", "arguments": "{\"query\": \"how to fix a leaky faucet\"}"}}
  ]}
]}
```

`{{SYSTEM}}` is a placeholder — `build_dataset.py` substitutes it with the character's real system prompt at build time.

**`build_dataset.py` flattens tool_calls into content** at build time so the assistant turn renders as plain JSON text:

```
"content": "{\"name\": \"web_search\", \"arguments\": {\"query\": \"how to fix a leaky faucet\"}}"
```

This is required because mlabonne's abliterated chat template silently drops `tool_calls` on assistant messages — before the flatten was added, training targets came out as the literal string `"None"`. The flattened format matches Karin's fallback parser (`bridge/llm.py` Shape B: name + arguments keys).

### No-tool cases — same file, just no `tool_calls`

```json
{"messages": [
  {"role": "system", "content": "{{SYSTEM}}"},
  {"role": "user", "content": "hi"},
  {"role": "assistant", "content": "Hey there."}
]}
```

### DPO pairs — JSONL with prompt / chosen / rejected

```json
{
  "prompt": [{"role": "user", "content": "Don't remind me, I already know."}],
  "chosen": [{"role": "assistant", "content": "Got it, no reminder."}],
  "rejected": [{"role": "assistant", "content": null, "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "schedule_reminder", "arguments": "{}"}}
  ]}]
}
```

`chosen` / `rejected` get the same flatten treatment as SFT messages.

### Test set — `sft/eval_cases_novel.yaml`

88 cases (as of 2026-04-20), **hard-disjoint** from `phrase_library/train/`. `assert_disjoint.py` (wired into `build_dataset.py`) fails the build if any new training example duplicates a test prompt. The older `scripts/eval_cases.yaml` (216 cases, 82% training overlap) was deleted because its score was memorization-inflated and not a valid generalization metric for tuned models.

## Target counts for the next iteration

- **train/**: ~800 examples across 16 tools + no_tool bucket
  - Decoy-focused additions (priority): figurative uses of tool-trigger keywords — "call me when you're free", "memory lane", "schedule conflicts", "set aside some time", "forget it", "convert an argument", "what time is…" (non-time-fetch), "miles" (non-conversion)
- **dpo_pairs/**: ~120-150 pairs — 3-4 per tool-keyword covering figurative uses
- **eval_cases_novel.yaml**: already at 88 cases after the v2 expansion (compound queries, historical facts, tool-arg extraction, richer multi-turn, harder decoys, no-tool-available edges). Keep growing with fresh paraphrases; never reuse anything in train/.

## Workflow

1. Seed `train/*.jsonl` per tool (10-20 per tool to start; scale to target).
2. Run `python sft/scripts/build_dataset.py --character karin --voice general --out sft/sft_dataset.arrow`.
   - Asserts disjointness → errors with `TRAIN/TEST LEAKAGE` if any train prompt duplicates a test prompt.
   - Flattens tool_calls → content JSON.
   - Packs to `sft/sft_dataset.arrow`. Bundle for Drive upload with `cd sft && tar -cf sft_dataset.tar sft_dataset.arrow` (run from the `sft/` dir so the archive's internal layout is flat — the notebook's cell 3 extracts to `/content/dataset/` and expects `sft_dataset.arrow/` at that root, NOT `sft/sft_dataset.arrow/`).
3. Upload `sft_dataset.tar` to `MyDrive/Karin_SFT/input/` (overwrites prior).
4. Open `colab_sft.ipynb` on Colab (A100 recommended). Run all.
   - Hash-based run folder → identical CONFIG reuses the same Drive folder; each cell's `STATE` guard skips already-done steps.
   - `FORCE_REWRITE=True` in cell 0b wipes the current run folder for a clean redo.
   - Defaults: SFT → picker (by routing accuracy, not eval_loss) → DPO → LoRA GGUF → `runtime.unassign`. Merged-GGUF path off by default; flip `CONFIG['run_merge_gguf']=True` if needed.
5. Copy `runs/<hash>/karin-lora.gguf` from Drive to the Jetson.
6. On the Jetson: pick a deploy option in `colab_sft.ipynb` section 9 (Ollama `ADAPTER`, Ollama merged, or **llama.cpp recommended**).
7. Run `scripts/eval_routing.py --cases sft/eval_cases_novel.yaml` — compare to v4 mannix's 86.8%.

## Prompt-engineering baseline runs (pre-SFT, historical)

The v1-v7 runs were measured on the now-removed 216-case `scripts/eval_cases.yaml`. That set had 82% training-data overlap, so its scores were memorization-inflated and not directly comparable to tuned-model runs on the held-out set. The numbers are preserved here for provenance only:

| Run | Model + aid | Score on old 216-set |
|-----|-------------|----------------------|
| v1 | mannix, no aids | 70.4% |
| v2 | mannix + regex classifier | 75.5% |
| v3 | mannix + regex + spaCy veto | 79.2% |
| v4 | mannix + Phase-1 prompt hints | 79.2% (ceiling) |
| v5 | mannix + directive hints | 77.8% (regressed, reverted) |
| v6 | mannix + embedding fallback | 79.2% |
| v7a | hermes3:8b | VRAM spill, no data |
| v7b | qwen3-abliterated:4b | 18.5% (broken tool-use) |

On the apples-to-apples held-out set (`sft/eval_cases_novel.yaml`, pre-expansion 38 cases): v4 mannix scored 33/38 = 86.8%. Current tuned+cls-fixes setup (iter-3 LoRA + classifier patches) scored 36/38 = 94.7% on the same 38. See [docs/routing_eval_comparison.md](../docs/routing_eval_comparison.md) for the cross-backend comparison table.
