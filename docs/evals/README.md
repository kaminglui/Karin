# Eval artifacts

Raw per-case data from routing evals. Analysis + narrative lives in the parent `docs/` folder (mainly [eval_135_manual_review.md](../eval_135_manual_review.md), [routing_eval_comparison.md](../routing_eval_comparison.md), and [routing-pipeline.md](../routing-pipeline.md)). These files are the source data those analyses cite.

## Files

### 135-case evals (chronological)

| File | What it is | Score |
|---|---|---|
| `eval_135.json` | Iter-3 baseline on 135-case set (Ollama + classifier fixes, no rescue) | 96/135 = 71.1% routing |
| `eval_iter4.json` | Iter-4 LoRA on same set — rolled back | 100/135 = 74.1% routing, 27/56 = 48.2% tool-output |
| `eval_phase0_rescue.json` | iter-3 + Phase-0 classifier + rescue. Contains `reply_flags` + `reply_pass` from retroactive `--score-json` scan | 112/135 = 83.0% routing, 77/135 = 57.0% reply-pass (regex) |
| `eval_phase0_round2.json` | + Step 1 (alert-priority + compound picker + yaml fix). Contains `claude_*` fields from manual judge | 119/135 = 88.1% routing |
| `eval_phase0_round2_step2.json` | + Step 2 (continuation rescue). Contains `claude_*` fields | 125/135 = 92.6% routing, 36/135 = 26.7% reply-pass (manual) |
| `eval_phase0_round2_step2_twophase.json` | + two-phase compose ON. Contains `claude_*` fields | 125/135 = 92.6% routing, 62/135 = 45.9% reply-pass (manual) |

### Large narrative reference

| File | What it is |
|---|---|
| `iter3-reference.md` | Full iter-3 135-case table plus untruncated failing replies, split out of `../eval_135_manual_review.md` so the review remains readable. |

### 38-case cross-backend evals (iter-3 LoRA, varying serving backend)

| File | Backend | Score |
|---|---|---|
| `eval_ollama_baseline.json` + `.log` | Ollama, no classifier fixes | 35/38 = 92.1% |
| `eval_ollama_fixed.json` + `.log` | Ollama + classifier patches (schedule_reminder priority, in-YYYY wiki) | 36/38 = 94.7% |
| `eval_llamacpp_v25.log` | llama-server with custom jinja template | 29/38 = 76.3% (then blocked on Jetson memory) |
| `eval_mlc_hard.log` | MLC-LLM merged LoRA + hard coaching injection | 22/38 = 57.9% |
| `eval_mlc_soft.log` | MLC-LLM merged LoRA + softer coaching | 22/38 = 57.9% |

## Schema

Each `.json` file produced by `scripts/eval_routing.py` has:

```
{
  "model": "<ollama tag>",
  "base_url": "<ollama url>",
  "passed": int,                  # routing passes (traditional metric)
  "total": int,
  "pass_rate": float,
  "reply_passed": int,            # (added 2026-04-20) regex reply-pass count
  "reply_pass_rate": float,
  "claude_routing_pass": int,     # (added 2026-04-20) Claude manual-judge scores
  "claude_reply_pass": int,
  "cases": [
    {
      "prompt": str,
      "expected_tool": str | list | null,
      "expected_args": dict | null,
      "actual_tool": str | null,
      "actual_args": dict,
      "passed": bool,
      "reason": str,              # if failed
      "latency_s": float,
      "final_reply": str,
      "tools_called": [{"name", "args", "stub_result"}],
      "used_tool_output": bool,   # sentinel-number match heuristic
      "reply_flags": [str],       # (2026-04-20) heuristic flag list
      "reply_model_bugs": [str],  # subset that fails reply_pass (regex)
      "reply_pass": bool,         # routing_pass AND no model-bug flags
      "claude_routing": "pass" | "fail",    # manual judge
      "claude_reply": "pass" | "fail" | "na",
      "claude_reason": str
    }, ...
  ]
}
```

Older JSONs (`eval_135.json`, `eval_iter4.json`, `eval_ollama_baseline.json`, `eval_ollama_fixed.json`) pre-date the `reply_*` + `claude_*` fields — retroactively score them with `python scripts/eval_routing.py --score-json docs/evals/<file>.json` if the regex metric is needed.

## Reproduce

```bash
# Full 135-case eval (live, ~50 min on Jetson):
docker cp ~/Karin/scripts/eval_routing.py karin-web:/app/scripts/eval_routing.py
docker cp ~/karin_sft/eval_cases_novel.yaml karin-web:/app/scripts/eval_cases_novel.yaml
docker exec karin-web python3 /app/scripts/eval_routing.py \
    --cases /app/scripts/eval_cases_novel.yaml \
    --json /app/scripts/eval_<tag>.json

# With two-phase compose:
docker exec karin-web python3 /app/scripts/eval_routing.py \
    --cases /app/scripts/eval_cases_novel.yaml --two-phase \
    --json /app/scripts/eval_<tag>_twophase.json

# Retroactively score an existing JSON with the regex scanner:
python scripts/eval_routing.py --score-json docs/evals/<file>.json
```

Manual Claude judgment is applied per-session when an eval run needs an honest number — see the two-stage review rubric in `feedback_eval_read_replies.md` (memory).
