"""LLM-as-judge eval for reply quality.

Complements eval_routing.py (which grades tool SELECTION) by grading
reply QUALITY with a smaller judge model. Runs the held-out
sft/eval_cases_novel.yaml through the real LLM with stubbed tools,
then asks a judge model to score each reply on four dimensions.

Usage (on a host where Ollama is reachable):

    python scripts/eval_reply_quality.py
    python scripts/eval_reply_quality.py --judge-model qwen2.5:3b
    python scripts/eval_reply_quality.py --model llama3.1:8b
    python scripts/eval_reply_quality.py --json out.json
    python scripts/eval_reply_quality.py --verbose

Exit code: 0 if mean overall >= 3.0, else 1. Tune with --threshold.

Why a separate judge model: self-judging inflates scores. Using a
different model (qwen2.5:3b vs llama3.1:8b) gives a more honest read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bridge.llm import OllamaLLM  # noqa: E402
from bridge.tools import TOOL_SCHEMAS  # noqa: E402
from bridge.utils import load_config  # noqa: E402


# --- Stubbed tool execution (same as eval_routing.py) ---

def _stub_execute(name: str, args: dict) -> str:
    return f"(stub result for {name})"


def _monkey_patch_tools():
    """Replace bridge.tools.execute with the stub."""
    from bridge import tools as tools_mod
    tools_mod.execute = _stub_execute  # type: ignore[attr-defined]


# --- Judge prompt ---

JUDGE_SYSTEM = """\
You are a reply-quality judge. Score an AI assistant's response on four
dimensions, each 1-5. Output ONLY a JSON object, nothing else.

{"relevance": N, "conciseness": N, "persona": N, "grounding": N, "overall": N}

Rubric:
- relevance (1-5): Does the reply answer the user's question? 1=wrong topic, 5=directly answers.
- conciseness (1-5): Is it appropriately brief for a voice assistant? 1=wall of text or dump, 5=one crisp sentence.
- persona (1-5): Does it sound like a casual friend, not a formal bot? 1=robotic/corporate, 5=natural/casual.
- grounding (1-5): If a tool was called, is the reply based on tool output? 1=hallucinated facts, 5=clearly grounded. If no tool was expected, score 5 (N/A).
- overall (1-5): Holistic quality. Average of the above, rounded."""


def judge_reply(
    judge: OllamaLLM,
    prompt: str,
    reply: str,
    tool_used: str | None,
) -> dict:
    """Ask the judge model to score one (prompt, reply) pair."""
    user_msg = (
        f"User prompt: {prompt}\n"
        f"Tool called: {tool_used or '(none)'}\n"
        f"Assistant reply: {reply}\n\n"
        "Score this reply. Output JSON only."
    )
    judge.reset()
    raw = judge.chat(user_msg, commit_history=False)
    # Parse the JSON from the judge's reply. Be lenient — some models
    # wrap it in markdown fences or add commentary.
    raw = raw.strip()
    if raw.startswith("```"):
        # Strip markdown fences
        lines = raw.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        scores = json.loads(raw)
        if isinstance(scores, dict):
            return {
                "relevance": int(scores.get("relevance", 3)),
                "conciseness": int(scores.get("conciseness", 3)),
                "persona": int(scores.get("persona", 3)),
                "grounding": int(scores.get("grounding", 3)),
                "overall": int(scores.get("overall", 3)),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: couldn't parse judge output
    return {"relevance": 0, "conciseness": 0, "persona": 0, "grounding": 0,
            "overall": 0, "_raw": raw[:200]}


# --- Main ---

def load_cases(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw.get("cases") or []


def run_case(llm: OllamaLLM, prompt: str) -> tuple[str | None, str, float]:
    """Run a prompt through the LLM with stubbed tools.

    Returns (tool_name_or_none, reply_text, seconds).
    """
    llm.reset()
    captured_tool: list[str] = []

    def on_tool(name: str, args: dict, result: str) -> None:
        if not captured_tool:
            captured_tool.append(name)

    t0 = time.monotonic()
    try:
        reply = llm.chat(prompt, tools=TOOL_SCHEMAS, on_tool_call=on_tool,
                         commit_history=False)
    except Exception as e:
        reply = f"(error: {e})"
    dt = time.monotonic() - t0
    tool = captured_tool[0] if captured_tool else None
    return tool, reply.strip(), dt


# --- Claude API judge (external, higher-quality than local 8B models) ---

CLAUDE_JUDGE_SYSTEM = """\
You are a strict reply-quality judge for a local voice assistant called Karin. Your job is to read one eval case and return a JSON verdict.

You apply a two-stage rubric:

STAGE 1 — routing_pass (soft-grade)
  • Did the assistant call an acceptable tool for the user's prompt?
  • expected_tool may be a string OR a list (multiple routings are defensible).
  • Soft-grade judgment calls: if the prompt is plausibly two ways ("time zone in Tokyo" as get_time OR wiki; compound "weather and news" as either tool), accept any defensible reading.
  • If expected_tool is null/None, the prompt should produce NO tool call (chitchat / idiom / greeting).

STAGE 2 — reply_pass (only if routing_pass)
  • reply faithfully uses the tool output (no fabrication of numbers, commodities, cities, or topics not in the stub)
  • reply stays in character as Karin — NOT "Karaoke-chan" / vocaloid / "a large language model"
  • reply actually ADDRESSES the user's specific question (no topic drift, no meta-commentary like "You've asked me to..." / "This is a JSON object...")
  • reply doesn't leak internals: no NONCE_xxx tokens, no raw JSON schema, no widget phrases when no widget fired

Stub artifacts to discount (don't fail reply_pass over these):
  • "Trotsky" / "1842" when real wiki would be fine in prod
  • "Hitchhiker" / "42 is the answer" when real search would be fine in prod

Output ONLY a JSON object:
{"routing_pass": bool, "reply_pass": bool, "reason": "one-sentence explanation (≤25 words)"}"""


def _format_claude_case_prompt(case: dict) -> str:
    """Build the per-case user prompt for the Claude judge."""
    import json as _json
    tool_results = []
    for tc in case.get("tools_called") or []:
        tool_results.append(
            f"  • {tc.get('name')}({_json.dumps(tc.get('args') or {})}) "
            f"returned: {(tc.get('stub_result') or '')[:200]}"
        )
    tool_block = "\n".join(tool_results) or "(no tools called)"
    expected = case.get("expected_tool")
    return (
        f"User prompt: {case.get('prompt')!r}\n"
        f"Expected tool: {expected!r}\n"
        f"Actual tool called: {case.get('actual_tool')!r}\n"
        f"Actual args: {case.get('actual_args')!r}\n"
        f"Tool result(s):\n{tool_block}\n"
        f"Assistant's final reply:\n{case.get('final_reply') or '(empty)'}\n\n"
        f"Judge this case. Output JSON only."
    )


def judge_with_claude(
    eval_json_path: Path,
    claude_model: str,
    out_path: Path | None,
    max_cases: int | None = None,
) -> int:
    """Run every case in an existing eval JSON through Claude for
    routing + reply grading. Writes ``claude_api_*`` fields back onto
    each case and aggregates totals.

    Requires ANTHROPIC_API_KEY in the environment. Run BEFORE iter-5
    ships or any time an automated honest-judge score is needed — this
    is the deterministic, reproducible replacement for me hand-reading
    135 replies per session.

    Default model is ``claude-haiku-4-5-20251001`` (cheap, fast,
    plenty-good for this judging task). Override with ``--claude-model``
    for a stricter read from Sonnet.
    """
    import os
    import time
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY in the environment", file=sys.stderr)
        return 2
    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        return 2

    client = Anthropic(api_key=api_key)
    d = json.loads(eval_json_path.read_text(encoding="utf-8"))
    cases = d.get("cases") or []
    if max_cases:
        cases = cases[:max_cases]
    if not cases:
        print(f"No cases in {eval_json_path}", file=sys.stderr)
        return 2

    print(f"Judging {len(cases)} cases with {claude_model} (Claude API)")
    print("-" * 72)

    routing_pass = 0
    reply_pass = 0
    for i, c in enumerate(cases, 1):
        msg = _format_claude_case_prompt(c)
        try:
            resp = client.messages.create(
                model=claude_model,
                max_tokens=256,
                system=CLAUDE_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": msg}],
            )
            raw = resp.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                lines = raw.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines).strip()
            verdict = json.loads(raw)
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            print(f"  [{i:3d}] PARSE ERROR: {e}; raw={raw[:100] if 'raw' in dir() else '?'!r}")
            verdict = {"routing_pass": False, "reply_pass": False, "reason": f"parse error: {e}"}
        except Exception as e:
            print(f"  [{i:3d}] API ERROR: {e}; retrying in 5s...")
            time.sleep(5)
            try:
                resp = client.messages.create(
                    model=claude_model,
                    max_tokens=256,
                    system=CLAUDE_JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": msg}],
                )
                raw = resp.content[0].text.strip()
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    raw = "\n".join(lines).strip()
                verdict = json.loads(raw)
            except Exception as e2:
                print(f"  [{i:3d}] retry failed: {e2}")
                verdict = {"routing_pass": False, "reply_pass": False, "reason": f"api error: {e2}"}

        c["claude_api_routing_pass"] = bool(verdict.get("routing_pass"))
        c["claude_api_reply_pass"] = bool(verdict.get("reply_pass"))
        c["claude_api_reason"] = verdict.get("reason", "")
        if c["claude_api_routing_pass"]:
            routing_pass += 1
        if c["claude_api_reply_pass"]:
            reply_pass += 1
        mark = "✓" if c["claude_api_reply_pass"] else ("·" if c["claude_api_routing_pass"] else "✗")
        print(f"  [{i:3d}/{len(cases)}] {mark} {c.get('prompt','')[:50]!s}")

    total = len(cases)
    d["claude_api_routing_pass"] = routing_pass
    d["claude_api_reply_pass"] = reply_pass
    d["claude_api_routing_pass_rate"] = routing_pass / total if total else 0.0
    d["claude_api_reply_pass_rate"] = reply_pass / total if total else 0.0
    d["claude_api_model"] = claude_model

    print("-" * 72)
    print(f"routing_pass: {routing_pass}/{total} = {routing_pass/total*100:.1f}%")
    print(f"reply_pass  : {reply_pass}/{total} = {reply_pass/total*100:.1f}%")
    print(f"gap         : {(routing_pass-reply_pass)/total*100:+.1f} pp")

    dest = out_path or eval_json_path
    dest.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"Written to {dest}")
    return 0


def judge_existing(
    eval_json_path: Path,
    base_url: str,
    judge_model: str,
    out_path: Path | None,
) -> int:
    """Score every case in an existing eval JSON with an LLM judge.

    Skips the live LLM re-run — reuses ``prompt``, ``actual_tool``, and
    ``final_reply`` already present in the JSON (as produced by
    eval_routing.py). Writes ``judge_scores`` onto each case and
    recomputes ``judge_overall_pass_rate``.

    Use when:
      * you already ran eval_routing.py and want an honest reply-quality
        number without paying to run the LoRA again;
      * you want to retroactively grade historical runs.
    """
    d = json.loads(eval_json_path.read_text(encoding="utf-8"))
    cases = d.get("cases") or []
    if not cases:
        print(f"No cases in {eval_json_path}", file=sys.stderr)
        return 2

    judge = OllamaLLM(
        base_url=base_url,
        model=judge_model,
        system_prompt=JUDGE_SYSTEM,
        temperature=0.1,
        num_ctx=2048,
        options={"num_predict": 200},
    )

    print(f"Judging {len(cases)} cases from {eval_json_path.name} with {judge_model}")
    print("-" * 72)

    totals = {"relevance": 0, "conciseness": 0, "persona": 0,
              "grounding": 0, "overall": 0}
    scored = 0
    judge_pass = 0          # overall >= 3
    judge_pass_strict = 0   # overall >= 4
    for i, c in enumerate(cases, 1):
        prompt = c.get("prompt") or ""
        reply = c.get("final_reply") or ""
        tool = c.get("actual_tool")
        scores = judge_reply(judge, prompt, reply, tool)
        c["judge_scores"] = scores
        if scores.get("overall", 0) > 0:
            scored += 1
            for k in totals:
                totals[k] += scores.get(k, 0)
            if scores.get("overall", 0) >= 3:
                judge_pass += 1
            if scores.get("overall", 0) >= 4:
                judge_pass_strict += 1
        short = prompt[:50] if len(prompt) <= 50 else prompt[:47] + "..."
        print(f"[{i:3d}/{len(cases)}] overall={scores.get('overall','?')} {short}")

    means = {k: round(v / scored, 2) if scored else 0.0 for k, v in totals.items()}
    total = len(cases)
    d["judge_means"] = means
    d["judge_pass_lenient"] = judge_pass         # overall >= 3
    d["judge_pass_strict"] = judge_pass_strict   # overall >= 4
    d["judge_pass_lenient_rate"] = judge_pass / total if total else 0.0
    d["judge_pass_strict_rate"] = judge_pass_strict / total if total else 0.0

    print("-" * 72)
    print(f"Judge means: {means}")
    print(f"Lenient pass (overall>=3): {judge_pass}/{total} = {judge_pass/total*100:.1f}%")
    print(f"Strict  pass (overall>=4): {judge_pass_strict}/{total} = {judge_pass_strict/total*100:.1f}%")

    dest = out_path or eval_json_path
    dest.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"Written to {dest}")
    judge.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-judge reply quality eval")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "config" / "assistant.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the main LLM model")
    parser.add_argument("--judge-model", type=str, default="qwen2.5:3b",
                        help="Model for the judge (default: qwen2.5:3b)")
    parser.add_argument("--cases", type=Path,
                        default=REPO_ROOT / "sft" / "eval_cases_novel.yaml")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also dump results to this JSON file")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Min mean overall score for exit 0 (default: 3.0)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit to first N cases (for quick runs)")
    parser.add_argument("--judge-existing", type=Path, default=None,
                        help=("Skip live LLM re-run; score replies from an "
                              "existing eval_routing.py JSON file via the "
                              "local Ollama judge model. Writes judge_scores "
                              "back onto the file."))
    parser.add_argument("--judge-claude", type=Path, default=None,
                        help=("Use Claude API as judge on an existing eval "
                              "JSON. Gives honest, reproducible routing_pass "
                              "+ reply_pass grading per the two-stage rubric. "
                              "Requires ANTHROPIC_API_KEY. ~$0.02 per 135-case "
                              "run on Haiku. Writes claude_api_* fields."))
    parser.add_argument("--claude-model", type=str,
                        default="claude-haiku-4-5-20251001",
                        help="Claude model ID for --judge-claude (default: "
                             "haiku-4-5; try claude-sonnet-4-6 for strict "
                             "read at ~10x cost).")
    args = parser.parse_args()

    # Claude API judge — external, reproducible, replaces manual hand-reading.
    if args.judge_claude:
        return judge_with_claude(
            args.judge_claude, args.claude_model, args.json, args.max_cases,
        )

    # Retroactive judging path — no live re-run of the production LLM.
    if args.judge_existing:
        cfg = load_config(args.config)
        base_url = cfg["llm"]["base_url"]
        return judge_existing(
            args.judge_existing, base_url, args.judge_model, args.json,
        )

    _monkey_patch_tools()

    cfg = load_config(args.config)
    llm_cfg = cfg["llm"]
    model = args.model or llm_cfg["model"]
    base_url = llm_cfg["base_url"]

    cases = load_cases(args.cases)
    if args.max_cases:
        cases = cases[:args.max_cases]

    # Skip null-expected cases (chitchat) — we can't meaningfully judge
    # a tool-less casual reply against a rubric designed for tool turns.
    cases = [c for c in cases if c.get("expected_tool") is not None]

    print(f"Model: {model}")
    print(f"Judge: {args.judge_model}")
    print(f"Cases: {len(cases)} (tool-expected only)")
    print()

    # Build the main LLM
    llm = OllamaLLM(
        base_url=base_url,
        model=model,
        system_prompt=llm_cfg.get("system_prompt", ""),
        temperature=llm_cfg.get("temperature", 0.3),
        num_ctx=llm_cfg.get("num_ctx", 2048),
        options=llm_cfg.get("options", {}),
    )

    # Build the judge LLM
    judge = OllamaLLM(
        base_url=base_url,
        model=args.judge_model,
        system_prompt=JUDGE_SYSTEM,
        temperature=0.1,
        num_ctx=2048,
        options={"num_predict": 200},
    )

    results: list[dict] = []
    totals = {"relevance": 0, "conciseness": 0, "persona": 0,
              "grounding": 0, "overall": 0}
    scored = 0

    for i, case in enumerate(cases, 1):
        prompt = case["prompt"]
        expected = case.get("expected_tool")

        tool, reply, dt = run_case(llm, prompt)
        scores = judge_reply(judge, prompt, reply, tool)

        entry = {
            "prompt": prompt,
            "expected_tool": expected,
            "actual_tool": tool,
            "reply": reply[:200],
            "latency_s": round(dt, 2),
            **scores,
        }
        results.append(entry)

        if scores.get("overall", 0) > 0:
            for k in totals:
                totals[k] += scores.get(k, 0)
            scored += 1

        status = "OK" if scores.get("overall", 0) >= 3 else "LOW"
        if args.verbose or status == "LOW":
            print(f"  [{status}] {i}/{len(cases)} {prompt[:50]}")
            print(f"       tool={tool} reply={reply[:60]}...")
            print(f"       scores={scores}")
        else:
            print(f"  [{status}] {i}/{len(cases)} {prompt[:50]}: overall={scores.get('overall', '?')}")

    # --- Aggregate ---
    print(f"\n{'='*50}")
    if scored > 0:
        means = {k: round(v / scored, 2) for k, v in totals.items()}
        print(f"Mean scores ({scored} scored):")
        for k, v in means.items():
            print(f"  {k:14s}: {v}")
    else:
        means = {"overall": 0}
        print("No cases scored (judge model may be unreachable)")

    if args.json:
        args.json.write_text(
            json.dumps({"means": means, "results": results}, indent=2,
                       ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults written to {args.json}")

    llm.close()
    judge.close()

    passing = means.get("overall", 0) >= args.threshold
    print(f"\n{'PASS' if passing else 'FAIL'}: mean overall "
          f"{means.get('overall', 0)} {'≥' if passing else '<'} {args.threshold}")
    return 0 if passing else 1


if __name__ == "__main__":
    sys.exit(main())
