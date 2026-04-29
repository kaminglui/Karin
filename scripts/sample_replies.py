"""Capture a handful of representative LLM replies across categories.

Unlike ``eval_routing.py`` this runs the REAL tool dispatcher (no stubs)
so the captured replies reflect what a user would actually see. Use to
sanity-check persona, verbosity, and tool-synthesis quality after a
prompt or classifier change.

Each category is one prompt; we print the user prompt, the tools
picked, and the final reply. Output is also dumped to ``--json`` if
given, so deltas between runs can be diffed.

Usage (inside the karin-web container):

    python scripts/sample_replies.py
    python scripts/sample_replies.py --json /tmp/replies.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bridge.llm import OllamaLLM  # noqa: E402
from bridge.tools import TOOL_SCHEMAS  # noqa: E402
from bridge.utils import load_config  # noqa: E402


# Pick one prompt per category. Add a new line here whenever a category
# regresses so repeat runs catch it.
PROMPTS: list[tuple[str, str]] = [
    ("chitchat", "Hi Karin!"),
    ("chitchat", "How are you today?"),
    ("identity", "What's your name?"),
    ("classifier-hit:get_weather", "What's the weather in Tokyo?"),
    ("classifier-hit:get_news", "Any news today?"),
    ("classifier-hit:tracker", "What's the price of gold?"),
    ("classifier-hit:wiki", "Who was Alan Turing?"),
    ("classifier-hit:convert", "5 miles to km."),
    ("classifier-abstain", "How cold is it outside right now?"),
    ("basic-arithmetic", "What's 15 times 23?"),
    ("multi-tool", "What's the weather in Tokyo and any news today?"),
    ("ambiguity", "Thanks."),
]


def run_one(llm: OllamaLLM, prompt: str) -> dict:
    llm.reset()
    picks: list[dict] = []

    def on_tool_call(name: str, args: dict, result: str) -> None:
        picks.append({"name": name, "args": args, "result_preview": result[:140]})

    t0 = time.monotonic()
    try:
        reply = llm.chat(prompt, tools=TOOL_SCHEMAS, on_tool_call=on_tool_call, commit_history=False)
    except Exception as e:
        reply = f"<chat raised: {e}>"
    dt = time.monotonic() - t0
    return {"prompt": prompt, "picks": picks, "reply": reply, "latency_s": round(dt, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "assistant.yaml"))
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    llm_cfg = cfg["llm"]
    llm = OllamaLLM(
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        system_prompt=llm_cfg.get("system_prompt", ""),
        temperature=llm_cfg.get("temperature", 0.4),
        num_ctx=llm_cfg.get("num_ctx", 4096),
        options=llm_cfg.get("options", {}),
        request_timeout=float(llm_cfg.get("request_timeout", 180.0)),
    )

    results = []
    try:
        for category, prompt in PROMPTS:
            res = run_one(llm, prompt)
            res["category"] = category
            tools_str = ", ".join(p["name"] for p in res["picks"]) or "<no tool>"
            print("=" * 72)
            print(f"[{category}] ({res['latency_s']}s)")
            print(f"  user:  {prompt}")
            print(f"  tools: {tools_str}")
            print(f"  reply: {res['reply']}")
            results.append(res)
    finally:
        llm.close()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nJSON written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
