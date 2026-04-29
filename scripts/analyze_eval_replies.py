"""Pretty-print eval JSON for manual reply-quality review.

Usage: python scripts/analyze_eval_replies.py <path.json> [--only-fails]
"""
from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def trunc(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "..."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--only-fails", action="store_true")
    ap.add_argument("--show-flags", action="store_true",
                    help="Also surface passing-but-flagged cases")
    args = ap.parse_args()

    d = json.loads(Path(args.path).read_text(encoding="utf-8"))
    cases = d["cases"]
    print(f"=== {d['model']} ===")
    print(f"Routing pass: {d['passed']}/{d['total']} = {d['pass_rate']:.1%}")
    print(f"Reply pass:   {d['reply_passed']}/{d['total']} = {d['reply_pass_rate']:.1%}")
    print("-" * 80)

    for i, c in enumerate(cases, 1):
        passed = c.get("passed", False)
        reply_pass = c.get("reply_pass", False)
        flags = c.get("reply_flags", []) or []
        used = c.get("used_tool_output", None)
        show = True
        if args.only_fails:
            show = (not passed) or (not reply_pass) or flags
        if not show:
            continue
        mark = "PASS" if passed else "FAIL"
        rp = "rP" if reply_pass else "rF"
        tu = "T+" if used is True else ("T-" if used is False else "T?")
        fl = f" flags={flags}" if flags else ""
        exp = c.get("expected_tool")
        got = c.get("actual_tool")
        print(f"[{i:3d}] {mark} {rp} {tu}  {c['prompt'][:70]}")
        print(f"      exp={exp}  got={got}{fl}")
        if c.get("reason"):
            print(f"      reason: {c['reason']}")
        reply = c.get("final_reply") or ""
        print(f"      reply:  {trunc(reply)}")
        tc = c.get("tools_called") or []
        for t in tc[:2]:
            n = t.get("name", "?")
            st = (t.get("stub_result") or "")[:140].replace("\n", " ")
            print(f"      tool:   {n} -> {st}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
