"""Audit SFT dataset tool/no-tool balance before training.

Catches the iter-5 failure mode: concentrated no-tool rows that pass
every pre-train sanity check but wreck routing accuracy in production.
Iter-5 had 22 no-tool rows (5.2% of the set) clustered across two
files — not flagged by any metric, regressed 20 cases on the 135-case
eval.

What this does:
  1. Walks sft/phrase_library/train/*.jsonl
  2. For every assistant turn, classifies as tool-fire or no-tool
  3. Prints per-file + per-tool distributions, with % of total
  4. Raises WARNING lines on patterns known to cause regressions:
       - file contributes > 3% of assistant turns AND is > 80% no-tool
       - any single tool > 15% (over-concentration)
       - overall no-tool ratio outside 25-45% band

Run:
    python sft/scripts/audit_dataset_balance.py
    python sft/scripts/audit_dataset_balance.py --json out.json
    python sft/scripts/audit_dataset_balance.py --strict   # exit 1 on warnings

This is a dataset-level lint; run it BEFORE rebuilding the tar.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "sft" / "phrase_library" / "train"

# Matches the flattened tool-call shape used in this repo: the assistant's
# content is literal JSON like {"name": "circuit", "arguments": {...}}.
_TOOL_CALL_RE = re.compile(r'^\s*\{\s*"name"\s*:\s*"([a-z_][a-z0-9_]*)"')

# Thresholds derived from the iter-5 post-mortem. The actual iter-5
# regression was caused by iter5_persona_antiwidget.jsonl (10 turns /
# 541 = 1.85% of total) — below any reasonable "big concentration"
# threshold — so the per-file check has to flag quite small but fully-
# no-tool files. The overall-ratio band is a weaker signal (iter-5 at
# 22.4% still under-fired) but kept as a sanity check.
NO_TOOL_RATIO_MIN = 0.12   # below this: no-tool signal too weak → over-fire
NO_TOOL_RATIO_MAX = 0.35   # above this: no-tool dominates → under-fire
MAX_SINGLE_TOOL_SHARE = 0.15
NO_TOOL_FILE_SHARE_WARN = 0.015   # any file >1.5% of total assistant turns
NO_TOOL_FILE_RATIO_WARN = 0.80    # AND >80% of that file's rows are no-tool


def _classify(msg: dict) -> tuple[str, str]:
    """Return (category, tool_name). category is 'tool_fire' or 'no_tool'.

    Handles both raw library shape (structured ``tool_calls`` field) and
    the post-build flattened shape (tool call as JSON string in content).
    build_dataset.py flattens at build time, but for pre-build audits we
    need to recognise both forms.
    """
    tcs = msg.get("tool_calls") or []
    if tcs:
        first = tcs[0]
        fn = first.get("function") or {}
        name = fn.get("name") or first.get("name", "unknown")
        return ("tool_fire", name)
    content = (msg.get("content") or "").strip()
    m = _TOOL_CALL_RE.match(content)
    if m:
        return ("tool_fire", m.group(1))
    return ("no_tool", "")


def _scan_row(row: dict) -> list[tuple[str, str]]:
    """Return (category, tool_name) for every assistant turn in the row."""
    msgs = row.get("messages") or []
    return [_classify(m) for m in msgs if m.get("role") == "assistant"]


def audit(train_dir: Path) -> dict:
    files = sorted(train_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No jsonl files under {train_dir}")

    per_file: dict[str, dict] = {}
    total_tool = Counter()
    total_no_tool = 0
    total_assistant = 0
    total_rows = 0

    for f in files:
        file_tool = Counter()
        file_no_tool = 0
        file_assistant = 0
        file_rows = 0
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or "messages" not in row:
                    continue
                file_rows += 1
                for cat, name in _scan_row(row):
                    file_assistant += 1
                    if cat == "tool_fire":
                        file_tool[name] += 1
                    else:
                        file_no_tool += 1
        per_file[f.name] = {
            "rows": file_rows,
            "assistant_turns": file_assistant,
            "tool_fire": dict(file_tool),
            "no_tool": file_no_tool,
        }
        total_rows += file_rows
        total_assistant += file_assistant
        total_no_tool += file_no_tool
        total_tool.update(file_tool)

    no_tool_ratio = total_no_tool / max(total_assistant, 1)

    # Build warnings
    warnings: list[str] = []
    if no_tool_ratio > NO_TOOL_RATIO_MAX:
        warnings.append(
            f"overall no-tool ratio {no_tool_ratio:.1%} > {NO_TOOL_RATIO_MAX:.0%} "
            "— LoRA will under-fire in production (iter-5 regression pattern)"
        )
    if no_tool_ratio < NO_TOOL_RATIO_MIN:
        warnings.append(
            f"overall no-tool ratio {no_tool_ratio:.1%} < {NO_TOOL_RATIO_MIN:.0%} "
            "— LoRA will over-fire on chitchat"
        )
    for tool, count in total_tool.most_common():
        share = count / max(total_assistant, 1)
        if share > MAX_SINGLE_TOOL_SHARE:
            warnings.append(
                f"tool={tool} is {share:.1%} of all assistant turns "
                f"(> {MAX_SINGLE_TOOL_SHARE:.0%}) — over-concentration may "
                "over-generalize that tool on adjacent prompts"
            )
    for name, d in per_file.items():
        if d["assistant_turns"] == 0:
            continue
        share = d["assistant_turns"] / max(total_assistant, 1)
        file_no_tool_ratio = d["no_tool"] / d["assistant_turns"]
        if share > NO_TOOL_FILE_SHARE_WARN and file_no_tool_ratio > NO_TOOL_FILE_RATIO_WARN:
            warnings.append(
                f"{name}: {d['no_tool']}/{d['assistant_turns']} no-tool "
                f"({file_no_tool_ratio:.0%}) and {share:.1%} of total — "
                "iter-5 imbalance pattern, pair with tool-fire positives"
            )

    return {
        "total_rows": total_rows,
        "total_assistant_turns": total_assistant,
        "total_no_tool": total_no_tool,
        "no_tool_ratio": no_tool_ratio,
        "by_tool": dict(total_tool.most_common()),
        "by_file": per_file,
        "warnings": warnings,
    }


def _print_report(a: dict) -> None:
    print("=" * 68)
    print("SFT dataset balance audit")
    print("=" * 68)
    print(f"  rows scanned       : {a['total_rows']}")
    print(f"  assistant turns    : {a['total_assistant_turns']}")
    print(f"  no-tool turns      : {a['total_no_tool']} "
          f"({a['no_tool_ratio']:.1%})")
    print()
    print("Top tool distribution:")
    for tool, count in list(a["by_tool"].items())[:15]:
        share = count / max(a["total_assistant_turns"], 1)
        bar = "#" * int(share * 80)
        print(f"  {tool:22s} {count:4d}  {share:6.1%}  {bar}")
    print()
    print("Per-file breakdown (assistant turns, no-tool count, no-tool %):")
    sorted_files = sorted(
        a["by_file"].items(),
        key=lambda kv: kv[1]["assistant_turns"],
        reverse=True,
    )
    for name, d in sorted_files:
        if d["assistant_turns"] == 0:
            continue
        pct = d["no_tool"] / d["assistant_turns"]
        flag = " *" if pct > NO_TOOL_FILE_RATIO_WARN else ""
        print(f"  {name:42s} {d['assistant_turns']:4d}  "
              f"no_tool={d['no_tool']:3d} ({pct:5.0%}){flag}")
    print()
    if a["warnings"]:
        print("!! WARNINGS !!")
        for w in a["warnings"]:
            print(f"  - {w}")
    else:
        print("OK — no warnings.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-dir", default=str(TRAIN_DIR))
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Write audit as JSON to this path")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 if there are warnings")
    args = ap.parse_args()

    result = audit(Path(args.train_dir))
    _print_report(result)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2), encoding="utf-8",
        )
        print(f"\nJSON written to {args.json_out}")
    if args.strict and result["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
