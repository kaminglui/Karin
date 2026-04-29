"""Produce a bucket + tool breakdown from an eval_routing JSON dump.

Usage:
    docker exec karin-web python3 /app/scripts/bench_analyze.py /tmp/eval_mannix_216.json
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path


def _bucket_of(case: dict) -> str:
    if case.get("history") or case.get("_has_history"):
        return "multi-turn"
    notes = (case.get("notes") or "").split(",")[0].strip()
    return notes or "untagged"


def main(path: str) -> int:
    # Reload the eval YAML so we can recover bucket info (the JSON dump
    # doesn't carry the notes field by default — it does record prompt).
    import yaml
    repo_yaml = Path("/app/scripts/eval_cases.yaml")
    cases_by_prompt: dict[str, dict] = {}
    if repo_yaml.exists():
        for c in yaml.safe_load(repo_yaml.read_text()).get("cases", []):
            cases_by_prompt[c["prompt"]] = c

    data = json.loads(Path(path).read_text())
    cases = data["cases"]
    total = len(cases)
    passed = sum(1 for c in cases if c["passed"])

    print(f"=== {path} ===")
    print(f"Overall: {passed}/{total} ({passed/total:.1%})")

    # By bucket
    by_bucket: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cases:
        ref = cases_by_prompt.get(c["prompt"], {})
        if ref.get("history"):
            c["_has_history"] = True
            c["notes"] = ref.get("notes", "")
        else:
            c["notes"] = ref.get("notes", "")
        b = _bucket_of(c)
        by_bucket[b].append(c)

    print()
    print("=== By bucket ===")
    for name, lst in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
        p = sum(1 for x in lst if x["passed"])
        n = len(lst)
        rate = p / n if n else 0
        lats = [x["latency_s"] for x in lst]
        print(f"  {name:25s}  {p:3d}/{n:3d} ({rate:5.1%})  mean lat {statistics.mean(lats):.1f}s")

    # By expected tool
    by_tool: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cases:
        et = c.get("expected_tool") or "<null>"
        by_tool[et].append(c)

    print()
    print("=== By expected_tool ===")
    for name, lst in sorted(by_tool.items(), key=lambda kv: -len(kv[1])):
        p = sum(1 for x in lst if x["passed"])
        n = len(lst)
        rate = p / n if n else 0
        print(f"  {name:20s}  {p:3d}/{n:3d} ({rate:5.1%})")

    # Top failures grouped by expected tool
    print()
    print("=== Failures ===")
    fails = [c for c in cases if not c["passed"]]
    for c in fails:
        exp = c.get("expected_tool") or "<null>"
        act = c.get("actual_tool") or "<notool>"
        prompt = c["prompt"]
        notes = c.get("notes", "")
        prefix = "[MT] " if c.get("_has_history") else "     "
        print(f"  {prefix}exp={exp:16s} got={act:16s}  {prompt[:60]}  [{notes}]")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: bench_analyze.py path/to/eval.json", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
