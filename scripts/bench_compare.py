"""Compare eval_routing.py JSON dumps across models side-by-side.

Usage:
    docker exec karin-web python3 /app/scripts/bench_compare.py \\
        /tmp/eval_llama31.json /tmp/eval_hermes3.json /tmp/eval_dolphin3.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def main(paths: list[str]) -> int:
    datasets = [(Path(p).stem, load(p)) for p in paths]

    print("\n=== Pass rate + latency ===")
    for name, d in datasets:
        latencies = [c["latency_s"] for c in d["cases"]]
        print(f"  {name:20s}  {d['passed']:2d}/{d['total']:2d} "
              f"({d['passed']/d['total']:.1%})  "
              f"mean {statistics.mean(latencies):.1f}s  "
              f"median {statistics.median(latencies):.1f}s  "
              f"max {max(latencies):.1f}s")

    # Use first as baseline
    base_name, base = datasets[0]
    base_cases = {c["prompt"]: c for c in base["cases"]}

    for name, d in datasets[1:]:
        cand_cases = {c["prompt"]: c for c in d["cases"]}
        regressed = [(p, cand_cases[p])
                     for p, bc in base_cases.items()
                     if bc["passed"] and p in cand_cases
                     and not cand_cases[p]["passed"]]
        fixed = [(p, cand_cases[p])
                 for p, bc in base_cases.items()
                 if not bc["passed"] and p in cand_cases
                 and cand_cases[p]["passed"]]

        print(f"\n=== vs {base_name}: {name} ===")
        print(f"  Regressions (was-pass -> now-fail): {len(regressed)}")
        for p, c in regressed[:15]:
            exp = c.get("expected_tool") or "<null>"
            act = c.get("actual_tool") or "<no tool>"
            print(f"    - [{p[:60]}] exp={exp} got={act}")
        if len(regressed) > 15:
            print(f"    ...and {len(regressed) - 15} more")
        print(f"  Fixes (was-fail -> now-pass): {len(fixed)}")
        for p, c in fixed[:10]:
            print(f"    + [{p[:60]}] -> {c.get('actual_tool') or '<no tool>'}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: bench_compare.py result1.json [result2.json ...]",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
