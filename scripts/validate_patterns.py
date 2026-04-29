"""Offline routing-pattern validation.

Checks that every routing_pattern in TOOL_SCHEMAS:
  1. Compiles without error.
  2. Matches at least one eval case (not dead).

Also checks that:
  3. No eval case triggers two different tools (ambiguity regression).
  4. Classifier agrees with expected_tool on confident matches.

No Ollama needed — fast enough for CI.

Usage:
    python scripts/validate_patterns.py
    python scripts/validate_patterns.py --cases sft/eval_cases_novel.yaml
    python scripts/validate_patterns.py --verbose
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bridge.tools import TOOL_SCHEMAS  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw.get("cases") or []


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate routing patterns offline")
    parser.add_argument(
        "--cases", type=Path,
        default=REPO_ROOT / "sft" / "eval_cases_novel.yaml",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    prompts = [c["prompt"] for c in cases if c.get("prompt")]
    print(f"Loaded {len(prompts)} eval case prompts from {args.cases.name}")

    issues: list[str] = []

    # --- Build pattern index ---
    tool_patterns: dict[str, list[tuple[str, re.Pattern]]] = {}
    compile_errors = 0
    for schema in TOOL_SCHEMAS:
        fn = schema.get("function", {})
        name = fn.get("name", "")
        patterns = fn.get("routing_patterns", [])
        if not name or not patterns:
            continue
        compiled: list[tuple[str, re.Pattern]] = []
        for src in patterns:
            try:
                compiled.append((src, re.compile(src, re.IGNORECASE)))
            except re.error as e:
                issues.append(f"COMPILE ERROR: {name} pattern {src!r}: {e}")
                compile_errors += 1
        tool_patterns[name] = compiled

    total_patterns = sum(len(v) for v in tool_patterns.values())
    print(f"Compiled {total_patterns} patterns across {len(tool_patterns)} tools "
          f"({compile_errors} errors)")

    # --- Check 1: dead patterns (never match any eval case) ---
    dead = 0
    for tool_name, pats in tool_patterns.items():
        for src, pat in pats:
            matched_any = any(pat.search(p) for p in prompts)
            if not matched_any:
                dead += 1
                issues.append(f"DEAD PATTERN: {tool_name}: {src!r} matches zero eval cases")
                if args.verbose:
                    print(f"  [dead] {tool_name}: {src[:60]}")
    if dead:
        print(f"  {dead} dead pattern(s) — consider adding eval cases or removing the pattern")

    # --- Check 2: ambiguity (one prompt triggers two tools) ---
    ambiguous = 0
    for case in cases:
        prompt = case.get("prompt", "")
        if not prompt:
            continue
        hits: set[str] = set()
        for tool_name, pats in tool_patterns.items():
            for _, pat in pats:
                if pat.search(prompt):
                    hits.add(tool_name)
                    break
        if len(hits) > 1:
            expected = case.get("expected_tool")
            if expected is not None:
                # Ambiguity on a case that expects a specific tool is a regression.
                ambiguous += 1
                issues.append(
                    f"AMBIGUITY: {prompt!r} triggers {sorted(hits)} "
                    f"(expected {expected})"
                )
                if args.verbose:
                    print(f"  [ambig] {prompt[:50]} -> {sorted(hits)}")
    if ambiguous:
        print(f"  {ambiguous} ambiguity regression(s)")

    # --- Check 3: classifier agrees with expected_tool ---
    from bridge.routing.classifier import classify
    mismatches = 0
    for case in cases:
        prompt = case.get("prompt", "")
        expected = case.get("expected_tool")
        if not prompt or expected is None:
            continue
        got = classify(prompt)
        # Classifier abstaining (None) is acceptable — it's conservative.
        # But classifying to a WRONG tool is a regression.
        if got is not None and got != expected:
            mismatches += 1
            issues.append(
                f"MISMATCH: classifier({prompt!r}) = {got}, expected {expected}"
            )
            if args.verbose:
                print(f"  [mismatch] {prompt[:50]} -> {got} (expected {expected})")
    if mismatches:
        print(f"  {mismatches} classifier mismatch(es)")

    # --- Summary ---
    print(f"\n{'='*50}")
    if issues:
        print(f"ISSUES: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("ALL CLEAR — no dead patterns, no ambiguities, no mismatches.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
