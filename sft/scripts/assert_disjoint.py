"""Assert zero verbatim overlap between training prompts and the test set.

Called from build_dataset.py before writing the arrow dataset, so any
new training example that duplicates an eval prompt fails the build
immediately. The eval_cases yaml is the authoritative test set; training
must never contain its prompts verbatim.

Usage:
    python sft/scripts/assert_disjoint.py \\
        --train sft/phrase_library/train \\
        --eval  sft/eval_cases_novel.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _train_prompts(train_dir: Path) -> dict[str, Path]:
    """Return {normalized_prompt: source_file} for every user turn."""
    out: dict[str, Path] = {}
    for p in sorted(train_dir.glob("*.jsonl")):
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ex = json.loads(ln)
            for m in ex.get("messages") or []:
                if m.get("role") == "user":
                    text = (m.get("content") or "").strip().lower()
                    if text:
                        out.setdefault(text, p)
    return out


def _eval_prompts(eval_yaml: Path) -> list[dict]:
    import yaml
    data = yaml.safe_load(eval_yaml.read_text(encoding="utf-8"))
    return data.get("cases") or []


def check(train_dir: Path, eval_yaml: Path) -> int:
    train = _train_prompts(train_dir)
    cases = _eval_prompts(eval_yaml)

    collisions: list[tuple[str, Path]] = []
    for c in cases:
        p = (c.get("prompt") or "").strip().lower()
        if p and p in train:
            collisions.append((c.get("prompt", ""), train[p]))

    if collisions:
        print(f"TRAIN/TEST LEAKAGE: {len(collisions)} verbatim overlaps found", file=sys.stderr)
        for prompt, src in collisions[:10]:
            print(f"  {src.name}: {prompt!r}", file=sys.stderr)
        if len(collisions) > 10:
            print(f"  ... and {len(collisions) - 10} more", file=sys.stderr)
        print(
            "\nFix: remove these prompts from training or swap them for\n"
            "paraphrases. Test prompts must never appear verbatim in train.",
            file=sys.stderr,
        )
        return 1

    print(
        f"disjoint OK: {len(train)} train prompts, {len(cases)} eval prompts, "
        f"0 overlaps"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", dest="eval_yaml", type=Path, required=True)
    args = ap.parse_args()
    sys.exit(check(args.train, args.eval_yaml))


if __name__ == "__main__":
    main()
