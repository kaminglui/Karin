"""Build HuggingFace datasets from the SFT phrase library.

Reads JSONL files under ``sft/phrase_library/{train,dpo_pairs}/`` and
emits three HF ``Dataset`` objects (train_sft, train_dpo, optional
eval_held) to a target directory ready for ``load_from_disk``.

System prompt substitution:

  Training JSONL uses ``{{SYSTEM}}`` as a placeholder so the source
  files stay compact and re-playable with different personas. At build
  time we read the active character's ``system_prompt`` from
  ``characters/profile.yaml`` (joined with voices/{voice}/voice.yaml
  for the persona / language fields) and substitute it into every
  training example.

Usage (from repo root)::

    python sft/scripts/build_dataset.py \\
        --character karin \\
        --voice general \\
        --out sft/dataset.arrow

Then in the Colab notebook::

    from datasets import load_from_disk
    ds = load_from_disk("sft/dataset.arrow")
    train_sft = ds["train_sft"]
    train_dpo = ds["train_dpo"]

The notebook applies the model's chat template and tokenizes at that
point — this script stays template-agnostic so the same dataset works
across Llama 3 / Qwen 2.5 / Mistral instruct variants.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_ROOT = REPO_ROOT / "sft" / "phrase_library"
TEST_SET = REPO_ROOT / "sft" / "eval_cases_novel.yaml"


def _assert_disjoint() -> None:
    """Delegate to assert_disjoint.py so the same check runs in CI."""
    import subprocess
    r = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("assert_disjoint.py")),
            "--train", str(LIB_ROOT / "train"),
            "--eval", str(TEST_SET),
        ],
        capture_output=True,
        text=True,
    )
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, end="", file=sys.stderr)
        raise SystemExit(
            "Training dataset leaks into the test set. Aborting build."
        )


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Skip pure-comment rows. Some jsonl files carry a leading
            # "{ _comment: ... }" line as inline documentation so the
            # schema + rationale live with the data. Passing those to
            # _apply_system / _apply_system_dpo would produce invalid
            # training examples (empty prompt / no messages).
            if isinstance(row, dict) and set(row.keys()) == {"_comment"}:
                continue
            out.append(row)
    return out


def _flatten_tool_calls(msgs: list[dict]) -> list[dict]:
    """Materialize assistant tool_calls into content as plain JSON.

    mlabonne/huihui abliterated Llama 3.1 tokenizer's chat template
    silently ignores `tool_calls` on assistant messages and renders
    content=None as the literal string "None". Serializing tool_calls
    into content='{"name": ..., "arguments": ...}' makes the target
    survive the template and lands in the format that Karin's fallback
    parser accepts (bridge/llm.py Shape B: name + arguments keys).
    """
    out = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name", "unknown")
            args = fn.get("arguments") or tc.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            rendered = json.dumps(
                {"name": name, "arguments": args}, ensure_ascii=False
            )
            out.append({**m, "content": rendered, "tool_calls": None})
        else:
            out.append(m)
    return out


def _resolve_system_prompt(character: str, voice: str) -> str:
    """Load and render the character's system prompt.

    Mirrors what bridge/llm.py does at runtime: read the template from
    ``characters/<character>/profile.yaml`` (or fall back to the shared
    ``characters/profile.yaml``), then fill {persona} and {language_note}
    from the voice's ``voice.yaml``. If the voice yaml is missing (it's
    gitignored by default), we degrade to empty strings so training
    never blocks on it.
    """
    import yaml

    char_dir = REPO_ROOT / "characters" / character
    profile_path = char_dir / "profile.yaml"
    if not profile_path.exists():
        profile_path = REPO_ROOT / "characters" / "profile.yaml"
    with profile_path.open("r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)

    template = profile.get("system_prompt", "")

    # Voice yaml lives at one of (in priority order):
    #   characters/<char>/voices/<voice>/voice.yaml   (multi-voice layout)
    #   characters/<char>/voice.yaml                  (single-voice flat layout)
    # File may be gitignored — degrade gracefully to empty fields.
    candidates = [
        char_dir / "voices" / voice / "voice.yaml",
        char_dir / "voice.yaml",
    ]
    persona = ""
    language_note = ""
    for voice_path in candidates:
        if voice_path.exists():
            with voice_path.open("r", encoding="utf-8") as f:
                vdata = yaml.safe_load(f) or {}
            persona = vdata.get("persona", "")
            language_note = vdata.get("language_note", "")
            break

    # Use targeted replace, not .format() — the template contains
    # literal `{` / `}` in the JSON examples it quotes (e.g.
    # `{"function": ...}`), which would crash str.format.
    return (
        template
        .replace("{persona}", persona)
        .replace("{language_note}", language_note)
    )


def _apply_system(example: dict, system_prompt: str) -> dict:
    msgs = example.get("messages")
    if not msgs:
        return example
    patched = []
    for m in msgs:
        if m.get("role") == "system" and m.get("content") == "{{SYSTEM}}":
            patched.append({"role": "system", "content": system_prompt})
        else:
            patched.append(m)
    return {**example, "messages": _flatten_tool_calls(patched)}


def _apply_system_dpo(example: dict, system_prompt: str) -> dict:
    """DPO entries carry ``prompt`` / ``chosen`` / ``rejected``. We
    prepend a system message to ``prompt`` since the base library
    stores it without one. `chosen`/`rejected` are lists of assistant
    messages that may carry `tool_calls` — flatten those too."""
    prompt = example.get("prompt") or []
    sys_msg = {"role": "system", "content": system_prompt}
    patched_prompt = [sys_msg, *prompt]
    return {
        **example,
        "prompt": patched_prompt,
        "chosen": _flatten_tool_calls(example.get("chosen") or []),
        "rejected": _flatten_tool_calls(example.get("rejected") or []),
    }


def build(character: str, voice: str, out_dir: Path, include_held: bool = True) -> None:
    from datasets import Dataset, DatasetDict

    # Fail the build if any training prompt appears verbatim in the
    # held-out test set. Protects against accidentally re-introducing
    # the leakage that invalidated the early SFT eval numbers.
    _assert_disjoint()

    system_prompt = _resolve_system_prompt(character, voice)

    train_rows: list[dict] = []
    for p in sorted((LIB_ROOT / "train").glob("*.jsonl")):
        for row in _read_jsonl(p):
            train_rows.append(_apply_system(row, system_prompt))

    dpo_rows: list[dict] = []
    for p in sorted((LIB_ROOT / "dpo_pairs").glob("*.jsonl")):
        for row in _read_jsonl(p):
            dpo_rows.append(_apply_system_dpo(row, system_prompt))

    train_sft = Dataset.from_list(train_rows) if train_rows else None
    train_dpo = Dataset.from_list(dpo_rows) if dpo_rows else None

    splits: dict[str, Dataset] = {}
    if train_sft is not None:
        splits["train_sft"] = train_sft
    if train_dpo is not None:
        splits["train_dpo"] = train_dpo

    if not splits:
        raise SystemExit("No training examples found under sft/phrase_library/")

    ds = DatasetDict(splits)
    ds.save_to_disk(str(out_dir))

    print(f"Wrote {out_dir}")
    for name, split in splits.items():
        print(f"  {name}: {len(split)} examples")
    print(f"System prompt length: {len(system_prompt)} chars")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="karin")
    ap.add_argument("--voice", default="general")
    ap.add_argument("--out", default="sft/dataset.arrow")
    args = ap.parse_args()
    build(args.character, args.voice, Path(args.out))


if __name__ == "__main__":
    main()
