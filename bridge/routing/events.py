"""Append-only log of routing classifier decisions.

One JSONL line per user turn. Each line records:

* the user's prompt (truncated),
* the classifier's hint (or None if it abstained),
* which tool(s) the LLM actually picked that turn (empty list if the
  turn was plain chitchat / no-tool).

This is the raw data we need to tell whether:

* the classifier is useful (hint == picked often enough to matter);
* the classifier is wrong (hint ≠ picked — the LLM overrode us);
* the classifier abstains too often (None + picked means we could
  have helped but didn't).

Matches the "every LLM decision is logged" principle from
``docs/design.md`` §3. Output path is parallel to
``bridge/news/data/events.jsonl`` so ops-time tooling can tail both.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("bridge.routing.events")

# Default log path. Created lazily on first write so importing the
# module never touches disk.
_DEFAULT_DIR: Path = Path(__file__).resolve().parent / "data"
_DEFAULT_PATH: Path = _DEFAULT_DIR / "events.jsonl"

# Keep user-prompt excerpts short. Full prompts may contain PII we
# don't need for routing analysis — a 120-char head is plenty to
# see what the user was asking without bloating the log.
_PROMPT_MAX = 120


def log_decision(
    user_text: str | None,
    hint_tool: str | None,
    picked_tools: list[str],
    *,
    path: Path | None = None,
) -> None:
    """Append one record of (prompt, hint, picked) to events.jsonl.

    Silently swallows I/O errors: logging is best-effort and should
    never break the chat turn. A warning is emitted so the failure
    is visible in server logs.
    """
    target = path if path is not None else _DEFAULT_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": "routing_decision",
            "prompt": (user_text or "").strip()[:_PROMPT_MAX],
            "hint": hint_tool,
            "picked": list(picked_tools),
            # Derived flags to make tailing the log readable without
            # post-processing. Two booleans answer the most common
            # "is this working?" questions at a glance.
            "hint_followed": (
                hint_tool is not None
                and len(picked_tools) == 1
                and picked_tools[0] == hint_tool
            ),
            "abstained": hint_tool is None,
        }
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("failed to append routing event: %s", e)
