"""User + agent long-term memory / skills.

Two tiny markdown files the user edits freely:

- ``data/memory/user.md``  — facts about the user (name, location,
  preferences). Injected into the system prompt so Karin remembers
  across conversations.
- ``data/memory/agent.md`` — extra instructions or "skills" for Karin
  herself (tone, catch-phrases, task-specific behaviors).

Kept deliberately small because this repo targets tiny models — at
num_ctx=4096, a 400-token (~1600 char) memory block already costs 10%
of the window before the conversation even starts. We cap each file at
``MAX_MEMORY_CHARS`` and truncate when injecting.
"""
from __future__ import annotations

import logging
from pathlib import Path

from bridge.utils import REPO_ROOT

log = logging.getLogger("bridge.memory")

# Phase H: memory is per-profile. Both user.md and agent.md move so a
# profile can carry its own Karin persona on top of its own user facts.
# Kept as a module-level constant for the migration runner to locate.
LEGACY_MEMORY_DIR: Path = REPO_ROOT / "data" / "memory"

# Per-file hard cap. Keeps the memory block small enough that a 4 GB
# quantized model still has headroom. Users editing past the cap get
# truncated on save (with a warning in the response).
MAX_MEMORY_CHARS: int = 1000


def _default_paths() -> tuple[Path, Path]:
    from bridge.profiles import active_profile
    mem_dir = active_profile().memory_dir
    return mem_dir / "user.md", mem_dir / "agent.md"


class MemoryStore:
    """File-backed read/write for user + agent memory."""

    def __init__(
        self,
        user_path: Path | None = None,
        agent_path: Path | None = None,
    ) -> None:
        if user_path is None or agent_path is None:
            default_user, default_agent = _default_paths()
            user_path = user_path or default_user
            agent_path = agent_path or default_agent
        self.user_path = user_path
        self.agent_path = agent_path
        # mkdir each parent so callers with mixed custom paths (e.g.
        # tests) don't have to pre-create dirs. idempotent.
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent_path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self, path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as e:
            log.warning("failed to read %s: %s", path.name, e)
            return ""

    def _write(self, path: Path, text: str) -> tuple[str, bool]:
        """Write text, truncating at MAX_MEMORY_CHARS. Returns (saved, truncated)."""
        clean = (text or "").strip()
        truncated = len(clean) > MAX_MEMORY_CHARS
        if truncated:
            clean = clean[:MAX_MEMORY_CHARS].rstrip()
        path.write_text(clean, encoding="utf-8")
        return clean, truncated

    def get_user(self) -> str:
        return self._read(self.user_path)

    def get_agent(self) -> str:
        return self._read(self.agent_path)

    def set_user(self, text: str) -> tuple[str, bool]:
        return self._write(self.user_path, text)

    def set_agent(self, text: str) -> tuple[str, bool]:
        return self._write(self.agent_path, text)

    def build_prompt_block(self) -> str:
        """Compose the memory text that gets prepended to the system prompt.

        Returns an empty string when both files are empty — we don't
        want to waste tokens on blank section headers.
        """
        user = self.get_user()
        agent = self.get_agent()
        parts: list[str] = []
        if user:
            parts.append(f"[About the user]\n{user}")
        if agent:
            parts.append(f"[Your additional instructions]\n{agent}")
        return "\n\n".join(parts)
