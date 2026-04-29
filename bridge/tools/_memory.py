"""User memory tool."""
from __future__ import annotations

import logging

log = logging.getLogger("bridge.tools")


def _update_memory(fact: str = "") -> str:
    """Append a user fact to the memory store, with dedup."""
    fact = (fact or "").strip()
    if not fact:
        return "no fact provided"
    if len(fact) > 200:
        fact = fact[:200]

    from bridge.memory import MemoryStore, MAX_MEMORY_CHARS

    store = MemoryStore()
    current = store.get_user()

    existing = [line.strip().lower() for line in current.split("\n") if line.strip()]
    if fact.lower() in existing:
        return f"already known: {fact}"
    for line in existing:
        if fact.lower() in line or line in fact.lower():
            return f"similar fact already stored: {line}"

    separator = "\n" if current and not current.endswith("\n") else ""
    updated = current + separator + fact

    if len(updated) > MAX_MEMORY_CHARS:
        lines = updated.split("\n")
        while len("\n".join(lines)) > MAX_MEMORY_CHARS and len(lines) > 1:
            lines.pop(0)
        updated = "\n".join(lines)

    saved, truncated = store.set_user(updated)
    log.info("update_memory: saved fact %r (total %d chars)", fact, len(saved))
    return f"remembered: {fact}"

