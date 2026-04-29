"""Per-(rule_id, scope_key) cooldown tracking.

Kept in its own small module so the engine's logic stays readable and
so tests can pin cooldown semantics without spinning up the rest of
the subsystem.

Storage shape on disk (cooldowns.json):
  {
    "rule_id::scope_key": "2026-04-12T14:30:00+00:00",
    ...
  }

Keys use "::" as the separator so a scope_key containing ":" (e.g. a
cross-channel cartesian key like "gold_usd:cluster:abc123") doesn't
collide with the rule_id boundary.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bridge.utils import atomic_write_text


log = logging.getLogger("bridge.alerts.cooldown")
_KEY_SEP = "::"


class CooldownLedger:
    """File-backed cooldown map. Single-writer, same caveat as other ledgers."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, datetime] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            self._state = {}
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt file — start clean. The event log still preserves
            # history, but every cooldown is reset, which means alerts
            # that were suppressed will fire on the next scan. Surface
            # loudly so operators can notice and investigate.
            log.warning(
                "cooldown ledger %s is corrupt; resetting state — "
                "previously-suppressed alerts will re-fire on next scan",
                self._path,
            )
            self._state = {}
            return
        parsed: dict[str, datetime] = {}
        for key, ts in raw.items():
            try:
                dt = datetime.fromisoformat(ts)
            except (TypeError, ValueError):
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            parsed[key] = dt.astimezone(timezone.utc)
        self._state = parsed

    def _save(self) -> None:
        data = {k: v.astimezone(timezone.utc).isoformat() for k, v in self._state.items()}
        atomic_write_text(
            self._path,
            json.dumps(data, indent=2, ensure_ascii=False),
        )

    @staticmethod
    def _composite(rule_id: str, scope_key: str) -> str:
        return f"{rule_id}{_KEY_SEP}{scope_key}"

    def last_fired(self, rule_id: str, scope_key: str) -> datetime | None:
        self._ensure_loaded()
        return self._state.get(self._composite(rule_id, scope_key))

    def is_in_cooldown(
        self, rule_id: str, scope_key: str,
        cooldown_hours: int, now: datetime,
    ) -> bool:
        """True if the rule+scope fired within the last cooldown_hours."""
        last = self.last_fired(rule_id, scope_key)
        if last is None:
            return False
        return (now - last) < timedelta(hours=cooldown_hours)

    def mark_fired(
        self, rule_id: str, scope_key: str, now: datetime,
    ) -> None:
        self._ensure_loaded()
        self._state[self._composite(rule_id, scope_key)] = now
        self._save()
