"""Persistence for bridge/alerts.

Three artifacts under data/:
  - alerts.jsonl          append-only, every fire and every suppression
  - cooldowns.json        per (rule_id, scope_key) last-fired-at (owned by CooldownLedger)
  - advisory_state.json   last-known per-country travel advisory level

alerts.jsonl is the source of truth for audit. cooldowns.json is a
derived index — losing it would only forget cooldowns, not alert history.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bridge.alerts.models import Alert, SuppressedAlertRecord
from bridge.utils import atomic_write_text, json_default as _json_default

log = logging.getLogger("bridge.alerts.store")


class AlertStore:
    """File-backed alert persistence. Single-writer."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_path = self.data_dir / "alerts.jsonl"
        self.advisory_state_path = self.data_dir / "advisory_state.json"

    def append_alert(self, alert: Alert) -> None:
        entry = {"kind": "alert_fired", "alert": asdict(alert)}
        with self.alerts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    def append_suppression(self, record: SuppressedAlertRecord) -> None:
        entry = {"kind": "alert_suppressed", "record": asdict(record)}
        with self.alerts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    def append_event(self, kind: str, data: dict[str, Any] | None = None) -> None:
        """Generic event log. Used for scan-cycle metadata."""
        entry: dict[str, Any] = {
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if data:
            entry.update(data)
        with self.alerts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    def iter_fired_alerts(self):
        """Iterate every `alert_fired` entry in the log in file order."""
        if not self.alerts_path.exists():
            return
        with self.alerts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(
                        "alerts.jsonl: skipping malformed line (partial write?) — %d bytes",
                        len(line),
                    )
                    continue
                if obj.get("kind") != "alert_fired":
                    continue
                yield obj.get("alert") or {}

    def last_event(self, kind: str) -> dict[str, Any] | None:
        """Tail-scan for the last event of a given kind. Used by the TTL gate."""
        if not self.alerts_path.exists():
            return None
        last: dict[str, Any] | None = None
        with self.alerts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("kind") == kind:
                    last = obj
        return last

    # --- advisory state (separate file; small enough to rewrite on each update) ---

    def load_advisory_state(self) -> dict[str, int]:
        """Return {country_code: last_known_level}. Empty when first-run."""
        if not self.advisory_state_path.exists():
            return {}
        try:
            raw = json.loads(self.advisory_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {str(k): int(v) for k, v in raw.items() if isinstance(v, int)}

    def save_advisory_state(self, state: dict[str, int]) -> None:
        atomic_write_text(
            self.advisory_state_path,
            json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True),
        )
