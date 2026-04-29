"""JSON persistence for trackers.

Single-writer, same philosophy as bridge.news.ledger. Stores:
  - data/trackers.json   — all persisted TrackerRecords keyed by id
  - data/events.jsonl    — append-only fetch-ok/fetch-error log

Two module-level helpers are exposed for the service to call after a
successful fetch:
  - add_or_replace_reading(): dedup by timestamp (a monthly series that
    republishes the same month's value overwrites; a daily series with
    a new day's close appends).
  - prune_history(): drop readings older than history_days, keeping
    history bounded.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bridge.trackers.models import TrackerReading, TrackerRecord
from bridge.utils import atomic_write_text, json_default as _json_default

log = logging.getLogger("bridge.trackers.store")


from bridge.utils import parse_iso_utc as _parse_dt  # noqa: E402


class TrackerStore:
    """Thin JSON persistence. Single-writer. No locking.

    Full rewrites go through ``atomic_write_text`` so a concurrent
    reader never sees a half-written file. The JSONL event log stays
    plain append-mode (appends are atomic up to the fs-block boundary
    on POSIX; single-writer on Windows).
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trackers_path = self.data_dir / "trackers.json"
        self.events_path = self.data_dir / "events.jsonl"

    def load(self) -> dict[str, TrackerRecord]:
        if not self.trackers_path.exists():
            return {}
        raw = json.loads(self.trackers_path.read_text(encoding="utf-8"))
        out: dict[str, TrackerRecord] = {}
        for tid, d in raw.items():
            history = [
                TrackerReading(
                    timestamp=_parse_dt(r["timestamp"]),
                    value=float(r["value"]),
                )
                for r in d.get("history", [])
            ]
            last_fetched_at = None
            if d.get("last_fetched_at"):
                last_fetched_at = _parse_dt(d["last_fetched_at"])
            out[tid] = TrackerRecord(
                id=d["id"],
                label=d["label"],
                category=d["category"],
                history=history,
                last_fetched_at=last_fetched_at,
                last_fetch_error=d.get("last_fetch_error"),
            )
        return out

    def save(self, records: dict[str, TrackerRecord]) -> None:
        data = {tid: asdict(r) for tid, r in records.items()}
        atomic_write_text(
            self.trackers_path,
            json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        )

    def append_event(self, kind: str, data: dict[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
        }
        if data:
            entry.update(data)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")


# --- history helpers ------------------------------------------------------

def add_or_replace_reading(
    record: TrackerRecord,
    reading: TrackerReading,
) -> None:
    """Append a reading, OR overwrite an existing reading with the same
    timestamp. Sorts history ascending after the change.

    Timestamp equality as the dedup key works cleanly because:
      - Frankfurter stamps 15:00 UTC of the ECB publish date
      - BLS stamps noon UTC of the first-day-of-month
      - Stooq stamps the exact close time
    So re-fetching on the same day for FX/BLS overwrites (correct —
    upstream may revise); Stooq intraday refetch would append a slightly
    different timestamp (not ideal but rare in practice since we
    TTL-gate refetches in the service).
    """
    for i, existing in enumerate(record.history):
        if existing.timestamp == reading.timestamp:
            record.history[i] = reading
            break
    else:
        record.history.append(reading)
    record.history.sort(key=lambda r: r.timestamp)


def prune_history(
    record: TrackerRecord,
    history_days: int,
    now: datetime | None = None,
) -> None:
    """Drop readings older than history_days. Mutates in place.

    `now` is injectable for testability — default is wall-clock UTC.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=history_days)
    record.history = [r for r in record.history if r.timestamp >= cutoff]
    record.history.sort(key=lambda r: r.timestamp)
