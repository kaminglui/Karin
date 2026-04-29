"""JSON ledger: articles.json, clusters.json, events.jsonl.

Single-writer persistence for the news subsystem. V1 rewrites the whole
articles/clusters files on each save — fine at <10k entries, which is
the realistic ceiling for a 10-15 feed setup. Revisit if we ever grow
larger.

Datetimes are serialized as ISO-8601 UTC. On load, naive datetimes are
coerced to UTC rather than erroring (RSS feeds occasionally omit timezone
info); the fetcher stamps fetched_at as UTC-aware so this only matters
for published_at edge cases.

The event log is append-only JSONL. `last_event(kind)` tail-scans it to
support the TTL cache gate without needing a separate index.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    StoryCluster,
)
from bridge.utils import atomic_write_text

log = logging.getLogger("bridge.news.ledger")


def _json_default(obj: Any) -> Any:
    """json.dumps default= hook for our dataclass types."""
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"not serializable: {type(obj)!r}")


from bridge.utils import parse_iso_utc as _parse_dt  # noqa: E402


class Ledger:
    """Thin JSON persistence layer. No locking — single-writer by design.

    If we ever run ingest concurrently with get_news reads, revisit: JSON
    file writes are not atomic on Windows, so a concurrent reader could
    see a truncated file. Current usage is strictly sequential inside
    NewsService, which holds the single writer.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.articles_path = self.data_dir / "articles.json"
        self.clusters_path = self.data_dir / "clusters.json"
        self.events_path = self.data_dir / "events.jsonl"

    # --- articles ----------------------------------------------------------

    def load_articles(self) -> dict[str, NormalizedArticle]:
        if not self.articles_path.exists():
            return {}
        raw = json.loads(self.articles_path.read_text(encoding="utf-8"))
        # display_summary + language were added in the translation
        # phase; old ledger rows won't have them. Default to "" and "en"
        # (our existing feeds are all English) so migration is silent.
        # A future ingest pass overwrites with detected values when the
        # article is refreshed.
        return {
            aid: NormalizedArticle(
                article_id=d["article_id"],
                source_id=d["source_id"],
                url=d["url"],
                display_title=d["display_title"],
                normalized_title=d["normalized_title"],
                summary=d["summary"],
                fingerprint=d["fingerprint"],
                wire_attribution=d["wire_attribution"],
                published_at=_parse_dt(d["published_at"]),
                fetched_at=_parse_dt(d["fetched_at"]),
                display_summary=d.get("display_summary", ""),
                language=d.get("language", "en"),
            )
            for aid, d in raw.items()
        }

    def save_articles(self, articles: dict[str, NormalizedArticle]) -> None:
        data = {aid: asdict(a) for aid, a in articles.items()}
        atomic_write_text(
            self.articles_path,
            json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        )

    # --- clusters ----------------------------------------------------------

    def load_clusters(self) -> dict[str, StoryCluster]:
        if not self.clusters_path.exists():
            return {}
        raw = json.loads(self.clusters_path.read_text(encoding="utf-8"))
        return {
            cid: StoryCluster(
                cluster_id=d["cluster_id"],
                article_ids=list(d["article_ids"]),
                centroid_display_title=d["centroid_display_title"],
                centroid_normalized_title=d["centroid_normalized_title"],
                first_seen_at=_parse_dt(d["first_seen_at"]),
                latest_update_at=_parse_dt(d["latest_update_at"]),
                last_checked_at=_parse_dt(d["last_checked_at"]),
                last_state_change_at=_parse_dt(d["last_state_change_at"]),
                state=ConfidenceState(d["state"]),
                is_stale=bool(d["is_stale"]),
                independent_confirmation_count=int(d["independent_confirmation_count"]),
                article_count=int(d["article_count"]),
                syndicated_article_count=int(d["syndicated_article_count"]),
            )
            for cid, d in raw.items()
        }

    def save_clusters(self, clusters: dict[str, StoryCluster]) -> None:
        data = {cid: asdict(c) for cid, c in clusters.items()}
        atomic_write_text(
            self.clusters_path,
            json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        )

    # --- events ------------------------------------------------------------

    def append_event(self, kind: str, data: dict[str, Any] | None = None) -> None:
        """Append one JSON line to events.jsonl. Never rewrites history."""
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
        }
        if data:
            entry.update(data)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    def last_event(self, kind: str) -> dict[str, Any] | None:
        """Tail-scan events.jsonl for the most recent event of `kind`.

        Used by the TTL cache gate (service.ingest_latest). For V1 the log
        is small enough that a full scan is cheap; if it ever grows past
        a few MB, switch to reverse-line iteration.
        """
        if not self.events_path.exists():
            return None
        last: dict[str, Any] | None = None
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping malformed event log line")
                    continue
                if entry.get("kind") == kind:
                    last = entry
        return last
