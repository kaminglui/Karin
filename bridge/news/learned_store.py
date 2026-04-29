"""JSON-backed store for LLM-learned watchlist entities.

Schema on disk (``data/news/learned_keywords.json``):

.. code-block:: json

    {
      "regions": {
        "US": {
          "<label>": {
            "kind": "organization",
            "confidence": 80,
            "count": 12,
            "first_seen": "2026-04-15T00:00:00+00:00",
            "last_seen":  "2026-04-16T14:30:00+00:00"
          },
          ...
        },
        ...
      },
      "topics": { ... },
      "events": { ... }
    }

The three top-level keys mirror the watchlist taxonomy (regions /
topics / events). Each bucket holds a dict keyed by a **user-visible
watchlist label** ("US", "Mainland China", "AI / Tech", etc.), with
nested entity records keyed by lowercase label so dedup is natural
on update.

Design rules:

* Single-writer: the news service is the only writer. No locking.
* Idempotent updates: ``record()`` merges new sightings with the
  existing record — increments ``count``, bumps ``last_seen``,
  preserves ``first_seen``.
* TTL is write-time: ``sweep_expired()`` drops entries not seen in
  ``ttl_days`` days. Called by the service after each record pass
  so stale entries don't pile up.
* Empty buckets are cleaned up by the sweep so reading the file
  isn't noisy.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bridge.news.keyword_learn import LearnedEntity
from bridge.utils import atomic_write_text

log = logging.getLogger("bridge.news.learned_store")


DEFAULT_TTL_DAYS = 30

# The three canonical bucket keys. Keeping this list aligned with the
# preferences schema means any UI reading learned keywords can
# iterate the same order as Regions / Topics / Events tabs.
BUCKET_KEYS: tuple[str, ...] = ("regions", "topics", "events")


@dataclass(frozen=True)
class LearnedRecord:
    """One row on disk. Frozen so misuse shows as a TypeError,
    serialized via `to_dict` for writes."""
    label: str
    kind: str
    confidence: int
    count: int
    first_seen: datetime
    last_seen: datetime

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "count": self.count,
            "first_seen": self.first_seen.astimezone(timezone.utc).isoformat(),
            "last_seen": self.last_seen.astimezone(timezone.utc).isoformat(),
        }


def _parse_dt(s: str) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class LearnedStore:
    """File-backed bucket→bucket_label→entity_label→LearnedRecord."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    # --- I/O -----------------------------------------------------------

    def load(self) -> dict[str, dict[str, dict[str, LearnedRecord]]]:
        """Load the whole file.

        Missing / malformed file -> empty nested dict with all bucket
        keys present, so callers can blindly do
        ``store.load()["regions"][label]`` without branching.
        """
        empty: dict[str, dict[str, dict[str, LearnedRecord]]] = {
            k: {} for k in BUCKET_KEYS
        }
        if not self._path.exists():
            return empty
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("learned_keywords.json unreadable: %s; starting fresh", e)
            return empty
        if not isinstance(raw, dict):
            return empty
        for bucket in BUCKET_KEYS:
            src = raw.get(bucket, {})
            if not isinstance(src, dict):
                continue
            for wl_label, ents in src.items():
                if not isinstance(ents, dict):
                    continue
                bucket_out: dict[str, LearnedRecord] = {}
                for key, rec in ents.items():
                    if not isinstance(rec, dict):
                        continue
                    first = _parse_dt(rec.get("first_seen", ""))
                    last = _parse_dt(rec.get("last_seen", ""))
                    if first is None or last is None:
                        continue
                    try:
                        count = int(rec.get("count", 1))
                    except (TypeError, ValueError):
                        count = 1
                    try:
                        conf = int(rec.get("confidence", 0))
                    except (TypeError, ValueError):
                        conf = 0
                    bucket_out[key] = LearnedRecord(
                        label=str(rec.get("label", key)),
                        kind=str(rec.get("kind", "other")),
                        confidence=conf,
                        count=max(count, 1),
                        first_seen=first,
                        last_seen=last,
                    )
                if bucket_out:
                    empty[bucket][str(wl_label)] = bucket_out
        return empty

    def load_analyzed(self) -> dict[str, dict[str, set[str]]]:
        """Return the set of cluster_ids already analyzed per bucket.

        Shape: ``{bucket: {watchlist_label: {cluster_id, ...}}}``. A
        sidecar at the root of the on-disk JSON under the reserved
        ``_analyzed_clusters`` key. Lets the service skip re-learning
        when nothing new has landed since the last pass — the LLM
        isn't asked to re-extract what it's already seen.
        """
        out: dict[str, dict[str, set[str]]] = {k: {} for k in BUCKET_KEYS}
        if not self._path.exists():
            return out
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return out
        if not isinstance(raw, dict):
            return out
        sidecar = raw.get("_analyzed_clusters") or {}
        if not isinstance(sidecar, dict):
            return out
        for bucket in BUCKET_KEYS:
            b = sidecar.get(bucket) or {}
            if not isinstance(b, dict):
                continue
            for wl_label, cids in b.items():
                if not isinstance(cids, list):
                    continue
                out[bucket][str(wl_label)] = {
                    str(cid) for cid in cids if cid
                }
        return out

    def save(
        self,
        data: dict[str, dict[str, dict[str, LearnedRecord]]],
        analyzed: "dict[str, dict[str, set[str]]] | None" = None,
    ) -> None:
        """Write the whole dict to disk as one JSON blob.

        Creates the parent directory if needed. Atomic-ish: writes to
        `.tmp` sibling, then renames.

        ``analyzed`` is the optional sidecar tracking which cluster
        ids the LLM has already learned from per bucket — stored so
        the next pass can skip buckets that have no new clusters.
        Missing / None preserves whatever was on disk already.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, dict] = {}
        for bucket in BUCKET_KEYS:
            bdata = data.get(bucket, {})
            if not bdata:
                continue
            bucket_out: dict[str, dict] = {}
            for wl_label, ents in bdata.items():
                if not ents:
                    continue
                serialized: dict[str, dict] = {}
                for key, rec in ents.items():
                    d = rec.to_dict()
                    d["label"] = rec.label
                    serialized[key] = d
                if serialized:
                    bucket_out[wl_label] = serialized
            if bucket_out:
                out[bucket] = bucket_out

        # Preserve or overwrite the analyzed-clusters sidecar.
        if analyzed is None:
            # Read-through the existing sidecar so save() calls that
            # aren't mutating the analyzed set don't wipe it.
            if self._path.exists():
                try:
                    prev = json.loads(self._path.read_text(encoding="utf-8"))
                    if isinstance(prev, dict) and isinstance(prev.get("_analyzed_clusters"), dict):
                        out["_analyzed_clusters"] = prev["_analyzed_clusters"]
                except (json.JSONDecodeError, OSError):
                    pass
        else:
            sidecar: dict[str, dict[str, list[str]]] = {}
            for bucket in BUCKET_KEYS:
                bdata = analyzed.get(bucket) or {}
                bucket_out2: dict[str, list[str]] = {}
                for wl_label, cids in bdata.items():
                    if not cids:
                        continue
                    bucket_out2[wl_label] = sorted(cids)
                if bucket_out2:
                    sidecar[bucket] = bucket_out2
            if sidecar:
                out["_analyzed_clusters"] = sidecar

        atomic_write_text(
            self._path,
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        )


# --- mutations -----------------------------------------------------------

def record(
    data: dict[str, dict[str, dict[str, LearnedRecord]]],
    bucket: str,
    wl_label: str,
    entities: list[LearnedEntity],
    now: datetime,
) -> int:
    """Merge `entities` into `data[bucket][wl_label]`.

    Returns the number of NEW rows added (first-time sightings). Existing
    entries get their ``count`` incremented and ``last_seen`` bumped to
    ``now``; first_seen stays put. Caller drives the datetime so tests
    can pin an exact clock.
    """
    if bucket not in BUCKET_KEYS:
        raise ValueError(f"unknown bucket {bucket!r}; expected one of {BUCKET_KEYS}")
    if not entities:
        return 0
    bdata = data.setdefault(bucket, {})
    ents = bdata.setdefault(wl_label, {})
    new_rows = 0
    for ent in entities:
        key = ent.label.lower()
        existing = ents.get(key)
        if existing is None:
            ents[key] = LearnedRecord(
                label=ent.label,
                kind=ent.kind,
                confidence=ent.confidence,
                count=1,
                first_seen=now,
                last_seen=now,
            )
            new_rows += 1
        else:
            # Keep the higher confidence, the most recent kind vote,
            # and the most recent display form of the label (Qwen may
            # capitalise differently between calls — last-seen wins).
            ents[key] = LearnedRecord(
                label=ent.label or existing.label,
                kind=ent.kind or existing.kind,
                confidence=max(existing.confidence, ent.confidence),
                count=existing.count + 1,
                first_seen=existing.first_seen,
                last_seen=now,
            )
    return new_rows


def sweep_expired(
    data: dict[str, dict[str, dict[str, LearnedRecord]]],
    now: datetime,
    ttl_days: int = DEFAULT_TTL_DAYS,
) -> int:
    """Drop entries whose ``last_seen`` is older than ``ttl_days``.

    Also prunes empty watchlist-label dicts so the on-disk JSON stays
    tidy. Returns the number of entries removed.
    """
    cutoff = now - timedelta(days=ttl_days)
    dropped = 0
    for bucket in BUCKET_KEYS:
        bdata = data.get(bucket)
        if not bdata:
            continue
        empty_labels: list[str] = []
        for wl_label, ents in bdata.items():
            stale = [k for k, r in ents.items() if r.last_seen < cutoff]
            for k in stale:
                del ents[k]
                dropped += 1
            if not ents:
                empty_labels.append(wl_label)
        for wl_label in empty_labels:
            del bdata[wl_label]
    return dropped


# --- view helper ---------------------------------------------------------

def to_ui_payload(
    data: dict[str, dict[str, dict[str, LearnedRecord]]],
) -> dict:
    """Flatten the store into a JSON-friendly shape for the API.

    Each bucket becomes an array of ``{wl_label, entities: [...]}``
    rows. Entities are sorted by (count desc, last_seen desc) so the
    most-recurring show up first.
    """
    out: dict[str, list[dict]] = {k: [] for k in BUCKET_KEYS}
    for bucket in BUCKET_KEYS:
        bdata = data.get(bucket, {})
        for wl_label, ents in bdata.items():
            rows = [
                {
                    "label": r.label,
                    "kind": r.kind,
                    "confidence": r.confidence,
                    "count": r.count,
                    "first_seen": r.first_seen.isoformat(),
                    "last_seen": r.last_seen.isoformat(),
                }
                for r in ents.values()
            ]
            # Highest count first, then reverse-chronological tiebreak
            # via the ISO-string-as-float key. The earlier
            # ``rows.sort(key=lambda x: (-x['count'], x['last_seen']))``
            # was a noop — the second sort below superseded it — so the
            # noop has been removed.
            rows.sort(key=lambda x: (-x["count"], -_iso_sort_key(x["last_seen"])))
            out[bucket].append({"watchlist_label": wl_label, "entities": rows})
        out[bucket].sort(key=lambda x: x["watchlist_label"].lower())
    return out


def _iso_sort_key(iso: str) -> float:
    """Numeric sort key from an ISO timestamp. Used by to_ui_payload's
    tiebreak so rows with the same count come back in reverse-chron
    order. Invalid / missing timestamps push to the tail."""
    dt = _parse_dt(iso)
    if dt is None:
        return 0.0
    return dt.timestamp()
