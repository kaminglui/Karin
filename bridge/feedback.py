"""Per-turn preference feedback store.

Captures user thumbs-up/down ratings on assistant replies along with
an embedding of the user's message and the tools the model called to
produce the reply. The resulting log is what ``bridge.bandit`` later
uses to retrieve "turns like this one" and nudge routing toward tools
that worked in the past.

No model weight updates, no fine-tuning — this is a pure retrieval /
preference-lookup layer. Append-only JSONL on disk so you can grep or
tail the log for debugging.

Design notes:
- Ratings may arrive seconds or hours AFTER the turn ended (user sees
  the reply, decides, clicks). We record the entry immediately with
  ``rating=None`` and patch it via ``update_rating(turn_id, rating)``.
- Embeddings are supplied by a caller-provided callable so this module
  stays decoupled from whichever embedding provider (Ollama, a local
  sentence-transformers model, etc.) the server picks.
- If embedding fails we still append the entry with ``embedding=None``
  so the turn isn't lost — the k-NN lookup simply skips null-embedding
  entries. Graceful degradation over hard dependency.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger("bridge.feedback")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FeedbackEntry:
    """One turn's worth of preference data.

    ``turn_id`` is the browser-side stream/job id so the feedback API
    can patch the right row. ``embedding`` is the prompt's vector from
    the embedder, used for k-NN retrieval later; it may be None if the
    embedder was unavailable at turn time.
    """
    turn_id: str
    created_at: str
    prompt: str
    tool_chain: list[dict]
    reply: str
    conversation_id: str | None = None
    embedding: list[float] | None = None
    rating: int | None = None   # +1 / -1 / None (unrated)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "FeedbackEntry":
        data = json.loads(line)
        return cls(**data)


def ollama_embed(
    base_url: str,
    text: str,
    model: str = "nomic-embed-text",
    timeout: float = 10.0,
) -> list[float] | None:
    """Fetch an embedding vector from Ollama's ``/api/embeddings``.

    Returns the vector on success or None if the request fails — caller
    is expected to fail-soft (log the turn without an embedding rather
    than raise). Keeps this module importable even when Ollama is down.
    """
    if not text or not text.strip():
        return None
    from bridge._http import make_client

    url = base_url.rstrip("/") + "/api/embeddings"
    try:
        with make_client(timeout=timeout) as client:
            resp = client.post(url, json={"model": model, "prompt": text})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("embedding fetch failed (%s): %s", model, e)
        return None
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        return None
    return [float(x) for x in emb]


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Standard cosine similarity. Returns 0.0 on mismatched/empty inputs.

    We never raise here — feedback k-NN is best-effort; a weird vector
    shouldn't crash a turn.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class FeedbackStore:
    """Thread-safe append-only JSONL store for feedback entries.

    One JSONL file per machine, lives at ``data/feedback/entries.jsonl``.
    The whole file is loaded into memory on construction and re-serialized
    on rating updates — fine up to ~100k turns, a long way off for a
    single-user voice assistant. If we ever cross that threshold we
    swap in SQLite; nothing about the public API needs to change.
    """

    def __init__(
        self,
        path: Path,
        embedder: Callable[[str], list[float] | None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._entries: list[FeedbackEntry] = []
        self._lock = threading.Lock()
        self._load()

    # ---- persistence -----------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._entries.append(FeedbackEntry.from_json(line))
                    except Exception as e:
                        log.warning("skipping unreadable feedback line: %s", e)
        except Exception as e:
            log.warning("couldn't load feedback log (%s) — starting empty", e)

    def _rewrite(self) -> None:
        """Serialize all entries back to disk. Called after rating updates
        and resets. Atomic via tmp-file rename so a crash mid-write can't
        corrupt the log.
        """
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(e.to_json() + "\n")
        tmp.replace(self.path)

    # ---- public API ------------------------------------------------------

    def append(
        self,
        turn_id: str,
        prompt: str,
        tool_chain: list[dict],
        reply: str,
        conversation_id: str | None = None,
    ) -> FeedbackEntry:
        """Record a new turn. Embedding is fetched if an embedder is
        configured; failure is logged but not raised.
        """
        embedding: list[float] | None = None
        if self._embedder is not None and prompt.strip():
            try:
                embedding = self._embedder(prompt)
            except Exception as e:
                log.warning("embedder raised on append: %s", e)

        entry = FeedbackEntry(
            turn_id=turn_id,
            created_at=_now_iso(),
            prompt=prompt,
            tool_chain=list(tool_chain or []),
            reply=reply,
            conversation_id=conversation_id,
            embedding=embedding,
            rating=None,
        )
        with self._lock:
            self._entries.append(entry)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")
        return entry

    def update_rating(self, turn_id: str, rating: int) -> bool:
        """Patch an existing entry's rating. Returns True on success,
        False if no matching ``turn_id`` was found.
        """
        if rating not in (-1, 1):
            raise ValueError(f"rating must be +1 or -1, got {rating!r}")
        with self._lock:
            for e in self._entries:
                if e.turn_id == turn_id:
                    e.rating = rating
                    self._rewrite()
                    return True
            return False

    def get(self, turn_id: str) -> FeedbackEntry | None:
        with self._lock:
            for e in self._entries:
                if e.turn_id == turn_id:
                    return e
        return None

    def all_entries(self) -> list[FeedbackEntry]:
        """Snapshot copy of the entries list. Safe to iterate without
        holding the lock."""
        with self._lock:
            return list(self._entries)

    def knn_similar(
        self,
        embedding: list[float],
        k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[tuple[FeedbackEntry, float]]:
        """Return the top ``k`` entries most similar to ``embedding`` by
        cosine similarity, filtered to entries with a rating and a
        similarity >= ``min_similarity``.

        Null-embedding entries and unrated entries are skipped (they
        have no signal for the bandit). Returns (entry, similarity)
        pairs sorted by similarity descending.
        """
        if not embedding:
            return []
        with self._lock:
            scored: list[tuple[FeedbackEntry, float]] = []
            for e in self._entries:
                if e.embedding is None or e.rating is None:
                    continue
                sim = cosine_similarity(embedding, e.embedding)
                if sim >= min_similarity:
                    scored.append((e, sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def tool_stats(self) -> dict[str, dict[str, float]]:
        """Aggregate per-tool rating stats across all rated entries.

        Returns ``{tool_name: {"count": N, "sum": S, "avg": S/N}}``. Used
        by the debug stats endpoint and any future smarter bandit logic.
        """
        stats: dict[str, dict[str, float]] = {}
        with self._lock:
            for e in self._entries:
                if e.rating is None:
                    continue
                for tc in e.tool_chain:
                    name = tc.get("name") or ""
                    if not name:
                        continue
                    s = stats.setdefault(name, {"count": 0.0, "sum": 0.0, "avg": 0.0})
                    s["count"] += 1
                    s["sum"] += float(e.rating)
                    s["avg"] = s["sum"] / s["count"]
        return stats

    def reset(self) -> None:
        """Wipe the log — both in-memory and on disk."""
        with self._lock:
            self._entries.clear()
            if self.path.exists():
                self.path.unlink()
