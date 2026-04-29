"""Persistent conversation history + token-budget compaction.

Stores each conversation as a JSON file under ``data/conversations/`` and
keeps a lightweight index so the server can resume the most recent
conversation on boot. Compaction is driven by the target LLM's
``num_ctx`` so the policy scales with whatever model you swap in.

This module owns two related pieces:

* :class:`ConversationStore` — disk persistence (write after every turn,
  load on boot) and budget enforcement (summarize the oldest half when
  the list outgrows the context window).
* :class:`HistorySession` — the in-memory rolling list of messages plus
  the ``threading.RLock`` that guards every mutation. Both the async
  request paths (via ``run_in_threadpool``) and the worker-thread chat
  loop acquire the same lock, which prevents the torn-list races a
  separate ``asyncio.Lock`` and ``threading.Lock`` would otherwise
  permit.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from bridge.utils import REPO_ROOT, atomic_write_text

log = logging.getLogger("bridge.history")


class HistorySession:
    """Thread-safe owner of the rolling conversation list.

    Wraps a ``list[dict]`` of chat messages with a ``threading.RLock``.
    All mutations go through ``append`` / ``extend`` / ``replace`` /
    ``clear``; reads go through ``snapshot`` (defensive copy) plus the
    cheap ``len`` / ``bool`` / iteration / indexing protocols.

    Why this exists: previously ``OllamaLLM._history`` was a raw list,
    mutated from both the chat loop (worker thread) and the FastAPI
    routes (async via ``run_in_threadpool``). The two outer locks
    (``asyncio.Lock`` for async routes, ``threading.Lock`` for jobs)
    didn't intersect, so a DELETE racing with an in-flight chat could
    corrupt the list. Routing every mutation through one ``RLock``
    here removes that class of bug without forcing the route layer to
    serialize everything globally.

    The lock is re-entrant so a caller holding it can call multiple
    methods nested without self-deadlock (e.g. snapshot-then-replace
    under a single critical section via the ``lock`` property).
    """

    __slots__ = ("_messages", "_lock")

    def __init__(self, initial: Iterable[dict] = ()) -> None:
        self._messages: list[dict] = list(initial)
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        """Expose the internal lock for callers that need to bracket
        multiple ops atomically — e.g. compaction reads then conditional
        replace. Use sparingly; the per-method locking covers the common
        cases. """
        return self._lock

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self) -> Iterator[dict]:
        # Iterate over a snapshot so callers can't observe a
        # mid-mutation tear and don't keep the lock for the duration of
        # whatever they're doing inside the loop body.
        return iter(self.snapshot())

    def __getitem__(self, idx):
        with self._lock:
            return self._messages[idx]

    def __eq__(self, other) -> bool:
        # Compare by content so tests and callers can write
        # `session == []` or `session == [{"role": "user", ...}]` and
        # get the same semantics they would for a plain list.
        if isinstance(other, HistorySession):
            return self.snapshot() == other.snapshot()
        if isinstance(other, list):
            return self.snapshot() == other
        return NotImplemented

    # Mutable container — explicitly opt out of hashing rather than
    # inheriting the default (which would change identity-based hashing
    # to raise once __eq__ is defined; setting None makes the intent
    # explicit and the error message clearer).
    __hash__ = None  # type: ignore[assignment]

    def snapshot(self) -> list[dict]:
        """Return a shallow defensive copy of the current message list."""
        with self._lock:
            return list(self._messages)

    def append(self, message: dict) -> None:
        with self._lock:
            self._messages.append(message)

    def extend(self, messages: Iterable[dict]) -> None:
        # Materialize the input outside the lock in case it's a
        # generator that does I/O — keeps the critical section short.
        msgs = list(messages)
        with self._lock:
            self._messages.extend(msgs)

    def replace(self, messages: Iterable[dict]) -> None:
        msgs = list(messages)
        with self._lock:
            self._messages = msgs

    def clear(self) -> None:
        with self._lock:
            self._messages = []

# Phase H: conversations live under the active profile. The legacy
# data/conversations/ location is kept as a constant for the migration
# runner to find (see bridge/profiles/migration.py) and for tests that
# want a stable default — passing a custom root to ConversationStore()
# still takes precedence. Resolving at call time (not import time)
# means the active profile can be chosen by the time we actually write.
LEGACY_CONVERSATIONS_DIR: Path = REPO_ROOT / "data" / "conversations"


def _default_conversations_dir() -> Path:
    from bridge.profiles import active_profile
    return active_profile().conversations_dir

# Compaction thresholds, expressed as a fraction of num_ctx. 60% trigger
# gives the next turn + tool output + reply ~40% of the window to work
# with, which is enough at num_ctx=4096 and scales linearly upward.
# Compact more aggressively for small models (2B–4B): they degrade
# meaningfully with longer context, so keeping more headroom helps
# tool selection accuracy. 0.50 trigger leaves 50% of num_ctx for the
# new turn + tools + reply.
_COMPACT_TRIGGER_RATIO: float = 0.50
# After compaction, the kept-history + summary should fit in this
# fraction of num_ctx so compaction doesn't re-trigger immediately.
_COMPACT_TARGET_RATIO: float = 0.30

# Rough English tokens-per-char. Real tokenizers vary by 10-20% but
# we're computing a budget, not a precise count — close enough.
_CHARS_PER_TOKEN: float = 4.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Timestamp-based id so ``ls`` naturally sorts chronologically.

    Plus a 4-hex-char random tail so two calls in the same wall-clock
    second produce unique IDs (Windows' system clock has ~15 ms
    resolution; even microseconds collide on rapid creation).
    """
    import secrets
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"_{secrets.token_hex(2)}Z"


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token count across a message list. English-tuned."""
    total_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        # name / role / tool_calls etc. are tiny — ignore.
    return int(total_chars / _CHARS_PER_TOKEN)


class ConversationStore:
    """Manages one current conversation + an index of past ones."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else _default_conversations_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        # Index file lives inside `root` so passing a custom root
        # (tests, multiple stores) doesn't bleed into the global
        # singleton's index.
        self._index_path = self.root / "index.json"
        self._index = self._read_index()

    # ---- index helpers ----------------------------------------------------

    def _read_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("conversation index unreadable (%s) — rebuilding", e)
        return {"current": None, "conversations": []}

    def _write_index(self) -> None:
        atomic_write_text(
            self._index_path,
            json.dumps(self._index, indent=2, ensure_ascii=False),
        )

    def _conv_path(self, cid: str) -> Path:
        return self.root / f"{cid}.json"

    # ---- public API -------------------------------------------------------

    def load_current_or_new(self) -> tuple[str, list[dict]]:
        """Return (id, messages) for the current conversation.

        If there's a "current" entry in the index, load its messages.
        Otherwise create a fresh conversation and return it empty.
        """
        cid = self._index.get("current")
        if cid and self._conv_path(cid).exists():
            try:
                data = json.loads(self._conv_path(cid).read_text(encoding="utf-8"))
                return cid, data.get("messages", [])
            except Exception as e:
                log.warning("failed to load conversation %s (%s) — starting fresh", cid, e)
        return self.new_conversation()

    def save(self, cid: str, messages: list[dict]) -> None:
        """Write the current conversation's messages to disk."""
        preview = ""
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                preview = m["content"][:80]
                break

        # Preserve existing turn_notes when re-saving (compaction
        # rewrites messages but shouldn't discard notes).
        existing_notes: list[dict] = []
        path = self._conv_path(cid)
        if path.exists():
            try:
                existing_notes = json.loads(
                    path.read_text(encoding="utf-8")
                ).get("turn_notes", [])
            except Exception:
                pass

        payload = {
            "id": cid,
            "updated_at": _now_iso(),
            "message_count": len(messages),
            "messages": messages,
            "turn_notes": existing_notes,
        }
        atomic_write_text(
            self._conv_path(cid),
            json.dumps(payload, indent=2, ensure_ascii=False),
        )

        for entry in self._index.get("conversations", []):
            if entry.get("id") == cid:
                entry["updated_at"] = payload["updated_at"]
                entry["message_count"] = payload["message_count"]
                if preview and not entry.get("preview"):
                    entry["preview"] = preview
                break
        self._write_index()

    def new_conversation(self) -> tuple[str, list[dict]]:
        """Archive current (if any) and start a fresh conversation."""
        cid = _new_id()
        created_at = _now_iso()
        entry = {
            "id": cid,
            "created_at": created_at,
            "updated_at": created_at,
            "message_count": 0,
            "preview": "",
        }
        self._index.setdefault("conversations", []).insert(0, entry)
        self._index["current"] = cid
        atomic_write_text(
            self._conv_path(cid),
            json.dumps(
                {"id": cid, "updated_at": created_at, "message_count": 0, "messages": []},
                indent=2, ensure_ascii=False,
            ),
        )
        self._write_index()
        log.info("started new conversation %s", cid)
        return cid, []

    def current_id(self) -> str | None:
        return self._index.get("current")

    def list_conversations(self) -> list[dict]:
        """Return the conversation index (most-recent first).

        Each entry has ``id``, ``created_at``, ``updated_at``,
        ``message_count``, ``preview``. Safe to serialize as JSON.
        """
        return list(self._index.get("conversations", []))

    def load_conversation(self, cid: str) -> list[dict] | None:
        """Load a specific conversation's messages by id, or None."""
        path = self._conv_path(cid)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("messages", [])
        except Exception as e:
            log.warning("failed to load conversation %s: %s", cid, e)
            return None

    def switch_to(self, cid: str) -> bool:
        """Make ``cid`` the current conversation. Returns True on success."""
        if not self._conv_path(cid).exists():
            return False
        self._index["current"] = cid
        self._write_index()
        return True

    def append_turn_note(
        self,
        cid: str,
        user_text: str,
        tools_called: list[str],
        routing_hint: str | None,
        reply_preview: str,
        audio_id: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Append a turn-level flow note to the conversation.

        These notes are observability metadata — they record what
        happened on each turn (which tools fired, what the routing
        classifier guessed, a short preview of the user prompt and
        the assistant reply). They are NOT injected into the LLM
        context; they're for debugging and UI display only.

        When ``audio_id`` + ``duration_ms`` are supplied, they pin this
        turn to a cached WAV under ``data/audio_cache/<cid>/<audio_id>``
        so the browser can replay after a refresh. The cache is tmpfs
        in prod — a missing file after restart is recoverable (the
        frontend degrades to "replay unavailable").
        """
        note = {
            "timestamp": _now_iso(),
            "user": (user_text or "")[:120],
            "tools": tools_called or [],
            "routing_hint": routing_hint,
            "reply": (reply_preview or "")[:120],
        }
        if audio_id:
            note["audio_id"] = audio_id
        if duration_ms is not None:
            note["duration_ms"] = int(duration_ms)
        path = self._conv_path(cid)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        notes = data.get("turn_notes", [])
        notes.append(note)
        data["turn_notes"] = notes
        atomic_write_text(
            path,
            json.dumps(data, indent=2, ensure_ascii=False),
        )

    def get_turn_notes(self, cid: str) -> list[dict]:
        """Return turn notes for a conversation, or empty list."""
        path = self._conv_path(cid)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("turn_notes", [])
        except Exception:
            return []

    def delete_conversation(self, cid: str) -> str | None:
        """Delete a conversation file and its index entry.

        Cascades: any cached TTS audio for this conversation is removed
        from the audio cache too, since audio outlives neither the
        conversation nor the user's intent to keep it.

        Returns the id of the new "current" conversation after deletion:
        unchanged if ``cid`` wasn't current, otherwise the most recent
        remaining id (or a freshly created one if the list is now empty).
        Returns None if ``cid`` didn't exist.
        """
        path = self._conv_path(cid)
        if not path.exists() and not any(c.get("id") == cid for c in self._index.get("conversations", [])):
            return None
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            log.warning("failed to unlink %s: %s", path, e)

        # Cascade: drop any cached TTS audio for this conversation.
        # Import inline to avoid a top-level cycle (audio_cache may
        # import from bridge.utils which is already loaded here).
        try:
            from bridge.audio_cache import delete_conversation_audio
            delete_conversation_audio(cid)
        except Exception as e:
            log.debug("audio cascade-delete failed (non-fatal): %s", e)

        self._index["conversations"] = [
            c for c in self._index.get("conversations", []) if c.get("id") != cid
        ]

        if self._index.get("current") != cid:
            self._write_index()
            return self._index.get("current")

        # We deleted the current conversation — pick the next most recent,
        # or start a fresh one if nothing remains.
        if self._index["conversations"]:
            self._index["current"] = self._index["conversations"][0]["id"]
            self._write_index()
            return self._index["current"]

        # All gone — start a fresh conversation so the app always has one.
        new_cid, _ = self.new_conversation()
        return new_cid


# ---- compaction -----------------------------------------------------------

_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a concise note-taker. Summarize the conversation snippet "
    "below in 2-4 short sentences. Preserve concrete facts the user "
    "mentioned (names, places, preferences), any decisions reached, and "
    "open questions. Do not use first or second person — write as "
    "neutral notes. Output only the summary text, nothing else."
)


def maybe_compact(
    history: list[dict],
    num_ctx: int,
    summarize_fn: Callable[[list[dict], str], str],
    trigger_ratio: float = _COMPACT_TRIGGER_RATIO,
    target_ratio: float = _COMPACT_TARGET_RATIO,
) -> list[dict]:
    """Return a (possibly shorter) history, summarizing old turns if over budget.

    Args:
        history: The LLM's current rolling history (no system prompt).
        num_ctx: The model's context window in tokens.
        summarize_fn: Callback (messages, system_prompt) -> summary str.
            Kept as a callback so this module doesn't import the LLM.
        trigger_ratio: Compact when token estimate exceeds this * num_ctx.
        target_ratio: Keep only this * num_ctx worth of recent turns.

    Returns:
        The (possibly compacted) history. Callers should replace the LLM's
        ``_history`` with the return value.

    Strategy:
        - Estimate tokens; if below trigger, return unchanged.
        - Split into an "old" prefix (to summarize) and a "keep" suffix
          (recent messages fitting within target budget).
        - Call summarize_fn on old prefix; prepend a single synthetic
          ``role=system`` message containing the summary.
        - Never split a tool_calls / tool-result pair across the boundary —
          keep them together on the recent side if needed.
    """
    tokens = _estimate_tokens(history)
    trigger = int(trigger_ratio * num_ctx)
    if tokens <= trigger:
        return history

    log.info(
        "compaction triggered: %d tokens >= %d (trigger). Target ≤%d.",
        tokens, trigger, int(target_ratio * num_ctx),
    )

    target_budget = int(target_ratio * num_ctx)

    # Walk from the end, keeping messages until we fill the target budget.
    keep_idx = len(history)
    kept_tokens = 0
    for i in range(len(history) - 1, -1, -1):
        msg_tokens = _estimate_tokens([history[i]])
        if kept_tokens + msg_tokens > target_budget and keep_idx < len(history):
            break
        kept_tokens += msg_tokens
        keep_idx = i

    # Don't split tool-call / tool-result pairs. If the first kept message
    # is a tool result, slide the boundary earlier to include its call.
    while keep_idx < len(history) and history[keep_idx].get("role") == "tool":
        keep_idx -= 1
    if keep_idx <= 0:
        # Would summarize nothing — history is already short or all-recent.
        return history

    old = history[:keep_idx]
    recent = history[keep_idx:]

    try:
        summary = summarize_fn(old, _SUMMARIZE_SYSTEM_PROMPT).strip()
    except Exception as e:
        log.warning("summarization failed (%s) — leaving history uncompacted", e)
        return history

    if not summary:
        log.warning("summarizer returned empty — leaving history uncompacted")
        return history

    summary_msg = {
        "role": "system",
        "content": f"[Prior conversation summary]\n{summary}",
    }
    compacted = [summary_msg, *recent]
    log.info(
        "compacted %d old messages -> 1 summary (%d chars). New history: %d messages, ~%d tokens.",
        len(old), len(summary), len(compacted), _estimate_tokens(compacted),
    )
    return compacted
