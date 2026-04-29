"""Tests for bridge.history — conversation persistence + token-budget compaction.

Covers:
  - new_conversation / load_current_or_new / save round-trip
  - list_conversations ordering + metadata
  - switch_to + delete_conversation (+ the 'current' pointer cycling)
  - maybe_compact: no-op below budget, summarize old half above trigger
  - Preview extraction picks the first user message
"""
from __future__ import annotations

import pytest

from bridge.history import (
    ConversationStore,
    _estimate_tokens,
    maybe_compact,
)


@pytest.fixture
def store(tmp_path):
    """Isolated store backed by a tmp dir; no shared state between tests."""
    return ConversationStore(root=tmp_path)


# --- basic lifecycle --------------------------------------------------------

class TestNewAndLoad:
    def test_load_current_or_new_on_empty(self, store):
        cid, messages = store.load_current_or_new()
        assert cid is not None and len(cid) > 0
        assert messages == []
        # The new id is marked current.
        assert store.current_id() == cid

    def test_new_conversation_bumps_current(self, store):
        cid1, _ = store.load_current_or_new()
        cid2, _ = store.new_conversation()
        assert cid2 != cid1
        assert store.current_id() == cid2

    def test_save_and_load_round_trip(self, store):
        cid, _ = store.load_current_or_new()
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        store.save(cid, msgs)
        loaded = store.load_conversation(cid)
        assert loaded == msgs


# --- list + metadata --------------------------------------------------------

class TestList:
    def test_list_newest_first(self, store):
        cid1, _ = store.new_conversation()
        cid2, _ = store.new_conversation()
        cid3, _ = store.new_conversation()
        ids = [c["id"] for c in store.list_conversations()]
        assert ids.index(cid3) < ids.index(cid2) < ids.index(cid1)

    def test_list_includes_preview_from_first_user_msg(self, store):
        cid, _ = store.load_current_or_new()
        store.save(cid, [
            {"role": "system", "content": "[prior]"},
            {"role": "user", "content": "what's the weather"},
            {"role": "assistant", "content": "partly cloudy"},
        ])
        entry = next(c for c in store.list_conversations() if c["id"] == cid)
        assert entry["preview"].startswith("what's the weather")
        assert entry["message_count"] == 3


# --- switch + delete --------------------------------------------------------

class TestSwitchDelete:
    def test_switch_to_existing(self, store):
        cid1, _ = store.new_conversation()
        cid2, _ = store.new_conversation()
        assert store.current_id() == cid2
        assert store.switch_to(cid1) is True
        assert store.current_id() == cid1

    def test_switch_to_unknown_returns_false(self, store):
        assert store.switch_to("bogus_id") is False

    def test_delete_non_current(self, store):
        cid1, _ = store.new_conversation()
        cid2, _ = store.new_conversation()   # becomes current
        new_current = store.delete_conversation(cid1)
        # Current unchanged since we deleted the non-current one.
        assert new_current == cid2
        # cid1 is gone from the list.
        ids = [c["id"] for c in store.list_conversations()]
        assert cid1 not in ids

    def test_delete_current_switches_to_next_recent(self, store):
        cid1, _ = store.new_conversation()
        cid2, _ = store.new_conversation()   # current
        # Delete current → should switch back to cid1.
        new_current = store.delete_conversation(cid2)
        assert new_current == cid1
        assert store.current_id() == cid1

    def test_delete_last_conversation_creates_fresh(self, store):
        cid, _ = store.load_current_or_new()
        new_current = store.delete_conversation(cid)
        # A fresh conversation was auto-created.
        assert new_current is not None
        assert new_current != cid
        assert len(store.list_conversations()) == 1

    def test_delete_unknown_returns_none(self, store):
        store.new_conversation()
        assert store.delete_conversation("nope") is None


# --- compaction -------------------------------------------------------------

class TestCompaction:
    def test_no_op_below_trigger(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        # num_ctx big enough that the trigger is way above current tokens.
        result = maybe_compact(
            history, num_ctx=4096,
            summarize_fn=lambda msgs, sp: "SHOULD NOT BE CALLED",
        )
        assert result == history   # unchanged

    def test_compacts_when_above_trigger(self):
        # Build a history big enough to cross the default 50% trigger.
        # Each message ~400 chars; tokens ≈ chars/4 so 25 msgs ≈ 2500 tokens;
        # num_ctx 2048 * 0.5 = 1024 trigger, so 2500 tokens >> trigger.
        history = []
        for i in range(25):
            history.append({"role": "user", "content": "x" * 400})
            history.append({"role": "assistant", "content": "y" * 400})
        called = []
        def fake_summarize(msgs, sys_prompt):
            called.append((len(msgs), sys_prompt))
            return "condensed summary"
        result = maybe_compact(
            history, num_ctx=2048, summarize_fn=fake_summarize,
        )
        assert called, "summarizer should have been invoked"
        # First message of result should be the synthetic summary.
        assert result[0]["role"] == "system"
        assert "[Prior conversation summary]" in result[0]["content"]
        assert "condensed summary" in result[0]["content"]
        # Compacted history is shorter than original.
        assert len(result) < len(history)

    def test_summarizer_exception_leaves_history(self):
        history = [{"role": "user", "content": "x" * 50000}]
        def bad(msgs, sp):
            raise RuntimeError("LLM fell over")
        result = maybe_compact(
            history, num_ctx=1024, summarize_fn=bad,
        )
        # Fall-through policy: don't lose the history on summarize failure.
        assert result == history

    def test_empty_summary_also_falls_back(self):
        history = [{"role": "user", "content": "x" * 50000}]
        result = maybe_compact(
            history, num_ctx=1024, summarize_fn=lambda msgs, sp: "",
        )
        assert result == history


# --- token estimator --------------------------------------------------------

class TestTokenEstimate:
    def test_chars_over_4(self):
        # 40 chars in content ≈ 10 tokens.
        msgs = [{"role": "user", "content": "x" * 40}]
        assert _estimate_tokens(msgs) == 10

    def test_empty(self):
        assert _estimate_tokens([]) == 0

    def test_non_string_content_ignored(self):
        # Tool-call messages have dict/list content; estimator ignores them.
        msgs = [
            {"role": "user", "content": "abcd"},
            {"role": "tool", "content": None},
            {"role": "assistant", "content": ""},
        ]
        assert _estimate_tokens(msgs) == 1


# --- turn notes ---------------------------------------------------------------

class TestTurnNotes:
    """Tests for append_turn_note / get_turn_notes (Phase C)."""

    def test_append_and_retrieve(self, store):
        """Basic round-trip: append a note, then retrieve it."""
        cid, _ = store.load_current_or_new()
        store.append_turn_note(
            cid, "what's the weather",
            ["get_weather"], "get_weather", "Partly cloudy",
        )
        notes = store.get_turn_notes(cid)
        assert len(notes) == 1
        assert notes[0]["tools"] == ["get_weather"]
        assert notes[0]["routing_hint"] == "get_weather"
        assert "timestamp" in notes[0]

    def test_multiple_notes_in_order(self, store):
        """Multiple notes accumulate in chronological order."""
        cid, _ = store.load_current_or_new()
        store.append_turn_note(cid, "time?", ["get_time"], "get_time", "Friday 3pm")
        store.append_turn_note(cid, "weather?", ["get_weather"], "get_weather", "Sunny")
        notes = store.get_turn_notes(cid)
        assert len(notes) == 2
        assert notes[0]["user"].startswith("time")
        assert notes[1]["user"].startswith("weather")

    def test_empty_conversation_has_no_notes(self, store):
        """A fresh conversation has an empty notes list."""
        cid, _ = store.load_current_or_new()
        assert store.get_turn_notes(cid) == []

    def test_nonexistent_conversation_returns_empty(self, store):
        """Querying notes for a non-existent cid returns [], not an error."""
        assert store.get_turn_notes("nonexistent_id_xyz") == []

    def test_notes_survive_save(self, store):
        """save() preserves existing turn_notes (doesn't overwrite them)."""
        cid, _ = store.load_current_or_new()
        store.append_turn_note(cid, "hi", [], None, "hello")
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        store.save(cid, msgs)
        notes = store.get_turn_notes(cid)
        assert len(notes) == 1
        assert notes[0]["user"] == "hi"

    def test_notes_survive_multiple_saves(self, store):
        """Two save() calls don't erase notes appended between them."""
        cid, _ = store.load_current_or_new()
        store.append_turn_note(cid, "first", ["get_time"], "get_time", "3pm")
        store.save(cid, [{"role": "user", "content": "first"}])
        store.append_turn_note(cid, "second", ["wiki"], "wiki", "Einstein")
        store.save(cid, [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ])
        notes = store.get_turn_notes(cid)
        assert len(notes) == 2

    def test_truncation_120_chars(self, store):
        """Long user_text and reply_preview are truncated to 120 chars."""
        cid, _ = store.load_current_or_new()
        long_text = "a" * 200
        store.append_turn_note(cid, long_text, [], None, long_text)
        notes = store.get_turn_notes(cid)
        assert len(notes[0]["user"]) == 120
        assert len(notes[0]["reply"]) == 120
