"""Tests for bridge.memory — user + agent memory files.

Covers:
  - Fresh install (no files yet)
  - Normal round-trip (set → get)
  - Truncation at MAX_MEMORY_CHARS
  - UTF-8 round-trip (Unicode content survives write/read)
  - build_prompt_block composition (empty / one-sided / both)
"""
from __future__ import annotations

import pytest

from bridge.memory import MAX_MEMORY_CHARS, MemoryStore


@pytest.fixture
def store(tmp_path):
    """Fresh MemoryStore backed by tmp files so tests don't touch real data/."""
    return MemoryStore(
        user_path=tmp_path / "user.md",
        agent_path=tmp_path / "agent.md",
    )


class TestFreshInstall:
    def test_get_empty_when_files_missing(self, store):
        assert store.get_user() == ""
        assert store.get_agent() == ""

    def test_build_prompt_block_empty_when_both_unset(self, store):
        assert store.build_prompt_block() == ""


class TestRoundTrip:
    def test_set_and_get_user(self, store):
        text, truncated = store.set_user("I live in Philadelphia.")
        assert truncated is False
        assert text == "I live in Philadelphia."
        assert store.get_user() == "I live in Philadelphia."

    def test_set_and_get_agent(self, store):
        text, truncated = store.set_agent("Keep replies short.")
        assert truncated is False
        assert store.get_agent() == "Keep replies short."

    def test_user_and_agent_are_independent(self, store):
        store.set_user("A")
        store.set_agent("B")
        assert store.get_user() == "A"
        assert store.get_agent() == "B"

    def test_overwrite_replaces(self, store):
        store.set_user("first")
        store.set_user("second")
        assert store.get_user() == "second"

    def test_strips_surrounding_whitespace(self, store):
        text, _ = store.set_user("  padded text  \n")
        assert text == "padded text"
        assert store.get_user() == "padded text"

    def test_empty_write_produces_empty_read(self, store):
        store.set_user("something")
        store.set_user("")
        assert store.get_user() == ""


class TestTruncation:
    def test_truncates_at_cap(self, store):
        long_text = "x" * (MAX_MEMORY_CHARS + 100)
        saved, truncated = store.set_user(long_text)
        assert truncated is True
        assert len(saved) <= MAX_MEMORY_CHARS
        assert store.get_user() == saved

    def test_under_cap_not_truncated(self, store):
        under = "a" * (MAX_MEMORY_CHARS - 1)
        saved, truncated = store.set_user(under)
        assert truncated is False
        assert saved == under


class TestUnicode:
    def test_unicode_round_trip_user(self, store):
        content = "I love Kärin 🎸 — bass player from Otaru 小樽"
        store.set_user(content)
        assert store.get_user() == content

    def test_unicode_agent(self, store):
        content = "仕事のメモ — keep it casual"
        store.set_agent(content)
        assert store.get_agent() == content


class TestPromptBlock:
    def test_user_only(self, store):
        store.set_user("I live in Philly.")
        block = store.build_prompt_block()
        assert "[About the user]" in block
        assert "I live in Philly." in block
        assert "[Your additional instructions]" not in block

    def test_agent_only(self, store):
        store.set_agent("Always say please.")
        block = store.build_prompt_block()
        assert "[Your additional instructions]" in block
        assert "Always say please." in block
        assert "[About the user]" not in block

    def test_both_with_separator(self, store):
        store.set_user("U facts")
        store.set_agent("A rules")
        block = store.build_prompt_block()
        assert block.startswith("[About the user]")
        assert "[Your additional instructions]" in block
        # Verifies the two blocks are separated by a blank line.
        assert "U facts\n\n[Your additional instructions]" in block

    def test_empty_returns_empty_string(self, store):
        assert store.build_prompt_block() == ""
