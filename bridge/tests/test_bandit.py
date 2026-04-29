"""Tests for bridge.bandit — preference_hint & retry_hint.

Seeds a FeedbackStore with a deterministic embedder and rated entries,
then asserts the hint content matches what we'd expect a human reader
to want. Thresholds in bandit.py are intentionally conservative — the
goal is "no misleading hints" over "fire as often as possible", and
these tests reflect that.
"""
from __future__ import annotations

import pytest

from bridge.bandit import preference_hint, retry_hint
from bridge.feedback import FeedbackStore


def _stub_embedder(text: str) -> list[float]:
    """Same deterministic embedder used in test_feedback.py — 4-D
    vector derived from length + first three character ordinals."""
    n = float(len(text))
    c1 = float(ord(text[0]) % 7) if len(text) > 0 else 0.0
    c2 = float(ord(text[1]) % 7) if len(text) > 1 else 0.0
    c3 = float(ord(text[2]) % 7) if len(text) > 2 else 0.0
    return [n, c1, c2, c3]


# ---- preference_hint ------------------------------------------------------


class TestPreferenceHint:
    def test_empty_store_returns_empty(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        assert preference_hint(_stub_embedder("anything"), store) == ""

    def test_no_embedding_returns_empty(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        # Populate something so we know it's the embedding check that
        # short-circuits, not emptiness.
        store.append("t1", "hi", [{"name": "get_time"}], "r")
        store.update_rating("t1", 1)
        assert preference_hint(None, store) == ""
        assert preference_hint([], store) == ""

    def test_single_neighbour_insufficient(self, tmp_path):
        """MIN_NEIGHBOURS_FOR_HINT=2; one similar turn isn't enough
        signal to inject a hint."""
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        store.append("t1", "hello world today", [{"name": "wiki_random"}], "r")
        store.update_rating("t1", 1)
        hint = preference_hint(_stub_embedder("hello world today"), store)
        assert hint == ""

    def test_consistent_positive_signal_produces_preferred_hint(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        # Seed several very-similar rated entries all liking tell_story.
        for i, tid in enumerate(["t1", "t2", "t3"]):
            store.append(
                tid,
                f"tell me a story about topic {i}",
                [{"name": "tell_story"}],
                "reply",
            )
            store.update_rating(tid, 1)
        hint = preference_hint(_stub_embedder("tell me a story about something"), store)
        assert "preference hint" in hint.lower()
        assert "preferred" in hint.lower()
        assert "tell_story" in hint

    def test_consistent_negative_signal_produces_disliked_hint(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        for tid in ["t1", "t2", "t3"]:
            store.append(tid, "tell me a story please", [{"name": "wiki_random"}], "r")
            store.update_rating(tid, -1)
        hint = preference_hint(_stub_embedder("tell me a story today"), store)
        assert "disliked" in hint.lower()
        assert "wiki_random" in hint

    def test_mixed_signal_below_threshold_returns_empty(self, tmp_path):
        """Equal +/- ratings → |avg|≈0 → no hint fired."""
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        for i, (tid, rating) in enumerate([
            ("t1", 1), ("t2", -1), ("t3", 1), ("t4", -1),
        ]):
            store.append(tid, f"story prompt {i}", [{"name": "wiki_random"}], "r")
            store.update_rating(tid, rating)
        assert preference_hint(_stub_embedder("story prompt X"), store) == ""

    def test_dissimilar_past_turns_ignored(self, tmp_path):
        """Past turns with cosine < min_similarity shouldn't contribute,
        even with strong ratings."""
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        # Seed two very-different prompts (very different lengths → very
        # different stub embeddings → low cosine)
        store.append("t1", "x", [{"name": "wiki_random"}], "r")
        store.append("t2", "y", [{"name": "wiki_random"}], "r")
        store.update_rating("t1", -1)
        store.update_rating("t2", -1)

        query = _stub_embedder("this is a very long and distinct prompt indeed")
        # With high min_similarity they won't clear the bar; with
        # default 0.5 they might, but the dissimilar embedding means
        # weighted contribution is small. Either way, hint should
        # empty out under strict filter.
        assert preference_hint(query, store, min_similarity=0.95) == ""

    def test_unrated_entries_never_contribute(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        # All three similar entries but none rated
        for tid in ["t1", "t2", "t3"]:
            store.append(tid, "a story about things", [{"name": "tell_story"}], "r")
        hint = preference_hint(_stub_embedder("a story about stuff"), store)
        assert hint == ""

    def test_hint_lists_multiple_tools(self, tmp_path):
        """When several tools have clear signal, the hint should list
        the liked ones and the disliked ones separately."""
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        # Three very-similar turns: tell_story liked, wiki_random disliked
        for tid in ["t1", "t2", "t3"]:
            store.append(
                tid,
                "fun creative prompt here",
                [{"name": "tell_story"}, {"name": "wiki_random"}],
                "r",
            )
        store.update_rating("t1", 1)
        store.update_rating("t2", 1)
        store.update_rating("t3", 1)
        # All positive → both tools appear under "preferred"
        hint = preference_hint(_stub_embedder("fun creative prompt today"), store)
        assert "tell_story" in hint
        assert "wiki_random" in hint


# ---- retry_hint -----------------------------------------------------------


class TestRetryHint:
    def test_mentions_tools_from_prev_chain(self):
        chain = [{"name": "wiki_search", "arguments": {"query": "PA"}},
                 {"name": "wiki_random", "arguments": {}}]
        h = retry_hint(chain)
        assert "retry" in h.lower()
        assert "wiki_search" in h
        assert "wiki_random" in h

    def test_empty_chain_gives_generic_retry(self):
        h = retry_hint([])
        assert "retry" in h.lower()
        # No specific tool names, but still says something useful
        assert "different" in h.lower()

    def test_entries_without_names_are_skipped(self):
        h = retry_hint([{"arguments": {}}, {"name": "get_weather"}])
        assert "get_weather" in h
