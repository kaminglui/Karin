"""Tests for bridge.feedback — storage, retrieval, rating updates.

The Ollama ``ollama_embed`` helper isn't covered here (it hits the
network); it's exercised in a live smoke test on the Jetson instead.
All the store / cosine / k-NN logic is pure Python and easy to pin.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bridge.feedback import (
    FeedbackEntry,
    FeedbackStore,
    cosine_similarity,
)


# ---- cosine_similarity ----------------------------------------------------


class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_mismatched_lengths_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_empty_or_none(self):
        assert cosine_similarity(None, [1.0]) == 0.0
        assert cosine_similarity([1.0], None) == 0.0
        assert cosine_similarity([], []) == 0.0

    def test_zero_vector(self):
        # Degenerate but shouldn't divide by zero
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---- FeedbackStore persistence + lifecycle --------------------------------


def _stub_embedder(text: str) -> list[float]:
    """Deterministic stand-in for the real embedder. Produces a 4-D
    vector where the first component reflects prompt length (gives us
    distinguishable neighbours) and the rest encode the first three
    character ordinals mod 7 for a bit of variety."""
    n = float(len(text))
    c1 = float(ord(text[0]) % 7) if len(text) > 0 else 0.0
    c2 = float(ord(text[1]) % 7) if len(text) > 1 else 0.0
    c3 = float(ord(text[2]) % 7) if len(text) > 2 else 0.0
    return [n, c1, c2, c3]


class TestFeedbackStoreLifecycle:
    def test_append_creates_file_and_persists(self, tmp_path):
        store = FeedbackStore(tmp_path / "entries.jsonl", embedder=_stub_embedder)
        e = store.append("t1", "hello", [{"name": "get_time", "arguments": {}}], "reply here")
        assert e.turn_id == "t1"
        assert e.rating is None
        assert e.embedding is not None
        # File written
        assert (tmp_path / "entries.jsonl").exists()

    def test_load_reads_prior_entries(self, tmp_path):
        p = tmp_path / "entries.jsonl"
        s1 = FeedbackStore(p, embedder=_stub_embedder)
        s1.append("t1", "hi", [], "reply1")
        s1.append("t2", "bye", [], "reply2")
        # Reopen
        s2 = FeedbackStore(p, embedder=_stub_embedder)
        ids = [e.turn_id for e in s2.all_entries()]
        assert ids == ["t1", "t2"]

    def test_update_rating_patches_entry(self, tmp_path):
        store = FeedbackStore(tmp_path / "entries.jsonl", embedder=_stub_embedder)
        store.append("t1", "hello", [], "reply")
        assert store.update_rating("t1", 1) is True
        assert store.get("t1").rating == 1
        # Persistence: reopen and check
        reopened = FeedbackStore(tmp_path / "entries.jsonl", embedder=_stub_embedder)
        assert reopened.get("t1").rating == 1

    def test_update_rating_unknown_turn_returns_false(self, tmp_path):
        store = FeedbackStore(tmp_path / "entries.jsonl", embedder=_stub_embedder)
        assert store.update_rating("nope", 1) is False

    def test_update_rating_rejects_invalid_value(self, tmp_path):
        store = FeedbackStore(tmp_path / "entries.jsonl", embedder=_stub_embedder)
        store.append("t1", "hello", [], "r")
        with pytest.raises(ValueError):
            store.update_rating("t1", 0)
        with pytest.raises(ValueError):
            store.update_rating("t1", 2)

    def test_reset_clears_disk_and_memory(self, tmp_path):
        p = tmp_path / "entries.jsonl"
        store = FeedbackStore(p, embedder=_stub_embedder)
        store.append("t1", "hi", [], "r")
        store.reset()
        assert store.all_entries() == []
        assert not p.exists()

    def test_embedder_none_still_appends(self, tmp_path):
        """If no embedder configured, we still record the turn —
        rating-based stats still work, only k-NN retrieval skips it."""
        store = FeedbackStore(tmp_path / "entries.jsonl", embedder=None)
        e = store.append("t1", "hi", [], "r")
        assert e.embedding is None

    def test_embedder_raising_is_swallowed(self, tmp_path):
        def angry(_text):
            raise RuntimeError("embedding server down")

        store = FeedbackStore(tmp_path / "entries.jsonl", embedder=angry)
        e = store.append("t1", "hi", [], "r")
        assert e.embedding is None
        assert e.turn_id == "t1"


# ---- k-NN retrieval -------------------------------------------------------


class TestKnn:
    def test_knn_returns_rated_nearest_neighbours(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        # Seed three rated entries with varying prompts
        store.append("t1", "hello world", [], "r1")
        store.append("t2", "hello earth", [], "r2")
        store.append("t3", "goodbye martians", [], "r3")
        store.update_rating("t1", 1)
        store.update_rating("t2", 1)
        store.update_rating("t3", -1)

        query_emb = _stub_embedder("hello planet")
        results = store.knn_similar(query_emb, k=2)
        assert len(results) == 2
        ids = [e.turn_id for e, _ in results]
        # "hello world" / "hello earth" are nearer than "goodbye martians"
        assert "t3" not in ids

    def test_knn_skips_unrated_entries(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        store.append("t1", "hello", [], "r")  # unrated
        store.append("t2", "hello", [], "r")
        store.update_rating("t2", 1)

        results = store.knn_similar(_stub_embedder("hello"), k=5)
        ids = [e.turn_id for e, _ in results]
        assert ids == ["t2"]

    def test_knn_skips_null_embedding_entries(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=None)
        store.append("t1", "hello", [], "r")  # embedding=None
        store.update_rating("t1", 1)
        # Query still runs — just returns empty
        results = store.knn_similar([1.0, 2.0, 3.0, 4.0], k=5)
        assert results == []

    def test_knn_filters_by_min_similarity(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        store.append("t1", "completely different topic here", [], "r")
        store.update_rating("t1", 1)
        query = _stub_embedder("hi")  # tiny, different prompt

        # Strict threshold filters out the dissimilar entry
        assert store.knn_similar(query, k=5, min_similarity=0.99) == []

    def test_knn_empty_query_returns_empty(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        store.append("t1", "hello", [], "r")
        store.update_rating("t1", 1)
        assert store.knn_similar([], k=3) == []


# ---- tool_stats -----------------------------------------------------------


class TestToolStats:
    def test_aggregates_across_rated_entries(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        store.append("t1", "a", [{"name": "get_weather"}], "r")
        store.append("t2", "b", [{"name": "get_weather"}, {"name": "get_time"}], "r")
        store.append("t3", "c", [{"name": "get_time"}], "r")
        store.update_rating("t1", 1)
        store.update_rating("t2", -1)
        store.update_rating("t3", 1)

        stats = store.tool_stats()
        assert stats["get_weather"]["count"] == 2
        assert stats["get_weather"]["sum"] == 0   # +1 and -1
        assert stats["get_weather"]["avg"] == 0.0
        assert stats["get_time"]["count"] == 2
        assert stats["get_time"]["avg"] == 0.0    # +1 (t3) and -1 (t2)

    def test_ignores_unrated(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        store.append("t1", "a", [{"name": "get_time"}], "r")  # unrated
        store.append("t2", "b", [{"name": "get_time"}], "r")
        store.update_rating("t2", 1)

        stats = store.tool_stats()
        assert stats["get_time"]["count"] == 1

    def test_empty_store_returns_empty_dict(self, tmp_path):
        store = FeedbackStore(tmp_path / "e.jsonl", embedder=_stub_embedder)
        assert store.tool_stats() == {}


# ---- malformed-line resilience --------------------------------------------


class TestCorruptLines:
    def test_bad_line_is_skipped_on_load(self, tmp_path, caplog):
        p = tmp_path / "e.jsonl"
        # Mix one good entry, one garbage line, another good one
        good1 = FeedbackEntry(
            turn_id="t1", created_at="x", prompt="a",
            tool_chain=[], reply="r", rating=1,
        )
        good2 = FeedbackEntry(
            turn_id="t2", created_at="y", prompt="b",
            tool_chain=[], reply="r", rating=-1,
        )
        p.write_text(
            good1.to_json() + "\n"
            + "this is not valid json\n"
            + good2.to_json() + "\n",
            encoding="utf-8",
        )
        store = FeedbackStore(p, embedder=None)
        ids = [e.turn_id for e in store.all_entries()]
        assert ids == ["t1", "t2"]
