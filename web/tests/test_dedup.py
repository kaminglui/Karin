"""Unit tests for the repeat-prompt dedup cache in `web/server.py`.

Covers:
  * Normalization (case + whitespace) collapses equivalence classes.
  * Time-sensitive prompts bypass the cache (lookup + store).
  * Round-trip: store + lookup hits within TTL.
  * TTL expiry: stale entries miss + auto-prune.
  * LRU bound: cache never grows past _DEDUP_MAX entries.

The cache is process-local state; tests reset it via _dedup_clear().
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from web.server import (
    _DEDUP_MAX,
    _DEDUP_TTL_S,
    _dedup_cache,
    _dedup_clear,
    _dedup_is_time_sensitive,
    _dedup_lookup,
    _dedup_normalize,
    _dedup_store,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    _dedup_clear()
    yield
    _dedup_clear()


def _payload(answer: str = "ok") -> dict:
    """Minimal response shape — same fields the live endpoint returns."""
    return {
        "user": "x",
        "assistant": answer,
        "audio_b64": "",
        "audio_mime": None,
    }


def _arun(coro):
    """Run an async coroutine to completion. Used instead of pytest-asyncio
    so the test file has no plugin dependency — repo's pytest config has
    anyio installed but no asyncio plugin."""
    return asyncio.run(coro)


# --- normalization ---------------------------------------------------------

def test_normalize_collapses_case_and_whitespace() -> None:
    assert _dedup_normalize("  Say  Hello  World ") == "say hello world"
    assert _dedup_normalize("HELLO") == "hello"
    assert _dedup_normalize("\t\nfoo\nbar  ") == "foo bar"


def test_normalize_idempotent() -> None:
    once = _dedup_normalize("  WHAT IS 2+2 ")
    twice = _dedup_normalize(once)
    assert once == twice


# --- time-sensitive detection ---------------------------------------------

@pytest.mark.parametrize("prompt", [
    "what time is it",
    "weather in tokyo",
    "latest news",
    "gold price today",
    "any alerts I should know about",
    "what's the temperature",
    "today's digest",
    "is it raining",
])
def test_time_sensitive_prompts_match(prompt: str) -> None:
    assert _dedup_is_time_sensitive(prompt), prompt


@pytest.mark.parametrize("prompt", [
    "what is the capital of France",
    "say hello world",
    "how do you spell mississippi",
    "tell me a joke",
    "explain photosynthesis",
])
def test_non_time_sensitive_prompts_skip(prompt: str) -> None:
    assert not _dedup_is_time_sensitive(prompt), prompt


# --- round-trip ------------------------------------------------------------

def test_store_then_lookup_hits_within_ttl() -> None:
    payload = _payload("forty-two")

    async def body():
        await _dedup_store("what is 2+2", payload)
        return await _dedup_lookup("what is 2+2")

    assert _arun(body()) == payload


def test_lookup_normalizes_before_matching() -> None:
    payload = _payload("forty-two")

    async def body():
        await _dedup_store("what is 2+2", payload)
        # Different case + extra spaces — same equivalence class
        return await _dedup_lookup("  WHAT IS 2+2 ")

    assert _arun(body()) == payload


def test_miss_for_unseen_prompt() -> None:
    async def body():
        await _dedup_store("seen prompt", _payload())
        return await _dedup_lookup("unseen prompt")

    assert _arun(body()) is None


# --- time-sensitive bypass ------------------------------------------------

def test_time_sensitive_store_is_no_op() -> None:
    """Storing a time-sensitive prompt is a no-op — cache stays empty."""
    _arun(_dedup_store("what time is it", _payload()))
    assert "what time is it" not in _dedup_cache


def test_time_sensitive_lookup_returns_none() -> None:
    """Even if a prior store somehow happened, time-sensitive lookups
    skip the cache. Belt + suspenders against false negatives in the
    classifier."""
    # Force the cache to contain a time-sensitive entry directly,
    # bypassing the store-time guard.
    _dedup_cache["weather in tokyo"] = (time.monotonic(), _payload())
    assert _arun(_dedup_lookup("weather in tokyo")) is None


# --- TTL expiry -----------------------------------------------------------

def test_ttl_expiry_returns_none_and_prunes() -> None:
    payload = _payload()
    _arun(_dedup_store("benign question", payload))
    # Fast-forward past TTL by patching time.monotonic.
    fake_future = time.monotonic() + _DEDUP_TTL_S + 1.0
    with patch("web.server._time.monotonic", return_value=fake_future):
        miss = _arun(_dedup_lookup("benign question"))
    assert miss is None
    # Stale entry should have been removed during the lookup.
    assert "benign question" not in _dedup_cache


# --- LRU bound ------------------------------------------------------------

def test_lru_eviction_caps_at_max() -> None:
    """Storing _DEDUP_MAX + 5 distinct entries leaves cache exactly at
    _DEDUP_MAX, with the oldest entries evicted."""
    async def body():
        for i in range(_DEDUP_MAX + 5):
            await _dedup_store(
                f"benign question number {i}", _payload(f"reply {i}"),
            )
    _arun(body())
    assert len(_dedup_cache) == _DEDUP_MAX
    # The first 5 should have been evicted.
    assert "benign question number 0" not in _dedup_cache
    assert "benign question number 4" not in _dedup_cache
    # The most recently inserted should be present.
    assert f"benign question number {_DEDUP_MAX + 4}" in _dedup_cache


def test_recent_hit_bumps_to_lru_front() -> None:
    """A successful lookup should move the entry to the front so it
    survives subsequent eviction. Otherwise frequently-asked questions
    would be evicted before infrequent ones."""
    async def body():
        await _dedup_store("first question", _payload("a"))
        for i in range(_DEDUP_MAX - 1):  # fill to cap (incl first)
            await _dedup_store(f"filler {i}", _payload())
        # Hit "first question" — moves it to LRU front
        await _dedup_lookup("first question")
        # New entry triggers eviction of the oldest item, which is now
        # "filler 0", not "first question".
        await _dedup_store("brand new question", _payload())
    _arun(body())
    assert "first question" in _dedup_cache
    assert "filler 0" not in _dedup_cache
