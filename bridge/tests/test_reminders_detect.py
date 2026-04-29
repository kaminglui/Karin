"""Tests for bridge.reminders.detect — regex anchors, time parsing,
guard rails (no past, no too-far-future, no unanchored prompts).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bridge.reminders.detect import detect_reminder


_NOW = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Confident-match cases: each prompt has a clear anchor + a clear time.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prompt", [
    "remind me to call mom at 5pm",
    "Remind me to take out the trash tomorrow",
    "remind me that I have a meeting in 2 hours",
    "don't forget to feed the cat at 7pm",
    "ping me in 30 minutes",
    "set a reminder to check the oven in 10 minutes",
    "set a reminder for dinner at 7pm",
])
def test_detects_with_clear_time(prompt):
    out = detect_reminder(prompt, now=_NOW)
    assert out is not None, f"expected detection for: {prompt!r}"
    assert out.trigger_at > _NOW
    assert out.message  # non-empty


def test_returns_typed_DetectedReminder():
    out = detect_reminder("remind me to call mom at 5pm", now=_NOW)
    assert out is not None
    assert out.matched_phrase == "remind_me_to"
    # 5pm same day in UTC.
    assert out.trigger_at.hour == 17
    assert out.trigger_at.minute == 0


def test_relative_time_anchors_to_now():
    out = detect_reminder("remind me to call back in 30 minutes", now=_NOW)
    assert out is not None
    delta = out.trigger_at - _NOW
    assert timedelta(minutes=29) <= delta <= timedelta(minutes=31)


def test_message_strips_time_clause():
    """Message stored should NOT contain the time clause —
    "call mom at 5pm" → message="call mom" + trigger_at=5pm."""
    out = detect_reminder("remind me to call mom at 5pm", now=_NOW)
    assert out is not None
    assert "5pm" not in out.message.lower()
    assert "call mom" in out.message.lower()


# ---------------------------------------------------------------------------
# Abstain cases: no anchor / no usable time / past time / too far.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prompt", [
    "",
    "   ",
    None,
    "what's the weather today",
    "I have a meeting at 5pm",                   # mentions time but no anchor
    "5pm tomorrow",                              # no anchor at all
    "tell me about ukraine",
    "remind me",                                 # anchor but no content
    "remind me to call mom",                     # anchor + content but no time
    "don't forget",                              # anchor but nothing else
    "remind me to do the thing whenever",        # vague time word, no parse
])
def test_abstains_when_unsafe(prompt):
    assert detect_reminder(prompt, now=_NOW) is None


def test_past_time_is_rejected():
    """Time-of-day in the past TODAY would normally bump to tomorrow
    via PREFER_DATES_FROM=future — but if dateparser falls back to
    today anyway, the MIN_CLEARANCE guard rejects it."""
    # 9am — but it's already noon, so 9am today is in the past.
    out = detect_reminder("remind me to call at 9am", now=_NOW)
    if out is not None:
        # If detected, must be 9am TOMORROW (next future occurrence).
        assert out.trigger_at > _NOW + timedelta(hours=12)


def test_far_future_is_rejected():
    """Beyond the 365-day horizon → reject. Protects against
    dateparser turning random tokens into multi-year dates."""
    out = detect_reminder("remind me to vote in 2050", now=_NOW)
    # Either the parse fails OR the horizon guard catches the year.
    if out is not None:
        assert out.trigger_at <= _NOW + timedelta(days=365)


# ---------------------------------------------------------------------------
# Anchor variations
# ---------------------------------------------------------------------------
def test_each_anchor_recognized():
    cases = [
        ("remind me to X at 5pm",        "remind_me_to"),
        ("remind me that I owe Y at 5pm", "remind_me_to"),
        ("don't forget to do Z at 5pm",  "dont_forget"),
        ("dont forget to do W at 5pm",   "dont_forget"),
        ("ping me at 5pm",               "ping_me"),
        ("set a reminder to do A at 5pm", "set_a_reminder"),
        ("set reminder for B at 5pm",    "set_a_reminder"),
    ]
    for prompt, expected_phrase in cases:
        out = detect_reminder(prompt, now=_NOW)
        assert out is not None, prompt
        assert out.matched_phrase == expected_phrase, prompt


def test_anchor_is_case_insensitive():
    out = detect_reminder("REMIND ME TO call mom at 5pm", now=_NOW)
    assert out is not None


def test_anchor_must_be_word_boundary():
    """`unremind` shouldn't match `remind`. Conservative — keeps
    weird verbs from triggering false reminders."""
    assert detect_reminder("preremind me to X at 5pm", now=_NOW) is None
