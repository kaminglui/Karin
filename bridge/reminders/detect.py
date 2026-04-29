"""Code-based reminder detection from chat input.

Pure-Python regex + ``dateparser`` — no LLM, no learning. The chat
turn handler calls :func:`detect_reminder` on every user message
*before* dispatching to the LLM. When detection returns a hit, the
reminder is created via :func:`bridge.reminders.create_reminder`,
the LLM still runs (so Karin can acknowledge), and the reminder
fires at its trigger time through the notify pipeline.

Design rules:

* **High precision over recall.** A false positive (system creates a
  reminder the user didn't intend) is more annoying than a false
  negative (user has to be more explicit). Quiet-by-default: when
  the regex / time parse aren't both confident, return None.

* **Code, not LLM.** This module never touches the model. Adheres to
  the design.md rule "code is the trust path; LLM is a supplement."
  An optional LLM fallback is documented in docs/ideas.md (Phase 5)
  for cases code misses, gated by a feature flag.

* **Anchor phrases up front.** Match a small set of intent-revealing
  prefixes ("remind me to ...", "don't forget ...", "ping me at ...").
  Without one of these anchors, we never create a reminder — even if
  the user mentions "5pm". This is the precision/recall tradeoff.

* **Time parse must succeed AND land in the future.** ``dateparser``
  is permissive; we add a guard rail so "remind me to call mom"
  (no time given) doesn't create a reminder for "now."
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

log = logging.getLogger("bridge.reminders.detect")


@dataclass(frozen=True)
class DetectedReminder:
    """One successful detection. Caller persists via create_reminder."""
    trigger_at: datetime    # tz-aware UTC, in the future
    message: str            # cleaned reminder content
    matched_phrase: str     # which anchor phrase fired (audit / debug)


# Anchor phrases that MUST appear for a reminder to fire. Each pattern
# captures `content` and (optionally) leaves the time clause attached
# for downstream parsing. Order matters — first match wins.
#
# `(?P<content>...)` captures the action; the rest is parsed for the
# trigger time. Patterns are deliberately conservative — we'd rather
# miss than misfire. Add new phrasings as we see them in real usage.
_ANCHORS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (
        "remind_me_to",
        # "remind me to call mom at 5pm"
        # "remind me that I need to ..."
        re.compile(
            r"\bremind\s+me\s+(?:to|that)\s+(?P<content>.+)",
            re.IGNORECASE,
        ),
    ),
    (
        "dont_forget",
        re.compile(
            r"\b(?:don'?t|do\s+not)\s+forget\s+(?:to\s+)?(?P<content>.+)",
            re.IGNORECASE,
        ),
    ),
    (
        "ping_me",
        # Keep the time-prefix word ("at", "in", "on") inside the
        # captured content — the time-clause regex needs it. Earlier
        # version consumed it, leaving "30 minutes" with no "in"
        # prefix, which the time pattern couldn't re-find.
        re.compile(
            r"\bping\s+me\s+(?P<content>.+)",
            re.IGNORECASE,
        ),
    ),
    (
        "set_a_reminder",
        # "set a reminder to ...", "set a reminder for ..."
        re.compile(
            r"\bset\s+(?:an?\s+)?reminder\s+(?:to|for|that)\s+(?P<content>.+)",
            re.IGNORECASE,
        ),
    ),
)

# Time clauses we lift out of the captured content before storing the
# message. Order matters: longer phrases first so "in 30 minutes" wins
# over "in 30". Each pattern is anchored with \b on both sides so it
# can match anywhere in the string.
_TIME_CLAUSE_PATTERNS: tuple[str, ...] = (
    # absolute date + time:  "at 5pm tomorrow", "on Friday at 3"
    r"\b(?:at|on)\s+\S+(?:\s+\S+){0,4}",
    # "in 30 minutes", "in 2 hours", "in 1 hour"
    r"\bin\s+\d+(?:\.\d+)?\s+(?:second|sec|minute|min|hour|hr|day|week|month|year)s?\b",
    # "5pm", "5:30pm", "5 pm" — bare time-of-day at end
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    # "tomorrow morning/afternoon/evening/night" / "tonight"
    r"\b(?:tomorrow|tonight)(?:\s+(?:morning|afternoon|evening|night))?\b",
    # "next Monday" / "Friday" alone
    r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
)
_TIME_CLAUSE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _TIME_CLAUSE_PATTERNS),
    re.IGNORECASE,
)

# How far in the future a parsed time can land. Beyond this, treat
# the parse as junk — protects against dateparser interpreting random
# numbers as years (e.g. "2050" turning into a 25-year-out reminder).
_MAX_HORIZON = timedelta(days=365)
# Minimum clearance into the future. Prevents "remind me at 3pm" from
# scheduling for 3pm TODAY when current time is already past 3pm —
# dateparser will then bump to tomorrow IF we set PREFER_DATES_FROM.
_MIN_CLEARANCE = timedelta(seconds=30)


def detect_reminder(
    text: str,
    *,
    now: datetime | None = None,
) -> DetectedReminder | None:
    """Try to pull a reminder out of ``text``. Returns None when the
    prompt isn't a reminder request, OR when an anchor matched but no
    usable time clause came out.

    ``now`` is the reference timestamp for relative phrases ("in 2 h").
    Defaults to wall-clock UTC; tests pin it for determinism.
    """
    if not text or not text.strip():
        return None
    ref_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    for phrase_id, anchor_re in _ANCHORS:
        m = anchor_re.search(text)
        if not m:
            continue
        captured = m.group("content").strip(" .!?;:,")
        if not captured:
            return None
        # Pull a time clause out of the captured content. dateparser
        # also accepts the full string, but extracting the clause
        # gives us a cleaner reminder message ("call mom" vs
        # "call mom at 5pm").
        time_clause, message = _split_time_and_message(captured)
        if not time_clause:
            log.debug("anchor %s matched but no time clause: %r", phrase_id, captured)
            return None
        trigger = _parse_time(time_clause, ref_now)
        if trigger is None:
            log.debug("anchor %s time-parse failed: %r", phrase_id, time_clause)
            return None
        # Guard rails: must be in the future, within reasonable horizon.
        if trigger <= ref_now + _MIN_CLEARANCE:
            log.debug("trigger %s not far enough in future (now=%s)", trigger, ref_now)
            return None
        if trigger > ref_now + _MAX_HORIZON:
            log.debug("trigger %s past max horizon", trigger)
            return None
        if not message:
            # Anchor matched + time parsed, but no task content
            # remained after stripping the time clause. Fall back to
            # a generic message keyed by the anchor — "ping me at
            # 5pm" without a task is still a useful self-prompt.
            message = _DEFAULT_MESSAGE.get(phrase_id, "reminder")
        return DetectedReminder(
            trigger_at=trigger,
            message=message,
            matched_phrase=phrase_id,
        )
    return None


# Used when an anchor matched + time parsed, but the user didn't say
# WHAT to remind them about. We still create the reminder with a
# default message keyed on the anchor — "ping" feels right for
# ping_me, generic "reminder" for the rest.
_DEFAULT_MESSAGE: dict[str, str] = {
    "ping_me":        "ping",
    "remind_me_to":   "reminder",
    "dont_forget":    "reminder",
    "set_a_reminder": "reminder",
}


def _split_time_and_message(captured: str) -> tuple[str | None, str]:
    """Extract the FIRST time clause from ``captured`` and return
    (time_clause, cleaned_message). When no clause matches, returns
    (None, captured)."""
    m = _TIME_CLAUSE_RE.search(captured)
    if not m:
        return None, captured.strip()
    time_clause = m.group(0).strip()
    # Remove the clause from the message to clean up the stored text.
    # Replace once (not all) so a sentence containing the same words
    # twice doesn't get over-stripped.
    cleaned = (captured[:m.start()] + captured[m.end():]).strip(" .!?;:,")
    # Collapse whitespace.
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return time_clause, cleaned


def _parse_time(clause: str, ref_now: datetime) -> datetime | None:
    """Run ``dateparser`` on the clause, anchored at ``ref_now``,
    preferring future interpretations. Returns tz-aware UTC datetime
    or None on parse failure. Lazy-imports dateparser so the
    reminders module is importable in environments without it
    (the test suite for store.py shouldn't need dateparser)."""
    try:
        import dateparser   # type: ignore[import-untyped]
    except ImportError:
        log.warning("dateparser not installed; reminder detection disabled")
        return None
    parsed = dateparser.parse(
        clause,
        settings={
            "RELATIVE_BASE": ref_now.replace(tzinfo=None),
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        },
    )
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
