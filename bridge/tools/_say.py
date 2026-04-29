"""Repeat-back tool — echoes a phrase verbatim.

Without this, "repeat after me: hello world" tends to route to wiki or
web_search (the LoRA reads it as a lookup query). The `say` tool gives
the routing layer an explicit target so the phrase comes back unchanged
and reaches TTS as the user intended.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("bridge.tools")

# Cap to keep TTS turns bounded and prevent prompt-injection abuse
# (e.g. someone telling Karin to repeat a 5000-char block).
_MAX_LEN = 500


def _say(text: str = "") -> str:
    """Return the user's phrase verbatim, trimmed to _MAX_LEN chars."""
    text = (text or "").strip()
    if not text:
        return "(nothing to say — no text was provided)"
    if len(text) > _MAX_LEN:
        text = text[:_MAX_LEN] + "…"
    log.info("say: %r", text[:80])
    return text


# Patterns to extract the verbatim phrase directly from the user message,
# in priority order. Covers the common forms; anything else falls back to
# the LoRA's argument extraction. Compiled once at import.
_EXTRACTORS: tuple[re.Pattern[str], ...] = (
    # `repeat/say/echo/parrot after me [:,] X` — colon, comma, or space
    re.compile(r"(?:repeat|say|echo|parrot)\s+after\s+me[\s:,]+(.+?)\s*$", re.IGNORECASE),
    # `repeat/echo/parrot this/the following [:,] X`
    re.compile(r"(?:repeat|echo|parrot)\s+(?:this|these|the\s+following)[\s:,]+(.+?)\s*$", re.IGNORECASE),
    # `say/repeat/echo "X"` or `say 'X'` — quoted form
    re.compile(r"(?:say|repeat|echo|parrot)\s+[\"'](.+?)[\"']", re.IGNORECASE),
    # `repeat/say exactly [:,] X`
    re.compile(r"(?:repeat|say)\s+exactly[\s:,]+(.+?)\s*$", re.IGNORECASE),
    # Bare `repeat: X` / `say: X` (post-colon)
    re.compile(r"^\s*(?:repeat|say|echo)\s*:\s*(.+?)\s*$", re.IGNORECASE),
    # Plain `say <text>` at the start of the message. Lowest priority —
    # listed last so the more-specific patterns above (`say after me`,
    # `say "..."`, `say exactly`, `say:`) win when they apply. Required
    # to extract the text when the new ^say routing pattern fires.
    re.compile(r"^\s*say\s+(.+?)\s*$", re.IGNORECASE),
)


def extract_verbatim_phrase(user_text: str | None) -> str | None:
    """Extract the phrase to echo from the current user message.

    Used by the bridge to override the LoRA's `text` argument when the
    LoRA latches onto a stale phrase from conversation history (an
    iter-3 limitation — the model wasn't trained on `say` and over-uses
    the most-recent string-shaped span in context).

    Returns the phrase with surrounding whitespace + matching quotes
    stripped, or None when no pattern fits — in which case the caller
    keeps whatever the LoRA produced.
    """
    if not user_text:
        return None
    for pat in _EXTRACTORS:
        m = pat.search(user_text)
        if m:
            phrase = m.group(1).strip()
            # Strip matching surrounding quotes from the captured group.
            if len(phrase) >= 2 and phrase[0] in "\"'" and phrase[-1] == phrase[0]:
                phrase = phrase[1:-1].strip()
            if phrase:
                return phrase
    return None
