"""Script-based language detection for the news pipeline.

Cheap, dependency-free, deterministic. Counts code-point hits across
the unicode blocks that matter for our current + near-term feeds
(Latin / Han / Hiragana+Katakana / Hangul) and picks the dominant one.

Why not langdetect / fasttext:
  - Both pull wheels we don't otherwise need (adds to the Jetson image).
  - Our headlines are short (~10 words). Probabilistic detectors flip
    language between "Trump reacts" and "Trump responds" because n-gram
    evidence is thin. Script-counting is stable on short input.
  - We only need four buckets + "unknown"; a full 70+ language detector
    is overkill and costs accuracy at the edges we care about.

Tradeoff we accept: can't distinguish languages that share a script
(en vs es vs de all look Latin here). For Phase A that's fine — current
feeds are English-only on the Latin side; if/when a Spanish or German
feed lands we'll bolt on a small stopword check. Flagged in
SUPPORTED_LANGUAGES so callers know the contract.

Public API:
    detect_language(text) -> str        # one of SUPPORTED_LANGUAGES or "und"
    SUPPORTED_LANGUAGES: frozenset[str]
"""
from __future__ import annotations

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en", "zh", "ja", "ko"})

# Minimum number of script-bearing characters required before we trust
# the count. Titles shorter than this (e.g. "BBC", numbers-only) fall
# back to "und" — the caller can decide what to do with unknown.
_MIN_SCRIPT_CHARS = 3

# Fraction of script-bearing chars that must fall in the winning bucket
# for us to commit to a call. Below this we return "und" rather than
# guess on a mixed-script blurb.
_DOMINANCE_THRESHOLD = 0.55


def _bucket(cp: int) -> str | None:
    """Map one code point to one of our language buckets, or None
    (non-script — digits, punctuation, whitespace, symbols).

    Ranges follow the unicode block tables; we accept the small amount
    of overlap at edges (e.g. CJK extensions) because the dominance
    threshold absorbs it.
    """
    # Latin: Basic Latin letters + Latin-1 Supplement letters + extensions
    if 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:
        return "latin"
    if 0x00C0 <= cp <= 0x024F:
        return "latin"
    # Hangul (Korean). Jamo + Syllables + Compatibility Jamo ranges.
    if 0x1100 <= cp <= 0x11FF or 0xAC00 <= cp <= 0xD7AF or 0x3130 <= cp <= 0x318F:
        return "hangul"
    # Hiragana + Katakana (Japanese kana). Pure-kana text is unambiguously JP.
    if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
        return "kana"
    # Han / CJK Unified Ideographs. Shared by zh + ja (kanji). We treat
    # han as Chinese BY DEFAULT and only override to Japanese when kana
    # also appears — kana is the reliable JP marker.
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "han"
    return None


def detect_language(text: str) -> str:
    """Return one of SUPPORTED_LANGUAGES, or "und" when undecidable.

    Rule order (matters):
      1. If kana is present, it's Japanese. (Even mixed with han/latin.)
      2. Otherwise pick whichever bucket carries >= _DOMINANCE_THRESHOLD
         of the script-bearing characters.
      3. If no bucket dominates, or too few script chars overall,
         return "und".

    Latin-dominant -> "en" reflects current feed reality; a future
    multi-Latin split would need stopword/word-shape checks layered on.
    """
    if not text:
        return "und"

    counts = {"latin": 0, "hangul": 0, "kana": 0, "han": 0}
    for ch in text:
        b = _bucket(ord(ch))
        if b is not None:
            counts[b] += 1

    total = sum(counts.values())
    if total < _MIN_SCRIPT_CHARS:
        return "und"

    # Kana is the tiebreaker for JP vs ZH. Any kana at all -> Japanese.
    if counts["kana"] > 0:
        return "ja"

    # Otherwise pick dominant bucket.
    winner = max(counts, key=lambda k: counts[k])
    if counts[winner] / total < _DOMINANCE_THRESHOLD:
        return "und"

    return {
        "latin": "en",
        "hangul": "ko",
        "han": "zh",
        "kana": "ja",  # unreachable given the kana early-return, kept for completeness
    }[winner]
