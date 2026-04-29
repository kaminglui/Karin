"""Normalization: RawArticle -> NormalizedArticle.

Produces the fields clustering and the brief builder depend on:
  - display_title: preserved case/punctuation, bracketed tags removed
  - normalized_title: lowercased + stripped form of display_title
  - summary: lowercased, HTML stripped, whitespace collapsed
  - display_summary: case-preserved, HTML stripped, collapsed
  - fingerprint: 5-gram shingle hash for near-dup detection
  - wire_attribution: "AP" / "Reuters" / "AFP" / None
  - language: detected script bucket ("en"/"zh"/"ja"/"ko"/"und")
  - article_id: stable hash of url

All functions are pure and deterministic — same input, same output — so
tests can pin exact strings.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

from bridge.news.detect import detect_language
from bridge.news.models import NormalizedArticle, RawArticle

# U+FFFD is the Unicode replacement character — it appears in feed text
# when the upstream encoding pipeline mangled bytes that couldn't be
# decoded. Stripping it leaves the rest of the title intact (some
# information is lost, but better than rendering a literal box glyph).
_REPLACEMENT_CHAR = "\ufffd"

# --- regexes ---------------------------------------------------------------

# Bracketed tags like "[LIVE UPDATES]", "(Video)", up to 40 chars inside.
# Conservative char limit avoids eating long parenthetical clauses that
# are part of the actual headline.
_BRACKETED_RE = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]")

# Common RSS boilerplate words used as prefixes. Stripped from both display
# and normalized titles because they add noise to clustering without
# carrying information.
_BOILERPLATE_RE = re.compile(
    r"\b(live updates?|breaking|watch live|video|photos?|analysis|opinion)\b[:\-]?\s*",
    re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Wire attribution markers. Patterns look for the classic byline forms
# that wire stories carry; covers most AP/Reuters/AFP syndication. Does
# NOT attempt to detect every possible wire (deferred — current three
# cover ~95% of V1 feeds).
_WIRE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\(AP\)|\bAP\s+—|\bAssociated Press\b", re.IGNORECASE), "AP"),
    (re.compile(r"\(Reuters\)|\bReuters\s+—", re.IGNORECASE), "Reuters"),
    (re.compile(r"\(AFP\)|\bAgence France-Presse\b", re.IGNORECASE), "AFP"),
]


# --- title / summary cleanup ----------------------------------------------

def clean_display_title(title: str) -> str:
    """Strip bracketed tags + boilerplate; preserve case. Collapse whitespace.

    Also performs Unicode hygiene: NFKC normalization (handles full-width
    and other compatibility forms) and removal of U+FFFD replacement
    characters left over from upstream encoding errors.
    """
    t = unicodedata.normalize("NFKC", title)
    t = t.replace(_REPLACEMENT_CHAR, "")
    t = _BRACKETED_RE.sub("", t)
    t = _BOILERPLATE_RE.sub("", t)
    return _WS_RE.sub(" ", t).strip(" -:")


def normalize_title(title: str) -> str:
    """Display-clean, then lowercased. Used for clustering/matching."""
    return clean_display_title(title).lower()


def clean_display_summary(summary: str) -> str:
    """Strip HTML tags + Unicode hygiene, preserve case.

    Mirrors clean_display_title for summaries: removes HTML,
    NFKC-normalizes, drops U+FFFD replacement chars, collapses
    whitespace. Output is what the translated-brief path should pull
    from when extracted body text isn't available.
    """
    s = unicodedata.normalize("NFKC", summary)
    s = s.replace(_REPLACEMENT_CHAR, "")
    s = _HTML_TAG_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def normalize_summary(summary: str) -> str:
    """Strip HTML tags, lowercase, collapse whitespace.

    Lowercased form powers watchlist matching + fingerprinting. The
    case-preserved sibling is clean_display_summary; we derive the
    lowercase from it so both stay in lockstep.
    """
    return clean_display_summary(summary).lower()


# --- attribution ----------------------------------------------------------

def infer_wire_attribution(raw: RawArticle) -> str | None:
    """Return 'AP' / 'Reuters' / 'AFP' if the title or summary clearly
    carries that wire byline; else None.

    Conservative by design: false positives here inflate the
    syndication_count and suppress independent_confirmation_count,
    which hurts confidence scoring. Only matches explicit markers.
    """
    hay = f"{raw.title} {raw.summary}"
    for pat, name in _WIRE_PATTERNS:
        if pat.search(hay):
            return name
    return None


# --- identity & fingerprint -----------------------------------------------

def article_id(url: str) -> str:
    """Stable 16-hex-char id derived from the article URL."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def compute_fingerprint(
    normalized_title: str,
    normalized_summary: str,
    url: str = "",
) -> str:
    """5-gram shingle hash across title + summary.

    Two articles with near-identical wording (e.g. a wire story that
    multiple outlets republish verbatim) produce the same fingerprint,
    which the clusterer uses to collapse them to one independent entity.

    For thin content (<5 words combined), the 5-gram approach can't work
    and a naive hash of the text would collide across different articles
    that share a stock headline ("Here's the latest." with empty summary
    is the canonical case). Fall back to a URL-derived fingerprint so
    those articles stay distinct. If no URL is provided, degrade to
    text-only hashing for determinism in tests.
    """
    words = (normalized_title + " " + normalized_summary).split()
    if len(words) < 5:
        seed = url if url else " ".join(words)
        return hashlib.sha1(f"thin:{seed}".encode("utf-8")).hexdigest()[:16]
    shingles = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    # sorted + set => order-independent bag-of-shingles; small reordering
    # of sentences won't change the fingerprint.
    joined = "|".join(sorted(set(shingles)))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


# --- end-to-end -----------------------------------------------------------

def normalize(raw: RawArticle) -> NormalizedArticle:
    """Convert one RawArticle to a NormalizedArticle."""
    display_title = clean_display_title(raw.title)
    normalized_title_str = display_title.lower()
    display_summary = clean_display_summary(raw.summary)
    summary = display_summary.lower()
    # Detect on the case-preserved title+summary pair: the detector
    # counts unicode blocks, so case doesn't matter mathematically, but
    # the preserved-case text is the only form we keep around long-term
    # — saves one allocation and keeps the input obvious.
    language = detect_language(f"{display_title} {display_summary}")
    return NormalizedArticle(
        article_id=article_id(raw.url),
        source_id=raw.source_id,
        url=raw.url,
        display_title=display_title,
        normalized_title=normalized_title_str,
        summary=summary,
        fingerprint=compute_fingerprint(normalized_title_str, summary, raw.url),
        wire_attribution=infer_wire_attribution(raw),
        published_at=raw.published_at,
        fetched_at=raw.fetched_at,
        display_summary=display_summary,
        language=language,
    )


def normalize_many(raws: Iterable[RawArticle]) -> list[NormalizedArticle]:
    return [normalize(r) for r in raws]
