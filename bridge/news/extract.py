"""Full-article extraction via trafilatura.

RSS feeds + NewsAPI both give only title + description truncated at a
few hundred characters — nowhere near the full body. This module
fetches the publisher's URL and runs trafilatura's readability-style
extractor to recover the clean article text.

Design notes:
  - Extraction is **separate storage** from the article ledger
    (``data/extracted.json``). The base `NormalizedArticle` stays
    small — the full text is a sidecar so the confidence / cluster
    pipeline never has to pay the byte cost of the body.
  - Extraction is **opt-in** at the caller level. If trafilatura
    isn't installed (e.g. a minimal Docker image), this module logs
    a one-time warning and returns empty — the rest of the news path
    keeps working without full-text.
  - We use httpx (not trafilatura.fetch_url) so the request shares
    the same User-Agent + timeout policy as the rest of the codebase,
    and so we can follow redirects cleanly.
  - Failures are **cached** with an error reason. We don't retry for
    ``_FAILURE_RETRY_HOURS`` — NYT paywalls etc. are persistent and
    re-fetching every poll is wasted budget.
  - Each ingest cycle caps extraction at
    ``MAX_EXTRACTIONS_PER_CYCLE``. Over time new articles get picked
    up; we don't try to backfill the whole ledger on one pass.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from bridge.utils import atomic_write_text

log = logging.getLogger("bridge.news.extract")

# --- tunables -------------------------------------------------------------

FETCH_TIMEOUT = 15.0
FETCH_USER_AGENT = "Mozilla/5.0 (compatible; karin/1.0)"

# How many articles we're willing to extract per ingest cycle. Each
# extraction is ~1-3s (download + parse), so 20 per cycle = ~30-60s of
# work. The news poller runs every 20 min, so over a day we cover
# ~1,400 articles — plenty of headroom for our volume.
MAX_EXTRACTIONS_PER_CYCLE = 20

# After a failed extraction, don't retry for this long. Most failures
# are permanent (paywall, sign-in wall, 404). 24h lets us pick up the
# occasional transient outage within reason.
_FAILURE_RETRY_HOURS = 24

# --- trafilatura is optional ---------------------------------------------
# Let the module import even when the library isn't available — callers
# check ``EXTRACTION_AVAILABLE`` and skip gracefully.

try:
    import trafilatura
    EXTRACTION_AVAILABLE = True
except ImportError:
    trafilatura = None  # type: ignore[assignment]
    EXTRACTION_AVAILABLE = False
    log.warning(
        "trafilatura not installed — full-text extraction disabled. "
        "Install with: pip install trafilatura"
    )


# --- data types -----------------------------------------------------------


@dataclass
class ExtractedArticle:
    """One extraction result, stored per article_id in extracted.json.

    When ``text`` is non-empty, extraction succeeded; when ``error`` is
    set, it failed and we remember the reason so we don't retry until
    the failure expiry window passes. ``attempted_at`` is the upper
    bound of the cooldown.
    """

    article_id: str
    text: str = ""
    title: str = ""
    author: str = ""
    date: str = ""
    error: str = ""
    attempted_at: datetime | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error


# --- store ----------------------------------------------------------------


class ExtractStore:
    """JSON-file-backed store for extracted article bodies.

    File layout: ``{article_id: {text, title, author, date, error,
    attempted_at}}`` with datetime stored as ISO 8601. Missing keys are
    tolerated on load so we can extend the schema without a migration.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, ExtractedArticle]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("extracted.json unreadable: %s — starting fresh", e)
            return {}
        out: dict[str, ExtractedArticle] = {}
        for aid, rec in raw.items():
            when = None
            w = rec.get("attempted_at")
            if w:
                try:
                    when = datetime.fromisoformat(w)
                except ValueError:
                    when = None
            out[aid] = ExtractedArticle(
                article_id=aid,
                text=rec.get("text", "") or "",
                title=rec.get("title", "") or "",
                author=rec.get("author", "") or "",
                date=rec.get("date", "") or "",
                error=rec.get("error", "") or "",
                attempted_at=when,
            )
        return out

    def save(self, records: dict[str, ExtractedArticle]) -> None:
        raw = {
            aid: {
                "text": r.text,
                "title": r.title,
                "author": r.author,
                "date": r.date,
                "error": r.error,
                "attempted_at": r.attempted_at.isoformat() if r.attempted_at else None,
            }
            for aid, r in records.items()
        }
        atomic_write_text(self._path, json.dumps(raw, indent=2, ensure_ascii=False))


# --- extraction primitives ------------------------------------------------


def extract_one(url: str, *, client: httpx.Client | None = None) -> ExtractedArticle:
    """Fetch one URL and run trafilatura on the HTML.

    Returns an ExtractedArticle with ``text`` populated on success or
    ``error`` populated on failure. Never raises — callers don't need a
    try/except around this. Timeouts, non-200 responses, bad HTML,
    paywalled pages, and extraction-returned-empty are all normal
    outcomes that return an error-flavored record.
    """
    now = datetime.now(timezone.utc)
    if not EXTRACTION_AVAILABLE:
        return ExtractedArticle(
            article_id="", error="trafilatura not installed",
            attempted_at=now,
        )
    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": FETCH_USER_AGENT})
    try:
        try:
            resp = client.get(url, timeout=FETCH_TIMEOUT, follow_redirects=True)
        except httpx.HTTPError as e:
            return ExtractedArticle(
                article_id="", error=f"http: {type(e).__name__}",
                attempted_at=now,
            )
        if resp.status_code != 200:
            return ExtractedArticle(
                article_id="", error=f"http {resp.status_code}",
                attempted_at=now,
            )
        try:
            result_json = trafilatura.extract(
                resp.text,
                include_comments=False,
                output_format="json",
                with_metadata=True,
                url=url,
            )
        except Exception as e:
            return ExtractedArticle(
                article_id="", error=f"trafilatura: {e!s}",
                attempted_at=now,
            )
        if not result_json:
            return ExtractedArticle(
                article_id="", error="empty extraction",
                attempted_at=now,
            )
        try:
            data = json.loads(result_json)
        except (json.JSONDecodeError, ValueError):
            return ExtractedArticle(
                article_id="", error="bad extractor output",
                attempted_at=now,
            )
        text = (data.get("text") or "").strip()
        if not text:
            return ExtractedArticle(
                article_id="", error="no text",
                attempted_at=now,
            )
        return ExtractedArticle(
            article_id="",   # caller fills this in
            text=text,
            title=(data.get("title") or "").strip(),
            author=(data.get("author") or "").strip(),
            date=(data.get("date") or "").strip(),
            error="",
            attempted_at=now,
        )
    finally:
        if own_client:
            client.close()


def extract_missing(
    articles: dict,                    # dict[article_id, NormalizedArticle]
    store: ExtractStore,
    *,
    max_extractions: int = MAX_EXTRACTIONS_PER_CYCLE,
    now: datetime | None = None,
) -> dict[str, int]:
    """Top up the extraction store for articles we don't have yet.

    An article is considered "pending" if:
      - it's not in the store, OR
      - its last attempt errored AND the retry cooldown has passed.

    Returns summary counts ``{pending, attempted, succeeded, failed}``
    so callers can log rate + success ratio. Does NOT raise on any
    per-article failure — those are recorded in the store and counted
    as ``failed``. The whole module is meant to be fail-soft; the news
    subsystem should never be degraded because extraction had a bad day.
    """
    if not EXTRACTION_AVAILABLE:
        return {"pending": 0, "attempted": 0, "succeeded": 0, "failed": 0}

    now = now or datetime.now(timezone.utc)
    existing = store.load()
    retry_cutoff = now - timedelta(hours=_FAILURE_RETRY_HOURS)

    pending: list[tuple[str, str]] = []
    for aid, art in articles.items():
        rec = existing.get(aid)
        if rec is None:
            pending.append((aid, art.url))
            continue
        if rec.ok:
            continue   # already have it
        # errored record — retry only if cooldown passed
        if rec.attempted_at is None or rec.attempted_at < retry_cutoff:
            pending.append((aid, art.url))

    if not pending:
        return {"pending": 0, "attempted": 0, "succeeded": 0, "failed": 0}

    # Prefer newer articles for extraction — freshness matters more than
    # completeness. Sort by published_at descending via articles dict.
    def _ts(aid_url):
        aid, _ = aid_url
        art = articles.get(aid)
        return getattr(art, "published_at", now) or now
    pending.sort(key=_ts, reverse=True)

    to_try = pending[:max_extractions]
    succeeded = 0
    failed = 0
    with httpx.Client(headers={"User-Agent": FETCH_USER_AGENT}) as client:
        for aid, url in to_try:
            rec = extract_one(url, client=client)
            rec.article_id = aid
            existing[aid] = rec
            if rec.ok:
                succeeded += 1
            else:
                failed += 1

    store.save(existing)
    return {
        "pending": len(pending),
        "attempted": len(to_try),
        "succeeded": succeeded,
        "failed": failed,
    }
