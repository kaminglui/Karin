"""NewsAPI.org fetcher — bolts the wire-service feed onto the news subsystem.

Why this module exists:
  The RSS-only V1 feed list (BBC / NPR / Guardian / NYT / etc.) is all
  REPUTABLE tier, so the confidence machinery in `confidence.py`
  rarely promotes stories past PROVISIONALLY_CONFIRMED (the CONFIRMED
  rule requires ≥3 independent sources with at least 1 WIRE). NewsAPI
  lets us pull AP, Reuters, Bloomberg, AFP directly — register those
  as tier=WIRE in sources.json and the existing evaluation pipeline
  starts producing CONFIRMED states without any new evaluation code.

Design:
  - One function, `fetch_newsapi()`, returns a list[RawArticle] shaped
    identically to what `fetch_all()` produces from RSS. Callers
    concatenate the two lists and feed the result to
    `normalize_many()` — the rest of the pipeline is unchanged.
  - Keyed on `KARIN_NEWSAPI_KEY`. If the env var is unset, returns
    an empty list (no error, no log noise beyond a one-time INFO).
    That keeps the feature strictly opt-in: no key = behaves as if
    this module didn't exist.
  - Uses the ``/v2/top-headlines?sources=<csv>`` endpoint, not
    ``/v2/everything``. top-headlines is rate-tier-friendly (one call
    returns up to 100 recent articles across all requested sources),
    and its results are current-day rather than the 24h-delayed
    dev-tier window on /v2/everything.
  - Filters responses to the source_ids we KNOW — if NewsAPI ever
    returns articles from a source we haven't registered in
    sources.json, those are dropped to keep the ownership_group /
    tier accounting clean.

TOS note: NewsAPI's free dev plan is labeled "development use only."
A single-host personal assistant is a gray area; if this ever gets
deployed more broadly, switch to a paid plan or rotate to a different
provider.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Iterable

import httpx

from bridge.news.models import RawArticle, Source

log = logging.getLogger("bridge.news.newsapi")

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/top-headlines"

# Per-call max articles to pull. NewsAPI free tier returns up to 100;
# we request fewer to keep our ledger tight and to tolerate multiple
# polls per day without blowing the 100-req/day limit.
_DEFAULT_PAGE_SIZE = 30

# HTTP timeout. NewsAPI is usually snappy (<1s) but we match the
# RSS fetcher's conservative window so a slow network doesn't stall
# the whole ingest.
_FETCH_TIMEOUT = 8.0


def _api_key() -> str:
    """Read KARIN_NEWSAPI_KEY from env. Empty string means "feature off"."""
    return (os.environ.get("KARIN_NEWSAPI_KEY") or "").strip()


def fetch_newsapi(
    sources: dict[str, Source],
    *,
    country: str | None = None,
    category: str | None = None,
    page_size: int = _DEFAULT_PAGE_SIZE,
    client: httpx.Client | None = None,
) -> list[RawArticle]:
    """Fetch recent articles from NewsAPI's /top-headlines for every
    source in `sources` whose id is recognized by NewsAPI.

    The `sources` arg is the full Source registry; we select the subset
    whose id matches a NewsAPI source slug (by convention those are
    hyphenated, e.g. 'associated-press', 'reuters'). This keeps the
    registry as the single source of truth for tiering.

    Returns empty list when:
      - KARIN_NEWSAPI_KEY is unset
      - No registered sources match NewsAPI slugs
      - The HTTP call fails for any reason (caller treats as "no new
        wire input this cycle", not an error)
    """
    key = _api_key()
    if not key:
        log.debug("NewsAPI skipped: KARIN_NEWSAPI_KEY not set")
        return []

    # Select the sources NewsAPI should fetch. We use `is_wire_service`
    # as the selector: in this deploy, wire-tier sources (AP, Reuters,
    # Bloomberg) are the ones actually served by NewsAPI. RSS-only
    # REPUTABLE sources (BBC, NPR, etc.) have is_wire_service=False
    # and flow through `fetch_all` on their own.
    # If a non-wire NewsAPI source is ever needed, add it to sources.json
    # with is_wire_service=true (slight semantic abuse but keeps the
    # selector simple) or extend Source with an explicit "via" field.
    newsapi_source_ids = sorted(
        sid for sid, src in sources.items() if src.is_wire_service
    )
    if not newsapi_source_ids:
        return []

    params = {
        "apiKey": key,
        "sources": ",".join(newsapi_source_ids),
        "pageSize": str(page_size),
    }
    if country:
        # Note: can't pass BOTH sources and country to top-headlines;
        # we prefer sources for explicit wire selection. country is
        # here for callers that want to bypass sources entirely.
        params.pop("sources", None)
        params["country"] = country
    if category:
        params["category"] = category

    owns_client = client is None
    try:
        if owns_client:
            client = httpx.Client()
        resp = client.get(NEWSAPI_ENDPOINT, params=params, timeout=_FETCH_TIMEOUT)
    except httpx.HTTPError as e:
        log.warning("NewsAPI fetch failed: %s", e)
        return []
    finally:
        if owns_client and client is not None:
            client.close()

    if resp.status_code != 200:
        log.warning(
            "NewsAPI returned %d: %s", resp.status_code, resp.text[:200],
        )
        return []
    try:
        data = resp.json()
    except ValueError as e:
        log.warning("NewsAPI returned non-JSON: %s", e)
        return []
    if data.get("status") != "ok":
        log.warning("NewsAPI error: %s", data.get("message") or data)
        return []

    articles_raw = data.get("articles") or []
    out: list[RawArticle] = []
    for art in articles_raw:
        raw = _to_raw_article(art, sources)
        if raw is not None:
            out.append(raw)
    log.info("NewsAPI: %d articles from %d sources", len(out), len(newsapi_source_ids))
    return out


def _to_raw_article(
    art: dict, sources: dict[str, Source],
) -> RawArticle | None:
    """Convert one NewsAPI article dict to a RawArticle, or None to skip.

    Skips articles from sources we haven't registered (we need the
    Source metadata to assign tier/ownership_group correctly — an
    article we can't attribute is useless to the confidence machinery).
    """
    src_meta = art.get("source") or {}
    src_id = (src_meta.get("id") or "").strip()
    if not src_id or src_id not in sources:
        return None

    url = (art.get("url") or "").strip()
    title = (art.get("title") or "").strip()
    summary = (art.get("description") or "").strip()
    if not url or not title:
        return None

    # NewsAPI timestamps are ISO-8601 with a trailing 'Z'. Convert to
    # timezone-aware UTC so downstream datetime arithmetic works.
    published_raw = art.get("publishedAt") or ""
    try:
        published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
    except ValueError:
        # Malformed timestamp — default to "now" so the article still
        # shows up in current-window queries instead of getting dropped
        # for a bad upstream field.
        published_at = datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    return RawArticle(
        source_id=src_id,
        url=url,
        title=title,
        summary=summary,
        published_at=published_at,
        fetched_at=datetime.now(timezone.utc),
    )
