"""RSS fetch layer.

Pure fetch-and-parse: returns RawArticle objects straight from feed XML.
Normalization, persistence, and the TTL cache gate all live elsewhere
(normalize.py, ledger.py, service.py respectively) so this module can be
tested in isolation against fixture XML.

Per-feed timeout is hard-capped at PER_FEED_TIMEOUT seconds; failed feeds
are reported (not raised) so one dead feed doesn't kill the whole batch.
"""
from __future__ import annotations

import calendar
import logging
from datetime import datetime, timezone
from typing import Iterable

import feedparser
import httpx

from bridge.news.models import Feed, RawArticle

log = logging.getLogger("bridge.news.fetch")

# Per-RSS-feed HTTP timeout (seconds). Tunable via
# config/tuning.yaml → news.fetch.per_feed_timeout_s.
from bridge import tuning as _tuning
PER_FEED_TIMEOUT = _tuning.get("news.fetch.per_feed_timeout_s", 5.0)
USER_AGENT = "karin-news/0.1 (+https://github.com/kaminglui/Karin)"


def fetch_feed(feed: Feed, client: httpx.Client) -> tuple[list[RawArticle], str | None]:
    """Fetch one RSS feed.

    Returns (articles, error_message). On any HTTP or parse error the list
    is empty and error_message is populated; the caller logs and continues.
    """
    try:
        resp = client.get(feed.url, timeout=PER_FEED_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return [], f"http error: {e}"

    parsed = feedparser.parse(resp.content)
    # feedparser sets bozo=1 on any parse anomaly. If it also got zero entries
    # out, treat it as a hard failure; otherwise trust the partial parse.
    if parsed.bozo and not parsed.entries:
        return [], f"parse error: {parsed.bozo_exception!r}"

    fetched_at = datetime.now(timezone.utc)
    articles: list[RawArticle] = []
    for entry in parsed.entries:
        url = entry.get("link") or ""
        title = entry.get("title") or ""
        summary = entry.get("summary") or entry.get("description") or ""
        if not url or not title:
            continue
        published = _parse_published(entry) or fetched_at
        articles.append(RawArticle(
            source_id=feed.source_id,
            url=url,
            title=title,
            summary=summary,
            published_at=published,
            fetched_at=fetched_at,
        ))
    return articles, None


def fetch_all(feeds: Iterable[Feed]) -> tuple[list[RawArticle], int, int]:
    """Fetch every feed sequentially.

    Returns (articles, ok_feed_count, failed_feed_count). Sequential
    (not concurrent) by design in V1: 10-15 feeds at 5s timeout each is
    at most ~75s worst-case, and concurrency adds complexity we don't
    need for lazy-refresh usage. Revisit if feed list grows large.
    """
    articles: list[RawArticle] = []
    ok = fail = 0
    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for feed in feeds:
            got, err = fetch_feed(feed, client)
            if err:
                log.warning("feed %s failed: %s", feed.source_id, err)
                fail += 1
            else:
                ok += 1
            articles.extend(got)
    log.info("fetch_all: %d feeds ok, %d failed, %d articles", ok, fail, len(articles))
    return articles, ok, fail


def _parse_published(entry) -> datetime | None:
    """Pull the best-available publish timestamp out of a feedparser entry.

    feedparser yields time.struct_time in UTC; calendar.timegm is the
    correct conversion (time.mktime assumes local time and would shift
    by the server TZ offset).
    """
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)
    return None
