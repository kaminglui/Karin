"""State Department travel advisory fetcher + diff.

The State Department publishes a global RSS feed of travel advisories at
https://travel.state.gov/_res/rss/TAsTWs.xml. Each entry's title is
formatted like:

  "Egypt - Level 3: Reconsider Travel Travel Advisory"

We parse country name + numeric level (1..4) out of the title. The
country code mapping is a small hand-curated table — good enough for V1
since the set of countries at Level 3/4 is small and slow-moving.

First-poll behavior (spec decision): when advisory_state.json is absent,
the first poll silently establishes the baseline and emits zero changes.
Only SUBSEQUENT polls report diffs. This prevents day-one from flooding
the user with ~200 "changed to current level" false alarms.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

import feedparser
import httpx

log = logging.getLogger("bridge.alerts.advisory_fetch")

ADVISORY_RSS_URL = "https://travel.state.gov/_res/rss/TAsTWs.xml"
# Tunable via config/tuning.yaml → alerts.advisory.fetch_timeout_s.
from bridge import tuning as _tuning
FETCH_TIMEOUT = _tuning.get("alerts.advisory.fetch_timeout_s", 10.0)

# Pattern inside feed titles: "Country Name - Level N: ..."
# Uses a non-greedy country capture so "Saint Vincent and the Grenadines" works.
_LEVEL_RE = re.compile(r"^\s*(?P<country>.+?)\s*-\s*Level\s*(?P<level>[1-4])\b", re.IGNORECASE)


class AdvisoryFetchError(Exception):
    pass


def fetch_advisories(
    client: httpx.Client | None = None,
    url: str = ADVISORY_RSS_URL,
) -> dict[str, dict]:
    """Poll the State Dept feed. Return {country: {level, title}}.

    `country` is a normalized uppercase key (currently the country name
    uppercased + hyphens→underscores). For V1 this is good enough as a
    stable key; in a later phase we can map to ISO 3166 codes if needed.

    Passing an httpx.Client is optional — used in tests to inject a mock.
    """
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=FETCH_TIMEOUT)
    try:
        resp = client.get(url, timeout=FETCH_TIMEOUT)
    except httpx.HTTPError as e:
        raise AdvisoryFetchError(f"http error: {e}") from e
    finally:
        if own_client:
            client.close()
    if resp.status_code != 200:
        raise AdvisoryFetchError(f"status {resp.status_code}")
    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        raise AdvisoryFetchError(f"parse error: {parsed.bozo_exception!r}")

    out: dict[str, dict] = {}
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        m = _LEVEL_RE.match(title)
        if not m:
            continue
        country = _normalize_country(m.group("country"))
        try:
            level = int(m.group("level"))
        except ValueError:
            continue
        # If the feed lists the same country more than once, keep the
        # highest level. Rare but defensive.
        if country in out and out[country]["level"] >= level:
            continue
        out[country] = {"level": level, "title": title}
    return out


def _normalize_country(name: str) -> str:
    """Stable key from a country name. Uppercase ASCII with whitespace -> _."""
    return re.sub(r"\s+", "_", name.strip()).upper()


def diff_advisories(
    current: dict[str, dict],
    previous_state: dict[str, int],
) -> list[dict]:
    """Return per-country change records.

    A change means the level differs from the previous state, or a
    country appeared (previously absent, now advisory-listed). We
    intentionally do NOT emit changes for countries that dropped off
    the feed entirely — they'd produce "old_level=X, new_level=None"
    records, which aren't meaningful for alerting.

    Each record: {"country", "old_level", "new_level", "title"}.
    """
    changes: list[dict] = []
    for country, info in current.items():
        new_level = info["level"]
        old_level = previous_state.get(country)
        if old_level == new_level:
            continue
        changes.append({
            "country": country,
            "old_level": old_level,
            "new_level": new_level,
            "title": info.get("title", ""),
        })
    return changes


def snapshot_to_state(snapshot: dict[str, dict]) -> dict[str, int]:
    """Reduce a full fetch result to the per-country level mapping stored
    as advisory_state.json."""
    return {k: v["level"] for k, v in snapshot.items()}
