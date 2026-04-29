"""Year-card aggregator — "1985 in numbers" style facts.

Composes population + inflation data into a single packaged response
so the LLM can paraphrase a curated context card for a year. No
nested tool calls — the aggregator loads the underlying JSON files
directly (same source-of-truth as the inflation and population
tools), computes the comparisons, and returns one structured dict.

Architecture:
- Reuses the same region keys as inflation + population
  ("us", "hk_sar", "cn_mainland", "japan", "south_korea", "world")
- Default region: "us" for inflation (CPI baseline), "world" for the
  population headline (with the user's region as a secondary card if
  specified).
- v1: world population, region population, inflation baseline ($1 in
  YEAR ≈ $X today), CPI value, source citations.
- v2 (this iteration): cohort-age compute, wages snapshot (US 1964+),
  item-price highlights (US 1980+), wiki year-article snippet.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("bridge.tools")


def _facts(year: int | str, region: str | None = None) -> str:
    """Build a packaged 'year in numbers' card for the given year.

    Args:
        year: the year to summarize (integer; required).
        region: optional region for the regional inflation + population
                breakdowns. Default behavior: world population + US
                inflation (the baseline most users expect). Set to e.g.
                "japan" to get Japan population + Japan CPI alongside
                the world baseline.

    Returns a JSON string containing a curated aggregate. The LLM
    paraphrases — never invent values not present in the response.
    """
    # Defer to the underlying tool implementations so all the validation
    # (range checks, unknown-region rejection, source citations) is
    # consistent across tools without duplicating logic.
    from bridge.tools._inflation import _inflation, _REGIONS as INFL_REGIONS
    from bridge.tools._population import _population, _REGIONS as POP_REGIONS

    try:
        year = int(year)
    except (TypeError, ValueError):
        return json.dumps({"error": f"year must be an integer, got {year!r}"})

    region = (region or "").strip().lower() or None

    out: dict[str, Any] = {
        "year": year,
        "region": region,
        "sections": {},
        "interpretation_lines": [],
    }

    # World population — always include; this is the universal anchor.
    # Also fetch the latest year so the widget can show a then-vs-now
    # comparison (e.g. "4.85B in 1985 → 8.14B today, +68%").
    pop_world_raw = _population(region="world", year=year)
    pop_world = json.loads(pop_world_raw)
    if "error" not in pop_world:
        pop_world_now_raw = _population(region="world")  # latest by default
        pop_world_now = json.loads(pop_world_now_raw)
        if "error" not in pop_world_now:
            pop_world["now"] = _delta_payload(
                pop_world["population"], pop_world_now["population"],
                pop_world["year"], pop_world_now["year"],
            )
            pop_world["now"]["population"] = pop_world_now["population"]
            pop_world["now"]["year"] = pop_world_now["year"]
        out["sections"]["world_population"] = pop_world
        out["interpretation_lines"].append(pop_world["interpretation"])
    else:
        out["sections"]["world_population"] = {"error": pop_world["error"]}

    # Region-specific population (if region was given AND it's known
    # to the population tool — population has region "us"/"world"/etc.).
    if region and region in POP_REGIONS and region != "world":
        pop_region_raw = _population(region=region, year=year)
        pop_region = json.loads(pop_region_raw)
        if "error" not in pop_region:
            pop_region_now_raw = _population(region=region)
            pop_region_now = json.loads(pop_region_now_raw)
            if "error" not in pop_region_now:
                pop_region["now"] = _delta_payload(
                    pop_region["population"], pop_region_now["population"],
                    pop_region["year"], pop_region_now["year"],
                )
                pop_region["now"]["population"] = pop_region_now["population"]
                pop_region["now"]["year"] = pop_region_now["year"]
            out["sections"]["region_population"] = pop_region
            out["interpretation_lines"].append(pop_region["interpretation"])

    # Inflation baseline — "$1 in YEAR vs today" for the region's
    # currency. If no region given, use US (the BLS CPI-U baseline).
    infl_region = region if region in INFL_REGIONS else "us"
    infl_raw = _inflation(amount=1.0, from_year=year, region=infl_region)
    infl = json.loads(infl_raw)
    if "error" not in infl:
        out["sections"]["inflation_baseline"] = {
            "region": infl["region"],
            "country": infl["country"],
            "currency": infl["currency"],
            "amount_input": infl["amount_input"],
            "from_year": infl["from_year"],
            "to_year": infl["to_year"],
            "amount_output": infl["cpi"]["amount_output"],
            "ratio": infl["cpi"]["ratio"],
            "confidence": infl["cpi"]["confidence"],
            "source": infl["cpi"]["source"],
            "source_url": infl["cpi"]["source_url"],
            "interpretation": infl["interpretation"],
        }
        out["interpretation_lines"].append(infl["interpretation"])
    else:
        out["sections"]["inflation_baseline"] = {"error": infl["error"]}

    # --- v2 sections ----------------------------------------------------

    # Cohort age — trivial compute, big "interesting" payoff. Skip for
    # future years (cohort doesn't exist yet) and ridiculous past
    # years (older than ~125 are all dead, so the framing is morbid).
    from datetime import date as _date
    today_year = _date.today().year
    age_now = today_year - year
    if 0 < age_now <= 120:
        out["sections"]["cohort_age"] = {
            "year_born": year,
            "current_year": today_year,
            "age": age_now,
        }
        out["interpretation_lines"].append(
            f"People born in {year} are about {age_now} years old today."
        )

    # Wages snapshot — US AHETPI hourly wages, 1964+. Single point.
    wages_block = _wages_snapshot(year)
    if wages_block:
        out["sections"]["wages_snapshot"] = wages_block
        out["interpretation_lines"].append(wages_block["interpretation"])

    # Item-price highlights — US BLS Average Price Series, 1980+.
    # Pick a few well-known items so the year card surfaces concrete,
    # memorable prices ("a gallon of gas was $1.20").
    items_block = _item_highlights(year)
    if items_block:
        out["sections"]["item_highlights"] = items_block
        out["interpretation_lines"].append(items_block["interpretation"])

    # Wiki year-article snippet — public-domain extract from the
    # dedicated "YYYY" Wikipedia article. Surfaces notable events.
    wiki_block = _wiki_year_snippet(year)
    if wiki_block:
        out["sections"]["wiki_year"] = wiki_block
        if wiki_block.get("extract"):
            out["interpretation_lines"].append(wiki_block["extract"])

    # Top-line summary for paraphrasing — joins all interpretation
    # sentences into one coherent paragraph the LLM can lightly rewrite.
    if out["interpretation_lines"]:
        out["interpretation"] = " ".join(out["interpretation_lines"])
    else:
        out["interpretation"] = (
            f"{year} is outside the loaded data ranges — try a year "
            f"between 1960 and the current year."
        )

    log.info(
        "facts: year=%d region=%s sections=%s",
        year, region, list(out["sections"].keys()),
    )
    return json.dumps(out, ensure_ascii=False)


# --- v2 helpers ------------------------------------------------------

# Items to surface in the year card. Order matters — picked for memorability.
# All US BLS AP series with 1980+ coverage. Keys must match _ITEM_PHRASE_MAP
# entries in _inflation.py so _resolve_item finds them.
_FACTS_ITEM_KEYS: tuple[str, ...] = ("gasoline", "bread", "eggs", "milk_gallon", "coffee")


def _delta_payload(then_value: float, now_value: float,
                   then_year: int, now_year: int) -> dict:
    """Build a then-vs-now comparison payload. Used by every section
    that has a queried-year point and a current-year point."""
    if then_value <= 0 or now_value is None:
        return {"change_abs": None, "change_pct": None, "ratio": None}
    diff = now_value - then_value
    pct = (diff / then_value * 100) if then_value else None
    return {
        "change_abs": round(diff) if isinstance(diff, (int, float)) and abs(diff) >= 1 else round(diff, 4),
        "change_pct": round(pct, 2) if pct is not None else None,
        "ratio": round(now_value / then_value, 4) if then_value else None,
        "years_span": now_year - then_year,
    }


def _wages_snapshot(year: int) -> dict | None:
    """US AHETPI hourly wage at the given year + the latest year's
    wage for then-vs-now comparison."""
    from bridge.tools._inflation import _load_wages
    wages = _load_wages()
    if wages is None:
        return None
    annual = wages.get("annual", {})
    if str(year) not in annual:
        return None
    wage = float(annual[str(year)])
    # Latest year in the dataset for the "today" comparison.
    available = sorted(int(y) for y in annual.keys())
    now_year = available[-1]
    wage_now = float(annual[str(now_year)])
    delta = _delta_payload(wage, wage_now, year, now_year)
    return {
        "year": year,
        "wage_hourly_usd": round(wage, 2),
        "now": {
            "year": now_year,
            "wage_hourly_usd": round(wage_now, 2),
            **delta,
        },
        "source": wages["_metadata"].get("series_name", "BLS AHETPI"),
        "source_url": wages["_metadata"].get("source_url_primary", ""),
        "interpretation": (
            f"In {year}, the average production-worker hourly wage in "
            f"the US was about ${wage:.2f} (vs ${wage_now:.2f} today, "
            f"a {delta['ratio']:.2f}× nominal increase)."
        ),
    }


def _item_highlights(year: int) -> dict | None:
    """US item prices for the year, picked for memorability, each with
    the latest available price for then-vs-now comparison."""
    from bridge.tools._inflation import _load_items, _resolve_item
    items_data = _load_items()
    if items_data is None:
        return None
    picks: list[dict] = []
    for key in _FACTS_ITEM_KEYS:
        entry = _resolve_item(key, items_data)
        if entry is None:
            continue
        annual = entry.get("annual", {})
        if str(year) not in annual:
            continue
        price = float(annual[str(year)])
        # Latest year for comparison.
        years_avail = sorted(int(y) for y in annual.keys())
        now_year = years_avail[-1]
        price_now = float(annual[str(now_year)])
        delta = _delta_payload(price, price_now, year, now_year)
        picks.append({
            "key": entry["key"],
            "label": entry["label"],
            "unit": entry["unit"],
            "price": round(price, 2),
            "price_now": round(price_now, 2),
            "now_year": now_year,
            "ratio": delta["ratio"],
            "change_pct": delta["change_pct"],
        })
    if not picks:
        return None
    # Top 3 inline summary.
    parts = [
        f"{p['label'].split(',')[0].strip()} ${p['price']:.2f} → ${p['price_now']:.2f} ({p['ratio']:.2f}×)"
        for p in picks[:3]
    ]
    now_year_any = picks[0]["now_year"] if picks else year
    return {
        "year": year,
        "now_year": now_year_any,
        "items": picks,
        "source": "BLS Average Price Series",
        "source_url": "https://www.bls.gov/cpi/factsheets/average-prices.htm",
        "interpretation": (
            f"US prices in {year} vs {now_year_any}: " + "; ".join(parts) + "."
        ),
    }


_WIKI_HEADERS = {
    "User-Agent": (
        "Karin/1.0 (https://github.com/kaminglui/Karin; "
        "kaminglui+karin@users.noreply.github.com) httpx"
    ),
    "Accept": "application/json",
}


# How long to remember a Wikipedia fetch FAILURE before retrying.
# Successes are cached forever (year articles change slowly); failures
# get a short TTL so a transient 503 or network blip during one boot
# doesn't permanently mask that year for the lifetime of the process.
_WIKI_FAIL_TTL_SECONDS = 600


def _wiki_year_snippet(year: int) -> dict | None:
    """Fetch notable Events from the "YYYY" Wikipedia article and
    return a list of 3 highlights, with a fallback to the lead summary
    when section parsing fails. Cached process-local — successful year
    articles change slowly, so re-fetching them is wasteful. Failures
    are cached only for ``_WIKI_FAIL_TTL_SECONDS`` so a transient
    network error during one boot doesn't permanently disable that
    year."""
    import time
    cache = _wiki_year_snippet._cache
    entry = cache.get(year)
    if entry is not None:
        if entry["ok"]:
            return entry["block"]
        # Negative cache — recheck after TTL.
        if time.time() < entry["expires_at"]:
            return None
    try:
        import httpx  # noqa: F401  — re-checks availability before fetch
    except ImportError:
        return None
    title = str(year)
    base_url = f"https://en.wikipedia.org/wiki/{title}"

    events = _fetch_year_events(year, title)
    if events:
        block = {
            "title": title,
            "events": events,
            # Render the events as a single sentence for paraphrasing.
            "extract": (
                f"Notable events in {year}: " +
                "; ".join(events[:3]) +
                "."
            ),
            "source": "Wikipedia",
            "source_url": base_url,
        }
        cache[year] = {"ok": True, "block": block}
        return block

    # Fallback: the REST summary endpoint (calendar lead). Better than
    # nothing — surfaces the article link so the user can click through.
    summary = _fetch_year_summary(title)
    if summary:
        block = {
            "title": summary.get("title", title),
            "extract": summary.get("extract", ""),
            "source": "Wikipedia",
            "source_url": summary.get("source_url", base_url),
        }
        cache[year] = {"ok": True, "block": block}
        return block

    cache[year] = {"ok": False, "expires_at": time.time() + _WIKI_FAIL_TTL_SECONDS}
    return None


_wiki_year_snippet._cache = {}  # type: ignore[attr-defined]


# Match leading "* " or "** " bullet lines; capture rest after stripping
# wikilinks/templates. Used to extract events from the Events section's
# wikitext.
_BULLET_RE = re.compile(r"^\*+\s*(.+?)\s*$")
_WIKILINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_REF_RE = re.compile(r"<ref[^/]*/>|<ref[^>]*>.*?</ref>", re.DOTALL)


def _clean_wikitext_line(line: str) -> str:
    """Strip references, templates, wikilink syntax, HTML tags, common
    HTML entities, and bold/italic markup from a wikitext bullet line."""
    line = _REF_RE.sub("", line)
    line = _TEMPLATE_RE.sub("", line)
    # Replace [[Link|Display]] with Display, [[Link]] with Link.
    line = _WIKILINK_RE.sub(lambda m: m.group(1), line)
    line = _HTML_TAG_RE.sub("", line)
    # Common HTML entities used in year articles.
    line = (line
            .replace("&ndash;", "–")
            .replace("&mdash;", "—")
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&#39;", "'"))
    # Wikitext bold (''') and italic (''). Strip the markup, keep text.
    line = re.sub(r"'''(.+?)'''", r"\1", line)
    line = re.sub(r"''(.+?)''", r"\1", line)
    # Normalize whitespace.
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _fetch_year_events(year: int, title: str) -> list[str]:
    """Use the MediaWiki Action API to fetch the "Events" section of
    the year article and extract a few bullet entries.

    Returns a list of cleaned-up event strings (most-notable first per
    Wikipedia's lead-events ordering when present, otherwise just the
    first three from January). Empty list on any failure path so the
    caller can fall back to the REST summary."""
    try:
        import httpx
    except ImportError:
        return []
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True,
                          headers=_WIKI_HEADERS) as client:
            # Step 1: section list.
            r = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "parse",
                    "page": title,
                    "format": "json",
                    "prop": "sections",
                    "redirects": 1,
                },
            )
            if r.status_code != 200:
                return []
            j = r.json()
            sections = (j.get("parse") or {}).get("sections") or []
            # Find the top-level "Events" section (toclevel == 1).
            target_idx = None
            for s in sections:
                if (s.get("line") or "").strip().lower() == "events" and \
                        s.get("toclevel") == 1:
                    target_idx = s.get("index")
                    break
            if target_idx is None:
                return []
            # Step 2: wikitext of that section.
            r2 = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "parse",
                    "page": title,
                    "format": "json",
                    "prop": "wikitext",
                    "section": str(target_idx),
                    "redirects": 1,
                },
            )
            if r2.status_code != 200:
                return []
            wt = ((r2.json().get("parse") or {}).get("wikitext") or {}).get("*", "")
            if not wt:
                return []
    except Exception as e:
        log.warning("wiki events fetch for %d failed: %s", year, e)
        return []

    # Pull bullet lines until we have ~3 usable ones.
    out: list[str] = []
    for raw in wt.splitlines():
        m = _BULLET_RE.match(raw)
        if not m:
            continue
        cleaned = _clean_wikitext_line(m.group(1))
        # Skip very short lines (often residual markup) or section
        # headers that snuck through. Keep things that read like
        # event statements.
        if len(cleaned) < 20:
            continue
        # Drop trailing reference markers like " [1][2]"
        cleaned = re.sub(r"\s*\[\d+\]\s*$", "", cleaned)
        out.append(cleaned)
        if len(out) >= 3:
            break
    return out


def _fetch_year_summary(title: str) -> dict | None:
    """REST v1 summary fallback when the Events parse fails."""
    try:
        import httpx
    except ImportError:
        return None
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True,
                          headers=_WIKI_HEADERS) as client:
            r = client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            )
            if r.status_code != 200:
                return None
            j = r.json()
            extract = (j.get("extract") or "").strip()
            if not extract:
                return None
            sentences = extract.split(". ")
            short = ". ".join(sentences[:2])
            if not short.endswith("."):
                short += "."
            return {
                "title": j.get("title", title),
                "extract": short,
                "source_url": (
                    j.get("content_urls", {})
                    .get("desktop", {}).get("page")
                    or f"https://en.wikipedia.org/wiki/{title}"
                ),
            }
    except Exception as e:
        log.warning("wiki year summary fetch for %s failed: %s", title, e)
        return None


# Pre-extractor for the bridge merge layer. Pulls just `year` (and
# optional `region`) out of the user message — facts is simpler than
# inflation/population since it has only those two args.

import re

_YEAR_RE = re.compile(r"\b(\d{4})\b")
_REGION_PHRASES: list[tuple[str, str]] = [
    (r"\bworld(?:wide)?\b|\bglobal\b|\bearth\b", "world"),
    (r"\bhong\s*kong\s+sar\b|\bhong\s*kong\b", "hk_sar"),
    (r"\bmainland\s+china\b|\bchina\s+mainland\b|\bprc\b", "cn_mainland"),
    (r"\b(?<!south\s)(?<!east\s)(?<!west\s)china\b", "cn_mainland"),
    (r"\bjapan(?:ese)?\b", "japan"),
    (r"\bsouth\s+korea(?:n)?\b|\brok\b", "south_korea"),
    (r"\b(?<!north\s)korea\b", "south_korea"),
    (r"\b(?:united\s+states|america|usa|u\.?s\.?a?)\b", "us"),
]


def extract_facts_args(user_text: str | None) -> dict[str, Any]:
    """Pull year + optional region from user text. Year is required;
    if absent, return {} (facts tool needs it)."""
    if not user_text:
        return {}
    text = user_text.strip()
    text_lower = text.lower()
    out: dict[str, Any] = {}
    for ym in _YEAR_RE.finditer(text):
        try:
            y = int(ym.group(1))
        except ValueError:
            continue
        if 1700 <= y <= 2100:
            out["year"] = y
            break
    for pat, key in _REGION_PHRASES:
        if re.search(pat, text_lower):
            out["region"] = key
            break
    return out
