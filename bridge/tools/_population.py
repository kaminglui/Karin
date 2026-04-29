"""Population lookup tool — World Bank SP.POP.TOTL midyear estimates.

Same architecture as the inflation tool: deterministic, code does ALL
math, LLM only picks the tool + extracts args + paraphrases the
returned JSON. Reuses the same region keys as inflation so a future
``facts(year=YYYY)`` aggregator can query both side-by-side.

Coverage: 1960-current for the 5 inflation regions (us / hk_sar /
cn_mainland / japan / south_korea) plus a "world" aggregate (WLD).

Returned JSON shape (paraphrasing target — never invent values):

```
{
  "region":      "us",
  "country":     "United States",
  "year":        1985,
  "population":  238466000,
  "interpretation": "The United States had a population of about 238.5 million in 1985.",
  "source":      "World Bank Open Data (SP.POP.TOTL)",
  "source_url":  "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=USA",
  "data_as_of":  "2026-04-27",
  "data_range":  [1960, 2024]
}
```

When ``from_year`` and ``to_year`` are both supplied, also returns
``change_pct`` and ``change_abs`` so the LLM can paraphrase a "X grew
by Y% over Z years" comparison without doing the math itself.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("bridge.tools")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "population"
_ALL_COUNTRIES_PATH = _DATA_DIR / "pop_all_countries.json"

# Map our region keys to World Bank ISO3 codes for rank lookups.
_REGION_TO_ISO3: dict[str, str] = {
    "us": "USA",
    "hk_sar": "HKG",
    "cn_mainland": "CHN",
    "japan": "JPN",
    "south_korea": "KOR",
    # "world" is the aggregate, not a country — no rank entry.
}

# Keep the keys identical to _inflation._REGIONS so a `facts` tool can
# pass the same region key to both. "world" is population-specific
# (the inflation tool has no world CPI aggregate).
_REGIONS: dict[str, dict] = {
    "world": {
        "label": "World",
        "path": _DATA_DIR / "pop_world.json",
    },
    "us": {
        "label": "United States",
        "path": _DATA_DIR / "pop_us.json",
    },
    "hk_sar": {
        "label": "Hong Kong SAR, China",
        "path": _DATA_DIR / "pop_hk_sar.json",
    },
    "cn_mainland": {
        "label": "China (Mainland)",
        "path": _DATA_DIR / "pop_cn_mainland.json",
    },
    "japan": {
        "label": "Japan",
        "path": _DATA_DIR / "pop_japan.json",
    },
    "south_korea": {
        "label": "South Korea",
        "path": _DATA_DIR / "pop_south_korea.json",
    },
}

# Region detection — mirrors the inflation extractor so phrasings work
# the same way. "world" / "global" / "earth" detect the world aggregate.
_REGION_DETECT_PATTERNS: list[tuple[str, str]] = [
    (r"\bworld(?:wide)?\s+population\b|\bglobal\s+population\b|\bpopulation\s+of\s+(?:the\s+)?(?:world|earth|planet)\b", "world"),
    (r"\bhong\s*kong\s+sar\b", "hk_sar"),
    (r"\bhong\s*kong\b", "hk_sar"),
    (r"\bin\s+hk\b", "hk_sar"),
    (r"\bmainland\s+china\b|\bchina\s+mainland\b|\bprc\b", "cn_mainland"),
    (r"\b(?<!south\s)(?<!east\s)(?<!west\s)china\b", "cn_mainland"),
    (r"\bjapan(?:ese)?\b", "japan"),
    (r"\bsouth\s+korea(?:n)?\b|\brok\b", "south_korea"),
    (r"\b(?<!north\s)korea\b", "south_korea"),
    (r"\b(?:united\s+states|america|usa|u\.?s\.?a?)\b", "us"),
    # "earth" / "world" without an explicit "population" anchor —
    # safer to match only when the user clearly asked for population.
    (r"\b(?:earth|world|planet)\b", "world"),
]

_REGION_ALIASES: dict[str, str] = {
    "us": "us", "usa": "us", "u.s.": "us", "u.s.a.": "us",
    "united states": "us", "america": "us", "american": "us",
    "world": "world", "global": "world", "earth": "world", "planet": "world",
    "hk": "hk_sar", "hong kong": "hk_sar", "hong kong sar": "hk_sar", "hkg": "hk_sar",
    "china": "cn_mainland", "mainland china": "cn_mainland", "china mainland": "cn_mainland",
    "prc": "cn_mainland", "chn": "cn_mainland",
    "japan": "japan", "japanese": "japan", "jpn": "japan",
    "korea": "south_korea", "south korea": "south_korea",
    "rok": "south_korea", "kor": "south_korea",
}


def _load(region: str) -> dict[str, Any]:
    cfg = _REGIONS.get(region)
    if cfg is None:
        raise ValueError(f"Unknown region {region!r}. Available: {sorted(_REGIONS)}.")
    path: Path = cfg["path"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Population dataset for region={region!r} missing at {path}. "
            f"Run scripts/fetch_worldbank_population.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _format_population(n: int) -> str:
    """Human-readable population: 238466000 → '238.5 million'."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} billion"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} million"
    if n >= 1_000:
        return f"{n / 1_000:.1f} thousand"
    return str(n)


def _population(
    region: str = "world",
    year: int | str | None = None,
    from_year: int | str | None = None,
    to_year: int | str | None = None,
    metric: str = "value",
    top: int | str = 10,
) -> str:
    """Look up population for a region at a year, or compute change
    between two years, or rank countries by population.

    Args:
        region: "world" (default), "us", "hk_sar", "cn_mainland",
                "japan", or "south_korea".
        year: single-year query — population at this year.
        from_year + to_year: range query — population at both years
                             plus computed change_pct and change_abs.
        metric: "value" (default) for the population number; "rank"
                for top-N most populous countries plus the queried
                region's rank.
        top: when metric="rank", how many countries to return (default 10).

    Returns a JSON string the LLM paraphrases. Never invent numbers.
    """
    region = (region or "world").strip().lower()
    region = _REGION_ALIASES.get(region, region)
    metric = (metric or "value").strip().lower()

    if metric == "rank":
        return _population_rank(region, year, top)

    cfg = _REGIONS.get(region)
    if cfg is None:
        return json.dumps({
            "error": (
                f"Unknown region {region!r}. Available: "
                f"{', '.join(sorted(_REGIONS))}."
            ),
        })

    try:
        data = _load(region)
    except (FileNotFoundError, ValueError) as e:
        log.error("population tool: %s", e)
        return json.dumps({"error": str(e)})

    annual = data["annual"]
    available = sorted(int(y) for y in annual.keys())
    floor, ceil = available[0], available[-1]

    # If from_year + to_year given, do range query; else single-year.
    if from_year is not None and to_year is not None:
        try:
            fy = int(from_year)
            ty = int(to_year)
        except (TypeError, ValueError):
            return json.dumps({
                "error": f"from_year/to_year must be integers, got {from_year!r}/{to_year!r}",
            })
        if fy == ty:
            year = fy  # collapse to single-year path
            from_year = to_year = None
        else:
            for y in (fy, ty):
                if y < floor or y > ceil:
                    return json.dumps({
                        "error": (
                            f"year={y} is outside {cfg['label']} data range "
                            f"({floor}-{ceil})."
                        ),
                    })
            pop_from = int(annual[str(fy)])
            pop_to = int(annual[str(ty)])
            change_abs = pop_to - pop_from
            change_pct = round((pop_to - pop_from) / pop_from * 100, 2) if pop_from else 0.0
            grew_or_shrank = "grew" if change_abs >= 0 else "shrank"
            return json.dumps({
                "region": region,
                "country": cfg["label"],
                "from_year": fy,
                "to_year": ty,
                "population_from": pop_from,
                "population_to": pop_to,
                "change_abs": change_abs,
                "change_pct": change_pct,
                "interpretation": (
                    f"{cfg['label']}'s population {grew_or_shrank} from "
                    f"{_format_population(pop_from)} in {fy} to "
                    f"{_format_population(pop_to)} in {ty} — a "
                    f"{change_pct:+.2f}% change ({change_abs:+,} people)."
                ),
                "source": "World Bank Open Data (SP.POP.TOTL)",
                "source_url": data["_metadata"]["source_url_primary"],
                "data_as_of": data["_metadata"]["data_as_of"],
                "data_range": [floor, ceil],
            }, ensure_ascii=False)

    # Single-year path.
    if year is None:
        year = ceil  # default to latest
    try:
        year = int(year)
    except (TypeError, ValueError):
        return json.dumps({"error": f"year must be an integer, got {year!r}"})
    if year < floor or year > ceil:
        return json.dumps({
            "error": (
                f"year={year} is outside {cfg['label']} data range "
                f"({floor}-{ceil})."
            ),
        })
    pop = int(annual[str(year)])
    log.info("population: %s @ %d -> %s", region, year, _format_population(pop))
    return json.dumps({
        "region": region,
        "country": cfg["label"],
        "year": year,
        "population": pop,
        "interpretation": (
            f"{cfg['label']} had a population of about {_format_population(pop)} "
            f"in {year}."
        ),
        "source": "World Bank Open Data (SP.POP.TOTL)",
        "source_url": data["_metadata"]["source_url_primary"],
        "data_as_of": data["_metadata"]["data_as_of"],
        "data_range": [floor, ceil],
    }, ensure_ascii=False)


def _population_rank(
    region: str, year: int | str | None, top: int | str = 10,
) -> str:
    """Return the top-N most populous countries for a year, plus the
    queried region's rank within that list. Loads from the cached
    all-countries file (scripts/fetch_worldbank_pop_all.py)."""
    if not _ALL_COUNTRIES_PATH.is_file():
        return json.dumps({
            "error": (
                f"All-countries population dataset missing at "
                f"{_ALL_COUNTRIES_PATH}. Run "
                f"scripts/fetch_worldbank_pop_all.py."
            ),
        })
    data = json.loads(_ALL_COUNTRIES_PATH.read_text(encoding="utf-8"))
    countries = data.get("countries") or {}
    if not countries:
        return json.dumps({"error": "all-countries dataset is empty"})

    try:
        top_n = max(1, min(50, int(top)))  # clamp to [1, 50]
    except (TypeError, ValueError):
        top_n = 10

    # Year resolution: default to the most recent year present in the dataset.
    if year is None:
        all_years: set[int] = set()
        for entry in countries.values():
            for y in entry.get("annual", {}).keys():
                try:
                    all_years.add(int(y))
                except ValueError:
                    pass
        if not all_years:
            return json.dumps({"error": "no year data found in dataset"})
        target_year = max(all_years)
    else:
        try:
            target_year = int(year)
        except (TypeError, ValueError):
            return json.dumps({"error": f"year must be an integer, got {year!r}"})

    # Build (iso3, name, region, population) tuples for everyone with data.
    rows: list[tuple[str, str, str, int]] = []
    for iso3, entry in countries.items():
        annual = entry.get("annual") or {}
        v = annual.get(str(target_year))
        if v is None:
            continue
        try:
            rows.append((iso3, entry.get("name", iso3),
                         entry.get("region", ""), int(v)))
        except (TypeError, ValueError):
            continue
    if not rows:
        return json.dumps({
            "error": f"no country data found for year={target_year}",
        })
    rows.sort(key=lambda r: r[3], reverse=True)
    total_countries = len(rows)

    top_list = [
        {
            "rank": i + 1,
            "iso3": r[0],
            "country": r[1],
            "region": r[2],
            "population": r[3],
        }
        for i, r in enumerate(rows[:top_n])
    ]

    # Where does the queried region sit?
    region_position = None
    target_iso3 = _REGION_TO_ISO3.get(region)
    if target_iso3:
        for i, r in enumerate(rows):
            if r[0] == target_iso3:
                region_position = {
                    "region": region,
                    "iso3": target_iso3,
                    "country": r[1],
                    "rank": i + 1,
                    "population": r[3],
                    "out_of": total_countries,
                }
                break

    # Build the interpretation paragraph the LLM paraphrases.
    leader = top_list[0]
    leader_pop = leader["population"]
    leader_phrase = (
        f"{leader['country']} ({_format_population(leader_pop)})"
    )
    interp = (
        f"In {target_year}, the most populous country was "
        f"{leader_phrase}. Top {top_n}: " +
        ", ".join(f"{x['rank']}. {x['country']}" for x in top_list) +
        f". {total_countries} sovereign countries with data."
    )
    if region_position:
        interp += (
            f" {region_position['country']} ranks "
            f"{region_position['rank']} of {region_position['out_of']} "
            f"with {_format_population(region_position['population'])}."
        )

    log.info(
        "population rank: year=%d top=%d region=%s region_rank=%s",
        target_year, top_n, region,
        region_position["rank"] if region_position else None,
    )
    return json.dumps({
        "year": target_year,
        "top": top_list,
        "total_countries": total_countries,
        "region_position": region_position,
        "interpretation": interp,
        "source": "World Bank Open Data (SP.POP.TOTL)",
        "source_url": data["_metadata"].get("source_url_primary", ""),
        "data_as_of": data["_metadata"].get("data_as_of", ""),
    }, ensure_ascii=False)


def population_widget_data(
    region: str = "world",
    year: int | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
) -> dict[str, Any]:
    """Build a widget-friendly dict: same calc the LLM tool returns
    plus the full year-by-year series for plotting."""
    region = (region or "world").strip().lower()
    region = _REGION_ALIASES.get(region, region)
    raw = _population(
        region=region, year=year, from_year=from_year, to_year=to_year,
    )
    calc = json.loads(raw)
    if "error" in calc:
        return calc
    try:
        data = _load(region)
    except (FileNotFoundError, ValueError):
        return calc
    cfg = _REGIONS.get(region) or {}
    annual = data["annual"]
    years_sorted = sorted(int(y) for y in annual.keys())
    series = {
        "years": years_sorted,
        "values": [int(annual[str(y)]) for y in years_sorted],
        "label": f"Population ({cfg.get('label', region)})",
    }
    out = dict(calc)
    out["series"] = series
    return out


# Year + region pre-extractor for the bridge merge layer.

_YEAR_RE = re.compile(r"\b(\d{4})\b")
_YEAR_FROM_TO_RE = re.compile(
    r"\b(?:from|between)\s+(\d{4})\s+(?:to|and|through)\s+(\d{4})\b",
    re.IGNORECASE,
)
# Rank-query indicators. Presence promotes metric→"rank".
_RANK_RE = re.compile(
    r"\bmost\s+populous\b|"
    r"\b(?:biggest|largest)\s+(?:countr(?:y|ies)|nations?)\b|"
    r"\bpopulation\s+rank(?:ing)?s?\b|"
    r"\btop\s+\d+\s+(?:countr(?:y|ies)|most\s+populous)\b|"
    r"\bwhere\s+does\s+\w+\s+rank\b|"
    r"\branks?\s+\d+(?:st|nd|rd|th)?\s+(?:in\s+population|by\s+population)\b",
    re.IGNORECASE,
)
# "Top N" extraction for the count.
_TOP_N_RE = re.compile(r"\btop\s+(\d{1,2})\b", re.IGNORECASE)


def extract_population_args(user_text: str | None) -> dict[str, Any]:
    """Pull region + year(s) hints out of the user message. Same role
    as extract_inflation_args — fills missing args the LoRA omitted."""
    if not user_text:
        return {}
    text = user_text.strip()
    text_lower = text.lower()
    out: dict[str, Any] = {}

    # Region
    for pat, key in _REGION_DETECT_PATTERNS:
        if re.search(pat, text_lower):
            out["region"] = key
            break

    # Range query first ("from 1990 to 2020", "between 1980 and 2010")
    m = _YEAR_FROM_TO_RE.search(text)
    if m:
        try:
            out["from_year"] = int(m.group(1))
            out["to_year"] = int(m.group(2))
            return out
        except ValueError:
            pass

    # Single-year query — pick the first 4-digit number that's a
    # plausible year (1700-2100).
    for ym in _YEAR_RE.finditer(text):
        try:
            y = int(ym.group(1))
        except ValueError:
            continue
        if 1700 <= y <= 2100:
            out["year"] = y
            break

    # Rank inference — explicit phrasings promote metric=rank, and a
    # "top N" pulls out the N. Don't override the LoRA's metric if it
    # already said something else.
    if _RANK_RE.search(text):
        out["metric"] = "rank"
        m = _TOP_N_RE.search(text)
        if m:
            try:
                out["top"] = int(m.group(1))
            except ValueError:
                pass

    return out
