"""Fetch World Bank SP.POP.TOTL for ALL sovereign countries (217)
across 1960-current, plus the country metadata needed to filter out
aggregates (regions, income groups, etc.). Writes a single file:

    bridge/data/population/pop_all_countries.json

Used by the population tool's `metric="rank"` mode to answer "top N
most populous in YYYY" / "where does Japan rank?" queries without
hitting the API per request. Ranks are computed at query time from
this cache, so the file ships annual values per country and the
rank logic lives in `bridge/tools/_population.py`.

Usage:
    python scripts/fetch_worldbank_pop_all.py
    python scripts/fetch_worldbank_pop_all.py --full
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "population" / "pop_all_countries.json"
)
INDICATOR = "SP.POP.TOTL"
DATA_FLOOR = 1960
REVISION_LOOKBACK_YEARS = 2


def fetch_country_metadata() -> dict[str, dict]:
    """Return {iso3: {name, region}} for sovereign countries only.
    World Bank flags aggregates with region.id == 'NA'."""
    from _fetch_helpers import http_request
    url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    body = http_request(url, timeout=60)
    out: dict[str, dict] = {}
    for rec in (body[1] or []):
        region = (rec.get("region") or {}).get("id")
        if region == "NA":
            continue  # aggregate, not a country
        iso3 = rec.get("id")
        if not iso3 or len(iso3) != 3:
            continue
        out[iso3] = {
            "name": rec.get("name", iso3),
            "region": (rec.get("region") or {}).get("value", ""),
        }
    return out


def fetch_population_range(start: int, end: int) -> list[dict]:
    """Pull all-country population for a year range. Returns the raw
    record list."""
    from _fetch_helpers import http_request
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{INDICATOR}"
        f"?date={start}:{end}&format=json&per_page=20000"
    )
    body = http_request(url, timeout=120)
    if not isinstance(body, list) or len(body) < 2 or body[1] is None:
        raise RuntimeError(f"unexpected shape: {body!r}")
    return body[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument(
        "--full",
        action="store_true",
        help=(
            "Force a complete rebuild from %d. Default: incremental — "
            "fetch only the last %d years and merge into existing JSON."
            % (DATA_FLOOR, REVISION_LOOKBACK_YEARS + 1)
        ),
    )
    args = p.parse_args()

    existing: dict = {}
    incremental = False
    if args.out.is_file() and not args.full:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        incremental = bool(existing.get("countries"))

    if incremental:
        start = max(DATA_FLOOR, date.today().year - REVISION_LOOKBACK_YEARS)
        print(
            f"Incremental refresh from {start} (existing has "
            f"{len(existing.get('countries', {}))} countries).",
            file=sys.stderr,
        )
    else:
        start = DATA_FLOOR
        print(
            f"Full backfill all-country population from {DATA_FLOOR} ...",
            file=sys.stderr,
        )

    from _fetch_helpers import write_status
    print("Fetching country metadata ...", file=sys.stderr)
    try:
        meta = fetch_country_metadata()
    except Exception as e:
        write_status(args.out.parent, "fetch_worldbank_pop_all", "metadata",
                     ok=False, error=str(e))
        raise
    print(f"  {len(meta)} sovereign countries", file=sys.stderr)

    print(f"Fetching population {start}-{date.today().year} ...", file=sys.stderr)
    try:
        raw = fetch_population_range(start, date.today().year)
    except Exception as e:
        write_status(args.out.parent, "fetch_worldbank_pop_all", "all_countries",
                     ok=False, error=str(e))
        raise
    print(f"  {len(raw)} raw records", file=sys.stderr)

    # Group raw records into {iso3: {year: value}}, filtered to known
    # sovereign countries.
    fresh: dict[str, dict[str, int]] = {}
    for rec in raw:
        iso3 = rec.get("countryiso3code")
        if iso3 not in meta:
            continue
        try:
            y = int(rec["date"])
            v = rec.get("value")
        except (KeyError, ValueError, TypeError):
            continue
        if v is None:
            continue
        try:
            fresh.setdefault(iso3, {})[str(y)] = int(round(float(v)))
        except (TypeError, ValueError):
            continue

    # Merge: existing[countries][iso3].annual gets new years overlaid.
    countries: dict[str, dict] = dict(existing.get("countries", {}) or {})
    for iso3 in meta:
        prev = countries.get(iso3) or {"name": meta[iso3]["name"],
                                       "region": meta[iso3]["region"],
                                       "annual": {}}
        # Always refresh name/region from latest metadata.
        prev["name"] = meta[iso3]["name"]
        prev["region"] = meta[iso3]["region"]
        prev_annual = dict(prev.get("annual") or {})
        new_annual = fresh.get(iso3) or {}
        prev_annual.update(new_annual)
        prev["annual"] = prev_annual
        countries[iso3] = prev

    # Drop any old entries that aren't in current metadata (rare —
    # country code changes are uncommon).
    countries = {k: v for k, v in countries.items() if k in meta}

    payload = {
        "_metadata": {
            "indicator": INDICATOR,
            "source_primary": "World Bank Open Data",
            "source_url_primary": (
                f"https://data.worldbank.org/indicator/{INDICATOR}"
            ),
            "data_as_of": date.today().isoformat(),
            "country_count": len(countries),
            "year_range": [DATA_FLOOR, date.today().year],
            "notes": (
                "Sovereign countries only (World Bank aggregates / income "
                "groups / regional totals filtered out via region.id != 'NA'). "
                "Each entry: {name, region, annual: {year: midyear_population}}."
            ),
        },
        "countries": countries,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    total_obs = sum(len(c["annual"]) for c in countries.values())
    print(
        f"Wrote {args.out} — {len(countries)} countries, {total_obs} observations",
        file=sys.stderr,
    )
    write_status(args.out.parent, "fetch_worldbank_pop_all", "all_countries",
                 ok=True, country_count=len(countries),
                 total_observations=total_obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
