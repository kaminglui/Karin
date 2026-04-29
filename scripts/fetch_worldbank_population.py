"""Fetch population (SP.POP.TOTL) from World Bank for the regions the
inflation tool already supports, plus the world aggregate. Writes one
JSON per region to bridge/data/population/pop_<region>.json.

Why World Bank: same auth-less, multi-region source we already use
for inflation v3. Reusing the same region keys keeps the population
and inflation tools symmetric — when a future ``facts`` aggregator
wants both for a single year, it queries the same key in both
datasets.

Indicator: SP.POP.TOTL (total population, midyear estimate). Most
countries cover 1960-current. World aggregate (WLD) covers 1960+.

Usage:
    python scripts/fetch_worldbank_population.py
    python scripts/fetch_worldbank_population.py world hk_sar
    python scripts/fetch_worldbank_population.py --full
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "population"
)
INDICATOR = "SP.POP.TOTL"

# Mirrors REGIONS in fetch_worldbank_cpi.py so a query that works for
# inflation also works for population. "world" is added as the
# aggregate (WLD is the World Bank "World" code).
REGIONS: dict[str, dict] = {
    "world": {
        "iso3": "WLD",
        "label": "World",
        "data_floor": 1960,
    },
    "us": {
        "iso3": "USA",
        "label": "United States",
        "data_floor": 1960,
    },
    "hk_sar": {
        "iso3": "HKG",
        "label": "Hong Kong SAR, China",
        "data_floor": 1960,
    },
    "cn_mainland": {
        "iso3": "CHN",
        "label": "China (Mainland)",
        "data_floor": 1960,
    },
    "japan": {
        "iso3": "JPN",
        "label": "Japan",
        "data_floor": 1960,
    },
    "south_korea": {
        "iso3": "KOR",
        "label": "South Korea",
        "data_floor": 1960,
    },
    # Taiwan, China is not in World Bank — DGBAS would need a separate
    # populator (tracked alongside the inflation v3 Taiwan TODO).
}

REVISION_LOOKBACK_YEARS = 2


def fetch_worldbank(iso3: str, start_year: int, end_year: int) -> dict[int, int]:
    url = (
        f"https://api.worldbank.org/v2/country/{iso3}/indicator/{INDICATOR}"
        f"?format=json&per_page=200&date={start_year}:{end_year}"
    )
    from _fetch_helpers import (
        WORLDBANK_RECORD_SHAPE, ShapeError, check_shape, http_request,
    )
    body = http_request(url, timeout=60)
    if not isinstance(body, list) or len(body) < 2 or body[1] is None:
        raise RuntimeError(f"World Bank returned unexpected shape for {iso3}: {body!r}")
    if body[1]:
        try:
            check_shape(body[1][0], WORLDBANK_RECORD_SHAPE)
        except ShapeError as e:
            raise RuntimeError(f"World Bank upstream schema drift: {e}") from e
    out: dict[int, int] = {}
    for rec in body[1]:
        try:
            y = int(rec["date"])
            v = rec.get("value")
        except (KeyError, ValueError, TypeError):
            continue
        if v is None:
            continue
        try:
            # SP.POP.TOTL values are plain integers; keep as int.
            out[y] = int(round(float(v)))
        except (TypeError, ValueError):
            continue
    return out


def write_region(region: str, full: bool) -> int:
    cfg = REGIONS[region]
    out_path = DATA_DIR / f"pop_{region}.json"
    existing_annual: dict[str, int] = {}
    incremental = False
    if out_path.is_file() and not full:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        existing_annual = dict(prev.get("annual", {}))
        incremental = bool(existing_annual)

    if incremental:
        start_year = max(cfg["data_floor"], date.today().year - REVISION_LOOKBACK_YEARS)
        print(
            f"[{region}] incremental refresh from {start_year} "
            f"(existing has {len(existing_annual)} years).",
            file=sys.stderr,
        )
    else:
        start_year = cfg["data_floor"]
        print(
            f"[{region}] full backfill {cfg['iso3']} from {start_year} ...",
            file=sys.stderr,
        )

    from _fetch_helpers import write_status
    try:
        fresh = fetch_worldbank(cfg["iso3"], start_year, date.today().year)
    except Exception as e:
        write_status(DATA_DIR, "fetch_worldbank_population", region,
                     ok=False, error=str(e))
        raise
    print(f"  got {len(fresh)} annual values", file=sys.stderr)
    annual: dict[str, int] = dict(existing_annual)
    for y, v in fresh.items():
        annual[str(y)] = v
    if not annual:
        print(f"  ERROR: no annual data for {region}", file=sys.stderr)
        write_status(DATA_DIR, "fetch_worldbank_population", region,
                     ok=False, error="no annual data")
        return 1

    payload = {
        "_metadata": {
            "series_name": (
                f"World Bank — Total Population (SP.POP.TOTL), {cfg['label']}"
            ),
            "indicator": INDICATOR,
            "iso3": cfg["iso3"],
            "country_or_region": cfg["label"],
            "source_primary": "World Bank Open Data",
            "source_url_primary": (
                f"https://data.worldbank.org/indicator/{INDICATOR}"
                f"?locations={cfg['iso3']}"
            ),
            "data_as_of": date.today().isoformat(),
            "notes": (
                "Midyear total population estimate. World Bank republishes "
                "underlying official statistics (UN Population Division for "
                "world aggregates; national stats agencies for country values)."
            ),
        },
        "annual": annual,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    years = sorted(int(y) for y in annual)
    print(
        f"  wrote {out_path} — {len(annual)} years ({years[0]}-{years[-1]})",
        file=sys.stderr,
    )
    write_status(DATA_DIR, "fetch_worldbank_population", region,
                 ok=True, year_range=[years[0], years[-1]])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "regions",
        nargs="*",
        default=list(REGIONS.keys()),
        help=f"Region keys to fetch. Available: {', '.join(REGIONS)}",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Force a complete backfill instead of incremental refresh.",
    )
    args = p.parse_args()

    rc = 0
    for region in args.regions:
        if region not in REGIONS:
            print(f"unknown region: {region}", file=sys.stderr)
            rc = 1
            continue
        try:
            r = write_region(region, full=args.full)
            if r != 0:
                rc = r
        except RuntimeError as e:
            print(f"[{region}] {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
