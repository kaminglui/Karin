"""Fetch CPI data from the World Bank Open Data API for one or more
countries/regions and write per-region JSON files under
bridge/data/inflation/.

Why World Bank: their FP.CPI.TOTL series (Consumer Price Index,
2010=100) is publicly accessible without auth, covers most countries
back to ~1960-1980, and uses a consistent indexing baseline so
multi-region comparisons can share normalisation logic. Source agencies
underneath are the same official statistics offices each region
publishes itself (e.g. C&SD for HK SAR), so values match.

Used to seed v3 international support for the inflation tool. Each
region's data goes to its own file (cpi_<region>.json) so the tool can
load them independently.

Usage:
    python scripts/fetch_worldbank_cpi.py                   # all configured regions
    python scripts/fetch_worldbank_cpi.py hk_sar            # one region
    python scripts/fetch_worldbank_cpi.py --full hk_sar     # full backfill
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "inflation"
)
INDICATOR = "FP.CPI.TOTL"  # CPI All Items, 2010 = 100

# Region registry. Add new entries here as v3 expands. `iso3` is the
# World Bank country code (3-letter ISO except for some aggregates).
# `label` is the user-facing display name (mirrors the official "X,
# China" naming convention where applicable).
REGIONS: dict[str, dict] = {
    "hk_sar": {
        "iso3": "HKG",
        "label": "Hong Kong SAR, China",
        "currency": "HKD",
        "data_floor": 1981,  # World Bank coverage start
    },
    "cn_mainland": {
        "iso3": "CHN",
        # World Bank labels this "China"; we use "China (Mainland)" per
        # the project's preferred naming convention.
        "label": "China (Mainland)",
        "currency": "CNY",
        "data_floor": 1986,
    },
    "japan": {
        "iso3": "JPN",
        "label": "Japan",
        "currency": "JPY",
        "data_floor": 1960,
    },
    "south_korea": {
        "iso3": "KOR",
        # World Bank labels this "Korea, Rep."; use the common short name.
        "label": "South Korea",
        "currency": "KRW",
        "data_floor": 1960,
    },
    # Taiwan, China is not a World Bank member — fetched separately
    # via the DGBAS national statistics source (see fetch_dgbas_tw_cpi.py
    # if shipped, otherwise pending).
}

REVISION_LOOKBACK_YEARS = 2


def fetch_worldbank(iso3: str, start_year: int, end_year: int) -> dict[int, float]:
    """Pull annual CPI values from the World Bank API. Returns
    {year: value} (skipping nulls). Public, no key."""
    url = (
        f"https://api.worldbank.org/v2/country/{iso3}/indicator/{INDICATOR}"
        f"?format=json&per_page=200&date={start_year}:{end_year}"
    )
    from _fetch_helpers import (
        WORLDBANK_RECORD_SHAPE, ShapeError, check_shape, http_request,
    )
    body = http_request(url, timeout=60)
    if not isinstance(body, list) or len(body) < 2 or body[1] is None:
        raise RuntimeError(
            f"World Bank API returned unexpected shape for {iso3}: {body!r}"
        )
    # Tier A: spot-check a sample record so a key rename surfaces clearly.
    if body[1]:
        try:
            check_shape(body[1][0], WORLDBANK_RECORD_SHAPE)
        except ShapeError as e:
            raise RuntimeError(f"World Bank upstream schema drift: {e}") from e
    out: dict[int, float] = {}
    for rec in body[1]:
        try:
            y = int(rec["date"])
            v = rec.get("value")
        except (KeyError, ValueError, TypeError):
            continue
        if v is None:
            continue
        try:
            out[y] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def write_region(region: str, full: bool) -> int:
    cfg = REGIONS[region]
    out_path = DATA_DIR / f"cpi_{region}.json"
    existing_annual: dict[str, float] = {}
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
        write_status(DATA_DIR, "fetch_worldbank_cpi", region,
                     ok=False, error=str(e))
        raise
    print(f"  got {len(fresh)} annual values", file=sys.stderr)
    annual: dict[str, float] = dict(existing_annual)
    for y, v in fresh.items():
        annual[str(y)] = round(v, 4)
    if not annual:
        print(f"  ERROR: no annual data for {region}", file=sys.stderr)
        write_status(DATA_DIR, "fetch_worldbank_cpi", region,
                     ok=False, error="no annual data")
        return 1

    payload = {
        "_metadata": {
            "series_name": (
                f"World Bank — Consumer Price Index, {cfg['label']} "
                f"(2010 = 100)"
            ),
            "indicator": INDICATOR,
            "iso3": cfg["iso3"],
            "country_or_region": cfg["label"],
            "currency": cfg["currency"],
            "source_primary": "World Bank Open Data",
            "source_url_primary": (
                f"https://data.worldbank.org/indicator/{INDICATOR}"
                f"?locations={cfg['iso3']}"
            ),
            "data_as_of": date.today().isoformat(),
            "notes": (
                "World Bank republishes the underlying official statistics "
                "(C&SD for Hong Kong SAR, China; NBS for China (Mainland); "
                "DGBAS for Taiwan, China; etc.). Index baseline 2010 = 100."
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
    write_status(DATA_DIR, "fetch_worldbank_cpi", region,
                 ok=True, year_range=[years[0], years[-1]])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "regions",
        nargs="*",
        default=list(REGIONS.keys()),
        help=f"Region keys to fetch (default: all). Available: {', '.join(REGIONS)}",
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
