"""Fetch BLS CPI-U All Items (series CUUR0000SA0) annual averages
1913-current and write to bridge/data/inflation/cpi_u_us.json.

This is the modern CPI-U that the inflation tool uses for purchasing-
power calculations. The dataset originally shipped hand-curated from
usinflationcalculator.com (which republishes the BLS values verbatim);
this script replaces that with a programmatic fetch from the BLS API
v2 so updates are reproducible.

Note: pre-1913 history comes from a separate one-shot script
(scripts/fetch_nber_pre1913_cpi.py). The inflation tool merges both
files at request time via _load_cpi().

Usage:
    python scripts/fetch_bls_cpi.py             # incremental refresh (default)
    python scripts/fetch_bls_cpi.py --full      # full backfill from 1913
    python scripts/fetch_bls_cpi.py --key KEY   # raises BLS rate limits

Without a key the BLS API allows 10 years per request and 25 requests
per day. The full 1913-current backfill needs ~12 chunks; incremental
refreshes need 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

SERIES_ID = "CUUR0000SA0"
SERIES_NAME = "BLS CPI-U All Items, U.S. City Average, Annual Average"
SOURCE_URL_PRIMARY = f"https://data.bls.gov/timeseries/{SERIES_ID}"
SOURCE_URL_API = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{SERIES_ID}"
API_ENDPOINT = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

DATA_FLOOR = 1913
REVISION_LOOKBACK_YEARS = 2  # incremental window covers BLS revisions
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "inflation" / "cpi_u_us.json"
)


def fetch_chunk(start_year: int, end_year: int, key: str | None) -> list[dict]:
    payload: dict = {
        "seriesid": [SERIES_ID],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if key:
        payload["registrationkey"] = key
    from _fetch_helpers import (
        BLS_API_V2_SHAPE, ShapeError, check_shape, http_request,
    )
    body = http_request(API_ENDPOINT, data=payload, timeout=60)
    if body.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(
            f"BLS API error: status={body.get('status')!r} "
            f"message={body.get('message')}"
        )
    # Tier A: validate response shape so a future BLS schema change
    # surfaces as a clear "upstream changed" error instead of a silent
    # KeyError mid-loop.
    try:
        check_shape(body, BLS_API_V2_SHAPE)
    except ShapeError as e:
        raise RuntimeError(
            f"BLS upstream schema drift: {e}. The fetcher's expected "
            f"shape (BLS_API_V2_SHAPE in scripts/_fetch_helpers.py) "
            f"needs updating."
        ) from e
    series = body.get("Results", {}).get("series", [])
    if not series:
        return []
    return series[0].get("data", []) or []


def fetch_all(key: str | None, start_year: int | None = None) -> list[dict]:
    chunk_size = 20 if key else 10
    current_year = date.today().year
    out: list[dict] = []
    start = start_year if start_year is not None else DATA_FLOOR
    while start <= current_year:
        end = min(start + chunk_size - 1, current_year)
        print(f"  fetching {start}-{end} ...", file=sys.stderr)
        chunk = fetch_chunk(start, end, key)
        out.extend(chunk)
        start = end + 1
    return out


def annual_averages(records: list[dict]) -> tuple[dict[str, float], list[int]]:
    by_year: dict[int, list[float]] = {}
    for rec in records:
        period = rec.get("period", "")
        if not period.startswith("M") or period == "M13":
            continue
        try:
            y = int(rec["year"])
            v = float(rec["value"])
        except (KeyError, ValueError):
            continue
        by_year.setdefault(y, []).append(v)
    annual: dict[str, float] = {}
    preliminary: list[int] = []
    for y in sorted(by_year):
        months = by_year[y]
        annual[str(y)] = round(sum(months) / len(months), 3)
        if len(months) < 12:
            preliminary.append(y)
    return annual, preliminary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--key", help="Optional BLS registration key")
    p.add_argument("--out", type=Path, default=OUTPUT_PATH)
    p.add_argument(
        "--full",
        action="store_true",
        help=(
            "Force a complete backfill from %d. Default: incremental — "
            "re-fetch only the last %d years and merge into existing JSON. "
            "First run with no existing file always does a full backfill."
            % (DATA_FLOOR, REVISION_LOOKBACK_YEARS + 1)
        ),
    )
    args = p.parse_args()

    existing_annual: dict[str, float] = {}
    existing_meta: dict = {}
    incremental = False
    if args.out.is_file() and not args.full:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        existing_annual = dict(existing.get("annual", {}))
        existing_meta = dict(existing.get("_metadata", {}) or {})
        incremental = bool(existing_annual)

    from _fetch_helpers import write_status
    if incremental:
        start_year = max(DATA_FLOOR, date.today().year - REVISION_LOOKBACK_YEARS)
        print(
            f"Incremental refresh from {start_year} (existing has "
            f"{len(existing_annual)} years). Use --full to rebuild.",
            file=sys.stderr,
        )
    else:
        start_year = DATA_FLOOR
        print(f"Full backfill {SERIES_ID} from {DATA_FLOOR} ...", file=sys.stderr)
    try:
        records = fetch_all(args.key, start_year=start_year)
    except Exception as e:
        write_status(args.out.parent, "fetch_bls_cpi", SERIES_ID,
                     ok=False, error=str(e))
        raise
    print(f"  got {len(records)} monthly records", file=sys.stderr)

    fresh_annual, preliminary = annual_averages(records)
    annual = dict(existing_annual)
    annual.update(fresh_annual)  # BLS revisions for recent years win
    if not annual:
        print("ERROR: no annual data computed", file=sys.stderr)
        write_status(args.out.parent, "fetch_bls_cpi", SERIES_ID,
                     ok=False, error="no annual data")
        return 1

    metadata = {
        "series_id": SERIES_ID,
        "series_name": SERIES_NAME,
        "source_primary": "U.S. Bureau of Labor Statistics",
        "source_url_primary": SOURCE_URL_PRIMARY,
        "source_url_api": SOURCE_URL_API,
        "data_as_of": date.today().isoformat(),
        "notes": (
            "Annual averages computed from M01-M12 monthly observations. "
            "Years with <12 months are flagged in preliminary_years. "
            "Pre-1913 data lives in cpi_u_us_pre1913.json (NBER M04051 "
            "rebased) and is merged at request time."
        ),
    }
    # Preserve any pre-existing keys we didn't explicitly set (e.g.
    # legacy provenance fields from the original hand-curated file).
    for k, v in existing_meta.items():
        metadata.setdefault(k, v)

    payload = {
        "_metadata": metadata,
        "annual": annual,
        "preliminary_years": preliminary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    years = sorted(int(y) for y in annual)
    print(
        f"Wrote {args.out} — {len(annual)} years "
        f"({years[0]}-{years[-1]}); preliminary: {preliminary or 'none'}",
        file=sys.stderr,
    )
    write_status(args.out.parent, "fetch_bls_cpi", SERIES_ID,
                 ok=True, year_range=[years[0], years[-1]],
                 preliminary_years=preliminary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
