"""Fetch BLS AHETPI (Average Hourly Earnings, production + nonsupervisory)
and write annual averages to bridge/data/inflation/wages_us.json.

Series: CES0500000008 — Total Private, production and nonsupervisory
employees, average hourly earnings (USD/hour, nominal, seasonally
adjusted). Coverage: 1964-01 through current month.

Why this series? It's the longest-running BLS hourly-earnings series
that goes back to 1964, matching common "wages then vs now" questions.
The "All Employees" series (CES0500000003) only starts in 2006, so we'd
lose 42 years of history.

Ships under v1.5 of the inflation tool. See docs/inflation-tool-plan.md.

Usage:
    python scripts/fetch_bls_wages.py [--key BLS_API_KEY]

Without a key, BLS allows 10 years per request and 25 requests per day,
so the full 1964-current fetch needs ~7 requests and stays under the
limit. With a registered key the request size grows to 20 years (cuts
to ~4 requests) and the daily cap rises to 500.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

SERIES_ID = "CES0500000008"
SERIES_NAME = (
    "BLS CES Average Hourly Earnings of Production and Nonsupervisory "
    "Employees, Total Private (USD/hour, nominal, seasonally adjusted)"
)
SOURCE_URL_PRIMARY = f"https://data.bls.gov/timeseries/{SERIES_ID}"
SOURCE_URL_API = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{SERIES_ID}"
API_ENDPOINT = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

DATA_FLOOR = 1964  # AHETPI series starts here
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "inflation" / "wages_us.json"
)


def fetch_chunk(start_year: int, end_year: int, key: str | None) -> list[dict]:
    """POST one chunk to BLS API v2, return raw monthly data records."""
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
    try:
        check_shape(body, BLS_API_V2_SHAPE)
    except ShapeError as e:
        raise RuntimeError(f"BLS upstream schema drift: {e}") from e
    series = body.get("Results", {}).get("series", [])
    if not series:
        return []
    return series[0].get("data", []) or []


def fetch_all(key: str | None, start_year: int | None = None) -> list[dict]:
    """Paginate over the requested year range, respecting BLS chunk limits.

    ``start_year`` defaults to ``DATA_FLOOR`` (full backfill). Pass a
    later year for an incremental refresh that only re-fetches recent
    observations + the in-progress year's revisions.
    """
    chunk_size = 20 if key else 10  # BLS limits per their docs
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
    """Group monthly records by year and compute the annual average.

    Returns (annual_dict, preliminary_years_list). A year is flagged
    preliminary when it has fewer than 12 monthly observations (the
    in-progress current year, basically).
    """
    by_year: dict[int, list[float]] = {}
    for rec in records:
        period = rec.get("period", "")
        # M01-M12 are months; M13 is the BLS-published "annual average"
        # row (only present after the year closes). We ignore M13 and
        # compute our own average from M01-M12 so the field is uniform.
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
        annual[str(y)] = round(sum(months) / len(months), 2)
        if len(months) < 12:
            preliminary.append(y)
    return annual, preliminary


REVISION_LOOKBACK_YEARS = 2  # incremental window: current_year - N through current_year


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--key", help="Optional BLS registration key (raises rate limits)")
    p.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help=(
            "Force a complete backfill from %d. Default: incremental — "
            "re-fetch only the last %d years (captures BLS revisions to "
            "recent observations + the in-progress year's monthly updates), "
            "merging into the existing JSON. First run with no existing "
            "file always does a full backfill regardless."
            % (DATA_FLOOR, REVISION_LOOKBACK_YEARS + 1)
        ),
    )
    args = p.parse_args()

    existing_annual: dict[str, float] = {}
    incremental = False
    if args.out.is_file() and not args.full:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        existing_annual = dict(existing.get("annual", {}))
        incremental = True

    if incremental:
        start_year = max(DATA_FLOOR, date.today().year - REVISION_LOOKBACK_YEARS)
        print(
            f"Incremental refresh from {start_year} (existing has "
            f"{len(existing_annual)} years). Use --full to rebuild.",
            file=sys.stderr,
        )
    else:
        start_year = DATA_FLOOR
        print(
            f"Full backfill {SERIES_ID} from {DATA_FLOOR} ...",
            file=sys.stderr,
        )
    from _fetch_helpers import write_status
    try:
        records = fetch_all(args.key, start_year=start_year)
    except Exception as e:
        write_status(args.out.parent, "fetch_bls_wages", SERIES_ID,
                     ok=False, error=str(e))
        raise
    print(f"  got {len(records)} monthly records", file=sys.stderr)

    fresh_annual, preliminary = annual_averages(records)
    annual = dict(existing_annual)
    annual.update(fresh_annual)  # fresh values overwrite (BLS revisions win)
    if not annual:
        print("ERROR: no annual data computed", file=sys.stderr)
        write_status(args.out.parent, "fetch_bls_wages", SERIES_ID,
                     ok=False, error="no annual data")
        return 1

    payload = {
        "_metadata": {
            "series_id": SERIES_ID,
            "series_name": SERIES_NAME,
            "source_url_primary": SOURCE_URL_PRIMARY,
            "source_url_api": SOURCE_URL_API,
            "data_as_of": date.today().isoformat(),
            "units": "USD/hour, nominal, seasonally adjusted",
            "notes": (
                "Annual averages computed from M01-M12 monthly observations. "
                "Years with <12 months are flagged in preliminary_years."
            ),
        },
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
    write_status(args.out.parent, "fetch_bls_wages", SERIES_ID,
                 ok=True, year_range=[years[0], years[-1]],
                 preliminary_years=preliminary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
