"""Fetch BLS Average Price (AP) series for common consumer items and
write annual averages to bridge/data/inflation/items_us.json.

The AP series are nominal national average retail prices published
monthly by BLS — same authoritative source as CPI-U and AHETPI. This
gives the inflation tool concrete item-level prices ("how much was a
gallon of gas in 1985?") instead of relying on the LLM's training-data
guesses or computed CPI-equivalents.

Coverage: most series start January 1980. A few start later (ground
beef 1984+, fortified whole milk gallon 1995+).

Ships under v2 of the inflation tool. See docs/inflation-tool-plan.md.

Usage:
    python scripts/fetch_bls_items.py [--key BLS_API_KEY]

Each item is its own series, but BLS API v2 lets you batch up to 25
series per request. We chunk by year-range (10/20 years) and group
items together so we make ~2-4 requests per fetch — well under the
unkeyed daily cap.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Curated item set. Order matters — used as the canonical aliases the
# LLM/extractor matches against. Each entry: stable_key, BLS series id,
# unit phrase ("per gallon"), human label, search aliases.
ITEMS: list[dict] = [
    {
        "key": "gasoline",
        "series_id": "APU000074714",
        "label": "Gasoline (regular, all formulations)",
        "unit": "per gallon",
        "aliases": ["gas", "gasoline", "petrol", "fuel"],
    },
    {
        "key": "bread",
        "series_id": "APU0000702111",
        "label": "Bread, white (1 lb loaf)",
        "unit": "per loaf",
        "aliases": ["bread", "loaf of bread", "white bread"],
    },
    {
        "key": "eggs",
        "series_id": "APU0000708111",
        "label": "Eggs, Grade A large",
        "unit": "per dozen",
        "aliases": ["eggs", "dozen eggs", "carton of eggs"],
    },
    {
        "key": "milk_gallon",
        "series_id": "APU0000709112",
        "label": "Milk, fresh whole (fortified)",
        "unit": "per gallon",
        "aliases": ["milk", "gallon of milk", "whole milk"],
        # 1995+ only — note in JSON
    },
    {
        "key": "milk_halfgal",
        "series_id": "APU0000709111",
        "label": "Milk, fresh whole (half gallon)",
        "unit": "per half-gallon",
        "aliases": ["half gallon of milk", "half-gallon milk"],
    },
    {
        "key": "ground_beef",
        "series_id": "APU0000703112",
        "label": "Ground beef, 100% beef",
        "unit": "per lb",
        "aliases": ["ground beef", "hamburger meat", "beef", "mince"],
    },
    {
        "key": "chicken",
        "series_id": "APU0000706111",
        "label": "Chicken, fresh, whole",
        "unit": "per lb",
        "aliases": ["chicken", "whole chicken", "broiler"],
    },
    {
        "key": "bacon",
        "series_id": "APU0000704111",
        "label": "Bacon, sliced",
        "unit": "per lb",
        "aliases": ["bacon"],
    },
    {
        "key": "bananas",
        "series_id": "APU0000711211",
        "label": "Bananas",
        "unit": "per lb",
        "aliases": ["bananas", "banana"],
    },
    {
        "key": "tomatoes",
        "series_id": "APU0000712311",
        "label": "Tomatoes, field grown",
        "unit": "per lb",
        "aliases": ["tomatoes", "tomato"],
    },
    {
        "key": "coffee",
        "series_id": "APU0000717311",
        "label": "Coffee, 100% ground roast",
        "unit": "per lb",
        "aliases": ["coffee", "ground coffee", "bag of coffee"],
    },
    {
        "key": "sugar",
        "series_id": "APU0000715211",
        "label": "Sugar, white, all sizes",
        "unit": "per lb",
        "aliases": ["sugar"],
    },
    {
        "key": "electricity",
        "series_id": "APU000072610",
        "label": "Electricity",
        "unit": "per kWh",
        "aliases": ["electricity", "kwh", "kilowatt hour", "power"],
    },
    {
        "key": "natural_gas",
        "series_id": "APU000072620",
        "label": "Utility (piped) natural gas",
        "unit": "per therm",
        "aliases": ["natural gas", "utility gas", "therm of gas"],
    },
]

DATA_FLOOR = 1980
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "inflation" / "items_us.json"
)
API_ENDPOINT = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def fetch_chunk(series_ids: list[str], start_year: int, end_year: int,
                key: str | None) -> dict[str, list[dict]]:
    """POST one chunk; return {series_id: [monthly records]}.

    Routes through ``_fetch_helpers.http_request`` for User-Agent +
    retry-with-backoff on 429/5xx — BLS occasionally rate-limits or
    serves transient 502s during peak hours.
    """
    from _fetch_helpers import (
        BLS_API_V2_SHAPE, ShapeError, check_shape, http_request,
    )
    payload: dict = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if key:
        payload["registrationkey"] = key
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
    out: dict[str, list[dict]] = {sid: [] for sid in series_ids}
    for s in body.get("Results", {}).get("series", []):
        sid = s.get("seriesID")
        if sid in out:
            out[sid] = s.get("data", []) or []
    return out


REVISION_LOOKBACK_YEARS = 2  # incremental window: current_year - N through current_year


def fetch_all_items(
    key: str | None, start_year: int | None = None,
) -> dict[str, list[dict]]:
    """Iterate year-range chunks, accumulate monthly data per series.

    ``start_year`` defaults to ``DATA_FLOOR`` (full backfill). For an
    incremental refresh, pass a more recent year — only the requested
    window is re-fetched.
    """
    chunk_size = 20 if key else 10
    current_year = date.today().year
    series_ids = [it["series_id"] for it in ITEMS]
    accumulated: dict[str, list[dict]] = {sid: [] for sid in series_ids}
    start = start_year if start_year is not None else DATA_FLOOR
    while start <= current_year:
        end = min(start + chunk_size - 1, current_year)
        print(f"  fetching {start}-{end} ({len(series_ids)} series)...",
              file=sys.stderr)
        chunk = fetch_chunk(series_ids, start, end, key)
        for sid, records in chunk.items():
            accumulated[sid].extend(records)
        start = end + 1
    return accumulated


def annual_averages(records: list[dict]) -> tuple[dict[str, float], list[int]]:
    """Group monthly records by year → annual average, plus list of
    years where we have <12 observations (preliminary / partial)."""
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
        annual[str(y)] = round(sum(months) / len(months), 4)
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
            "Force a complete backfill from %d for all series. Default: "
            "incremental — re-fetch only the last %d years (BLS revisions + "
            "in-progress monthly updates), merging into existing per-item "
            "annual dicts. First run with no existing file always does a "
            "full backfill regardless." % (DATA_FLOOR, REVISION_LOOKBACK_YEARS + 1)
        ),
    )
    args = p.parse_args()

    existing_by_key: dict[str, dict] = {}
    incremental = False
    if args.out.is_file() and not args.full:
        existing_payload = json.loads(args.out.read_text(encoding="utf-8"))
        for it in existing_payload.get("items", []) or []:
            existing_by_key[it["key"]] = it
        incremental = bool(existing_by_key)

    from _fetch_helpers import write_status
    if incremental:
        start_year = max(DATA_FLOOR, date.today().year - REVISION_LOOKBACK_YEARS)
        print(
            f"Incremental refresh from {start_year} ({len(existing_by_key)} "
            f"existing items). Use --full to rebuild.",
            file=sys.stderr,
        )
    else:
        start_year = DATA_FLOOR
        print(f"Full backfill {len(ITEMS)} BLS AP series ...", file=sys.stderr)
    try:
        raw_by_series = fetch_all_items(args.key, start_year=start_year)
    except Exception as e:
        write_status(args.out.parent, "fetch_bls_items", "all",
                     ok=False, error=str(e))
        raise

    items_out = []
    for it in ITEMS:
        sid = it["series_id"]
        fresh_annual, fresh_preliminary = annual_averages(raw_by_series[sid])
        prior = existing_by_key.get(it["key"])
        merged_annual: dict[str, float] = dict(prior.get("annual", {})) if prior else {}
        merged_annual.update(fresh_annual)
        if not merged_annual:
            print(f"  warning: {sid} ({it['key']}) returned no data — skipping",
                  file=sys.stderr)
            continue
        years = sorted(int(y) for y in merged_annual)
        items_out.append({
            "key": it["key"],
            "label": it["label"],
            "unit": it["unit"],
            "aliases": it["aliases"],
            "series_id": sid,
            "source_url": f"https://data.bls.gov/timeseries/{sid}",
            "year_range": [years[0], years[-1]],
            "annual": merged_annual,
            "preliminary_years": fresh_preliminary,
        })

    payload = {
        "_metadata": {
            "source": "BLS Average Price (AP) Series — U.S. City Average",
            "source_url_index": "https://www.bls.gov/cpi/factsheets/average-prices.htm",
            "data_as_of": date.today().isoformat(),
            "notes": (
                "Annual averages computed from M01-M12 monthly retail prices. "
                "Years with <12 months are flagged in preliminary_years. "
                "Different series have different start years."
            ),
        },
        "items": items_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"Wrote {args.out} — {len(items_out)} items "
        f"(skipped {len(ITEMS) - len(items_out)})",
        file=sys.stderr,
    )
    write_status(args.out.parent, "fetch_bls_items", "all",
                 ok=True,
                 item_count=len(items_out),
                 skipped=len(ITEMS) - len(items_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
