"""Fetch EIA weekly retail regular gasoline (US avg) and write annual
averages to bridge/data/inflation/items_us_eia_overrides.json.

EIA's EPMR_PTE_NUS_DPG_W series is the canonical "what consumers
actually pay at the pump" — regular grade only, volume-weighted across
all PADD regions. It's published every Monday afternoon.

The BLS AP series (APU000074714) we already cache is "Gasoline, all
types" — it includes premium and midgrade, so the headline number
runs a few cents above what people see at the pump. For the facts
year-card and "how much was gas in 1985" questions, EIA regular is
the more accurate signal.

Coverage: 1990-08-20 onward. For 1980-1989 we keep the BLS values
in items_us.json — _load_items() merges EIA on top of BLS at read
time, so those early years still resolve.

Usage:
    python scripts/fetch_eia_gasoline.py [--key EIA_API_KEY]

The key can also come from KARIN_EIA_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

DATA_FLOOR = 1990
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "inflation" / "items_us_eia_overrides.json"
)
API_ENDPOINT = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"

# duoarea=NUS = United States average; product=EPMR = regular all formulations.
DUOAREA = "NUS"
PRODUCT = "EPMR"


def _build_url(api_key: str, length: int, offset: int) -> str:
    params = [
        ("api_key", api_key),
        ("frequency", "weekly"),
        ("data[]", "value"),
        ("facets[duoarea][]", DUOAREA),
        ("facets[product][]", PRODUCT),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("length", str(length)),
        ("offset", str(offset)),
    ]
    return f"{API_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _fetch_all_weekly(api_key: str) -> list[dict]:
    """Page through all weekly observations (~1900+ rows since 1990).
    EIA caps `length` at 5000 per request; pagination is via offset."""
    all_rows: list[dict] = []
    offset = 0
    page_size = 5000
    while True:
        url = _build_url(api_key, page_size, offset)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"EIA HTTP {e.code} at offset={offset}: {body}"
            ) from e
        response = payload.get("response") or {}
        rows = response.get("data") or []
        if not rows:
            break
        all_rows.extend(rows)
        total = int(response.get("total") or len(all_rows))
        offset += len(rows)
        if offset >= total:
            break
    return all_rows


def annual_averages(rows: list[dict]) -> tuple[dict[str, float], list[int]]:
    """Group weekly rows by year, average to annual.

    Returns (annual map, preliminary years). A year is preliminary when
    it has fewer weekly observations than expected (52 ± a couple).
    The current year is almost always preliminary."""
    by_year: dict[int, list[float]] = {}
    for row in rows:
        period = row.get("period")
        value = row.get("value")
        if period is None or value is None:
            continue
        try:
            year = int(str(period)[:4])
            v = float(value)
        except ValueError:
            continue
        by_year.setdefault(year, []).append(v)

    annual: dict[str, float] = {}
    preliminary: list[int] = []
    today_year = date.today().year
    for year in sorted(by_year):
        observations = by_year[year]
        if not observations:
            continue
        annual[str(year)] = round(sum(observations) / len(observations), 4)
        # Less than ~50 weeks → partial. 1990 is partial because EIA
        # weekly data starts mid-August. Current year is always partial.
        if year == today_year or len(observations) < 50:
            preliminary.append(year)
    return annual, preliminary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--key", default=None,
        help="EIA API key. Falls back to KARIN_EIA_API_KEY env var.",
    )
    p.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()

    api_key = (args.key or os.environ.get("KARIN_EIA_API_KEY") or "").strip()
    if not api_key:
        print(
            "error: need EIA key (--key or KARIN_EIA_API_KEY env). "
            "Register at https://www.eia.gov/opendata/register.php",
            file=sys.stderr,
        )
        return 2

    from _fetch_helpers import write_status
    print(
        f"Fetching EIA weekly retail gasoline (duoarea={DUOAREA}, "
        f"product={PRODUCT}) ...",
        file=sys.stderr,
    )
    try:
        rows = _fetch_all_weekly(api_key)
    except Exception as e:
        write_status(args.out.parent, "fetch_eia_gasoline", "gasoline",
                     ok=False, error=str(e))
        raise
    print(f"  got {len(rows)} weekly observations", file=sys.stderr)

    annual, preliminary = annual_averages(rows)
    if not annual:
        msg = "no annual data computed (empty response?)"
        write_status(args.out.parent, "fetch_eia_gasoline", "gasoline",
                     ok=False, error=msg)
        print(f"error: {msg}", file=sys.stderr)
        return 1

    years = sorted(int(y) for y in annual)
    payload = {
        "_metadata": {
            "source": "EIA Weekly Retail Gasoline Prices (regular, US avg)",
            "source_url": (
                "https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm"
            ),
            "series": f"{PRODUCT}_PTE_{DUOAREA}_DPG.W",
            "data_as_of": date.today().isoformat(),
            "notes": (
                "Annual averages computed from weekly retail prices. EIA "
                "regular grade only — preferred over BLS APU000074714 "
                "(all-types) for consumer-facing pump-price questions. "
                "_load_items() merges these onto items_us.json gasoline "
                "for years where EIA has coverage; pre-1990 falls back to "
                "BLS."
            ),
        },
        "items": [
            {
                "key": "gasoline",
                "label": "Gasoline (regular, US avg)",
                "unit": "per gallon",
                "aliases": ["gas", "gasoline", "petrol", "fuel"],
                "series_id": f"{PRODUCT}_PTE_{DUOAREA}_DPG.W",
                "source_url": (
                    "https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm"
                ),
                "year_range": [years[0], years[-1]],
                "annual": annual,
                "preliminary_years": preliminary,
            }
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.out} — gasoline {years[0]}-{years[-1]} "
        f"({len(annual)} years, {len(preliminary)} preliminary)",
        file=sys.stderr,
    )
    write_status(args.out.parent, "fetch_eia_gasoline", "gasoline",
                 ok=True,
                 year_count=len(annual),
                 year_range=f"{years[0]}-{years[-1]}",
                 preliminary=preliminary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
