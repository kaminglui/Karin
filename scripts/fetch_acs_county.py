"""Fetch Census ACS 5-year county-level data for the county-overlay
analytical surface.

Pulls four ACS tables for every US county across multiple vintages,
splits them into per-metric JSON files under
`bridge/data/county/` matching the schema documented in that
directory's README:

  - B19083_001E  Gini index (income inequality, 0-1)
  - B01003_001E  Total population
  - B25064_001E  Median gross rent ($/month)
  - B19001       16-bucket household income distribution

Plus `county_names.json` mapping FIPS → "<County Name>, <ST>".

Free Census API key required: https://api.census.gov/data/key_signup.html
Set as KARIN_CENSUS_API_KEY in .env or pass via --key.

Vintages fetched: 2018, 2020, 2022, 2024. Each is the published
ACS 5-year vintage (e.g. 2022 = 2018-2022 average), which is the
stable + lagged series — the 1-year is more current but noisy at
the county level.

Usage:
    python scripts/fetch_acs_county.py [--key CENSUS_API_KEY]
                                       [--vintages 2018,2022]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "county"
)
ENDPOINT = "https://api.census.gov/data/{vintage}/acs/acs5"

# Default vintages — every two years gives us biennial coverage that
# pairs with the FBI UCR + CDC WONDER cadences. The most-recent ACS
# 5-year published as of 2025 is the 2019-2023 release (vintage 2023);
# we treat 2024 as forward-looking and add it once Census ships it.
DEFAULT_VINTAGES = [2018, 2020, 2022, 2023]

# B19001 has 17 estimates: total + 16 income brackets. We keep all 16
# brackets; total is implied by the sum.
INCOME_BRACKETS = [f"B19001_{n:03d}E" for n in range(2, 18)]
INCOME_BRACKET_LABELS = [
    "lt_10k", "10_15k", "15_20k", "20_25k", "25_30k", "30_35k",
    "35_40k", "40_45k", "45_50k", "50_60k", "60_75k", "75_100k",
    "100_125k", "125_150k", "150_200k", "200k_plus",
]

# State FIPS → 2-letter code, for friendly county names.
STATE_FIPS_TO_CODE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


def _get_json(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "Karin/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Census ACS HTTP {e.code} at {url[:120]}...: {body}"
        ) from e


def _fetch_acs_year(api_key: str, vintage: int, vars_: list[str]) -> dict[str, dict]:
    """Fetch a single ACS 5-year vintage for all US counties.

    Returns {fips: {var: float|None}}.
    """
    qs = urllib.parse.urlencode({
        "get": "NAME," + ",".join(vars_),
        "for": "county:*",
        "in": "state:*",
        "key": api_key,
    })
    url = f"{ENDPOINT.format(vintage=vintage)}?{qs}"
    rows = _get_json(url)
    if not rows or not isinstance(rows, list):
        raise RuntimeError(f"Census ACS {vintage}: empty response")
    header = rows[0]
    name_idx = header.index("NAME")
    state_idx = header.index("state")
    county_idx = header.index("county")
    var_indices = [(v, header.index(v)) for v in vars_]
    out: dict[str, dict] = {}
    for row in rows[1:]:
        state = str(row[state_idx]).zfill(2)
        county = str(row[county_idx]).zfill(3)
        fips = state + county
        entry: dict = {"_name": row[name_idx]}
        for v, idx in var_indices:
            raw = row[idx]
            if raw is None or raw == "" or raw == "-":
                entry[v] = None
                continue
            try:
                # Census uses negative sentinels (-666666666 etc.) for
                # suppressed/insufficient-sample. Flag those as missing.
                fv = float(raw)
                entry[v] = fv if fv > -100_000_000 else None
            except (TypeError, ValueError):
                entry[v] = None
        out[fips] = entry
    return out


def _format_county_name(raw: str) -> str:
    """Census names look like "Los Angeles County, California". Tighten
    to "Los Angeles County, CA" for tooltip + drill UI.

    Some entries don't end in "County" (Louisiana parishes, Alaska
    boroughs, the city-counties). Preserve those suffixes verbatim."""
    if "," not in raw:
        return raw
    locale, _, state_full = raw.rpartition(", ")
    # Reverse the STATE_FIPS_TO_CODE mapping with full-name lookup.
    # Rare enough that a small static map is fine — beats parsing.
    state_full_to_code = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ",
        "Arkansas": "AR", "California": "CA", "Colorado": "CO",
        "Connecticut": "CT", "Delaware": "DE",
        "District of Columbia": "DC", "Florida": "FL",
        "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
        "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
        "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
        "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
        "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
        "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
        "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
        "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
        "South Carolina": "SC", "South Dakota": "SD",
        "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
        "Puerto Rico": "PR",
    }
    code = state_full_to_code.get(state_full, state_full)
    return f"{locale}, {code}"


def _write_metric(
    metric_key: str, label: str, unit: str,
    by_county_year: dict[int, dict[str, float | None]],
    *, source: str, vintage: str, out_dir: Path,
) -> Path:
    """Write a single metric file: {fips: {year: value}}."""
    flat: dict[str, dict[str, float]] = {}
    for year, county_map in by_county_year.items():
        for fips, val in county_map.items():
            if val is None:
                continue
            flat.setdefault(fips, {})[str(year)] = val
    payload = {
        "_metadata": {
            "version": 1,
            "data_as_of": date.today().isoformat(),
            "metric": metric_key,
            "label": label,
            "unit": unit,
            "source": source,
            "vintage": vintage,
        },
        "by_county": flat,
    }
    path = out_dir / f"{metric_key}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--key",
        default=None,
        help="Census API key (falls back to KARIN_CENSUS_API_KEY env)",
    )
    p.add_argument(
        "--from-prefs",
        action="store_true",
        help=(
            "Also try the active profile's api_keys.census_api_key if "
            "neither --key nor KARIN_CENSUS_API_KEY is set. Lets the "
            "Settings UI's saved key reach this script without an "
            "extra env-var export."
        ),
    )
    p.add_argument(
        "--vintages",
        default=",".join(str(v) for v in DEFAULT_VINTAGES),
        help="Comma-separated 4-digit ACS 5-year vintages",
    )
    p.add_argument(
        "--out", type=Path, default=OUT_DIR,
        help="Output directory (default: bridge/data/county)",
    )
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _fetch_helpers import resolve_api_key  # noqa: E402
    api_key = resolve_api_key(
        "census_api_key",
        cli_value=args.key,
        env_var="KARIN_CENSUS_API_KEY",
        from_prefs=args.from_prefs,
    )
    if not api_key:
        print(
            "error: need Census API key (--key, KARIN_CENSUS_API_KEY env, "
            "or save one in Settings → Optional API keys and pass "
            "--from-prefs). Register free at "
            "https://api.census.gov/data/key_signup.html",
            file=sys.stderr,
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    vintages = [int(v.strip()) for v in args.vintages.split(",") if v.strip()]
    print(f"Fetching ACS 5-year vintages: {vintages}", file=sys.stderr)

    # All variables we need across all metrics — fetched in one pass per
    # vintage to minimize request count.
    vars_ = ["B19083_001E", "B01003_001E", "B25064_001E", *INCOME_BRACKETS]

    by_year: dict[int, dict] = {}
    for v in vintages:
        print(f"  vintage {v} ...", file=sys.stderr)
        try:
            data = _fetch_acs_year(api_key, v, vars_)
        except RuntimeError as e:
            print(f"    skipped ({e})", file=sys.stderr)
            continue
        print(f"    got {len(data)} counties", file=sys.stderr)
        by_year[v] = data
        time.sleep(0.5)

    if not by_year:
        print("error: no vintages fetched", file=sys.stderr)
        return 1

    # county_names.json: take the most recent vintage's NAME field.
    latest = max(by_year)
    name_map = {
        fips: _format_county_name(entry["_name"])
        for fips, entry in by_year[latest].items()
    }
    names_path = args.out / "county_names.json"
    names_path.write_text(
        json.dumps({
            "_metadata": {
                "version": 1,
                "data_as_of": date.today().isoformat(),
                "vintage": f"ACS 5-yr {latest}",
            },
            "by_county": name_map,
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {names_path}", file=sys.stderr)

    # Single-variable metric files.
    metric_specs = [
        ("gini", "Gini index", "ratio", "B19083_001E", "ACS B19083 (Gini)"),
        ("population", "Population", "people", "B01003_001E", "ACS B01003"),
        ("rent", "Median gross rent", "USD/month", "B25064_001E", "ACS B25064"),
    ]
    for metric_key, label, unit, var, source in metric_specs:
        per_year = {
            v: {fips: e.get(var) for fips, e in d.items()}
            for v, d in by_year.items()
        }
        path = _write_metric(
            metric_key, label, unit, per_year,
            source=source, vintage=f"ACS 5-yr {','.join(str(v) for v in vintages)}",
            out_dir=args.out,
        )
        print(f"  wrote {path}", file=sys.stderr)

    # Income brackets — multi-variable, write as nested dict.
    bracket_payload = {
        "_metadata": {
            "version": 1,
            "data_as_of": date.today().isoformat(),
            "metric": "income_brackets",
            "label": "Household income distribution (16 buckets)",
            "unit": "share",
            "source": "ACS B19001",
            "vintage": f"ACS 5-yr {','.join(str(v) for v in vintages)}",
            "bracket_keys": INCOME_BRACKET_LABELS,
        },
        "by_county": {},
    }
    for fips in by_year[latest]:
        bracket_payload["by_county"][fips] = {}
        for v, d in by_year.items():
            entry = d.get(fips)
            if not entry:
                continue
            total = entry.get("B19001_001E") if "B19001_001E" in entry else None
            buckets = {}
            for var, label_key in zip(INCOME_BRACKETS, INCOME_BRACKET_LABELS):
                val = entry.get(var)
                if val is None:
                    continue
                # Store as fraction-of-total so different vintages compose
                # without relying on a separate population denominator.
                buckets[label_key] = val
            if buckets:
                bracket_payload["by_county"][fips][str(v)] = buckets
    bracket_path = args.out / "income_brackets.json"
    bracket_path.write_text(
        json.dumps(bracket_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {bracket_path}", file=sys.stderr)

    print(
        f"Done. {len(by_year[latest])} counties × {len(vintages)} vintages.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
