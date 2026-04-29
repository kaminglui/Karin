"""Fetch FBI UCR county-level violent + property crime rates from
the Crime Data Explorer (CDE) API.

Approach:
  1. List every agency in each state via
     ``/agencies/byStateAbbr/{state_abbr}`` — 51 calls (50 states + DC).
  2. For each reporting agency, fetch annual offense counts via
     ``/summarized/agency/{ori}/{offense}/{from}/{to}``.
  3. Aggregate to county FIPS:
        county_rate = (sum agency_offense_count) /
                      (sum agency_population) * 100,000
     using the county FIPS lookup baked into ``county_names.json``
     from ``fetch_acs_county.py``.

Two metrics written:
  - ``violent_crime_rate.json`` — offense="violent-crime" / per 100K
  - ``property_crime_rate.json`` — offense="property-crime" / per 100K

Free API key required from https://api.data.gov/signup/. Set as
``KARIN_FBI_API_KEY`` in ``.env`` or pass ``--key``.

Runtime warning: there are ~18,000 reporting agencies in the US.
api.data.gov's free-tier rate limit is 1,000 requests/hour per key.
A full run = 51 (state lists) + N_agencies × N_offenses calls. With
the default ``--rate-limit 4.0`` and both offenses enabled, a full
US fetch takes 20-40 hours wall clock. Use ``--states`` to test
on a smaller scope first, ``--year`` to limit to a single vintage,
and ``--offenses`` to populate one metric at a time.

Coverage caveat: agencies with ``nibrs_months_reported < 12`` for
the target year are excluded — partial-year reporting would skew
the rate. Counties with no fully-reporting agency are written with
``coverage: 0`` so the panel can flag them.

Usage:
    KARIN_FBI_API_KEY=... python scripts/fetch_fbi_ucr.py --year 2022
    python scripts/fetch_fbi_ucr.py --states CA --year 2022
    python scripts/fetch_fbi_ucr.py --offenses violent-crime
    python scripts/fetch_fbi_ucr.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
ENDPOINT = "https://api.usa.gov/crime/fbi/cde"

# Resume state under bridge/data/county/_fbi_ucr_resume.json. Lets a
# slow full run survive a network blip — re-running with --resume
# skips agencies already fetched. Safe to delete for a fresh run.
RESUME_FILE = OUT_DIR / "_fbi_ucr_resume.json"

STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY",
]

OFFENSE_METRICS: dict[str, dict[str, str]] = {
    "violent-crime": {
        "metric_key": "violent_crime_rate",
        "label": "Violent crime",
    },
    "property-crime": {
        "metric_key": "property_crime_rate",
        "label": "Property crime",
    },
}

DEFAULT_RATE_LIMIT = 4.0  # seconds between calls; 900/hour
DEFAULT_YEAR = date.today().year - 2  # FBI data lags ~12-18 months


def _http_get_json(
    url: str, *, attempts: int = 3, backoff: float = 2.0,
) -> dict | list:
    """GET with retry on transient 5xx and rate-limit 429."""
    last_err: Exception | None = None
    for n in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Karin/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and n < attempts - 1:
                wait = backoff * (n + 1)
                print(
                    f"    HTTP {e.code}; retrying in {wait:.0f}s ({body[:80]})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            last_err = RuntimeError(
                f"FBI CDE HTTP {e.code} at {url[:140]}: {body}"
            )
            break
        except urllib.error.URLError as e:
            if n < attempts - 1:
                wait = backoff * (n + 1)
                print(
                    f"    URL error; retrying in {wait:.0f}s ({e})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            last_err = RuntimeError(f"FBI CDE network error: {e}")
            break
    assert last_err is not None
    raise last_err


def _normalize_county_name(raw: str) -> str:
    """Strip ", ST" suffix and collapse county-suffix variants for matching.

    FBI uses "ADAMS" while ACS county_names.json uses "Adams County, IL"
    or "Adams Parish, LA" or "Anchorage, AK" (no suffix). We normalize
    both sides to lowercase + drop the trailing suffix word.
    """
    name = (raw or "").strip().lower()
    # Drop trailing "county", "parish", "borough", "census area", "city
    # and borough", "municipality" — common Census Bureau suffixes.
    name = re.sub(
        r"\s+(county|parish|borough|census area|city and borough|"
        r"municipality|municipio)\s*$",
        "",
        name,
    )
    return name.strip()


def _build_fips_lookup(county_names_path: Path) -> dict[tuple[str, str], str]:
    """Build {(state_abbr, normalized_county_name): fips} from
    ``county_names.json`` produced by ``fetch_acs_county.py``."""
    if not county_names_path.is_file():
        raise RuntimeError(
            f"Missing {county_names_path}. Run scripts/fetch_acs_county.py "
            f"first — its county_names.json is required for FIPS lookup."
        )
    data = json.loads(county_names_path.read_text(encoding="utf-8"))
    by_county = data.get("by_county") or {}
    lookup: dict[tuple[str, str], str] = {}
    for fips, full_name in by_county.items():
        if "," not in full_name:
            continue
        county_part, _, state_code = full_name.rpartition(", ")
        norm = _normalize_county_name(county_part)
        lookup[(state_code.strip().upper(), norm)] = str(fips).zfill(5)
    return lookup


def _load_resume() -> dict:
    if not RESUME_FILE.is_file():
        return {}
    try:
        return json.loads(RESUME_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_resume(state: dict) -> None:
    RESUME_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESUME_FILE.write_text(json.dumps(state) + "\n", encoding="utf-8")


def fetch_agencies_for_state(
    state_abbr: str, api_key: str, rate_limit: float,
) -> list[dict]:
    """List every agency in a state. Single API call per state."""
    qs = urllib.parse.urlencode({"API_KEY": api_key})
    url = f"{ENDPOINT}/agencies/byStateAbbr/{state_abbr}?{qs}"
    body = _http_get_json(url)
    time.sleep(rate_limit)
    if isinstance(body, dict):
        # Some responses wrap the list under "results" or "agencies"
        for k in ("results", "agencies", "data"):
            if isinstance(body.get(k), list):
                return body[k]
        return []
    if isinstance(body, list):
        return body
    return []


def _agency_population(agency: dict, year: int) -> int | None:
    """Pull the agency's population for the target year. Schema varies —
    some entries include ``population_<year>`` keys, others nest under
    ``population``."""
    direct = agency.get(f"population_{year}")
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            pass
    pop = agency.get("population")
    if isinstance(pop, dict):
        for key in (str(year), year):
            if key in pop:
                try:
                    return int(pop[key])
                except (TypeError, ValueError):
                    pass
    if isinstance(pop, (int, float)):
        return int(pop)
    return None


def _months_reported(agency: dict, year: int) -> int | None:
    """Months of NIBRS reporting for the target year. Schema varies —
    most recent CDE responses use ``nibrs_months_reported_<year>``."""
    direct = agency.get(f"nibrs_months_reported_{year}")
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            pass
    months = agency.get("months_reported")
    if isinstance(months, dict):
        for key in (str(year), year):
            if key in months:
                try:
                    return int(months[key])
                except (TypeError, ValueError):
                    pass
    return None


def fetch_agency_offense_count(
    ori: str, offense: str, year: int, api_key: str, rate_limit: float,
) -> int | None:
    """Annual count of a single offense category for a single agency.

    Returns None when the API has no data for that agency/offense/year
    pair — distinguishes "no data" from "0 incidents reported"."""
    qs = urllib.parse.urlencode({"API_KEY": api_key})
    url = (
        f"{ENDPOINT}/summarized/agency/{ori}/{offense}/{year}/{year}"
        f"?{qs}"
    )
    body = _http_get_json(url)
    time.sleep(rate_limit)
    if not isinstance(body, dict):
        return None
    rows = (
        body.get("results")
        or body.get("offenses")
        or body.get("data")
        or []
    )
    if not isinstance(rows, list) or not rows:
        return None
    total = 0
    found = False
    for r in rows:
        try:
            ry = int(r.get("data_year") or r.get("year") or year)
        except (TypeError, ValueError):
            continue
        if ry != year:
            continue
        v = r.get("actual") or r.get("value") or r.get("count")
        if v is None:
            continue
        try:
            total += int(v)
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


def aggregate_to_county(
    agencies: list[dict],
    state_abbr: str,
    fips_lookup: dict[tuple[str, str], str],
    offense_counts: dict[str, int],
    year: int,
) -> dict[str, dict[str, float | int]]:
    """Sum agency offense counts + populations per county FIPS.

    Returns ``{fips: {"offenses": int, "population": int, "n_agencies": int}}``.
    Only includes counties with at least one fully-reporting agency.
    """
    county: dict[str, dict[str, float | int]] = {}
    for agency in agencies:
        ori = agency.get("ori") or ""
        if ori not in offense_counts:
            continue
        county_name = (
            agency.get("county_name")
            or agency.get("county")
            or ""
        )
        norm = _normalize_county_name(county_name)
        fips = fips_lookup.get((state_abbr.upper(), norm))
        if not fips:
            continue
        pop = _agency_population(agency, year)
        if pop is None or pop <= 0:
            continue
        entry = county.setdefault(
            fips,
            {"offenses": 0, "population": 0, "n_agencies": 0},
        )
        entry["offenses"] = int(entry["offenses"]) + int(offense_counts[ori])
        entry["population"] = int(entry["population"]) + int(pop)
        entry["n_agencies"] = int(entry["n_agencies"]) + 1
    return county


def write_metric(
    metric_key: str,
    *,
    label: str,
    by_county_year: dict[int, dict[str, dict[str, float | int]]],
    out_dir: Path,
    year_range: tuple[int, int],
) -> Path:
    """Convert per-county aggregated counts into the JSON envelope.

    ``by_county_year`` is ``{year: {fips: {offenses, population, n_agencies}}}``.
    The output file uses ``{fips: {year: rate_per_100K}}`` for the
    primary value, plus ``_coverage`` for n_agencies and population by
    (fips, year) so the panel can surface coverage warnings.
    """
    flat_rates: dict[str, dict[str, float]] = {}
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for year, county_map in by_county_year.items():
        for fips, agg in county_map.items():
            pop = int(agg["population"])
            if pop <= 0:
                continue
            offenses = int(agg["offenses"])
            rate = round(offenses / pop * 100_000, 2)
            flat_rates.setdefault(fips, {})[str(year)] = rate
            coverage.setdefault(fips, {})[str(year)] = {
                "n_agencies": int(agg["n_agencies"]),
                "covered_population": pop,
                "offenses": offenses,
            }
    payload = {
        "_metadata": {
            "version": 1,
            "data_as_of": date.today().isoformat(),
            "metric": metric_key,
            "label": label,
            "unit": "incidents per 100,000",
            "source": "FBI Crime Data Explorer (UCR/NIBRS, agency-level)",
            "vintage": (
                f"FBI UCR {year_range[0]}-{year_range[1]} (12-month "
                f"reporting only)"
            ),
            "notes": (
                "Population-weighted county aggregation: rate = "
                "sum(agency_offenses) / sum(agency_population) * 100,000. "
                "Only agencies with 12 months of NIBRS reporting in the "
                "target year are included. _coverage tracks how much of "
                "each county is covered."
            ),
        },
        "by_county": flat_rates,
        "_coverage": coverage,
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
        "--key", default=None,
        help="api.data.gov key (falls back to KARIN_FBI_API_KEY env)",
    )
    p.add_argument(
        "--from-prefs",
        action="store_true",
        help=(
            "Also try the active profile's api_keys.fbi_api_key if "
            "neither --key nor KARIN_FBI_API_KEY is set. Lets the "
            "Settings UI's saved key reach this script without an "
            "extra env-var export."
        ),
    )
    p.add_argument(
        "--year", type=int, default=DEFAULT_YEAR,
        help=f"Target year (default: {DEFAULT_YEAR}, 2 years lag)",
    )
    p.add_argument(
        "--offenses",
        default=",".join(OFFENSE_METRICS),
        help=(
            f"Comma-separated offense categories. Available: "
            f"{', '.join(OFFENSE_METRICS)}"
        ),
    )
    p.add_argument(
        "--states",
        default=",".join(STATE_ABBRS),
        help="Comma-separated state abbreviations (default: all 50 + DC)",
    )
    p.add_argument(
        "--rate-limit", type=float, default=DEFAULT_RATE_LIMIT,
        help=(
            f"Seconds between API calls (default: {DEFAULT_RATE_LIMIT}; "
            f"keeps under api.data.gov free 1,000/hour limit)"
        ),
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Reuse cached agency offense counts from prior run",
    )
    p.add_argument(
        "--out", type=Path, default=OUT_DIR,
        help="Output directory (default: bridge/data/county)",
    )
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _fetch_helpers import resolve_api_key, write_status  # noqa: E402
    api_key = resolve_api_key(
        "fbi_api_key",
        cli_value=args.key,
        env_var="KARIN_FBI_API_KEY",
        from_prefs=args.from_prefs,
    )
    if not api_key:
        print(
            "error: need FBI/api.data.gov key (--key, KARIN_FBI_API_KEY "
            "env, or save one in Settings → Optional API keys and pass "
            "--from-prefs). Register free at "
            "https://api.data.gov/signup/",
            file=sys.stderr,
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    offenses = [o.strip() for o in args.offenses.split(",") if o.strip()]
    unknown_offenses = [o for o in offenses if o not in OFFENSE_METRICS]
    if unknown_offenses:
        print(
            f"error: unknown offense(s) {unknown_offenses}. Known: "
            f"{list(OFFENSE_METRICS)}",
            file=sys.stderr,
        )
        return 2

    fips_lookup = _build_fips_lookup(args.out / "county_names.json")
    print(
        f"FIPS lookup loaded: {len(fips_lookup)} (state, county_name) "
        f"entries.",
        file=sys.stderr,
    )

    resume = _load_resume() if args.resume else {}
    if resume:
        print(
            f"Resuming with {sum(len(v) for v in resume.values())} "
            f"cached agency entries.",
            file=sys.stderr,
        )

    year = args.year
    offense_counts: dict[str, dict[str, int]] = {o: {} for o in offenses}
    agencies_by_state: dict[str, list[dict]] = {}

    print(
        f"Plan: {len(states)} states × {len(offenses)} offense(s), year {year}. "
        f"Rate limit {args.rate_limit}s/call.",
        file=sys.stderr,
    )

    # --- pass 1: list agencies per state ---
    for s in states:
        print(f"[{s}] listing agencies ...", file=sys.stderr)
        try:
            agencies = fetch_agencies_for_state(s, api_key, args.rate_limit)
        except Exception as e:
            print(f"  error listing {s}: {e}", file=sys.stderr)
            agencies = []
        agencies_by_state[s] = agencies
        full = [
            a for a in agencies
            if (_months_reported(a, year) or 0) >= 12
        ]
        print(
            f"  {len(agencies)} agencies, {len(full)} with 12 months "
            f"in {year}",
            file=sys.stderr,
        )

    # --- pass 2: per-agency offense fetch (the slow part) ---
    for offense in offenses:
        cache = resume.get(offense, {}) if args.resume else {}
        for s in states:
            agencies = agencies_by_state.get(s) or []
            full = [
                a for a in agencies
                if (_months_reported(a, year) or 0) >= 12
            ]
            print(
                f"[{s}/{offense}] fetching {len(full)} agencies ...",
                file=sys.stderr,
            )
            for i, agency in enumerate(full, 1):
                ori = agency.get("ori") or ""
                if not ori:
                    continue
                if ori in cache:
                    offense_counts[offense][ori] = cache[ori]
                    continue
                try:
                    count = fetch_agency_offense_count(
                        ori, offense, year, api_key, args.rate_limit,
                    )
                except Exception as e:
                    print(
                        f"    {ori}: {e}; treating as missing",
                        file=sys.stderr,
                    )
                    count = None
                if count is None:
                    continue
                offense_counts[offense][ori] = count
                cache[ori] = count
                if i % 50 == 0:
                    resume[offense] = cache
                    _save_resume(resume)
                    print(
                        f"    [{s}/{offense}] {i}/{len(full)} done",
                        file=sys.stderr,
                    )
            resume[offense] = cache
            _save_resume(resume)

    # --- pass 3: aggregate to county + write metric files ---
    rc = 0
    for offense in offenses:
        spec = OFFENSE_METRICS[offense]
        by_county_year: dict[int, dict[str, dict[str, float | int]]] = {}
        per_year_county = by_county_year.setdefault(year, {})
        for s in states:
            agencies = agencies_by_state.get(s) or []
            full = [
                a for a in agencies
                if (_months_reported(a, year) or 0) >= 12
            ]
            agg = aggregate_to_county(
                full, s, fips_lookup, offense_counts[offense], year,
            )
            for fips, entry in agg.items():
                per_year_county[fips] = entry
        if not per_year_county:
            msg = f"no county aggregations for {offense} in {year}"
            print(f"warning: {msg}", file=sys.stderr)
            write_status(
                args.out, "fetch_fbi_ucr", spec["metric_key"],
                ok=False, error=msg,
            )
            rc = 1
            continue
        path = write_metric(
            spec["metric_key"],
            label=spec["label"],
            by_county_year=by_county_year,
            out_dir=args.out,
            year_range=(year, year),
        )
        print(
            f"  wrote {path} — {len(per_year_county)} counties for "
            f"{offense} in {year}",
            file=sys.stderr,
        )
        write_status(
            args.out, "fetch_fbi_ucr", spec["metric_key"],
            ok=True,
            year=year,
            county_count=len(per_year_county),
        )

    return rc


if __name__ == "__main__":
    sys.exit(main())
