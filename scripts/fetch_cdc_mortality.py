"""Fetch CDC NCHS county-level mortality from data.cdc.gov.

Pulls county-level age-adjusted death rates from CDC's open Socrata
endpoint and writes them into the JSON envelope documented in
``bridge/data/county/README.md``.

Two target metrics:

- ``mortality_overdose`` — NCHS "Drug Poisoning Mortality by County"
  model-based age-adjusted rate, ~3,140 counties × ~22 years of
  coverage. Pulled via the Socrata REST API.
- ``mortality_all_cause`` — DEFERRED. CDC WONDER's underlying-cause-
  of-death dataset isn't mirrored on Socrata at county granularity;
  the canonical source uses an XML POST API that requires a different
  client. This script writes a "deferred" status entry so the
  diagnostics endpoint surfaces the gap clearly.

Free dataset. An optional Socrata app token raises the per-IP rate
limit but isn't required for an occasional populator run; register
one at https://data.cdc.gov/profile/edit/developer_settings and set
as ``KARIN_CDC_APP_TOKEN`` in ``.env``.

Usage:
    python scripts/fetch_cdc_mortality.py
    python scripts/fetch_cdc_mortality.py --dataset pbkm-d27e
    python scripts/fetch_cdc_mortality.py --app-token <token>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "county"
)
SOCRATA_HOST = "https://data.cdc.gov"

# NCHS - Drug Poisoning Mortality by County: United States.
# Slug is the trailing 8-char base62 segment of the dataset URL.
# As of the 2024 release this is "pbkm-d27e"; if CDC re-publishes
# under a fresh slug the script will surface a clear HTTP 404 and
# you can pass the new slug via --dataset.
DEFAULT_OVERDOSE_DATASET = "pbkm-d27e"

# Field-name candidates: NCHS has renamed columns across vintages,
# so we accept whichever one the live response carries.
_YEAR_CANDIDATES = ["year"]
_FIPS_CANDIDATES = ["fips_code", "fips", "county_fips_code"]

# CDC's published dataset uses BANDED rate categories (e.g. "8-9.9",
# "20+") rather than continuous values — small-cell suppression for
# privacy on low-population counties. We parse band strings to their
# midpoint so the values are usable in correlation analysis. Listed
# in priority order so a future re-publish that includes a continuous
# rate column gets picked up automatically.
_RATE_CANDIDATES_NUMERIC = [
    "model_based_death_rate",
    "estimated_age_adjusted_rate",
    "age_adjusted_rate",
]
_RATE_CANDIDATES_BANDED = [
    "estimated_age_adjusted_death_rate_11_categories_in_ranges",
    "age_adjusted_death_rate_11_categories_in_ranges",
]

PAGE_SIZE = 50_000


class _SocrataAuth:
    """Mutable auth state for a single fetcher run.

    Holds the app token and a ``disabled`` flag flipped on by 401/403.
    Once disabled, further calls skip the token — the public endpoint
    still serves data unauthenticated (just at the lower rate limit),
    so a bad token shouldn't fail the populator. Using a small class
    instead of a module-level global keeps the resolver reentrant +
    testable.
    """

    __slots__ = ("token", "disabled")

    def __init__(self, token: str | None) -> None:
        self.token = token or None
        self.disabled = False


def _socrata_get(url: str, auth: _SocrataAuth) -> list[dict]:
    """GET a Socrata endpoint with graceful fallback on invalid tokens."""
    use_token = bool(auth.token) and not auth.disabled
    headers = {"User-Agent": "Karin/1.0", "Accept": "application/json"}
    if use_token:
        headers["X-App-Token"] = auth.token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if use_token and e.code in (401, 403):
            print(
                f"  CDC Socrata rejected app token (HTTP {e.code}); "
                f"falling back to unauthenticated requests for the rest "
                f"of this run.",
                file=sys.stderr,
            )
            auth.disabled = True
            req2 = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Karin/1.0",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req2, timeout=60) as resp2:
                    return json.loads(resp2.read().decode("utf-8"))
            except urllib.error.HTTPError as e2:
                body = e2.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(
                    f"CDC Socrata HTTP {e2.code} at {url[:120]}... "
                    f"(no-token retry also failed): {body}"
                ) from e2
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"CDC Socrata HTTP {e.code} at {url[:120]}...: {body}"
        ) from e


def _fetch_paginated(
    dataset: str,
    auth: _SocrataAuth,
    *,
    where: str | None = None,
) -> list[dict]:
    """Page through every row of a Socrata dataset.

    Socrata caps ``$limit`` at 50,000 per request; pagination is via
    ``$offset``. Ordering by ``:id`` is required for deterministic
    pagination — without it Socrata can return overlapping pages.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        params = [
            ("$limit", str(PAGE_SIZE)),
            ("$offset", str(offset)),
            ("$order", ":id"),
        ]
        if where:
            params.append(("$where", where))
        url = (
            f"{SOCRATA_HOST}/resource/{dataset}.json"
            f"?{urllib.parse.urlencode(params)}"
        )
        page = _socrata_get(url, auth)
        if not isinstance(page, list):
            raise RuntimeError(
                f"CDC Socrata returned non-list at offset={offset}: "
                f"{type(page).__name__}"
            )
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def _pick_field(sample: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in sample:
            return c
    return None


def _parse_band(value: str) -> float | None:
    """Convert NCHS banded rate strings to a numeric midpoint.

    Handles common forms:
      "0-1.9"  -> 0.95
      "8-9.9"  -> 8.95
      "20+"    -> 22.5  (open-ended top band; assumes ~5-pt width)
      ">20"    -> 22.5
      "<2"     -> 1.0
      "Suppressed" / "" / "*" -> None
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"suppressed", "*", "n/a", "na", "unreliable"}:
        return None
    # Open-ended top band
    m = re.match(r"^(?:>=?|≥)\s*([\d.]+)$", s)
    if m:
        return float(m.group(1)) + 2.5
    if s.endswith("+"):
        try:
            return float(s[:-1]) + 2.5
        except ValueError:
            return None
    # Open-ended bottom band
    m = re.match(r"^(?:<|≤)\s*([\d.]+)$", s)
    if m:
        return max(float(m.group(1)) / 2, 0.5)
    # Range: "8-9.9" → midpoint 8.95
    m = re.match(r"^([\d.]+)\s*[-–]\s*([\d.]+)$", s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        return round((lo + hi) / 2, 3)
    # Bare number — already continuous
    try:
        return float(s)
    except ValueError:
        return None


def fetch_overdose(
    dataset: str, auth: _SocrataAuth,
) -> tuple[dict[str, dict[str, float]], bool]:
    """Returns ``({fips: {year: rate}}, banded)`` from the NCHS dataset.

    ``banded`` is True when the source uses categorical ranges (e.g.
    "8-9.9") that we converted to band midpoints. The metadata note
    surfaces this so consumers know the values aren't exact.
    """
    print(
        f"Fetching CDC NCHS drug overdose mortality "
        f"(Socrata dataset {dataset}) ...",
        file=sys.stderr,
    )
    rows = _fetch_paginated(dataset, auth)
    if not rows:
        raise RuntimeError("CDC overdose dataset returned 0 rows")
    print(f"  got {len(rows)} rows", file=sys.stderr)

    year_field = _pick_field(rows[0], _YEAR_CANDIDATES)
    fips_field = _pick_field(rows[0], _FIPS_CANDIDATES)
    if year_field is None or fips_field is None:
        raise RuntimeError(
            f"missing year/fips fields. Available: {sorted(rows[0].keys())}"
        )
    rate_field_numeric = _pick_field(rows[0], _RATE_CANDIDATES_NUMERIC)
    rate_field_banded = _pick_field(rows[0], _RATE_CANDIDATES_BANDED)
    if rate_field_numeric:
        rate_field = rate_field_numeric
        banded = False
    elif rate_field_banded:
        rate_field = rate_field_banded
        banded = True
    else:
        raise RuntimeError(
            f"no recognized rate field. Tried numeric "
            f"{_RATE_CANDIDATES_NUMERIC} and banded "
            f"{_RATE_CANDIDATES_BANDED}. Available: "
            f"{sorted(rows[0].keys())}"
        )
    print(
        f"  using fields year={year_field!r}, fips={fips_field!r}, "
        f"rate={rate_field!r} ({'banded midpoints' if banded else 'continuous'})",
        file=sys.stderr,
    )

    by_county: dict[str, dict[str, float]] = {}
    skipped = 0
    for row in rows:
        try:
            year = str(int(row[year_field]))
            fips = str(row[fips_field]).zfill(5)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if not fips.isdigit() or len(fips) != 5:
            skipped += 1
            continue
        raw = row.get(rate_field)
        if raw is None or raw == "":
            continue
        value = _parse_band(raw) if banded else None
        if value is None and not banded:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                skipped += 1
                continue
        if value is None:
            continue
        by_county.setdefault(fips, {})[year] = value
    if skipped:
        print(f"  skipped {skipped} malformed rows", file=sys.stderr)
    print(f"  {len(by_county)} counties parsed", file=sys.stderr)
    return by_county, banded


def write_metric(
    metric_key: str,
    *,
    label: str,
    unit: str,
    source: str,
    vintage: str,
    by_county: dict[str, dict[str, float]],
    out_dir: Path,
    extras: dict | None = None,
) -> Path:
    metadata = {
        "version": 1,
        "data_as_of": date.today().isoformat(),
        "metric": metric_key,
        "label": label,
        "unit": unit,
        "source": source,
        "vintage": vintage,
    }
    if extras:
        metadata.update(extras)
    payload = {
        "_metadata": metadata,
        "by_county": by_county,
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
        "--dataset",
        default=DEFAULT_OVERDOSE_DATASET,
        help=(
            "Socrata dataset slug for the overdose dataset "
            f"(default: {DEFAULT_OVERDOSE_DATASET})"
        ),
    )
    p.add_argument(
        "--app-token",
        default=None,
        help="data.cdc.gov app token (falls back to KARIN_CDC_APP_TOKEN env)",
    )
    p.add_argument(
        "--from-prefs",
        action="store_true",
        help=(
            "Also try the active profile's api_keys.cdc_app_token if "
            "neither --app-token nor KARIN_CDC_APP_TOKEN is set. Lets "
            "the Settings UI's saved key reach this script without an "
            "extra env-var export step."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="Output directory (default: bridge/data/county)",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _fetch_helpers import resolve_api_key, write_status  # noqa: E402

    app_token = resolve_api_key(
        "cdc_app_token",
        cli_value=args.app_token,
        env_var="KARIN_CDC_APP_TOKEN",
        from_prefs=args.from_prefs,
    ) or None
    auth = _SocrataAuth(app_token)

    rc = 0

    # --- Drug overdose mortality ---
    try:
        by_county, banded = fetch_overdose(args.dataset, auth)
    except Exception as e:
        write_status(
            args.out, "fetch_cdc_mortality", "mortality_overdose",
            ok=False, error=str(e),
        )
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not by_county:
        msg = "no rows returned for overdose dataset"
        write_status(
            args.out, "fetch_cdc_mortality", "mortality_overdose",
            ok=False, error=msg,
        )
        print(f"error: {msg}", file=sys.stderr)
        return 1
    years = sorted(
        {int(y) for series in by_county.values() for y in series}
    )
    label_suffix = ", band midpoint" if banded else ""
    note = (
        "CDC publishes county-level rates in 11 categorical bands for "
        "small-cell privacy suppression. Values are band midpoints, "
        "not exact rates — fine for percentile/correlation analysis "
        "but don't quote as a precise number."
    ) if banded else ""
    payload_extras = {"binning_note": note} if note else {}
    path = write_metric(
        "mortality_overdose",
        label=f"Drug-overdose mortality (age-adjusted{label_suffix})",
        unit="deaths per 100,000",
        source="CDC NCHS - Drug Poisoning Mortality by County",
        vintage=f"NCHS {years[0]}-{years[-1]}",
        by_county=by_county,
        out_dir=args.out,
        extras=payload_extras,
    )
    print(
        f"  wrote {path} — {len(by_county)} counties × "
        f"{len(years)} years ({years[0]}-{years[-1]})"
        f"{' (banded midpoints)' if banded else ''}",
        file=sys.stderr,
    )
    write_status(
        args.out, "fetch_cdc_mortality", "mortality_overdose",
        ok=True,
        county_count=len(by_county),
        year_range=[years[0], years[-1]],
        banded=banded,
    )

    # --- All-cause mortality (deferred) ---
    print(
        "  mortality_all_cause: skipped (CDC WONDER county-level "
        "all-cause uses XML POST API; pending follow-up fetcher)",
        file=sys.stderr,
    )
    write_status(
        args.out, "fetch_cdc_mortality", "mortality_all_cause",
        ok=False,
        error=(
            "not implemented — CDC WONDER county-level all-cause "
            "mortality uses XML POST API rather than Socrata"
        ),
        status="deferred",
    )

    return rc


if __name__ == "__main__":
    sys.exit(main())
