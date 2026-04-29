"""Fetch NBER Macrohistory series M04051 (cost-of-living index, 1860-1939
monthly) and compile a pre-1913 annual table rebased to modern BLS CPI-U.

Why this source: M04051 is the standard NBER cost-of-living series for
the period before BLS CPI began (1913), originally compiled from
historical wholesale price + retail price index estimates. It is
public-domain US-government-funded research data, distributed by NBER
without a key/login.

Why rebase: M04051 uses an arbitrary base index. To make pre-1913
values comparable to the modern CPI-U series (1982-84 = 100), we
compute the scale factor at the overlap year 1913 (M04051's annual
average ÷ BLS CPI-U's annual average at 1913) and divide every
pre-1913 NBER value by it. This produces a continuous spliced series.

Output: bridge/data/inflation/cpi_u_us_pre1913.json — annual averages
1860 through 1912, indexed to modern CPI-U baseline. The tool's
`_load_cpi()` merges these into the main annual dict at request time.

Usage:
    python scripts/fetch_nber_pre1913_cpi.py
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

NBER_URL = "https://data.nber.org/databases/macrohistory/rectdata/04/m04051.dat"
NBER_DOC_URL = "https://data.nber.org/databases/macrohistory/contents/chapter04.html"

CPI_PATH = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "inflation" / "cpi_u_us.json"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "bridge" / "data" / "inflation" / "cpi_u_us_pre1913.json"
)

OVERLAP_YEAR = 1913  # first year where both NBER and BLS CPI-U exist
PRE_DATA_FLOOR = 1860  # NBER M04051 starts here


def fetch_nber_monthly() -> dict[int, list[float]]:
    """Fetch the NBER .dat file and group monthly values by year.

    File format: fixed-width whitespace `YEAR MONTH VALUE` per line.
    Lines may end with a `.` placeholder for missing months — those
    are skipped (filtered out before annual averaging).
    """
    req = urllib.request.Request(NBER_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    by_year: dict[int, list[float]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            y = int(parts[0])
            m = int(parts[1])
            v = float(parts[2])
        except ValueError:
            continue
        if not (1 <= m <= 12):
            continue
        by_year.setdefault(y, []).append(v)
    return by_year


def annual_averages(by_year: dict[int, list[float]]) -> dict[int, float]:
    """Compute mean of monthly values per year. Years with <12 months
    are still kept (early/late truncation is rare in NBER's series)."""
    return {y: round(sum(vs) / len(vs), 4) for y, vs in by_year.items() if vs}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUTPUT_PATH)
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-fetch even when the output file already exists. The "
            "NBER M04051 series ended in 1939, so this script is "
            "fetch-once by design — pre-1913 values never change. Use "
            "--force only after a methodology change (e.g. switching "
            "the overlap year)."
        ),
    )
    args = p.parse_args()

    if args.out.is_file() and not args.force:
        print(
            f"{args.out} already exists. NBER pre-1913 data is frozen "
            f"by history (series ended 1939) — nothing to refresh. "
            f"Pass --force to re-fetch anyway.",
            file=sys.stderr,
        )
        return 0

    print(f"Fetching NBER M04051 from {NBER_URL} ...", file=sys.stderr)
    by_year = fetch_nber_monthly()
    annual = annual_averages(by_year)
    if OVERLAP_YEAR not in annual:
        print(f"ERROR: NBER series missing overlap year {OVERLAP_YEAR}", file=sys.stderr)
        return 1

    cpi_data = json.loads(CPI_PATH.read_text(encoding="utf-8"))
    bls_annual = cpi_data["annual"]
    if str(OVERLAP_YEAR) not in bls_annual:
        print(
            f"ERROR: modern CPI-U dataset is missing overlap year {OVERLAP_YEAR}",
            file=sys.stderr,
        )
        return 1
    bls_at_overlap = float(bls_annual[str(OVERLAP_YEAR)])
    nber_at_overlap = annual[OVERLAP_YEAR]
    scale = bls_at_overlap / nber_at_overlap
    print(
        f"Overlap rebase: NBER@{OVERLAP_YEAR}={nber_at_overlap:.3f}, "
        f"BLS-CPI-U@{OVERLAP_YEAR}={bls_at_overlap:.3f}, scale={scale:.6f}",
        file=sys.stderr,
    )

    pre_1913: dict[str, float] = {}
    for y in sorted(annual):
        if y >= OVERLAP_YEAR:
            continue
        if y < PRE_DATA_FLOOR:
            continue
        pre_1913[str(y)] = round(annual[y] * scale, 3)

    payload = {
        "_metadata": {
            "series_name": (
                "NBER Macrohistory M04051 — Cost of Living Index for "
                "Production Workers, rebased to BLS CPI-U baseline at "
                f"{OVERLAP_YEAR}"
            ),
            "series_id": "M04051USM324NNBR",
            "source_primary": "NBER Macrohistory Database (Chapter 04)",
            "source_url_primary": NBER_DOC_URL,
            "source_url_data": NBER_URL,
            "rebased_to": cpi_data["_metadata"]["series_name"],
            "overlap_year": OVERLAP_YEAR,
            "rebase_scale": round(scale, 6),
            "data_as_of": date.today().isoformat(),
            "year_range": [PRE_DATA_FLOOR, OVERLAP_YEAR - 1],
            "confidence": "low",
            "notes": (
                "Pre-1913 estimates use 19th-century retail/wholesale "
                "price proxies and a smaller commodity basket than the "
                "modern CPI-U. Confidence is 'low' relative to post-1913 "
                "BLS data. Use for rough purchasing-power comparisons "
                "only. The series is rebased so 1913 NBER ≈ 1913 BLS, "
                "but methodology drift means values further back are "
                "less directly comparable to modern CPI."
            ),
        },
        "annual": pre_1913,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    years = sorted(int(y) for y in pre_1913)
    print(
        f"Wrote {args.out} — {len(pre_1913)} annual values "
        f"({years[0]}-{years[-1]})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
