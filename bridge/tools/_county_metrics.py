"""County-level metric tool — reads per-metric JSON caches under
`bridge/data/county/` and exposes choropleth-friendly + comparison +
drill payloads to the API and (eventually) the LLM.

Phase A scaffolding: tool, schema, API. Real data comes from the
fetchers (`scripts/fetch_acs_county.py` shipped; CDC WONDER + FBI UCR
fetchers deferred). When a metric file is missing, the tool returns
a structured empty response with a clear "data not yet fetched"
message rather than raising.

All correlation math is pure numpy — `_analyze.py`'s correlate op
already uses this exact pattern. Phase C will add scipy for partial
correlations and OLS regression once the user has Phase B in hand
and confirms the raw Pearson r is leaving questions on the table.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from bridge.tools._regions import resolve_year as _resolve_year

log = logging.getLogger("bridge.tools")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "county"

# Registry of known metrics. Each entry maps a short key to the file
# under bridge/data/county/. Adding a new metric is: (1) add an entry
# here with sensible label + unit, (2) write a fetcher script, (3)
# run the fetcher to populate the JSON. The tool picks it up
# automatically.
_METRICS: dict[str, dict[str, str]] = {
    "gini": {
        "file": "gini.json",
        "label": "Gini index",
        "unit": "ratio (0-1)",
        "category": "demographics",
    },
    "population": {
        "file": "population.json",
        "label": "Population",
        "unit": "people",
        "category": "demographics",
    },
    "rent": {
        "file": "rent.json",
        "label": "Median gross rent",
        "unit": "USD/month",
        "category": "housing",
    },
    "mortality_all_cause": {
        "file": "mortality_all_cause.json",
        "label": "All-cause mortality (age-adjusted)",
        "unit": "deaths per 100,000",
        "category": "health",
    },
    "mortality_overdose": {
        "file": "mortality_overdose.json",
        "label": "Drug-overdose mortality",
        "unit": "deaths per 100,000",
        "category": "health",
    },
    "violent_crime_rate": {
        "file": "violent_crime_rate.json",
        "label": "Violent crime",
        "unit": "incidents per 100,000",
        "category": "crime",
    },
    "property_crime_rate": {
        "file": "property_crime_rate.json",
        "label": "Property crime",
        "unit": "incidents per 100,000",
        "category": "crime",
    },
    "alice_pct": {
        "file": "alice_pct.json",
        "label": "% ALICE households",
        "unit": "share",
        "category": "economic",
    },
}


def _load_metric_file(key: str) -> dict[str, Any] | None:
    """Read a metric JSON. Returns None when the file isn't on disk —
    the caller surfaces a clear empty-data message rather than raising,
    so the panel can render a "no data yet" state."""
    spec = _METRICS.get(key)
    if not spec:
        return None
    path = _DATA_DIR / spec["file"]
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("county metric file %s unreadable: %s", path, e)
        return None


def _load_county_names() -> dict[str, str]:
    """FIPS → human-readable county name map. Empty dict when not
    fetched yet."""
    path = _DATA_DIR / "county_names.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("by_county") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def list_available_metrics() -> dict[str, Any]:
    """Tell the panel which metrics actually have data on disk + which
    are listed in the registry but missing. The UI hides missing ones
    from selectors (or flags them as "coming soon")."""
    available = []
    missing = []
    for key, spec in _METRICS.items():
        path = _DATA_DIR / spec["file"]
        entry = {
            "key": key,
            "label": spec["label"],
            "unit": spec["unit"],
            "category": spec["category"],
        }
        if path.is_file():
            available.append(entry)
        else:
            missing.append(entry)
    return {"available": available, "missing": missing}


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "count": len(arr),
    }


# --- public API -------------------------------------------------------

def county_metric(metric: str, year: int | None = None) -> dict[str, Any]:
    """Per-county values + summary stats for a single metric/year.

    Choropleth-friendly: the panel maps `values_by_county[fips]`
    directly onto the us-counties TopoJSON.
    """
    spec = _METRICS.get(metric)
    if not spec:
        return {
            "error": (
                f"unknown metric {metric!r}. Available: "
                f"{', '.join(sorted(_METRICS))}"
            ),
        }
    data = _load_metric_file(metric)
    if data is None:
        return {
            "metric": metric,
            "label": spec["label"],
            "unit": spec["unit"],
            "data_status": "not_fetched_yet",
            "values_by_county": {},
            "stats": None,
            "year": None,
            "available_years": [],
            "interpretation": (
                f"Data for '{metric}' has not been fetched yet. Run the "
                f"corresponding fetcher in scripts/ to populate "
                f"bridge/data/county/{spec['file']}."
            ),
        }

    by_county = data.get("by_county", {}) or {}
    # Collect available years across all counties (different counties
    # may have data for different years, e.g. when a metric is added
    # mid-vintage).
    years_set: set[int] = set()
    for series in by_county.values():
        for ystr in (series or {}).keys():
            try:
                years_set.add(int(ystr))
            except ValueError:
                pass
    available_years = sorted(years_set)
    chosen = _resolve_year(year, available_years)

    values_by_county: dict[str, float] = {}
    if chosen is not None:
        ystr = str(chosen)
        for fips, series in by_county.items():
            v = (series or {}).get(ystr)
            if v is not None:
                values_by_county[fips] = float(v)

    stats = _stats(list(values_by_county.values()))
    interpretation = ""
    if stats:
        interpretation = (
            f"{spec['label']} across {stats['count']} US counties in "
            f"{chosen}: range {stats['min']:,.4g}–{stats['max']:,.4g} "
            f"({spec['unit']}), median {stats['median']:,.4g}."
        )
    else:
        interpretation = (
            f"No counties had {spec['label']} values for the requested "
            f"year. Try a different year — available: {available_years}."
        )

    return {
        "metric": metric,
        "label": spec["label"],
        "unit": spec["unit"],
        "category": spec["category"],
        "year": chosen,
        "year_requested": year,
        "available_years": available_years,
        "values_by_county": values_by_county,
        "stats": stats,
        "source": (data.get("_metadata") or {}).get("source", ""),
        "vintage": (data.get("_metadata") or {}).get("vintage", ""),
        "data_status": "loaded",
        "interpretation": interpretation,
    }


def county_compare(
    metric_a: str, metric_b: str, year: int | None = None,
) -> dict[str, Any]:
    """Pairwise Pearson r between two metrics across all counties that
    have both. Returns the joined data + scatter-plot-friendly pairs.

    The panel uses this for the Compare view: side-by-side choropleths
    + a scatterplot + the r value with a "correlation, not causation"
    note baked into the response.
    """
    a = county_metric(metric_a, year=year)
    b = county_metric(metric_b, year=year)
    if "error" in a:
        return a
    if "error" in b:
        return b
    # If either metric isn't fetched, surface that explicitly so the
    # panel can show a "fetch this metric to enable comparison" hint.
    if a.get("data_status") != "loaded" or b.get("data_status") != "loaded":
        return {
            "metric_a": metric_a,
            "metric_b": metric_b,
            "data_status": "incomplete",
            "missing": [
                m for m, payload in (
                    (metric_a, a), (metric_b, b),
                ) if payload.get("data_status") != "loaded"
            ],
            "interpretation": (
                f"Both metrics need data on disk to compute a "
                f"correlation. Missing: see `missing` field."
            ),
        }

    # Year alignment — use whichever year both metrics actually have
    # for the most counties.
    target_year = a["year"] if a["year"] == b["year"] else None
    if target_year is None:
        # Fall back to the latest year both have.
        common = sorted(
            set(a["available_years"]) & set(b["available_years"])
        )
        if not common:
            return {
                "metric_a": metric_a,
                "metric_b": metric_b,
                "data_status": "no_overlap",
                "interpretation": "Metrics share no overlapping year.",
            }
        target_year = max(common)
        a = county_metric(metric_a, year=target_year)
        b = county_metric(metric_b, year=target_year)

    a_vals = a["values_by_county"]
    b_vals = b["values_by_county"]
    common_fips = sorted(set(a_vals) & set(b_vals))
    if len(common_fips) < 3:
        return {
            "metric_a": metric_a,
            "metric_b": metric_b,
            "year": target_year,
            "data_status": "insufficient_overlap",
            "n": len(common_fips),
            "interpretation": (
                "Need at least 3 counties with both metrics to compute "
                "a correlation."
            ),
        }

    pairs = [
        {"fips": f, "a": a_vals[f], "b": b_vals[f]}
        for f in common_fips
    ]
    a_arr = np.array([p["a"] for p in pairs], dtype=float)
    b_arr = np.array([p["b"] for p in pairs], dtype=float)
    r = float(np.corrcoef(a_arr, b_arr)[0, 1]) if a_arr.std() > 0 and b_arr.std() > 0 else 0.0

    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "metric_a_label": a["label"],
        "metric_b_label": b["label"],
        "year": target_year,
        "n": len(common_fips),
        "pearson_r": round(r, 4),
        "pairs": pairs,
        "stats_a": a["stats"],
        "stats_b": b["stats"],
        "data_status": "loaded",
        "interpretation": (
            f"Pearson r = {r:+.3f} between {a['label']} and {b['label']} "
            f"across {len(common_fips)} US counties in {target_year}. "
            f"Correlation, not causation — both metrics likely covary "
            f"with population density and urbanicity."
        ),
    }


def county_drill(fips: str) -> dict[str, Any]:
    """All metrics for a single county across all available years +
    its national-distribution percentile per metric.

    The panel uses this for the click-to-drill slide-in card.
    """
    fips = (fips or "").strip().zfill(5)
    if not fips or len(fips) != 5 or not fips.isdigit():
        return {"error": f"FIPS must be a 5-digit numeric string, got {fips!r}"}

    names = _load_county_names()
    name = names.get(fips, fips)

    metrics_out: dict[str, Any] = {}
    for key, spec in _METRICS.items():
        data = _load_metric_file(key)
        if data is None:
            continue
        series = (data.get("by_county") or {}).get(fips)
        if not series:
            continue

        # Compute this county's percentile rank vs all counties in the
        # latest year — a quick "is this county high or low for this
        # metric?" line for the drill UI.
        years = sorted(int(y) for y in series)
        latest = years[-1]
        my_value = float(series[str(latest)])
        all_values = [
            float(s[str(latest)])
            for s in (data.get("by_county") or {}).values()
            if s and str(latest) in s
        ]
        if all_values:
            below = sum(1 for v in all_values if v < my_value)
            percentile = round((below / len(all_values)) * 100, 1)
        else:
            percentile = None

        metrics_out[key] = {
            "label": spec["label"],
            "unit": spec["unit"],
            "category": spec["category"],
            "series": {y: float(series[str(y)]) for y in years},
            "latest_year": latest,
            "latest_value": my_value,
            "percentile_among_counties": percentile,
        }

    if not metrics_out:
        return {
            "fips": fips,
            "name": name,
            "data_status": "not_fetched_yet",
            "metrics": {},
            "interpretation": (
                f"No county-level metrics on disk yet for {name}. Run "
                f"the fetchers in scripts/ to populate."
            ),
        }

    return {
        "fips": fips,
        "name": name,
        "metrics": metrics_out,
        "data_status": "loaded",
        "interpretation": (
            f"{name} ({fips}): {len(metrics_out)} metrics loaded. "
            f"Percentile values rank this county against all other "
            f"counties in the same year — 50 = median, 95 = "
            f"top-5%-extreme."
        ),
    }
