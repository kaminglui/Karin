"""Statistical analysis over the cached economic series.

Phase 1 of the SOTA-economic-analysis plan: a single `analyze` tool that
operates on whatever series is already cached locally (CPI, wages, BLS AP
items, World Bank population) and returns deterministic numeric answers
for "what was unusual about year X?", "what's the trend?", "how
volatile?", "what's that in real dollars?", and "are these two series
correlated?".

Pure numpy — no scipy, no statsmodels. The richer ops (Hodrick-Prescott
trend extraction, ARIMA forecasts, structural-break tests) are deferred
to Phase 3 and only land when there's a clear use case that this tier
can't cover.

The LLM picks the tool + extracts args; this module does the math and
returns a structured JSON the LLM paraphrases. Same architectural rule
as `inflation`/`population`/`facts`: code does the math, LLM only writes
the sentence.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

log = logging.getLogger("bridge.tools")


# --- series resolution --------------------------------------------------

# A `series` arg looks like: "cpi", "cpi:us", "cpi:japan", "wages",
# "population", "population:us", "item:gasoline". Resolves to a
# {year_int: float} dict keyed by year, plus a label + unit + source for
# the response payload.
def _resolve_series(spec: str) -> dict[str, Any]:
    spec = (spec or "").strip().lower()
    if not spec:
        raise ValueError("series is required")

    if spec.startswith("cpi"):
        # cpi or cpi:<region>
        region = spec.split(":", 1)[1] if ":" in spec else "us"
        from bridge.tools._inflation import (
            _load_cpi, _REGIONS as _INFL_REGIONS, _REGION_ALIASES,
        )
        region = _REGION_ALIASES.get(region, region)
        if region not in _INFL_REGIONS:
            raise ValueError(
                f"Unknown CPI region {region!r}. "
                f"Available: {sorted(_INFL_REGIONS)}",
            )
        cpi = _load_cpi(region)
        cfg = _INFL_REGIONS[region]
        return {
            "values": {int(y): float(v) for y, v in cpi["annual"].items()},
            "label": f"CPI ({cfg['label']})",
            "unit": "index",
            "currency": cfg.get("currency"),
            "source": "BLS CPI-U" if region == "us" else "World Bank FP.CPI.TOTL",
        }

    if spec == "wages":
        from bridge.tools._inflation import _load_wages
        wages = _load_wages()
        if wages is None:
            raise FileNotFoundError(
                "wages dataset missing — run scripts/fetch_bls_wages.py",
            )
        return {
            "values": {int(y): float(v) for y, v in wages["annual"].items()},
            "label": "US hourly wage (AHETPI)",
            "unit": "USD/hour",
            "currency": "USD",
            "source": wages["_metadata"].get(
                "series_name", "BLS AHETPI",
            ),
        }

    if spec.startswith("population"):
        from bridge.tools._population import _load, _REGIONS as _POP_REGIONS, _REGION_ALIASES
        region = spec.split(":", 1)[1] if ":" in spec else "world"
        region = _REGION_ALIASES.get(region, region)
        if region not in _POP_REGIONS:
            raise ValueError(
                f"Unknown population region {region!r}. "
                f"Available: {sorted(_POP_REGIONS)}",
            )
        pop = _load(region)
        cfg = _POP_REGIONS[region]
        return {
            "values": {int(y): float(v) for y, v in pop["annual"].items()},
            "label": f"Population ({cfg['label']})",
            "unit": "people",
            "currency": None,
            "source": "World Bank SP.POP.TOTL",
        }

    if spec.startswith("item:"):
        key = spec.split(":", 1)[1]
        from bridge.tools._inflation import _load_items, _resolve_item
        items_data = _load_items()
        if items_data is None:
            raise FileNotFoundError(
                "items dataset missing — run scripts/fetch_bls_items.py",
            )
        entry = _resolve_item(key, items_data)
        if entry is None:
            raise ValueError(
                f"Unknown item {key!r}. Available: "
                f"{', '.join(it['key'] for it in items_data['items'])}",
            )
        return {
            "values": {int(y): float(v) for y, v in entry["annual"].items()},
            "label": entry["label"],
            "unit": entry["unit"],
            "currency": "USD",
            "source": entry.get("source_url", "BLS Average Price Series"),
        }

    raise ValueError(
        f"Unknown series {spec!r}. Use one of: cpi, cpi:<region>, "
        f"wages, population, population:<region>, item:<key>",
    )


def _slice(values: dict[int, float], year_from: int | None,
           year_to: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, values) numpy arrays clipped to [from, to]."""
    years = sorted(values.keys())
    if year_from is not None:
        years = [y for y in years if y >= year_from]
    if year_to is not None:
        years = [y for y in years if y <= year_to]
    if not years:
        raise ValueError("empty window — no observations in the requested range")
    arr_y = np.array(years, dtype=int)
    arr_v = np.array([values[y] for y in years], dtype=float)
    return arr_y, arr_v


# --- operations --------------------------------------------------------

def _op_peak(years: np.ndarray, vals: np.ndarray, *, label: str,
             unit: str) -> dict[str, Any]:
    idx = int(np.argmax(vals))
    return {
        "op": "peak",
        "year": int(years[idx]),
        "value": round(float(vals[idx]), 4),
        "interpretation": (
            f"{label} peaked at {vals[idx]:.4g} {unit} in {years[idx]}."
        ),
    }


def _op_trough(years: np.ndarray, vals: np.ndarray, *, label: str,
               unit: str) -> dict[str, Any]:
    idx = int(np.argmin(vals))
    return {
        "op": "trough",
        "year": int(years[idx]),
        "value": round(float(vals[idx]), 4),
        "interpretation": (
            f"{label} bottomed at {vals[idx]:.4g} {unit} in {years[idx]}."
        ),
    }


def _op_trend(years: np.ndarray, vals: np.ndarray, *, label: str,
              unit: str) -> dict[str, Any]:
    """Linear OLS y = a*x + b plus R² and a one-step projection."""
    if len(years) < 2:
        raise ValueError("trend needs at least 2 years")
    x = years.astype(float)
    y = vals
    slope, intercept = np.polyfit(x, y, deg=1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    next_year = int(years[-1]) + 1
    projected = float(slope * next_year + intercept)
    return {
        "op": "trend",
        "year_from": int(years[0]),
        "year_to": int(years[-1]),
        "slope_per_year": round(float(slope), 6),
        "intercept": round(float(intercept), 4),
        "r_squared": round(r2, 4),
        "projected": {
            "year": next_year,
            "value": round(projected, 4),
        },
        "interpretation": (
            f"{label} trended {slope:+.4g} {unit}/yr from "
            f"{years[0]} to {years[-1]} (R²={r2:.2f}); linear projection "
            f"to {next_year} ≈ {projected:.4g} {unit}."
        ),
    }


def _op_volatility(years: np.ndarray, vals: np.ndarray, *, label: str,
                   unit: str) -> dict[str, Any]:
    """Annualized volatility = std of YoY % changes."""
    if len(vals) < 2:
        raise ValueError("volatility needs at least 2 years")
    yoy_pct = np.diff(vals) / vals[:-1] * 100.0
    vol = float(np.std(yoy_pct, ddof=1)) if len(yoy_pct) > 1 else float(np.abs(yoy_pct[0]))
    mean_pct = float(np.mean(yoy_pct))
    return {
        "op": "volatility",
        "year_from": int(years[0]),
        "year_to": int(years[-1]),
        "yoy_mean_pct": round(mean_pct, 4),
        "yoy_std_pct": round(vol, 4),
        "interpretation": (
            f"{label} averaged {mean_pct:+.2f}%/yr changes from "
            f"{years[0]} to {years[-1]} with std {vol:.2f}pp; "
            f"higher std = more volatile."
        ),
    }


def _op_percentile_rank(years: np.ndarray, vals: np.ndarray, *, year: int,
                        label: str, unit: str) -> dict[str, Any]:
    if year not in years:
        raise ValueError(
            f"year={year} not in [{years[0]},{years[-1]}] window",
        )
    target = float(vals[years == year][0])
    rank = float(np.sum(vals < target)) / len(vals) * 100.0
    return {
        "op": "percentile_rank",
        "year": year,
        "value": round(target, 4),
        "percentile": round(rank, 2),
        "n_observations": int(len(vals)),
        "interpretation": (
            f"{label} in {year} ({target:.4g} {unit}) ranks at the "
            f"{rank:.0f}th percentile across {len(vals)} years "
            f"({years[0]}–{years[-1]})."
        ),
    }


def _op_zscore(years: np.ndarray, vals: np.ndarray, *, year: int,
               label: str, unit: str) -> dict[str, Any]:
    if year not in years:
        raise ValueError(
            f"year={year} not in [{years[0]},{years[-1]}] window",
        )
    target = float(vals[years == year][0])
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    z = (target - mean) / std if std > 0 else 0.0
    return {
        "op": "zscore",
        "year": year,
        "value": round(target, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "zscore": round(z, 4),
        "interpretation": (
            f"{label} in {year} ({target:.4g} {unit}) was "
            f"{z:+.2f} σ from the {years[0]}–{years[-1]} mean "
            f"({mean:.4g} {unit}, σ={std:.4g})."
        ),
    }


def _op_deflate(years: np.ndarray, vals: np.ndarray, *,
                base_year: int | None, region: str, label: str,
                unit: str) -> dict[str, Any]:
    """Convert nominal series → real series in `base_year` dollars by
    dividing each value by (CPI[year] / CPI[base_year]). Default base
    year = the latest year in the series."""
    from bridge.tools._inflation import _load_cpi
    try:
        cpi_data = _load_cpi(region)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"deflate needs CPI: {e}") from e
    cpi = {int(y): float(v) for y, v in cpi_data["annual"].items()}
    if base_year is None:
        base_year = int(years[-1])
    if base_year not in cpi:
        raise ValueError(
            f"base_year={base_year} missing in {region} CPI",
        )
    base_cpi = cpi[base_year]
    out_years: list[int] = []
    out_real: list[float] = []
    skipped: list[int] = []
    for y, v in zip(years, vals):
        c = cpi.get(int(y))
        if c is None:
            skipped.append(int(y))
            continue
        out_years.append(int(y))
        out_real.append(float(v) * base_cpi / c)
    if not out_years:
        raise ValueError("deflate: no overlap between series and CPI")
    head = {y: round(v, 4) for y, v in zip(out_years[:5], out_real[:5])}
    tail = {y: round(v, 4) for y, v in zip(out_years[-5:], out_real[-5:])}
    return {
        "op": "deflate",
        "base_year": base_year,
        "region_for_cpi": region,
        "n_years": len(out_years),
        "year_range": [out_years[0], out_years[-1]],
        "head": head,
        "tail": tail,
        "skipped_years": skipped,
        "interpretation": (
            f"Deflated {label} into {base_year} {unit} using {region.upper()} "
            f"CPI; {len(out_years)} years from {out_years[0]} to "
            f"{out_years[-1]} (skipped {len(skipped)} for missing CPI)."
        ),
    }


def _op_correlate(series_a: dict[str, Any], series_b_spec: str, *,
                  year_from: int | None, year_to: int | None,
                  lag: int) -> dict[str, Any]:
    """Pearson correlation between two series over their overlapping
    years (clipped to [from, to]). Optional lag: positive lag means we
    correlate A[t] with B[t-lag] (B leads A by `lag` years)."""
    series_b = _resolve_series(series_b_spec)
    a_years = set(series_a["values"]) & set(series_b["values"])
    if year_from is not None:
        a_years = {y for y in a_years if y >= year_from}
    if year_to is not None:
        a_years = {y for y in a_years if y <= year_to}
    a_years = sorted(a_years)
    if len(a_years) < 3:
        raise ValueError(
            f"correlate needs ≥3 overlapping years, got {len(a_years)}",
        )
    a_vals = np.array([series_a["values"][y] for y in a_years], dtype=float)
    if lag != 0:
        # Shift B by lag — pairs (a[t], b[t-lag]). Drop edges.
        b_lookup = {y: series_b["values"].get(y - lag) for y in a_years}
        keep = [(y, b_lookup[y]) for y in a_years if b_lookup[y] is not None]
        if len(keep) < 3:
            raise ValueError(
                f"correlate at lag={lag}: only {len(keep)} aligned pairs",
            )
        a_vals = np.array(
            [series_a["values"][y] for y, _ in keep], dtype=float,
        )
        b_vals = np.array([b for _, b in keep], dtype=float)
        years_used = [y for y, _ in keep]
    else:
        b_vals = np.array([series_b["values"][y] for y in a_years], dtype=float)
        years_used = a_years
    # Pearson r via numpy (no scipy needed).
    r = float(np.corrcoef(a_vals, b_vals)[0, 1])
    return {
        "op": "correlate",
        "series_a": series_a["label"],
        "series_b": series_b["label"],
        "lag_years": lag,
        "year_from": int(years_used[0]),
        "year_to": int(years_used[-1]),
        "n_years": len(years_used),
        "pearson_r": round(r, 4),
        "interpretation": (
            f"Pearson r between {series_a['label']} and "
            f"{series_b['label']} (lag={lag}): {r:+.3f} "
            f"over {len(years_used)} years "
            f"({years_used[0]}–{years_used[-1]})."
        ),
    }


# --- public entry point ------------------------------------------------

_VALID_OPS = (
    "peak", "trough", "trend", "volatility",
    "percentile_rank", "zscore", "deflate", "correlate",
)


def _analyze(
    series: str | None = None,
    op: str | None = None,
    year_from: int | str | None = None,
    year_to: int | str | None = None,
    year: int | str | None = None,
    base_year: int | str | None = None,
    series_b: str | None = None,
    lag: int | str = 0,
) -> str:
    """Statistical operations over a cached economic series.

    Args:
        series: which series to analyze. One of: "cpi", "cpi:<region>",
            "wages", "population", "population:<region>", "item:<key>".
        op: which operation. One of: peak, trough, trend, volatility,
            percentile_rank, zscore, deflate, correlate.
        year_from / year_to: optional clipping window.
        year: required for percentile_rank and zscore.
        base_year: optional for deflate (default = latest year).
        series_b: required for correlate.
        lag: optional lag years for correlate (default 0).

    Returns a JSON string the LLM paraphrases.
    """
    try:
        if not series:
            return json.dumps({"error": "series is required"})
        if not op:
            return json.dumps({"error": "op is required"})
        op = str(op).strip().lower()
        if op not in _VALID_OPS:
            return json.dumps({
                "error": f"unknown op {op!r}. valid: {', '.join(_VALID_OPS)}",
            })

        # Coerce year args.
        def _maybe_int(x: Any) -> int | None:
            if x is None or x == "":
                return None
            try:
                return int(x)
            except (TypeError, ValueError):
                raise ValueError(f"could not parse year-like arg {x!r}")

        yf = _maybe_int(year_from)
        yt = _maybe_int(year_to)
        y_ref = _maybe_int(year)
        by = _maybe_int(base_year)
        try:
            lag_int = int(lag) if lag is not None else 0
        except (TypeError, ValueError):
            return json.dumps({"error": f"lag must be integer, got {lag!r}"})

        s = _resolve_series(series)
        years, vals = _slice(s["values"], yf, yt)
        label, unit = s["label"], s["unit"]

        if op == "peak":
            payload = _op_peak(years, vals, label=label, unit=unit)
        elif op == "trough":
            payload = _op_trough(years, vals, label=label, unit=unit)
        elif op == "trend":
            payload = _op_trend(years, vals, label=label, unit=unit)
        elif op == "volatility":
            payload = _op_volatility(years, vals, label=label, unit=unit)
        elif op == "percentile_rank":
            if y_ref is None:
                return json.dumps({"error": "percentile_rank requires year"})
            payload = _op_percentile_rank(
                years, vals, year=y_ref, label=label, unit=unit,
            )
        elif op == "zscore":
            if y_ref is None:
                return json.dumps({"error": "zscore requires year"})
            payload = _op_zscore(
                years, vals, year=y_ref, label=label, unit=unit,
            )
        elif op == "deflate":
            # Use US CPI by default; if the series is regional CPI itself,
            # use that region's CPI for self-consistency.
            region = "us"
            if series.startswith("cpi:"):
                region = series.split(":", 1)[1]
            payload = _op_deflate(
                years, vals, base_year=by, region=region,
                label=label, unit=unit,
            )
        elif op == "correlate":
            if not series_b:
                return json.dumps({"error": "correlate requires series_b"})
            payload = _op_correlate(
                s, series_b, year_from=yf, year_to=yt, lag=lag_int,
            )
        else:
            return json.dumps({"error": f"unhandled op {op!r}"})

        # Common metadata.
        payload.update({
            "series": series,
            "label": label,
            "unit": unit,
            "source": s["source"],
        })
        return json.dumps(payload, ensure_ascii=False)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        log.exception("analyze tool failed")
        return json.dumps({"error": f"analyze failed: {type(e).__name__}: {e}"})


# --- arg pre-extractor (mirrors _facts/_population pattern) -----------

_ANALYZE_OP_PHRASES: tuple[tuple[str, str], ...] = (
    (r"\b(?:peak|highest|maximum|max|record\s+high|all[-\s]time\s+high)\b", "peak"),
    (r"\b(?:trough|lowest|minimum|min|record\s+low|all[-\s]time\s+low)\b", "trough"),
    (r"\b(?:trend|trajectory|slope|growth\s+rate)\b", "trend"),
    (r"\b(?:volatil(?:e|ity)|swings?|how\s+much\s+does\s+it\s+vary)\b", "volatility"),
    (r"\b(?:percentile|rank(?:ed)?|how\s+(?:unusual|rare|extreme))\b", "percentile_rank"),
    (r"\b(?:z[-\s]?score|standard\s+deviations?\s+from)\b", "zscore"),
    (r"\b(?:deflate|real\s+(?:dollars|terms)|inflation[-\s]adjusted)\b", "deflate"),
    (r"\b(?:correl|relat(?:ed|ionship)|track\s+together)\b", "correlate"),
)

_YEAR_RE = re.compile(r"\b(?:19|20|21)\d{2}\b")


def extract_analyze_args(user_text: str | None) -> dict[str, Any]:
    """Cheap NLP for "what was the peak inflation year?" style queries.
    Pulls op + likely year refs; the LLM still has to pick `series`."""
    out: dict[str, Any] = {}
    text = (user_text or "").lower()
    if not text:
        return out
    for pat, op in _ANALYZE_OP_PHRASES:
        if re.search(pat, text, re.I):
            out["op"] = op
            break
    yrs = _YEAR_RE.findall(text)
    if yrs:
        # Single year → percentile/zscore subject. Two years → window.
        if len(yrs) == 1:
            out["year"] = int(yrs[0])
        elif len(yrs) >= 2:
            sorted_yrs = sorted(int(y) for y in yrs)
            out["year_from"] = sorted_yrs[0]
            out["year_to"] = sorted_yrs[-1]
    return out
