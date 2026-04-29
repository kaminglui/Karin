"""US choropleth map data — state-level rent / housing / cost-of-living.

Phase 1: median 2-bedroom rent by state, year-selectable. Pure data
provider — the rendering happens client-side in
`web/static/panels/map.js` using D3 + a vendored us-atlas TopoJSON.

Architecture mirrors the other data tools (`_inflation`, `_population`,
`_alice`): a tiny tool fn that reads the cached JSON and returns a
shape the panel + the LLM can both consume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "map"
_RENTS_PATH = _DATA_DIR / "state_rents.json"
_REGIONS_PATH = _DATA_DIR / "regions.json"
_PROFILES_PATH = _DATA_DIR / "state_profiles.json"


def _load_rents() -> dict[str, Any]:
    if not _RENTS_PATH.is_file():
        raise FileNotFoundError(
            f"state-rent dataset missing at {_RENTS_PATH}",
        )
    return json.loads(_RENTS_PATH.read_text(encoding="utf-8"))


def _load_regions() -> dict[str, Any]:
    if not _REGIONS_PATH.is_file():
        raise FileNotFoundError(
            f"regions dataset missing at {_REGIONS_PATH}",
        )
    return json.loads(_REGIONS_PATH.read_text(encoding="utf-8"))


def _load_profiles() -> dict[str, Any]:
    if not _PROFILES_PATH.is_file():
        raise FileNotFoundError(
            f"state profiles dataset missing at {_PROFILES_PATH}",
        )
    return json.loads(_PROFILES_PATH.read_text(encoding="utf-8"))


def map_regions() -> dict[str, Any]:
    """Curated historical/economic regions with state-code memberships +
    a per-region color suggestion. The choropleth panel renders these
    as toggleable overlay layers — pick one and the member states get
    a hatched outline on top of the rent fill."""
    data = _load_regions()
    regs = data.get("regions", {})
    # Sort regions by member-count descending so the dropdown sorts the
    # widest overlays first; users mostly want Sun Belt / Rust Belt
    # before they reach the niche ones.
    ordered = sorted(
        regs.items(),
        key=lambda kv: (-len(kv[1].get("states", [])), kv[1].get("label", "")),
    )
    return {
        "regions": [
            {
                "key": k,
                "label": v.get("label", k),
                "color": v.get("color", "#666666"),
                "description": v.get("description", ""),
                "states": list(v.get("states", [])),
            }
            for k, v in ordered
        ],
        "metadata": data.get("_metadata", {}),
    }


def map_state_profile(code: str) -> dict[str, Any]:
    """Per-state profile: top industries, resources, editorial 'why is
    rent the way it is' narrative, plus the region memberships derived
    from regions.json. Keyed by 2-letter state code (uppercase)."""
    code = (code or "").strip().upper()
    profiles = _load_profiles().get("by_state", {})
    profile = profiles.get(code)
    if not profile:
        return {"error": f"no profile for state code {code!r}"}

    # Walk regions.json for memberships so the answer is always in sync
    # with the canonical region list.
    regs = _load_regions().get("regions", {})
    memberships = [
        {"key": rk, "label": rv.get("label", rk), "color": rv.get("color")}
        for rk, rv in regs.items()
        if code in rv.get("states", [])
    ]

    # Try the rent value at the latest year for a quick context line.
    try:
        rents = _load_rents()
        years = sorted(int(y) for y in rents.get("by_year", {}))
        latest = years[-1] if years else None
        rent_now = rents.get("by_year", {}).get(str(latest), {}).get(code)
    except Exception:  # noqa: BLE001
        rent_now = None
        latest = None

    return {
        "code": code,
        "name": profile["name"],
        "top_industries": profile.get("top_industries", []),
        "resources": profile.get("resources", []),
        "narrative": profile.get("narrative", ""),
        "region_memberships": memberships,
        "rent_context": {
            "year": latest,
            "median_2br_rent": rent_now,
        } if rent_now is not None else None,
    }


def map_state_rents(year: int | str | None = None,
                    metric: str = "median_2br_rent") -> dict[str, Any]:
    """Return a choropleth-friendly payload for state-level rent.

    Shape:
      {
        "year": 2024,
        "available_years": [2020, 2022, 2024],
        "metric": "median_2br_rent",
        "metric_label": "Median 2-bedroom rent (USD/month)",
        "values_by_state": { "CA": 2178, "TX": 1411, ... },
        "values_by_fips": { "06": 2178, "48": 1411, ... },
        "state_names": { "CA": "California", ... },
        "stats": { "min": 822, "max": 2462, "median": 1267 },
        "source": "...",
        "as_of": "...",
        "interpretation": "..."
      }
    """
    data = _load_rents()
    by_year = data.get("by_year", {})
    available = sorted(int(y) for y in by_year)
    if not available:
        return {"error": "rent dataset has no years"}

    requested: int | None
    if year is None or year == "":
        requested = None
    else:
        try:
            requested = int(year)
        except (TypeError, ValueError):
            return {"error": f"year must be integer, got {year!r}"}
    if requested is None:
        chosen = available[-1]
    elif requested in available:
        chosen = requested
    else:
        older = [y for y in available if y <= requested]
        chosen = older[-1] if older else available[-1]

    values = {k: float(v) for k, v in by_year[str(chosen)].items()}
    fips_to_code: dict[str, str] = data.get("fips_to_code", {})
    values_by_fips = {
        fips: values[code] for fips, code in fips_to_code.items()
        if code in values
    }
    sorted_vals = sorted(values.values())
    n = len(sorted_vals)
    median = (
        sorted_vals[n // 2] if n % 2 else
        (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )
    stats = {
        "min": round(min(values.values()), 2),
        "max": round(max(values.values()), 2),
        "median": round(median, 2),
        "mean": round(sum(values.values()) / n, 2),
        "count": n,
    }

    # Pick the cheapest + most expensive states for an at-a-glance line.
    by_value = sorted(values.items(), key=lambda kv: kv[1])
    cheapest = by_value[:3]
    priciest = by_value[-3:]
    state_names = data.get("state_names", {})

    def _fmt_pair(code: str, val: float) -> str:
        name = state_names.get(code, code)
        return f"{name} (${val:,.0f}/mo)"

    interpretation = (
        f"Median 2-bedroom rent by US state in {chosen}: "
        f"range ${stats['min']:,.0f}–${stats['max']:,.0f}/mo "
        f"(median ${stats['median']:,.0f}). "
        f"Cheapest: {', '.join(_fmt_pair(c, v) for c, v in cheapest)}. "
        f"Priciest: {', '.join(_fmt_pair(c, v) for c, v in priciest)}."
    )

    return {
        "year": chosen,
        "year_requested": requested,
        "available_years": available,
        "metric": metric,
        "metric_label": "Median 2-bedroom rent (USD/month)",
        "values_by_state": values,
        "values_by_fips": values_by_fips,
        "state_names": state_names,
        "stats": stats,
        "source": data.get("_metadata", {}).get("source", ""),
        "as_of": data.get("_metadata", {}).get("data_as_of", ""),
        "metadata": data.get("_metadata", {}),
        "interpretation": interpretation,
    }


def _map(year: int | str | None = None) -> str:
    """LLM tool wrapper — returns a JSON string the LLM paraphrases.
    For now the choropleth visualization is the main payload; the LLM
    just gets the numeric summary so it can answer 'where is rent
    cheapest?' style questions textually."""
    payload = map_state_rents(year=year)
    return json.dumps(payload, ensure_ascii=False)
