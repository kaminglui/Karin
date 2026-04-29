"""Historical purchasing-power equivalence using authoritative BLS data.

Deterministic: all math runs from cached BLS series shipped under
`bridge/data/inflation/`. No live API calls in the request path. The
LLM picks the tool + extracts args; this module does the calculation
and returns a structured JSON the LLM paraphrases.

v1:   CPI-only (US, 1913–present, annual averages).
v1.5: wages (BLS AHETPI, US, 1964–present, nominal hourly earnings).
v2:   pre-1913 (Measuring Worth) + cross-validation with FRED.
v3:   international (Mainland China, Hong Kong SAR, Japan, Korea, Taiwan).

See docs/inflation-tool-plan.md for the full design.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("bridge.tools")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "inflation"
_CPI_PATH = _DATA_DIR / "cpi_u_us.json"
_CPI_PRE1913_PATH = _DATA_DIR / "cpi_u_us_pre1913.json"
_WAGES_PATH = _DATA_DIR / "wages_us.json"
_ITEMS_PATH = _DATA_DIR / "items_us.json"
# Optional per-item overrides from a non-BLS source (e.g. EIA for retail
# gasoline — closer to the pump price than BLS's all-types weighted
# series). Shape mirrors items_us.json; entries are merged into the
# matching item by key, replacing per-year values for the years EIA
# covers and falling back to BLS for years it doesn't.
_ITEMS_OVERRIDES_PATH = _DATA_DIR / "items_us_eia_overrides.json"

# Confidence cutoffs — see docs/inflation-tool-plan.md "Confidence levels".
# Pre-1860 has no NBER coverage in this dataset and is rejected explicitly.
_CPI_HIGH_CONFIDENCE_FROM = 1947
_CPI_MEDIUM_CONFIDENCE_FROM = 1913  # 1913-1946: early BLS, pre-1947 method
_CPI_DATA_FLOOR = 1860  # NBER M04051 spliced, rebased to BLS CPI-U at 1913
_WAGES_DATA_FLOOR = 1964  # CES0500000008 series start
_ITEMS_DATA_FLOOR = 1980  # BLS AP series typically start here

# Region registry — v3 international support. Each entry maps a region
# key (used as the `region` arg in the tool) to its on-disk dataset.
# Add an entry here when shipping a new region. Keep "us" first so the
# default behaves exactly like v2 (US-only).
_REGIONS: dict[str, dict] = {
    "us": {
        "label": "United States",
        "currency": "USD",
        "currency_symbol": "$",
        "cpi_path": _CPI_PATH,
        "cpi_pre1913_path": _CPI_PRE1913_PATH,  # v2 NBER splice
        "supports_wages": True,   # v1.5 BLS AHETPI
        "supports_items": True,   # v2 BLS AP
    },
    "hk_sar": {
        "label": "Hong Kong SAR, China",
        "currency": "HKD",
        "currency_symbol": "HK$",
        "cpi_path": _DATA_DIR / "cpi_hk_sar.json",
        "cpi_pre1913_path": None,
        "supports_wages": False,  # v3 region-specific datasets pending
        "supports_items": False,
    },
    "cn_mainland": {
        "label": "China (Mainland)",
        "currency": "CNY",
        "currency_symbol": "¥",
        "cpi_path": _DATA_DIR / "cpi_cn_mainland.json",
        "cpi_pre1913_path": None,
        "supports_wages": False,
        "supports_items": False,
    },
    "japan": {
        "label": "Japan",
        "currency": "JPY",
        "currency_symbol": "¥",
        "cpi_path": _DATA_DIR / "cpi_japan.json",
        "cpi_pre1913_path": None,
        "supports_wages": False,
        "supports_items": False,
    },
    "south_korea": {
        "label": "South Korea",
        "currency": "KRW",
        "currency_symbol": "₩",
        "cpi_path": _DATA_DIR / "cpi_south_korea.json",
        "cpi_pre1913_path": None,
        "supports_wages": False,
        "supports_items": False,
    },
    # Taiwan, China is pending — DGBAS data needs scraping (no clean
    # public API found 2026-04-27); World Bank doesn't carry Taiwan.
}

# Region aliases (user-facing → canonical key). The bridge extractor
# matches against this so phrasings like "in Hong Kong" route to the
# right dataset.
_REGION_ALIASES: dict[str, str] = {
    "us": "us", "u.s.": "us", "u.s.a.": "us", "usa": "us",
    "united states": "us", "america": "us", "american": "us",
    "hk": "hk_sar", "hong kong": "hk_sar", "hong kong sar": "hk_sar",
    "hkg": "hk_sar", "hkd": "hk_sar",
    "china": "cn_mainland", "china mainland": "cn_mainland",
    "mainland china": "cn_mainland", "prc": "cn_mainland",
    "chn": "cn_mainland", "cny": "cn_mainland", "rmb": "cn_mainland",
    "renminbi": "cn_mainland", "yuan": "cn_mainland",
    "japan": "japan", "japanese": "japan", "jpn": "japan",
    "jpy": "japan", "yen": "japan",
    "korea": "south_korea", "south korea": "south_korea",
    "rok": "south_korea", "kor": "south_korea",
    "krw": "south_korea", "won": "south_korea",
}

_BOILERPLATE_CAVEATS = (
    "CPI measures the average urban consumer's basket — not your specific spending.",
    "Long-range comparisons hide that what people buy today differs from what they bought decades ago.",
)


def _load_cpi(region: str = "us") -> dict[str, Any]:
    """Cached read of the CPI series for the given region. Raises if
    the file is missing — that indicates a setup failure (the JSON
    ships with the repo for v1, World Bank fetch for v3 regions).

    For region="us", optionally merges the pre-1913 NBER spliced file
    so the caller sees a continuous series 1860-current."""
    cfg = _REGIONS.get(region)
    if cfg is None:
        raise ValueError(
            f"Unknown region {region!r}. Available: {sorted(_REGIONS)}",
        )
    cpi_path: Path = cfg["cpi_path"]
    if not cpi_path.is_file():
        raise FileNotFoundError(
            f"CPI dataset for region={region!r} missing at {cpi_path}. "
            f"Run the appropriate fetch script (scripts/fetch_*.py).",
        )
    cpi = json.loads(cpi_path.read_text(encoding="utf-8"))
    pre_path = cfg.get("cpi_pre1913_path")
    if pre_path is not None and pre_path.is_file():
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        # Pre-1913 values come from a different source (NBER M04051);
        # modern values win on overlap (defensive — rebase is at 1913).
        merged = dict(pre.get("annual", {}))
        merged.update(cpi.get("annual", {}))
        cpi = dict(cpi)
        cpi["annual"] = merged
        cpi["_metadata"] = dict(cpi.get("_metadata", {}))
        cpi["_metadata"]["pre1913_source"] = pre.get("_metadata", {})
    return cpi


def _load_wages() -> dict[str, Any] | None:
    """Cached read of the wages series. Returns None when not shipped
    (e.g. pre-v1.5 deployments) — the caller surfaces a friendly error
    rather than crashing."""
    if not _WAGES_PATH.is_file():
        return None
    return json.loads(_WAGES_PATH.read_text(encoding="utf-8"))


def _load_items() -> dict[str, Any] | None:
    """Cached read of the BLS AP item-prices dataset (v2). Returns None
    when not shipped — caller surfaces a friendly error.

    If ``items_us_eia_overrides.json`` is present, its entries are
    merged onto the BLS items by key. Annual maps merge year-by-year
    (override wins on overlap); year_range, source_url, label, and
    series_id are replaced wholesale so the overridden entry advertises
    its actual provenance."""
    if not _ITEMS_PATH.is_file():
        return None
    base = json.loads(_ITEMS_PATH.read_text(encoding="utf-8"))
    if not _ITEMS_OVERRIDES_PATH.is_file():
        return base
    try:
        overrides = json.loads(
            _ITEMS_OVERRIDES_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as e:
        log.warning("inflation tool: bad overrides file %s: %s",
                    _ITEMS_OVERRIDES_PATH.name, e)
        return base
    by_key = {it["key"]: it for it in base.get("items", [])}
    for ov in overrides.get("items", []):
        key = ov.get("key")
        if not key:
            continue
        target = by_key.get(key)
        if target is None:
            base.setdefault("items", []).append(dict(ov))
            by_key[key] = ov
            continue
        merged_annual = dict(target.get("annual", {}))
        merged_annual.update(ov.get("annual", {}))
        target["annual"] = merged_annual
        for field in ("label", "unit", "series_id", "source_url",
                      "preliminary_years"):
            if field in ov:
                target[field] = ov[field]
        years = sorted(int(y) for y in merged_annual)
        if years:
            target["year_range"] = [years[0], years[-1]]
    return base


def _resolve_item(item_query: str | None, items_data: dict | None = None) -> dict | None:
    """Match a user-facing item phrase ("gas", "loaf of bread", "milk")
    to the canonical item entry in items_us.json. Exact key wins, then
    longest-alias-substring match (so "gallon of milk" resolves before
    "milk" alone). Returns None on miss; caller produces an error.
    """
    if not item_query:
        return None
    items_data = items_data or _load_items()
    if items_data is None:
        return None
    q = str(item_query).strip().lower()
    if not q:
        return None
    for entry in items_data["items"]:
        if entry["key"].lower() == q:
            return entry
    # Score by longest alias hit so multi-word aliases beat short ones.
    best: tuple[int, dict] | None = None
    for entry in items_data["items"]:
        for alias in entry.get("aliases", []):
            a = alias.lower()
            if a == q or a in q or q in a:
                score = len(a)
                if best is None or score > best[0]:
                    best = (score, entry)
    return best[1] if best else None


def _confidence_for(year: int) -> str:
    """high (1947+, modern BLS methodology), medium (1913-1946, early
    BLS), low (1860-1912, NBER M04051 historical estimates rebased to
    BLS at 1913)."""
    if year >= _CPI_HIGH_CONFIDENCE_FROM:
        return "high"
    if year >= _CPI_MEDIUM_CONFIDENCE_FROM:
        return "medium"
    return "low"


def _inflation(
    amount: float | str | None = None,
    from_year: int | str = 0,
    to_year: int | str | None = None,
    measure: str = "cpi",
    item: str | None = None,
    region: str = "us",
    regions: str | None = None,
) -> str:
    """Calculate purchasing-power equivalence between two years.

    Args:
        amount: dollar amount in source-year terms.
        from_year: source year (1913-current; 1860+ for US, 1981+ for
                 hk_sar — varies by region).
        to_year: target year (defaults to most recent year in dataset).
        measure: "cpi" (default) for prices, "wages" for AHETPI hourly
                 wages (US only), "both" for CPI + wages + real-wage
                 delta.
        item: optional consumer item key/alias (US only — BLS AP
              series): gas, bread, eggs, milk, beef, chicken, bacon,
              bananas, tomatoes, coffee, sugar, electricity, natural gas.
        region: "us" (default) or "hk_sar" / "cn_mainland" / "japan" /
              "south_korea". Used when ``regions`` is unset.
        regions: optional CSV of region keys (e.g. "us,japan,hk_sar")
              for cross-region comparison mode. When set, returns a
              ``comparison`` map with one CPI block per region; the
              top-level ``cpi`` block is the FIRST region in the list.

    Returns a JSON string the LLM paraphrases. The numbers are
    authoritative — the LLM must NOT cite different values.
    """
    if regions:
        return _inflation_comparison(
            amount=amount, from_year=from_year, to_year=to_year,
            regions_csv=regions,
        )

    region = (region or "us").strip().lower()
    region = _REGION_ALIASES.get(region, region)
    cfg = _REGIONS.get(region)
    if cfg is None:
        return json.dumps({
            "error": (
                f"Unknown region {region!r}. Available: "
                f"{', '.join(sorted(_REGIONS))}."
            ),
        })
    # Default to $1 when amount is missing/zero — natural reading of
    # "in 1970 money?" / "1970 dollars" is the unit comparison. The
    # iter-3 LoRA frequently omits amount on those phrasings.
    if amount is None or amount == "" or amount == 0:
        amount = 1.0
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return json.dumps({"error": f"amount must be numeric, got {amount!r}"})
    try:
        from_year = int(from_year)
    except (TypeError, ValueError):
        return json.dumps({"error": f"from_year must be an integer, got {from_year!r}"})
    if to_year is not None:
        try:
            to_year = int(to_year)
        except (TypeError, ValueError):
            return json.dumps({"error": f"to_year must be an integer, got {to_year!r}"})

    if amount <= 0:
        return json.dumps({"error": "amount must be positive"})

    measure = (measure or "cpi").strip().lower()
    if measure not in {"cpi", "wages", "both"}:
        return json.dumps({"error": f"measure must be 'cpi' / 'wages' / 'both', got {measure!r}"})

    try:
        cpi_data = _load_cpi(region)
    except (FileNotFoundError, ValueError) as e:
        log.error("inflation tool: %s", e)
        return json.dumps({"error": str(e)})

    annual = cpi_data["annual"]
    available_years = sorted(int(y) for y in annual.keys())
    data_floor = available_years[0]
    latest_year = available_years[-1]
    cur_sym = cfg["currency_symbol"]
    cur_code = cfg["currency"]
    region_label = cfg["label"]

    if to_year is None:
        to_year = latest_year

    # Range checks — explicit rejection beats silently extrapolating.
    if from_year < data_floor:
        return json.dumps({
            "error": (
                f"from_year={from_year} is before the {region_label} "
                f"dataset floor ({data_floor})."
            ),
        })
    if from_year > latest_year:
        return json.dumps({
            "error": f"from_year={from_year} is beyond the latest data point ({latest_year}).",
        })
    if to_year < data_floor or to_year > latest_year:
        return json.dumps({
            "error": (
                f"to_year={to_year} is outside the {region_label} data "
                f"range ({data_floor}-{latest_year})."
            ),
        })

    cpi_from = annual[str(from_year)]
    cpi_to = annual[str(to_year)]
    ratio = cpi_to / cpi_from
    amount_output = round(amount * ratio, 2)

    result: dict[str, Any] = {
        "amount_input": amount,
        "from_year": from_year,
        "to_year": to_year,
        "region": region,
        "country": region_label,  # back-compat: widget reads `country`
        "currency": cur_code,
        "cpi": {
            "amount_output": amount_output,
            "ratio": round(ratio, 4),
            "cpi_index_from": cpi_from,
            "cpi_index_to": cpi_to,
            "source": cpi_data["_metadata"]["series_name"],
            "source_url": cpi_data["_metadata"]["source_url_primary"],
            "data_as_of": cpi_data["_metadata"]["data_as_of"],
            "confidence": _confidence_for(from_year) if region == "us" else "medium",
        },
        "interpretation": (
            f"{cur_sym}{amount:,.2f} in {from_year} had the purchasing power of "
            f"about {cur_sym}{amount_output:,.2f} in {to_year} ({region_label}, CPI). "
            f"That's a ratio of {ratio:.2f}× across {to_year - from_year} years."
        ),
        "caveats": list(_BOILERPLATE_CAVEATS),
    }

    # Flag preliminary (in-progress) years so the LLM can mention it.
    preliminary = set(cpi_data.get("preliminary_years") or [])
    if from_year in preliminary or to_year in preliminary:
        prelim_year = to_year if to_year in preliminary else from_year
        result["caveats"].append(
            f"{prelim_year} is a preliminary annual figure (in-progress year); "
            f"the official annual average finalizes after the year ends."
        )

    if measure in {"wages", "both"} and not cfg["supports_wages"]:
        result["wages"] = {
            "error": (
                f"Wages data is not yet loaded for {region_label}. "
                f"Wages currently support region='us' only (BLS AHETPI)."
            ),
        }
    elif measure in {"wages", "both"}:
        wages_block = _compute_wages_block(from_year, to_year, ratio)
        result["wages"] = wages_block
        if "error" not in wages_block and measure == "both":
            # Real-wage delta = nominal wage growth / nominal price growth.
            # >1 means hourly wages outpaced inflation; <1 means they fell
            # behind. This is the "honest" comparison for "are workers
            # better off" questions and worth surfacing in interpretation.
            real_delta = wages_block["wage_ratio"] / ratio
            result["real_wage_delta"] = round(real_delta, 4)
            direction = "outpaced" if real_delta >= 1.0 else "lagged"
            change_pct = abs(real_delta - 1.0) * 100
            result["interpretation"] = (
                f"${amount:,.2f} in {from_year} = about ${amount_output:,.2f} in "
                f"{to_year} by CPI ({ratio:.2f}× prices). Average production-worker "
                f"hourly wages went from ${wages_block['wage_from']:.2f} to "
                f"${wages_block['wage_to']:.2f} ({wages_block['wage_ratio']:.2f}× "
                f"nominal). Real wages {direction} prices by about {change_pct:.1f}% "
                f"over those {to_year - from_year} years."
            )
        elif measure == "wages":
            if "error" in wages_block:
                result["interpretation"] = wages_block["error"]
            else:
                # Pre-compute the real-wage delta + direction here too
                # so a wages-only call still has unambiguous text. The
                # LoRA tends to invert "9.4× wages vs 8.5× prices" when
                # forced to do the comparison itself.
                real_delta = wages_block["wage_ratio"] / ratio
                result["real_wage_delta"] = round(real_delta, 4)
                direction = "outpaced" if real_delta >= 1.0 else "lagged"
                change_pct = abs(real_delta - 1.0) * 100
                result["interpretation"] = (
                    f"Average production-worker hourly wages went from "
                    f"${wages_block['wage_from']:.2f} in {from_year} to "
                    f"${wages_block['wage_to']:.2f} in {to_year} — a "
                    f"{wages_block['wage_ratio']:.2f}× nominal increase. "
                    f"Over the same period prices rose {ratio:.2f}× (CPI), so "
                    f"real wages {direction} prices by about {change_pct:.1f}%."
                )

    if item and not cfg["supports_items"]:
        result["item"] = {
            "error": (
                f"Item-level prices are not yet loaded for {region_label}. "
                f"Items currently support region='us' only (BLS AP series)."
            ),
        }
    elif item:
        item_block = _compute_item_block(
            item_query=item, from_year=from_year, to_year=to_year,
            cpi_annual=annual,
        )
        result["item"] = item_block
        if "error" not in item_block:
            # Item interpretation supersedes the generic CPI line — it's
            # the more concrete, less misleading framing for the user.
            line = (
                f"{item_block['label']} ({item_block['unit']}) cost "
                f"${item_block['price_from']:.2f} in {from_year}"
            )
            if "price_to" in item_block:
                item_change = item_block["nominal_change"]
                vs_inflation = "outpaced" if item_change > ratio else "lagged"
                pct_diff = abs(item_change / ratio - 1.0) * 100
                line += (
                    f" and ${item_block['price_to']:.2f} in {to_year} "
                    f"({item_change:.2f}× nominal). Overall prices rose "
                    f"{ratio:.2f}× (CPI) over the same period, so this item "
                    f"{vs_inflation} general inflation by about {pct_diff:.1f}%."
                )
            else:
                line += (
                    f". Adjusted for general inflation, that's about "
                    f"${item_block['today_cpi_equivalent']:.2f} in {to_year} "
                    f"dollars (item-level price for {to_year} not in dataset)."
                )
            result["interpretation"] = line

    log.info(
        "inflation: %s%s @ %d -> %s%s @ %d (ratio %.4f, conf=%s, "
        "region=%s, measure=%s, item=%s)",
        cur_sym, amount, from_year, cur_sym, amount_output, to_year, ratio,
        result["cpi"]["confidence"], region, measure, item,
    )
    return json.dumps(result, ensure_ascii=False)


def _inflation_comparison(
    amount: float | str | None,
    from_year: int | str,
    to_year: int | str | None,
    regions_csv: str,
) -> str:
    """Cross-region CPI comparison. Loops the per-region path, packages
    results into a ``comparison`` map. Returns the FIRST region's CPI
    block as the top-level for backwards-compatible widget paths.
    """
    # Split + canonicalize the region list. Skip unknowns instead of
    # erroring so a typo on one region doesn't kill the whole call.
    raw_keys = [k.strip().lower() for k in regions_csv.split(",") if k.strip()]
    canonical: list[str] = []
    for k in raw_keys:
        ck = _REGION_ALIASES.get(k, k)
        if ck in _REGIONS and ck not in canonical:
            canonical.append(ck)
    if not canonical:
        return json.dumps({
            "error": (
                f"No valid regions in {regions_csv!r}. Available: "
                f"{', '.join(sorted(_REGIONS))}."
            ),
        })
    if len(canonical) == 1:
        # Single-region call dressed up as comparison — fall through to
        # the regular path so the response shape stays simple.
        return _inflation(
            amount=amount, from_year=from_year, to_year=to_year,
            region=canonical[0],
        )

    # Run each region individually, collect results.
    comparison: dict[str, Any] = {}
    interp_lines: list[str] = []
    fy_resolved: int | None = None
    ty_resolved: int | None = None
    for r in canonical:
        sub_raw = _inflation(
            amount=amount, from_year=from_year, to_year=to_year, region=r,
        )
        sub = json.loads(sub_raw)
        if "error" in sub:
            comparison[r] = {"error": sub["error"]}
            continue
        cpi = sub.get("cpi") or {}
        cfg = _REGIONS.get(r) or {}
        comparison[r] = {
            "country": sub.get("country"),
            "currency": sub.get("currency"),
            "currency_symbol": cfg.get("currency_symbol", "$"),
            "from_year": sub.get("from_year"),
            "to_year": sub.get("to_year"),
            "amount_input": sub.get("amount_input"),
            "amount_output": cpi.get("amount_output"),
            "ratio": cpi.get("ratio"),
            "confidence": cpi.get("confidence"),
            "source": cpi.get("source"),
            "source_url": cpi.get("source_url"),
            "data_as_of": cpi.get("data_as_of"),
        }
        # Capture from/to years from the first successful sub-result.
        if fy_resolved is None:
            fy_resolved = sub.get("from_year")
            ty_resolved = sub.get("to_year")
        # Per-region interpretation line.
        sym = cfg.get("currency_symbol", "$")
        interp_lines.append(
            f"{sub.get('country')}: {sym}{sub.get('amount_input'):,.2f} "
            f"in {sub.get('from_year')} → {sym}{cpi.get('amount_output'):,.2f} "
            f"in {sub.get('to_year')} ({cpi.get('ratio'):.2f}× CPI)"
        )

    if not comparison:
        return json.dumps({"error": "no successful regions in comparison"})

    # Build comparative interpretation that calls out the spread.
    valid = {k: v for k, v in comparison.items() if "error" not in v}
    if len(valid) >= 2:
        ratios = [(k, v["ratio"]) for k, v in valid.items() if v.get("ratio") is not None]
        ratios.sort(key=lambda p: p[1], reverse=True)
        hi = ratios[0]
        lo = ratios[-1]
        spread_line = (
            f"Across {fy_resolved}-{ty_resolved}, "
            f"{valid[hi[0]]['country']} had the highest cumulative "
            f"inflation ({hi[1]:.2f}×) and {valid[lo[0]]['country']} "
            f"the lowest ({lo[1]:.2f}×) — a {hi[1] / lo[1]:.1f}× spread."
        )
    else:
        spread_line = ""

    interpretation = " ".join(interp_lines)
    if spread_line:
        interpretation = f"{interpretation} {spread_line}"

    log.info(
        "inflation comparison: regions=%s years=%s-%s",
        canonical, fy_resolved, ty_resolved,
    )
    return json.dumps({
        "amount_input": valid[canonical[0]]["amount_input"] if canonical[0] in valid else None,
        "from_year": fy_resolved,
        "to_year": ty_resolved,
        "comparison": comparison,
        "regions_order": canonical,
        "interpretation": interpretation,
    }, ensure_ascii=False)


def _compute_item_block(
    item_query: str,
    from_year: int,
    to_year: int,
    cpi_annual: dict[str, float],
) -> dict[str, Any]:
    """Look up an item's BLS AP price at ``from_year`` and (when
    available) at ``to_year``. Always also computes the today's-dollar
    equivalent of the from_year price via CPI so the user sees both the
    nominal historical price AND its inflation-adjusted value.
    """
    items_data = _load_items()
    if items_data is None:
        return {
            "error": (
                "Item-prices dataset (BLS AP series) is not loaded on "
                "this server. Run scripts/fetch_bls_items.py to populate it."
            ),
        }
    entry = _resolve_item(item_query, items_data)
    if entry is None:
        known = sorted(it["key"] for it in items_data["items"])
        return {
            "error": (
                f"Unknown item {item_query!r}. Available items: "
                f"{', '.join(known)}."
            ),
            "available": known,
        }
    annual = entry["annual"]
    available = sorted(int(y) for y in annual.keys())
    floor, ceil = available[0], available[-1]
    if from_year < floor:
        return {
            "error": (
                f"BLS AP data for {entry['label']} starts in {floor}; "
                f"{from_year} is too early."
            ),
            "data_floor": floor,
            "label": entry["label"],
        }
    if from_year > ceil:
        return {
            "error": f"BLS AP data for {entry['label']} only goes through {ceil}.",
            "label": entry["label"],
        }
    price_from = float(annual[str(from_year)])
    cpi_from = float(cpi_annual[str(from_year)])
    cpi_to = float(cpi_annual[str(to_year)])
    today_equiv = round(price_from * cpi_to / cpi_from, 2)

    block: dict[str, Any] = {
        "key": entry["key"],
        "label": entry["label"],
        "unit": entry["unit"],
        "price_from": round(price_from, 2),
        "year_from": from_year,
        "today_cpi_equivalent": today_equiv,
        "year_to": to_year,
        "source": "BLS Average Price Series",
        "source_url": entry["source_url"],
        "data_range": entry["year_range"],
    }
    if str(to_year) in annual:
        price_to = float(annual[str(to_year)])
        block["price_to"] = round(price_to, 2)
        if price_from > 0:
            block["nominal_change"] = round(price_to / price_from, 4)
    preliminary = set(entry.get("preliminary_years") or [])
    if from_year in preliminary or to_year in preliminary:
        prelim = to_year if to_year in preliminary else from_year
        block["caveat"] = (
            f"{prelim} is a preliminary partial-year average; "
            f"the official annual figure finalizes after the year ends."
        )
    return block


def inflation_widget_data(
    amount: float = 1.0,
    from_year: int = 1970,
    to_year: int | None = None,
    measure: str = "cpi",
    item: str | None = None,
    region: str = "us",
    regions: str | None = None,
) -> dict[str, Any]:
    """Build a widget-friendly dict: the same calculation the LLM tool
    returns, plus year-by-year series arrays for charting.

    The series start at ``from_year`` so the chart visualises the
    purchasing-power evolution of $``amount``-in-``from_year`` going
    forward to the latest data point. Wages are included when the
    dataset is loaded and the year range overlaps 1964+. When ``item``
    is set, an item-price series is added so the chart shows a
    concrete consumer item alongside CPI/wages. When ``regions`` is
    set (CSV), returns a comparison shape with per-region CPI blocks
    and overlay-friendly per-region series.
    """
    if regions:
        return _inflation_comparison_widget_data(
            amount=amount, from_year=from_year, to_year=to_year,
            regions_csv=regions,
        )
    region = (region or "us").strip().lower()
    region = _REGION_ALIASES.get(region, region)
    raw = _inflation(
        amount=amount, from_year=from_year, to_year=to_year,
        measure=measure, item=item, region=region,
    )
    calc = json.loads(raw)
    if "error" in calc:
        return calc

    try:
        cpi_data = _load_cpi(region)
    except (FileNotFoundError, ValueError):
        return calc  # Calc has the error already
    cfg = _REGIONS[region]
    cur_sym = cfg["currency_symbol"]
    annual = cpi_data["annual"]
    fy = int(calc["from_year"])
    ty = int(calc["to_year"])
    base_cpi = float(annual[str(fy)])
    available = sorted(int(y) for y in annual.keys() if fy <= int(y))
    cpi_series = {
        "years": available,
        # Equivalent value of $amount-in-from_year, plotted forward.
        "values": [round(amount * float(annual[str(y)]) / base_cpi, 4) for y in available],
        "label": f"{cur_sym}{amount:g} ({fy}) in CPI-adjusted {cfg['currency']}",
    }

    series: dict[str, Any] = {"cpi": cpi_series}

    # Wages + items only for regions that ship those datasets (US in v3).
    wages_data = _load_wages() if cfg["supports_wages"] else None
    if wages_data is not None:
        wages_annual = wages_data["annual"]
        wages_years = sorted(int(y) for y in wages_annual.keys() if fy <= int(y))
        if wages_years and str(fy) in wages_annual:
            base_wage = float(wages_annual[str(fy)])
            if base_wage > 0:
                series["wages_nominal"] = {
                    "years": wages_years,
                    "values": [round(float(wages_annual[str(y)]), 4) for y in wages_years],
                    "label": f"Avg hourly wage (nominal USD)",
                }
                # Real-wage = nominal wage / CPI ratio relative to fy.
                # Plotted in fy-equivalent purchasing-power dollars per
                # hour so the line is comparable across the series.
                real_values: list[float] = []
                real_years: list[int] = []
                for y in wages_years:
                    if str(y) not in annual:
                        continue
                    cpi_ratio_y = float(annual[str(y)]) / base_cpi
                    real = float(wages_annual[str(y)]) / cpi_ratio_y
                    real_values.append(round(real, 4))
                    real_years.append(y)
                if real_years:
                    series["wages_real"] = {
                        "years": real_years,
                        "values": real_values,
                        "label": f"Avg hourly wage (in {fy} dollars)",
                    }

    # Item time-series — only for regions that ship items, when the LLM
    # picked one and it resolved. Restricted to from_year+ for visual
    # consistency with the CPI line.
    item_block = calc.get("item")
    if cfg["supports_items"] and item_block and "error" not in item_block:
        items_data = _load_items()
        if items_data is not None:
            entry = _resolve_item(item_block.get("key") or item, items_data)
            if entry is not None:
                item_annual = entry["annual"]
                item_years = sorted(int(y) for y in item_annual.keys() if fy <= int(y))
                if item_years:
                    series["item"] = {
                        "years": item_years,
                        "values": [round(float(item_annual[str(y)]), 4) for y in item_years],
                        "label": f"{entry['label']} ({entry['unit']})",
                    }

    return {
        "amount_input": calc.get("amount_input", amount),
        "from_year": fy,
        "to_year": ty,
        "region": region,
        "country": calc.get("country", cfg["label"]),
        "currency": calc.get("currency", cfg["currency"]),
        "currency_symbol": cur_sym,
        "measure": measure,
        "cpi": calc.get("cpi"),
        "wages": calc.get("wages"),
        "item": item_block,
        "real_wage_delta": calc.get("real_wage_delta"),
        "interpretation": calc.get("interpretation"),
        "caveats": calc.get("caveats", []),
        "series": series,
    }


def _inflation_comparison_widget_data(
    amount: float | str | None,
    from_year: int | str,
    to_year: int | str | None,
    regions_csv: str,
) -> dict[str, Any]:
    """Comparison-mode widget payload — packages the comparison call
    plus a per-region series so the chart can overlay multiple lines.
    Each series is normalized to amount=1 at from_year so curves are
    directly comparable across currencies."""
    raw = _inflation(
        amount=amount, from_year=from_year, to_year=to_year,
        regions=regions_csv,
    )
    calc = json.loads(raw)
    if "error" in calc:
        return calc
    fy = int(calc["from_year"])
    series_per_region: dict[str, Any] = {}
    for region in calc.get("regions_order", []):
        block = calc.get("comparison", {}).get(region) or {}
        if "error" in block:
            continue
        try:
            cpi_data = _load_cpi(region)
        except (FileNotFoundError, ValueError):
            continue
        annual = cpi_data["annual"]
        if str(fy) not in annual:
            continue
        base = float(annual[str(fy)])
        if base <= 0:
            continue
        years_sorted = sorted(int(y) for y in annual.keys() if fy <= int(y))
        # Normalized: each region's series starts at 1.0 at fy and
        # tracks its own currency-of-origin inflation forward.
        cfg = _REGIONS.get(region) or {}
        series_per_region[region] = {
            "years": years_sorted,
            "values": [round(float(annual[str(y)]) / base, 4) for y in years_sorted],
            "label": f"{block.get('country', region)} ({cfg.get('currency_symbol', '$')})",
            "currency_symbol": cfg.get("currency_symbol", "$"),
        }
    out = dict(calc)
    out["series_per_region"] = series_per_region
    return out


def _compute_wages_block(from_year: int, to_year: int, cpi_ratio: float) -> dict[str, Any]:
    """Build the ``wages`` JSON sub-block for the given year span.

    Returns either a populated block (wage_from, wage_to, wage_ratio,
    source, etc.) or ``{"error": "..."}`` when wages can't be computed
    (file not shipped, year out of range). Errors are friendly enough to
    paraphrase directly to the user.
    """
    wages_data = _load_wages()
    if wages_data is None:
        return {
            "error": (
                "Wages dataset (BLS AHETPI) is not loaded on this server. "
                "Run scripts/fetch_bls_wages.py to populate it."
            ),
        }
    annual = wages_data["annual"]
    available = sorted(int(y) for y in annual.keys())
    floor, ceil = available[0], available[-1]
    if from_year < floor or to_year < floor:
        bad = from_year if from_year < floor else to_year
        return {
            "error": (
                f"Wages data starts in {floor} (BLS AHETPI series begins "
                f"there); {bad} is too early."
            ),
        }
    if from_year > ceil or to_year > ceil:
        bad = from_year if from_year > ceil else to_year
        return {
            "error": f"Wages data only goes through {ceil}; {bad} is too recent.",
        }

    wage_from = float(annual[str(from_year)])
    wage_to = float(annual[str(to_year)])
    if wage_from <= 0:
        return {"error": f"Wage data for {from_year} is zero/missing — can't compute ratio."}
    wage_ratio = wage_to / wage_from
    block: dict[str, Any] = {
        "wage_from": round(wage_from, 2),
        "wage_to": round(wage_to, 2),
        "wage_ratio": round(wage_ratio, 4),
        "units": wages_data["_metadata"].get("units", "USD/hour, nominal"),
        "source": wages_data["_metadata"]["series_name"],
        "source_url": wages_data["_metadata"]["source_url_primary"],
        "data_as_of": wages_data["_metadata"]["data_as_of"],
    }
    preliminary = set(wages_data.get("preliminary_years") or [])
    if from_year in preliminary or to_year in preliminary:
        prelim_year = to_year if to_year in preliminary else from_year
        block["caveat"] = (
            f"{prelim_year} wages are a preliminary partial-year average; "
            f"the official annual figure finalizes after the year ends."
        )
    return block


# Phrase patterns for amount + year extraction. Compiled once. The bridge
# calls extract_inflation_args() to fill in args the iter-3 LoRA omits or
# misreads (most common failure: dropping `amount` for "a dollar in 1970
# money?", which then errors out).
_AMOUNT_WORDS = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20, "fifty": 50, "hundred": 100, "thousand": 1000,
}

# `$50`, `$1.25`, `50 dollars`, `100 bucks`, `5 cents`
_AMOUNT_DOLLAR_RE = re.compile(
    r"\$\s*(\d{1,3}(?:[,]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)",
)
_AMOUNT_DOLLAR_WORD_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s+(?:dollars?|bucks?)\b",
    re.IGNORECASE,
)
_AMOUNT_CENT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+cents?\b",
    re.IGNORECASE,
)
_AMOUNT_WORD_DOLLAR_RE = re.compile(
    r"\b(a|one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|hundred|thousand)\s+(?:dollar|buck)s?\b",
    re.IGNORECASE,
)

# `1970`, `in 1970`, `from 1970`, `since 1970`, `in 1970 money/dollars`
_YEAR_FROM_RE = re.compile(
    r"\b(?:in|from|since|of|during|back\s+in)\s+(\d{4})\b",
    re.IGNORECASE,
)
_YEAR_BARE_RE = re.compile(r"\b(\d{4})\s*(?:dollars?|money|prices?|wages?)\b", re.IGNORECASE)
_YEAR_TO_RE = re.compile(
    r"\b(?:to|in|by)\s+(\d{4})\b(?:\s*(?:dollars?|money|prices?))?",
    re.IGNORECASE,
)

# Wage-question keywords. Their presence promotes measure→'wages' so
# the LoRA's default 'cpi' doesn't strip the wage block out of the
# response. "Real wages" / "kept up with inflation" / "wages vs
# inflation" want both sides, so they promote to 'both'.
_MEASURE_BOTH_RE = re.compile(
    r"\b(?:"
    r"real\s+wages?|"
    r"wages?\s+(?:and|vs\.?|versus)\s+inflation|"
    r"inflation\s+(?:and|vs\.?|versus)\s+wages?|"
    r"wages?\s+(?:keep|kept|keeping|kept\s+up|outpace[ds]?|outpacing|lag(?:ged|ging)?)|"
    r"(?:keep|kept|keeping)\s+(?:up\s+)?with\s+inflation|"
    r"purchasing\s+power\s+of\s+(?:a\s+)?(?:wage|salary|paycheck)"
    r")\b",
    re.IGNORECASE,
)
_MEASURE_WAGES_RE = re.compile(
    r"\b(?:wages?|salar(?:y|ies)|paycheck|earnings?|hourly\s+(?:pay|rate|wage))\b",
    re.IGNORECASE,
)

# Region-detection regex. Each pattern maps to a region key. Multi-word
# phrases listed first so they beat shorter overlaps.
_REGION_DETECT_PATTERNS: list[tuple[str, str]] = [
    # Multi-word forms first to beat shorter overlaps.
    (r"\bhong\s*kong\s+sar\b", "hk_sar"),
    (r"\bhong\s*kong\b", "hk_sar"),
    (r"\bin\s+hk\b", "hk_sar"),
    (r"\bhkd\b|\bhk\$", "hk_sar"),
    (r"\bmainland\s+china\b|\bchina\s+mainland\b|\bprc\b", "cn_mainland"),
    # Bare "china" maps to mainland (most common usage; HK and TW have
    # their own keywords). Don't trigger on "south china sea" etc.
    (r"\b(?<!south\s)(?<!east\s)(?<!west\s)china\b", "cn_mainland"),
    (r"\bcny\b|\brenminbi\b|\bchinese\s+yuan\b|\b\d+\s+yuan\b", "cn_mainland"),
    (r"\bjapan(?:ese)?\b", "japan"),
    (r"\bjpy\b|\b\d+\s+yen\b|\bjapanese\s+yen\b", "japan"),
    (r"\bsouth\s+korea\b|\bsouth\s+korean\b|\brok\b", "south_korea"),
    (r"\b(?<!north\s)korea\b", "south_korea"),
    (r"\bkrw\b|\b\d+\s+won\b|\bkorean\s+won\b", "south_korea"),
    # US patterns — only used in comparison mode (single-region calls
    # default to "us" when nothing else matches). Conservative —
    # require either an explicit unambiguous form OR a context anchor
    # that makes "us" clearly the country (in the US, US inflation,
    # the US's CPI, etc.) and not the pronoun.
    (
        r"\bunited\s+states\b|\busa\b|\bu\.s\.a?\b|\bamerica(?:n)?\b|"
        r"\bin\s+(?:the\s+)?us\b|"
        r"\b(?:the\s+)?us\s+(?:dollar|inflation|cpi|economy|population|prices?|wages?|gov|government)\b|"
        r"\bus\s+vs\.?\b|\bvs\.?\s+us\b",
        "us",
    ),
]


# Wiki-query shapes that signal "this should have been inflation".
# Used by the bridge's wiki→inflation redirect (in llm.py): when the
# LoRA picks wiki but its query string echoes one of these patterns
# AND extract_inflation_args produces a from_year, the call is
# rewritten. iter-3 was trained without the inflation tool, so it
# defaults to wiki for "dollar in YYYY"/"X worth today" phrasings.
_WIKI_TO_INFLATION_REDIRECT_RE = re.compile(
    r"\b(?:"
    r"dollar\s+(?:in|from|of)\s+\d{4}|"   # "dollar in 1865"
    r"\d{4}\s+dollars?|"                  # "1865 dollars"
    r"worth\s+(?:today|now)|"             # "X worth today"
    r"in\s+\d{4}\s+(?:money|dollars?|prices?)|"  # "in 1865 money"
    r"purchasing\s+power|"
    r"adjusted\s+for\s+inflation|"
    # Foreign currency + year — "1 yuan in 1990", "100 yen in 1970",
    # "1000 won in 1980". Wiki has no useful answer for these; the
    # inflation tool with the right region does.
    r"\d+\s+(?:yuan|yen|won|hkd|cny|jpy|krw)\s+(?:in|from|of)\s+\d{4}|"
    r"(?:yuan|yen|won|hkd|cny|jpy|krw)\s+(?:in|from|of|since)\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)

# Item phrase → canonical key. Multi-word entries listed first so they
# beat shorter overlapping matches ("natural gas" wins over bare "gas",
# "ground beef" over "beef"). Order is sorted at use-time by phrase
# length descending.
_ITEM_PHRASE_MAP: dict[str, str] = {
    "natural gas": "natural_gas",
    "utility gas": "natural_gas",
    "therm of gas": "natural_gas",
    "gallon of milk": "milk_gallon",
    "whole milk": "milk_gallon",
    "loaf of bread": "bread",
    "white bread": "bread",
    "ground beef": "ground_beef",
    "hamburger meat": "ground_beef",
    "whole chicken": "chicken",
    "dozen eggs": "eggs",
    "carton of eggs": "eggs",
    "ground coffee": "coffee",
    "ground roast coffee": "coffee",
    "kilowatt hour": "electricity",
    "milk": "milk_gallon",
    "bread": "bread",
    "eggs": "eggs",
    "beef": "ground_beef",
    "chicken": "chicken",
    "bacon": "bacon",
    "bananas": "bananas",
    "banana": "bananas",
    "tomatoes": "tomatoes",
    "tomato": "tomatoes",
    "coffee": "coffee",
    "sugar": "sugar",
    "electricity": "electricity",
    "kwh": "electricity",
    "gasoline": "gasoline",
    "petrol": "gasoline",
    "gas": "gasoline",  # bare "gas" defaults to gasoline (most common query)
}


def extract_inflation_args(user_text: str | None) -> dict[str, Any]:
    """Pull amount + year hints out of the current user message.

    Returns a dict with whichever of {amount, from_year, to_year} could
    be extracted. The bridge merges this on top of the LoRA's args so we
    fix the common misses (omitted amount, swapped years) without
    overriding correct extractions when both agree.

    "today" / "now" deliberately do NOT set to_year — the tool defaults
    to the latest dataset year, which is the right answer.
    """
    if not user_text:
        return {}
    text = user_text.strip()
    out: dict[str, Any] = {}

    # Amount — try $-prefix first (most explicit), then word-form, then cents.
    # `$1925` is unambiguous; `1925 dollars` reads as "dollars from 1925"
    # in this domain. Skip word-form matches that look like a 4-digit year.
    m = _AMOUNT_DOLLAR_RE.search(text)
    if m:
        try:
            out["amount"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    if "amount" not in out:
        m = _AMOUNT_DOLLAR_WORD_RE.search(text)
        if m:
            try:
                v = float(m.group(1).replace(",", ""))
                if not (1700 <= v <= 2100 and v.is_integer()):
                    out["amount"] = v
            except ValueError:
                pass
    if "amount" not in out:
        m = _AMOUNT_CENT_RE.search(text)
        if m:
            try:
                out["amount"] = round(float(m.group(1)) / 100.0, 4)
            except ValueError:
                pass
    if "amount" not in out:
        m = _AMOUNT_WORD_DOLLAR_RE.search(text)
        if m:
            word = m.group(1).lower()
            if word in _AMOUNT_WORDS:
                out["amount"] = float(_AMOUNT_WORDS[word])

    # Years — gather all 4-digit candidates with their context.
    years_found: list[tuple[int, int]] = []  # (position, year)
    for m in re.finditer(r"\b(\d{4})\b", text):
        try:
            y = int(m.group(1))
        except ValueError:
            continue
        if 1700 <= y <= 2100:  # plausible year range
            years_found.append((m.start(), y))

    if years_found:
        # "1970 dollars/money" wins first — bare year glued to a currency
        # word is the strongest source-year signal ("1925 dollars in
        # 2026" should put 1925 in from_year, not 2026).
        m = _YEAR_BARE_RE.search(text)
        if m:
            try:
                out["from_year"] = int(m.group(1))
            except ValueError:
                pass
        # Otherwise the first "in/from/since/of/during YEAR" wins from_year.
        if "from_year" not in out:
            m = _YEAR_FROM_RE.search(text)
            if m:
                try:
                    out["from_year"] = int(m.group(1))
                except ValueError:
                    pass
        # Fallback: earliest year mentioned is the from_year.
        if "from_year" not in out:
            out["from_year"] = years_found[0][1]
        # If two distinct years appear, the later one becomes to_year
        # (covers "1970 to 2000", "1980 in 2020 dollars").
        distinct = sorted({y for _, y in years_found})
        if len(distinct) >= 2:
            from_y = out.get("from_year")
            for y in distinct:
                if y != from_y:
                    out["to_year"] = y
                    break

    # Item detection — match the longest phrase first so "natural gas"
    # beats "gas". A hit pre-fills the LoRA's `item` arg so the tool can
    # return a real BLS price instead of the LoRA hallucinating one.
    text_lower = text.lower()
    for phrase in sorted(_ITEM_PHRASE_MAP.keys(), key=lambda p: -len(p)):
        if re.search(rf"\b{re.escape(phrase)}\b", text_lower):
            out["item"] = _ITEM_PHRASE_MAP[phrase]
            break

    # Measure inference — keywords pull 'cpi' default toward 'wages' or
    # 'both'. Comparison phrasings ("real wages", "kept up with
    # inflation") force 'both' so the JSON includes the real-wage delta
    # the LoRA can't reliably compute on its own.
    if _MEASURE_BOTH_RE.search(text):
        out["measure"] = "both"
    elif _MEASURE_WAGES_RE.search(text):
        out["measure"] = "wages"

    # Region inference — pull explicit "Hong Kong" / "HKD" cues out so
    # the tool routes to the right dataset. iter-3 doesn't know the
    # `region` arg exists; the bridge merge will fill it from here.
    # Collect ALL matching regions to support cross-region comparison.
    matched_regions: list[str] = []
    for pat, key in _REGION_DETECT_PATTERNS:
        if re.search(pat, text_lower) and key not in matched_regions:
            matched_regions.append(key)
    if matched_regions:
        # Comparison cues: "compare", "vs", "versus", "and" between two
        # regions — promote to multi-region mode.
        is_compare = bool(re.search(
            r"\bcompare\b|\bvs\.?\b|\bversus\b|\bbetween\b|\bacross\b",
            text_lower,
        )) or len(matched_regions) >= 2
        if is_compare and len(matched_regions) >= 2:
            out["regions"] = ",".join(matched_regions)
        else:
            out["region"] = matched_regions[0]

    return out
