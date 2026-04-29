"""ALICE estimator — Asset Limited, Income Constrained, Employed.

ALICE households are those that earn ABOVE the federal poverty line
(so don't qualify for most benefits) but BELOW a "household survival
budget" (so still can't actually afford basics). Methodology mirrors
United for ALICE (https://www.unitedforalice.org). They publish state-
and national-level reports every two years; we cross-validate against
their numbers.

Architecture:
- Hardcoded constants live in bridge/data/alice/alice_baseline.json
  (FPL, HUD FMR, KFF premium, USDA food, childcare, ACS income brackets,
  household composition). Refresh in-place; fetchers can be added later.
- Variable components SHOULD pull from our existing series when present:
  - food: cross-check via BLS AP basket sum vs USDA TFP total
  - transport gas component: cross-check via EIA retail gasoline
  - energy: cross-check via BLS electricity + natural_gas
- Code does ALL math; LLM only paraphrases.

Outputs:
1. Survival budget breakdown (housing, food, transport, healthcare,
   childcare, technology, taxes) for the canonical 4-person household.
2. ALICE threshold = sum of those line items.
3. Poverty rate + ALICE rate estimates derived by interpolating the
   ACS B19001 income brackets between the household-size-weighted FPL
   and the household-size-weighted ALICE threshold.
4. Comparison vs United for ALICE published values.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from bridge.tools._regions import resolve_year as _resolve_year

log = logging.getLogger("bridge.tools")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "alice"
_BASELINE_PATH = _DATA_DIR / "alice_baseline.json"


# --- household composition templates --------------------------------
#
# Each composition encodes the structural choices that drive the
# survival budget: how many bedrooms (housing), how many people +
# kids vs adults (food + healthcare coverage choice), how many
# drivers, and the federal filing status (which determines the
# bracket schedule + standard deduction + which CTC/EITC schedule
# applies). United for ALICE publishes per-state budget tables for
# similar compositions; we mirror their canonical set so cross-
# validation lines up.

COMPOSITIONS: dict[str, dict[str, Any]] = {
    "1A0K": {
        "label": "Single adult",
        "adults": 1,
        "kids": 0,
        "bedrooms": 1,
        "drivers": 1,
        "filing_status": "single",
        "coverage": "single",
        "household_size": 1,
    },
    "2A0K": {
        "label": "Couple, no children",
        "adults": 2,
        "kids": 0,
        "bedrooms": 1,
        "drivers": 2,
        "filing_status": "MFJ",
        "coverage": "two_singles",
        "household_size": 2,
    },
    "1A1K": {
        "label": "Single parent + 1 child",
        "adults": 1,
        "kids": 1,
        "bedrooms": 2,
        "drivers": 1,
        "filing_status": "HoH",
        "coverage": "family",
        "household_size": 2,
    },
    "2A1K": {
        "label": "Couple + 1 child",
        "adults": 2,
        "kids": 1,
        "bedrooms": 2,
        "drivers": 2,
        "filing_status": "MFJ",
        "coverage": "family",
        "household_size": 3,
    },
    "2A2K": {
        "label": "Couple + 2 children (4-person canonical)",
        "adults": 2,
        "kids": 2,
        "bedrooms": 2,
        "drivers": 2,
        "filing_status": "MFJ",
        "coverage": "family",
        "household_size": 4,
    },
    "2A3K": {
        "label": "Couple + 3 children",
        "adults": 2,
        "kids": 3,
        "bedrooms": 3,
        "drivers": 2,
        "filing_status": "MFJ",
        "coverage": "family",
        "household_size": 5,
    },
}

# Backward-compat: when a caller passes `household_size` instead of
# `composition`, map to a sensible default. Childless singles → 1A0K,
# couples → 2A0K, three-person assumes a couple + 1 kid (more common
# than single + 2 kids), four → canonical, five-plus → 2A3K (caps).
_SIZE_TO_COMPOSITION: dict[int, str] = {
    1: "1A0K", 2: "2A0K", 3: "2A1K", 4: "2A2K",
    5: "2A3K", 6: "2A3K", 7: "2A3K",
}


def _load_baseline() -> dict[str, Any]:
    if not _BASELINE_PATH.is_file():
        raise FileNotFoundError(
            f"ALICE baseline missing at {_BASELINE_PATH}. "
            "Restore from git or rebuild the alice/ data directory.",
        )
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def _scale_alice_threshold_for_size(threshold_4p: float,
                                    household_size: int) -> float:
    """Approximate per-household-size ALICE threshold scaling.
    United for ALICE published per-state thresholds vary roughly as:
      1-person ≈ 0.42 × 4-person threshold
      2-person ≈ 0.66 × 4-person
      3-person ≈ 0.83 × 4-person
      4-person = 1.00
      5-person ≈ 1.18 × 4-person
      6-person ≈ 1.35 × 4-person
      7+      ≈ 1.50 × 4-person
    Calibrated against United for ALICE 2022 NJ + national budget tables."""
    factors = {1: 0.42, 2: 0.66, 3: 0.83, 4: 1.00,
               5: 1.18, 6: 1.35, 7: 1.50}
    return threshold_4p * factors.get(household_size, factors[7])


# ---- bracket interpolation -------------------------------------------

# ACS B19001 brackets are right-open ranges. The midpoints we use to
# interpolate "% of households below threshold T" are linear within
# each bracket: if T sits at fraction f through the bracket, we count
# f × bracket_share of households as below T.
_BRACKETS: tuple[tuple[str, float, float | None], ...] = (
    ("lt_10k",     0,      10_000),
    ("10_15k",     10_000, 15_000),
    ("15_20k",     15_000, 20_000),
    ("20_25k",     20_000, 25_000),
    ("25_30k",     25_000, 30_000),
    ("30_35k",     30_000, 35_000),
    ("35_40k",     35_000, 40_000),
    ("40_45k",     40_000, 45_000),
    ("45_50k",     45_000, 50_000),
    ("50_60k",     50_000, 60_000),
    ("60_75k",     60_000, 75_000),
    ("75_100k",    75_000, 100_000),
    ("100_125k",   100_000, 125_000),
    ("125_150k",   125_000, 150_000),
    ("150_200k",   150_000, 200_000),
    # 200k+ has no upper bound; treat its midpoint at 250k for
    # interpolation purposes (rough, but only matters for thresholds
    # well above 200k which aren't ALICE-relevant).
    ("200k_plus",  200_000, 300_000),
)


def _pct_below(threshold: float,
               brackets: dict[str, float]) -> float:
    """Return fraction of households earning below `threshold` using
    linear interpolation across the ACS bracket distribution."""
    cumulative = 0.0
    for key, lo, hi in _BRACKETS:
        share = float(brackets.get(key, 0.0))
        if hi is None or threshold >= hi:
            cumulative += share
            continue
        if threshold <= lo:
            break
        # Threshold falls inside this bracket: count the proportional
        # share linearly.
        f = (threshold - lo) / (hi - lo)
        cumulative += share * f
        break
    return cumulative


def _weighted_threshold(per_size_value_fn,
                        composition: dict[str, float]) -> float:
    """Compose a household-size-weighted scalar from a function that
    returns the per-size value. composition keys are like '1_person',
    '2_person', '7_plus'."""
    keymap = {"1_person": 1, "2_person": 2, "3_person": 3,
              "4_person": 4, "5_person": 5, "6_person": 6,
              "7_plus": 7}
    total = 0.0
    weight_sum = 0.0
    for key, weight in composition.items():
        size = keymap.get(key)
        if size is None or weight <= 0:
            continue
        total += weight * per_size_value_fn(size)
        weight_sum += weight
    return total / weight_sum if weight_sum > 0 else 0.0


# ---- variable-component cross-checks --------------------------------

def _bls_food_basket_check(year: int) -> dict[str, Any] | None:
    """Cross-validate USDA TFP against a sum of BLS AP item prices ×
    typical-basket quantities for a 4-person family. Returns None when
    items dataset isn't available — non-fatal."""
    try:
        from bridge.tools._inflation import _load_items, _resolve_item
        items_data = _load_items()
    except Exception:  # noqa: BLE001
        return None
    if items_data is None:
        return None
    # Annual quantities for a 4-person household, USDA TFP-aligned:
    # bread 80 loaves, eggs 60 dozen, milk 78 gallons, ground_beef 80 lb,
    # chicken 100 lb, bananas 80 lb, tomatoes 60 lb, coffee 12 lb,
    # sugar 40 lb. (Coverage subset; doesn't include produce/dairy that
    # isn't in BLS AP.)
    basket = {
        "bread": 80, "eggs": 60, "milk_gallon": 78,
        "ground_beef": 80, "chicken": 100, "bananas": 80,
        "tomatoes": 60, "coffee": 12, "sugar": 40,
    }
    total = 0.0
    contributors: list[dict] = []
    for key, qty in basket.items():
        entry = _resolve_item(key, items_data)
        if entry is None:
            continue
        annual = entry.get("annual", {})
        price = annual.get(str(year))
        if price is None:
            continue
        line = float(price) * qty
        total += line
        contributors.append({
            "item": entry["label"],
            "unit_price": round(float(price), 4),
            "qty_per_year": qty,
            "annual_cost": round(line, 2),
        })
    if not contributors:
        return None
    return {
        "subtotal_annual": round(total, 2),
        "year": year,
        "contributors": contributors,
        "note": (
            "BLS AP partial basket — covers staples only. USDA Thrifty "
            "Food Plan also includes produce, dairy, and prepared foods "
            "not in BLS AP, so the BLS basket runs ~50-70% of the TFP "
            "total. Use as a directional sanity check, not a substitute."
        ),
    }


def _eia_gas_check(year: int) -> dict[str, Any] | None:
    """Annual gasoline cost using EIA retail price + a 2-driver baseline."""
    try:
        from bridge.tools._inflation import _load_items, _resolve_item
        items_data = _load_items()
        if items_data is None:
            return None
        gas = _resolve_item("gasoline", items_data)
        if gas is None:
            return None
        annual = gas.get("annual", {})
        price = annual.get(str(year))
        if price is None:
            return None
    except Exception:  # noqa: BLE001
        return None
    # 2 drivers × 12,000 mi/yr ÷ 24 mpg = 1,000 gal/yr
    gallons_per_year = 1000.0
    annual_cost = float(price) * gallons_per_year
    return {
        "year": year,
        "price_per_gallon": round(float(price), 4),
        "gallons_per_year": gallons_per_year,
        "annual_gas_cost": round(annual_cost, 2),
        "source": gas.get("source_url",
                          "EIA retail gasoline (regular, US avg) merged "
                          "with BLS APU000074714"),
        "note": (
            "Gasoline alone — full transport budget also includes "
            "maintenance, insurance, and depreciation, which roughly "
            "double the gas-only number. The baseline transport line "
            "uses ALICE-aligned $/mile."
        ),
    }


# ---- core compute ----------------------------------------------------

def _compute_survival_budget_for_composition(yr: dict, comp: dict, year: int) -> dict:
    """Build the survival budget for a specific household composition.
    Each line item is computed from per-composition factors (bedrooms,
    food persons, coverage tier, drivers, kids) rather than a single
    scaling on a 4-person canonical — that fixes the "1-person budget
    still includes 4-person childcare" bug from the v1 model."""

    # Housing: HUD FMR 2-bedroom national avg, scaled by bedroom count.
    # Calibrated against typical FMR 1bdr/2bdr/3bdr ratios — a 1bdr
    # runs ~85% of a 2bdr, a 3bdr runs ~30% above.
    bdr_factor = {1: 0.85, 2: 1.00, 3: 1.30}.get(comp["bedrooms"], 1.0)
    housing_annual = yr["hud_fmr_2bdr_us_avg_monthly"] * 12 * bdr_factor

    # Food: USDA Thrifty Food Plan adult-equivalents. The published
    # 4-person TFP value is 2 adults + 2 kids ≈ 3.7 adult-equivalents
    # (kids ~85% of an adult's caloric need). Reverse to per-AE then
    # rebuild for this composition.
    pp_adult_eq = (yr["usda_thrifty_food_4person_monthly"] * 12) / 3.7
    food_ae = comp["adults"] * 1.0 + comp["kids"] * 0.85
    food_annual = pp_adult_eq * food_ae

    # Healthcare: tier varies by composition.
    coverage = comp["coverage"]
    if coverage == "single":
        single_share = yr.get("kff_single_worker_share_annual", 1400)
        single_oop = yr.get("kff_single_oop_avg_annual", 800)
        healthcare_annual = single_share + single_oop
        coverage_label = "Single coverage"
    elif coverage == "two_singles":
        single_share = yr.get("kff_single_worker_share_annual", 1400)
        single_oop = yr.get("kff_single_oop_avg_annual", 800)
        healthcare_annual = (single_share + single_oop) * 2
        coverage_label = "2× single coverage"
    else:  # family
        healthcare_annual = (
            yr["kff_family_worker_share_annual"]
            + yr["kff_family_oop_avg_annual"]
        )
        coverage_label = "Family coverage"

    # Transport: per-driver. Cached value is for 2 drivers; halve.
    transport_per_driver = yr["transport_2_drivers_annual"] / 2
    transport_annual = transport_per_driver * comp["drivers"]

    # Childcare: zero if no kids, else per-kid scaled. The cached
    # 2-kid value is the typical center-based national avg.
    if comp["kids"] > 0:
        per_kid = yr["childcare_2_kids_annual"] / 2
        childcare_annual = per_kid * comp["kids"]
    else:
        childcare_annual = 0

    # Technology: roughly fixed per household — couple sharing one
    # internet bill costs slightly more (more devices) than a single.
    tech_factor = 1.0 if comp["adults"] == 1 else 1.15
    technology_annual = yr["technology_annual"] * tech_factor

    pretax_subtotal = (
        housing_annual + food_annual + healthcare_annual
        + childcare_annual + transport_annual + technology_annual
    )

    # Bracket-based gross-up — solve for gross s.t. gross - tax(gross)
    # == pretax_subtotal. The tax module handles federal brackets per
    # filing status, CTC, simplified EITC, payroll, and avg state.
    from bridge.tools._alice_tax import gross_up
    gross_total, tax_breakdown = gross_up(
        pretax_subtotal,
        status=comp["filing_status"],
        num_kids=comp["kids"],
        year=year,
    )

    return {
        "housing": {
            "annual": round(housing_annual, 2),
            "source": yr["hud_fmr_source"],
            "explanation": (
                f"{comp['bedrooms']}-bedroom apartment rent at HUD's "
                f"Fair Market Rent national average — scaled from the "
                f"2-bedroom baseline by ×{bdr_factor:.2f} for "
                f"{comp['bedrooms']} bedroom(s). Includes utilities."
            ),
        },
        "food": {
            "annual": round(food_annual, 2),
            "source": yr["usda_food_source"],
            "explanation": (
                f"USDA Thrifty Food Plan: cheapest grocery basket "
                f"meeting USDA dietary guidelines, scaled to "
                f"{comp['adults']} adult(s) + {comp['kids']} child(ren) "
                f"= {food_ae:.2f} adult-equivalents."
            ),
        },
        "healthcare": {
            "annual": round(healthcare_annual, 2),
            "source": yr["kff_source"],
            "coverage_tier": coverage_label,
            "explanation": (
                f"{coverage_label}: worker contribution to "
                f"employer-sponsored insurance plus average out-of-"
                f"pocket spending (KFF EHBS). Childless singles use "
                f"single coverage (~$1.5K/yr); families need family "
                f"coverage (~$8K/yr)."
            ),
        },
        "childcare": {
            "annual": round(childcare_annual, 2),
            "source": (
                yr["childcare_source"] if comp["kids"] > 0
                else "N/A — no children in this household"
            ),
            "explanation": (
                f"Center-based care for {comp['kids']} child(ren), US "
                f"national average. Zero for childless households."
                if comp["kids"] > 0 else
                "N/A — no children in this composition."
            ),
        },
        "transport": {
            "annual": round(transport_annual, 2),
            "source": yr["transport_source"],
            "explanation": (
                f"{comp['drivers']} vehicle(s) at 12,000 mi/yr each — "
                f"gas + routine maintenance only. Vehicle ownership "
                f"(insurance, depreciation) sits outside this line, "
                f"which is why our number runs lower than UFA's."
            ),
        },
        "technology": {
            "annual": round(technology_annual, 2),
            "source": yr["technology_source"],
            "explanation": (
                "Smartphone with data plan plus home broadband. "
                "Excludes streaming subs and devices beyond replacement "
                "amortization. Family households scale ~15% higher "
                "for additional devices."
            ),
        },
        "subtotal_pretax": round(pretax_subtotal, 2),
        "subtotal_explanation": (
            "Sum of the six basic-needs line items above. Pre-tax — "
            "the gross income needed to take home this much (after "
            "federal income tax, CTC + EITC offsets, payroll, and "
            "average state tax) is on the next lines."
        ),
        "taxes": {
            **tax_breakdown,
            "explanation": (
                f"Federal income tax via 2024 brackets for "
                f"{tax_breakdown['filing_status']} filers, less Child "
                f"Tax Credit ({tax_breakdown['num_kids']} kid(s) × "
                f"$2,000) and EITC, plus 7.65% FICA (Social Security + "
                f"Medicare) and a 5% national-average state income "
                f"tax. Iteratively solved because EITC phases out "
                f"non-linearly with income."
            ),
        },
        "total_for_size": round(gross_total, 2),
        "total_explanation": (
            f"Gross annual income required to cover the basics for a "
            f"{comp['label'].lower()} in {year}. United for ALICE "
            f"publishes the canonical 4-person threshold for cross-"
            f"validation; we report ours alongside theirs in the "
            f"comparison block below."
        ),
    }


def _compute_survival_budget(yr: dict, household_size: int) -> dict:
    """Build the per-household survival budget. Line items are computed
    for the 4-person canonical (the household composition United for
    ALICE publishes for cross-validation), then a single scaling factor
    derives the threshold for other sizes — UFA doesn't publish full
    per-size line-item breakdowns, so a single-line scaling is the
    most honest projection we can make.

    The output always includes:
      total_4person:    4-person baseline gross income needed (cross-
                        validation anchor; matches what UFA publishes)
      total_for_size:   threshold scaled for the requested household
                        size (= total_4person when household_size == 4)
      scaling_factor:   1.0 when size == 4, else per-size factor
    """
    housing_annual = yr["hud_fmr_2bdr_us_avg_monthly"] * 12
    food_annual = yr["usda_thrifty_food_4person_monthly"] * 12
    healthcare_annual = (
        yr["kff_family_worker_share_annual"]
        + yr["kff_family_oop_avg_annual"]
    )
    childcare_annual = yr["childcare_2_kids_annual"]
    transport_annual = yr["transport_2_drivers_annual"]
    technology_annual = yr["technology_annual"]
    pretax_subtotal = (
        housing_annual + food_annual + healthcare_annual
        + childcare_annual + transport_annual + technology_annual
    )
    tax_rate = float(yr["effective_tax_rate"])
    # Gross-up: subtotal is the after-tax cost; gross income needed
    # = subtotal / (1 - tax_rate).
    total_4person = pretax_subtotal / (1 - tax_rate) if tax_rate < 1 else pretax_subtotal
    taxes = total_4person - pretax_subtotal
    scaling_factor = _scale_alice_threshold_for_size(1.0, household_size)
    total_for_size = total_4person * scaling_factor

    return {
        "housing": {
            "annual": round(housing_annual, 2),
            "source": yr["hud_fmr_source"],
            "explanation": (
                "Two-bedroom apartment rent at HUD's Fair Market Rent "
                "national average — the 40th-percentile asking rent "
                "across US metro areas. Includes utilities."
            ),
        },
        "food": {
            "annual": round(food_annual, 2),
            "source": yr["usda_food_source"],
            "explanation": (
                "USDA Thrifty Food Plan for a family of four (2 adults "
                "age 20-50, 2 children age 6-8 + 9-11). Lowest-cost "
                "grocery basket meeting USDA dietary recommendations."
            ),
        },
        "healthcare": {
            "annual": round(healthcare_annual, 2),
            "source": yr["kff_source"],
            "explanation": (
                "Worker contribution to employer-sponsored family "
                "health insurance plus average family out-of-pocket "
                "spending (KFF EHBS). Assumes employer coverage; ACA "
                "marketplace prices for the uninsured run higher."
            ),
        },
        "childcare": {
            "annual": round(childcare_annual, 2),
            "source": yr["childcare_source"],
            "explanation": (
                "Annual cost of center-based care for two children, US "
                "national average (Child Care Aware America). Relative "
                "or in-home care costs less; in-home nannying costs more."
            ),
        },
        "transport": {
            "annual": round(transport_annual, 2),
            "source": yr["transport_source"],
            "explanation": (
                "Operating two vehicles at 12,000 mi/yr each — gas + "
                "routine maintenance only. Vehicle ownership "
                "(insurance, depreciation) sits outside this line, "
                "which is why our number runs lower than UFA's."
            ),
        },
        "technology": {
            "annual": round(technology_annual, 2),
            "source": yr["technology_source"],
            "explanation": (
                "Smartphone with data plan plus home broadband internet "
                "at typical baseline plans. Excludes streaming "
                "subscriptions and devices beyond replacement amortization."
            ),
        },
        "subtotal_pretax": round(pretax_subtotal, 2),
        "subtotal_explanation": (
            "Sum of the six basic-needs line items above. Pre-tax — "
            "the gross income needed to actually take home this much "
            "is on the next line."
        ),
        "taxes": {
            "annual": round(taxes, 2),
            "rate_pct": round(tax_rate * 100, 2),
            "source": yr["tax_source"],
            "explanation": (
                "Effective rate combining federal income tax (post-"
                "Child Tax Credit), payroll tax (FICA), and an "
                "average state income tax for a family in the ALICE "
                "income band."
            ),
        },
        "total_4person": round(total_4person, 2),
        "total_4person_explanation": (
            "Gross annual income required to cover the basics for a "
            "4-person household. United for ALICE publishes this as "
            "the canonical ALICE threshold for cross-validation."
        ),
        "household_size": household_size,
        "scaling_factor": round(scaling_factor, 4),
        "total_for_size": round(total_for_size, 2),
        "scaling_explanation": (
            "Per-household-size scaling factor calibrated to United "
            "for ALICE's published per-state budget tables: "
            "1p≈0.42, 2p≈0.66, 3p≈0.83, 4p=1.00, 5p≈1.18, 6p≈1.35, "
            "7p+≈1.50 of the 4-person baseline."
        ),
    }


def _estimate_population_shares(yr: dict,
                                threshold_4p_total: float) -> dict:
    """Estimate % poverty + % ALICE for the US using a household-size-
    weighted FPL and a household-size-weighted ALICE threshold against
    the ACS B19001 bracket distribution."""
    composition = yr["household_composition_pct"]
    fpl = yr["fpl"]
    fpl_for_size = lambda s: float(  # noqa: E731
        fpl.get(str(s), fpl.get("6", 0)) + fpl["extra_per_person"] * max(s - 6, 0)
    )
    weighted_fpl = _weighted_threshold(fpl_for_size, composition)
    weighted_alice = _weighted_threshold(
        lambda s: _scale_alice_threshold_for_size(threshold_4p_total, s),
        composition,
    )
    brackets = yr["household_income_brackets"]
    pct_below_fpl = _pct_below(weighted_fpl, brackets)
    pct_below_alice = _pct_below(weighted_alice, brackets)
    pct_alice = max(0.0, pct_below_alice - pct_below_fpl)
    pct_above_alice = max(0.0, 1.0 - pct_below_alice)
    return {
        "weighted_fpl": round(weighted_fpl, 2),
        "weighted_alice_threshold": round(weighted_alice, 2),
        "pct_poverty": round(pct_below_fpl, 4),
        "pct_alice": round(pct_alice, 4),
        "pct_above_alice": round(pct_above_alice, 4),
        "method": (
            "Linear interpolation across ACS B19001 bracketed counts; "
            "FPL + ALICE thresholds weighted by ACS B11016 household-size "
            "distribution."
        ),
    }


def _resolve_composition(
    composition: str | None,
    household_size: int | str | None,
) -> tuple[str, dict[str, Any]] | tuple[None, str]:
    """Resolve `composition` arg first; fall back to `household_size`.
    Returns (key, comp_dict) on success or (None, error_msg) on failure."""
    if composition:
        key = str(composition).strip().upper()
        if key in COMPOSITIONS:
            return key, COMPOSITIONS[key]
        return None, (
            f"Unknown composition {composition!r}. Available: "
            f"{', '.join(COMPOSITIONS)}"
        )
    if household_size is not None:
        try:
            size = int(household_size)
        except (TypeError, ValueError):
            return None, (
                f"household_size must be integer, got {household_size!r}"
            )
        if size < 1:
            return None, "household_size must be ≥ 1"
        size = min(size, 7)
        key = _SIZE_TO_COMPOSITION.get(size, "2A2K")
        return key, COMPOSITIONS[key]
    # Default: 4-person canonical
    return "2A2K", COMPOSITIONS["2A2K"]


def _alice(year: int | str | None = None,
           composition: str | None = None,
           household_size: int | str | None = None) -> str:
    """Estimate the US ALICE share for a given year + household composition.

    Args:
        year: 2016, 2018, 2020, 2022, 2023, 2024 (default = latest).
        composition: one of "1A0K" (single), "2A0K" (couple),
                     "1A1K" (single parent + 1 child),
                     "2A1K" (couple + 1 child),
                     "2A2K" (couple + 2 children, canonical),
                     "2A3K" (couple + 3 children).
        household_size: 1-7 — backward-compatible alternative to
                     `composition`. Maps via `_SIZE_TO_COMPOSITION`.
    """
    try:
        baseline = _load_baseline()
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    by_year = baseline.get("by_year", {})
    available = sorted(int(y) for y in by_year)
    if not available:
        return json.dumps({"error": "ALICE baseline has no years"})

    requested: int | None
    if year is None or year == "":
        requested = None
    else:
        try:
            requested = int(year)
        except (TypeError, ValueError):
            return json.dumps({"error": f"year must be integer, got {year!r}"})
    chosen_year = _resolve_year(requested, available)
    yr = by_year[str(chosen_year)]

    comp_key, comp_or_err = _resolve_composition(composition, household_size)
    if comp_key is None:
        return json.dumps({"error": comp_or_err})
    comp = comp_or_err

    survival_budget = _compute_survival_budget_for_composition(
        yr, comp, chosen_year,
    )
    total_for_size = survival_budget["total_for_size"]

    # Cross-validation anchor: always recompute the 4-person canonical
    # so we can compare to UFA's published threshold even when the
    # user picked a different composition.
    if comp_key == "2A2K":
        canonical_budget = survival_budget
    else:
        canonical_budget = _compute_survival_budget_for_composition(
            yr, COMPOSITIONS["2A2K"], chosen_year,
        )
    total_4person = canonical_budget["total_for_size"]

    population = _estimate_population_shares(yr, total_4person)

    ref = yr.get("united_for_alice_reference", {})
    ufa_threshold = ref.get("alice_threshold_4person_us", 0)
    threshold_delta = total_4person - ufa_threshold if ref else None
    pct_alice_delta = (
        population["pct_alice"] - ref.get("pct_alice", 0)
        if ref else None
    )

    bls_food_check = _bls_food_basket_check(chosen_year)
    eia_gas_check = _eia_gas_check(chosen_year)

    # Interpretation: composition-aware leading line.
    if comp_key == "2A2K":
        budget_line = (
            f"For a {comp['label'].lower()} in {chosen_year}, the "
            f"survival budget totals ${total_for_size:,.0f}/yr "
            f"(vs United for ALICE's ${ufa_threshold:,.0f}, "
            f"delta ${threshold_delta:+,.0f}). Filing as "
            f"{comp['filing_status']}; effective tax "
            f"{survival_budget['taxes']['effective_rate_pct']}% "
            f"after CTC/EITC."
        )
    else:
        budget_line = (
            f"For a {comp['label'].lower()} in {chosen_year}, the "
            f"survival budget is ${total_for_size:,.0f}/yr "
            f"(filing {comp['filing_status']}; effective tax "
            f"{survival_budget['taxes']['effective_rate_pct']}% "
            f"after CTC/EITC). The 4-person canonical baseline for "
            f"cross-validation against UFA: ${total_4person:,.0f} "
            f"vs UFA's ${ufa_threshold:,.0f}."
        )
    population_line = (
        f"Estimated US-wide: {population['pct_poverty']*100:.1f}% in "
        f"poverty + {population['pct_alice']*100:.1f}% ALICE = "
        f"{(population['pct_poverty']+population['pct_alice'])*100:.1f}% "
        f"below survival; United for ALICE published "
        f"{ref.get('pct_poverty', 0)*100:.0f}%+{ref.get('pct_alice', 0)*100:.0f}% "
        f"= {(ref.get('pct_poverty', 0)+ref.get('pct_alice', 0))*100:.0f}%."
    )

    # Build serializable composition descriptors for the panel.
    available_compositions = [
        {"key": k, "label": v["label"], "household_size": v["household_size"],
         "adults": v["adults"], "kids": v["kids"],
         "filing_status": v["filing_status"]}
        for k, v in COMPOSITIONS.items()
    ]

    out: dict[str, Any] = {
        "year": chosen_year,
        "year_requested": requested,
        "available_years": available,
        "composition": comp_key,
        "composition_details": {
            "label": comp["label"],
            "adults": comp["adults"],
            "kids": comp["kids"],
            "bedrooms": comp["bedrooms"],
            "drivers": comp["drivers"],
            "filing_status": comp["filing_status"],
            "household_size": comp["household_size"],
        },
        "available_compositions": available_compositions,
        "household_size": comp["household_size"],
        "survival_budget": survival_budget,
        # Aliases kept for backward compat with consumers reading
        # the old shape.
        "alice_threshold_4person_us": total_4person,
        "alice_threshold_for_size": total_for_size,
        "canonical_4person_budget": (
            canonical_budget if comp_key != "2A2K" else None
        ),
        "population_shares": population,
        "fpl": yr["fpl"],
        "reference_united_for_alice": {
            **ref,
            "delta_threshold_4person": (
                round(threshold_delta, 2) if threshold_delta is not None else None
            ),
            "delta_pct_alice": (
                round(pct_alice_delta, 4) if pct_alice_delta is not None else None
            ),
        },
        "data_cross_check": {
            "bls_food_basket": bls_food_check,
            "eia_gasoline": eia_gas_check,
        },
        "metadata": baseline.get("_metadata", {}),
        "interpretation": f"{budget_line} {population_line}",
    }
    return json.dumps(out, ensure_ascii=False)


# --- arg pre-extractor ------------------------------------------------
#
# Mirrors the pattern in _facts.py / _population.py / _inflation.py — the
# bridge merges this output into the LoRA's tool args before dispatch,
# so phrasings like "ALICE rate for a 3-person family in 2020" still
# end up with the right year + size even when the LoRA only emits
# `{}` (or stale defaults).

_ALICE_YEAR_RE = re.compile(r"\b(20[12]\d)\b")
# "3-person", "1 person", "two-person" (alpha numbers handled minimally)
_ALICE_SIZE_PEOPLE_RE = re.compile(r"\b(\d+)[-\s](?:person|people)\b", re.I)
_ALICE_SIZE_FAMILY_RE = re.compile(
    r"\b(?:family\s+of|household\s+of|for\s+a)\s+(\d+)\b", re.I,
)
# Numeric words → int, capped at 10 (anything bigger lands in 7+).
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_ALICE_SIZE_WORD_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten)[-\s]"
    r"(?:person|people|adult)\b",
    re.I,
)


def extract_alice_args(user_text: str | None) -> dict[str, Any]:
    """Pull `year`, `composition`, and `household_size` from the user's
    phrasing if present. Returns an empty dict on no match.

    Composition detection prefers explicit phrasings ("single parent",
    "single adult", "couple", "family of 4"). When only a number is
    found, falls back to `household_size` and lets `_resolve_composition`
    map it to a default composition.
    """
    out: dict[str, Any] = {}
    text = (user_text or "").strip()
    if not text:
        return out

    # Year — accept 2010-2029.
    m = _ALICE_YEAR_RE.search(text)
    if m:
        out["year"] = int(m.group(1))

    # Composition phrases — match before falling back to size.
    lower = text.lower()
    if re.search(r"\bsingle\s+(?:parent|mom|mother|dad|father)\b", lower):
        # "single parent" — default to 1 child unless a count comes up
        if re.search(r"\b(?:two|2|three|3|four|4)\s+(?:kid|child|son|daughter)", lower):
            out["composition"] = "1A0K"  # placeholder; size below corrects
        else:
            out["composition"] = "1A1K"
    elif re.search(r"\bsingle\s+adult\b|\b(?:childless|alone)\s+adult\b|\blone\s+adult\b", lower):
        out["composition"] = "1A0K"
    elif re.search(r"\bcouple\s+(?:with(?:out)?|no|with\s+no)\s+(?:kid|child|children)\b", lower):
        out["composition"] = "2A0K"
    elif re.search(r"\bcouple\s+(?:with|and)\s+(?:two|2)\s+(?:kid|child)", lower):
        out["composition"] = "2A2K"
    elif re.search(r"\bcouple\s+(?:with|and)\s+(?:one|1)\s+(?:kid|child)", lower):
        out["composition"] = "2A1K"
    elif re.search(r"\bcouple\s+(?:with|and)\s+(?:three|3)\s+(?:kid|child)", lower):
        out["composition"] = "2A3K"

    # Household size — numeric forms.
    if "composition" not in out:
        for pat in (_ALICE_SIZE_PEOPLE_RE, _ALICE_SIZE_FAMILY_RE):
            m = pat.search(text)
            if m:
                try:
                    n = int(m.group(1))
                    if 1 <= n <= 10:
                        out["household_size"] = n
                        break
                except ValueError:
                    continue
        if "household_size" not in out:
            m = _ALICE_SIZE_WORD_RE.search(text)
            if m:
                n = _NUM_WORDS.get(m.group(1).lower())
                if n:
                    out["household_size"] = n

    return out
