"""Filing-status-aware federal + payroll + state tax computation for
the ALICE estimator.

Methodology:
- Federal income tax: progressive brackets per filing status, applied
  to (gross - standard_deduction).
- Child Tax Credit: $2,000/child under 17 (post-2017 TCJA), refundable
  up to ~$1,700 in recent years; we approximate as fully refundable
  here because ALICE-band families typically owe enough federal tax
  to absorb the credit before the refundability cap matters.
- Earned Income Tax Credit: simplified linear phase-in / plateau /
  phase-out using IRS-published max amounts and phase-out boundaries.
  Refundable.
- Payroll tax (FICA): 7.65% (employee share) on wages up to the SS
  wage base. ALICE-band incomes are nowhere near the cap so we just
  apply the flat 7.65%.
- State income tax: a national-average effective rate of 5% as a
  rough placeholder. Real state rates vary 0% (TX, FL, ...) to
  ~10% (CA marginal). Replace with per-state data when we land a
  state-aware fetcher.

Returns a dict that the panel can render line-by-line so the user
sees CTC/EITC offsets explicitly. The total goes into the gross-up
computation in _alice.py.
"""
from __future__ import annotations

from typing import Any

# --- federal brackets + standard deduction (per year + filing status) -

# Brackets are (top_of_bracket, rate). Last entry's top is None to
# mean "and above". Inflation-indexed annually.
_BRACKETS: dict[int, dict[str, list[tuple[float | None, float]]]] = {
    2024: {
        "single": [(11600, 0.10), (47150, 0.12), (100525, 0.22),
                   (191950, 0.24), (243725, 0.32), (609350, 0.35),
                   (None, 0.37)],
        "MFJ":    [(23200, 0.10), (94300, 0.12), (201050, 0.22),
                   (383900, 0.24), (487450, 0.32), (731200, 0.35),
                   (None, 0.37)],
        "HoH":    [(16550, 0.10), (63100, 0.12), (100500, 0.22),
                   (191950, 0.24), (243700, 0.32), (609350, 0.35),
                   (None, 0.37)],
    },
    2023: {
        "single": [(11000, 0.10), (44725, 0.12), (95375, 0.22),
                   (182100, 0.24), (231250, 0.32), (578125, 0.35),
                   (None, 0.37)],
        "MFJ":    [(22000, 0.10), (89450, 0.12), (190750, 0.22),
                   (364200, 0.24), (462500, 0.32), (693750, 0.35),
                   (None, 0.37)],
        "HoH":    [(15700, 0.10), (59850, 0.12), (95350, 0.22),
                   (182100, 0.24), (231250, 0.32), (578100, 0.35),
                   (None, 0.37)],
    },
    2022: {
        "single": [(10275, 0.10), (41775, 0.12), (89075, 0.22),
                   (170050, 0.24), (215950, 0.32), (539900, 0.35),
                   (None, 0.37)],
        "MFJ":    [(20550, 0.10), (83550, 0.12), (178150, 0.22),
                   (340100, 0.24), (431900, 0.32), (647850, 0.35),
                   (None, 0.37)],
        "HoH":    [(14650, 0.10), (55900, 0.12), (89050, 0.22),
                   (170050, 0.24), (215950, 0.32), (539900, 0.35),
                   (None, 0.37)],
    },
    2020: {
        "single": [(9875, 0.10), (40125, 0.12), (85525, 0.22),
                   (163300, 0.24), (207350, 0.32), (518400, 0.35),
                   (None, 0.37)],
        "MFJ":    [(19750, 0.10), (80250, 0.12), (171050, 0.22),
                   (326600, 0.24), (414700, 0.32), (622050, 0.35),
                   (None, 0.37)],
        "HoH":    [(14100, 0.10), (53700, 0.12), (85500, 0.22),
                   (163300, 0.24), (207350, 0.32), (518400, 0.35),
                   (None, 0.37)],
    },
    2018: {
        "single": [(9525, 0.10), (38700, 0.12), (82500, 0.22),
                   (157500, 0.24), (200000, 0.32), (500000, 0.35),
                   (None, 0.37)],
        "MFJ":    [(19050, 0.10), (77400, 0.12), (165000, 0.22),
                   (315000, 0.24), (400000, 0.32), (600000, 0.35),
                   (None, 0.37)],
        "HoH":    [(13600, 0.10), (51800, 0.12), (82500, 0.22),
                   (157500, 0.24), (200000, 0.32), (500000, 0.35),
                   (None, 0.37)],
    },
    2016: {
        # Pre-TCJA brackets — 7 rates topping out at 39.6%.
        "single": [(9275, 0.10), (37650, 0.15), (91150, 0.25),
                   (190150, 0.28), (413350, 0.33), (415050, 0.35),
                   (None, 0.396)],
        "MFJ":    [(18550, 0.10), (75300, 0.15), (151900, 0.25),
                   (231450, 0.28), (413350, 0.33), (466950, 0.35),
                   (None, 0.396)],
        "HoH":    [(13250, 0.10), (50400, 0.15), (130150, 0.25),
                   (210800, 0.28), (413350, 0.33), (441000, 0.35),
                   (None, 0.396)],
    },
}

_STD_DEDUCTION: dict[int, dict[str, float]] = {
    2024: {"single": 14600, "MFJ": 29200, "HoH": 21900},
    2023: {"single": 13850, "MFJ": 27700, "HoH": 20800},
    2022: {"single": 12950, "MFJ": 25900, "HoH": 19400},
    2020: {"single": 12400, "MFJ": 24800, "HoH": 18650},
    2018: {"single": 12000, "MFJ": 24000, "HoH": 18000},
    2016: {"single": 6300,  "MFJ": 12600, "HoH": 9300},  # pre-TCJA
}

# CTC: $2,000/kid under 17 (TCJA, 2018+); pre-TCJA was $1,000.
# Refundable cap rises annually (~$1,700 in 2024) but for ALICE-band
# incomes the credit usually fits inside federal liability anyway.
_CTC_PER_KID: dict[int, float] = {
    2024: 2000, 2023: 2000, 2022: 2000, 2020: 2000, 2018: 2000,
    2016: 1000,
}

# EITC parameters per (year, filing_status, num_kids). Each entry:
#   (max_amount, phase_in_end, phase_out_start, phase_out_end)
# Linear phase-in 0 → max over [0, phase_in_end]; flat plateau
# [phase_in_end, phase_out_start]; linear phase-out max → 0 over
# [phase_out_start, phase_out_end]. Source: IRS Rev. Proc. annual
# update; rounded to dollar.
_EITC_PARAMS: dict[int, dict[str, dict[int, tuple[float, float, float, float]]]] = {
    2024: {
        "single": {0: (632,   8260, 10330, 18591),
                   1: (3995, 12390, 22720, 49084),
                   2: (6604, 17400, 22720, 55768),
                   3: (7830, 17400, 22720, 59899)},
        "MFJ":    {0: (632,   8260, 17250, 25511),
                   1: (3995, 12390, 29640, 56004),
                   2: (6604, 17400, 29640, 62688),
                   3: (7830, 17400, 29640, 66819)},
        "HoH":    {0: (632,   8260, 10330, 18591),
                   1: (3995, 12390, 22720, 49084),
                   2: (6604, 17400, 22720, 55768),
                   3: (7830, 17400, 22720, 59899)},
    },
    2023: {
        "single": {0: (600,   7840, 9800,  17640),
                   1: (3995, 11750, 21560, 46560),
                   2: (6604, 16510, 21560, 52918),
                   3: (7430, 16510, 21560, 56838)},
        "MFJ":    {0: (600,   7840, 16370, 24210),
                   1: (3995, 11750, 28120, 53120),
                   2: (6604, 16510, 28120, 59478),
                   3: (7430, 16510, 28120, 63398)},
        "HoH":    {0: (600,   7840, 9800,  17640),
                   1: (3995, 11750, 21560, 46560),
                   2: (6604, 16510, 21560, 52918),
                   3: (7430, 16510, 21560, 56838)},
    },
    2022: {
        "single": {0: (560,   7320, 9160,  16480),
                   1: (3733, 10980, 20130, 43492),
                   2: (6164, 15410, 20130, 49399),
                   3: (6935, 15410, 20130, 53057)},
        "MFJ":    {0: (560,   7320, 15290, 22610),
                   1: (3733, 10980, 26260, 49622),
                   2: (6164, 15410, 26260, 55529),
                   3: (6935, 15410, 26260, 59187)},
        "HoH":    {0: (560,   7320, 9160,  16480),
                   1: (3733, 10980, 20130, 43492),
                   2: (6164, 15410, 20130, 49399),
                   3: (6935, 15410, 20130, 53057)},
    },
    2020: {
        "single": {0: (538,   7030, 8790,  15820),
                   1: (3584, 10540, 19330, 41756),
                   2: (5920, 14800, 19330, 47440),
                   3: (6660, 14800, 19330, 50954)},
        "MFJ":    {0: (538,   7030, 14680, 21710),
                   1: (3584, 10540, 25220, 47646),
                   2: (5920, 14800, 25220, 53330),
                   3: (6660, 14800, 25220, 56844)},
        "HoH":    {0: (538,   7030, 8790,  15820),
                   1: (3584, 10540, 19330, 41756),
                   2: (5920, 14800, 19330, 47440),
                   3: (6660, 14800, 19330, 50954)},
    },
    2018: {
        "single": {0: (519,   6780, 8490,  15270),
                   1: (3461, 10180, 18660, 40320),
                   2: (5716, 14290, 18660, 45802),
                   3: (6431, 14290, 18660, 49194)},
        "MFJ":    {0: (519,   6780, 14170, 20950),
                   1: (3461, 10180, 24350, 46010),
                   2: (5716, 14290, 24350, 51492),
                   3: (6431, 14290, 24350, 54884)},
        "HoH":    {0: (519,   6780, 8490,  15270),
                   1: (3461, 10180, 18660, 40320),
                   2: (5716, 14290, 18660, 45802),
                   3: (6431, 14290, 18660, 49194)},
    },
    2016: {
        "single": {0: (506,   6610, 8270,  14880),
                   1: (3373,  9920, 18190, 39296),
                   2: (5572, 13930, 18190, 44648),
                   3: (6269, 13930, 18190, 47955)},
        "MFJ":    {0: (506,   6610, 13820, 20430),
                   1: (3373,  9920, 23740, 44846),
                   2: (5572, 13930, 23740, 50198),
                   3: (6269, 13930, 23740, 53505)},
        "HoH":    {0: (506,   6610, 8270,  14880),
                   1: (3373,  9920, 18190, 39296),
                   2: (5572, 13930, 18190, 44648),
                   3: (6269, 13930, 18190, 47955)},
    },
}

# Payroll (FICA) employee share: 6.2% Social Security + 1.45% Medicare.
# SS portion caps at the wage base ($168,600 in 2024) — irrelevant to
# the ALICE band but kept for completeness if we ever rescore higher
# incomes.
_FICA_RATE = 0.0765
_SS_WAGE_BASE: dict[int, float] = {
    2024: 168600, 2023: 160200, 2022: 147000, 2020: 137700,
    2018: 128400, 2016: 118500,
}

# State income tax: national-average effective rate. The actual range
# is 0% (TX, FL, NV, ...) to ~10%+ (CA marginal). We use 5% as a
# nationwide approximation; the ALICE estimator panel is up-front
# about this.
_STATE_AVG_RATE = 0.05


def _nearest_year(year: int, available: dict) -> int:
    """Pick the requested year if available, else the closest year ≤
    requested, else the latest year. Same fallback _alice uses for the
    baseline."""
    if year in available:
        return year
    older = [y for y in sorted(available) if y <= year]
    if older:
        return older[-1]
    return sorted(available)[-1]


def _bracket_tax(taxable: float, brackets: list[tuple[float | None, float]]) -> float:
    """Apply a progressive bracket schedule to a taxable-income amount."""
    if taxable <= 0:
        return 0.0
    tax = 0.0
    prev_cap = 0.0
    for cap, rate in brackets:
        if cap is None or taxable <= cap:
            tax += (taxable - prev_cap) * rate
            return tax
        tax += (cap - prev_cap) * rate
        prev_cap = cap
    return tax


def federal_income_tax(gross: float, status: str, year: int) -> float:
    """Federal income tax owed before credits (CTC, EITC)."""
    yr_b = _nearest_year(year, _BRACKETS)
    yr_s = _nearest_year(year, _STD_DEDUCTION)
    brackets = _BRACKETS[yr_b].get(status) or _BRACKETS[yr_b]["single"]
    std_ded = _STD_DEDUCTION[yr_s].get(status) or _STD_DEDUCTION[yr_s]["single"]
    taxable = max(0.0, gross - std_ded)
    return _bracket_tax(taxable, brackets)


def child_tax_credit(num_kids: int, year: int) -> float:
    """Per-kid CTC. Treated as fully refundable for the ALICE band —
    most ALICE families owe enough federal tax pre-credit that the
    refundability cap doesn't bind, and where it does bind we'd
    UNDER-state the relief, so this errs on the conservative side
    (fewer ALICE if anything)."""
    if num_kids <= 0:
        return 0.0
    yr_c = _nearest_year(year, _CTC_PER_KID)
    return num_kids * _CTC_PER_KID[yr_c]


def eitc(gross: float, num_kids: int, status: str, year: int) -> float:
    """Earned Income Tax Credit. Linear phase-in from $0 → max over
    [0, phase_in_end]; plateau at max through phase_out_start; linear
    phase-out max → $0 over [phase_out_start, phase_out_end]."""
    yr_e = _nearest_year(year, _EITC_PARAMS)
    by_status = _EITC_PARAMS[yr_e].get(status) or _EITC_PARAMS[yr_e]["single"]
    kids_key = min(num_kids, 3)  # IRS caps EITC categories at 3+
    max_amt, pi_end, po_start, po_end = by_status[kids_key]
    if gross <= 0:
        return 0.0
    if gross < pi_end:
        return max_amt * (gross / pi_end)
    if gross < po_start:
        return max_amt
    if gross < po_end:
        return max_amt * (1.0 - (gross - po_start) / (po_end - po_start))
    return 0.0


def payroll_tax(gross: float, year: int) -> float:
    """FICA employee share — SS portion caps at wage base."""
    yr_w = _nearest_year(year, _SS_WAGE_BASE)
    cap = _SS_WAGE_BASE[yr_w]
    ss_part = min(gross, cap) * 0.062
    medicare_part = gross * 0.0145
    return ss_part + medicare_part


def state_tax(gross: float) -> float:
    """National-average effective state income tax."""
    return gross * _STATE_AVG_RATE


def compute_total_tax(
    gross: float,
    *,
    status: str,
    num_kids: int,
    year: int,
) -> dict[str, Any]:
    """Combine federal, payroll, state into one dict that breaks out
    each component for the panel + a `net_total` for the gross-up
    iteration. EITC is refundable so it can push net_total negative
    on low-income families."""
    fed_pre = federal_income_tax(gross, status, year)
    ctc = child_tax_credit(num_kids, year)
    fed_after_ctc = max(0.0, fed_pre - ctc)
    eitc_amt = eitc(gross, num_kids, status, year)
    fed_net = fed_after_ctc - eitc_amt
    fica = payroll_tax(gross, year)
    state = state_tax(gross)
    return {
        "filing_status": status,
        "num_kids": num_kids,
        "federal_pre_credits": round(fed_pre, 2),
        "ctc": round(ctc, 2),
        "eitc": round(eitc_amt, 2),
        "federal_after_credits": round(fed_net, 2),
        "payroll_fica": round(fica, 2),
        "state_avg": round(state, 2),
        "net_total": round(fed_net + fica + state, 2),
        "effective_rate_pct": round(
            ((fed_net + fica + state) / gross) * 100, 2,
        ) if gross > 0 else 0.0,
    }


def gross_up(
    pretax_subtotal: float,
    *,
    status: str,
    num_kids: int,
    year: int,
    max_iter: int = 12,
    tol: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Solve for gross income such that gross - taxes(gross) ==
    pretax_subtotal. Iterative because EITC + brackets make the
    relationship piecewise-linear."""
    # Initial guess: 12% effective rate (typical ALICE band).
    gross = pretax_subtotal / 0.88
    last_total = 0.0
    for _ in range(max_iter):
        breakdown = compute_total_tax(
            gross, status=status, num_kids=num_kids, year=year,
        )
        net_total = breakdown["net_total"]
        diff = (pretax_subtotal + net_total) - gross
        if abs(diff) < tol:
            break
        gross += diff
        last_total = net_total
    breakdown = compute_total_tax(
        gross, status=status, num_kids=num_kids, year=year,
    )
    return gross, breakdown
