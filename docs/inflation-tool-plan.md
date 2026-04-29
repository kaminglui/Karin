# `inflation` tool — implementation plan

Status: planning approved 2026-04-26. Build in progress.

## Goals

- Answer "how much is $X in year Y worth today" with verifiable, cited
  numbers — never LLM-fabricated.
- Bundle CPI-based AND wage-based equivalence in one call so the user sees
  both "what $1 buys" and "what $1 of wages bought" without needing
  separate prompts.
- Code does all math from cached authoritative data; LLM only picks the
  tool, extracts args, paraphrases the JSON return into prose. Existing
  market-fabrication scrub guards against LoRA citing wrong numbers.

## Non-goals (deferred)

- **Item-level prices** ("how much was bread in 1970"). Requires
  curated per-item series; partial coverage. v3 if user demand surfaces.
- **International / East Asia**. Each tracked as its own
  country/region entry with its own statistics office + currency:
  - **Hong Kong SAR, China** (HKD) — Census and Statistics Department (C&SD) CPI
  - **China (Mainland)** (CNY) — National Bureau of Statistics (NBS) CPI
  - **Taiwan, China** (TWD) — DGBAS
  - **Japan** (JPY) — Statistics Bureau of Japan
  - **South Korea** (KRW) — Statistics Korea (KOSIS)

  Mainland and HK are deliberately **separate** entries despite the
  political grouping — different currencies, different statistics
  agencies, different consumer baskets. Conflating them would be
  inaccurate. v3 — schema description in v1 says US-only.
- **"Production power"** in the sense of GDP/capita or productivity-
  adjusted wages — distinct concept; would need additional series. v3.

## Data sources

| Series | Source | Range | API | Use |
|---|---|---|---|---|
| **CPI-U All Items, Annual Average** (`CUUR0000SA0`) | BLS | 1913–present | `api.bls.gov/publicAPI/v2`, free, no key needed for ≤25 req/day | Primary inflation measure |
| **Historical CPI-equivalent** | Measuring Worth (static) | 1774–1912 | None — embedded constant table | Pre-BLS reconstruction; `confidence: low` |
| **Average Hourly Earnings, Production & Nonsupervisory** (`CES0500000003`) | BLS | 1964–present | Same BLS API | Primary wage measure |
| **Historical "Unskilled Wage" series** | Measuring Worth (static) | 1774–1963 | None — embedded constant table | Pre-1964 wage reconstruction; `confidence: low` |
| (cross-validation, optional) FRED `CPIAUCNS` and `CES0500000003` | Federal Reserve | 1947–present | `api.stlouisfed.org`, free with API key | Sanity-check that BLS cache matches; flag divergence > 0.1% |

**Caching:** ship `bridge/data/inflation/cpi_u_us.json` and
`bridge/data/inflation/wages_us.json` populated from BLS at build time.
Tool reads cache only — no live API calls in the request path. Background
poller refreshes monthly (16th of the month, when BLS publishes the prior
month). Pre-1913 / pre-1964 static tables ship with the repo.

## Tool API

```python
inflation(
    amount: float,           # required, positive
    from_year: int,          # required, 1774 ≤ from_year ≤ current
    to_year: int = None,     # default = current calendar year
    measure: str = "both",   # "cpi" | "wages" | "both" (default both)
)
```

## Return shape

```json
{
    "amount_input":     1.0,
    "from_year":        1970,
    "to_year":          2026,
    "country":          "US",
    "cpi": {
        "amount_output": 7.92,
        "ratio":         7.92,
        "source":        "BLS CPI-U All Items, Annual Average",
        "source_url":    "https://data.bls.gov/cgi-bin/surveymost?cu",
        "data_as_of":    "2026-03-31",
        "confidence":    "high"
    },
    "wages": {
        "amount_output": 11.45,
        "ratio":         11.45,
        "source":        "BLS Average Hourly Earnings, Production and Nonsupervisory Employees (CES0500000003)",
        "source_url":    "https://data.bls.gov/timeseries/CES0500000003",
        "data_as_of":    "2026-03-31",
        "confidence":    "high"
    },
    "interpretation": (
        "$1 in 1970 had the purchasing power of about $7.92 today (prices). "
        "An hour of typical wages in 1970 paid about $1; today it pays "
        "about $11.45 in nominal terms — wages outgrew prices by a factor "
        "of about 1.45×."
    ),
    "caveats": [
        "CPI measures the average urban consumer's basket — not your specific spending.",
        "Wage equivalence uses production / nonsupervisory workers' average; salaried and high-skill paths diverge.",
        "Long-range comparisons hide that what people buy in 2026 differs structurally from 1970."
    ]
}
```

`interpretation` and `caveats` are **server-generated strings**, not LLM
narration — the LLM is allowed to paraphrase them but the values are
deterministic from the data.

## Confidence levels

| Year range | Source | Confidence | Notes |
|---|---|---|---|
| 1947 → present | BLS direct, post-WW2 methodology stable | `high` | Current methodology since 1947 |
| 1913 → 1946 | BLS direct, but methodology shifts | `medium` | Pre-WW2 BLS series; cross-walked |
| 1774 → 1912 | Measuring Worth reconstruction | `low` | Best-effort historical reconstruction |
| < 1774 | — | n/a | Reject with explanation |

For wages specifically: 1964+ = `high`, 1900–1963 = `medium`, pre-1900 =
`low`.

## LLM boundary enforcement

| Allowed | NOT allowed |
|---|---|
| Pick `inflation` based on prompt | Make up CPI / wage ratios |
| Extract `amount`, `from_year` from natural language | Interpolate between years (tool does it) |
| Paraphrase `interpretation` and `caveats` into prose | Cite a number that doesn't match the tool's return |
| Suggest a `graph` follow-up | Estimate pre-1774 values |
| Pick `measure="cpi"` / `"wages"` / `"both"` based on question phrasing | Decide a confidence level |

**Enforcement extension to existing scrub** (in `bridge/llm.py`): the
`_MARKET_FABRICATION_PATTERNS` scrub already catches fabricated dollar
figures on chitchat. Extend a sibling check that fires when the
`inflation` tool ran AND the reply contains a dollar figure that
doesn't match `cpi.amount_output` or `wages.amount_output` (within
$0.01). Substitute persona fallback if mismatch.

## Routing patterns (in `_schemas.py`)

```python
"routing_patterns": [
    # "in 1970 money", "in 1965 dollars"
    r"\bin\s+\d{4}s?\s+(?:money|dollars?|prices?|terms|wages?)\b",
    # "how much was X in 1970"
    r"\bhow\s+much\s+(?:was|were|did|is)\s+.+\s+(?:in|back\s+in)\s+\d{4}s?\b",
    # "$X in YEAR worth today"
    r"\$\d[\d,.]*\s+(?:in|from)\s+\d{4}s?\b",
    # "inflation since YEAR" / "inflation from YEAR to YEAR"
    r"\binflation\s+(?:since|from|between)\s+\d{4}\b",
    # "purchasing power of"
    r"\bpurchasing\s+power\s+(?:of|in)\b",
    # "what would $X have been worth"
    r"\bwhat\s+would\s+\$?\d.+\s+(?:be|have\s+been)\s+worth\b",
]
```

## Graph integration (v1)

Tool's return JSON optionally includes a `series` block when the LLM
passes `include_series=True`:

```json
"series": [
    {"year": 1970, "cpi_index": 38.8, "wage_hourly": 1.05},
    {"year": 1971, "cpi_index": 40.5, "wage_hourly": 1.10},
    ...
]
```

The LLM can then chain to the existing `graph` tool with this series.
No new graph type needed — the existing line-plot widget renders
multi-series data already (per `bridge/tools/_graph.py`).

## Widget (v1)

```
┌─────────────────────────────────────────────────────┐
│ $1 in 1970 (US)                       Source: BLS  │
│                                                     │
│  Prices (CPI):  $7.92    confidence: high          │
│  Wages (avg):   $11.45   confidence: high          │
│                                                     │
│  $1 of 1970 wages would buy what $11.45 of today's │
│  wages buys — wages outgrew prices by 1.45×.       │
│                                                     │
│  CPI data as of 2026-03-31                         │
│  [more about CPI vs wages ▼]                       │
└─────────────────────────────────────────────────────┘
```

## Files

```
NEW:  bridge/tools/_inflation.py                ~150 LOC
NEW:  bridge/data/inflation/cpi_u_us.json       ~50 KB cached BLS series
NEW:  bridge/data/inflation/wages_us.json       ~25 KB cached BLS series
NEW:  bridge/data/inflation/historical_us.json  ~5 KB pre-1913 / pre-1964 static
NEW:  bridge/data/inflation/refresh.py          monthly poller — fetch BLS, write cache
NEW:  web/static/widgets/inflation.js           widget renderer
EDIT: bridge/tools/_schemas.py                  add inflation schema + routing_patterns
EDIT: bridge/tools/_dispatch.py                 wire inflation
EDIT: bridge/tests/test_tools.py                count: 16 → 17, plus inflation unit tests
EDIT: bridge/llm.py                             extend scrub to flag mismatched-figure replies
EDIT: bridge/pollers.py                         schedule cpi_poller monthly
EDIT: README.md, RUNBOOK.md, docs/tools.md      add inflation row, count update
```

## Phasing

**v1 (this session) — CPI only:**
- BLS CPI-U cached as static JSON in `bridge/data/inflation/cpi_u_us.json`
  (real annual averages 1913–2026 from BLS via usinflationcalculator.com,
  which republishes BLS data verbatim)
- Tool implementation with `measure="cpi"` (the `measure` arg accepts
  `"wages"` and `"both"` but those return `{"error": "wages dataset not
  yet loaded — see docs/inflation-tool-plan.md"}` until v1.5)
- Schema + routing + dispatch
- Tests
- Doc/count updates
- NO widget yet (text-only reply for v1)
- NO background poller yet (cached data, manual refresh)

**v1.5 (next session):**
- Fetch BLS Average Hourly Earnings (AHETPI) — multiple sources got 403
  during v1 build; need to use BLS API with a free key OR run a populator
  script on the Jetson which has internet access
- Wires the `wages` measure to actual data
- Background poller for monthly auto-refresh

**v2 (later):**
- Pre-1913 historical CPI from Measuring Worth
- Pre-1964 historical wages
- Cross-source FRED validation
- LLM boundary scrub extension
- Widget renderer

**v3 (much later):**
- East Asian countries/regions (Hong Kong SAR China, China Mainland, Taiwan China, Japan, Korea)
- Per-country/region sources via national/regional stat agencies
- Item-level prices (BLS Average Price Series for tracked items)
- Currency-of-the-era conversion

## Future enhancements (cross-cutting)

For data-fetch resilience and user-visible diagnostics that apply to
ALL `scripts/fetch_*.py` (not just inflation), see
[docs/data-fetch-resilience.md](data-fetch-resilience.md).

## Accuracy guardrails to ship in v1

1. Reject `from_year < 1913` in v1 (pre-1913 data isn't shipped yet)
   — return `{"error": "data only goes back to 1913 in this version"}`.
2. Cap `from_year` and `to_year` at current calendar year.
3. Reject `amount <= 0` or non-numeric.
4. `data_as_of` returned in every response so user knows data freshness.
5. `confidence` always present.
6. `source` and `source_url` always cited.
7. Widget shows the source attribution prominently — not buried in a
   tooltip.

## What this does NOT do

- Does not predict future inflation.
- Does not adjust for region within US (Manhattan vs rural Iowa CPI
  diverge — we use national CPI-U).
- Does not adjust for income bracket (top-decile spending differs
  from median).
- Does not use chained CPI (BLS C-CPI-U) — that's a research series.
- Does not handle sub-year granularity in v1 — annual averages only.
