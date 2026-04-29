# County-level overlay analysis: crime × mortality × demographics × ALICE

**Status:** planned, awaiting user decision on starting phase
**Owner:** dev branch
**Last updated:** 2026-04-27

## Goal

Extend the existing `/ui/map` panel beyond rent into a multi-metric
county-level analytical surface. Layer crime rates, mortality,
demographics (income diversity / population), and ALICE share onto
the same geographic frame so the user can visually probe relationships
("are ALICE-heavy counties also high-crime?", "where does the death
rate decouple from poverty?", "what's the residual after controlling
for population density?").

The product is descriptive analysis with honest stats, not causal
claims. Confounders (segregation history, urbanicity, healthcare
access, age structure) are surfaced explicitly in the UI alongside
any correlation we compute.

## What we already have (dependencies)

| Piece | Status | Location |
|---|---|---|
| Choropleth infrastructure | shipped | `web/static/panels/map.js` (D3 + us-atlas TopoJSON, state-level) |
| ALICE estimator | shipped | `bridge/tools/_alice.py` (national, 6 years) |
| Statistical analyze tool | shipped | `bridge/tools/_analyze.py` (peak/trough/trend/percentile_rank/correlate) |
| Vendored D3 + topojson-client | shipped | `web/static/vendor/` |
| Data-fetch resilience helpers | shipped | `scripts/_fetch_helpers.py` |
| Cross-frame nav postMessage hook | shipped | `karin:focus-region` listener in `app.js` |

## Phase A — Data assembly (~3-4 days)

Goal: build a single canonical county-level table with all metrics.

### A.1 Fetchers

Three new scripts under `scripts/`:

1. **`fetch_cdc_mortality.py`** — CDC WONDER county-level
   age-adjusted death rates (all causes, drug overdose, cardiovascular).
   Requires no API key for the public data export. Manual download
   fallback from `https://wonder.cdc.gov/Deaths-by-Underlying-Cause.html`
   if the programmatic API is rate-limited.
2. **`fetch_fbi_ucr.py`** — FBI Crime Data Explorer API
   (`https://api.usa.gov/crime/fbi/cde/`). Per-agency violent crime +
   property crime; aggregate to county. Free API key required (already
   pattern-matches `KARIN_*_API_KEY`).
3. **`fetch_acs_county.py`** — Census ACS via existing API:
   - B19083 (Gini index)
   - B01003 (total population)
   - B19001 (income brackets, for county-level ALICE disaggregation)
   - B25064 (median gross rent)
   - B17001 (poverty status by age)

### A.2 Data store

`bridge/data/county/` directory:
- `mortality_by_county.json` — `{fips: {year: {all_cause: float, ...}}}`
- `crime_by_county.json` — same shape, violent + property
- `demographics_by_county.json` — gini, population, density
- `alice_by_county.json` — disaggregated ALICE share per county
  using ACS B19001 brackets per county against the household-size-
  weighted survival budget (already derived for the state-level model)
- `_metadata.json` — single source-of-truth for vintages + as-of dates

Total payload: ~3,200 counties × ~12 metrics × 3 vintages ≈ 3 MB JSON.
Fits on Jetson without strain (currently 600 MiB headroom; this is
0.5% of that).

### A.3 ALICE county disaggregation

Currently ALICE % is computed nationally. To get per-county estimates:
1. Use ACS county-level B19001 brackets (Phase A.1.3 fetch)
2. Apply the same household-size-weighted FPL + survival-budget logic
   from `_alice.py` per county
3. Survival budget components scale by per-county HUD FMR (already
   per-county in the HUD data)
4. Other components (food, healthcare, childcare) stay at the national
   level for v1 — they don't vary as much county-to-county as housing

This produces a county-level `pct_alice` we can put on the map.

### A.4 New tool: `_county_metrics.py`

Reads all the above caches and exposes:
- `county_metric(metric, year)` → `{fips: value}` for choropleth
- `county_compare(metric_a, metric_b, year)` → joined data + Pearson r
- `county_drill(fips)` → all metrics for a single county

### A.5 New endpoint: `/api/county`

```
GET /api/county?metric=mortality_all_cause&year=2022
  → {fips: value, stats, source, ...}

GET /api/county/compare?a=alice_pct&b=violent_crime&year=2022
  → {pairs: [{fips, a, b}], pearson_r, scatter, ...}

GET /api/county/drill?fips=06037   (LA county)
  → {fips, name, all metrics, time series, ...}
```

## Phase B — Visualization (~1-2 days)

Goal: extend `/ui/map` to support county-level choropleth, multiple
metrics, and a comparison view.

### B.1 County TopoJSON

Add `web/static/vendor/us-counties-10m.json` (~1.5 MB from us-atlas).
First-paint cost: ~1s on Jetson tailnet; mitigated by `cache: "force-
cache"` in the fetch (already used for states).

### B.2 Map panel updates

- Granularity selector: state (current) vs county
- Metric selector grows to ~12 options (rent, ALICE, population,
  Gini, mortality variants, crime variants)
- Color scale switches between sequential (rate metrics) and diverging
  (residuals after controlling for X — phase C territory)

### B.3 Comparison view

New "Compare" mode in the map panel:
- Two metric pickers (A and B)
- Side-by-side mini-choropleths OR an overlay with bivariate color
  scale (red-blue grid)
- Below the map: scatterplot of A vs B across counties
- Pearson r displayed prominently with confidence interval
- Click any county on either choropleth → drill view

### B.4 Drill view

Clicking a county opens a slide-in panel showing:
- All metrics for that county
- Where it sits on the national distribution (percentile per metric)
- Time series for available metrics (2018-2024)
- Comparable counties (similar population + urbanicity)

## Phase C — Honest stats (deferred until Phase B ships)

Adding scipy (~30 MB) unlocks:

- **Partial correlations** — "after controlling for population density,
  the ALICE-crime correlation drops from r=0.6 to r=0.3" tells the
  user something a Pearson r alone hides.
- **OLS regression** — `crime ~ alice_pct + population + gini +
  region_dummies`; coefficient + p-value per predictor.
- **Residual maps** — for any pair of metrics, show counties where
  the actual value is unexpectedly high/low given a simple regression
  prediction. These outliers are the interesting cases — naive
  correlation just rediscovers "urban areas have more of everything".

Scipy memory cost on Orin Nano: ~30 MB resident when loaded; only
loaded by the analyze worker. Fits within current 600 MiB headroom.

## Honesty rules (UI-level)

Every analysis panel must:

1. **Label the analysis "descriptive" or "exploratory" — never
   "causal".** The Pearson r card should literally say "correlation,
   not causation" in fine print.
2. **Show confounders alongside any pairwise comparison.** If the
   user picks "ALICE vs crime", auto-include population density and
   Gini in the breakdown so they see the obvious confounders.
3. **Explain the units.** Crime per 100K, deaths per 100K
   age-adjusted, ALICE as a household-share fraction.
4. **Source-cite every metric.** CDC WONDER vintage, FBI UCR vintage,
   ACS table — each surfaces in the drill panel.

## Open questions / decisions for the user

1. **Crime metric choice:** FBI UCR is the standard but has known
   reporting biases (urban under-report, rural over-report depending
   on the era). Alternative: NCVS victimization survey (state-level
   only, but cleaner methodology). Default plan: ship UCR and surface
   the caveat.
2. **Mortality metric choice:** all-cause death rate is the most
   defensible single number; "deaths of despair" (drugs + alcohol +
   suicide) is the more relevant subset for ALICE-relationship
   analysis. Default: ship both and let the user pick.
3. **Year alignment:** ACS 5-year (2018-2022, 2019-2023, etc.) vs
   ACS 1-year (less stable, more current). UCR + CDC WONDER are
   single-year. Default: align everything to 5-year ACS midpoints
   for stability; offer 1-year as a comparison toggle.
4. **County boundaries vs city boundaries:** county is the standard
   unit for federal data, but a county like LA contains huge
   internal variation. Default: ship counties; add tract-level if
   demand surfaces. Tract data is ~74K rows — the JSON would be
   ~50 MB, still fine for Jetson.

## Success criteria

Phase A done when:
- `/api/county?metric=alice_pct&year=2022` returns 3,200 entries
- `/api/county/compare?a=alice_pct&b=violent_crime&year=2022`
  returns Pearson r within ±0.1 of independently-computed value
- All four cached JSON files have current `_last_fetch_status.json`
  showing OK

Phase B done when:
- The county-level choropleth renders in <2s on Jetson tailnet
- Comparison view displays both metrics + scatterplot + Pearson r
- Drill click works on both choropleths

Phase C done when:
- Partial correlation displayed alongside raw Pearson r
- A residual map can be generated for any pair of metrics
- All UI surfaces carry the descriptive-not-causal label

## Effort summary

| Phase | Effort | New deps | New disk | Memory delta |
|---|---|---|---|---|
| A — data assembly | 3-4 days | none (numpy only) | ~3 MB JSON | negligible |
| B — visualization | 1-2 days | none | +1.5 MB TopoJSON | negligible |
| C — honest stats | ~2 days when needed | scipy ~30 MB | ~30 MB Python | ~30 MB resident |

Total Phase A+B: ~5 days. Phase C deferred until the comparison
view is in user hands and they confirm they want partial correlations.

## Resume instructions

If context compacts before this lands, check `MEMORY.md` for the
`project_county_overlay_plan.md` pointer. The first move is `Phase
A.1` — write `scripts/fetch_cdc_mortality.py` against the WONDER
public export. Subsequent moves follow the table above; nothing
in this plan blocks on the LLM model or training.
