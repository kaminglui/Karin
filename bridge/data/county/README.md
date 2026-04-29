# County-level metrics

Per-county data files for the county-overlay analytical surface
(`/api/county` + `/ui/map` county granularity, planned phase B).

## File shape

Every metric file follows the same JSON envelope so
`bridge/tools/_county_metrics.py` can load any of them uniformly:

```json
{
  "_metadata": {
    "version": 1,
    "data_as_of": "YYYY-MM-DD",
    "metric": "short_key",
    "label": "Human-readable label",
    "unit": "USD/mo" | "per 100K" | "ratio" | "people" | ...,
    "source": "Source name + table/series ID",
    "vintage": "ACS 5-yr 2022" | "FBI UCR 2022" | ...
  },
  "by_county": {
    "<5-digit FIPS>": {
      "2018": <number>,
      "2020": <number>,
      "2022": <number>,
      "2024": <number>
    },
    ...
  }
}
```

FIPS codes are zero-padded 5-digit strings (state FIPS + county FIPS),
e.g. `"06037"` for Los Angeles County, CA. ACS, FBI UCR, and CDC
WONDER all use the same codes, so files compose cleanly.

## Files (planned)

| File | Source | Status |
|---|---|---|
| `county_names.json` | Census Bureau | shipped — `{fips: "Los Angeles County, CA"}` |
| `gini.json` | ACS B19083 | fetcher landed (`scripts/fetch_acs_county.py`); run with API key |
| `population.json` | ACS B01003 | fetcher landed |
| `rent.json` | ACS B25064 | fetcher landed |
| `income_brackets.json` | ACS B19001 (16 buckets) | fetcher landed |
| `mortality_all_cause.json` | CDC WONDER (XML POST API) | deferred — Socrata mirror not available; needs WONDER XML client |
| `mortality_overdose.json` | CDC NCHS — Drug Poisoning Mortality by County | fetcher landed (`scripts/fetch_cdc_mortality.py`); free Socrata API, optional `KARIN_CDC_APP_TOKEN` |
| `violent_crime_rate.json` | FBI Crime Data Explorer (UCR/NIBRS, agency-aggregated) | fetcher landed (`scripts/fetch_fbi_ucr.py`); requires `KARIN_FBI_API_KEY`. ~20-40 hr full US run |
| `property_crime_rate.json` | FBI Crime Data Explorer | fetcher landed (same script as violent crime) |
| `alice_pct.json` | derived from ACS + per-county HUD FMR | deferred — depends on ALICE refactor |

## How to populate

Each fetcher needs an API key registered to the user's profile:

- **Census ACS**: https://api.census.gov/data/key_signup.html (free)
- **FBI Crime Data Explorer**: https://api.usa.gov/crime/fbi/cde/
- **CDC WONDER**: programmatic API rate-limited; manual download fallback

Set keys in `.env` then:

```bash
# Census ACS — runs in ~30s, fetches all ~3,200 counties × 4 vintages
KARIN_CENSUS_API_KEY=... python scripts/fetch_acs_county.py

# CDC NCHS drug overdose mortality — Socrata; runs in ~10-30s
# (free, no key strictly needed; KARIN_CDC_APP_TOKEN raises rate limit)
python scripts/fetch_cdc_mortality.py

# FBI UCR violent + property crime — agency-aggregated; SLOW
# (~20-40 hours full US, run overnight or use --states/--year flags)
KARIN_FBI_API_KEY=... python scripts/fetch_fbi_ucr.py --year 2022
```

Output JSON files land in this directory and are read on-demand by
`bridge/tools/_county_metrics.py`. No restart needed for the bridge
to pick up new data — re-loaded per request.

## Honesty rules

The county-overlay panel (planned phase B) labels every analysis as
descriptive, not causal. Pearson r between any two metrics is shown
with a "correlation, not causation" note + auto-displays population
density and Gini as visible confounders alongside the comparison.
See `docs/county-overlay-plan.md` for the full plan.
