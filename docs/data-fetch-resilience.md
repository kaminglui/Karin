# Data-fetch resilience (future)

Cross-cutting design notes for every script under `scripts/fetch_*.py`
that pulls data from an external JSON/CSV/XLSX API and writes a cache
file under `bridge/data/<tool>/`. Currently in-scope:

- `scripts/fetch_bls_cpi.py` — BLS CPI-U
- `scripts/fetch_bls_wages.py` — BLS AHETPI hourly wages
- `scripts/fetch_bls_items.py` — BLS Average Price Series (14 items)
- `scripts/fetch_nber_pre1913_cpi.py` — NBER M04051 cost-of-living
- `scripts/fetch_worldbank_cpi.py` — World Bank CPI for v3 regions
- `scripts/fetch_worldbank_population.py` — World Bank population
- (any future fetcher follows the same pattern)

**Status: NOT IMPLEMENTED YET — design notes only.**

## Problem 1: Schema drift adaptation

Today every fetcher hand-parses an upstream JSON shape. If an upstream
API changes its structure (key renamed, nesting rearranged, units
changed), today's parser raises `KeyError`/`TypeError` and either:

- The script crashes and the cache stays stale silently, OR
- The script writes empty/malformed data that breaks the tool downstream.

Three escalating mitigation tiers, pick by cost vs. flexibility:

### Tier A — schema validation (cheap)

Pin the expected JSON shape per fetcher and validate at fetch time.
On mismatch, abort with a clear "upstream schema changed at <key
path>" error pointing at the diff. Expected shapes can live as JSON
schema files under `scripts/_expected_shapes/` and be loaded next
to the parser.

Cost: minimal runtime, ~50 lines per fetcher. Doesn't fix anything
on its own — just turns silent breakage into loud, actionable errors.

### Tier B — forgiving extraction (medium)

Replace strict-path parsing with key-name-based dict walking and
regex-over-JSON-string fallbacks. Cosmetic reshapes (e.g. World
Bank wraps records in an extra envelope) don't break us. Examples:

- "find any key named `value` under any record that also has `year`"
  instead of `body[1][i]["value"]`
- regex `r'"value"\s*:\s*([\d.]+)'` as last-resort extraction

Cost: rewrite each parser. More tolerant, but if the new shape changes
SEMANTICS (units, baseline, definition), forgiving parsing silently
adopts the wrong values — still need Tier A on top.

### Tier C — LLM-assisted extraction (expensive but flexible)

When tier A/B fail, route the raw response to the local LLM with a
prompt like:

> "Extract `{year: value}` pairs from this JSON. Output only valid
> JSON: `{\"1985\": 312.4, \"1986\": ...}`. If you can't, output `{}`."

Handles unforeseen reshapes at runtime cost (~1-3s per fetch on the
Jetson). Use only as a fallback when A/B fail. The LLM is the same
karin-tuned model that handles user turns — no extra dependency.

Risks: LLM hallucination on numeric extraction is real. Sanity-check
the LLM's output against the expected magnitude/range from the cached
prior values before accepting (e.g. "all values must be within 50% of
last year's").

## Problem 2: User-visible fetch error surface

Today fetch errors only appear in `stderr` when someone runs the
script manually. Should add a per-dataset "last fetch status"
mechanism so the web UI can show a red indicator when a source is
stale or broken — including the error message so the user (or me)
can diagnose.

### Design sketch

Each fetcher writes `bridge/data/<tool>/_last_fetch_status.json`
alongside its data files:

```json
{
  "scripts/fetch_worldbank_cpi.py:hk_sar": {
    "ok": true,
    "ran_at": "2026-04-27T10:30:00",
    "fetched_years": [2024, 2025, 2026],
    "error": null
  },
  "scripts/fetch_worldbank_cpi.py:tw_china": {
    "ok": false,
    "ran_at": "2026-04-27T10:30:30",
    "error": "HTTP 404 — Taiwan not in World Bank dataset",
    "stale_since": "2026-04-27T10:30:30"
  }
}
```

Web UI: a `/api/diagnostics/data-sources` endpoint returns the merged
status across all `_last_fetch_status.json` files. A new Settings
toggle "Show data-source diagnostics" reveals a panel that lists each
source with green/red dot + last-fetched timestamp + error message
when failed.

Default: hidden (the panel is only useful when something breaks).
Enable via Settings when investigating "why is the inflation tool
saying the data is stale?".

## Suggested implementation order

1. Tier A (schema validation) for the two highest-impact fetchers
   (`fetch_bls_cpi.py`, `fetch_worldbank_cpi.py`) — covers the most
   user-visible breakage path.
2. The `_last_fetch_status.json` writer in every fetcher (uniform
   format; no UI yet).
3. The web UI diagnostics panel + Settings toggle.
4. Tier B (forgiving extraction) only after we see a real schema
   change in production that Tier A flags.
5. Tier C (LLM-assisted extraction) only if Tier B becomes impractical
   to maintain across 6+ different upstream APIs.
