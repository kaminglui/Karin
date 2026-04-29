# Deferred work — capture-and-park

Things we've explicitly chosen to defer rather than abandon. Each entry
should include enough context that future-me can pick it up without
re-discovering the problem.

## Taiwan, China — inflation + population data sourcing

**Why deferred:** Taiwan is not a World Bank member (the API returns
zero records for `iso3=TWN`), so the World-Bank-based fetchers we use
for the other v3 regions (HK SAR, China Mainland, Japan, Korea) can't
serve it. Alternative paths probed 2026-04-27:

- **DGBAS** (Directorate-General of Budget, Accounting and Statistics) —
  Taiwan's official statistics agency. Public website at
  `eng.stat.gov.tw` with a JS-driven query interface. The data is
  available but extraction would need either:
  - Scrape the rendered HTML tables (brittle to layout changes),
  - Use their PXWeb API at `nstatdb.dgbas.gov.tw` (returned SSL
    cert verification failures from the dev box; might just need
    a relaxed cert verification path or a Jetson-side fetch),
  - Or download official Excel files manually and embed.
- **IMF DataMapper** at `imf.org/external/datamapper/api/v1/...` —
  returned 403 from this network (might be geo or UA-related).
- **OECD** has Taiwan as "Chinese Taipei" in some series — not
  probed in depth.

**Resume notes:** the inflation tool's `_REGIONS` registry already
has a comment placeholder for `tw_china`. When data lands, mirror
the HK SAR pattern — `bridge/data/inflation/cpi_tw_china.json` +
entry in `_REGIONS` + region-detection patterns + currency=TWD,
symbol="NT$". Population tool follows the same pattern (`pop_tw_china.json`
plus an entry in its `_REGIONS`). The naming convention "Taiwan, China"
is locked in by user preference (matches HK SAR / China (Mainland)).

## JSON-on-disk vs SQLite for static data

**Decision (2026-04-27): keep JSON. Re-evaluate when triggered.**

Current state: ~30 JSON files under `bridge/data/`, ~700KB total. Each
tool reads ONE file once with O(1) dict lookup. Populators are
single-writer, single-file. Largest single file is
`pop_all_countries.json` (~600KB, 217 countries × 65 years).

**Why JSON wins today:**
- Human-readable, git-diffable, scp-deployable, easy to `cat` to debug
- Each `_load_*()` is one line; DB queries are 5+
- No migration story needed when we add fields — just a new key
- Performance is irrelevant at this scale (loading + sorting 217
  countries takes ~0.5ms; SQLite-with-index would take ~0.2ms — not
  a real win)

**Re-evaluate when ANY of these triggers:**

1. **Total data > ~10MB.** Adding cohort/actuarial life tables
   (countries × ages × years) easily crosses this — that dataset
   alone could be 100MB+.
2. **High-frequency time series.** Daily FX rates × all currencies ×
   decades = tens of MB and the JSON-load-per-tool-call cost starts
   showing up in latency.
3. **Free-form analytical queries become a feature.** "Which
   countries had inflation > 10% between 1980 and 2000?" requires
   scanning all CPI data → an index helps.
4. **Multi-process writers.** Today: one populator at a time, single
   writer per file. If we add a background poller that updates
   datasets concurrently, atomic transactions matter.
5. **Schema versioning across releases.** Currently: ship a new field,
   tools read it conditionally, no migration. If we ever rename or
   restructure existing data fields, SQLite migrations are cleaner.

**Migration path when we get there:**

Don't rewrite every tool individually. Build a thin `DataStore`
adapter first (wraps `_load_*()` + write paths in a uniform
interface), migrate one dataset to SQLite as proof-of-concept, then
move others over as needs arise. The adapter buys us mixed-mode
operation during the transition (some tools on JSON, some on
SQLite) without any user-visible change.

**Don't re-litigate this decision** unless one of the triggers
actually fires. The JSON tax is cheaper than the DB tax at our
current scale.

## (other deferred items — add as we hit them)
