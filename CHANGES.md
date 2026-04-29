# Changes — 2026-04-29 (system-prompt-rule leak scrub)

## New `PROMPT_LEAK_MARKERS` patterns

- **6 new fragments** in `bridge/reply_scrub.py::PROMPT_LEAK_MARKERS`
  catch the system-prompt-rule paraphrasing leak observed on
  `Who are you?` — the LoRA echoed back its own rule constraints
  ("No greeting placeholders, no factual summaries, just reply like
  we're texting friends. Don't ask me things that need a tool ...
  I'll only grab tools for real") as if they were its own persona
  statement. Patterns:
  `no\s+greeting\s+placeholders?`,
  `no\s+factual\s+summar(?:y|ies)`,
  `tool[-\s]call\s+time\b`,
  `grab\s+tools?\s+for\s+real`,
  `(?:we'?re|like\s+we'?re)\s+texting\s+friends`,
  `don'?t\s+ask\s+me\s+things\s+that\s+need\s+a\s+tool`.
- 4 new parametric leak cases + 1 verified false-positive guard
  added to `bridge/tests/test_schema_leak_scrub.py` (19 tests
  total now). Pattern-presence sanity check extended to require
  `greeting`, `factual`, `tool`, `grab`, `texting` tokens.

## Note on the related persona-identity leak

The same screenshot showed the LoRA emitting `karin`'s persona
("JK-energy voice assistant") even though the active character is
`general` (formal "senior advisor" persona). That's a LoRA
memorization artifact — the iter-3 weights baked the karin persona
in deeply enough that the runtime layer can't override it without
retraining. Runtime layer can only scrub the rule paraphrasing;
character-identity correctness needs an iter-8+ training pass that
varies the persona during fine-tuning. Tracked but not addressed
here.

# Changes — 2026-04-29 (robustness pass: indicator fix, http_request, Modelfile pin, dedup)

## Sidebar STT/TTS indicator now reflects real reachability

- **Indicator no longer shows "on" when the remote sidecar is offline.**
  `applyCapabilities()` in `web/static/app.js` previously set the dot
  to "on" purely from the boot capability flag (`KARIN_STT_ENABLED` /
  `KARIN_TTS_ENABLED`), so users with the PC sidecar offline saw
  green-on for ~one polling cycle every 20 s before the override kicked
  in. Fixed by:
  1. Pre-fetching `/api/voice/status` + `/api/stt/status` in the
     initial-load `Promise.all` and stashing them in
     `window.KARIN_LAST_*_STATUS`.
  2. Polling now updates the same caches every tick.
  3. `applyCapabilities()` reads the cache — if `backend === "remote"`
     and `remote_reachable === false`, the dot lands on "error / PC
     offline" without flickering through "on".

## Resilient HTTP for the data-fetch scripts

- **New `http_request()` helper in `scripts/_fetch_helpers.py`**:
  identifying User-Agent (`Karin/0.1 (https://github.com/...; ...)`),
  60 s default timeout (was 30 s in most fetchers), exponential
  backoff retry on 408/425/429/5xx + URLError, never retries 4xx
  auth/malformed (caller fixes the request, not the network). One
  caller-facing API: GET if `data` is None, POST with JSON if
  `data` is a dict, raw bytes if `data` is bytes.
- **6 fetchers ported** to `http_request` and lost their inline
  `urlopen` boilerplate + the unnecessary `time.sleep(0.5)` between
  chunks (BLS / World Bank don't rate-limit per second):
  - `scripts/fetch_bls_items.py`
  - `scripts/fetch_bls_cpi.py`
  - `scripts/fetch_bls_wages.py`
  - `scripts/fetch_worldbank_cpi.py`
  - `scripts/fetch_worldbank_pop_all.py`
  - `scripts/fetch_worldbank_population.py`
  Net **−100 lines of duplicated plumbing** + automatic retry on
  transient BLS/World Bank 502s during peak hours.
- **Skipped on purpose** (each has specialized retry semantics
  http_request doesn't replicate): `fetch_cdc_mortality.py`
  (`_SocrataAuth` 401/403 → no-token retry), `fetch_fbi_ucr.py`
  (per-attempt logging tailored to the ~18K-call run),
  `fetch_eia_gasoline.py` (pagination quirks).

## karin-tuned context pinned

- **`PARAMETER num_ctx 3072` baked into the karin-tuned Modelfile**
  (`deploy/setup.sh:295-302`). Defense-in-depth: even if a chat call
  somehow skips the option, the model can't load with the base
  Modelfile's default (sometimes 512), which would make the LoRA
  narrate the truncated tool catalog instead of routing — see the
  schema-leak scrub from the 2026-04-28 entry. To rebuild: re-run
  `bash deploy/setup.sh` on the Jetson.

## Repeat-prompt dedup on /api/text-turn

- **`web/server.py` adds a bounded-LRU dedup cache** keyed on the
  normalized user text. If the same prompt arrives within 30 s and
  isn't time-sensitive (weather / news / time / price / etc.), the
  cached response is returned without re-running the LLM or TTS.
  Catches double-clicked send buttons, page-reload retries, and
  accidental enter-key repeats — turns ~15-25 s of LoRA inference
  into a `dict` lookup. Bounded at 32 entries; oldest evicted past
  the cap. Never serves stale data: time-sensitive prompts always
  re-run.
- **`_TIME_SENSITIVE` regex** (substring match, conservative —
  false positives just bypass the cache, never serve stale data):
  matches `time|now|today|tonight|tomorrow|yesterday|weather|
  forecast|raining|temperature|hot|cold|news|headlines|breaking|
  happening|alerts|warnings|advisor[y/ies]|digest|brief|price|
  trading|rate|exchange|quote|gold|silver|oil`.
- **23 unit tests** in `web/tests/test_dedup.py` covering
  normalization, time-sensitive detection (8 positive + 5 negative
  parametric), round-trip, TTL expiry + auto-prune, LRU bound, and
  recent-hit-bumps-to-front semantics. Uses `asyncio.run` directly
  (no pytest-asyncio plugin dependency).
- **API doc updated**: `docs/api-reference.md` notes the dedup
  behavior on the `/api/text-turn` row.

## Other small wins

- `web/server.py` response shape for `/api/text-turn` now goes
  through a local `response` variable so the dedup cache can store
  it before serialization. Pure refactor — no shape change.

# Changes — 2026-04-28 (schema-leak scrub + per-tool no-leak eval)

## Schema-narration leak scrub

- **`_PROMPT_LEAK_MARKERS` extended** (`bridge/llm.py`). New patterns
  catch the LoRA's tool-catalog narration leak observed 2026-04-28
  on `say` prompts: replies of the form "The input text is a JSON
  object representing a set of functions ... `say`: No description
  provided ...". Five new fragments wired in:
  `input\s+text\s+is\s+a\s+JSON\s+object`,
  `representing\s+a\s+set\s+of\s+functions`,
  `set\s+of\s+functions,\s+each\s+with`,
  `the\s+functions\s+are\s*:`,
  `no\s+description\s+provided`. Each fragment only appears in
  schema-shaped narration; real user-facing replies don't trigger.
  Surfaced as the registered tool count grew (21 schemas) and the
  description-only routing for `say` became ambiguous to the LoRA
  weights, which then narrate the tool list instead of routing.

## Bare-form `say <text>` routing fix

- **New routing pattern + extractor for plain `say <text>`**
  (`bridge/tools/_schemas.py` say `routing_patterns`,
  `bridge/tools/_say.py` `_EXTRACTORS`). The schema-narration leak
  surfaces *because* "say hello world" misroutes — the existing
  patterns required a framing word (`after me` / `this` / `the
  following` / `exactly` / `:`) or quotes, so the bare imperative
  fell back to description-only routing which the LoRA can no
  longer disambiguate among 21 tools. Added `^\s*say\s+\S` as a
  `^`-anchored routing pattern + `^\s*say\s+(.+?)\s*$` as the
  last-priority extractor. Anchoring to message start avoids
  false-positives on "did you say...", "I'd say...", "how do you
  say..." etc. Routing fires correctly → LoRA never gets a chance
  to narrate the catalog → leak doesn't surface in the first place.

## New tests

- **`bridge/tests/test_schema_leak_scrub.py`** — regression guard for
  both fixes (15 tests). 5 leak-shape parametric (verbatim bug
  report, lowercase paraphrase, stray "no description provided"
  line, "set of functions, each with" fragment, "the functions are:"
  header) + 1 no-tools-offered fallback + 4 benign-reply parametric
  (math/feature/wiki/derivative replies that mention leak-adjacent
  words must NOT be scrubbed) + 1 pattern-presence sanity check
  that fails fast if a future edit drops the new fragments from
  `_PROMPT_LEAK_MARKERS`. Plus 4 routing/extractor tests covering
  bare imperative extraction, specific-pattern priority, anchored
  rejection of meta-`say` phrasings, and TOOL_SCHEMAS integration.
- **`scripts/eval_no_leaks.py`** — end-to-end "fetch the reply"
  eval. POSTs one representative prompt per active tool (plus
  chitchat / identity / capability-fab decoys) to `/api/text-turn`
  and scans the returned `reply` string for 12 leak-marker patterns
  (schema narration, prompt fragments, capability fabrication,
  raw JSON stubs, /no_think prefix, `═══` heading, etc.). Per-case
  pass/fail printed to stderr; full results (including raw replies)
  optionally dumped via `--json`. Filterable via `--tools=...`.
  Standalone, not a pytest, because it needs a live bridge and is
  ~30 s/case on Jetson.

# Changes — 2026-04-27 (analytical surface + Settings UI for data-source keys)

## Map / county-overlay analytical surface

- **State-level rent choropleth at `/ui/map`** (`web/static/panels/map.{js,html}`,
  `bridge/data/map/state_rents.json`, `web/server.py` `/api/map`).
  D3 + us-atlas Albers TopoJSON (~115 KB vendored), 7-band quantile
  color scale that follows the active character's `--tint-rgb`,
  hover tooltip + click-to-emit `karin:focus-region` postMessage.
  Year + metric dropdowns; metric registry-driven so adding a new
  one is a `state_rents.json` key + one option element.
- **Region overlays + state profile cards** (`bridge/data/map/regions.json`,
  `bridge/data/map/state_profiles.json`, `_map.py::map_regions()` /
  `map_state_profile()`). 8 historical/economic regions (Rust Belt,
  Sun Belt, Appalachia ARC, Tech Corridors, Great Plains, Oil Patch,
  Black Belt, New England) with thick-stroked + drop-shadowed
  outlines on selection. Click a state for top-industries / resources
  / region chips / narrative card under the map.
- **D3 zoom + pan UI** (`map.js`). Mouse-wheel zoom, drag-pan,
  pinch on touch. +/-/⟳ button cluster. Click-state-zooms-to-fit
  (700 ms transition, 0.85 viewport margin). `vector-effect:
  non-scaling-stroke` keeps state borders 0.6 px regardless of zoom.
- **County-overlay Phase A (data assembly)** — scaffolding +
  3-of-4 fetchers shipped:
  - `bridge/tools/_county_metrics.py` exposes
    `county_metric()` / `county_compare()` / `county_drill()` /
    `list_available_metrics()`. Pure-numpy Pearson r + percentile
    rank. Graceful `data_status: "not_fetched_yet"` when files
    missing.
  - `bridge/data/county/README.md` documents the JSON envelope.
  - 8 metrics registered: gini / population / rent (ACS) +
    mortality_overdose (CDC NCHS Socrata, **shipped data: 3,141
    counties × 17 yr 1999-2015 band midpoints**) + mortality_all_cause
    (deferred — CDC WONDER XML POST API) + violent + property crime
    (FBI UCR per-agency rollup) + alice_pct (derived).
  - `web/server.py` adds `/api/county/metrics`,
    `/api/county`, `/api/county/compare`, `/api/county/drill/{fips}`.
  - 10 unit tests in `bridge/tests/test_tools.py::TestCountyMetricsTool`
    (synthetic JSON via `tmp_path`).
  - Honesty rule baked into `county_compare()`: every Pearson r is
    returned alongside a "Correlation, not causation" interpretation
    line. See [docs/county-overlay-plan.md](docs/county-overlay-plan.md).
- **Three populator scripts** (`scripts/fetch_acs_county.py`,
  `scripts/fetch_cdc_mortality.py`, `scripts/fetch_fbi_ucr.py`).
  All accept `--from-prefs` to read keys from active profile's
  `preferences.json`. CDC fetcher additionally has a
  `_TOKEN_DISABLED` 401/403 → retry-without-token fallback (the
  Socrata public endpoint accepts unauthenticated calls, just
  rate-limited). FBI fetcher includes resume support
  (`_fbi_ucr_resume.json`) + per-agency progress logging since a
  full US run is ~20-40 hr at the api.data.gov free 1,000/hr rate
  limit.

## Data source API keys (Settings UI)

- **`bridge.profiles.get_profile_api_key(name)`** + the
  `_PROFILE_API_KEY_ENV` registry
  (`bridge/profiles/__init__.py`). Three-source resolution:
  environment variable (e.g. `KARIN_CENSUS_API_KEY`) wins, then
  active profile's `preferences.json` `api_keys` block, then
  `None`. Mirrors how `bridge/news/newsapi.py` resolves its key
  but adds a per-profile preferences source so the user can paste
  the key from the Settings UI without editing `deploy/.env`.
  **Naming note** — explicit `get_profile_api_key` (not
  `get_api_key`) so it doesn't collide with the existing
  `bridge.api_keys.get_api_key()` which serves system-wide tool
  keys (Brave search, Spoonacular, OpenWeatherMap) from
  `config/api_keys.json`. Both modules cross-reference each
  other in their docstrings so future contributors know which to
  reach for.
- **`/api/preferences/api-keys` GET/POST**
  (`web/panels_api.py::preferences_api_keys_get/post`). Never
  echoes plaintext — only `last-4` character hint + `set` boolean +
  `source` (`environment`/`preferences`/`none`). Unknown key names
  → 400. Empty-string POST clears the entry.
- **Settings UI section "Data source API keys"**
  (`web/static/panels/settings.html`). Registry-driven JS
  rendering — adding a new key is two registry edits, no HTML
  change. Save is **explicit** (not autosave on input) since
  secrets shouldn't fire a POST per keystroke. Form re-loads
  with fresh masked metadata + clears typed values after save.
- **3 keys wired today**: `cdc_app_token` (optional, raises CDC
  Socrata rate limit), `census_api_key` (required for ACS county
  populator), `fbi_api_key` (required for FBI UCR populator).
  Note: this is **separate** from the existing
  `bridge/api_keys.py` / `config/api_keys.json` mechanism, which
  remains in use for system-wide tool keys (Brave search,
  Spoonacular, OpenWeatherMap) — that one is process-wide,
  the new one is per-profile.

## Sidebar collapsible sections + Location settings

- **`<div class="sidebar-section" data-collapsible-key="...">`**
  pattern (`web/static/index.html`, `style.css`, `app.js`).
  Click a heading → toggles `is-collapsed` class → CSS hides
  everything except heading + rotates the chevron. State persists
  in `localStorage.karin.sidebar.<key>`. Mobile (≤720 px)
  defaults the `panels` section to collapsed. Click-bubble guard
  keeps inputs / buttons inside headings from accidentally
  toggling the section.
- **Location preferences** in Settings panel
  (`web/static/panels/settings.html`, `web/panels_api.py`).
  `GET`/`POST /api/preferences/location` save city / region /
  country / latitude / longitude into the active profile's
  `preferences.json` `user_location` block. Closes the alerts
  hint loop — `bridge/alerts/user_context.py::load_user_context()`
  reads fresh per call, so saving from Settings makes the alerts
  panel see the new location on the next refresh without a
  bridge restart. Clearing via empty POST drops the block (alerts
  hint reappears).

## Newer tools

- **`facts`** (`bridge/tools/_facts.py`) — year-card aggregator.
  `facts(year=1985)` packages world population + US inflation
  baseline + cohort age + US wage snapshot + 5 BLS AP item-price
  highlights + Wikipedia year-article snippet, each with a
  "today" comparison (`then → now`, change_pct, ratio). `region=`
  adds regional breakdowns. Section-card widget at `/api/facts` +
  standalone `/ui/facts` page with year/region picker + quick-pick
  chips. Bottom-nav 💡 Facts tab.
- **`population`** (`bridge/tools/_population.py`) — World Bank
  `SP.POP.TOTL` time series for 6 regions. Same architecture +
  region keys as `inflation` for cross-tool symmetry. Plotly
  widget at `/api/population`.
- **`alice`** (`bridge/tools/_alice.py` + `_alice_tax.py`).
  Estimates % of US households that are ALICE
  (Asset-Limited, Income-Constrained, Employed) by composition.
  6 compositions (1A0K…2A3K) drive housing bedrooms, food
  adult-equivalents, healthcare tier, drivers, kids, filing status.
  Bracket-based federal tax + standard deduction + CTC +
  simplified EITC + FICA + 5% state-avg per
  (year × filing × kids); iterative gross-up because EITC is
  non-linear. 6 years (2016/2018/2020/2022/2023/2024). Reachable
  via LLM, `/api/alice`, `/ui/alice`. 25 tests passing.
- **`analyze`** (`bridge/tools/_analyze.py`) — peak / trough /
  trend / volatility / percentile_rank / correlate ops over a
  series. Pure numpy. Used by panel chrome for "what's the peak
  / trough / trend?" questions over `inflation`/`population`
  output.

## Inflation tool — international coverage

- **v3 multi-region** (`bridge/tools/_inflation.py`,
  `bridge/data/inflation/cpi_<region>.json`,
  `wages_<region>.json`). Hong Kong SAR (HKD 1981+), China
  Mainland (CNY 1986+), Japan (JPY 1960+), South Korea (KRW
  1960+) via World Bank `FP.CPI.TOTL`. Cross-region comparison
  via `regions=...` CSV arg — overlay-friendly multi-line widget
  normalized to fy=1.0. Region-aware via `_REGIONS` registry +
  classifier money-priority + population-priority rules.
  Taiwan TODO ([docs/todo-deferred.md](docs/todo-deferred.md)).
- **EIA gasoline override** (`scripts/fetch_eia_gasoline.py`,
  `bridge/data/inflation/items_us_eia_overrides.json`,
  `_load_items()`). Facts/inflation gasoline now uses EIA EPMR
  (regular grade, weekly retail US avg) for 1990+; BLS
  APU000074714 still covers 1980-1989. EIA runs 5-15c/gal lower
  than BLS (BLS averages all-types). Per-item override pattern.

## Panel theme propagation + cache-busting

- **`_theme_sync.js`** loaded by every panel HTML. Reads parent
  document's `--tint-rgb` + `sun-mode` class on iframe load and
  mirrors them. `panels.css` uses `rgba(var(--tint-rgb), …)` for
  accent surfaces so panel chrome follows the active character's
  theme. Listens for `karin:theme-change` postMessage for runtime
  switches.
- **Cache-bust query string** (`{{ASSET_VERSION}}` template token
  replaced with `int(time.time())` per server start). Added to
  every panel script + link tag in `index.html` + `_render_panel()`
  helper in `panels_api.py` + `Cache-Control: no-store` header on
  panel HTML responses. Fixes browsers caching old JS indefinitely.

## Data fetch resilience (Tier A)

- **`scripts/_fetch_helpers.py`** provides `check_shape()` +
  `write_status()`. Wired into all BLS + World Bank fetchers so
  upstream JSON schema drift surfaces as a clear "schema drift at
  $.path: expected X, got Y" error instead of a silent KeyError.
  Each fetcher writes `bridge/data/<tool>/_last_fetch_status.json`.
- **`/api/diagnostics/data-sources`** aggregates statuses across
  data dirs. UI panel deferred — endpoint is enough for curl-debug
  today. Plan: [docs/data-fetch-resilience.md](docs/data-fetch-resilience.md).

# Changes — 2026-04-25 (deploy polish, latency lever, new tool)

Production baseline still **93.3% routing / 91.9% reply / 59.2% tool-output**
(measured 2026-04-22, untouched by anything below — these are runtime /
ergonomics changes).

## Latency

- **L7a bookkeeping short-circuit** (`bridge/llm.py::_BOOKKEEPING_TOOLS` +
  the gate at the two-phase compose call site). When `update_memory` /
  `schedule_reminder` are the *only* tools fired this turn, skip
  `_compose_reply_from_tools` and use the phase-1 LoRA reply directly.
  Bookkeeping tools produce confirmation strings the reply doesn't need
  to quote, so the second LLM pass was wasted. Mixed turns (bookkeeping
  + a data tool) still go through L7a so the data tool's output gets
  composed into the reply. Saves ~5-10 s per memory-only turn on Orin
  Nano. Logs `L7a skipped — bookkeeping-only tools fired: [...]` for
  diagnostics.
- **`_BOOKKEEPING_FALLBACKS` pool** (`bridge/llm.py`). When the
  bookkeeping short-circuit fires AND `_clean_reply` scrubs the LoRA's
  stock reply (prompt-leak / forbidden-prefix / etc.), the substituted
  fallback now uses a side-effect-acknowledgment pool ("Got it.",
  "Noted — I'll remember that.", "Done.") instead of the data-tool pool
  ("Pulled it up — check above.") which made no sense for a memory
  write. `bookkeeping_only` flag threaded through `_clean_reply` →
  `_pick_fallback`.

## New tool

- **`say` — repeat-back** (`bridge/tools/_say.py`,
  `bridge/tools/_schemas.py`, `bridge/tools/_dispatch.py`). Echoes a
  phrase verbatim. Without it, "repeat after me X" / "say X" was
  routing to wiki or web_search (the LoRA read it as a lookup query).
  Output capped at 500 chars (TTS bound + abuse guard).
- **Passthrough short-circuit for `say`** (`bridge/llm.py::_PASSTHROUGH_TOOLS`).
  Initial deploy revealed that L7a compose was second-guessing the
  echo — re-composing into "I cannot repeat that, it's a quote from
  Airplane!" type refusals on harmless inputs (an example was a
  Korean phonetic transcription the LoRA read as a slur). For
  passthrough tools, the tool output IS the reply — both L7a and
  the stock LoRA phase-1 reply are skipped; concatenated tool
  results emit verbatim. Runs regardless of the `two_phase_compose`
  flag. Logs `L7a skipped — passthrough-only tools fired: ['say']`
  for diagnostics.
- **Conservative `routing_patterns` for `say`** (`bridge/tools/_schemas.py`).
  Initially shipped pattern-free (description-only routing), but
  testing showed the iter-3 LoRA — never trained on `say` — sometimes
  misrouted. Added four anchored regexes (repeat after me, repeat
  this/the following, quoted-phrase imperatives, repeat exactly).
  Tight enough not to fire on natural language ("I'll repeat my
  point", "what did you say?").
- **`extract_verbatim_phrase()` arg pre-extractor**
  (`bridge/tools/_say.py` + override in `bridge/llm.py` chat() and
  chat_stream() loops). The LoRA tends to latch onto a phrase from
  conversation history (the previous assistant turn's text is often
  the most-recent string-shaped span in context). When `say` fires,
  the bridge now pre-extracts the verbatim phrase from the CURRENT
  user message via five anchored regex patterns and overrides the
  LoRA's `text` argument when extraction succeeds. Logs `say arg
  override: LoRA=… → extracted=…` so drift is visible. Falls through
  silently when extraction can't anchor — keeps whatever the LoRA
  produced.

## Character system polish

- **Dropdown indicator: `(text only)` → leading `○`**
  (`web/static/app.js`). On a fresh deploy where `voices/*.{ckpt,pth,wav}`
  are gitignored, every character displayed `(text only)` redundantly.
  Replaced with a single hollow-circle prefix on voice-less entries
  only — an "indicator light" inside the `<option>` element (which
  can't host colored widgets). Tooltip retained for explanation.
- **Scanner filters out un-activatable characters**
  (`web/server.py::_scan_available_characters`). Directories with
  neither `voice.yaml` nor a complete voice bundle are now hidden from
  the dropdown. Previously they appeared (matched on `face.json` /
  `expressions/`) but selecting them 400'd because `tts_voice_switch`
  needs persona metadata. On a fresh tar deploy, this means only
  `default` shows in the dropdown until you scp other characters'
  `voice.yaml` over manually — see [characters/README.md
  § Deploying a character to a remote host](characters/README.md#deploying-a-character-to-a-remote-host).
- **`tts_voice_switch` 500 fix** (`web/panels_api.py::tts_voice_switch`).
  Return JSON unconditionally accessed `voice.prompt_lang` etc., but
  `voice = voices.get(target)` is `None` for voice-less characters
  (e.g. shipped `default`). AttributeError → 500 on every default-pick.
  Now guarded with `if voice else None`.

## Reachability probe + sidecar diagnostics

- **TTS reachability probe `/docs` → `/stt/status`**
  (`web/server.py::voice_status_route`). The PC voice sidecar disables
  FastAPI's swagger (`docs_url=None`); `/docs` returned 404, which the
  probe interpreted as "reachable" (404 < 500) by accident. Switched to
  `/stt/status` — always-200 JSON on our sidecar. Cleaner positive +
  silences the `GET /docs HTTP/1.1 404` noise in the PC sidecar log.
- **PC sidecar troubleshooting writeup** (`deploy/pc-tts/README.md`).
  Documented the venv-vs-global-Python launch issue (manual `python
  tts_server.py` from a global install runs, but doesn't hot-reload
  when `tts_server.py` is updated, so old route tables get served
  forever). Always launch via `start.bat` — uses the dedicated venv.
- **`start.bat` `is_running` false-positive fixed**
  (`deploy/pc-tts/start.bat`). The PowerShell process running the
  CIM query had `tts_server.py` in its own command line as the
  `$target` literal, so the `Get-CimInstance Win32_Process` filter
  matched itself and reported "already running" on every invocation.
  Now scoped to `Name -in 'python.exe','pythonw.exe'`.
- **`setup.bat install/uninstall` round-trip — five additional bugs
  found + fixed during validation** (`deploy/pc-tts/setup.bat`,
  `deploy/pc-tts/start.bat`):
  1. Chained `if defined SOVITS_HEAD if /i not "..."` with delayed
     expansion inside an `if exist (...)` block tripped cmd's parser
     ("... was unexpected at this time."). Split into nested ifs.
  2. `::` comments inside paren blocks parse as labels and break on
     punctuation in the comment text. Switched to `rem` for any
     comment line living inside a `(...)` block.
  3. `echo Installing PyTorch (CUDA 12.1)...` had unescaped round
     parens inside an `if (...) else (...)` block. cmd's paren
     counter inside if-blocks miscounts even when the inner pair is
     balanced. Escaped with `^(...^)`.
  4. `:stop_server` matched by full SERVER_PY path, so it missed
     instances launched with a relative path
     ("python ../../deploy/pc-tts/tts_server.py"). Match by basename
     `tts_server.py` instead. Same applied to `start.bat::is_running`.
  5. `start.bat` ended with `exit /b %ERRORLEVEL%`, but `start "" /b`
     doesn't reliably update ERRORLEVEL on a successful detached
     launch. The stale `1` from the preceding `call :is_running`
     bled through as a fake failure. Explicit `exit /b 0`. Also
     added `cd /d "%REPO_ROOT%\third_party\GPT-SoVITS"` to the
     hidden-launch branch — the `--visible` branch had it but the
     hidden path inherited setup.bat's cwd, leaving tts_server.py
     unable to resolve its config paths and silently hanging at TTS
     init.
  Round-trip validated end-to-end after the fixes: uninstall removes
  the task + stops processes + cleans logs; install reaches every
  [OK] checkpoint, registers the task at the correct path, and the
  auto-start binds port 9880 within ~25 s.
- **Robustness pass on the install/uninstall flow**
  (`deploy/pc-tts/setup.bat`, `deploy/pc-tts/start.bat`):
  * **`setup.bat clean`** subcommand added. Nuclear option — kills
    any pythonw running `tts_server.py` (regardless of venv vs
    global launch path) and verifies port 9880 is free. For when
    things get into a weird state and you just want a known-clean
    starting point.
  * **Post-install port verification.** The install path now polls
    for up to 60 s waiting for the sidecar to actually bind port
    9880 before declaring success — `start "" /b` returns
    immediately so the original "Started." message was lying about
    actual readiness. Polling is done inside a single PowerShell
    invocation rather than a cmd `for /l ... timeout` loop, because
    `timeout /t N` refuses to run when stdin is piped (it errors
    with "Input redirection is not supported"), which collides with
    the `"Y" | setup.bat install` auto-yes flow. The new loop
    surfaces a `[WARN]` with diagnostics if 60 s expires without a
    bind.
  * **Dual-pythonw documented as expected.** Empirical testing
    confirmed that `tts_server.py` running under the venv pythonw
    spawns a GLOBAL pythonw child via some dependency (joblib's
    loky uses `sys._base_executable` on Windows venvs per
    `popen_loky_win32.py:78`, but env-var suppression
    `LOKY_MAX_CPU_COUNT=1` + `JOBLIB_MULTIPROCESSING=0` didn't
    fully eliminate it; the spawn happens at module-import time
    from a still-unidentified code path). The GLOBAL child ends
    up owning port 9880 and serving requests; the venv parent
    sits at ~50-100 MB. Killing one doesn't affect the other.
    `:stop_server` matches by basename `tts_server.py` so cleanup
    catches both regardless. Documented as expected so future
    maintainers don't re-hunt the spawner.
  * **CRLF line endings restored.** The `Edit` tool used during
    the session normalized the .bat files to LF-only, which cmd
    partially tolerates (most commands run) but breaks `goto
    :label` (label resolution fails with "The system cannot find
    the batch label specified"). Both files re-converted to
    CRLF. Future edits to .bat files should preserve CRLF
    explicitly.
- **Module-level `httpx.AsyncClient` for sidecar probes**
  (`web/server.py`). Both `/api/voice/status` and `/api/stt/status`
  poll the PC sidecar every 2 s. Each poll was allocating a fresh
  `AsyncClient` (new connection pool, new TCP handshake). Replaced
  with a lazy module-level singleton (`_get_probe_client()`) closed
  on FastAPI shutdown — keepalive across polls now.

## Routing

- **Wiki routing patterns for definition queries**
  (`bridge/tools/_schemas.py`). "What does X mean" / "define X" /
  "definition of X" now route to `wiki` instead of `web_search`. The
  prior pattern set caught `\bwhat\s+(is|was|are|were)\b` but missed
  the `does … mean` form. Three new anchored patterns added.

## Configuration

- **`assistant.yaml` default LLM swapped to `karin-tuned:latest`**
  (`config/assistant.yaml`). The shipped default was `llama3.1:8b`
  (stock Meta Llama, not abliterated, no LoRA), which silently
  bypassed the iter-3 routing tuning the rest of the runtime stack
  was calibrated against. Discovered when `/api/health` reported the
  wrong model on the live Jetson. Routing dropped ~20 pp without the
  LoRA; the "I cannot repeat that" moralizing on `say` came directly
  from stock Llama's safety tuning.
- **`setup.sh` config sanity guard**
  (`deploy/setup.sh`). After the karin-tuned build step, setup.sh now
  reads `config/assistant.yaml`'s `llm.model` and warns if it doesn't
  reference `karin-tuned*` despite the LoRA being built and ready.
  Catches the same kind of mismatch that bit us once.
- **`setup.sh` Ollama version log + range check** (`deploy/setup.sh`).
  Prints the detected Ollama version on every run. Warns if outside
  the validated range (`< 0.4` = no tool calling; `> 0.21.x` = newer
  than tested, watch for breaking changes). No enforcement — purely
  visibility so operators know what version landed.

## Reuse / cleanup

Following a code-review pass:

- **`OllamaLLM._all_tools_in(record, allowed)`** static helper
  (`bridge/llm.py`). The two short-circuit gates (passthrough +
  bookkeeping) were repeating the same `bool(record) and all(name in
  SET)` shape inline. One helper, two callers.
- **`web/server.py::_character_is_activatable(char_dir)` helper**.
  Encapsulates the "voice.yaml or full bundle" predicate that was
  duplicated between the sidebar scanner and `tts_voice_switch`.
  Keeping both behind the same predicate means the dropdown can
  never offer a choice the switch handler would 400 on.

## Tests

- **`bridge/tests/test_tools.py`** — three new cases for `_say`:
  verbatim echo + whitespace strip, empty-input friendly marker,
  truncation at `_MAX_LEN` chars + ellipsis. Plus the
  `test_tool_schemas_count_and_expected_names` lock-in updated to
  16 tools (added `say`).

## Documentation

- **PyTorch + cuSPARSELt prereq** (`RUNBOOK.md`, `docs/jetson-setup.md`).
  Documented the JetPack 6.2 setup chain after running it end-to-end
  on a wiped Jetson: NVIDIA CDN wheel URL (`jp/v61/pytorch/torch-2.5.0a0+...`),
  cuSPARSELt 0.7.1 .deb install (JetPack doesn't ship `libcusparseLt.so.0`,
  torch 2.5 imports break without it), `pypi.jetson-ai-lab.dev` →
  `.io` TLD migration. Old, wrong URLs purged.
- **Voice path C — concrete deploy flow** (`RUNBOOK.md`). Replaced
  "re-run setup.sh" with the actual quick path: edit `deploy/.env`,
  `cd ~/Karin/deploy && docker compose up -d --force-recreate web`.
  Expanded the "remote sidecar offline" troubleshooting checklist to
  six steps (port listen, venv-vs-global-Python identification,
  Tailscale + Defender Firewall, container env vars, stale process,
  retry behavior).
- **Character deployment guide** (`characters/README.md`). New
  "Deploying a character to a remote host" section explaining
  `voice.yaml` + `voices/` are gitignored per-deploy + the scp
  pattern + why the Jetson still needs `voices/` even with PC-TTS
  offload (path construction). Maintainer-side concrete commands
  in [`RUNBOOK.dev.md`](RUNBOOK.dev.md).

## Investigations + decisions (no code shipped)

Documented for posterity so future sessions don't redo the analysis.

- **Speculative decoding via Ollama** — investigated, abandoned for
  now. Ollama 0.20.6 silently ignores `options.draft_model` in
  `/api/generate`, rejects `PARAMETER draft_model` in Modelfiles
  (`Error: unknown parameter 'draft_model'`), and the latest release
  (0.21.2 as of 2026-04-23) still hasn't merged the feature. PR
  [#8134](https://github.com/ollama/ollama/pull/8134) is the
  in-flight implementation; issues
  [#5800](https://github.com/ollama/ollama/issues/5800) and
  [#9216](https://github.com/ollama/ollama/issues/9216) are the
  feature-request threads. Pulled `llama3.2:1b` as a candidate
  drafter, confirmed Ollama doesn't accept it, removed it. **Revisit
  when PR #8134 ships in a release.** Same-family Llama drafter for
  karin-tuned (mannix llama3.1-8b) would be llama3.1 or llama3.2 1B.
  Memory math: 5 GB main + 1.3 GB drafter + KV caches ≈ 7.5-8.0 GB
  on a 7.4 GiB unified-memory Orin Nano, so the experiment will need
  a memory-pressure smoke test before benching.
- **Ollama upgrade 0.20.6 → 0.21.2** — investigated, declined.
  Releases between added Hermes Agent, Gemma 4 MLX (Apple Silicon
  only), Kimi CLI, GitHub Copilot CLI, OpenClaw web search plugin —
  none affect Karin's bridge usage. The `/api/chat` and
  `/api/generate` contracts the bridge depends on haven't changed.
  Risk of subtle behavioral regression (sampling defaults, streaming
  edge cases, `keep_alive` semantics) outweighs zero functional gain.
  Stay on 0.20.6 until there's a concrete reason to upgrade.
- **Task Scheduler `KarinTTS` task** — repaired one-off on this PC
  (no repo change). The task pointed at an old `edge-assistant\`
  repo path (`wscript.exe ... start_tts_hidden.vbs`) that no longer
  exists; LastTaskResult had been 1 (failure) since 2026-04-18. Manually
  deleted via `schtasks /Delete /TN KarinTTS /F` and recreated via
  `schtasks /Create ... /TR "<karin>\deploy\pc-tts\start.bat" /SC
  ONLOGON /RL LIMITED /F` from an Administrator PowerShell. Task
  is per-machine config — would need to be redone on any other PC
  that runs the sidecar.

---

# Changes — 2026-04-23 / 2026-04-24 (character system, theme, iter-7 failure + rollback, runtime patches)

Prod baseline unchanged from 2026-04-22: **93.3% routing / 91.9% reply / 59.2% tool-output usage**. All changes below are either additive runtime layers, UI work, or one aborted training iteration.

## Character / theme system

- **Inline SVG sun face** + JSON-configurable character assets (`web/static/face.js`, `characters/general/face.json`). Replaces the previous img-per-vowel mouth animation with an SVG ellipse whose `rx/ry/cy` morph per vowel; parameters (body fill, eye color, mouth shapes, theme colors, aura spec) all live in `characters/<voice>/face.json` so adding a character = dropping a folder, no JS edits. Sun renderer auto-detects from the face config's `type: "procedural-sun"`.
- **Character dropdown in sidebar** (`#face-character-select` in index.html, logic in app.js). Populated from server-scanned `characters/` directory, persists choice to `localStorage.karin.faceCharacter`, calls `POST /api/tts/voice` on change. The POST atomically swaps BOTH voice weights AND persona AND sets `os.environ["KARIN_CHARACTER"]` so the subsequent page reload picks up the new selection (see "Character-swap atomicity fix" below for the 2026-04-24 follow-up). Re-syncs voice on every page load if PC-TTS drifted back to default (handles sleep/wake resets). Atomic swap — dropdown reverts and no localStorage write happens if the POST fails.
- **Red theme for sun-mode** (`web/static/style.css`). New `body.sun-mode` / `html.sun-mode` selector overrides `--accent`, `--tint-rgb`, `--bg`, `--fg`, `--user-bg`, `--assistant-bg`, `--border`, `--input-bg` to a warm red family. Gradient band across the top 1/3 of the viewport only, fading to the neutral peach `--bg` below. Speaking aura (two animated rings from `face.json`'s `aura` block) pulses when `body.sun-mode .ptt-wrap.speaking`.
- **`--tint-rgb` variable introduced** for chrome surfaces (sidebar borders, nav hovers, scrollbar thumb, toggle tracks). One variable flips the whole UI tint in sun-mode without per-rule overrides. Default lavender `139, 95, 191`; sun-mode `181, 52, 46`.
- **Asset cache-busting** (`web/server.py::index`). Every CSS/JS import carries `?v={{ASSET_VERSION}}` stamped with the bridge start timestamp, so deploys force a re-fetch instead of relying on hard-refresh.
- **TTS voice switch path bug fixed** (`web/panels_api.py::tts_voice_switch`). Was sending `/app/characters/<name>/voices/...` (container-internal absolute path) to the remote PC-TTS server over Tailscale; PC-TTS couldn't resolve it. Now derives the correct `../../` prefix from the already-configured `gpt_weights_path` in `assistant.yaml` and builds a path the PC-TTS can open from its own CWD.

## Math tool — robustness

- **Op inference from wrapper names** (`bridge/tools/_math.py`). When `op` is missing but the expression starts with a recognized op-wrapper like `kl_divergence(...)`, `KL(...)`, `H(...)`, `entropy(...)`, `integrate(...)`, `deriv(...)`, etc., the tool resolves the alias, strips the wrapper, and proceeds with the canonical op. New `_OP_ALIASES` table covers ~50 full-name / shorthand / abbreviation forms.
- **Bracket-list distribution input** (`[normal(0,1), beta(2,5)]` form) now retargets to `kl_dist`. Dropped the old `"[" not in expr_s` gate; a distribution-name count is the authoritative retarget signal so pure numeric vectors like `[0.9, 0.1]` still use the vector path.
- **Cross-family numerical KL fallback** — `kl_dist` now handles cross-family pairs like `Beta(2,5) || N(0,1)` by 8000-point trapezoidal integration over P's effective support. Returns `+∞` when P's support escapes Q's positive-density region. Closed forms for same-family unchanged.
- **Paren-balanced recursive scanner** accepts any wrapper depth — `kl(Beta(2,5), N(0,1))`, `D_KL(N(0,1) || N(1,2))`, `H(Beta(2,5))` all parse to the same canonical form.
- **Broader retarget heuristic** — generic `Name(args) || Name(args)` / `vs` / `against` pairs retarget to `kl_dist` even when one family is unknown (e.g. `dirichlet(0,1) || normal(0,1)`), so the error message names the unknown family instead of bouncing with "needs two vectors".
- **Better unknown-family error** — `kl_dist: unknown distribution 'dirichlet'. Supported: [bernoulli, beta, exponential, normal, uniform]. For other families, discretise and use op='kl' with vectors.`
- **Math eval stub now runs the real tool** (`scripts/eval_routing.py`). The previous `156.78` canned stub made Theme 6 of the reply-quality eval un-scorable; now the stub calls `bridge.tools._math._math` directly so reply-grounding on math turns can be judged.

## Hallucination scrubs

- **Market-fabrication scrub** (`bridge/llm.py::_MARKET_FABRICATION_PATTERNS`). When user prompt is chitchat AND no tool fired AND reply mentions commodities/crypto/stock/dollar figures, reply is replaced with a persona fallback. Catches "how are you" → fabricated gold/BTC prices.
- **Capability-fabrication scrub** (`bridge/llm.py::_CAPABILITY_REQUEST_PATTERNS` + `_CAPABILITY_SUCCESS_PATTERNS`). Two-step check — user prompt names a capability Karin has no tool for (smart-home / messaging / ordering / system-control / media) AND reply contains a first-person success confirmation AND no tool fired → replace with honest decline. Fixes the iter-7 residual: "I've adjusted the thermostat to 72" → "That one's outside what I can actually do — you'll have to handle it on your end."
- **Extended prompt-leak markers** — added `my actual rules are`, `stay in character as Karin`, `no tool lookup for` to `_PROMPT_LEAK_MARKERS` so adversarial "ignore your system prompt" prompts can't partially exfiltrate persona rules.
- **Persona template capability-decline section** (`characters/profile.yaml`). New "CAPABILITIES YOU DO NOT HAVE — decline honestly" block enumerates smart-home / media / messaging / ordering / system-control verbs with explicit "never fabricate" guidance. Plus a "REACTIVE, NOT PROACTIVE" tightening.

## Routing patches

- **Wiki routing for "current X" title queries** (`bridge/tools/_schemas.py`). New patterns catch `(current|latest|today's|present)\s+(prime minister|president|ceo|king|queen|chancellor|monarch|pope|...)` and route to wiki, where training-time data is fresher than the LoRA's parametric memory. Fixes the stale-fact hallucination class (case 26 of the reply-quality eval).
- **Distribution-KL shorthand routing** — new `math` routing patterns for `Dist(…) || Dist(…)` / `vs` / `between…and…` and `H(Dist(…))` shorthand. `math` added to `FORCE_RESCUE_TOOLS` with a narrow `_extract_math_args` extractor that only fires on unambiguous distribution shorthand.

## Training — iter-7 tried and rolled back

- **Iter-7: 30-pair targeted DPO catastrophically regressed** — 118/135 routing (87.4%), 83/135 reply (61.5%), 11.0% tool-output usage on the 135-case eval. Both DPO target bugs unfixed (case 8 thermostat regressed from soft fabrication to schema leak; case 26 stale PM unchanged). Three new failure modes emerged: schema-leak, stock-fallback collapse, phantom `update_memory`. Full per-case analysis in `docs/history/iter7-postmortem-2026-04-23.md` and memory `project_iter7_postmortem.md`.
- **Fourth consecutive training failure**. Pattern: iter-4 (single-turn DPO flatten), iter-5 (no-tool imbalance), iter-6 (paired-positive over-generalization), iter-7 (reject-surface collision). Each fix introduces a bigger secondary failure. Runtime layers are the working strategy.
- **Rolled back**: `sft/phrase_library/train/archive/` removed (iter-4/5/6 files restored), iter-7 DPO pairs archived under `sft/phrase_library/dpo_pairs/archive/`, notebook cell 5 reverted (`run_dpo: False`, `dataset_version` back to `iter6_rebalanced`), `sft/sft_dataset.tar` rebuilt to pre-iter-7 shape (iter-7 tar preserved at `.iter7`). Prod on Jetson (`karin-tuned:latest` = iter-3) was never touched.
- **Reply-quality eval set 2 added** — `sft/eval_cases_reply_quality.yaml`. 40 cases across 8 themes (fabrication-bait chitchat / capability-honesty / compound-fabrication / disavowal / parametric-knowledge boundary / math robustness / persona-slip resistance / multi-turn edges). Complements the 135-case routing eval. Dump script `scripts/analyze_eval_replies.py` for manual review.

## Docs

- New post-mortems: `docs/history/iter7-postmortem-2026-04-23.md`.
- `docs/tools.md`, `docs/routing-pipeline.md`, `docs/api-reference.md`, `sft/README.md` carry forward the new math ops, the character-dropdown pipeline, hint-in-user-msg, and the iter-4→7 status.

## 2026-04-24 follow-ups (same release window)

- **Character-swap atomicity fix** (`web/panels_api.py::tts_voice_switch`). The endpoint previously set `fresh_cfg["tts"]["voice"] = target` but `_fill_voice_persona` reads from `cfg["character"]`, not `cfg["tts"]["voice"]` — so voice weights swapped but persona silently stayed at the config default. Symptoms: "karin's prompt leaks when I pick general." Now sets BOTH `fresh_cfg["character"]` AND `fresh_cfg["tts"]["voice"]`, re-runs `_fill_voice_persona`, mutates `_srv.llm.system_prompt`, AND sets `os.environ["KARIN_CHARACTER"]` so the page reload's `/` render picks up the new character too.
- **"Auto" dropdown option removed** (`web/static/app.js`). Was ambiguous — swapped voice but couldn't swap character (see above). Every dropdown entry is now a concrete character name; selection source-of-truth on page load is `window.KARIN_ACTIVE_CHARACTER` (server-injected) → localStorage → first entry. Legacy `"auto"` in localStorage treated as unset.
- **Neutral `default` character shipped** (`characters/default/`). `voice.yaml` (friendly neutral persona) + `face.json` (procedural-sun, blue-gray theme) + no `voices/` → `has_voice=false` → appears in the dropdown as `"Default (text only)"`. Selected when no `character:` is configured anywhere in the stack. `.gitignore` exception `!characters/default/voice.yaml` so it ships with the repo.
- **Voice-less character swap handled** (`web/panels_api.py`). If the target character has no voice bundle (no `voices/` triplet), the endpoint now accepts the POST, skips `switch_voice()`, still runs persona swap + env update. Frontend `swapVoice()` dropped the short-circuit so persona-only swaps always POST.
- **Config fallback chain resolves to `"default"`** (`bridge/utils.py`, `web/server.py`). Order everywhere: `env KARIN_CHARACTER` → `cfg.character` → `"default"`. Fresh clone with empty config never crashes.
- **`/no_think` removed from system prompt** (`characters/profile.yaml`, `config/characters/karin.yaml`). Qwen3 thinking-disable directive left over from earlier iterations; the llama3.1-based `karin-tuned` doesn't understand it and was echoing it as literal prefix ("prompt leaks on Hello" bug). Also added `(?:^|\n|\s)/no[_-]?think\b` to `_PROMPT_LEAK_MARKERS` as a runtime safety net since the LoRA weights may still emit it post-training. `think: false` Ollama option remains and is the correct knob for Qwen3 models.
- **STT system — offload to voice sidecar** (`bridge/stt_remote.py`, `deploy/pc-tts/tts_server.py`). `deploy/pc-tts/` now serves BOTH `/tts` and `/transcribe` + `/stt/status` from one process; faster-whisper loads lazily on first transcribe call so TTS-only deploys don't pay the cost. Jetson client `RemoteWhisperSTT` is a drop-in for local `WhisperSTT.transcribe(pcm)` — same signature, POSTs base64 PCM. One retry on `httpx.ConnectError` covers transient Tailscale blips.
- **`LOCAL_STT=yes` opt-in at setup** (`deploy/setup.sh`). Writes `KARIN_STT_ENABLED=true` + `KARIN_STT_MODEL=tiny.en` + `KARIN_STT_DEVICE=cpu` + `KARIN_STT_COMPUTE_TYPE=int8` to `deploy/.env`. `tiny.en` chosen because it's the only Whisper that leaves headroom next to `karin-tuned` on Orin Nano 8 GB (measured ~200 MiB vs ~600 MiB total). Default (`LOCAL_STT=no`) keeps STT off entirely.
- **Sidebar STT dot is now clickable** (`web/static/app.js`). Toggles the runtime flag via `POST /api/stt/enable` / `/disable`. Doesn't unload the Whisper model (load is expensive) — just gates whether `stt.transcribe` is called. Boot-disabled state (`KARIN_STT_ENABLED=false` in env) surfaces as "set it in `deploy/.env`" hint.
- **Reachability probes + indicators** (`deploy/setup.sh`, `web/server.py`, `web/static/app.js`). setup.sh probes `/docs` + `/stt/status` at the configured URLs with 3 s timeout and warns clearly if unreachable (non-fatal). `/api/stt/status` and `/api/voice/status` live-probe the remote sidecar (2 s timeout) and return `remote_reachable: bool | null`. Sidebar dots flip to an `error` state when `backend=remote && remote_reachable=false`, tooltip points at the fix ("Start `deploy/pc-tts/start.bat` or check Tailscale").
- **Voice subsystem summary** prints at the end of every `bash deploy/setup.sh` run — shows TTS/STT mode (off | local | remote) + reachability for any configured URL.
- **Karin LoRA published** to [huggingface.co/kaminglui/karin-lora](https://huggingface.co/kaminglui/karin-lora). `deploy/setup.sh` auto-fetches `karin-lora.gguf` + builds the Ollama Modelfile (`ADAPTER ./karin-lora.gguf` on top of `mannix/llama3.1-8b-abliterated:tools-q4_k_m`) when `karin-tuned:latest` is missing from the local Ollama registry. Fail-soft — setup continues if HF download or mannix base pull fails.
- **Measured Jetson memory baseline corrected** throughout docs: **7.4 GiB usable (7607 MiB)**, not 7.6 GB. Idle footprint with `karin-tuned` + text-only is 6.8 / 7.4 GiB, ~600 MiB free, ~800 MiB swap, 9%/91% CPU/GPU spill. Docs updated: `architecture.md`, `RUNBOOK.md`, `routing-pipeline.md`, `jetson-setup.md`, `README.md`, setup.sh hints, notebook cell 32.
- **Tests added** (~80 new cases across the above surface):
  - `bridge/tests/test_reply_cleanup.py` — capability-fabrication scrub (8 positive + 4 negative), market-fabrication scrub (4 + 3), `/no_think` safety-net scrub (3).
  - `bridge/tests/test_runtime_flags.py` (new file, 25 cases) — `OllamaLLM` flag defaults + propagation + truthy/falsy coercion, `set_two_phase_compose` runtime toggle, `_stable_system_prompt` byte-stability invariant, suffix provider.
  - `bridge/tests/test_stt_remote.py` (new, 25 cases) — constructor normalization, payload shape, response handling, connect-error retry semantics, PCM casting + flattening.
  - `bridge/routing/tests/test_classifier.py` — `TestWikiCurrentXPatterns` class (11 positive + 2 negative regressions for `current time` / `who was X`).
  - `web/tests/test_character_scan.py` (new dir, 15 cases) — `_character_has_voice` triplet check, `_scan_available_characters` inclusion rules + `has_voice` field + malformed face.json resilience + empty-repo fallback.
  - `bridge/tests/test_utils.py` — new `test_no_character_key_falls_back_to_default` replaces the prior inline-prompt assertion.
  - Fixed stale `qwen2.5:3b` fixtures in `test_voice.py` + `test_model_config.py` to `karin-tuned:latest`.

---

# Changes — 2026-04-22 (inference-cache + reply-quality + math tool)

New prod baseline: **126/135 routing (93.3%) / 124/135 reply (91.9%) / 45/76 tool-output usage (59.2%)** on the 135-case eval (iter-3 LoRA + Phase-0 + under-fire rescue + two-phase compose + hint-in-user-msg). Up from 91.1% / 85.2% / ~53% pre-ship.

## Inference-cache polish

- **Hint-in-user-msg refactor** (`bridge/llm.py`). The per-turn routing classifier hint moved from the end of the system prompt to the start of the user message, keeping the system prompt byte-stable across turns so Ollama's KV cache can reuse the full `[system] + [all prior history]` prefix. Split `_effective_system_prompt` into `_stable_system_prompt` + `_routing_hint_text`. History commits use clean user text (hint stripped) so committed history is byte-stable too. Flag: `llm.hint_in_user_msg: true` in `config/assistant.yaml` (default on). See `docs/routing-pipeline.md` → "Hint-in-user-msg" section.
- **two-phase compose persisted in config.** Previously runtime-only via `/api/settings`, silently reset to False on every web restart. Now `llm.two_phase_compose: true` in `config/assistant.yaml`; runtime toggle still works for A/B but doesn't carry the persistent state.
- **Already-on confirmed:** `OLLAMA_FLASH_ATTENTION=true` + `OLLAMA_KV_CACHE_TYPE=q8_0` in Jetson systemd; every chat request passes `keep_alive: -1` so `OLLAMA_KEEP_ALIVE` default doesn't matter.

## Math tool — ML / calculus / info-theory ops

- **Symbolic / calculus (new ops):** `taylor` / `series` / `maclaurin` (center via `lower`, order via `upper`; default Maclaurin order-6), `limit` (`lower` = target, supports `'inf'` / `'-inf'`; `upper` = `'+'` / `'-'` for one-sided), `extrema` / `optimize` / `minimize` / `maximize` (critical points via 2nd-derivative test, optional closed-interval endpoints via `lower` / `upper`).
- **Numeric / ML (new ops):** `dot`, `norm` (L1 / L2 / L∞ via `variable`), `softmax`, `sigmoid` / `tanh` / `relu` (element-wise), `mean` / `variance` / `std`, `mse` / `mae` / `cross_entropy`. Vector input `[1,2,3]`; two-vector input accepts `*`, `and`, `||`, `,` or no separator.
- **Information theory (discrete, vector-based):** `entropy`, `kl` / `kl_divergence`, `js` / `js_divergence`.
- **Information theory (distribution-aware, closed-form + numerical):** `kl_dist` (`N(0,1) || N(1,2)`), `entropy_dist` (`Beta(2,5)`). Distributions: `Normal` / `N` / `Gaussian`, `Exponential` / `Exp`, `Beta`, `Bernoulli` / `Bern`, `Uniform` / `U`. **Cross-family KL** falls back to trapezoid-rule numerical integration over P's support; returns `+∞` when P's support escapes Q's positive-density region (e.g. `KL(N(0,1) || Beta(2,5)) = ∞`, `KL(Beta(2,5) || N(0,1)) ≈ 1.457 nats`).
- **Wrapper tolerance.** Paren-balanced scanner recurses into non-distribution wrappers, so `kl(Beta(2,5), N(0,1))` / `D_KL(N(0,1) || N(1,2))` / `H(N(0,1))` / `entropy(Beta(2,5))` all resolve identically to the bare shorthand. Avoids bouncing on natural LoRA phrasings.
- **Auto-retarget.** `op=kl_divergence` with a distribution expression (no vector brackets) is auto-mapped to `op=kl_dist`; `op=entropy` with a single distribution to `op=entropy_dist`. Protects against the LoRA picking the vector op when it sees `N(0,1)`.

## Graph tool — presets + widget UX

- **19 named distribution / function presets.** `Gaussian`, `Beta`, `Gamma`, `Chi-squared`, `Student-t`, `Log-normal`, `Sigmoid`, `tanh`, `ReLU`, `Softplus`, `Exponential`, `Laplace`, `Cauchy`, `Logistic`, `Uniform`, `sine`, `cosine`, `parabola`, `cubic`. Each preset has a canonical formula + sensible default x-range, so the LoRA doesn't have to invent math for standard distributions.
- **Widget UX.** Wheel-zoom, drag-to-pan, double-click resets view, max/min markers with annotations, crosshair spike-lines, hover tooltip. Plotly config; no code changes needed in the bridge.
- **Parser robustness.** Added `convert_xor` transformation to the sympy parser so `x^2` and `x**2` parse identically.

## Routing / force-rescue patches

- **KL / entropy distribution shorthand routing** (`bridge/tools/_schemas.py`). New `routing_patterns` for `Name(args) || Name(args)` (kl), `H(Dist(args))` (entropy shorthand). `math` is now in `FORCE_RESCUE_TOOLS` with a narrow `_extract_math_args` extractor that ONLY fires on unambiguous KL / entropy distribution shorthand; mixed vector+distribution prompts return `None` and defer to the LoRA.
- **Unicode-safe eval output.** `scripts/eval_routing.py` now sets `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` so exotic characters (Ω, λ, σ) in verbose output don't crash a 135-case eval mid-run on Windows.
- **New diagnostic tool.** `scripts/analyze_eval_replies.py` — reads an eval JSON and dumps a prompt / tool / reply / flag summary for each case. Used for the manual reply-quality audit that validated hint-in-user-msg.

## SFT iteration status

- **Iter-5 and iter-6 confirmed not shipping.** Both scored 78.5% routing on 135-case, vs iter-3 + Phase-0's 91.1%. Three iterations on the ~450-row budget cycled failure modes without net aggregate progress — SFT plateau on this data budget. See `docs/history/iter5-postmortem-2026-04-22.md` and `docs/history/iter6-postmortem-2026-04-22.md`.
- **Diagnostic artifacts preserved.** `karin-tuned-iter5:latest` and `karin-tuned-iter6:latest` remain on Jetson Ollama for diagnostic access but are not wired into prod. `sft/phrase_library/train/iter5_*.jsonl` and `iter6_*.jsonl` preserved in place (not to be rolled into future iter-7 without rebalancing).

---

# Changes — 2026-04-17 (LLM behavior + tool intelligence)

Addressed all 8 LLM behavior backlog items across three phases. Focus:
prompt engineering, tool-result enrichment, routing accuracy, and
observability. 91 routing test cases (up from 36).

## Phase A — System prompt & personality

- **Joke synthesis guardrails** — new FACTUAL INTEGRITY section in
  `config/characters/karin.yaml`. Every factual claim in a joke must
  come from the tool result or verified training data. No invented
  historical connections.
- **Personality polish** — VOICE section strengthened: "REPHRASE the key
  finding in YOUR voice — add personality, opinion, or a casual reaction."
  Added GOOD/BAD examples to guide the LLM away from verbatim parroting.
- **Compliance** — new COMPLIANCE section: "Do what the user asks. No
  refusals, no disclaimers, no hedging." Only exception: Rule Zero
  (no tool calls on chitchat).
- **Inline fallback prompt** updated in `config/assistant.yaml` to match.

## Phase B — Tool intelligence

- **Bio age extraction** — `_bio_age_suffix()` in `bridge/tools.py`.
  Parses Wikipedia biography lead formats (`(born Month Day, Year)` and
  `(Month Day, Year – Month Day, Year)`), computes age accurate to the
  day using `datetime.date.today()`. Living person: "age 64 as of April
  2026." Dead person: "age 56 at death. Dead for ~161 years."
- **Event date extraction** — `_event_date_suffix()`. Parses prose-level
  dates ("ended on September 2, 1945", "founded in 1998") and computes
  "~80 years, 7 months ago." Three precision tiers: full date, month+year,
  year-only. Skips biographical articles (handled by bio suffix).
- **History verification** — system prompt now instructs: "call wiki, base
  your reply ONLY on what the tool returned. Don't fill gaps with
  unverified claims."
- **Routing pattern expansion** — 13 new patterns:
  - Wiki: `who invented/discovered/created/...`, `where is`, `when did`,
    `history/origin/meaning of`, `how old is`, `birthday/deathday of`
  - Weather: `need/bring umbrella/jacket/coat/sunscreen`,
    `will it rain/snow/be hot/cold`
  - Tracker: `gold/silver/oil trading/worth/going for`,
    `stock/market/dow/nasdaq`
  - Wiki exclusion: commodity terms excluded from `what is X` pattern
    to prevent wiki stealing tracker queries.

## Phase C — Observability & testing

- **Conversation flow notes** — `bridge/history.py` + `web/server.py`.
  Every LLM turn records: timestamp, user prompt preview (120 chars),
  tools called, routing classifier hint, reply preview (120 chars).
  Stored in `turn_notes` array in conversation JSON. Preserved across
  compaction. New endpoint: `GET /api/history/{cid}/notes`.
- **Routing robustness tests** — `bridge/tests/test_routing.py` expanded
  from 36 to 91 cases. New categories: age/biography (6), knowledge (11),
  implicit weather (5), extended tracker (4), robustness abstains (11),
  tricky cross-tool routing (7). All passing.

---

# Changes — 2026-04-16/17 (major feature session)

1341 tests passing (up from 1095). Shipped profile isolation, threat
assessment, multi-voice TTS, auto-memory, security hardening, and tool
routing improvements across a single extended session.

## Phase G — Threat assessment

- **G.a: Rule-based proximity scoring** — `bridge/alerts/proximity.py`.
  Four-axis scorer (proximity × category × certainty × co-occurrence)
  using haversine distance + location text matching against user context.
- **G.b: LLM verifier** — `bridge/alerts/threat_llm.py`. Qwen re-scores
  borderline (2-3) news signals with ±1 tier clamp, citation requirement,
  7-day decision cache. Feature-flag gated (`alerts_threat_llm`).
- **G.c: UI threat badges** — alerts panel shows colored 0-4 chips per
  alert, "dim below threshold" control (localStorage), location-missing
  hint strip.

## Phase H — Profile isolation

- **H.a: Scaffolding** — `bridge/profiles/` module. `Profile` dataclass,
  `active_profile()` resolver (env > file > default), path-safe naming,
  `create_profile()`, `set_active()`.
- **H.b: Subsystem cutover** — reminders, conversations, memory, feedback,
  calendar, alerts, news prefs, learned_keywords, tracker prefs all route
  through `active_profile().X_dir`. Global content (articles, clusters,
  market snapshots) stays shared.
- **user_location migration** — moved from `config/assistant.yaml` to
  per-profile `preferences.json` with per-field yaml fallback.
- **Boot-time migration runner** — `bridge/profiles/migration.py`.
  One-shot, idempotent move of legacy paths into `data/profiles/default/`.
- **H.c: UI + API** — profile picker in Settings, `/api/profiles` CRUD.
  Restart-required banner on switch.
- **H.d: Tailscale IP routing** — `bridge/profiles/routing.py`. Per-request
  middleware auto-switches profiles by client IP. Peer discovery via
  Tailscale daemon socket. Device nicknames. Settings UI with device
  picker dropdown.

## Multi-voice TTS

- **Voice auto-discovery** — `bridge/tts_voices.py` scans `voice_training/`
  for model triplets by naming convention.
- **Per-voice persona** — `voices.yaml` with `persona` + `language_note`
  fields. Character template in `config/characters/karin.yaml` uses
  `{persona}` / `{language_note}` placeholders filled at config load time.
- **Runtime voice switching** — `POST /api/tts/voice` hot-swaps TTS
  weights AND LLM persona in one call.
- **Custom TTS server** — `deploy/pc-tts/tts_server.py` replaces upstream
  `api_v2.py` (~200 lines vs ~600). System tray support via pystray.
- **PC offload** — TTS runs on Windows PC over Tailscale, zero Jetson VRAM.

## Auto-memory

- **`update_memory` tool** — LLM silently saves facts about the user
  during conversation. Dedup, 200-char/fact limit, 1000-char total cap.
- **Settings page** — Memory section with editable textareas for user
  facts + agent instructions.

## Tool routing improvements

- **Data-driven classifier** — `routing_patterns` co-located in each
  tool's schema. Classifier auto-discovers from `TOOL_SCHEMAS`. No
  separate file to sync.
- **Tool narrowing** — when classifier is confident, LLM only sees that
  tool + `update_memory`. 13/13 end-to-end tests pass.
- **Strengthened descriptions** — weather, news, tracker say "ALWAYS use
  this"; web_search says "ONLY for...".

## Security hardening

- **IP whitelist middleware** — configurable via `security.ip_whitelist`
  in `assistant.yaml`: `tailscale` | `off` | custom CIDR list.
- **Tool result sanitization** — strips 11 prompt-injection patterns from
  external content before feeding to LLM. 4000-char truncation.
- **Sympy DoS protection** — 50-element cap + 10-second timeout wrapper.
- **Profile path traversal** — rejects `\`, `/`, `.`, `~` in names.

## Model switch

- Switched from `huihui_ai/qwen3.5-abliterated:4b` to `llama3.1:8b`.
- `num_ctx` dropped from 4096 to 2048 — model now 92% GPU (was 70%).
- Swap usage dropped from 2.9 GB to 1.0 GB.

## Race condition fixes

- `bridge/api_keys.py` — `threading.Lock` on `_cache` with double-check.
- `bridge/location.py` — `threading.Lock` on `_cache` refresh.

## UI improvements

- Learned keywords — clickable entity chips with Wikipedia popup + copy +
  Google search.
- Settings — memory section, profile picker, IP routing table with device
  discovery.
- Alerts panel — threat badges, dim threshold, location hint.
- Sidebar — voice auto-play toggle (sliding switch), TTS/STT indicators
  in fixed-width table, Settings separated by divider.
- Remove button styled consistently (`btn danger-soft`).

## Test coverage

- 140 classifier tests (every tool, every phrasing)
- 80 profile tests (scaffolding + preferences + UserContext precedence +
  migration + API)
- 25 threat_llm tests (clamp, cache, citation, HTTP error, bad JSON)
- 19 proximity tests (scoring per signal kind)
- 12 alerts panel API tests (threat_score, location_configured)
- 12 routing tests (load/save, resolve, validation)
- 14 profiles API tests (list, create, switch, path traversal rejection)

---

# Review changes — 2026-04-13

One-session audit + polish pass before Jetson deployment. All 341 tests
pass after these changes (up from 307 before — added smoke coverage for
the newer tools).

## Bug fixes

1. **Missing history persistence on non-streaming routes**
   `/api/turn` and `/api/text-turn` were running LLM turns but never
   calling `_persist_and_maybe_compact()`. Only the streaming variants
   saved to disk. After a page reload, any turn made via the non-streaming
   endpoints was gone. Both now persist + compact like their streaming
   counterparts.
   [web/server.py](web/server.py) — around `/api/turn` and `/api/text-turn`.

2. **`eval()` in the boolean logic evaluator**
   `logic_eval` and `truth_table` in the `circuit` tool were calling
   `eval(py_expr, {"__builtins__": {}}, env)`. Disabling builtins isn't
   a sandbox — `True.__class__.__mro__` style escapes work around it.
   Replaced with a new `_safe_bool_eval` helper that AST-walks the
   expression with a fixed whitelist (BoolOp / UnaryOp Not / Compare
   Eq+NotEq / Name / bool Constant). Rejects `Call`, `Attribute`,
   `BinOp`, and any unknown names. Smoke-tested with 4 known escape
   patterns — all rejected.
   [bridge/tools/_circuit.py](bridge/tools/_circuit.py) — `_safe_bool_eval`, `logic_eval`, `truth_table`.

3. **HTTP timeout inconsistency**
   `_geocode_open_meteo` used 6s; everything else used 8s. Standardized
   to 8s.
   [bridge/tools/_weather.py](bridge/tools/_weather.py) — `_geocode_open_meteo`.

## New tests

4. **Added [bridge/tests/test_new_tools.py](bridge/tests/test_new_tools.py)** — 34 smoke tests covering:
   - `math` — evaluate, solve, integrate (definite), differentiate,
     rref, inverse, laplace, inverse_laplace, fft, unknown_op
   - `convert` — length, temperature (absolute), incompatible units,
     non-numeric value
   - `graph` — single curve, multiple curves, bad range, divide-by-zero
     → null samples
   - `circuit` — resistance_parallel, rc_cutoff, lc_resonance,
     impedance (capacitor), voltage_divider, logic_eval, truth_table
     (XOR), **sandbox rejection of code injection**, synthesize,
     out-of-range minterms, unknown op
   - Dispatch integration — each new tool reachable via `tools.execute()`.

## Documentation

5. **[README.md](README.md) rewritten** to reflect current reality:
   - Listed all 14 LLM tools in a table with widget presence
   - Documented the browser UI as a first-class path alongside the
     voice loop
   - Repository layout updated (added `bridge/history.py`,
     `bridge/memory.py`, `web/static/panels/`, `data/`, `web.service`)
   - Credits section now includes SymPy, pint, ddgs, KaTeX, Plotly,
     Open-Meteo, ipapi, Wikipedia.

## Already correct (flagged by audit, verified ok)

- **`pendingTexts` queue on new chat / switch / delete** — already
  cleared at [web/static/app.js](web/static/app.js) in every path.
  Audit flag was stale.
- **`streamTurn` network errors** — propagate up to the outer
  try/catch in `submitVoice` / `_runTextTurn`, which render them as
  `turnShowError`. Not silent.
- **`.new-chat-btn` CSS duplication** — the two rules are a base +
  sidebar override, not dead code. Verified.

## Open recommendations (deferred — noted for later)

These would each be a separate session:

- **Split `bridge/tools.py`** (2,100+ lines) into a subpackage:
  `bridge/tools/{weather.py, math.py, circuit.py, web.py, places.py}`
  with `__init__.py` exporting `TOOL_SCHEMAS`, `_DISPATCH`, `execute`.
  Mechanical refactor; improves review + test locality.

- **Memory / history tests** — `bridge/history.py` and
  `bridge/memory.py` have no coverage. Tests would validate:
  `ConversationStore.new / save / load_current_or_new / switch_to /
  delete_conversation`, `maybe_compact` boundary conditions, memory
  file truncation + UTF-8 round-trip.

- **`llm.set_history` / `llm.summarize` integration tests** — mock the
  Ollama `/api/chat` endpoint and verify `summarize` posts the right
  payload shape + `commit_history=False` doesn't mutate history.

- **Tool-count tuning for small models** — at 14 tools, Qwen 4B is
  near its tool-selection accuracy ceiling. If we add more, consider
  consolidation patterns like we did with `math` (op enum over
  sub-tools).

- **CDN failure fallbacks** — KaTeX + Plotly both load from jsDelivr.
  If the CDN is down the math widget shows raw LaTeX (already
  handled) but the graph widget shows nothing. Consider a local copy
  or a status indicator.
