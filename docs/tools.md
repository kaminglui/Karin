# Tools available to the LLM

Each tool schema is defined once in
[bridge/tools/_schemas.py](../bridge/tools/_schemas.py), and runtime
dispatch lives in
[bridge/tools/_dispatch.py](../bridge/tools/_dispatch.py). The repo has
21 tool schemas total; 20 are active by default because
`schedule_reminder` stays hidden unless the `reminders_llm_tool`
feature flag is enabled. Multi-step tool calls are supported: the LLM
can chain (look up a constant → run `math` on it → reply) up to the
iteration cap in [bridge/llm.py](../bridge/llm.py).

"Widget?" = whether the browser UI renders a dedicated inline card
for the tool's output. Tools without widgets return plain text that
the LLM paraphrases into the reply.

| Tool | Purpose | Widget? |
|---|---|---|
| `get_time` | Current date / time in any timezone | ✓ |
| `get_weather` | Current conditions; IP-fallback if no location given | ✓ |
| `get_news` | Top news headline (RSS feeds: Guardian, NPR) | ✓ |
| `get_alerts` | Active system alerts / advisories | ✓ |
| `get_digest` | Today's digest: top news, active alerts, tracker moves | ✓ |
| `tracker` | Gold, FX rates, gasoline, food CPI, stock indices | ✓ |
| `math` | **Symbolic** (algebra): evaluate, solve, simplify, factor, expand, rref, det, inverse, transpose, eigenvalues / eigenvectors, multiply. **Calculus**: integrate (definite via `lower`/`upper`), differentiate, **`taylor` / `series` / `maclaurin`** (center via `lower`, order via `upper`; default Maclaurin order-6), **`limit`** (`lower` = target; supports `'inf'` / `'-inf'`; `upper` = `'+'` / `'-'` for one-sided), **`extrema` / `optimize` / `minimize` / `maximize`** (critical points via 2nd-derivative test, optional closed-interval endpoints via `lower`/`upper`). **Transforms**: laplace / inverse_laplace, fourier / inverse_fourier, fft / ifft. **Numeric / ML**: `dot`, `norm` (L1 / L2 / L∞ via `variable`), `softmax`, `sigmoid` / `tanh` / `relu` (element-wise), `mean` / `variance` / `std`, `mse` / `mae` / `cross_entropy`. **Information theory (discrete)**: `entropy`, `kl` / `kl_divergence`, `js` / `js_divergence`. Vector input `"[1,2,3]"`; two-vector input `"[0.9, 0.1] * [0.5, 0.5]"` (permissive: `*` / `and` / `,` / `\|\|` or no separator all work). **Information theory (continuous, closed-form + numerical)**: `kl_dist` (`"N(0,1) \|\| N(1,2)"`), `entropy_dist` (`"Beta(2, 5)"`). Distributions supported: `Normal`/`N`/`Gaussian` (μ, σ), `Exponential`/`Exp` (rate), `Beta` (α, β), `Bernoulli`/`Bern` (p), `Uniform`/`U` (a, b). **Cross-family KL (added 2026-04-22)** falls back to trapezoid-rule numerical integration over P's effective support; returns `+∞` when P's support escapes Q's positive-density region (e.g. `KL(N(0,1) \|\| Beta(2,5)) = ∞`). Parses `x^2` and `x**2` identically (sympy `convert_xor`). **Wrapper tolerance**: the paren-balanced scanner recurses into non-distribution function calls, so `kl(Beta(2,5), N(0,1))`, `D_KL(N(0,1) \|\| N(1,2))`, `entropy(Beta(2,5))`, `H(N(0,1))` all resolve the same as the bare shorthand. | ✓ (KaTeX) |
| `graph` | Plot a function (one or many curves) over a range. Parses both `x**2` and `x^2` (sympy `convert_xor`). **Named presets** (added 2026-04-22): `Gaussian` / `sigmoid` / `tanh` / `relu` / `softplus` / `exponential` / `laplace` / `cauchy` / `logistic` / `uniform` / `sine` / `cosine` / `parabola` / `cubic` resolve to canonical formula + sensible x-range, so the LoRA doesn't have to invent the math for standard distributions. **Widget** (updated 2026-04-22): mouse-wheel zoom, drag to pan, double-click resets view, max/min markers with annotations, crosshair spike-lines, hover tooltip. | ✓ (Plotly) |
| `circuit` | Analog (R-series / parallel, RC-τ / cutoff, RL-cutoff, LC-resonance, impedance, voltage divider) + digital (logic_eval, truth_table, **synthesize** from minterms) | ✓ |
| `convert` | Units: length, mass, time, temperature, energy, power, pressure, speed, volume (pint) | ✓ |
| `wiki` | Wikipedia lookup: query = topic summary (with computed bio ages + event dates); empty = random article | ✓ |
| `web_search` | DuckDuckGo (no API key) for web-shaped queries | ✓ |
| `find_places` | "Best pizza near me" — location from IP if not given | ✓ |
| `schedule_reminder` | Set a push notification for a future time. Feature-flagged (`reminders_llm_tool`) | ✓ |
| `update_memory` | Save a fact about the user (name, location, preference). Called silently by the LLM | — |
| `say` | Repeat back user-provided text for explicit "say", "echo", or "repeat after me" requests. | — |
| `inflation` | Historical purchasing-power equivalence via cached BLS data: CPI-U (US, 1913-present) + AHETPI hourly wages (US, 1964-present). Returns the equivalent dollar amount, the wage ratio, and a real-wage delta with explicit "outpaced/lagged" direction. Inline widget plots the value-of-$1 curve over time + nominal-vs-real wages overlay (Plotly, mouse-wheel zoom + drag-pan). Deterministic — code does all math, LLM only picks tool + extracts args + paraphrases the cited result. International coverage via `region`/`regions` arg: Hong Kong SAR (HKD 1981+), China Mainland (CNY 1986+), Japan (JPY 1960+), South Korea (KRW 1960+) via World Bank. Cross-region overlay normalized to fy=1.0. See [docs/inflation-tool-plan.md](inflation-tool-plan.md). | ✓ (Plotly) |
| `population` | World Bank `SP.POP.TOTL` time series for 6 regions (US, Hong Kong SAR, China Mainland, Japan, South Korea, World aggregate). Same architecture + region keys as `inflation` for cross-tool symmetry. Plotly widget shows full-history line with markers for the queried year(s). | ✓ (Plotly) |
| `facts` | Year-card aggregator: `facts(year=1985)` packages world population + US inflation baseline + cohort age + US wage snapshot + 5 BLS AP item-price highlights + Wikipedia year-article snippet — each with a "today" comparison (`then → now`, change_pct, ratio). `region=` arg adds regional breakdowns. Section-card widget at `/api/facts` + standalone `/ui/facts` page with year/region picker + quick-pick chips. | ✓ |
| `analyze` | Operates on a series produced by another tool (or passed in directly). Operations: `peak`, `trough`, `trend` (slope), `volatility` (stddev), `percentile_rank`, `correlate` (Pearson r between two series). Pure numpy, no scipy dependency. Used by the panel chrome for "what's the peak / trough / trend?" questions over inflation/population output. | — |
| `alice` | Estimates % of US households that are ALICE (Asset-Limited, Income-Constrained, Employed — above poverty, below survival budget) by household composition. 6 compositions (1A0K through 2A3K) drive housing bedrooms, food adult-equivalents, healthcare coverage tier. `_alice_tax.py` runs federal brackets + standard deduction + CTC + simplified EITC + FICA + 5% state-avg per (year × filing status × num kids); iterative gross-up because EITC is non-linear. 6 years (2016/2018/2020/2022/2023/2024). Reachable via LLM, `/api/alice`, and `/ui/alice` page. | ✓ |

## Routing safety nets (small-model guardrails)

The bridge applies these *across* all tools to compensate for 3B-class
routing imperfection:

- **Per-tool cap (`max_per_tool`, default 1-2)** — a single tool can be
  called at most N times per turn, regardless of args. Configurable per
  model in `config/models.yaml`. Prevents wiki loops through synonym
  probes.
- **Same-(name, args) suppression** — the LLM doesn't get a second
  shot at an already-tried call; it sees a synthetic "already
  attempted" tool result and is nudged to try a different approach.
- **`max_tool_iters` (default 3-5)** — at most N tool-call iterations per
  turn. Configurable per model in `config/models.yaml`.
- **Forced final reply on cap exhaustion** — instead of returning
  `"(gave up after too many tool calls)"`, the bridge issues one
  more `/api/chat` *without* the `tools` field so the model has to
  produce a real answer from what it has.
- **Auto-retry on Ollama 5xx** (one extra attempt with a 3 s delay)
  hides transient runner crashes from sovits memory contention.

## Adding a new tool

1. Append a JSON-Schema block to `TOOL_SCHEMAS` in
   [bridge/tools/_schemas.py](../bridge/tools/_schemas.py) describing
   the name, purpose, and arguments.
2. Implement the function and register it in the dispatcher in
   [bridge/tools/_dispatch.py](../bridge/tools/_dispatch.py).
3. (Optional) Add an inline widget under
   [web/static/panels/](../web/static/panels/) using the shared
   `Panels.mountPanel(...)` helper from
   [panels.js](../web/static/panels/panels.js), then wire it into
   [chat.js](../web/static/panels/chat.js) (`PANEL_MOUNTERS` +
   `PANEL_TITLES`) and load the script from
   [index.html](../web/static/index.html).
4. Add a row above with a 1-line purpose and the widget flag.

## Testing routing accuracy

After adding or renaming a tool, run the live routing eval to make
sure the LLM still picks the right tool:

```bash
python scripts/eval_routing.py --verbose
```

Add cases covering the new tool to
[sft/eval_cases_novel.yaml](../sft/eval_cases_novel.yaml) so regressions
are caught. The build guard (`sft/scripts/assert_disjoint.py`) will fail
the dataset build if the new eval prompts overlap with the training
phrase library, so keep them fresh.
