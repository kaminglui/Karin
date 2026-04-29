# Karin — design principles

A snapshot of the architectural philosophy and decisions behind this
repo. Meant to be read after you've poked around the code for an hour
and want to know *why* things are the way they are. The code is the
source of truth; this doc is the mental model.

Not exhaustive. Captured here so future-us (and future contributors)
don't have to re-derive the reasoning from git archaeology.

---

## 1. Active vs passive class — the core taxonomy

Every tool / data source lands in exactly one of two buckets:

**Active class** — on-demand, hit-upstream-on-request, user accepts latency.
  - `get_time`, `get_weather`, `math`, `convert`, `graph`, `circuit`,
    `find_places`, `web_search`, `wiki`, `tracker(id=fx)`
  - Typical call path: user asks → tool → external API → ~1-5 s reply
  - Characteristic: data is volatile or arbitrary-input (what time is
    it, convert 5 mile to km). Pre-fetching doesn't make sense.

**Passive class** — pre-fetched on an interval, served from ledger,
instant on user query.
  - `get_news`, `get_alerts`, `get_digest`, slow-moving trackers
    (`gas_retail`, `gold`, CPI)
  - Typical call path: background poller → code pipeline (normalize →
    cluster → confidence → ledger). User query → ledger read → reply
    in <100 ms of local I/O.
  - Characteristic: data changes slowly relative to query frequency.
    Precomputing filters + state is cheaper than recomputing per turn.

**Rule for new features:** ask "would the user tolerate a 5 s wait on
the first query, or do they want instant answers?" That decides active
vs passive, which decides whether the feature needs a poller + ledger
or lives purely on the hot path.

## 2. Code is the trust path. LLM is a supplement.

The LLM never makes *decisions about truth*. Every piece of state the
user sees comes out of deterministic code:

  - News confirmation state (`DEVELOPING` → `PROVISIONALLY_CONFIRMED` →
    `CONFIRMED`) is computed by
    [bridge/news/confidence.py](../bridge/news/confidence.py) from
    source-tier bucketing rules. No LLM vote.
  - Alert levels come from
    [bridge/alerts/rules.py](../bridge/alerts/rules.py) — explicit
    threshold-based rules on structured signals. No LLM classifier.
  - Digest composition is deterministic aggregation over the three
    subsystems ([bridge/digest/service.py](../bridge/digest/service.py)).
  - Tracker values come from named-source fetchers with published
    prices. No LLM interpretation.

**Where the LLM participates:**

  - **Tool routing.** LLM picks which tool fits the user's prompt.
    Guardrailed by per-tool call caps
    ([bridge/llm.py](../bridge/llm.py) `MAX_PER_TOOL`,
    `MAX_TOOL_ITERS`) and JSON-leak recovery.
  - **Voice.** LLM paraphrases pre-computed `voice_line` strings in
    Karin's persona. It reads a 1-sentence fact and outputs a
    1-sentence casual remark. It does not rewrite or synthesize facts.
  - **Semantic dupe check** ([bridge/news/cross_verify.py](../bridge/news/cross_verify.py)).
    **Merge-only**, never split. Only acts on pairs the code already
    marked as borderline (Jaccard 0.30–0.55). Can't force a merge
    below 0.30 regardless of what the LLM votes. Every decision
    logged to `events.jsonl` for audit.

**Why this split:** LLMs hallucinate. We want a system that degrades
to "code-only" cleanly when the LLM is unavailable, slow, or wrong.
The LLM adds flavor and handles ambiguity, but removing it should
never break correctness — only UX.

## 3. Every LLM decision is logged

`cross_verify_decision` events go into `bridge/news/data/events.jsonl`
with prompt + response + outcome. Same pattern for `extract_ok`,
`ingest_ok`, etc. Two reasons:

  - **Audit.** "Why did we merge these two clusters?" is always
    answerable by grepping events.
  - **Future RL.** The `(input, model_output, outcome)` triples are
    exactly what a reward model or an offline evaluation harness would
    consume. We're not training one today; the log is free insurance.

## 4. Fail-soft everywhere

Every external integration degrades gracefully on failure. Missing API
keys → feature disabled, rest of app works. External API down →
function returns an empty list, not an exception. LLM returns empty →
retry once, then surface a clean error to the user.

This is enforced by contract in most modules: error paths log a warning
and return sentinels (`[]`, `None`, friendly-string). A production
outage never cascades into a full server failure.

## 5. Feature flags > conditionals

Anything that's user-toggleable goes in
[config/features.yaml](../config/features.yaml), consumed via
`bridge.features.is_enabled(name)`. The registry already covers STT,
TTS, bandit, holidays, news wires, and cross-verify. Env vars
override YAML (deploy-time flipping without file edits).

Rule: a second env-var flag for a new subsystem becomes a YAML entry
+ one call to `features.is_enabled`. Don't bake conditionals into the
code directly.

## 6. Per-model tuning overlay

[config/models.yaml](../config/models.yaml) holds per-model caps
(`max_per_tool`, `max_tool_iters`, `temperature`, `num_ctx`, `think`,
`history_pairs`, `request_timeout`). Model-swap is a config edit
(`KARIN_LLM_MODEL` env + restart), not a code change.

This lets us A/B models without touching `bridge/llm.py`. It also
makes it explicit that 3B vs 8B vs 4B-abliterated each want
different knobs — no one-size-fits-all default.

## 7. Token budget discipline

Every tool schema costs tokens in the system prompt. With `num_ctx`
typically 3072-4096 on Orin Nano, an unbounded tool list would
crowd out conversation context.

Decisions following from this:

  - Tool schemas **trim descriptions to one short sentence** of when
    to use it (post-consolidation). Long prose killed routing.
  - Retired / obsolete tools get removed from `TOOL_SCHEMAS` (LLM
    never sees them) but kept in `_DISPATCH` if any persisted
    conversation history references them (so widget replay still
    works). See the legacy `wiki_search` alias.
  - Shared code (`active_tool_schemas()`) filters the registry by the
    features flag so deploys that disable tools also reclaim tokens.

Corollary: we **do not** stuff routing rules into karin.yaml beyond
what the model actually needs. Heavy prompting reliably degrades
routing on smaller models because specific examples get copied
verbatim into replies.

## 8. The interactive-widget philosophy

If the user can do it themselves in the browser via local JS, *don't*
round-trip through the LLM. FX conversion, unit conversion, math on a
displayed value — all deterministic, all done in the page. The LLM's
role stops at "here's the number and a widget to play with."

This keeps latency flat (no turn cost per tweak) and avoids errors
that would otherwise come from the LLM re-deriving arithmetic.

## 9. No MCP, no C++ rewrite

**MCP** (Anthropic's Model Context Protocol) is not worth the overhead
for a single-host personal assistant. MCP shines in cross-process /
cross-language / third-party-tool contexts. Ours is a single-host
Python app with local tools. Adding MCP would add latency, a
subprocess boundary, and a dependency without unlocking anything.

**C++ rewrite** of our Python hot paths is a red herring. Our
workloads are dominated by:
  - httpx round-trips to external APIs (network-bound, not CPU)
  - Ollama LLM inference (already C++ via llama.cpp)
  - ~400-element set intersections (microseconds in Python)

Moving any of this to C++ costs weeks for zero measurable win. The
perceived-slow parts are LLM inference and external API latency —
neither is our code.

## 10. Hardware-aware defaults

Orin Nano 8 GB shared memory (measured 7.4 GiB usable) forces decisions:

  - LLM default is `karin-tuned:latest` — an iter-3 LoRA on
    `mannix/llama3.1-8b-abliterated:tools-q4_k_m` at Q4_K_M (~4.9 GB
    weights, 9%/91% CPU/GPU spill in practice). `qwen2.5:3b` is a
    smaller pre-LoRA fallback; `llama3.1:8b` raw is the training base.
  - STT (`faster-whisper`, config default `tiny.en` on CPU int8) and
    TTS (GPT-SoVITS) both default to OFF. Flipping them on is
    opt-in at setup (`LOCAL_STT=yes` writes `.env` with `tiny.en` —
    Orin-Nano-safe; `LOCAL_SOVITS=yes` for bigger boxes) or via PC
    offload (`KARIN_STT_BASE_URL` + `KARIN_TTS_BASE_URL`).
  - `num_ctx` tuned per-model in
    [config/models.yaml](../config/models.yaml) so KV cache doesn't
    spill to CPU.

A PC deploy can flip all these on by overriding `models.yaml` +
`features.yaml`. Nothing is hardcoded to "Jetson only" — just the
defaults lean conservative.

## 11. LLM behaves reactively, not proactively

The LLM only invokes data-fetch tools (`get_news`, `get_alerts`,
`get_digest`, `tracker`, `get_weather`) when the user's message
**explicitly requests that kind of info**. It never calls them as a
greeting filler or as a fallback for an unclear prompt. Ambiguous
prompts route to a short conversational reply, not a data fetch.

The reasoning: **the user should always know what the LLM is doing.**
If the user said "hi," a news query lands out of context and surprises
them. If the user said "latest news," the query is expected. The
passive-class pollers handle "keep data fresh" in the background;
the LLM doesn't need to prefetch on user-facing turns.

## 12. The digest is the canonical passive surface

`/ui/digest` is the one page that answers "what do I care about today?"
It composes from news + alerts + trackers. Other UIs (`/ui/news`,
`/ui/alerts`, `/ui/trackers`) are drill-downs.

New passive data feeds should aim to surface in the digest first, then
gain their own drill-down only if the digest's representation isn't
enough.

---

## Meta: what this doc is NOT

- Not a list of features. Read [README.md](../README.md).
- Not operational runbook. Read [RUNBOOK.md](../RUNBOOK.md).
- Not a module API reference. Read each module's top docstring.

It's the "why" document. When you're tempted to add a feature that
violates one of these rules, reread it and make sure the deviation is
worth the complexity.
