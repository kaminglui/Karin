# Ideas — feasibility backlog

Forward-looking ideas that aren't scoped or scheduled yet. Each entry
is a one-paragraph capture + a quick feasibility read (not a plan).
Move items into `TODO.md` or a milestone once the tradeoffs are clear
and the user wants to commit.

Captured 2026-04-15.

---

## Outbound notifications (Discord / message / API)

**The idea.** Let Karin push out messages on her own — not just reply
in the chat panel. Channels could be a Discord webhook / bot, SMS via
Twilio, Pushover / ntfy.sh for phone notifications, or a plain webhook
other tooling can subscribe to.

**Feasibility:** high. Easiest path is **Discord webhook + ntfy.sh**.
Both are free, no OAuth dance, one HTTP POST per message. The hard
part is deciding *when* Karin should push, not *how*. Natural triggers
already exist:

* `get_alerts` hits a level ≥ MODERATE
* a tracker moves more than N% since the last digest
* a news item's confidence flips to `CONFIRMED`

Wire a thin `bridge/notify/` module: one function per channel, plus a
`notify(event, message)` dispatcher that reads which channels are
enabled from `config/features.yaml`. The existing poller threads
(`bridge/pollers.py`) are the natural caller. Keep messages fail-soft
— a 500 from Discord should log a warning and move on, not break the
poller loop.

Scope guess: half a day once the trigger rules are agreed.

---

## Video summarization

**The idea.** Feed Karin a video (YouTube link, local file) and have
her produce a useful, compressed summary — both text and maybe a
clipped highlight reel.

**Feasibility:** low on-device, medium with an off-box step. This is
genuinely a research problem. To do it well on the Jetson alone we'd
need all of:

* audio track extraction + Whisper transcription (already have STT;
  add ffmpeg)
* VAD-based filler removal (already have Silero VAD — can reuse)
* BGM / noise separation (new — demucs or spleeter; both heavy)
* keyframe / scene-change detection so we skip invariant frames (pyscenedetect)
* visual captioning per keyframe (BLIP-2 / InternVL / LLaVA — all
  >4GB VRAM; won't fit next to Ollama on the Orin Nano)
* summarization over the joined transcript + captions

The vision-captioning step is the blocker: no model that fits the
Jetson's spare VRAM is *good* at this. The realistic path is to
offload captioning to a service (an API, or a second PC) and keep
only audio+scene work on the Jetson. Even then, end-to-end quality
depends on the video — a scripted lecture summarizes well, a vlog
with chatty asides is much harder.

Recommend: defer until (a) there's a concrete use case (which videos,
how often) and (b) smaller vision models catch up. Build the audio
side first — transcription + filler-removal + summary works today
and covers lectures / podcasts / meetings without the hard parts.

---

## Post-training (LoRA / GRPO) + RAG / embeddings

**The idea.** Instead of fighting model-behavior issues with prompt
engineering + regex guards, teach a small open model to behave the
way we want. Also add RAG for long-term memory / personal knowledge.

**Feasibility:** higher than most people think for LoRA; questionable
payoff for full GRPO; straightforward for RAG.

**LoRA** for tool-routing + persona voice is feasible. The events
logging we just added (`bridge/routing/data/events.jsonl`,
`bridge/news/data/events.jsonl`) is already the shape of a training
set: `(prompt, system-prompt, correct-tool)` triples. With ~2-5k
hand-curated examples we could fine-tune a LoRA adapter on top of
llama3.1:8b or qwen2.5:7b that bakes Karin's voice + routing rules
into weights. Rough cost: one GPU-day on a 24 GB card (rented) per
iteration. The real cost is curating the dataset — 2-5k examples is
maybe 10-15 hours of labeling.

**GRPO / RLHF** has a much worse ratio here. It shines when you have
a reward signal you can compute cheaply at scale (math correctness,
code execution). For "did Karin sound right?" the reward is human
preference and we'd need to label a lot of pairs. Skip unless we
discover a quantifiable reward we already have data for (e.g.
"did the user thumbs-up the reply" — the `bridge/feedback.py` signal).

**RAG / embeddings** is the clearer win and is lighter-weight. Useful
for: Karin remembering past conversations, pulling user-specific
context ("last time you asked about X I said Y"), and grounding wiki
answers. Stack to try:

* `bge-small-en-v1.5` or `all-MiniLM-L6-v2` for embeddings — both
  run fast on CPU, no VRAM cost, sub-100 ms per doc.
* `sqlite-vec` or `chromadb` for storage — SQLite is simpler and
  ships with Python; no server to run.
* Index: conversation history, the news `articles.json`, any docs the
  user drops in.

The hard part is choosing when to retrieve. Retrieving on every turn
bloats the prompt with junk; retrieving only on explicit cues misses
the value. A small classifier ("does this prompt need grounding?")
can decide — and that's exactly the pattern we already built for
tool routing.

**Recommended order** if pursuing any of this:

1. Start logging thumbs-up/thumbs-down consistently so any future
   post-training has real preference data. (Partially done —
   `bridge/feedback.py`.)
2. Ship a small RAG index over past conversations. Low risk, modest
   complexity, immediately useful.
3. Only then consider LoRA on the routing/voice dataset. By that
   point the events log will have months of data.
4. GRPO remains parked unless a concrete reward signal appears.

---

## TTS offload to a PC peer

**The idea.** GPT-SoVITS v2Pro is the heaviest single thing on the
Jetson (~2-3 GB VRAM + slow inference on Orin Nano). The PC has more
VRAM and is usually idle. Split TTS into a standalone HTTP service
running on the PC; Jetson becomes a thin client over tailnet.

**Architecture.** The existing `tts-server/` directory is the seed —
promote it into a proper service:

* PC side: FastAPI process wrapping the existing GPT-SoVITS `api_v2.py`
  (pinned to commit 2d9193b0 per memory). Exposes
  `POST /synthesize {text, voice, speed} → audio/wav stream`.
* Jetson side: replace `bridge/tts.py`'s in-process synth with a
  thin HTTP client pointed at `http://pc.tailnet:PORT/synthesize`.
  Keep the streaming interface identical so audio playback code
  is untouched.
* Fallback: on timeout / PC unreachable, log a warning and return
  text-only. Never break the chat turn because TTS is down.

**Tradeoffs.** One extra hop (~5-20 ms LAN) but TTS is only on
user-facing turns, never the hot path. Reclaims the ~2-3 GB VRAM on
the Jetson for a larger LLM (fits a 13B-class Q4 model next to STT).
PC doesn't need to be always-on — when it's off, Karin simply replies
in text. Biggest watch-out: voice-training artifacts (the reference
audio + the trained checkpoints) stay in one place — on the PC where
they were produced.

**Scope guess:** 1 day once we pin the API shape and write the client
swap. STT stays on the Jetson unchanged (small model, hot path).

---

## Refined: self-improving routing classifier

(Refinement of the original "use events.jsonl to tune patterns" idea.)

**Script:** `scripts/suggest_patterns.py`. Runs nightly (or manually).
**Does not auto-edit code** — emits a markdown proposal for review.

**Logic.**

1. Load `bridge/routing/data/events.jsonl`, filter to
   `abstained == true AND picked` non-empty (the classifier missed
   but the model routed to a real tool anyway).
2. Group by `picked[0]`. For each group with ≥ 5 prompts in the last
   30 days, tokenize the prompts and find tokens that appear in ≥ 60%
   of them but aren't in that tool's existing patterns.
3. Emit a proposal block per group:
   ```
   ## tracker — 8 abstain-prompts with common token "quote"
   candidate regex: \bquote\s+(for|of)\b
   examples:
     - "get me a quote for copper"
     - "quote of tin futures"
   apply? (y/n) — then edit classifier.py manually
   ```
4. Human reviews + applies. Keeps the trust path for routing where it
   belongs: in deterministic code, not in an auto-trained model.

**Why not auto-apply:** a bad regex silently misroutes prompts. The
cost of a false regex >> cost of a missed one. Review is cheap
(~30 s per proposal); a regression takes hours to unwind.

---

## Refined: reminders / working memory

**Blocker:** needs the **Outbound notifications** module above (fires
due reminders via Discord / ntfy / etc.). Build notifications first.

**Module layout:** `bridge/reminders/`:

* `store.py` — SQLite (`reminders.db`). JSON files lose races against
  the 60 s scheduler tick.
* `scheduler.py` — daemon thread in `bridge/pollers.py`. Ticks every
  60 s, fetches due rows, calls `bridge.notify.notify(...)`, marks
  them delivered.
* `parser.py` — wraps `dateparser` for natural-language time phrases
  ("5pm tomorrow", "in 2 hours", "next Friday morning"). Timezone
  from `bridge/location.user_timezone()` so users don't have to
  specify.
* `tool.py` — registers a `reminder` tool with `op=create|list|cancel`,
  added to `TOOL_SCHEMAS` behind a feature flag.

**Edge cases worth flagging up front:**

* User deletes reminder while scheduler is firing it — use
  `UPDATE ... WHERE delivered = 0` guards.
* System clock skew after a Jetson reboot — log ts of last tick
  vs. now on startup; if > 5 min gap, drain missed reminders as
  "delayed" with a note in the message.
* Recurring reminders ("every Monday at 9") — out of scope for v1.
  Keep one-shot only; add recurrence in v2 if used.

**Scope guess:** 1 day for v1 (after notifications is in).

---

## Refined: LLM-as-judge reply quality eval

**Judge model:** `qwen2.5:3b` — small, fast, *different from prod*
(self-judging inflates scores). Runs locally, no API cost.

**Rubric** (Karin's voice rules as discrete checks):

| Check | How |
|---|---|
| length ≤ 20 words (single-tool turn) | programmatic |
| length ≤ 60 words (multi-tool turn) | programmatic |
| no markdown / bullets / headers | programmatic regex |
| no forbidden prefix (`Note:`, `The output is`, etc.) | programmatic |
| stays in persona voice | LLM-judged, 0-3 scale |
| reply uses ≥ 1 fact from tool result (if any) | LLM-judged, 0-1 |

Programmatic checks are free + exact. Only the voice/factuality
checks need the LLM. Composite score = weighted average, 0-100.

**Harness:** `scripts/eval_voice.py`. Samples N recent turns from
`data/conversations/*.json`, scores each, writes
`data/eval/voice_scores_YYYY-MM-DD.jsonl`. Prints trend over last
30 days. Hook into a nightly cron eventually; run manually for now.

**Payoff:** quantifies the wiki-verbose problem (issue 4 from the
reply audit) so we can compare models + prompt changes objectively.
Also makes prompt-cut experiments (trim karin.yaml further) safe —
score before + after, keep the cut only if score holds.

**Scope guess:** half a day.

---

## Refined: code-aware mode

**Phase 1 (build first):** ripgrep-backed tool.

* Tool: `code_search(query: str, path?: str, max_results=10)`
* Config: `config/code_paths.yaml` listing allowed repo roots. Guard
  against accidental `/etc` or `~/.ssh` scans — path must match one
  of the configured roots.
* Runtime: subprocess `rg --json -n <terms>` → parse → return top
  matches as `file:line → snippet` with ±10 lines of context.
* Cap: total response ≤ 50 lines across all results; truncate with
  a `... (N more matches)` tail. Protects the context window.
* LLM flow: model sees the snippets in tool output and synthesizes
  an answer ("The function that does X is in `bridge/foo.py:42`,
  it looks like ...").

**Phase 2 (only if Phase 1 gets used):** semantic index.

* Embed each file chunk with `bge-small-en-v1.5`.
* Store in `sqlite-vec`.
* New op: `code_search(query, mode="semantic")`.
* Rebuild on file-system watcher events (watchdog lib).

**Rationale for phasing:** grep answers the most common questions
("where is X defined", "what calls Y") for free. Semantic retrieval
earns its complexity only for fuzzy questions ("what's the error-
handling strategy"), and those are rarer. Don't build Phase 2
preemptively.

**Scope guess:** half a day Phase 1, 2-3 days Phase 2 if we go.

---

## Digest as an LLM-written summary with source links

(Captured 2026-04-15 during the digest-bug triage.)

**The idea.** The current digest page just re-renders the same news /
alerts / tracker cards you'd see on each individual tab — it duplicates
those surfaces instead of summarizing them. An alternative: have the LLM
produce one short paragraph per day ("today's news trend is X, one alert
active about Y, gold moved Z%") with inline links back to the source tab
for any reader who wants detail.

**Why defer.** Needs a `digest_summarize` pipeline: collect the same
raw data we have now, feed it to the LLM, store the paragraph in the
digest JSON alongside the structured items. The structured items stay
(they're cheap to ship; they back the "open News →" links). Also
needs prompt design so the summary reliably cites sources — probably a
small JSON output with `{summary: "...", citations: [...]}`.

**Adjacent fix that's already landed.** The `title` vs `headline`
attribute-name mismatch on StoryBrief was dropping news headlines
from the digest rendering (cards showed just a state pill). Fixed
in [bridge/digest/service.py](../bridge/digest/service.py) with a
multi-name fallback so it's robust to future model renames.

---

## Emergency alerts: truly passive delivery (SSE / push)

(Captured 2026-04-15 after tightening the alerts poll interval to 2 min.)

**The idea.** Today the browser + the Jetson both pull: the Jetson
polls NWS / state.gov every 2 min, and any open alerts panel manually
refreshes when loaded. A new severe-weather CAP entry still waits up
to ~2 min for the Jetson to notice, plus however long until the user
next opens the panel. We want the user to see it *immediately* —
ideally without opening any panel.

**What "passive" actually looks like.**
* NWS does not push: no CAP webhook, no WebSocket. So the *Jetson*
  must keep polling. We can't remove that leg.
* Between Jetson and the browser: add a **Server-Sent Events (SSE)
  stream** at `/api/alerts/stream`. When the alert poller lands a new
  `CRITICAL` or `ADVISORY` row in the ledger, it publishes an event
  on the stream. Any connected browser receives it within ~1 s and
  can (a) raise a toast over the chat UI, (b) auto-refresh the alerts
  panel if it's open, (c) forward to Discord/ntfy via the
  outbound-notifications module (see above).

**Implementation sketch.**
* New endpoint: `/api/alerts/stream` (FastAPI `StreamingResponse`
  with `text/event-stream`). Holds the connection open; emits
  `data: {...}\n\n` per new alert.
* `bridge/alerts/service.py` gets a pub-sub: `subscribe()` returns a
  `queue.Queue` the endpoint reads from; `scan()` pushes to all
  subscribers after persisting new rows.
* Frontend: `app.js` opens an `EventSource` on startup, listens for
  `alert` events, shows a toast + a banner.
* Cross-browser gotcha: EventSource reconnects on disconnect
  automatically; throw in `retry: 5000` on stream open to keep
  reconnect latency bounded.

**Scope guess:** ~1 day. Standalone from the outbound-notifications
module but shares the same trigger rules.

---

## Residential utility trackers (electricity / natural gas / water)

(Captured 2026-04-15 alongside the RBOB removal + crypto-placeholder
work.)

**The idea.** Expand the `energy` tracker category to cover what the
user actually pays at home, not what the market pays at wholesale.
Retail gasoline is already here (EIA weekly, PADD region configurable);
what's missing is residential electricity, residential natural gas,
and residential water.

**Electricity — feasible.** EIA exposes
`/v2/electricity/retail-sales/data/` with `sectorid=RES` and a
`stateid` facet. Monthly, national or per-state. Needs a new
fetcher (current `fetch_eia` hardcodes the petroleum endpoint). An
hour's worth of work to generalize it: add an
`"eia_endpoint"` param to the config schema, let `_eia_fetch_rows`
switch endpoints based on that param. Cadence: monthly.

**Residential natural gas — feasible.** EIA
`/v2/natural-gas/pri/sum/data/` with `process=PRS` for residential.
Same fetcher generalization covers it.

**Water — not feasible without manual data entry.** No national API.
Water billing is fragmented across ~50,000 US utilities, each with
its own rate schedule. EPA publishes aggregate statistics but nothing
usable per-utility programmatically. Realistic options:

* Manual entry — user types their last bill into a tiny `config/water_rates.json`, we just display it.
* Scrape the local utility site (State College Borough Water Authority for PA) — fragile, breaks when they redesign the page.
* Skip water entirely for now. My recommendation.

**Location coupling.** We already know the user's location from
`bridge/location.py`. A new tracker preference layer (see the
category-preferences idea above) can default `stateid=PA` for the
electricity + natural-gas trackers based on that. If the user travels,
the trackers stay tied to their home state until overridden — this
is a "what do I pay at home" feature, not a local-price-where-I-am
feature.

**Scope guess:** half a day for electricity + natural gas (one new
fetcher, two config entries, one test case per endpoint). Water is
deferred.

---

## Location-aware tracker config (auto-pick PADD / state from home)

(Captured 2026-04-15 while de-hardcoding "University Park" and "PADD 1B"
out of the gas_retail info popover.)

**The idea.** The frontend now derives the PADD region + state list
dynamically from the tracker's *configured* `duoarea` (via a label
lookup) — no more hardcoded "NY, NJ, PA, DE, MD, DC" or "not
University Park specifically". What's still manual is the config
itself: when a user moves states, they have to edit
`trackers.example.json` by hand to switch `duoarea` from `R1Y`
(Central Atlantic) to whatever PADD their new address falls under.

**What's needed.** Let tracker configs use a `${home_state_padd}`
placeholder the fetcher resolves at refresh time:

```json
{
  "id": "gas_retail",
  "params": {"duoarea": "${home_state_padd}", "product": "EPMR"},
  ...
}
```

The trackers preferences layer (just shipped, `home_state: "PA"`)
already has the state; we need:

1. A STATE → PADD lookup table in `bridge/trackers/fetch.py`
   (PA → R1Y, CA → R50 …).
2. The service reads `home_state` from tracker_preferences and
   substitutes `${home_state_padd}` (and `${home_state}` itself for
   BLS state-level series) before calling the fetcher.
3. Tracker `label` also accepts a `${home_state_padd_name}`
   placeholder so the display line stays in sync. Absent a preference,
   fall back to the literal PADD as authored in the config.

**Scope guess:** ~2 hours. Includes tests for substitution +
documentation of the placeholder syntax.

---

## Calendar integration (.ics polling, not OAuth)

(Captured 2026-04-15 alongside the reminders idea below.)

**The idea.** Pull the user's calendar events into the same notification
pipeline `bridge/notify/` already runs. N minutes before each event,
push a Discord / ntfy reminder ("Meeting with X in 15 min"). No write-
back; read-only.

**Why .ics first, not Google API.** Every major calendar (Google,
Apple iCloud, Outlook web, Nextcloud, Fastmail) exposes a per-calendar
*"secret URL"* — a long-tokened HTTPS endpoint serving the calendar in
iCalendar (`.ics`) format. Pasting that URL into config beats the
Google Cloud Console / OAuth client / refresh-token dance by an order
of magnitude. One Python dep (`icalendar`), one new poller, ~half a
day end-to-end.

**Module shape.**
* `bridge/calendar/fetch.py` — httpx GET on the .ics URL, parse with
  `icalendar.Calendar.from_ical`, return list of `(start_utc, summary,
  uid)`.
* `bridge/calendar/store.py` — SQLite table of upcoming events keyed
  by `(uid, start_utc)` so we don't re-notify on every poll. Mirrors
  the cooldown-ledger pattern.
* `bridge/pollers.py` — new `calendar_poller` at 5-15 min cadence;
  scans the next 24 h window, emits `calendar.upcoming` NotifyEvents
  for events crossing their lead-time threshold (default 15 min before).
* Config: `config/calendar.example.yaml` with `lead_time_minutes`,
  list of `.ics` URLs (env-var-only — don't commit secret URLs to
  the file), per-calendar overrides.

**Defer:** Google Calendar API native integration (writes, attachments,
mutual-busy view), CalDAV server, recurrence-rule edge cases that
icalendar doesn't already handle.

---

## Reminders + auto-detection from chat

(Captured 2026-04-15 — pairs with the calendar idea above and the
notifications module that just shipped.)

**The pieces, in order:**

1. **Reminders backend** — `bridge/reminders/`:
   * SQLite store keyed on `(reminder_id, trigger_at_utc)`.
   * Scheduler daemon thread in `bridge/pollers.py` (60 s tick) that
     fetches due rows, calls `notify(NotifyEvent(kind="reminders.fired",
     ...))`, marks `delivered=True` atomically.
   * Manual API: `bridge/reminders/api.py` with `create()`, `cancel()`,
     `list_upcoming()`. Used by the auto-detection layer + by a future
     "/reminder" tool.

2. **Inline UI card** — when a reminder is created (any source), show
   a small card in the chat stream: `⏰ Reminder set: <text> at
   <time> — [undo] [edit]`. Card stays interactive for ~10 s, then
   commits silently. Same-page pattern as the news action popup
   we already built.

3. **Code-based auto-detection** — `bridge/reminders/detect.py`:
   * Regex layer: `(remind me|don't forget|ping me|set a reminder)
     (to|that) <content> (at|on|in) <time>`. Several variants.
   * Time parsing: `dateparser` library handles "5pm tomorrow",
     "in 2 hours", "next Monday morning", "March 15 at 3", etc.
   * Routed BEFORE the LLM in the chat pipeline. On match, calls
     `reminders.create()` and lets the chat turn proceed normally;
     the UI gets the card via the streaming event channel.

4. **LLM fallback for ambiguity** — feature-flagged off by default.
   When regex returns nothing AND the prompt contains a time keyword
   ("tomorrow", "later", time-of-day phrases), one extra LLM call
   asks: *"Does this prompt request a future reminder? If yes,
   return JSON `{trigger_at, message}`. If no, return `null`."*
   Same card UX, same undo path.

**Why hybrid, not pure-code.** The card-with-undo UX makes precision
a non-issue: false positives are visibly shown + one-click cancelable,
false negatives are recoverable by re-asking explicitly. So the LLM
fallback is low-risk *additive* coverage, not a precision crutch.
Matches the design.md rule "code is the trust path, LLM is a
supplement."

**Why the regex layer first.** Most "remind me" phrasings are formulaic
and `dateparser` already handles the time grammar. We can ship #1-#3
without ever calling the LLM, then measure how often a real prompt
falls outside the regex coverage before deciding whether the LLM
fallback earns its keep.

**Effort estimate (whole stack):** ~3 days. #1+#2 alone (~1.5 days)
give a usable manual reminder system; #3 adds the magic; #4 is a
half-day add-on once the framework is proven.

---

## Build-order recommendation

**Already shipped:**
- ~~Notifications~~ — Discord webhook is default, ntfy legacy
- ~~TTS offload to PC~~ — running over Tailscale
- ~~LLM behavior backlog~~ — all 8 items shipped 2026-04-17
- ~~Calendar integration~~ — full stack: ICS fetch + SQLite dedup +
  10-min poller + Discord push. Feature-flagged off; activate by
  copying calendar.example.yaml → calendar.yaml + setting env vars.
- ~~Reminders~~ — backend + regex auto-detection + LLM fallback tool

**Ready to build (no blockers):**

1. **LLM-as-judge eval** (no deps; useful for tracking further changes)
2. **Self-improving classifier** (needs accumulated log data — defer
   a few weeks so there's enough abstain-cases to mine)
3. **Code-aware mode** (no deps; standalone)
