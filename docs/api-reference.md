# API reference

Everything external-facing: the bridge's configuration schema, the main
browser-facing `/api/*` routes, the upstream HTTP endpoints the bridge
talks to, and the systemd unit templates.

## assistant.yaml

Full schema reference for
[config/assistant.yaml](../config/assistant.yaml). All paths are resolved
relative to the repo root (the directory containing `bridge/`), not
relative to the yaml file.

### audio

| Key            | Type       | Default | Notes |
|----------------|------------|---------|-------|
| `sample_rate`  | int        | 16000   | STT/VAD capture rate in Hz. TTS output uses its own rate (32 kHz for v2Pro). |
| `input_device` | str\|null  | null    | Substring match against `sounddevice.query_devices()`. `null` = system default. |
| `output_device`| str\|null  | null    | Same as `input_device`, but for output. |
| `frame_ms`     | int        | 32      | Mic frame size. **Pinned to 32** — Silero VAD 5.x requires exactly 512 samples at 16 kHz. |

### vad

| Key              | Type  | Default | Notes |
|------------------|-------|---------|-------|
| `backend`        | str   | silero  | Only `silero` is supported. |
| `threshold`      | float | 0.5     | Speech probability cutoff (0.0-1.0). Raise for noisier environments. |
| `min_silence_ms` | int   | 700     | Silence duration that ends an utterance. |
| `speech_pad_ms`  | int   | 200     | Pre/post-padding around detected speech. |
| `min_speech_ms`  | int   | 250     | Utterances shorter than this are dropped. |

### stt

| Key            | Type   | Default         | Notes |
|----------------|--------|-----------------|-------|
| `backend`      | str    | faster-whisper  | Only faster-whisper is supported. |
| `model`        | str    | tiny.en         | Any faster-whisper model name. Orin-Nano-safe default (CPU, ~400 MB). Bump to `base.en` (~700 MB) on Orin NX 16 GB or larger; `small.en` / `medium.en` need a bigger GPU. |
| `device`       | str    | cpu             | Checked-in, Orin-Nano-safe default. Use `cuda` only when you have VRAM to spare. |
| `compute_type` | str    | int8            | CPU-safe default. Use `int8_float16` on CUDA Jetson builds, `float16` on larger desktop GPUs. |
| `language`     | str    | en              | Whisper language code. |
| `beam_size`    | int    | 1               | Beam size for decoding. 1 = fastest, 5 = most accurate. |
| `vad_filter`   | bool   | false           | Don't enable — upstream VAD already segmented. |

### llm

| Key             | Type   | Default                                | Notes |
|-----------------|--------|----------------------------------------|-------|
| `backend`       | str    | ollama                                 | `ollama`, `llamacpp`, and `mlc` are supported. `llamacpp`/`mlc` use OpenAI-compatible `/v1/chat/completions`. |
| `base_url`      | str    | http://localhost:11434                 | Ollama HTTP endpoint. |
| `model`         | str    | llama3.1:8b                            | Checked-in fallback default. `deploy/setup.sh` writes `KARIN_LLM_MODEL=karin-tuned:latest` when that production tag is present; operators can still override this per host. |
| `temperature`   | float  | 0.3                                    | Default sampling temperature (fallback if not in `options`). |
| `num_ctx`       | int    | 2048                                   | Default context window (fallback if not in `options`). |
| `options`       | dict   | `{num_predict: 80}`                    | Passed through to the backend's options/body translation layer. Top-level `temperature` / `num_ctx` are merged in if absent. |
| `system_prompt` | str    | (see config)                           | System message prepended to every chat turn. |
| `under_fire_rescue` | bool | true | When the LoRA under-fires on a tool-worthy prompt + the classifier has a confident hint, force-execute the hinted tool with best-effort args. One rescue per turn. See [routing-pipeline.md § L6](routing-pipeline.md). |
| `two_phase_compose` | bool | true | On tool turns, run a second LLM call with only user prompt + scrubbed tool outputs (no tool schema) to produce the reply. Adds +5-10 s/turn for big reply-quality win. Runtime override via `POST /api/settings {"two_phase_compose": bool}` — does not persist. |
| `hint_in_user_msg` | bool | true | Move the per-turn routing classifier hint from the end of the system prompt to the start of the user message. Keeps the system prompt byte-stable for Ollama KV-cache reuse. Shipped 2026-04-22; validated +2-7 pp routing/reply-pass on the 135-case eval. |

### tts

| Key                    | Type   | Default                                | Notes |
|------------------------|--------|----------------------------------------|-------|
| `backend`              | str    | gpt-sovits                             | Only gpt-sovits is supported. |
| `base_url`             | str    | http://localhost:9880                  | GPT-SoVITS api_v2.py HTTP endpoint. |
| `endpoint`             | str    | /tts                                   | Synthesis route path. |
| `voice`                | str    | general                                | Active character/voice bundle name under `characters/`. |
| `voice_dir`            | str\|null | null                                | Optional discovery root override. Leave unset for `characters/*/voices/`; set only for legacy flat-dir layouts. |
| `ref_audio_path`       | str    | ../../characters/general/voices/ref.wav | Fallback path to the reference clip when voice auto-discovery is unavailable. |
| `prompt_text`          | str    | ""                                     | Transcript of the reference clip. Empty = server infers. |
| `prompt_lang`          | str    | ko                                     | Language of the reference clip. |
| `text_lang`            | str    | en                                     | Language of text to synthesize. Must be in `tts_config.languages`. |
| `gpt_weights_path`     | str    | ../../characters/general/voices/gpt_model_32.ckpt | Fallback GPT weights path pushed via `/set_gpt_weights` at startup. |
| `sovits_weights_path`  | str    | ../../characters/general/voices/sovits_model_16.pth | Fallback SoVITS weights path pushed via `/set_sovits_weights` at startup. |
| `top_k`                | int    | 5                                      | Sampler top-k. |
| `top_p`                | float  | 1.0                                    | Sampler top-p. |
| `temperature`          | float  | 0.8                                    | Sampler temperature. |
| `speed_factor`         | float  | 1.0                                    | Playback speed multiplier. |
| `streaming_mode`       | bool   | false                                  | Kept for config compatibility. Browser streaming uses the bridge's `synthesize_stream()` path rather than treating this flag as the feature switch. |
| `request_timeout`      | float  | 300                                    | Per-request timeout for the GPT-SoVITS HTTP client. |

### runtime

| Key                   | Type   | Default | Notes |
|-----------------------|--------|---------|-------|
| `log_level`           | str    | INFO    | DEBUG / INFO / WARNING / ERROR. |
| `max_utterance_ms`    | int    | 15000   | Hard cap per-turn recording length. |
| `interrupt_on_speech` | bool   | false   | **Not implemented.** Set to true logs a warning and runs half-duplex anyway. |

## Browser `/api/*` routes

These are the app routes the shipped FastAPI UI uses. They sit
alongside the upstream service calls documented later in this file.

### Core UI and helper routes

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/health` | Lightweight readiness + active STT/LLM/TTS model snapshot. |
| `GET` | `/api/features` | Full feature registry snapshot, including tool denylist and runtime `two_phase_compose`. |
| `GET` | `/api/capabilities` | Legacy STT/TTS-only snapshot for older clients. |
| `POST` | `/api/settings` | Runtime settings patching (`two_phase_compose` today). |
| `GET` | `/api/active-character` | Current character metadata for theme/avatar selection. |
| `GET` | `/api/characters/available` | Discover switchable character bundles. |
| `GET` | `/api/characters/{char_name}/expressions/{filename}` | Serve character expression images. |
| `GET` | `/api/audio/{conversation_id}/{audio_id}` | Replay cached reply WAV for a past turn. |
| `GET` | `/api/weather`, `/api/convert`, `/api/math`, `/api/circuit`, `/api/graph`, `/api/find-places`, `/api/web-search` | Structured helper endpoints used by inline widgets without going back through the LLM. |

### Turn execution routes

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/api/turn` | One-shot voice turn: upload audio, run STT → LLM → TTS, return JSON with transcript/reply/base64 WAV. |
| `POST` | `/api/text-turn` | One-shot text turn: LLM (+ TTS when enabled), same reply shape as `/api/turn`. **Repeat-prompt dedup** — if the same normalized text arrives within 30 s and isn't time-sensitive (weather/news/time/etc.), the cached response is returned without re-running the LLM. Bounded LRU (32 entries). Catches double-clicked sends + page-reload retries. |
| `POST` | `/api/turn-stream` | Streaming voice turn: NDJSON stream of tool calls, transcript, audio chunks, and done/error events. |
| `POST` | `/api/text-turn-stream` | Streaming text turn: same NDJSON event shape without audio upload/STT. |
| `POST` | `/api/turn/start` | Background text turn enqueue; returns `job_id` immediately. |
| `GET` | `/api/turn/{job_id}/stream` | Replay + live-tail the persisted NDJSON event stream for a background job. |
| `GET` | `/api/turn/{job_id}/status` | Poll background job status without streaming. |

### Conversation, memory, and feedback routes

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/history` | Restore the current conversation, including tool events and cached `audio_id`s. |
| `POST` | `/api/history/new` | Archive the current conversation and start a fresh one. |
| `DELETE` | `/api/history/{cid}` | Delete a conversation and switch current history if needed. |
| `GET` | `/api/history/list` | List saved conversations + current conversation id. |
| `GET` | `/api/history/{cid}/notes` | Turn-level routing/debug notes for a conversation. |
| `POST` | `/api/history/switch` | Switch the active conversation. |
| `GET` | `/api/memory` | Read user + agent long-term memory. |
| `PUT` | `/api/memory` | Overwrite user and/or agent memory. |
| `POST` | `/api/feedback` | Apply thumbs-up/down feedback to a past turn. |
| `GET` | `/api/feedback/stats` | Feedback aggregates for debugging. |
| `POST` | `/api/feedback/reset` | Clear the feedback log. |

### Voice, holiday, panels, and profile routes

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/voice/status` | Report whether the voice backend is disabled, stopped, or running. |
| `POST` | `/api/voice/start`, `/api/voice/stop` | Start/stop the SoVITS container from the UI. |
| `GET` | `/api/holiday/today`, `/api/holiday/today/audio` | Holiday banner metadata + optional synthesized greeting audio. |
| `GET` | `/api/alerts/active`, `/api/news/briefs`, `/api/news/cluster/{cluster_id}`, `/api/digest/today`, `/api/news/graph`, `/api/news/learned-keywords`, `/api/trackers/catalog`, `/api/trackers/snapshots`, `/api/trackers/snapshot` | Data sources for the Alerts, News, Digest, and Trackers panels. |
| `GET`/`POST` | `/api/preferences/news`, `/api/preferences/trackers`, `/api/preferences/notify` | Read/write news, tracker, and notification settings. |
| `GET`/`POST` | `/api/preferences/location` | Read/write user_location (city / region / country / lat / lon). Empty POST clears, alerts proximity scoring picks up the new value on next refresh. |
| `GET`/`POST` | `/api/preferences/api-keys` | Read/write per-profile data-source API keys. GET returns metadata only (label, purpose, register URL, `set` boolean, `source` ∈ `environment`/`preferences`/`none`, last-4 hint). POST accepts `{data: {<name>: <plaintext>}}` — empty string clears. **Never echoes plaintext.** |
| `GET` | `/api/county/metrics` | List metric registry: `available` (file on disk) vs `missing` (registered but not yet fetched). |
| `GET` | `/api/county?metric=<key>&year=<yyyy>` | Per-county values for one metric/year + summary stats (min, max, mean, median, std, count) + interpretation string. |
| `GET` | `/api/county/compare?a=<key>&b=<key>&year=<yyyy>` | Pairwise Pearson r between two metrics across counties that have both. Returns scatter pairs + r + a "correlation, not causation" note baked into the interpretation field. |
| `GET` | `/api/county/drill/{fips}` | All available metrics for a single 5-digit FIPS county + national-distribution percentile rank per metric. |
| `GET` | `/api/map?year=<yyyy>&metric=<key>` | Map panel state-level data; returns `values_by_fips` for the choropleth + stats + interpretation. |
| `GET` | `/api/map/regions` | Region overlay metadata (Rust Belt / Sun Belt / etc. with state-code member lists + signature colors). |
| `GET` | `/api/map/state/{code}` | Per-state profile card (top industries, resources, region memberships, narrative). |
| `GET` | `/api/facts?year=<yyyy>[&region=<key>]` | Year-card aggregator: world population + US inflation baseline + wages + item prices + Wikipedia snippet, each with a then-vs-now comparison. |
| `GET` | `/api/population?region=<key>` | World Bank `SP.POP.TOTL` time series for a single region. |
| `GET` | `/api/inflation?from=<yyyy>&to=<yyyy>[&regions=<csv>]` | CPI-based purchasing-power equivalence; multi-region overlay when `regions=` is set. |
| `GET` | `/api/alice?year=<yyyy>&composition=<key>` | ALICE share estimator with the survival-budget breakdown + tax components. |
| `GET` | `/api/diagnostics/data-sources` | Aggregated `_last_fetch_status.json` across data dirs (Tier A resilience). |
| `GET`/`POST` | `/api/profiles`, `/api/profiles/active` | List, create, and switch user profiles. |
| `GET`/`POST` | `/api/profiles/routing`, `/api/profiles/routing/peers` | IP-to-profile routing config and Tailscale peer discovery. |
| `GET` | `/api/tts/voices` | Discover available character voices and the active one. |
| `POST` | `/api/tts/voice` | Hot-swap the active character — swaps voice weights (when the target has a bundle), persona (re-renders `{persona}`/`{language_note}` from the target's `voice.yaml`), and persists via `os.environ["KARIN_CHARACTER"]` so reloads pick up the new selection. Voice-less characters (`has_voice=false`, e.g. the shipped `default`) still swap persona; the weight step is skipped. |
| `GET`/`POST` | `/api/stt/status`, `/api/stt/enable`, `/api/stt/disable` | Inspect + runtime-toggle STT. `status` reports `boot_enabled`, `runtime_enabled`, `backend` (`local`/`remote`), `remote_url`, and `remote_reachable` (live 2 s probe of the sidecar when `KARIN_STT_BASE_URL` is set). `enable`/`disable` flip the runtime gate without unloading the in-process Whisper model. |
| `GET` | `/api/voice/status` | TTS-side parallel — reports local sovits container state, or probes the remote sidecar and returns `remote_reachable` when `KARIN_TTS_BASE_URL` is set. |
| `GET` | `/api/reminders/upcoming` | List pending reminders. |
| `POST` | `/api/reminders/{reminder_id}/cancel` | Cancel a pending reminder. |

### Standalone panel pages (`/ui/*`)

These are HTML pages that mount a single panel widget without the chat
chrome. Used for direct linking (e.g. the Facts tab in the bottom nav)
and inside iframes elsewhere in the UI. Each loads
`web/static/_theme_sync.js` so the page mirrors the parent's
`--tint-rgb` and `sun-mode` class.

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/ui/alerts` | Alerts panel page (proximity-scored + threat tier). |
| `GET` | `/ui/news` | News panel page (clusters + entity graph link). |
| `GET` | `/ui/trackers` | Tracker grid page (markets, FX, food, gas). |
| `GET` | `/ui/digest` | Daily digest page. |
| `GET` | `/ui/settings` | Settings panel (preferences, profiles, API keys, location). |
| `GET` | `/ui/facts` | Year-card page with year/region picker and quick-pick chips. |
| `GET` | `/ui/graph` | Standalone graph panel (Plotly). |
| `GET` | `/ui/alice` | ALICE-share estimator page. |
| `GET` | `/ui/map` | US state-level choropleth + region overlays + drill-in. |

### Dev / eval surface

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/api/chat/stream` | Standalone NDJSON chat-only stream. **Not** used by the shipped UI; kept for eval scripts (`sft/_endpoint_battery.sh`) and external tooling that wants tool-call + reply events without the audio stack. Uses a separate `OllamaLLM` singleton so importing the panels API doesn't drag Whisper/SoVITS into memory. |

## GPT-SoVITS api_v2.py endpoints

The bridge talks to an unmodified upstream
[GPT-SoVITS api_v2.py](https://github.com/RVC-Boss/GPT-SoVITS/blob/2d9193b0d3c0eae0c3a14d8c68a839f1bae157dc/api_v2.py)
server, pinned to commit `2d9193b0`. See [bridge/tts.py](../bridge/tts.py)
for the client.

### `POST /tts`

Request body (JSON):

```json
{
  "text": "Text to synthesize",
  "text_lang": "en",
  "ref_audio_path": "/abs/path/to/ref.wav",
  "prompt_text": "",
  "prompt_lang": "ja",
  "top_k": 5,
  "top_p": 1.0,
  "temperature": 1.0,
  "speed_factor": 1.0,
  "media_type": "wav",
  "streaming_mode": false
}
```

Required fields: `text`, `text_lang`, `ref_audio_path`, `prompt_lang`.
Everything else has server-side defaults.

Response on success: HTTP 200, `content-type: audio/wav`, body is a full
WAV blob with header (sample rate in the header, typically 32 kHz for
v2Pro).

Response on error: HTTP 400, JSON body like:

```json
{
  "message": "text_lang: zzz is not supported in version v2Pro",
  "Exception": "..."
}
```

### `GET /set_gpt_weights?weights_path=<abs>`

Loads a GPT checkpoint into the running pipeline. Returns 200 + `"success"`
on success, 400 + JSON error on failure. The bridge calls this once at
startup with `tts.gpt_weights_path` from the config.

### `GET /set_sovits_weights?weights_path=<abs>`

Same as above, for SoVITS weights.

### `GET /docs`

FastAPI's built-in Swagger UI. Used by
[deploy/wait-for-tts.sh](../deploy/wait-for-tts.sh) as a readiness probe
— it responds with 200 as soon as Uvicorn is accepting connections,
which happens AFTER the synchronous TTS pipeline construction. So a 200
on `/docs` means both "HTTP up" and "pipeline loaded."

## Ollama `/api/chat`

The bridge talks to Ollama via the non-streaming chat endpoint. See
[bridge/llm.py](../bridge/llm.py) for the client.

### Request

```json
{
  "model": "karin-tuned:latest",
  "messages": [
    {"role": "system", "content": "You are Karin..."},
    {"role": "user", "content": "Hello there"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "What's the weather?"}
  ],
  "stream": false,
  "options": {
    "num_predict": 80,
    "temperature": 0.3,
    "num_ctx": 2048
  },
  "keep_alive": -1
}
```

The bridge maintains a rolling `_history` list and prepends the system
prompt to every request. History is appended **only after** a successful
response — a failed POST does not mutate state.

### Response

Non-streaming response shape:

```json
{
  "model": "...",
  "created_at": "2026-04-11T...",
  "message": {
    "role": "assistant",
    "content": "The text you want."
  },
  "done": true,
  "total_duration": "...",
  "...": "..."
}
```

Only `message.content` is consumed by the bridge.

### Thinking mode

When you run a Qwen 3 model, you can put `think: "off"` (or `false`) in
`llm.options` in `assistant.yaml`. The bridge hoists that value to the
top-level Ollama request field that newer Ollama versions expect.

Without it, replies look like:

```
<think>
Let me think about this...
</think>
Actually, here's my answer.
```

Which the TTS would synthesize verbatim.

## Systemd unit templates

Both `.service` files in [deploy/](../deploy/) are templates with
placeholders that `setup.sh` substitutes via `sed` at install time:

- `@REPO_ROOT@` — absolute path to the repo root on the Jetson
- `@USER@` — the user running setup.sh (from `id -un`)

### sovits-server.service

Runs upstream's unmodified `api_v2.py` inside `.venv/tts-server/`.

```ini
[Unit]
Description=GPT-SoVITS api_v2.py inference server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=@USER@
WorkingDirectory=@REPO_ROOT@/third_party/GPT-SoVITS
Environment="PYTHONUNBUFFERED=1"
Environment="PATH=@REPO_ROOT@/.venv/tts-server/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=@REPO_ROOT@/.venv/tts-server/bin/python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
Restart=on-failure
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

Key points:

- `Type=simple` means systemd considers the service "started" as soon
  as the process forks, which is BEFORE Uvicorn is accepting
  connections. The bridge waits for real readiness via
  `wait-for-tts.sh` in its own `ExecStartPre`.
- `TimeoutStartSec=300` gives first-boot model loading (up to 60s +
  weight swap overhead) plenty of headroom.
- `WorkingDirectory` is important — `api_v2.py` reads paths relative to
  its CWD.

### assistant.service

Runs `bridge.main` inside `.venv/bridge/`.

```ini
[Unit]
Description=Karin voice bridge
After=network-online.target ollama.service sovits-server.service
Wants=network-online.target
Requires=ollama.service sovits-server.service

[Service]
Type=simple
User=@USER@
WorkingDirectory=@REPO_ROOT@
Environment="PYTHONUNBUFFERED=1"
Environment="PATH=@REPO_ROOT@/.venv/bridge/bin:/usr/local/bin:/usr/bin:/bin"
ExecStartPre=/bin/bash @REPO_ROOT@/deploy/wait-for-tts.sh http://127.0.0.1:9880 120
ExecStart=@REPO_ROOT@/.venv/bridge/bin/python -m bridge.main --config @REPO_ROOT@/config/assistant.yaml
Restart=on-failure
RestartSec=15
TimeoutStartSec=180

[Install]
WantedBy=multi-user.target
```

Key points:

- `Requires=` (not `Wants=`) means the bridge is stopped if either dep
  dies. This is intentional — a voice loop without Ollama or TTS is
  useless.
- `ExecStartPre` blocks until `/docs` returns 200 on the TTS server, so
  the bridge's first `/set_gpt_weights` call doesn't race the server.
- `TimeoutStartSec=180` accounts for `ExecStartPre` (up to 120s) plus
  model loading in the bridge itself.
