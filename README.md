# Karin

A self-hosted voice assistant that runs on an NVIDIA Jetson Orin Nano 8 GB —
microphone in, locally-inferenced reply out, all on-device, reachable from
anywhere via Tailscale. Also usable from any browser on your tailnet.

Karin speaks in a cloned voice trained with GPT-SoVITS (multi-voice,
runtime-switchable), uses an Ollama-backed LLM for the chat/tool turn
(production model: `karin-tuned:latest` — an iter-3 LoRA on
`mannix/llama3.1-8b-abliterated:tools-q4_k_m`; the checked-in
`config/assistant.yaml` default is the base `llama3.1:8b` for reference,
and pre-LoRA alternates like `qwen2.5:3b` are supported via
`KARIN_LLM_MODEL` — see [RUNBOOK § Hardware sizing](RUNBOOK.md#hardware-sizing-which-model-fits)),
and uses faster-whisper for speech recognition.
No cloud APIs in the voice path. Optional open-web lookups go through
Wikipedia / DuckDuckGo / Open-Meteo / ipwho.is — no paid keys.

## Contents

- [Two ways to use it](#two-ways-to-use-it)
- [Tools available to the LLM](#tools-available-to-the-llm)
- [Key features](#key-features)
- [System requirements](#system-requirements)
- [Quick start (Jetson)](#quick-start-jetson)
- [Quick start (PC, for development)](#quick-start-pc-for-development)
- [Voice subsystem](#voice-subsystem--three-paths-default-is-off)
- [Training your own voice](#training-your-own-voice)
- [Hardware](#hardware)
- [Repository layout](#repository-layout)
- [Credits](#credits)
- [License](#license)

More docs in **[docs/](docs/)**. Start with
[docs/system-overview.md](docs/system-overview.md) for the technical map
and [docs/README.md](docs/README.md) for the full index.

> **Note — current default is text-only.** Voice (STT + TTS) is
> gated behind `KARIN_STT_ENABLED` / `KARIN_TTS_ENABLED` env flags,
> both defaulting to `false` so the LLM has room to reason over
> longer tool chains. See
> [RUNBOOK.md § Voice feature flags](RUNBOOK.md#voice-feature-flags-stt--tts) for
> the flip-back procedure.

## Two ways to use it

**Voice loop** (the original use case)
```
[Mic / Phone WebUI]
        │
        ▼
   VAD  (silero-vad)                        CPU, tiny
        │
        ▼
   STT  (faster-whisper tiny.en, CPU)       ~400 MB RAM, opt-in
        │
        ▼
   LLM  (Ollama model from config / .env)   shared VRAM      ← calls tools
        │                                                  ↕
        ▼                                  { weather, news, math, circuit,
   TTS  (GPT-SoVITS v2Pro, PC offload)       wiki, web_search, memory, ... }
        │
        ▼
   [Speaker]

Total VRAM on Jetson: ~5.5 GB of 8 GB shared (92% GPU)
TTS runs on PC via Tailscale — zero Jetson VRAM for voice synthesis
```

**Browser UI** (push-to-talk or text chat)
- FastAPI app on `:80` with per-turn panels — each tool call opens a
  collapsible widget inline: weather card, news list, tracker grid,
  Wikipedia summary, KaTeX-rendered math, Plotly-rendered graphs, and
  more.
- Persistent conversation history with automatic token-budget
  compaction, left sidebar for switching conversations, and
  user + agent memory that auto-learns from conversation.
- Settings panel with news watchlists, tracker preferences,
  notification channels, profile management, IP routing, and memory.

## Tools available to the LLM

21 tool schemas — time, weather, news, alerts, digest, trackers, math,
circuit analysis, graphing, unit conversion, Wikipedia, web search,
places, reminders, auto-memory, `say` (repeat-back), `inflation`
(historical purchasing power via BLS CPI-U + multi-region series),
`population` (World Bank time series), `facts` (year-card aggregator
with then-vs-now comparisons), `analyze` (peak/trough/trend/percentile
on time series), and `alice` (US ALICE-share estimator with bracket-
based federal tax). Schemas live in
[bridge/tools/_schemas.py](bridge/tools/_schemas.py), runtime
dispatch lives in [bridge/tools/_dispatch.py](bridge/tools/_dispatch.py),
and 20 tools are active by default (`schedule_reminder` is feature-gated).

**Layered routing + reply pipeline** (L1–L8 — see [docs/routing-pipeline.md](docs/routing-pipeline.md) for the full diagram):
1. **Chitchat guard** — "hi"/"thanks"/"cool" strips tools before the LLM call.
2. **Classifier stack** — regex pre-classifier (data-driven `routing_patterns` per tool) + spaCy negation veto + continuation fallback + embedding fallback.
3. **Tool narrowing** — classifier hint shrinks the tool list so the LLM can't pick the wrong one.
4. **LoRA call** — picks from the narrowed set.
5. **Leak recovery** — recovers JSON-in-`content` emissions back into structured tool calls.
6. **Under-fire rescue** — when the LoRA abstains on a clearly-tool-worthy prompt, force-execute the classifier's hinted tool with best-effort args.
7. **Two-phase compose** — on tool turns, run a second LLM call with only user prompt + scrubbed tool outputs (no tool schema), so the reply is grounded rather than schema-leaky.
8. **Reply scrubs** — `_clean_reply` strips JSON stubs, prompt-leak markers, fabricated market quotes, capability disavowals, and markdown noise.

Current prod baseline on the 135-case eval: **93.3% routing / 91.9% reply-pass / 59.2% tool-output usage** (iter-3 LoRA + full L1–L8 stack, shipped 2026-04-22).

Multi-step tool calls are supported (look up a fact → `math` on it → reply), with safety nets:

- **Per-tool cap (2 calls/turn)** prevents runaway retries.
- **Same-(name, args) suppression** stops exact duplicate calls.
- **Forced-final reply** on cap exhaustion.
- **Tool result sanitization** — strips prompt-injection payloads from external content before feeding back to the LLM.
- **Auto-retry on Ollama 5xx** (one extra attempt with 3 s delay).

See **[docs/tools.md](docs/tools.md)** for the full 21-tool catalog (20 active by default) and the recipe for adding a new tool, or **[docs/routing-pipeline.md](docs/routing-pipeline.md)** for the full pipeline rationale.

## Key features

### Profile isolation (Phase H)
Multiple named profiles on one device, each with independent
watchlists, reminders, alert cooldowns, conversations, memory, and
user location. Tailscale IP routing auto-switches profiles per
device. Migration from legacy paths is automatic on first boot.

### Threat assessment (Phase G)
Rule-based proximity scorer (NWS weather, travel advisories, news
watchlist matches, tracker shocks) with optional LLM verifier for
borderline scores (±1 tier clamp, citation required, 7-day cache).
Threat badges on the Alerts panel with configurable dim-threshold.

### Multi-character / multi-voice
- **Per-character folders** under `characters/<name>/`. Each carries its own voice models (`voices/`), persona (`voice.yaml`), and optional face definition (`face.json`). The sidebar dropdown `🎭 Character` picks between them; choosing one hot-swaps the TTS voice via `POST /api/tts/voice`, updates the face renderer, and reloads the UI theme.
- **Face rendering is code- or image-based per character.** With a `face.json` declaring `type: "procedural-sun"`, the face is an inline SVG sun with a mouth ellipse that morphs per vowel (colors, mouth geometry, theme, and speaking-aura spec all live in the JSON). Without `face.json`, the legacy PNG-per-vowel expression swap is used, reading from `characters/<name>/expressions/`. No JS edits needed to add a new character.
- **Auto-discovered voices** — `bridge/tts_voices.py::discover_voices()` scans `characters/*/voices/` for `ref.wav`, `gpt_model*.ckpt`, `sovits_model*.pth`. Legacy flat-dir layouts (`voice_training/`) still work as fallback.

### Auto-memory
The `update_memory` tool lets the LLM silently save facts about the
user (name, location, preferences) during conversation. Stored in
the profile's memory, editable from the Settings page. Both the LLM
and the user can read and write.

### Security
- **IP whitelist** — configurable: `tailscale` (default), `off`, or
  custom CIDR list. Non-tailnet IPs get 403.
- **Tool result sanitization** — strips prompt-injection patterns
  from external content before feeding to the LLM.
- **Sympy DoS protection** — expression complexity cap + 10s timeout.
- **Profile path traversal protection** — name validation rejects
  `../`, `\`, and other filesystem-escape characters.

### News subsystem
RSS feeds with lexical clustering, optional LLM cross-verification
for borderline duplicates, LLM-assisted keyword learning (Phase E),
translation to target language, and a D3.js entity relation graph.

### Map + county-overlay analytics
US state-level choropleth at `/ui/map` (D3 + us-atlas Albers
TopoJSON, ~115 KB vendored). Click a state to drill in;
historical/economic region overlays (Rust Belt, Sun Belt, Tech
Corridors, etc.) with per-region color highlights; mouse-wheel
zoom + drag pan + auto-zoom-to-fit on click. Click-to-emit
`karin:focus-region` postMessage hooks the alerts proximity scoring
and other panels into cross-frame navigation. County-overlay
analytical surface (`/api/county/*`) reads per-metric JSON files
under `bridge/data/county/` for descriptive correlation analysis
across demographics / housing / health / crime / economic shares
— **descriptive, not causal**, with the disclaimer baked into the
tool layer alongside Pearson r values. See [docs/county-overlay-plan.md](docs/county-overlay-plan.md).

### Data source API keys (Settings UI)
Optional + required upstream keys (CDC Socrata, Census ACS, FBI
Crime Data Explorer) can be saved per-profile via the Settings panel
under "Data source API keys". Three-source resolution: env var
(`KARIN_*`) wins, then `preferences.json` `api_keys` block, then
fallback to no-key behavior. The Settings UI never echoes plaintext
— only last-4-character hints. Populator scripts (`scripts/fetch_*.py`)
accept `--from-prefs` to pick up the saved value.

### Feature flags
All subsystems toggle via `config/features.yaml` + env vars. STT,
TTS, bandit, holidays, news wires, translation, keyword learning,
threat LLM, notifications, calendar, reminders — each independently
on/off without code changes.

## System requirements

| Component | Jetson (production) | PC (development) |
|---|---|---|
| Python | 3.10 (JetPack default) | **3.12** |
| Torch | NVIDIA JetPack wheel (CUDA) | CUDA 12.x |
| CUDA | Bundled with JetPack 6.x | 12.4+ |
| ffmpeg | apt `ffmpeg` | `winget install Gyan.FFmpeg.Shared` |
| OS | Ubuntu 22.04 (JetPack) | Windows 11 / Linux |
| Ollama | Installed by `deploy/setup.sh` | [ollama.com](https://ollama.com) |

## Quick start (Jetson)

Karin ships **text-only by default**. Voice (STT + TTS) is optional and
requires you to train a GPT-SoVITS bundle for your own voice — see
[Add a voice (optional)](#add-a-voice-optional) below.

```bash
git clone https://github.com/kaminglui/Karin.git
cd Karin

# One-shot provisioning. Installs apt prereqs, Ollama + LLM models,
# Python venvs, systemd units, and Tailscale Serve HTTPS fronting.
# Idempotent — safe to re-run.
sudo -v
bash deploy/setup.sh
```

Karin is now at `https://<your-jetson>.<your-tailnet>.ts.net/`.
The Tailscale Serve step is idempotent — re-run it any time you rename
the device. See
[docs/jetson-setup.md § 6](docs/jetson-setup.md#6-later-enable-https-fronting-for-the-browser-ui)
and [RUNBOOK § Tailscale HTTPS](RUNBOOK.md#tailscale-https-port-443-on-port-80-off)
for details.

If you prefer Docker Compose, the stack is also containerized:

```bash
cp .env.example .env
# Uncomment the Jetson COMPOSE_FILE line in .env
cd deploy && docker compose up -d
```

See [docs/jetson-setup.md](docs/jetson-setup.md) for the full
end-to-end walkthrough.

## Add a voice (optional)

Text is the default because GPT-SoVITS fine-tuning is a separate ~hour
of GPU work and needs voice samples you supply. Even once you have a
trained bundle, you get to choose **where** sovits actually runs:

- **PC-TTS offload (recommended for Orin Nano 8 GB)** — run GPT-SoVITS
  on a beefier PC over Tailscale. The Jetson just sends text + gets WAV.
  Keeps voice VRAM off the Jetson entirely. See
  [deploy/pc-tts/README.md](deploy/pc-tts/README.md).
- **Local sovits (Orin NX 16 GB or larger)** — run the whole stack on
  the Jetson. LLM + sovits co-loaded. Opt-in via the `LOCAL_SOVITS=yes`
  env flag since the combined memory footprint overflows the Orin Nano's
  7.4 GiB usable pool (measured: karin-tuned alone holds ~6.8 GiB at
  idle, leaving ~600 MiB headroom). Do **not** enable this on Orin
  Nano 8 GB.

When you're ready:

1. Train a voice per [docs/training-guide.md](docs/training-guide.md).
   You'll end up with three files per character:
   `ref.wav`, `gpt_model_*.ckpt`, `sovits_model_*.pth`.
2. Drop them into `characters/<name>/voices/` on the Jetson (for PC-TTS,
   drop them on the PC instead — the Jetson only needs the bundle where
   sovits actually runs).
3. Re-run provisioning:
   - **PC-TTS:** set `KARIN_TTS_BASE_URL=http://<pc-tailscale-ip>:9880`
     and `KARIN_TTS_ENABLED=true` in `deploy/.env`, then
     `cd deploy && docker compose up -d` (or `sudo systemctl restart web`).
   - **Local sovits:** `LOCAL_SOVITS=yes bash deploy/setup.sh` — detects
     the bundle, clones GPT-SoVITS, downloads its ~4 GB upstream
     pretrained weights, installs `sovits-server.service` +
     `assistant.service`, runs a TTS smoke test, and brings up the
     voice loop.

See [characters/README.md](characters/README.md) for the folder layout
and [docs/training-guide.md](docs/training-guide.md) for the training
workflow. Voice is also toggleable at runtime via `KARIN_TTS_ENABLED`
and the "Voice" button in the web UI.

## Quick start (PC, for development)

```bash
py -3.12 -m venv .venv\bridge
.venv\bridge\Scripts\activate
pip install -r bridge/requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

uvicorn web.server:app --host 127.0.0.1 --port 8001
```

Open `http://localhost:8001`.

## Voice subsystem — three paths, default is off

Karin ships with STT + TTS **both disabled by default**. Pick a path
that matches your hardware. Setup auto-detects unreachable PCs and
surfaces the state in the sidebar indicators.

### Path 1 — text-only (default)

Nothing to configure. The web UI runs over chat. No microphone, no
spoken replies. TTS dot stays "off", STT dot stays "off".

### Path 2 — local STT on the Jetson (opt-in at setup)

Enables in-process `faster-whisper tiny.en` on CPU (~400 MB RAM,
~200-500 ms latency on the idle Orin Nano cores):

```bash
LOCAL_STT=yes bash deploy/setup.sh
```

Writes `KARIN_STT_ENABLED=true` + `KARIN_STT_MODEL=tiny.en` +
`KARIN_STT_DEVICE=cpu` + `KARIN_STT_COMPUTE_TYPE=int8` to `deploy/.env`.
TTS stays off (there's no room for both on the Jetson).

Bigger Whisper (`base.en`, `small.en`) is supported but **not**
recommended on Orin Nano 8 GB — it eats the remaining ~600 MiB
headroom next to `karin-tuned`. If you have Orin NX 16 GB or larger,
edit `KARIN_STT_MODEL` in `deploy/.env` after the initial install.

### Path 3 — offload both to a PC (recommended when accuracy matters)

One sidecar on a PC runs both GPT-SoVITS (TTS) and faster-whisper
(STT). Jetson memory stays at the text-only baseline (~6.8 GiB idle).

```bash
# On the PC (Windows):
cd deploy/pc-tts
setup.bat install    # venv + deps (TTS + STT) + auto-start on login
start.bat            # tray icon / --visible for console
```

On the Jetson, set in `deploy/.env`:
```
KARIN_TTS_ENABLED=true
KARIN_TTS_BASE_URL=http://<pc-tailscale-ip>:9880
KARIN_STT_ENABLED=true
KARIN_STT_BASE_URL=http://<pc-tailscale-ip>:9880     # same sidecar serves both
```

Then re-run `bash deploy/setup.sh` — it will **probe both URLs** and
warn if the PC isn't reachable (sidecar down, Tailscale disconnected,
wrong IP). The web UI also surfaces an "offline" state in the sidebar
STT/TTS dots when the remote sidecar stops responding, so you see it
live rather than on the next failed turn.

With STT on the PC you can run `small.en`, `medium.en`, or `large-v3`
for free (Whisper ~2-4% WER vs the Jetson-local `tiny.en` ~10% WER)
without any Jetson memory hit. Latency adds ~50-150 ms round-trip on
a Tailscale LAN.

See [deploy/pc-tts/README.md](deploy/pc-tts/README.md).

## Training your own voice

See [docs/training-guide.md](docs/training-guide.md) for the
target-voice dataset prep and the GPT-SoVITS workflow that produces
the model triplet.

## Hardware

- **Compute:** NVIDIA Jetson Orin Nano 8 GB (ARM64/JetPack)
- **Audio in:** directional microphone (USB or I2S) / browser mic
- **Audio out:** speaker / browser audio
- **Network:** Ethernet or Wi-Fi + [Tailscale](https://tailscale.com/)

### LLM model sizing

| Hardware | Recommended model | Notes |
|---|---|---|
| Jetson Orin Nano 8 GB | **`karin-tuned:latest`** (iter-3 LoRA on mannix llama3.1-8B-abliterated Q4_K_M, ~4.9 GB) | Production model — 93.3% routing / 91.9% reply on 135-case eval. TTS offloaded to a PC over Tailscale so the LLM gets the full VRAM budget. `qwen2.5:3b` is a pre-LoRA fallback when karin-tuned is unavailable. |
| Jetson Orin NX 16 GB | `karin-tuned:latest` (same) | TTS can run co-located via `LOCAL_SOVITS=yes bash deploy/setup.sh`. `huihui_ai/qwen3.5-abliterated:4b` is an untuned alternative. |
| Desktop NVIDIA (≥12 GB) | `karin-tuned:latest` or a 7B+ base model of your choice | Use the `gpu` compose overlay. |

## Repository layout

```
Karin/
├── bridge/                  # Core Python runtime: LLM, tools, routing, state
│   ├── llm.py               # Ollama/OpenAI-compatible client + tool loop
│   ├── tools/               # Tool schemas, dispatch, and implementations
│   ├── routing/             # Classifier, vetoes, rescue, continuations
│   ├── news/ alerts/        # Background information subsystems
│   ├── trackers/ reminders/ # User-facing tool subsystems
│   ├── profiles/            # Profile isolation and device routing
│   └── tests/               # Backend tests
├── web/                     # FastAPI app plus browser UI assets
│   ├── server.py            # Main API runtime and middleware
│   ├── panels_api.py        # Supporting panel/chat/settings endpoints
│   └── static/              # Browser UI
├── characters/              # Character metadata, faces, and private voice bundles
├── config/                  # Checked-in defaults and optional local templates
├── deploy/                  # Docker Compose, systemd templates, setup scripts
│   └── pc-tts/              # Windows PC sidecar for TTS/STT offload
├── sft/                     # Routing fine-tune data/build/eval workflow
├── docs/                    # Current docs plus historical analysis
│   └── evals/               # Raw eval artifacts and large reference tables
└── data/                    # Runtime state; gitignored
```

## Credits

- **[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)** — RVC-Boss et al.
  Voice cloning and TTS inference.
- **[Ollama](https://ollama.com/)** — Local LLM inference runtime.
- **[llama3.1](https://ai.meta.com/llama/)** — Meta, base LLM.
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — SYSTRAN.
- **[Silero VAD](https://github.com/snakers4/silero-vad)** — Silero Team.
- **[SymPy](https://www.sympy.org/)** — algebra / calculus / transforms.
- **[pint](https://pint.readthedocs.io/)** — unit conversions.
- **[spaCy](https://spacy.io/)** — NER for the news entity graph.
- **[D3.js](https://d3js.org/)** — entity relation graph visualization.
- **[pystray](https://github.com/moses-palmer/pystray)** — system tray icon for PC TTS.
- **[ddgs](https://pypi.org/project/ddgs/)** — DuckDuckGo search wrapper.
- **[KaTeX](https://katex.org/)** + **[Plotly.js](https://plotly.com/javascript/)**
  — math + plot rendering in the browser.
- **[Open-Meteo](https://open-meteo.com/)** — weather + geocoding, no key.
- **[ipwho.is](https://ipwho.is/)** — IP-based geolocation.
- **[Wikipedia](https://www.wikipedia.org/)** — facts + entity summaries.
- **[Tailscale](https://tailscale.com/)** — remote access + IP routing.
- **NVIDIA Jetson Orin Nano** — target hardware + JetPack stack.

## License

[MIT](LICENSE) © 2026 Ka Ming Lui
