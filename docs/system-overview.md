# System overview

This is the shortest accurate map of how Karin is put together. Use it
when you need to orient yourself before reading the deeper architecture,
deployment, or routing docs.

## Reading path

1. [../README.md](../README.md) - what Karin is and the fastest way to try it.
2. [../RUNBOOK.md](../RUNBOOK.md) - how to deploy, operate, reset, and troubleshoot it.
3. [architecture.md](architecture.md) - process layout, voice data flow, and memory budget.
4. [routing-pipeline.md](routing-pipeline.md) - how the LLM decides whether to call tools and how replies are composed.
5. [api-reference.md](api-reference.md) - config schema and HTTP endpoints.

## System layers

### 1. User interfaces

- **Browser UI** lives under `web/static/` and talks to the FastAPI app in
  `web/server.py`.
- **Panel APIs** in `web/panels_api.py` expose supporting routes for chat,
  trackers, news, alerts, reminders, profiles, preferences, and TTS voice
  switching.
- **Legacy physical voice loop** in `bridge/main.py` runs the original
  microphone -> VAD -> STT -> LLM -> TTS -> speaker loop outside the
  browser UI.

### 2. Web/API runtime

`web/server.py` is the main process for normal browser use. It loads
`config/assistant.yaml`, applies environment overrides, starts optional
background pollers, wires the LLM client, and exposes `/api/*` routes for
text turns, health, settings, STT/TTS status, character switching, and
runtime controls.

`config/features.yaml` and environment variables decide which subsystems
are visible at runtime. Defaults are conservative: text chat works by
default, while STT and TTS are opt-in.

### 3. LLM and routing

`bridge/llm.py` wraps the chat backend. The production path is Ollama with
`karin-tuned:latest` when deployment has built that tag; the checked-in
config keeps a base-model fallback.

Tool definitions live in `bridge/tools/_schemas.py`. Runtime filtering
lives in `bridge/tools/_dispatch.py` via `active_tool_schemas()`, which
hides disabled or feature-gated tools before the LLM sees them.

Small-model safety layers live under `bridge/routing/`:

- regex classifier hints for obvious tool requests;
- optional structural vetoes for false positives;
- under-fire rescue for safe tools when the LLM misses a confident hint;
- continuation handling for follow-up prompts.

### 4. Tools and background subsystems

Tools are implemented in `bridge/tools/` and call into subsystem packages
when needed. Examples:

- news: `bridge/news/`
- alerts: `bridge/alerts/`
- trackers: `bridge/trackers/`
- reminders: `bridge/reminders/`
- memory/profile state: `bridge/memory.py`, `bridge/profiles/`, and
  related data stores

Runtime state belongs in `data/` or subsystem `data/` folders and is
gitignored.

### 5. Voice stack

Voice is optional and has three supported modes:

- **Text-only**: default. No STT/TTS models are loaded.
- **Local STT on the Jetson**: `LOCAL_STT=yes bash deploy/setup.sh`
  enables faster-whisper `tiny.en` on CPU.
- **PC sidecar offload**: `deploy/pc-tts/` runs GPT-SoVITS and
  faster-whisper on a Windows PC, while the Jetson keeps the LLM memory
  budget.

Local GPT-SoVITS on the Jetson is possible only on larger hardware or when
explicitly enabled with `LOCAL_SOVITS=yes` and a complete character voice
bundle is present.

### 6. Characters and voices

Character folders live under `characters/<name>/`.

- `voice.yaml` is persona and voice metadata and is private per deploy
  except for `characters/default/voice.yaml`.
- `voices/ref.wav`, `*.ckpt`, and `*.pth` are trained voice assets and are
  gitignored.
- `face.json` or `expressions/default.png` controls how the character is
  displayed in the UI.

See [../characters/README.md](../characters/README.md) for the full layout
and switching behavior.

### 7. Deployment

`deploy/setup.sh` is the Jetson host setup path. It installs system
packages, prepares Ollama, builds or reuses `karin-tuned:latest`, writes
deployment environment overrides when appropriate, creates Python venvs,
installs systemd units, and starts the web service.

Docker Compose files under `deploy/` provide the containerized path. The
Docker build context intentionally excludes private runtime artifacts via
`.dockerignore`.

## Current vs historical docs

The docs are intentionally split by purpose:

- **Current operating path**: README, RUNBOOK, system overview,
  architecture, deployment, API reference, tools, character docs.
- **Research/history**: routing pipeline, routing eval comparison, eval
  manual review, iter postmortems, and SFT notes.
- **Raw artifacts**: files under `docs/evals/`.

Historical docs are kept because they explain why the current runtime has
its specific guardrails. They should not be treated as setup instructions
unless they explicitly say so.
