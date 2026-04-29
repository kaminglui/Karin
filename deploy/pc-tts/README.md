# Karin voice sidecar — PC-side TTS + STT offload

Runs GPT-SoVITS (TTS) and faster-whisper (STT) on your Windows PC so
the Jetson doesn't spend VRAM on voice I/O. Communicates over Tailscale.

## Files

```
tts_server.py   Voice server — /tts + /transcribe + optional tray icon
start.bat       Double-click = hidden (pythonw); --visible = console
setup.bat       setup.bat install / setup.bat uninstall
README.md       This file
```

The script is named `tts_server.py` for path-stability but now serves
both TTS and STT. Renaming the file would break existing install/auto-
start wiring — the process inside is the "Karin voice sidecar".

## Quick start

```bash
setup.bat install      # venv + deps (incl. faster-whisper) + Task Scheduler auto-start
start.bat              # manual start (hidden, tray icon)
start.bat --visible    # console mode for debugging
```

## How it works

`tts_server.py` is a FastAPI server that wraps two voice engines:

- **TTS** — GPT-SoVITS's `TTS` inference class (loaded at startup).
- **STT** — faster-whisper, loaded LAZILY on first `/transcribe` call
  so TTS-only deployments don't pay the ~1 GB cost.

Endpoints:
- `GET /health` — sidecar health, TTS settings, and lazy STT settings
- `POST /tts` — synthesize text to WAV
- `GET /set_gpt_weights?weights_path=...` — load GPT model
- `GET /set_sovits_weights?weights_path=...` — load SoVITS model
- `POST /transcribe` — transcribe base64-encoded int16 PCM to text
- `GET /stt/status` — report loaded Whisper model + settings

The server binds to the Tailscale IP only (auto-detected), so it's
not reachable from the public internet.

## STT configuration

Whisper settings come from env vars — set them before `start.bat`
launches. Typical PC-side config in a wrapper batch file:

```bat
set KARIN_STT_MODEL=small.en
set KARIN_STT_DEVICE=cuda
set KARIN_STT_COMPUTE_TYPE=int8_float16
set KARIN_STT_LANGUAGE=en
start.bat
```

Defaults if unset: `small.en`, auto-detects cuda if available else cpu,
`int8_float16` on cuda / `int8` on cpu, language=en. Because the PC
usually has spare VRAM, `small.en` or bigger (`medium.en`, `large-v3`)
is the win here — this is where the accuracy upgrade lives.

## Resource management

`start.bat` sets conservative defaults before launching the sidecar:

```bat
set KARIN_VOICE_CPU_THREADS=4
set KARIN_STT_NUM_WORKERS=1
set KARIN_TTS_MAX_CHARS=1000
set KARIN_STT_MAX_SECONDS=30
```

Override these before `start.bat` if the PC has more headroom. The
sidecar also serializes GPU-heavy TTS/STT work, caps Uvicorn concurrency,
and rejects oversized TTS/STT requests before they can allocate large
buffers.

Closing the tray icon through **Quit** asks Uvicorn to stop, marks any
active GPT-SoVITS loop for cancellation, clears Python references to
the loaded TTS/STT models, runs garbage collection, and clears CUDA
caches. If Uvicorn does not exit within 15 seconds, the sidecar forces
the process down so VRAM and file handles are released.

## Security

The server binds to `tailscale ip -4` (not `0.0.0.0`), so only
devices on your tailnet can reach it. No authentication — anyone
on the tailnet can synthesize or transcribe. If you need auth, put
a reverse proxy in front.

## Jetson configuration

In `deploy/.env` on the Jetson:
```
KARIN_TTS_ENABLED=true
KARIN_TTS_BASE_URL=http://<your-pc-tailscale-ip>:9880
KARIN_STT_ENABLED=true                              # optional — omit if you don't want voice input
KARIN_STT_BASE_URL=http://<your-pc-tailscale-ip>:9880
```

Both URLs typically point at the same host + port (the sidecar serves
both endpoints on one port). Setting `KARIN_STT_BASE_URL` makes the
Jetson skip the local faster-whisper load entirely and POST PCM to the
PC — zero Whisper memory on the Jetson.

Then `docker compose up -d --force-recreate web`.

## Reachability + error handling

When a PC URL is configured, the Jetson detects sidecar state in two places:

1. **At setup time** — `bash deploy/setup.sh` probes `/health` + `/stt/status`
   on the configured URLs and warns clearly if either is unreachable.
   Non-fatal (it assumes you may boot the PC later); the warning tells
   you exactly how to re-check.

2. **At runtime** — the web server's `GET /api/voice/status` and
   `GET /api/stt/status` both probe the sidecar on every poll
   (2s timeout). The sidebar TTS/STT dots turn to an `error` state
   with a tooltip like *"PC sidecar offline (http://…:9880)"* when
   the remote goes away. The transcription client also retries once
   on a transient `ConnectError` — single brief Tailscale blips won't
   fail a turn.

So if the PC sleeps, restarts, or the sidecar tray icon is quit, you'll
see it on the indicator before your next mic press, not after.

## Voice models

The current repo layout stores voices inside character bundles:

```
characters/<name>/
  voice.yaml
  voices/
    ref.wav
    gpt_model_*.ckpt
    sovits_model_*.pth
```

The Jetson-side bridge discovers these bundles automatically and can
switch at runtime via `POST /api/tts/voice` by pushing the selected
weights to this PC-side server.

## Stopping

- Tray mode: right-click tray icon → Quit. This stops the server process
  and releases the loaded model resources.
- Any mode: `setup.bat uninstall` removes the auto-start task and stops
  only the KarinTTS `tts_server.py` process.

## Uninstalling

```bash
setup.bat uninstall    # removes Task Scheduler entry + kills processes
```

## Always launch via `start.bat` (not `python tts_server.py`)

`start.bat` runs the server from the dedicated venv at
`.venv\tts-server\Scripts\python(w).exe`. A manual `python
tts_server.py` from another shell uses your global Python, which:

1. Won't have the sidecar's pinned dependencies — endpoints can fail
   on import errors that the venv-launched copy doesn't see.
2. Doesn't auto-reload when you `git pull` updates to `tts_server.py`
   — Python freezes the module bytecode at start time. A stale
   process keeps serving the old route table; new endpoints (e.g.
   `/stt/status` after the STT addition) return 404 until you kill
   the process and relaunch via `start.bat`.

If endpoints are missing or behaving oddly, identify the running
process and confirm it's the venv copy:

```powershell
Get-NetTCPConnection -LocalPort 9880 -State Listen |
    ForEach-Object { Get-Process -Id $_.OwningProcess } |
    Select-Object Id, ProcessName, Path
```

`Path` should end in `.venv\tts-server\Scripts\python(w).exe`. If
it's a global Python install, `Stop-Process -Id <pid>` and relaunch
via `start.bat`.

## Health check

Use `/health` for scripts and manual checks:

```powershell
curl http://<your-pc-tailscale-ip>:9880/health
```

The sidecar intentionally disables FastAPI's `/docs` UI, so `/health`
is the stable readiness endpoint.
