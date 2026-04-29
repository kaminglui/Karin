# Architecture

For a layer-by-layer map of the whole repo, start with
[system-overview.md](system-overview.md). This file focuses on process
boundaries, voice/audio flow, and the Jetson memory budget.

## Runtime entrypoints

Karin has two runtime entrypoints that share the same core LLM, tool, and
voice clients:

- **Browser runtime, current default**: `web.service` runs the FastAPI app
  in `web/server.py`, with `web/static/` as the client. Text chat works by
  default; STT/TTS are optional and can be local or remote.
- **Physical voice loop, legacy/embedded path**: `bridge/main.py` runs the
  microphone -> VAD -> STT -> LLM -> TTS -> speaker loop directly.

The diagram below shows the physical/local voice path because it exposes
the audio timing and model boundaries most clearly. The browser runtime
uses the same LLM/tool/TTS/STT building blocks through HTTP routes.

## Component diagram

```
                    ┌─────────────────────────┐
                    │  microphone (USB/I2S)   │
                    └───────────┬─────────────┘
                                │  int16 PCM @ 16 kHz,
                                │  512-sample frames (32 ms)
                    ┌───────────▼─────────────┐
                    │  bridge/audio_io.py     │
                    │  MicStream (callback)   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  bridge/vad.py          │
                    │  Silero VAD 5.x         │  ← CPU, ~1 MB model
                    │  state-machine segmenter│
                    └───────────┬─────────────┘
                                │  int16 PCM utterances
                    ┌───────────▼─────────────┐
                    │  bridge/stt.py          │
                    │  faster-whisper model   │  ← CPU int8; tiny/base by config
                    └───────────┬─────────────┘
                                │  English text
                    ┌───────────▼─────────────┐          ┌─────────────┐
                    │  bridge/llm.py          │──HTTP───▶│  ollama     │
                    │  /api/chat client       │          │ karin-tuned │  ← ~4.9 GB weights
                    │  rolling history        │◀─JSON────│  (llama3.1) │    (iter-3 LoRA
                    └───────────┬─────────────┘          └─────────────┘     on mannix 8B)
                                │  reply text
                    ┌───────────▼─────────────┐          ┌─────────────┐
                    │  bridge/tts.py          │──HTTP───▶│  api_v2.py  │
                    │  SoVITSTTS client       │          │  GPT-SoVITS │  ← ~1-2 GB VRAM
                    │  /set_*_weights + /tts  │◀─WAV─────│  v2Pro      │
                    └───────────┬─────────────┘          └─────────────┘
                                │  int16 PCM @ 32 kHz
                    ┌───────────▼─────────────┐
                    │  bridge/audio_io.py     │
                    │  Speaker.play()         │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  speaker (I2S/USB)      │
                    └─────────────────────────┘
```

## Memory budget

Orin Nano 8 GB advertises "8 GB" but exposes **7.42 GiB (7607 MiB)** of
shared CPU+GPU unified memory to userspace. The table below is measured
on a running production Jetson (2026-04-24), not a spec-sheet estimate.
Numbers are bytes actually resident, not weight-file size on disk.

### Default deploy (text-only, the current default)

`KARIN_STT_ENABLED=false` + `LOCAL_SOVITS=no` + `KARIN_TTS_ENABLED=false`.
No Whisper, no sovits. The LLM owns the VRAM.

| Component                              | Measured | Notes                                                     |
|----------------------------------------|----------|-----------------------------------------------------------|
| OS + JetPack + bridge + Ollama daemon  | ~1.8 GB  | Ubuntu 22.04, uvicorn, systemd, ssh, tailscaled, buffers  |
| karin-tuned (Ollama)                   | ~5.0 GB  | 4.9 GB weights; `ollama ps` reports 9%/91% CPU/GPU split on Q4_K_M — the 4.9 GB doesn't fit 100% on GPU so a sliver spills to CPU. KV at `num_ctx=2048` adds ~300 MB during an active turn. |
| Silero VAD                             | ~0       | Not loaded when STT is off.                               |
| faster-whisper                         | ~0       | Not loaded when STT is off. Local STT uses CPU RAM from the same unified pool as VRAM. |
| GPT-SoVITS v2Pro                       | ~0       | Not loaded when `LOCAL_SOVITS=no`.                        |
| **Total at idle**                      | **~6.8 / 7.4 GiB** | ~600 MiB free for buffers. ~800 MiB into swap. **Tight but stable.** |

### With STT enabled (opt-in, runtime-toggleable)

The web server loads STT at boot only when `KARIN_STT_ENABLED=true`.
After that, the sidebar dot (`POST /api/stt/enable`) only gates whether
the already-loaded recognizer is used for incoming audio. Remote STT via
`KARIN_STT_BASE_URL` does not load faster-whisper on the Jetson.

The checked-in bridge config and the Jetson setup helper both default
to `tiny.en` (Orin-Nano-safe). `base.en` is a manual override for boxes
with more headroom.

| Component                              | Would need | Notes                                                  |
|----------------------------------------|------------|--------------------------------------------------------|
| Everything from "default" above        | ~6.8 GB    |                                                         |
| faster-whisper **tiny.en** (CPU int8)  | +~400 MB   | Default in `config/assistant.yaml` and `LOCAL_STT=yes bash deploy/setup.sh`; tight but practical on Orin Nano. |
| Projected total with tiny.en           | ~7.2 GB    | Fits with ~200 MB headroom; remote STT is safer if the box is also busy. |
| faster-whisper **base.en** (CPU int8)  | +~700 MB   | Manual override (`KARIN_STT_MODEL=base.en`). On Jetson **system RAM = VRAM** (unified), so this eats from the same pool. |
| Projected total with base.en           | ~7.5 GB    | **Over 7.4 GiB** — pushes swap from 800 MiB to ~1.5-2 GiB. Works but sluggish. Bigger models such as small.en are Orin NX 16 GB+ territory. |

### With full voice stack on (STT + local sovits, opt-in)

Adding `LOCAL_SOVITS=yes` installs and enables the local GPT-SoVITS
service and the physical-mic voice loop. This is Orin NX 16 GB+ territory;
on Orin Nano, prefer remote STT/TTS sidecars over Tailscale.

| Component                              | Would need | Notes                                                  |
|----------------------------------------|------------|--------------------------------------------------------|
| Everything above (incl. local STT)      | ~7.2-7.5 GB | Depends on tiny.en vs base.en.                         |
| GPT-SoVITS v2Pro                       | +1.5-2 GB  | Varies with text length + reference audio              |
| **Projected total**                    | **~8.7-9.5 GB** | **Does not fit 7.4 GiB.** Orin NX 16 GB required. On Orin Nano, use PC-TTS offload instead (sovits on a PC over Tailscale). |

### Levers when memory pressure shows up

1. **Drop Whisper size** — `base.en` → `tiny.en` (-300 MB) via
   `KARIN_STT_MODEL` in `deploy/.env`, or runtime-toggle STT off via the
   sidebar dot. Going up to `small.en` (+300 MB vs base) is only safe on
   Orin NX 16 GB+.
2. **Shrink `num_ctx`** — from 2048 → 1024 frees ~150 MB of KV cache but
   risks truncating the ~2100-token system prompt. Don't go below 2048
   for karin-tuned.
3. **Stop sovits** — `docker compose stop sovits` frees ~1.5-2 GB. In
   the web UI, the Voice toggle does the same.
4. **Strip desktop** — on a headless Jetson, disabling GNOME/snap frees
   ~1 GB. See [jetson-setup.md § "Strip the desktop environment"](jetson-setup.md#strip-the-desktop-environment-optional).
5. **Remote voice sidecars** — set `KARIN_STT_BASE_URL` and/or
   `KARIN_TTS_BASE_URL` to a PC over Tailscale. The Jetson sends audio or
   text and receives text or WAV back. Keeps the voice VRAM entirely off
   the Jetson. See [deploy/pc-tts/README.md](../deploy/pc-tts/README.md).

## Data flow (single turn)

### Browser runtime

1. **Text turn.** The browser sends text to `web/server.py` through
   `/api/text-turn-stream` or the browser-disconnect-safe background job
   path: `POST /api/turn/start`, then `GET /api/turn/{job_id}/stream`.
2. **Audio turn.** Browser-recorded audio is uploaded to the same server,
   decoded to 16 kHz mono PCM by `decode_to_pcm16k()`, and transcribed by
   either the local `WhisperSTT` instance or a remote STT sidecar selected
   with `KARIN_STT_BASE_URL`.
3. **Chat.** Both text and audio turns call the shared `OllamaLLM`
   wrapper. Active tool schemas, rolling history, model options, and any
   model-level `think` flag from `config/models.yaml` are forwarded to the
   configured chat backend.
4. **Synthesize.** If TTS is enabled, `web/server.py` calls `SoVITSTTS`
   through the local GPT-SoVITS server or the URL in `KARIN_TTS_BASE_URL`.
   The browser receives streamed events plus an optional replayable
   `/api/audio/{conversation_id}/{audio_id}` URL.

### Physical voice loop

1. **Capture.** `MicStream` opens a PortAudio `InputStream` with a callback
   that pushes 512-sample int16 chunks into a bounded queue. Under
   backpressure, frames are dropped rather than blocking the real-time
   audio callback.
2. **Segment.** `SileroVAD.segment()` iterates those chunks, calling the
   Silero model per frame to get a speech probability. A simple state
   machine tracks `idle`/`speaking`, keeps a ring buffer of the last
   ~200 ms of silence as pre-padding, and emits one complete utterance
   when 700 ms of trailing silence is observed.
3. **Transcribe.** `WhisperSTT.transcribe()` converts the int16 PCM to
   float32 in `[-1, 1]` and runs faster-whisper in English-only mode with
   `beam_size=1` (latency-first).
4. **Chat.** `OllamaLLM.chat()` POSTs the user text plus rolling history
   to `http://localhost:11434/api/chat`, forwarding configured generation
   options. If `config/models.yaml` defines `think`, `OllamaLLM` moves it
   to the top-level Ollama request field where the API expects it.
5. **Synthesize.** `SoVITSTTS.synthesize()` POSTs the reply text to
   `http://localhost:9880/tts` along with the reference audio path and
   trained-voice conditioning parameters. The server returns a WAV blob;
   the client decodes it to int16 PCM via soundfile.
6. **Play.** `Speaker.play()` opens a one-shot `sd.play(..., blocking=True)`
   at the TTS server's output rate (32 kHz for v2Pro).

Each stage is wrapped in `try/except` in `bridge/main.py` so a single-turn
failure — a 400 from Ollama, a misheard word, a TTS 500 — logs and
continues rather than killing the loop.

## Half-duplex (by design)

The loop is strictly half-duplex: the user speaks, the assistant replies,
and only then does the mic open again for the next turn.
`interrupt_on_speech` exists in the config schema for future work but is
currently a no-op — the bridge logs a warning if it's set to true.

Full barge-in (stopping TTS mid-reply when the user starts talking) would
require either:

- **Acoustic echo cancellation (AEC)** — subtract the TTS output from the
  mic signal in real time so VAD doesn't trigger on the assistant's own
  voice.
- **Speaker ducking** — mute the mic while TTS is playing, with a small
  fade to prevent false triggers at playback boundaries.

Neither is implemented. A directional mic and physical separation between
mic and speaker, at modest playback volume, is enough in practice for
this project's use case.

## Local voice process design

The default deployment installs and starts `web.service` only. When local
GPT-SoVITS is explicitly enabled with `LOCAL_SOVITS=yes`, the physical
voice path runs as separate processes:

- **`sovits-server.service`** — runs upstream's `api_v2.py` unmodified,
  inside a venv that inherits JetPack's CUDA PyTorch via
  `--system-site-packages`.
- **`assistant.service`** — runs `python -m bridge.main`, also inside a
  venv with `--system-site-packages`.

Why two processes instead of importing GPT-SoVITS as a library?

1. **Dependency isolation.** GPT-SoVITS has heavy ML dependencies (torch,
   transformers, phonemizers, modelscope) that would bloat the bridge
   process and create version conflicts with faster-whisper. Separating
   them means each venv has its own dependency tree.
2. **Upstream compatibility.** Using the unmodified upstream `api_v2.py`
   means we can bump the GPT-SoVITS commit pin without worrying about
   rebasing our own fork — only the HTTP schema matters, and the schema
   is pinned in [bridge/tts.py](../bridge/tts.py) to a specific commit.
3. **Restart independence.** If the TTS server OOMs, systemd restarts it
   without killing the voice loop state. Similarly, reloading the bridge
   doesn't reload the TTS model (~30 seconds saved).

The trade-off is an extra HTTP round-trip per synthesized phrase — usually
under 5 ms on loopback and negligible next to the synthesis cost itself.

## Sample rate handling

Two sample rates are in play:

- **16 kHz** for mic capture, VAD, and STT. This is `audio.sample_rate`
  in the config.
- **32 kHz** for TTS output. This is the GPT-SoVITS v2Pro native rate;
  the bridge reads it from the WAV response header rather than assuming.

The `Speaker` class accepts a sample rate per `play()` call, so the
mismatch is handled without resampling on the bridge side — PortAudio
handles any rate conversion needed by the output device.

## Silero VAD chunk-size constraint

Silero VAD 5.x requires exactly **512 samples at 16 kHz** (or 256 at 8 kHz)
per forward pass. Any other size raises an error. `audio.frame_ms: 32` in
the config pins the mic frame size to this requirement:
32 ms × 16 kHz = 512 samples. Don't "simplify" this to 20 or 30 ms — it
will crash.
