# Deployment

This document covers the Jetson side of Karin — what `setup.sh`
does, how to start and stop things, and how to troubleshoot the first
run.

For PC-side testing before touching the Jetson, see
[pc-testing.md](pc-testing.md).

## Prerequisites

`setup.sh` will **not** install these — they're hard requirements and the
script fails fast if they're missing:

1. **JetPack on an ARM64 Jetson.** Tested on Jetson Orin Nano 8 GB.
   Other Jetsons with enough VRAM should work but haven't been validated.
2. **NVIDIA-supplied PyTorch wheels**, installed system-wide, with
   `torch.cuda.is_available()` returning True from the default `python3`.
   See [NVIDIA's install guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/).
   Do **not** `pip install torch` — that pulls a CPU-only wheel from
   PyPI and silently breaks CUDA.
3. **Optional: a complete character voice bundle** under
   `characters/<name>/voices/` if you want to run GPT-SoVITS on the
   Jetson with `LOCAL_SOVITS=yes`:
    - `ref.wav`
    - `gpt_model_*.ckpt`
    - `sovits_model_*.pth`

   The matching metadata lives in `characters/<name>/voice.yaml`. Fresh
   installs can run text-only, or offload TTS/STT to a PC sidecar, without
   this bundle. See [training-guide.md](training-guide.md) and
   [../characters/README.md](../characters/README.md).

Optional but recommended:

- **Tailscale** for remote access from a phone or laptop without exposing
  the Jetson to the open internet.
- **Docker** for Open WebUI. If not installed, `setup.sh` skips the
  WebUI step with a warning.

## Running setup.sh

From the repo root on the Jetson:

```bash
# Cache your sudo password so the script doesn't stall mid-run
sudo -v

bash deploy/setup.sh
```

`setup.sh` is idempotent — safe to re-run after a failure or as part of
an update. Re-running will:

- Skip Ollama install if already present, but update models.
- Build or reuse `karin-tuned:latest` when the LoRA fetch succeeds.
- Reinstall the web systemd unit and bridge environment.
- Set `deploy/.env` to use `KARIN_LLM_MODEL=karin-tuned:latest` when
  that tag exists and no model override is already present.
- Only fetch GPT-SoVITS, create the TTS venv, install local voice units,
  restart `sovits-server`, and run the TTS smoke test when
  `LOCAL_SOVITS=yes` and a complete voice bundle is present.

### What setup.sh does

In order:

1. **Sanity checks.** Verifies ARM64 + JetPack + `python3 -c "import torch; torch.cuda.is_available()"`.
2. **Voice-bundle discovery.** Detects complete bundles under
   `characters/*/voices/`. Missing bundles do not fail setup; the script
   continues in text-only / PC-offload mode unless you explicitly opt into
   local SoVITS with `LOCAL_SOVITS=yes`.
3. **apt packages.** Installs `portaudio19-dev`, `ffmpeg`, `libsndfile1`,
   `python3-venv`, `build-essential`, etc.
4. **Ollama.** Runs the upstream installer if missing, enables and starts
   the `ollama.service`, waits for it to respond on :11434, pulls the
   models listed in [config/ollama_models.txt](../config/ollama_models.txt).
5. **Production LLM tag.** Builds `karin-tuned:latest` from the routing
   LoRA when possible. If that tag exists, writes
   `KARIN_LLM_MODEL=karin-tuned:latest` to `deploy/.env` unless you
   already set a model there.
6. **GPT-SoVITS clone, pretrained weights, and TTS venv.** Only when
   `LOCAL_SOVITS=yes` and a complete character voice bundle is present.
   Creates `.venv/tts-server/` with `--system-site-packages` so it
   inherits JetPack's CUDA torch, then installs GPT-SoVITS requirements
   with `torch`/`torchaudio`/`torchvision` stripped out.
7. **Bridge venv.** Creates `.venv/bridge/` with `--system-site-packages`,
   installs `bridge/requirements.txt`.
8. **Systemd units.** Substitutes `@REPO_ROOT@` and `@USER@` in the
   templates, installs `web.service`, and installs the local voice units
   only when local SoVITS is enabled. Runs `daemon-reload` and enables the
   installed units.
9. **Open WebUI.** If Docker is installed, `docker compose up -d` from
   the `deploy/` directory. Uses host networking mode so the container
   reaches Ollama at `localhost:11434`.
10. **Start services.** Starts `web.service`. When local SoVITS is
    enabled, also starts `sovits-server.service`, waits for :9880 via
    [deploy/wait-for-tts.sh](../deploy/wait-for-tts.sh), runs the TTS
    smoke test, optionally pauses for audition, then starts
    `assistant.service`.
11. **Voice summary.** Prints whether TTS/STT are off, local, remote, or
    enabled-but-unconfigured, and probes any configured PC-side URLs.

### Non-interactive runs

If you're running `setup.sh` via SSH/automation and can't interactively
audition the output:

```bash
AUDITION=no bash deploy/setup.sh
```

This skips the audition pause. The assistant service is enabled but not
started when local SoVITS is enabled — you start it manually after
verifying the WAV:

```bash
.venv/bridge/bin/python scripts/test_tts.py
# play tmp/tts_test.wav
sudo systemctl start assistant
```

## Operating the services

```bash
# Status
sudo systemctl status sovits-server
sudo systemctl status assistant

# Live logs
journalctl -fu sovits-server
journalctl -fu assistant
journalctl -fu ollama

# Restart the bridge without restarting the TTS server
sudo systemctl restart assistant

# Stop everything
sudo systemctl stop assistant
sudo systemctl stop sovits-server
# ollama.service will keep running — stop it manually if you want:
sudo systemctl stop ollama
```

The `assistant.service` has
`Requires=sovits-server.service ollama.service`, so stopping either will
also stop the assistant. Re-starting them won't automatically re-start
the assistant — you have to do that explicitly.

## Accessing from a phone

Two options:

- **Voice loop.** Physical mic + speaker at the Jetson, no phone needed.
  The bridge is always listening.
- **Open WebUI chat.** `http://<jetson-tailscale-ip>:8080` from any
  device on your tailnet. This talks to the same Ollama instance the
  voice loop uses, so you're chatting with the same model either way.
  Voice does not currently round-trip through the WebUI — it's a
  separate pathway.

## Troubleshooting

### `ollama pull` fails with 404

Almost certainly means the model tag in
[config/ollama_models.txt](../config/ollama_models.txt) doesn't exist on
the Ollama registry. `setup.sh` pulls whatever tags are listed there,
then builds `karin-tuned:latest` locally from the LoRA adapter when the
base model and Hugging Face download are available. If `karin-tuned`
exists, setup writes `KARIN_LLM_MODEL=karin-tuned:latest` to
`deploy/.env` unless you already set a model there. If a tag disappears
upstream, edit the relevant file and re-run setup.

### `setup.sh` dies on the torch-CUDA check

```
[fail] system python3 lacks torch with CUDA. Install JetPack PyTorch wheels first
```

You don't have CUDA-enabled PyTorch. This is a JetPack prerequisite, not
a setup.sh responsibility. Follow
[NVIDIA's install guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)
and re-run.

### GPT-SoVITS venv install fails on `pip install`

Usually one of:

- A transitive dependency pinning torch to a version incompatible with
  the JetPack wheel. Try `--no-deps` on the offending package.
- A native build failure (pyopenjtalk, pypinyin) — the ARM64 compilation
  needs matching header packages. Install `libopenblas-dev` and re-run.
- Network flake — `pip install` retries aren't always enough. Re-run
  setup.sh.

### sovits-server starts but the smoke test fails with `/tts` 400

Check `journalctl -u sovits-server --no-pager -n 50`. Common causes:

- `text_lang: auto` not in your server's languages list. Fix in
  [config/assistant.yaml](../config/assistant.yaml) — try `en` or `ja`.
- `ref_audio_path` not readable by the service user. Check permissions
  on the active character bundle's `ref.wav`
  (for example `characters/general/voices/ref.wav`).
- Model file corruption. Re-copy from source.

### Whisper fails with `compute_type not supported`

Change [config/assistant.yaml](../config/assistant.yaml)
`stt.compute_type` from `int8_float16` to `float16`, or further down to
`int8` (CPU fallback, slower but universally supported).

### Qwen replies contain `<think>...</think>` blocks

Thinking mode isn't being disabled. The `options.think: "off"` key in
the config may need to move to a top-level `think: false` request field
depending on your Ollama version. Edit
[bridge/llm.py](../bridge/llm.py)'s `chat()` method to pass
`"think": False` at the top level of the request body instead of inside
`options`.

### VAD triggers on the assistant's own TTS output

Half-duplex expects the mic not to hear the speaker. If it does:

- Lower speaker volume
- Physically separate mic and speaker further
- Switch to a more directional mic
- As a last resort, raise `vad.threshold` from 0.5 to 0.6-0.7 in config
  (will also make it harder for soft-spoken users to trigger)

### First boot of sovits-server takes forever or times out

Model loading + first pipeline construction can take 60+ seconds on a
cold cache. `TimeoutStartSec=300` in the service file gives it 5
minutes. If you're hitting that, check `dmesg` for OOM kills —
`sovits-server` sharing VRAM with Whisper and Qwen is tight on 8 GB,
and a heavy Whisper init can push TTS over the edge. Workaround: drop
Whisper to `base.en` or `tiny.en`.
