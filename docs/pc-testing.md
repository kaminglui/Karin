# Testing on a PC

Before flashing/provisioning a Jetson, you can run the full bridge on any
x86 desktop with CUDA. Everything in `bridge/` is plain Python — Ollama,
GPT-SoVITS `api_v2.py`, faster-whisper, and silero-vad all run identically
on desktop CUDA and JetPack. Only VRAM pressure and inference speed differ,
and neither can be meaningfully simulated anyway.

The point of PC testing is to surface schema mismatches, import errors,
and logic bugs in your clone of the repo — not to validate performance.

## What you need

- A Windows or Linux desktop with an NVIDIA GPU (anything RTX 2000-series
  or newer will do)
- Python 3.10+
- CUDA-enabled PyTorch installed in whatever Python environment you're
  using for the bridge and GPT-SoVITS
- Ollama installed (see [ollama.com](https://ollama.com))
- Git, curl

If you're on Windows 11, Ollama auto-registers a background service on
install. You do **not** need to run `ollama serve` manually — the service
is already bound to `:11434` and running `ollama serve` yourself will
conflict with it.

## Step 1 — Pull the Ollama model

```bash
# Pull the model your local config points at.
# The checked-in assistant.yaml currently uses llama3.1:8b.
ollama pull llama3.1:8b

# Optional: pull the smaller Jetson-oriented variants too if you want
# your PC smoke test to match those deploy profiles.
# ollama pull qwen2.5:3b
# ollama pull huihui_ai/qwen3.5-abliterated:4b
curl http://localhost:11434/api/version       # confirm service is responding
```

## Step 2 — Clone GPT-SoVITS at the pinned commit

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
git checkout 2d9193b0d3c0eae0c3a14d8c68a839f1bae157dc
```

Pinning matches what `deploy/setup.sh` does on the Jetson, so any schema
quirks you hit on PC will also apply there.

## Step 3 — Install GPT-SoVITS requirements

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

On **Windows**, this may fail on `pyopenjtalk` or `pypinyin` if your MSVC
toolchain is missing or out of date — those packages have native C
extensions. If you hit build errors, look for prebuilt wheels on
conda-forge or skip the Japanese phonemizer entirely (you'll lose
Japanese support but English will still work).

## Step 4 — Start the GPT-SoVITS server

From inside the GPT-SoVITS venv:

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

First startup downloads ~1 GB of pretrained weights from HuggingFace.
Once you see `Uvicorn running on http://127.0.0.1:9880`, it's ready.

Leave this terminal running.

## Step 5 — Drop your trained weights into Karin

In a different shell:

```bash
cd /path/to/Karin

# Create or update a character bundle.
# The simplest no-extra-config option is to replace the active bundle
# under characters/general/voices/. Otherwise create characters/<name>/
# and set tts.voice in config/assistant.yaml.
#
# See characters/README.md and docs/training-guide.md for the layout:
#   characters/<name>/voice.yaml
#   characters/<name>/voices/ref.wav
#   characters/<name>/voices/gpt_model_*.ckpt
#   characters/<name>/voices/sovits_model_*.pth
```

## Step 6 — Install bridge requirements

```bash
python -m venv .venv
# Activate the venv (see above)

# Torch must already be CUDA-enabled in your environment — do NOT
# `pip install torch`, that pulls a CPU wheel.
pip install -r bridge/requirements.txt
```

## Step 7 — Smoke test the TTS client

```bash
python scripts/test_tts.py
```

This writes `tmp/tts_test.wav`. Play it. If it sounds like your trained
voice, the TTS pathway is wired up. If not, check the GPT-SoVITS server's
terminal for errors — the most common one is `text_lang` validation (see
[deployment.md#troubleshooting](deployment.md#troubleshooting)).

## Step 8 — Run the full loop

```bash
python -m bridge.main
```

Speak into your PC's default microphone. You should see log lines like:

```
loading models ...
opening mic (rate=16000 Hz, frame=32 ms, device=default)
ready — listening
utterance captured (1.87s)
user: hello, can you hear me
assistant: Yes, I can hear you.
```

And you should hear the assistant reply through your default audio
output.

## compute_type fallback

On desktop CUDA, `int8_float16` in
[config/assistant.yaml](../config/assistant.yaml) may or may not work
depending on your ctranslate2 build. If `WhisperSTT` fails to construct
with a compute-type error, temporarily switch to `float16`:

```yaml
stt:
  compute_type: float16   # was int8_float16
```

This burns a bit more VRAM (+~300 MB) but runs everywhere. Switch it
back for the Jetson deploy — JetPack's ctranslate2 handles
`int8_float16`.

## What PC testing does NOT validate

- **Actual Jetson VRAM pressure.** Your desktop has 8-24 GB; the Jetson
  has 8 GB shared with the OS. Everything that fits on desktop may not
  fit on Jetson.
- **JetPack-specific torch wheel quirks.** You're using PyPI torch on
  PC, JetPack torch on Jetson. Rare to hit a bug that only exists on
  one and not the other, but it happens.
- **ARM64 native build issues.** Some packages have ARM wheels that
  behave differently than x86 wheels.
- **Audio device enumeration.** PortAudio device names differ between
  platforms and audio stacks. Your PC's default mic is not the Jetson's
  default mic.

For those, you have to actually run on the Jetson. PC testing gets you
past the code-correctness bugs so that when the Jetson surprises you,
you know it's a Jetson-specific issue and not a "the bridge was never
going to work" issue.
