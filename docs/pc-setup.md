# PC setup — gotchas

This is a focused troubleshooting reference for the Windows / Linux desktop
bring-up. It complements [pc-testing.md](pc-testing.md) (which walks
through running the bridge once setup works) by listing the things that
tripped us up the first time. If you're hitting cryptic errors during
setup, this page is probably where the answer is.

For full system requirements at a glance, see the
[System requirements](../README.md#system-requirements) table in the README.

---

## Python version

**Use Python 3.12. Do not use 3.14.**

Pillow (a transitive dependency of GPT-SoVITS, faster-whisper, and several
other parts of the stack) currently fails to build wheels for Python 3.14
on Windows. Older deps in the GPT-SoVITS pin chain hit similar issues.

If you accidentally created venvs with `py -3.14`, recreate them with:

```powershell
py -3.12 -m venv .venv\bridge
py -3.12 -m venv .venv\tts-server
```

Python 3.10 and 3.11 also work but 3.12 is the version actively tested.

## Ollama (Windows)

Installing Ollama on Windows registers a background service automatically.
You should **not** run `ollama serve` yourself — the service is already
listening on `:11434` and a manual `ollama serve` will conflict with it.

Verify the service is up:

```bash
curl http://127.0.0.1:11434/api/version
```

## CUDA toolkit

Match the CUDA toolkit to the torch wheel you install.

For torch 2.4+ on Windows, the recommended pairing is **CUDA 12.4 or
newer**. Install the toolkit (e.g. CUDA 12.8 to a custom location like
`E:\CUDA\`), then **add its `bin` directory to your `PATH`**:

```powershell
$env:Path += ";E:\CUDA\v12.8\bin"
```

If you skip the PATH step, faster-whisper / ctranslate2 will fail at
import time with a "cuBLAS DLL not found" error. The PyPI torch wheel
ships with bundled cuBLAS, but ctranslate2 looks up the system cuBLAS
separately.

**Important — appending vs. replacing PATH**: always *append* to `$env:Path`
with `+=`. Setting `$env:Path = "..."` will wipe the rest of your environment
and break venv activation in the same shell.

## ffmpeg (Windows: install the full-shared variant)

Plain `ffmpeg` won't satisfy torchcodec's DLL requirements on Windows.
You need the **full-shared** build:

```powershell
winget install Gyan.FFmpeg.Shared
```

Verify ffmpeg is on PATH after install:

```powershell
ffmpeg -version
```

If `winget` doesn't update the PATH automatically, restart the shell or
add `C:\Users\<you>\AppData\Local\Microsoft\WinGet\...\ffmpeg\bin` manually.

The web server's audio decode path also needs `ffmpeg` on PATH —
[web/server.py](../web/server.py)'s `decode_to_pcm16k` function shells out
to `ffmpeg -i pipe:0 -f s16le ...`.

## NLTK data (GPT-SoVITS English mode)

GPT-SoVITS in English mode requires three NLTK corpora that aren't
bundled. Run this once inside your **GPT-SoVITS venv** (not the bridge venv):

```bash
.venv\tts-server\Scripts\python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict'); nltk.download('punkt')"
```

If you skip this, the first English TTS request raises a confusing
`LookupError` deep inside the GPT-SoVITS preprocessing chain rather than
a clean "missing corpus" message.

## huggingface_hub version pin

GPT-SoVITS uses an older `tokenizers` API that broke when
`huggingface_hub` released 1.0. Pin it inside the GPT-SoVITS venv:

```bash
.venv\tts-server\Scripts\pip install "huggingface_hub<1.0"
```

This is only needed once per venv. The bridge venv is unaffected.

## fast_langdetect cache directory

If GPT-SoVITS is configured with `text_lang: auto`, `fast_langdetect`
attempts to write a model cache to a directory that may not exist on
Windows, raising `FileNotFoundError`.

Workaround: keep `text_lang: en` (and `prompt_lang: en`) in
[config/assistant.yaml](../config/assistant.yaml) under the `tts:` section.
This avoids the language-detection path entirely and matches how Karin's
voice was trained anyway.

## GPT-SoVITS pretrained model snapshot

If the smoke test fails because GPT-SoVITS can't find pretrained models,
the snapshot didn't come through. Pull it from Hugging Face directly:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lj1995/GPT-SoVITS",
    local_dir="third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models",
    local_dir_use_symlinks=False,
)
```

Run this from the repo root with the GPT-SoVITS venv's Python.

## Reference audio constraints

GPT-SoVITS expects the reference audio (for example
`characters/general/voices/ref.wav`) to be **between 3 and 10 seconds
long**. Outside that window the inference
silently splits or pads in odd ways. A 6-second clip with no leading or
trailing silence is the sweet spot.

If you're getting bizarre prosody or random extra audio in the output,
audio length is the first thing to check.

## Quick verification

After setup, this should all work:

```bash
# 1. Tests pass (CI runs bridge/ + web/; ~1,460 cases)
.venv\bridge\Scripts\python -m pytest bridge/ web/

# 2. Status helper runs cleanly
.venv\bridge\Scripts\python -m bridge.status

# 3. Ollama responds
curl http://127.0.0.1:11434/api/version

# 4. (after starting GPT-SoVITS api_v2.py) TTS responds
curl http://127.0.0.1:9880/control?command=ping
```

If any step fails, check the gotcha section above for that component
before diving into upstream docs.
