#!/usr/bin/env bash
#
# One-shot Jetson provisioning for Karin.
#
# Prerequisites (setup.sh will NOT install these — it fails fast if missing):
#   - Jetson running JetPack (ARM64 / aarch64)
#   - NVIDIA's JetPack PyTorch wheel installed system-wide with CUDA.
#     See https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
#   - git, curl, sudo
#
# Voice weights are OPTIONAL. If a complete bundle is present under
# characters/<name>/voices/ (ref.wav + gpt_model_*.ckpt + sovits_model_*.pth),
# the full voice stack is provisioned. If not, setup proceeds in text-only
# mode: only web.service is installed, TTS/STT stay at their features.yaml
# defaults (off), and the browser UI works over chat without TTS. To add
# voice later: train a bundle (docs/training-guide.md), drop it under
# characters/<name>/voices/, and re-run this script.
#
# Local GPT-SoVITS is OPT-IN. Even when voice weights are present, running
# sovits locally on the Jetson Orin Nano 8 GB pushes unified memory over
# its cap (the LLM spills to CPU and inference slows ~3x). So by default
# this script skips the GPT-SoVITS clone, pretrained-weights download,
# TTS venv, and sovits-server.service — voice is exercised via the PC-TTS
# offload instead (see deploy/pc-tts/README.md). Flip the opt-in if you
# have Orin NX 16 GB or larger:
#
#   LOCAL_SOVITS=yes bash deploy/setup.sh
#
# Usage (from the repo root, NOT from deploy/):
#   sudo -v                 # cache sudo so the script doesn't stall
#   bash deploy/setup.sh
#
# Env vars:
#   LOCAL_SOVITS=yes|no     (default: no — run GPT-SoVITS on a PC or skip
#                           voice entirely. `yes` installs + starts the
#                           sovits-server service on THIS host; only turn
#                           it on for Orin NX 16 GB or larger.)
#   LOCAL_STT=yes|no        (default: no — STT stays off unless you also
#                           point at a PC-side voice sidecar via
#                           KARIN_STT_BASE_URL. `yes` writes
#                           KARIN_STT_ENABLED=true + KARIN_STT_MODEL=tiny.en
#                           + KARIN_STT_DEVICE=cpu to deploy/.env so the
#                           smallest Whisper model loads in-process — the
#                           only variant that leaves enough headroom next
#                           to karin-tuned on Orin Nano 8 GB.)
#   AUDITION=auto|yes|no    (default: auto — interactive if TTY, skip if not)
#                           yes  = force interactive audition pause
#                           no   = skip audition; leave assistant.service stopped
#                           Ignored when LOCAL_SOVITS=no.
#
# Idempotent: safe to re-run. Existing venvs, git checkouts, and systemd
# units will be updated in place.

set -euo pipefail

# --- constants ---------------------------------------------------------------

GPT_SOVITS_REPO="https://github.com/RVC-Boss/GPT-SoVITS.git"
# Pinned to match the schema bridge/tts.py was written against.
GPT_SOVITS_COMMIT="2d9193b0d3c0eae0c3a14d8c68a839f1bae157dc"
THIRD_PARTY_DIR="third_party"
GPT_SOVITS_DIR="${THIRD_PARTY_DIR}/GPT-SoVITS"
BRIDGE_VENV=".venv/bridge"
TTS_VENV=".venv/tts-server"

AUDITION="${AUDITION:-auto}"
LOCAL_SOVITS="${LOCAL_SOVITS:-no}"
LOCAL_STT="${LOCAL_STT:-no}"

# --- helpers -----------------------------------------------------------------

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn] \033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m[fail] \033[0m %s\n" "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

# --- sanity checks -----------------------------------------------------------

[[ -f bridge/main.py && -d config ]] \
    || die "run this from the Karin repo root (not deploy/)"

log "checking system"

if [[ "$(uname -m)" != "aarch64" ]]; then
    warn "not on aarch64 — this script targets Jetson. continuing anyway."
fi
if [[ ! -f /etc/nv_tegra_release ]]; then
    warn "/etc/nv_tegra_release absent — JetPack not detected. continuing anyway."
fi

need python3
need git
need curl
need sudo

# torch + CUDA is a hard prerequisite we will not try to install.
# Bridge venv uses --system-site-packages to inherit system torch
# (avoids redownloading ~500 MB per venv).
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    _jetpack_line=$(head -1 /etc/nv_tegra_release 2>/dev/null || echo "")
    _cuda_idx="cu126"   # default for JP 6.1 / 6.2 (L4T R36.3+)
    if echo "$_jetpack_line" | grep -q "R36.*REVISION: 2"; then
        _cuda_idx="cu122"    # JP 6.0 (L4T R36.2) uses CUDA 12.2
    fi
    die "system python3 lacks torch with CUDA.
       Install the NVIDIA JetPack PyTorch wheel (one-time, ~500 MB):

           pip3 install --user --no-cache \\
               --index-url https://pypi.jetson-ai-lab.dev/jp6/$_cuda_idx \\
               torch torchvision torchaudio

       Verify:
           python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'

       Then re-run this script. See RUNBOOK.md for the per-JetPack
       wheel index (6.0 → cu122, 6.1+/6.2 → cu126) or the upstream
       docs: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/"
fi
log "system python torch + CUDA ok"

# --- voice model files check (OPTIONAL) --------------------------------------
#
# A complete voice triplet (ref.wav + *.ckpt + *.pth) under any
# characters/<name>/voices/ unlocks the full voice stack. If none is found,
# setup proceeds in text-only mode: sovits-server + assistant.service are
# NOT installed, and web.service starts without a TTS backend. The operator
# can add a voice later and re-run this script.

log "checking for trained voice model files"
HAS_VOICE=0
for char_dir in characters/*/; do
    [[ -d "${char_dir}voices" ]] || continue
    ref=$(find "${char_dir}voices" -maxdepth 1 -type f \( -name 'ref.wav' -o -name '*_ref.wav' \) | head -n 1)
    gpt=$(find "${char_dir}voices" -maxdepth 1 -type f -name '*.ckpt' | head -n 1)
    sov=$(find "${char_dir}voices" -maxdepth 1 -type f -name '*.pth'  | head -n 1)
    if [[ -n "$ref" && -n "$gpt" && -n "$sov" ]]; then
        log "  found voice in ${char_dir}: $(basename "$ref"), $(basename "$gpt"), $(basename "$sov")"
        HAS_VOICE=1
    fi
done

if (( HAS_VOICE == 0 )); then
    log "no voice bundle found — continuing in TEXT-ONLY mode."
    log "  To enable voice later:"
    log "    1. Train a bundle per docs/training-guide.md"
    log "    2. Drop ref.wav + gpt_model_*.ckpt + sovits_model_*.pth into"
    log "       characters/<name>/voices/"
    log "    3. Re-run: bash deploy/setup.sh (adding LOCAL_SOVITS=yes only"
    log "       if your Jetson has ≥ 16 GB unified memory)"
fi

# --- validate AUDITION + LOCAL_SOVITS ---------------------------------------

case "$AUDITION" in
    auto|yes|no) ;;
    *) die "AUDITION must be auto|yes|no, got: $AUDITION" ;;
esac

case "$LOCAL_SOVITS" in
    yes|no) ;;
    *) die "LOCAL_SOVITS must be yes|no, got: $LOCAL_SOVITS" ;;
esac

case "$LOCAL_STT" in
    yes|no) ;;
    *) die "LOCAL_STT must be yes|no, got: $LOCAL_STT" ;;
esac

# Resolve the effective voice-stack mode. `WANT_LOCAL_SOVITS=1` means
# "provision GPT-SoVITS locally on this host": clone the repo, download
# the 4 GB pretrained models, build the TTS venv, install + start the
# sovits-server systemd unit (or bring up the Docker sovits container).
# Requires voice weights to be present AND the operator to opt in via
# LOCAL_SOVITS=yes. Default (no) keeps the Jetson memory-safe — voice
# still works via the PC-TTS offload (KARIN_TTS_BASE_URL in deploy/.env).
WANT_LOCAL_SOVITS=0
if (( HAS_VOICE == 1 )) && [[ "$LOCAL_SOVITS" == "yes" ]]; then
    WANT_LOCAL_SOVITS=1
fi

if (( HAS_VOICE == 1 )) && (( WANT_LOCAL_SOVITS == 0 )); then
    log "voice weights present but LOCAL_SOVITS=no — skipping local GPT-SoVITS"
    log "  (default on Jetson Orin Nano 8 GB; 7.4 GiB usable and karin-tuned alone holds ~6.8 of it)."
    log "  To run sovits HERE (Orin NX 16 GB+): LOCAL_SOVITS=yes bash deploy/setup.sh"
    log "  To run sovits on a separate PC over Tailscale: see deploy/pc-tts/README.md"
    log "    then set KARIN_TTS_BASE_URL + KARIN_TTS_ENABLED=true in deploy/.env."
fi

# --- apt packages ------------------------------------------------------------

log "installing apt packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    git-lfs \
    libsndfile1 \
    portaudio19-dev \
    python3-dev \
    python3-venv

# --- ollama ------------------------------------------------------------------

if ! command -v ollama >/dev/null 2>&1; then
    log "installing ollama"
    curl -fsSL https://ollama.com/install.sh | sh
    hash -r
else
    log "ollama already installed"
fi

# Detect installed version + warn if outside the validated range.
# Karin needs >=0.4 for tool calling; we've validated through 0.21.x.
# This guard doesn't enforce anything — just surfaces the version so
# operators know what they're actually running (the install branch
# above takes whatever ollama.com ships that day, the skip branch
# takes whatever was already there).
_olm_ver=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [[ -z "$_olm_ver" ]]; then
    warn "could not parse ollama version output: $(ollama --version 2>&1 | head -1)"
else
    log "ollama version: $_olm_ver"
    _olm_major=${_olm_ver%%.*}
    _olm_rest=${_olm_ver#*.}
    _olm_minor=${_olm_rest%%.*}
    if (( _olm_major == 0 && _olm_minor < 4 )); then
        warn "ollama $_olm_ver is older than 0.4 — tool calling may not work."
        warn "  upgrade with: curl -fsSL https://ollama.com/install.sh | sh"
    elif (( _olm_major == 0 && _olm_minor > 21 )); then
        warn "ollama $_olm_ver is newer than the last validated version (0.21.x)."
        warn "  Karin should still work, but check Ollama release notes for"
        warn "  breaking changes if you see unexpected bridge behavior."
    fi
fi

sudo systemctl enable --now ollama

log "waiting for ollama to respond"
ollama_ready=0
for _ in {1..30}; do
    if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
        ollama_ready=1
        break
    fi
    sleep 1
done
(( ollama_ready == 1 )) || die "ollama did not come up on :11434"

# --- LLM models --------------------------------------------------------------

log "pulling LLM models from config/ollama_models.txt"
while IFS= read -r line || [[ -n "$line" ]]; do
    model="${line%%#*}"               # strip inline comments
    model="${model//[[:space:]]/}"    # strip whitespace
    [[ -z "$model" ]] && continue
    log "  ollama pull $model"
    ollama pull "$model"
done < config/ollama_models.txt

# --- karin-tuned: iter-3 routing LoRA on the mannix base ---------------------
#
# Fetches karin-lora.gguf from huggingface.co/kaminglui/karin-lora and builds
# karin-tuned:latest in Ollama on top of the mannix base. This is the
# production routing model — 93.3% / 91.9% / 59.2% on the 135-case eval.
# Idempotent: skipped when the tag already exists.
#
# Fail-soft: if HF download or the Modelfile build fails, setup continues
# and the deploy falls back to whatever bases are in ollama_models.txt. The
# web UI will work; routing accuracy just drops to the pre-LoRA baseline.

if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx 'karin-tuned:latest'; then
    log "karin-tuned:latest already present in Ollama — skipping HF fetch"
else
    log "fetching karin-lora.gguf from kaminglui/karin-lora"

    # hf CLI is optional — install on-demand if missing.
    if ! command -v hf >/dev/null 2>&1; then
        log "  installing huggingface_hub CLI"
        python3 -m pip install --user --quiet "huggingface_hub[cli]<1.0" || true
        # User-scoped pip installs land under ~/.local/bin on Linux.
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if command -v hf >/dev/null 2>&1; then
        LORA_STAGE="$(mktemp -d -t karin-lora.XXXXXX)"
        if hf download kaminglui/karin-lora karin-lora.gguf --local-dir "$LORA_STAGE" >/dev/null; then
            log "  downloaded $(du -h "$LORA_STAGE/karin-lora.gguf" 2>/dev/null | cut -f1)"
            # Build the Modelfile inside the stage dir so the ADAPTER relative
            # path resolves to the downloaded GGUF.
            if (
                cd "$LORA_STAGE" && \
                ollama show mannix/llama3.1-8b-abliterated:tools-q4_k_m --modelfile > Modelfile 2>/dev/null && \
                echo 'ADAPTER ./karin-lora.gguf' >> Modelfile && \
                # Pin num_ctx so the model loads with enough room for
                # the 21-tool schema even if a chat call ever skips
                # the option. Without this Ollama can fall back to the
                # base Modelfile's default (sometimes 512), which makes
                # the LoRA narrate the truncated tool catalog instead
                # of routing. Matches assistant.yaml + models.yaml.
                echo 'PARAMETER num_ctx 3072' >> Modelfile && \
                ollama create karin-tuned -f Modelfile
            ); then
                log "built karin-tuned:latest (iter-3 LoRA on mannix base)"
            else
                warn "karin-tuned build failed — mannix base probably missing."
                warn "  ensure mannix/llama3.1-8b-abliterated:tools-q4_k_m is in"
                warn "  config/ollama_models.txt and was pulled successfully above."
            fi
        else
            warn "HF download failed (kaminglui/karin-lora). karin-tuned not built."
            warn "  deploy continues with the fallback bases from ollama_models.txt."
            warn "  retry manually:"
            warn "    hf download kaminglui/karin-lora karin-lora.gguf --local-dir ."
            warn "    ollama show mannix/llama3.1-8b-abliterated:tools-q4_k_m --modelfile > Modelfile"
            warn "    echo 'ADAPTER ./karin-lora.gguf' >> Modelfile"
            warn "    echo 'PARAMETER num_ctx 3072' >> Modelfile"
            warn "    ollama create karin-tuned -f Modelfile"
        fi
        rm -rf "$LORA_STAGE"
    else
        warn "hf CLI install failed — skipping karin-tuned build."
        warn "  manual path documented above in the fail-soft warning."
    fi
fi

# --- Config sanity: warn if assistant.yaml ignores the LoRA we just built ----
#
# The LoRA flow (above) builds karin-tuned:latest in Ollama, but the bridge
# only reads it when config/assistant.yaml's `llm.model` line points there.
# A stale or hand-edited config that still references the stock base
# silently bypasses every routing improvement the LoRA contributes (~20 pp
# on the 135-case eval). Guard against the mismatch on every deploy.
if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx 'karin-tuned:latest'; then
    _yaml_model=$(awk '/^llm:/{f=1} f&&/^[[:space:]]+model:/{print $2; exit}' \
                  "$REPO_ROOT/config/assistant.yaml" 2>/dev/null)
    if [[ -n "$_yaml_model" && "$_yaml_model" != *"karin-tuned"* ]]; then
        warn "config/assistant.yaml has llm.model=$_yaml_model"
        warn "  but karin-tuned:latest is built and ready. The bridge will"
        warn "  use $_yaml_model — losing the iter-3 LoRA's routing tuning."
        warn "  To use the LoRA, edit config/assistant.yaml:"
        warn "    llm:"
        warn "      model: karin-tuned:latest"
    fi
fi

# --- GPT-SoVITS (local sovits only) ------------------------------------------
#
# Skipped in text-only mode AND in voice-but-PC-offload mode. The repo +
# its ~4 GB upstream pretrained weights only exist on disk when the
# operator has opted into running sovits locally (LOCAL_SOVITS=yes AND a
# character voice bundle is present). Re-running setup.sh with
# LOCAL_SOVITS=yes after dropping in voice weights triggers the whole
# block (clone + pretrained download + TTS venv).

if (( WANT_LOCAL_SOVITS == 1 )); then
    log "setting up GPT-SoVITS at commit $GPT_SOVITS_COMMIT"
    mkdir -p "$THIRD_PARTY_DIR"
    if [[ ! -d "$GPT_SOVITS_DIR/.git" ]]; then
        git clone "$GPT_SOVITS_REPO" "$GPT_SOVITS_DIR"
    fi
    (
        cd "$GPT_SOVITS_DIR"
        git fetch --quiet origin
        git checkout --quiet "$GPT_SOVITS_COMMIT"
    )

    # --- GPT-SoVITS upstream pretrained weights (~4 GB one-time) -----------
    PRETRAINED_DIR="$GPT_SOVITS_DIR/GPT_SoVITS/pretrained_models"
    # Heuristic: if the dir has at least one non-empty subdirectory, assume
    # the snapshot is already present. snapshot_download is idempotent
    # anyway (skips already-cached files), so the guard is just a speed win.
    if [[ ! -d "$PRETRAINED_DIR" ]] || \
       [[ -z "$(find "$PRETRAINED_DIR" -mindepth 2 -type f -print -quit 2>/dev/null)" ]]; then
        log "downloading GPT-SoVITS upstream pretrained weights (~4 GB, one-time)"
        python3 -m pip install --user --quiet "huggingface_hub<1.0"
        python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lj1995/GPT-SoVITS",
    local_dir="$PRETRAINED_DIR",
    local_dir_use_symlinks=False,
)
PY
    else
        log "GPT-SoVITS pretrained weights already present"
    fi

    # --- TTS venv ----------------------------------------------------------
    log "creating TTS venv ($TTS_VENV) with --system-site-packages"
    python3 -m venv --system-site-packages "$TTS_VENV"

    "$TTS_VENV/bin/pip" install --quiet --upgrade pip wheel setuptools

    # Filter out torch/torchaudio/torchvision before installing GPT-SoVITS
    # requirements. JetPack already provides a CUDA-enabled torch via
    # --system-site-packages; if we let pip see the unfiltered requirements
    # file it will "helpfully" install a CPU-only torch from PyPI and clobber CUDA.
    FILTERED_REQ="$(mktemp)"
    trap 'rm -f "$FILTERED_REQ"' EXIT
    grep -ivE '^[[:space:]]*(torch|torchaudio|torchvision)([[:space:]]|=|>|<|!|~|$)' \
        "$GPT_SOVITS_DIR/requirements.txt" > "$FILTERED_REQ"

    log "installing GPT-SoVITS requirements (torch family filtered out)"
    "$TTS_VENV/bin/pip" install -r "$FILTERED_REQ"
else
    log "skipping GPT-SoVITS clone + venv (local sovits not enabled)"
fi

# --- bridge venv -------------------------------------------------------------

log "creating bridge venv ($BRIDGE_VENV) with --system-site-packages"
python3 -m venv --system-site-packages "$BRIDGE_VENV"
"$BRIDGE_VENV/bin/pip" install --quiet --upgrade pip wheel setuptools
log "installing bridge requirements"
"$BRIDGE_VENV/bin/pip" install -r bridge/requirements.txt

# --- systemd units -----------------------------------------------------------

log "installing systemd units"
REPO_ROOT_ABS="$(pwd)"
USER_NAME="$(id -un)"

install_unit() {
    local src="$1"
    local dst="/etc/systemd/system/$(basename "$src")"
    sed \
        -e "s|@REPO_ROOT@|${REPO_ROOT_ABS}|g" \
        -e "s|@USER@|${USER_NAME}|g" \
        "$src" | sudo tee "$dst" >/dev/null
    sudo chmod 0644 "$dst"
}

install_unit deploy/web.service
if (( WANT_LOCAL_SOVITS == 1 )); then
    install_unit deploy/sovits-server.service
    install_unit deploy/assistant.service
fi

sudo systemctl daemon-reload
sudo systemctl enable web.service
if (( WANT_LOCAL_SOVITS == 1 )); then
    sudo systemctl enable sovits-server.service assistant.service
fi

# --- deploy/.env: flip voice flags based on LOCAL_SOVITS / LOCAL_STT --------
#
# Runtime defaults in config/features.yaml are STT=off + TTS=off on Jetson.
# The opt-in flags below update deploy/.env so the web container / service
# wire up the right clients at next start. PC-offload users set
# KARIN_TTS_BASE_URL / KARIN_STT_BASE_URL directly in .env — those take
# precedence over the local paths below (see web/server.py STT factory).

_ensure_env_file() {
    local env_file="deploy/.env"
    if [[ ! -f "$env_file" ]]; then
        cp .env.example "$env_file"
    fi
    printf "%s" "$env_file"
}

_set_env_line() {
    # Idempotent replace-or-append. If the key already exists with the
    # requested value, skip. Otherwise remove every existing line for
    # that key and append the new one.
    local env_file="$1"
    local key="$2"
    local value="$3"
    if grep -qE "^[[:space:]]*${key}=${value}\$" "$env_file"; then
        return
    fi
    sed -i "/^[[:space:]]*${key}=/d" "$env_file"
    echo "${key}=${value}" >> "$env_file"
    log "  wrote ${key}=${value} to $env_file"
}

_env_has_active_key() {
    local env_file="$1"
    local key="$2"
    [[ -f "$env_file" ]] || return 1
    grep -qE "^[[:space:]]*${key}=" "$env_file"
}

_has_ollama_model() {
    local model="$1"
    ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$model"
}

env_file="deploy/.env"
if _env_has_active_key "$env_file" "KARIN_LLM_MODEL"; then
    log "keeping existing KARIN_LLM_MODEL in $env_file"
elif _has_ollama_model "karin-tuned:latest"; then
    env_file="$(_ensure_env_file)"
    _set_env_line "$env_file" "KARIN_LLM_MODEL" "karin-tuned:latest"
else
    warn "karin-tuned:latest is not present; leaving KARIN_LLM_MODEL unset."
    warn "  The checked-in config fallback will be used unless you set a model"
    warn "  explicitly in $env_file."
fi

if (( WANT_LOCAL_SOVITS == 1 )); then
    env_file="$(_ensure_env_file)"
    _set_env_line "$env_file" "KARIN_TTS_ENABLED" "true"
fi

if [[ "$LOCAL_STT" == "yes" ]]; then
    env_file="$(_ensure_env_file)"
    # Orin-Nano-safe in-process STT: tiny.en on CPU int8. Leaves ~200
    # MiB headroom next to karin-tuned (measured 2026-04-24). Larger
    # variants (base.en / small.en) are covered in docs/jetson-setup.md;
    # this opt-in picks the smallest so setup.sh stays "just works".
    _set_env_line "$env_file" "KARIN_STT_ENABLED" "true"
    _set_env_line "$env_file" "KARIN_STT_MODEL" "tiny.en"
    _set_env_line "$env_file" "KARIN_STT_DEVICE" "cpu"
    _set_env_line "$env_file" "KARIN_STT_COMPUTE_TYPE" "int8"
    log "LOCAL_STT=yes — local Whisper (tiny.en on CPU) enabled"
fi

# --- bringup ----------------------------------------------------------------
#
# Two paths: Docker Compose (preferred) OR native systemd. Both unit files
# and Docker images were set up above. Bring up ONE of them here based on
# Docker availability so we don't end up with two copies of web/sovits
# running side-by-side.

HAS_DOCKER=0
if command -v docker >/dev/null 2>&1; then
    HAS_DOCKER=1
fi

if (( HAS_DOCKER == 1 )); then
    # --- Docker Compose bringup --------------------------------------------
    if (( WANT_LOCAL_SOVITS == 1 )); then
        log "starting Karin stack via Docker Compose (web + local voice profile)"
        (cd deploy && docker compose --profile voice up -d)
    else
        log "starting Karin stack via Docker Compose (web only)"
        (cd deploy && docker compose up -d)
    fi
    log "web UI at http://<jetson-ip>:80 (native container binds 127.0.0.1:80)"
    log "tail logs with: cd deploy && docker compose logs -f web"
    log "systemd units installed but left inactive (Docker is the running path)"
else
    # --- native systemd bringup --------------------------------------------
    warn "docker not installed — falling back to native systemd bringup."
    warn "to switch to Docker later:"
    warn "    curl -fsSL https://get.docker.com | sh"
    warn "    sudo usermod -aG docker $USER_NAME   # then log out and back in"
    warn "    sudo systemctl stop web assistant sovits-server 2>/dev/null || true"
    warn "    (cd deploy && docker compose up -d)"

    if (( WANT_LOCAL_SOVITS == 1 )); then
        log "starting sovits-server.service"
        sudo systemctl start sovits-server.service

        log "waiting for sovits-server to be ready (up to 5 minutes)"
        if ! bash deploy/wait-for-tts.sh http://127.0.0.1:9880 300; then
            die "sovits-server did not come up. Check logs with:
           journalctl -fu sovits-server
       To retry: sudo systemctl stop sovits-server && LOCAL_SOVITS=yes bash deploy/setup.sh"
        fi

        log "running TTS smoke test"
        if ! "$BRIDGE_VENV/bin/python" scripts/test_tts.py; then
            die "TTS smoke test failed. Check logs with:
           journalctl -fu sovits-server
       sovits-server is still running for debugging.
       To retry: LOCAL_SOVITS=yes bash deploy/setup.sh"
        fi
        log "smoke test OK — tmp/tts_test.wav produced"

        # Decide whether to run the interactive audition pause.
        should_audition=0
        case "$AUDITION" in
            yes) should_audition=1 ;;
            no)  should_audition=0 ;;
            auto)
                if [[ -t 0 ]]; then
                    should_audition=1
                else
                    log "stdin is not a TTY — skipping audition prompt (AUDITION=auto)."
                fi
                ;;
        esac

        if (( should_audition == 1 )); then
            echo ""
            echo "=============================================================="
            echo " TTS smoke test produced: tmp/tts_test.wav"
            echo ""
            echo " LISTEN TO IT NOW. Does it sound like your trained voice?"
            echo ""
            echo " Press Enter to start the bridge, or Ctrl-C to abort."
            echo "=============================================================="
            read -r _

            log "starting assistant.service (voice loop) and web.service (FastAPI UI)"
            sudo systemctl start assistant.service
            sudo systemctl start web.service
            log "voice loop running. tail logs with: journalctl -fu assistant"
            log "web UI at http://<jetson-ip>:8001"
        else
            log ""
            log "assistant.service + web.service are enabled but not started (AUDITION=$AUDITION)."
            log "after verifying tmp/tts_test.wav, start them manually:"
            log "    sudo systemctl start assistant    # physical-mic voice loop"
            log "    sudo systemctl start web          # browser PTT / chat UI"
            log "    journalctl -fu assistant"
        fi
    else
        log "starting web.service (text-only UI)"
        sudo systemctl start web.service
        log "web UI at http://<jetson-ip>:8001"
        log "tail logs with: journalctl -fu web"
    fi
fi

# --- Tailscale Serve: HTTPS proxy under the current MagicDNS name -----------
#
# Binds https://<current-device-name>.<tailnet>.ts.net → the upstream port
# that our web UI is actually listening on (Docker: 127.0.0.1:80, systemd:
# 0.0.0.0:8001). Re-run safe — `serve reset` clears any config bound to a
# previous device name, then we re-provision. Tailscale handles the Let's
# Encrypt cert automatically on first TLS handshake.
#
# In Docker mode the web container binds 127.0.0.1:80 + CAP_NET_BIND_SERVICE
# so plain HTTP is localhost-only; Tailscale Serve is the ONLY way in.
# In systemd mode uvicorn binds 0.0.0.0:8001, so :8001 IS reachable on the
# tailnet directly — serve still works, it's just not the sole entry point.
if (( HAS_DOCKER == 1 )); then
    SERVE_UPSTREAM="http://127.0.0.1:80"
else
    SERVE_UPSTREAM="http://127.0.0.1:8001"
fi

if command -v tailscale >/dev/null 2>&1; then
    log "provisioning Tailscale Serve (HTTPS -> ${SERVE_UPSTREAM})"
    sudo tailscale serve reset >/dev/null 2>&1 || true
    if sudo tailscale serve --bg "$SERVE_UPSTREAM"; then
        current_name=$(tailscale status --self --json 2>/dev/null \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))" \
            2>/dev/null || echo "<this host>")
        log "serve active: https://${current_name}/ → ${SERVE_UPSTREAM}"
    else
        warn "tailscale serve failed — HTTPS will not be available. Fix with:"
        warn "    sudo tailscale serve reset"
        warn "    sudo tailscale serve --bg ${SERVE_UPSTREAM}"
    fi
else
    warn "tailscale CLI not found — skipping HTTPS serve setup."
    warn "web is listening at ${SERVE_UPSTREAM}; reach it over your LAN or"
    warn "set up Tailscale / a reverse proxy yourself."
fi

# --- Voice subsystem status + reachability summary --------------------------
#
# At the end of setup, report what the voice stack will actually look like
# when the web service starts: local / remote / off for each of TTS + STT,
# plus a live reachability probe on any PC-side URLs already in deploy/.env.
# Non-fatal — a PC being offline at setup time is fine, the user may boot
# it after; we just want a clear signal.

_get_env_value() {
    local env_file="$1"
    local key="$2"
    [[ -f "$env_file" ]] || return
    grep -E "^[[:space:]]*${key}=" "$env_file" \
        | tail -n 1 \
        | cut -d= -f2- \
        | sed -E 's/^[[:space:]]*|[[:space:]]*$//g; s/^"//; s/"$//'
}

_probe_url() {
    # Returns 0 if the URL responds (any 2xx/3xx/4xx) within 3s, 1 otherwise.
    # Uses -o /dev/null + -w "%{http_code}" so we can tolerate 404s on
    # legacy sidecars while still distinguishing connection failures.
    local url="$1"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null || echo "000")
    [[ "$code" != "000" ]]
}

_env_file_final="deploy/.env"
_tts_enabled=$(_get_env_value "$_env_file_final" KARIN_TTS_ENABLED)
_stt_enabled=$(_get_env_value "$_env_file_final" KARIN_STT_ENABLED)
_tts_url=$(_get_env_value "$_env_file_final" KARIN_TTS_BASE_URL)
_stt_url=$(_get_env_value "$_env_file_final" KARIN_STT_BASE_URL)

# Resolve mode per subsystem: off / local / remote
_resolve_mode() {
    local enabled="$1"
    local url="$2"
    local local_flag="$3"
    if [[ "$enabled" != "true" ]]; then
        printf "off"
        return
    fi
    if [[ -n "$url" ]]; then
        printf "remote"
        return
    fi
    if [[ "$local_flag" == "1" ]]; then
        printf "local"
        return
    fi
    printf "enabled-but-unconfigured"
}

_tts_mode=$(_resolve_mode "$_tts_enabled" "$_tts_url" "$WANT_LOCAL_SOVITS")
_local_stt_flag=0
[[ "$LOCAL_STT" == "yes" ]] && _local_stt_flag=1
_stt_mode=$(_resolve_mode "$_stt_enabled" "$_stt_url" "$_local_stt_flag")

# Reachability probes for whatever URLs are configured
_tts_reach=""
if [[ -n "$_tts_url" ]]; then
    if _probe_url "${_tts_url%/}/health" || _probe_url "${_tts_url%/}/stt/status"; then
        _tts_reach=" (reachable)"
    else
        _tts_reach=" (UNREACHABLE — sidecar down?)"
    fi
fi
_stt_reach=""
if [[ -n "$_stt_url" ]]; then
    if _probe_url "${_stt_url%/}/stt/status"; then
        _stt_reach=" (reachable)"
    else
        _stt_reach=" (UNREACHABLE — sidecar down?)"
    fi
fi

log ""
log "============================================================"
log " Voice subsystem summary"
log "   TTS: ${_tts_mode}${_tts_reach}"
log "   STT: ${_stt_mode}${_stt_reach}"
log "============================================================"
if [[ "$_tts_reach" == *UNREACHABLE* || "$_stt_reach" == *UNREACHABLE* ]]; then
    warn "A PC-side voice URL was configured but did not respond. The web UI"
    warn "will still start; transcription/synthesis calls will fail until"
    warn "the sidecar is up. Fix:"
    warn "  1. Check the PC is awake and Tailscale is connected both ways."
    warn "  2. On the PC: cd deploy/pc-tts && start.bat (or check tray icon)."
    warn "  3. Re-run: curl -s <url>/stt/status   to confirm reachability."
fi
if [[ "$_tts_mode" == "off" && "$_stt_mode" == "off" ]]; then
    log "(Running TEXT-ONLY. Enable STT during setup with"
    log "  LOCAL_STT=yes bash deploy/setup.sh"
    log " or point at a PC sidecar via KARIN_STT_BASE_URL / KARIN_TTS_BASE_URL"
    log " in deploy/.env — see deploy/pc-tts/README.md.)"
fi

log "setup.sh done."
