# Karin — operational runbook

Deploy, run, and recover. Designed so a fresh host goes from `git clone`
to a working text assistant with **just one command**. Voice (STT + TTS)
is optional — add trained weights any time and re-run setup to turn it on.

> 🆕 **Provisioning a fresh Jetson?** The from-zero walkthrough lives in
> **[docs/jetson-setup.md](docs/jetson-setup.md)** —
> [Part 1 — Flash JetPack 6](docs/jetson-setup.md#part-1--flash-jetpack-6)
> via SDK Manager (skip if already flashed) and
> [Part 2 — Provision Karin on the Jetson](docs/jetson-setup.md#part-2--provision-karin-on-the-jetson)
> (Docker, GUI strip, Ollama, repo, optional voice, HTTPS). Come back
> here for ongoing operations once it's up.

## Contents

- [What you need on the host before `docker compose up`](#what-you-need-on-the-host-before-docker-compose-up)
- [Pick your platform once](#pick-your-platform-once)
- [Hardware sizing (which model fits)](#hardware-sizing-which-model-fits)
- [Performance tuning (latency)](#performance-tuning-latency)
- [First-time deploy](#first-time-deploy)
- [Cross-platform summary](#cross-platform-summary)
- [Tailscale HTTPS (port 443 on, port 80 off)](#tailscale-https-port-443-on-port-80-off)
- [Updating](#updating)
- [Stopping](#stopping)
- [Pinned versions](#pinned-versions)
- [Optional host configs (persist across container rebuilds)](#optional-host-configs-persist-across-container-rebuilds)
- [Health and state inspection](#health-and-state-inspection)
- [First-run behavior (derived labels)](#first-run-behavior-derived-labels)
- [Resetting / reseeding](#resetting--reseeding)
- [Tools Karin can call](#tools-karin-can-call)
- [Troubleshooting](#troubleshooting)
- [What to commit and what to keep local](#what-to-commit-and-what-to-keep-local)
- [Falling back to the host `setup.sh` path](#falling-back-to-the-host-setupsh-path)

## What you need on the host before `docker compose up`

For a text-only deploy (default), two things:

1. **Docker Engine + NVIDIA Container Toolkit.**
   - JetPack 6 has the toolkit pre-installed but you still need to
     install Docker itself. See [Install Docker Engine](docs/jetson-setup.md#install-docker-engine) for the exact commands.
   - Verify: `docker info | grep -i runtimes` should list `nvidia`.
   - On x86 Ubuntu: [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

1b. **PyTorch with CUDA + cuSPARSELt on the host Python.** `deploy/setup.sh`
   creates a Python virtualenv for the bridge with `--system-site-packages`
   so it can reuse your system torch rather than redownloading it into
   the venv. If system `python3 -c "import torch; print(torch.cuda.is_available())"`
   doesn't print `True`, setup.sh fails fast with an install hint.

   Check your JetPack version first — `head -1 /etc/nv_tegra_release`
   shows L4T release (e.g. `# R36 (release), REVISION: 4.0` = JetPack 6.2,
   CUDA 12.6). The steps below are verified on **JetPack 6.2 (R36.4/5,
   CUDA 12.6, Python 3.10)**. JetPack 6.1 works with the same wheel
   (CUDA 12.6 ABI-compatible). For 6.0 (CUDA 12.2) see the notes below.

   **Step 1 — install the NVIDIA PyTorch wheel.** Preferred: direct
   wheel from NVIDIA's CDN (stable, doesn't depend on community DNS).
   ~807 MB download, goes to `~/.local/lib/python3.10/site-packages/`:

   ```bash
   pip3 install --user --no-cache \
       'https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl'
   ```

   Fallback if you need `torchvision` / `torchaudio` too, or the direct
   URL 404s (NVIDIA occasionally rotates filenames): use the
   jetson-ai-lab community index. Note the TLD is `.io`, not `.dev`
   (the old `.dev` domain no longer resolves):

   ```bash
   pip3 install --user --no-cache \
       --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
       torch torchvision torchaudio
   ```

   **Step 2 — install cuSPARSELt.** Torch 2.5 on Jetson needs
   `libcusparseLt.so.0`, which JetPack 6.2 does NOT ship. Without it,
   `import torch` crashes with `ImportError: libcusparseLt.so.0:
   cannot open shared object file`. Install once via NVIDIA's tegra
   repo (~5 MB):

   ```bash
   cd /tmp
   wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
   sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
   sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get install -y libcusparselt0 libcusparselt-dev
   ```

   **Step 3 — verify.** Should print `<version> True`:

   ```bash
   python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   # Expected: 2.5.0a0+872d972e41.nv24.08 True
   ```

   A `UserWarning: Failed to initialize NumPy ... API version 0x10 but
   this version of numpy is 0xe` is harmless — system numpy is older
   than torch 2.5 expects, but the bridge venv installs numpy 1.26.4
   which resolves it at serve time.

   **Why `--user`, not a venv**: Installing under `~/.local/lib/python3.10/site-packages/`
   means every venv you make with `--system-site-packages` (which is
   what `setup.sh` creates) inherits torch for free. A pure-venv install
   would duplicate ~500 MB of torch weights per venv AND get wiped
   every time you `rm -rf ~/Karin/.venv/`, which our cleanup flow does.
   The `--user` install survives `rm -rf ~/Karin`.

   **Do NOT** `pip install torch` from default PyPI — you'll get a
   CPU-only x86 wheel (or a "no matching distribution" error on ARM64).
   NVIDIA's CDN + jetson-ai-lab `.io` are the canonical sources for
   Jetson wheels.

   **JetPack 6.0 (CUDA 12.2) note**: the 2.5 wheel above is CUDA 12.6,
   so it won't work. Use the jetson-ai-lab `.io` index with `jp6/cu122`
   instead, OR the older 2.3 wheel — the NVIDIA forum thread
   [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
   keeps canonical URLs per release.

For voice (optional — skip unless you want TTS/STT), pick a path:

2a. **PC-TTS offload (recommended for Orin Nano 8 GB).** GPT-SoVITS runs
   on a PC over Tailscale; the Jetson only sends text and receives WAV.
   Voice bundle lives on the PC, not the Jetson. See
   [deploy/pc-tts/README.md](deploy/pc-tts/README.md) for the sidecar
   setup, then set `KARIN_TTS_BASE_URL=http://<pc-ip>:9880` and
   `KARIN_TTS_ENABLED=true` in `deploy/.env`.

2b. **Local sovits (Orin NX 16 GB or larger only).** The whole stack
   runs on the Jetson. Opt in with `LOCAL_SOVITS=yes bash deploy/setup.sh`
   — `setup.sh` detects the voice bundle and auto-installs steps 3+4
   below. Skip this option on Orin Nano 8 GB; it does not fit.

3. **(local sovits only) At least one complete character voice bundle**
   in `characters/<name>/voices/` on the Jetson:
   - `ref.wav`
   - `gpt_model_*.ckpt`
   - `sovits_model_*.pth`
   The bundle metadata lives in `characters/<name>/voice.yaml`.
   See [docs/training-guide.md](docs/training-guide.md) and
   [characters/README.md](characters/README.md).

4. **(local sovits only) The GPT-SoVITS clone + upstream pretrained models**
   at `third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models/`. Bootstrap:

   ```bash
   git clone https://github.com/RVC-Boss/GPT-SoVITS.git third_party/GPT-SoVITS
   cd third_party/GPT-SoVITS
   git checkout 2d9193b0   # pinned commit — bridge/tts.py is written against this
   cd ../..

   # Download the ~4 GB upstream weights (one-time). Requires python3 + huggingface_hub:
   pip3 install --user "huggingface_hub<1.0"
   python3 - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="lj1995/GPT-SoVITS",
       local_dir="third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models",
       local_dir_use_symlinks=False,
   )
   PY
   ```

   The Docker stack bind-mounts this directory read-only into the
   `sovits` service. `LOCAL_SOVITS=yes bash deploy/setup.sh` handles
   steps 3+4 automatically when the voice bundle is already present.

Everything else — Python 3.12, ffmpeg, CUDA torch, the full Python
dependency graph, models — all installed inside containers with pinned
versions. The one host-side service is **Ollama** (next section
explains why).

## Pick your platform once

Copy the example env file and uncomment the line for your host:

```bash
cp .env.example .env
# edit .env: uncomment ONE COMPOSE_FILE line
```

| Host | Uncomment |
|---|---|
| Jetson Orin (ARM64 + JetPack 6) — **the deploy target** | `deploy/docker-compose.yml:deploy/docker-compose.jetson.yml` |
| Desktop Linux with NVIDIA GPU — dev / debug | `deploy/docker-compose.yml:deploy/docker-compose.gpu.yml` |
| CPU-only (slow sanity check anywhere) | `deploy/docker-compose.yml` |

What the overlays actually change:
- `docker-compose.jetson.yml` — swaps both custom Dockerfiles onto
  `dustynv/l4t-pytorch:r36.2.0` (ARM64, JetPack torch pre-installed)
  and enables GPU passthrough.
- `docker-compose.gpu.yml` — uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
  for GPT-SoVITS, enables GPU passthrough, sets Whisper to `float16`.
- Base file alone — CPU-only Whisper (`KARIN_STT_DEVICE=cpu`,
  `compute_type=int8`), no GPU reservations.

Windows users debugging on their local machine: Docker Desktop's GPU
story is still messy (needs WSL2 + NVIDIA Container Toolkit inside
WSL). Easier to use the host-venv flow described in
[docs/pc-setup.md](docs/pc-setup.md) for Windows dev, and the Docker
flow for Jetson production.

## Hardware sizing (which model fits)

Production model is **`karin-tuned:latest`** — an iter-3 LoRA merged on
`mannix/llama3.1-8b-abliterated:tools-q4_k_m` (~4.9 GB). All the eval
numbers (93.3% routing / 91.9% reply / 59.2% tool-output on 135-case)
are measured against this model. Drop-in alternatives listed below are
pre-LoRA baselines; they work, but routing accuracy drops ~20 pp.

| Hardware | Available VRAM | Primary (current prod) | Alternatives | Notes |
|---|---|---|---|---|
| Jetson Orin Nano 8 GB | 7.4 GiB usable total; **~600 MiB free at idle with karin-tuned loaded** (measured 2026-04-24: 6.8 / 7.4 GiB RAM, 788 MiB into swap, 9%/91% CPU/GPU spill) | **`karin-tuned:latest`** (iter-3 LoRA, ~4.9 GB weights + ~300 MB KV at num_ctx=2048) | `qwen2.5:3b` (Q4_K_M, ~2.0 GB) if karin-tuned isn't built yet — frees ~3 GB for headroom at the cost of ~20 pp routing accuracy | karin-tuned requires `num_ctx=2048` and sovits OFF (PC-TTS offload or Voice toggle). ~3-5 s warm turns. STT disabled by default to keep the ~600 MiB headroom. |
| Jetson Orin NX 16 GB | ~12 GB | **`karin-tuned:latest`** | `huihui_ai/qwen3.5-abliterated:4b` (untuned, ~2.5 GB) | Can run sovits concurrently via `LOCAL_SOVITS=yes`. |
| Jetson AGX Orin 32/64 GB | ~24 GB+ | `karin-tuned:latest` | Any 7B+ base you want to fine-tune | Plenty of headroom. |
| Desktop GPU (≥12 GB) | per card | `karin-tuned:latest` | 4B or 7B base models | Use the `gpu` overlay. |
| CPU-only | n/a | `karin-tuned:latest` (slow) | 3B base model | Dev / smoke testing only. |

**Memory levers, in order of impact** (each cuts shared-VRAM usage):

1. **Stop sovits when not using voice** — click the **Voice** toggle
   in the sidebar header (or `docker compose stop sovits`). Frees
   ~2 GB instantly. Text turns stay fast; voice replies fail-soft
   ("voice not available — TTS server is offline" footer).
2. **System-service trim** (~1 GB freed): on a headless Jetson, disable
   snap/bluetooth/cups/ModemManager/fwupd. See [docs/jetson-setup.md](docs/jetson-setup.md#strip-the-desktop-environment-optional).
3. **CLI mode** — `sudo systemctl set-default multi-user.target`
   frees ~1-2 GB used by GNOME / display manager.
4. **KV cache q8_0** — Ollama env `OLLAMA_KV_CACHE_TYPE=q8_0` halves
   the KV cache memory at minor quality cost.
5. **Flash attention** — Ollama env `OLLAMA_FLASH_ATTENTION=true`
   cuts attention working memory by ~30%.
6. **Drop `num_ctx`** — `KARIN_NUM_CTX=2048` cuts the KV cache linearly,
   but watch Ollama logs for `truncating input prompt limit=...` —
   our system prompt is ~3000 chars and gets sliced under 2048.
   3072 is the sweet spot for the current Karin character file.
7. **`OLLAMA_GPU_OVERHEAD`** — reserve N bytes for sovits before
   Ollama plans its layout (e.g. `OLLAMA_GPU_OVERHEAD=536870912` =
   500 MB). Forces partial CPU spill rather than OOM when both
   models are loaded. Currently set to 512 MB to leave room for
   sovits's startup spikes when the toggle is flipped.
8. **Lower-bit model variant** — `qwen2.5:3b-instruct-q3_K_S` is
   ~30% smaller than the default `q4_K_M` at modest quality cost.

## Performance tuning (latency)

Order of operations on a Jetson Orin Nano 8 GB:

1. **Browser sees first reply token in ~1-2s** thanks to the new
   streaming pipeline (`bridge/llm.py`'s `chat_stream` + the
   `token_delta` events flowing through the job queue).
2. **Total turn time** depends on tools used:
   - No tools (chat): **5-10s**
   - One tool (e.g. `get_weather`): **10-20s**
   - Multi-step tool chains: **20-40s**
3. **Model stays GPU-resident forever** between turns
   (`keep_alive: -1` in every Ollama call), so the 10-15s cold-load
   penalty is paid once per server start.

Optimization levers we haven't pulled yet (each is its own session):

- **CUDA `ctranslate2` build** for Jetson — drops STT from ~3s to
  ~0.5s per second of audio. Currently we use the PyPI ARM64
  CPU-only wheel because building from source against JetPack's
  CUDA is a 30-45 min compile.
- **Speculative decoding** — Ollama 0.4+ supports it; pair the 2B
  with a 0.5B drafter for ~2× streaming speedup.
- **NVIDIA Riva STT** — Jetson-native, dramatically faster than
  Whisper. Heavier install + more setup but supported.
- **Wake word + full-duplex audio** — replace push-to-talk with
  always-listening "hey Karin" via openWakeWord; AEC + speaker
  ducking so users can interrupt mid-sentence.

## First-time deploy

Two paths depending on whether you want one-command provisioning
(`setup.sh`, recommended) or direct Docker Compose. Both assume the
repo is cloned and you're standing in the repo root — see **Step 0**
below if you haven't cloned yet.

### Step 0 — get the Karin source onto the Jetson

```bash
cd ~
git clone https://github.com/kaminglui/Karin.git
cd Karin
```

GitHub requires a Personal Access Token or SSH key for HTTPS clones —
password auth was dropped in 2021. Set up a PAT at
<https://github.com/settings/tokens> (classic, scope: `repo`) and use
it as the password when `git clone` prompts, or register an SSH key
and clone via `git@github.com:…` instead.

### Recommended: `bash deploy/setup.sh`

```bash
cd ~/Karin
sudo -v                       # cache sudo so prompts don't stall
bash deploy/setup.sh
```

What it does (idempotent, safe to re-run):

1. Sanity-check JetPack + ARM64 + CUDA torch.
2. Install apt prereqs (ffmpeg, build-essential, python3-venv, …).
3. Install Ollama + pull `mannix/llama3.1-8b-abliterated:tools-q4_k_m`
   from `config/ollama_models.txt` (~4.9 GB, ~3-5 min).
4. Pull `karin-lora.gguf` from
   [huggingface.co/kaminglui/karin-lora](https://huggingface.co/kaminglui/karin-lora)
   and build `karin-tuned:latest` in Ollama via a Modelfile with
   `ADAPTER ./karin-lora.gguf`. Fail-soft — setup continues if HF is
   unreachable; the mannix base is still usable.
5. Create `.venv/bridge/` + install `bridge/requirements.txt`.
6. Install systemd units (`web.service` always; `sovits-server.service`
   + `assistant.service` only when `LOCAL_SOVITS=yes`).
7. Run `docker compose up -d` if Docker is available (preferred) OR
   start the native systemd units.
8. Provision Tailscale Serve (HTTPS → localhost upstream).
9. Print a **voice subsystem summary** showing TTS + STT mode
   (`off` / `local` / `remote`) and reachability of any configured
   PC-side URLs.

Opt-in flags (both default `no`):

```bash
# Enable local STT with tiny.en on CPU (fits next to karin-tuned):
LOCAL_STT=yes bash deploy/setup.sh

# Enable local sovits TTS (Orin NX 16 GB+ only — won't fit on Nano):
LOCAL_SOVITS=yes bash deploy/setup.sh
```

### Direct Docker Compose (if you've already bootstrapped Ollama)

```bash
cd ~/Karin

# One-time: pick your platform by uncommenting ONE COMPOSE_FILE line.
cp .env.example .env
sed -i 's|^# COMPOSE_FILE=deploy/docker-compose.yml:deploy/docker-compose.jetson.yml|COMPOSE_FILE=deploy/docker-compose.yml:deploy/docker-compose.jetson.yml|' .env
sed -i 's|^COMPOSE_FILE=deploy/docker-compose.yml$|# &|' .env

docker compose up -d
```

**Without this `.env` step**, docker-compose reads only the base
`deploy/docker-compose.yml` (CPU-only). You'll get no GPU acceleration
and the LLM will effectively not run on Orin Nano.

First run behavior:

- **Ollama is expected to be running on the HOST** (not in a container)
  by default — `setup.sh` installs it that way. The `web` container
  reaches it via `host.docker.internal` / `127.0.0.1`. The
  `docker-ollama` profile exists as an alternative if you don't want
  Ollama on the host, but then you must explicitly
  `docker compose --profile docker-ollama up -d`.
- **sovits** is gated behind the `voice` profile and only runs when
  `LOCAL_SOVITS=yes` wired it in OR you pass `--profile voice`.
  Default deploys run text-only — sovits is NOT started.
- **web** starts `uvicorn web.server:app` bound to **`127.0.0.1:80`**
  (inside the host-networked container, CAP_NET_BIND_SERVICE grants
  the low port). This is localhost-only by design — Tailscale Serve
  (provisioned by setup.sh) proxies :443 → :80 so the UI is reachable
  over HTTPS via your tailnet. **If you started via `setup.sh` you do
  NOT need to visit `http://<ip>:80` directly — open
  `https://<your-jetson>.<tailnet>.ts.net/`.**
- **Native systemd fallback** (when Docker isn't installed): web.service
  binds `0.0.0.0:8001` instead. The Tailscale Serve provisioning in
  `setup.sh` auto-switches upstream to `:8001` in that case.

Monitor:

```bash
docker compose ps                    # healthcheck status for all services
docker compose logs -f web           # web UI + bridge logs
docker compose logs -f sovits        # TTS load + synth activity
docker compose logs -f ollama        # LLM turns

# OR, if on the native systemd path:
journalctl -fu web
journalctl -fu ollama
```

### Optional: Open WebUI alongside Karin

```bash
docker compose --profile webui up -d
```

Open WebUI runs on `:8080` (not `:80`) — it's an independent chat UI
for the raw Ollama model, not Karin's bridge. Useful for debugging
tool-routing separately from the bridge layer.

## Cross-platform summary

The PC → Jetson path is "same repo, different `COMPOSE_FILE` line":

```bash
# On your x86 debug box (with NVIDIA GPU):
grep '^COMPOSE_FILE' .env
# -> COMPOSE_FILE=deploy/docker-compose.yml:deploy/docker-compose.gpu.yml
docker compose up -d

# Push the repo to Jetson (rsync, scp, or git):
rsync -av --exclude .venv --exclude third_party --exclude data/conversations \
      ./ jetson:~/Karin/

# On the Jetson:
ssh jetson
cd ~/Karin
cp .env.example .env
# uncomment the jetson COMPOSE_FILE line
docker compose up -d
```

Same Dockerfiles, same compose base, same pinned Python deps. The
only differences between PC and Jetson come from the overlay files.

## Tailscale HTTPS (port 443 on, port 80 off)

Karin is served over HTTPS on the tailnet using **Tailscale Serve**,
which terminates TLS with an auto-managed Let's Encrypt cert bound to
the device's current MagicDNS name. Plain HTTP on the tailnet IP is
intentionally closed.

### How it's wired

- **uvicorn inside the container** binds `127.0.0.1:80` (not
  `0.0.0.0:80`). See [`deploy/docker-compose.yml`](deploy/docker-compose.yml)
  under the `web.command` line. Nothing on the tailnet reaches :80
  directly — `curl http://<tailscale-ip>/` returns connection refused.
- **`tailscale serve`** on the host listens on :443 and proxies to
  `http://127.0.0.1:80`, which is where uvicorn actually is.
- Clients hit `https://<device>.<tailnet>.ts.net/` and land on Karin.

### Automatic provisioning on fresh deploy

Running `bash deploy/setup.sh` provisions Tailscale Serve as its final
phase. The script:

1. Calls `tailscale serve reset` to drop any stale binding from a
   previous device name (important after a device rename — the old
   name would otherwise keep its serve config).
2. Calls `tailscale serve --bg http://127.0.0.1:80` to (re)bind HTTPS
   to the **current** Tailscale DNSName — read directly from
   `tailscale status --self --json → Self.DNSName`, so it works no
   matter what the OS `/etc/hostname` is.
3. Logs the resolved URL.

The serve config persists across reboots (Tailscale stores it), so you
don't need to re-run setup.sh unless you rename the device or want to
reset the cert.

### Renaming the device

1. Rename via [login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines)
   or `tailscale set --hostname=<new-name>`. The IP stays the same.
2. Re-run the two commands (same ones `setup.sh` uses):
   ```bash
   sudo tailscale serve reset
   sudo tailscale serve --bg http://127.0.0.1:80
   ```
3. `tailscale serve status` should now show the new DNSName.
4. Browser hits to `https://<new-name>.<tailnet>.ts.net/` work
   immediately — Tailscale provisions a fresh cert on the first
   TLS handshake (~5 s).

### Verifying the setup

```bash
# On the Jetson:
tailscale serve status                               # expected proxy line
curl -sfS http://127.0.0.1/api/health                # local uvicorn: 200
curl -sfS -m 3 http://<tailscale-ip>/api/health      # should REFUSE (port closed)

# From any tailnet client:
curl -sfS https://<device>.<tailnet>.ts.net/api/health  # should 200
```

### Re-opening plain HTTP (if you ever need it)

Edit `deploy/docker-compose.yml` — change the web service command from
`--host 127.0.0.1` back to `--host 0.0.0.0`, then
`docker compose up -d web`. You'd do this only if you're running on a
LAN without Tailscale, or want to put a different reverse proxy in
front.

## Updating

Code changes:
```bash
docker compose build web             # rebuild just the bridge image
docker compose up -d web             # restart with new image
```

Upstream updates (Ollama, GPT-SoVITS):
```bash
docker compose pull                  # Ollama + Open WebUI
# GPT-SoVITS clone is bind-mounted; pull + rebuild:
cd third_party/GPT-SoVITS && git pull && cd ../..
docker compose build sovits
docker compose up -d sovits
```

## Stopping

```bash
docker compose down                  # stop; keep volumes (models, conversations)
docker compose down -v               # stop + wipe open-webui volume too
```

Bind-mounted data (voice weights, conversations, memory, news/trackers
state) is always preserved — it lives on the host filesystem, not in
Docker volumes.

## Pinned versions

| Component | Version | Why pinned |
|---|---|---|
| Python (container) | 3.12 | matches the x86 dev venv; not 3.14 (Pillow) |
| Ollama | `ollama/ollama:0.4.7` | tool-calling stable; upgrades break rarely but do break |
| Qwen | `huihui_ai/qwen3.5-abliterated:4b` | docker-ollama pre-pull tag; useful on larger Jetsons / desktop GPUs |
| GPT-SoVITS | commit `2d9193b0` | [bridge/tts.py](bridge/tts.py) schema is written against this exact api_v2 |
| torch (x86) | `>=2.4` via CUDA wheel index | Whisper ctranslate2 compat |
| torch (Jetson) | JetPack-supplied wheels | PyPI wheels on ARM64 are CPU-only |
| faster-whisper | `1.2.1` | int8_float16 compute type works on sm_87 |
| ctranslate2 | `4.7.1` | faster-whisper backend; pinned together |
| silero-vad | `6.2.1` | 5.x API changed; 6.x expects exactly 512-sample 16k frames |
| Everything else | see [bridge/requirements.txt](bridge/requirements.txt) | exact `==` pins |

All Python deps use exact `==` pins so rebuilds reproduce. Update by
editing the file + `docker compose build web`.

## Optional host configs (persist across container rebuilds)

Configs live on the host and are bind-mounted read-only. Safe to edit.
Your personal copies are gitignored.

### News watchlists (optional)

```bash
cp bridge/news/config/preferences.example.json bridge/news/config/preferences.json
# edit regions / topics / events
docker compose restart web
```

Without this, news ranking falls back to state + recency only.

### Trackers (optional — ships with sensible defaults)

```bash
cp bridge/trackers/config/trackers.example.json bridge/trackers/config/trackers.json
# enable/disable individual trackers
docker compose restart web
```

Without this, the default list (USD/CNY, USD/HKD, USD/JPY, gold, RBOB
gasoline, two CPI series) is used.

### Alerts

No user config. Rules are code in `bridge/alerts/rules.py`.

### Background pollers (news + alerts)

Two daemon threads start at web boot (see [bridge/pollers.py](bridge/pollers.py)):

- **News poller** — every 20 min, calls `ingest_latest(force=True)`
  which pulls RSS feeds plus (if `KARIN_NEWSAPI_KEY` is set) wire-tier
  sources from NewsAPI. Stays under NewsAPI's 100/day free quota.
- **Alerts poller** — every 10 min, calls `AlertService.scan(force=True)`
  which collects tracker / news / travel-advisory / NWS signals and
  runs the rule engine. NWS is keyless; queries `api.weather.gov/alerts/active?point=<lat,lon>`
  with the user's IP-detected coordinates.

Disable the pollers entirely: comment out `app.on_event("startup")`
hooks in [web/server.py](web/server.py), or remove them for a deploy
that only wants on-demand refresh.

Check poller health: `docker logs karin-web 2>&1 | grep poller`.

### NewsAPI (optional — enables wire-tier confirmation)

Without `KARIN_NEWSAPI_KEY`, stories cap at `PROVISIONALLY_CONFIRMED`
because our RSS feeds are all REPUTABLE tier. Register a free key at
https://newsapi.org/register and add:

```bash
# in ~/Karin/.env
KARIN_NEWSAPI_KEY=<your key>
```

Wire-tier sources (AP, Reuters, Bloomberg) are registered in
[bridge/news/config/sources.json](bridge/news/config/sources.json).
Add more NewsAPI sources by appending an entry with `tier: wire` +
`is_wire_service: true` and the exact NewsAPI source id.

**Free-tier caveats:** 100 requests/day (20-min polling stays under),
articles up to 24h delayed, "development use" per their TOS.

### NWS weather alerts (no key needed)

Severe-weather alerts for the user's IP-detected coordinates flow
through `get_alerts` automatically — no config, no key. Filters:

- Severity ≥ Moderate (Minor advisories are suppressed; tune in
  [bridge/alerts/nws_fetch.py](bridge/alerts/nws_fetch.py) `SIGNIFICANT_SEVERITIES`)
- Per-alert cooldown: 6h (so same storm doesn't refire across polls)
- AlertLevel mapping: Extreme→CRITICAL, Severe→ADVISORY, Moderate→WATCH

### Third-party API keys (optional)

Some tools can use paid / rate-limited APIs when you provide a key.
All are optional — the bridge degrades gracefully when a key is
missing (falls back to the no-key alternative or disables the tool).

```bash
cp config/api_keys.example.json config/api_keys.json
# edit config/api_keys.json — paste keys for services you want to enable
docker compose restart web    # if running; otherwise just start
```

Supported slots (see [config/api_keys.example.json](config/api_keys.example.json) for
signup links + per-service details):
`brave_search`, `spoonacular`, `openweathermap`, `newsapi`, `tavily`,
`alpha_vantage`, `bls`, `huggingface`, `anthropic`, `openai`.

Env-var override: `KARIN_API_KEY_<UPPER>` (e.g. `KARIN_API_KEY_BRAVE_SEARCH=...`)
wins over the JSON file. Useful when you inject secrets via Docker
environment rather than a mounted file.

### Character / system prompt (optional)

Swap the LLM's persona without editing `assistant.yaml`:

```bash
# List shipped character bundles
ls characters/

# Make your own
mkdir -p characters/my_assistant/voices
cp characters/template/voice.example.yaml characters/my_assistant/voice.yaml
$EDITOR characters/my_assistant/voice.yaml
# drop ref.wav + gpt_model_*.ckpt + sovits_model_*.pth into
# characters/my_assistant/voices/

# Activate by editing assistant.yaml's top-level `character:` key,
# or use the env override (no file edit needed):
KARIN_CHARACTER=my_assistant docker compose up -d --force-recreate web
```

Current deploys use `characters/profile.yaml` as the shared prompt
template and fill `{persona}` / `{language_note}` from each bundle's
`voice.yaml`. See [characters/README.md](characters/README.md) for the
current layout. [config/characters/README.md](config/characters/README.md)
now documents the legacy fallback path only.

Private character bundles can live outside git or be ignored via your
own local exclude rules if you don't want personal voices/personas in
commits.

## Health and state inspection

```bash
# Read-only one-screen snapshot of all subsystems (configs present,
# reading counts, persistent fetch errors, advisory baseline status,
# alert counts).
docker compose exec web python -m bridge.status
```

## First-run behavior (derived labels)

Most "smart" outputs need history before they activate.

**Tracker derived labels** (Phase 5.2 — direction, movement, shock):

| Label | History needed |
|---|---|
| `direction_1d` (volatility-aware) | ~11 daily readings |
| `direction_1d` (fallback 0.25%) | as soon as `change_1d_pct` is computed |
| `shock_label` | ~16 daily readings |
| `movement_label` | ~26 daily readings |

Until then, trackers still report values + simple deltas.

**Alert rules** that depend on tracker labels inherit those minimums.
Expect ~2-4 weeks of continuous polling before the full alert behavior
matures.

**News watchlist alerts** fire as soon as `preferences.json` exists
with watchlists AND news has fetched a matching cluster.

**Travel advisory alerts** fire only on changes from the baselined
state — day-one is silent by design.

## Resetting / reseeding

All subsystem state lives on the host under `data/` and
`bridge/{news,trackers,alerts}/data/`. Configs are never touched by a
reset.

```bash
# Re-baseline travel advisory state
rm bridge/alerts/data/advisory_state.json

# Wipe alert audit log + cooldowns
rm -rf bridge/alerts/data/

# Wipe news data (full re-fetch next call)
rm -rf bridge/news/data/

# Wipe tracker history (loses deltas; preserves config)
rm -rf bridge/trackers/data/

# Wipe conversation history + memory
rm -rf data/

# Full nuclear reset
rm -rf bridge/news/data/ bridge/trackers/data/ bridge/alerts/data/ data/

# Then restart the web service so any in-memory state gets rebuilt:
docker compose restart web
```

## Tools Karin can call

The LLM picks from **21 tools** on every turn (20 active by default —
`schedule_reminder` is feature-gated): time, weather, news, alerts,
digest, trackers, math, circuit, graph, convert, wiki, web search,
places, reminders, auto-memory, `say` (repeat-back), `inflation`
(historical purchasing power via BLS CPI-U + multi-region series),
`population` (World Bank time series), `facts` (year-card aggregator),
`analyze` (peak/trough/trend on time series), and `alice` (US
ALICE-share estimator). See [docs/tools.md](docs/tools.md) for the
full list + widget mapping.

## Feature flags

The declarative source of truth is [config/features.yaml](config/features.yaml).
It lists every toggleable subsystem (STT, TTS, bandit, holidays,
news wires) and a tool denylist. Environment variables named in
each entry's `env:` field override the YAML at process start —
this is how CI/compose deploys flip flags without editing the file.

**To toggle a subsystem:** edit `config/features.yaml` on the host
and `docker compose up -d --force-recreate web`. Or set the env var
mentioned in the YAML entry in your `.env` and restart.

**To disable a tool without code changes:** add its name to
`tools.disabled` in `config/features.yaml`. The tool stops being
shown to the LLM and any attempt to call it returns a friendly
"disabled on this server" message.

**To inspect current state:** `curl /api/features` on the web
service returns the resolved snapshot.

## Voice feature flags (STT / TTS)

Default deploy runs **text-only**: STT (faster-whisper) and TTS
(GPT-SoVITS) are both disabled via feature flags. This reclaims
~3 GB of RAM/VRAM for the LLM on Orin Nano 8 GB.

Current defaults in `deploy/docker-compose.yml`:

```yaml
- KARIN_STT_ENABLED=${KARIN_STT_ENABLED:-false}
- KARIN_TTS_ENABLED=${KARIN_TTS_ENABLED:-false}
# New (2026-04-24): remote offload URLs. Setting either makes the
# Jetson POST to a PC sidecar instead of loading the subsystem
# in-process.
- KARIN_STT_BASE_URL=${KARIN_STT_BASE_URL:-}
- KARIN_TTS_BASE_URL=${KARIN_TTS_BASE_URL:-}
```

The client reads `/api/capabilities` + `/api/stt/status` +
`/api/voice/status` on load and hides UI affordances or shows an
"offline" state per the resolved flag + reachability.

### Three voice paths, pick one

#### Path A — text-only (default, no action needed)

Nothing in `deploy/.env`. UI has no mic button, STT/TTS dots stay off.

#### Path B — local STT on the Jetson (opt-in at setup)

```bash
LOCAL_STT=yes bash deploy/setup.sh
```

Writes the following to `deploy/.env`:

```
KARIN_STT_ENABLED=true
KARIN_STT_MODEL=tiny.en
KARIN_STT_DEVICE=cpu
KARIN_STT_COMPUTE_TYPE=int8
```

`tiny.en` is the only Whisper size that leaves headroom next to
`karin-tuned` on Orin Nano 8 GB (measured: ~400 MB add vs ~600 MiB
free at idle). Larger variants are supported — edit `KARIN_STT_MODEL`
in `deploy/.env` — but only on Orin NX 16 GB or larger. TTS stays
off in this mode (can't fit both).

#### Path C — offload both TTS and STT to a PC (recommended for accuracy)

One sidecar on a PC runs both GPT-SoVITS (TTS) and faster-whisper
(STT). See [deploy/pc-tts/README.md](deploy/pc-tts/README.md) for the
Windows sidecar setup (`setup.bat install` once, then `start.bat`).

**On the PC**, confirm the sidecar is up before going to the Jetson:

```powershell
# Find the PC's Tailscale IP — note the value, it's needed below:
tailscale ip -4

# Verify port 9880 is listening AND the right process owns it
# (.venv\tts-server\Scripts\python*.exe, NOT global Python — see
# the Troubleshooting section if it's the wrong one):
Get-NetTCPConnection -LocalPort 9880 -State Listen |
    ForEach-Object { Get-Process -Id $_.OwningProcess } |
    Select-Object Id, ProcessName, Path

# Probe the live endpoints:
curl.exe -s http://<pc-tailscale-ip>:9880/stt/status     # JSON with model info
curl.exe -s -X POST http://<pc-tailscale-ip>:9880/tts    # 422 "Field required" = endpoint live
```

**On the Jetson**, append to `deploy/.env` (the four `KARIN_*` vars,
not the repo-root `.env`) and force-recreate the web container so it
picks them up:

```bash
cat >> ~/Karin/deploy/.env <<'EOF'
KARIN_TTS_ENABLED=true
KARIN_TTS_BASE_URL=http://<pc-tailscale-ip>:9880
KARIN_STT_ENABLED=true
KARIN_STT_BASE_URL=http://<pc-tailscale-ip>:9880
EOF
cd ~/Karin/deploy && docker compose up -d --force-recreate web
```

`cd ~/Karin/deploy` is required — `docker-compose.yml` lives under
`deploy/`, not the repo root. Running `docker compose up` from
`~/Karin` returns `no configuration file provided: not found`.

Both URLs typically point at the same host+port — the sidecar serves
`/tts`, `/transcribe`, `/stt/status` from one FastAPI process. The
Jetson probes `/stt/status` every 2 s and flips the sidebar dots to
green within a few seconds of the container restart.

With STT on the PC you can run `small.en`, `medium.en`, or
`large-v3` for free (no Jetson memory hit). Tailscale round-trip
adds ~50-150 ms per transcribe.

### Sidebar indicators + runtime toggle

Two dots in the sidebar header: `TTS` and `STT`. States:

| State | Meaning | Source |
|---|---|---|
| off | Flag disabled at boot. | `KARIN_*_ENABLED=false` |
| on | Loaded + runtime-enabled. | Feature flag + `_stt_runtime_enabled` + sidecar reachable (if remote) |
| paused | Loaded but runtime-disabled. | STT only — clickable to re-enable |
| starting | Sovits container warming up. | TTS only |
| error | Remote sidecar not responding. | `remote_reachable: false` from probe |

**STT dot is clickable** (toggles runtime via `POST /api/stt/enable`
/ `/api/stt/disable`). The Whisper model stays loaded; the flag just
gates whether `stt.transcribe` is called. Toggling is cheap
(milliseconds) and safe mid-session.

**TTS dot is read-only** — starting/stopping sovits is done via SSH
(`docker compose start|stop sovits`) because doing it from the UI
while the LLM is warm can OOM on Orin Nano.

### Troubleshooting "remote sidecar offline"

If the STT or TTS dot shows `error` (or stays `off` with a "PC offline"
tooltip), walk these checks in order. Stop at the first failure.

1. **Is the sidecar actually running on the PC?** From the PC:
   ```powershell
   Get-NetTCPConnection -LocalPort 9880 -State Listen `
       -ErrorAction SilentlyContinue |
       Select-Object LocalAddress, OwningProcess
   curl.exe -s http://localhost:9880/stt/status            # 200 + JSON
   ```
   If nothing listens on 9880, double-click [deploy/pc-tts/start.bat](deploy/pc-tts/start.bat)
   or run `start.bat --visible` to surface startup errors. Don't
   `python tts_server.py` directly — that uses global Python instead
   of the sidecar's venv and produces a stale process (see #5 below).

2. **Is the listening process the venv-launched one?**
   ```powershell
   Get-NetTCPConnection -LocalPort 9880 -State Listen |
       ForEach-Object { Get-Process -Id $_.OwningProcess } |
       Select-Object Id, ProcessName, Path
   ```
   `Path` should be `…\.venv\tts-server\Scripts\python(w).exe`. If it's
   `…\AppData\Local\Programs\Python\Python312\python.exe` (or any
   global install), kill it (`Stop-Process -Id <pid>`) and relaunch via
   `start.bat`. The global-Python instance won't have all the routes.

3. **Can the Jetson reach the PC over tailnet?** From the Jetson:
   ```bash
   curl -sf -m 3 http://<pc-tailscale-ip>:9880/stt/status
   ```
   If this times out while step 1 worked locally on the PC, the most
   common cause is **Windows Defender Firewall** blocking inbound 9880.
   When the sidecar first launches, Defender prompts to allow `python`
   on Private/Public — say yes to **Private** at minimum (tailnet
   counts as private if you've classified it that way; otherwise
   allow both, or `New-NetFirewallRule -DisplayName "Karin sidecar" `
   `-Direction Inbound -Protocol TCP -LocalPort 9880 -Action Allow`).

4. **Does the Jetson's `deploy/.env` have the URLs?**
   ```bash
   grep -E '^KARIN_(TTS|STT)_' ~/Karin/deploy/.env
   docker compose -f ~/Karin/deploy/docker-compose.yml \
                  -f ~/Karin/deploy/docker-compose.jetson.yml \
       exec web env | grep -E 'KARIN_(TTS|STT)_'
   ```
   If the env file has the vars but `docker compose exec` doesn't show
   them, the container hasn't been recreated — `cd ~/Karin/deploy &&
   docker compose up -d --force-recreate web`.

5. **Stale sidecar process serving old routes** (e.g. `/stt/status`
   returns 404 even though the file has it). Python doesn't
   hot-reload, so a process started before a code update is frozen
   at the old version. Fix: `Stop-Process -Id <pid>` then `start.bat`.

6. The Jetson retries transient connect errors once before failing a
   transcribe — brief Tailscale blips don't kill a turn.

**On the PC sidecar log:** `GET /stt/status HTTP/1.1 200 OK` every 2 s
is the Jetson's reachability poll — that's healthy traffic. If you
see `GET /docs HTTP/1.1 404 Not Found` instead, you're running a web
container built before the probe was switched off `/docs` (rebuild
with `docker compose up -d --build --force-recreate web`).

### Disabling just one

- `KARIN_STT_ENABLED=false`, `KARIN_TTS_ENABLED=true` → type in, hear
  spoken replies. Mic button hidden.
- `KARIN_STT_ENABLED=true`, `KARIN_TTS_ENABLED=false` → speak, read
  text replies. Voice-status dot + replay buttons hidden.
- Can be set in `deploy/.env` (permanent) or via the `/api/stt/{enable,disable}`
  endpoint (runtime, for STT only).

## Operating voice on-demand

(Applies only when `KARIN_TTS_ENABLED=true`.)

On Orin Nano 8 GB the LLM and sovits cannot both stay GPU-resident
reliably. Default workflow:

1. Sovits is **stopped** at boot — text turns are fast.
2. When you want spoken replies, click the **Voice** pill in the
   sidebar header. The dot goes amber (~30 s warm-up on first start
   of the day), then green when sovits's healthcheck passes.
3. While voice is on: every reply gets synthesized; expect 30-60 s
   per turn (LLM gets partially CPU-spilled to make room for sovits).
4. Click the pill again to stop sovits and reclaim memory.

The toggle is just a UI shortcut for `docker compose start sovits` /
`docker compose stop sovits` against the existing container. The
container itself must have been created once with
`docker compose --profile voice up -d sovits` — that happens
automatically the first time you click Voice if it doesn't exist
yet, **as long as** the container has been built (image present).

If you'd rather drive it by SSH:
```bash
cd ~/Karin
docker compose --profile voice up -d sovits   # start (creates container if missing)
docker compose stop sovits                     # stop and free memory
```

## Troubleshooting

### `docker compose: no configuration file provided: not found`

You're running it from `~/Karin` instead of `~/Karin/deploy/`. The
compose files live under `deploy/`. All hand-rolled invocations need
`cd ~/Karin/deploy &&` first. (`bash deploy/setup.sh` handles this
internally and can be run from any working directory.)

### Character dropdown shows `(text only)` on every entry

You're on a web image built before [web/static/app.js](web/static/app.js)
moved to the leading-circle indicator. Pull the latest source and
rebuild the web container: `cd ~/Karin/deploy && docker compose up -d
--build --force-recreate web`. Hard-reload the browser (Ctrl-Shift-R)
to bypass cached JS. After update, voice-less characters lead with
`○` instead of suffixing `(text only)`.

### Character switch returns 500 on the shipped `default` character

Old web image. Same fix: rebuild + hard-reload. The handler now
guards against `voice = None` for voice-less characters.

### Character `general` (or any non-shipped name) returns "not found"

By design after the scanner update — `general`'s `voice.yaml` is
gitignored, so a fresh tar/clone has the directory but no persona
metadata, and the dropdown filters it out. To restore: drop the
character's `voice.yaml` (and ideally `voices/*.{ckpt,pth,wav}`) into
`~/Karin/characters/<name>/` on the Jetson manually — they're
deliberately not part of the public repo.

### Voice toggle says "err" / can't start sovits

The toggle uses the docker socket (`/var/run/docker.sock`) mounted
into the web container. Check:

```bash
# is the socket actually mounted?
docker exec karin-web ls -la /var/run/docker.sock

# is the sovits container known to docker?
docker ps -a --filter name=karin-sovits
```

If the container doesn't exist yet (fresh deploy), bring it up once
manually so docker creates it; the toggle takes over after that:
```bash
cd ~/Karin && docker compose --profile voice up -d sovits
```

### "sovits" never becomes healthy

```bash
docker compose logs sovits
```

Common causes:
- Bind-mount missing. Check that `third_party/GPT-SoVITS/api_v2.py`
  exists on the host.
- Pretrained models missing. The `/gpt-sovits/GPT_SoVITS/pretrained_models`
  dir should have ~4 GB of files (chinese-hubert-base, etc.). Re-run
  the `snapshot_download` step from first-time setup.
- `register_pytree_node() got an unexpected keyword argument 'flatten_with_keys_fn'`
  — torch 2.2 / transformers ≥4.41 mismatch on JetPack 6. Already
  pinned in [Dockerfile.sovits](deploy/Dockerfile.sovits) (`transformers==4.40.2`,
  `peft==0.11.1`, `torchmetrics==1.4.0`); rebuild with
  `docker compose build sovits --no-cache` if you've edited that file.
- `Could not build wheels for opencc` — also pinned: we substitute
  `opencc-python-reimplemented` (pure Python, no C++ build) for
  Chinese text normalization. Lift if you ever need true OpenCC
  fidelity.

### LLM returns HTTP 500 mid-conversation

Almost always memory pressure when sovits is also up. The bridge
auto-retries 5xx once with a 3 s delay, which usually masks the flap;
if you still see it, options:
- Stop sovits via the Voice toggle to free GPU.
- Check `journalctl -u ollama -n 30` for `NvMapMemHandleAlloc error 12`
  (Tegra allocator failure) or `llama runner terminated`.
- Increase `OLLAMA_GPU_OVERHEAD` in the systemd override at
  `/etc/systemd/system/ollama.service.d/quantize-kv.conf` if sovits
  is regularly evicting the LLM (tradeoff: more CPU spill).

### Karin replies in Japanese / another language

Persona fields live in `characters/<name>/voice.yaml` (gitignored
per-deploy); the LANGUAGE rule itself sits in the shared template
[characters/profile.yaml](characters/profile.yaml) so persona-specific
"JK-energy" cues can't override it. Two fix paths:

- **All characters**: edit the LANGUAGE block in
  `characters/profile.yaml` (tracked, ships with the repo).
- **One specific character**: edit its `characters/<name>/voice.yaml`
  — the `language_note:` field fills the `{language_note}` placeholder
  in profile.yaml when that character is active.

Legacy note: `config/characters/<name>.yaml` was the old per-character
prompt location. It's a fallback path only — the new
`characters/<name>/voice.yaml` layout wins when present. On a fresh
clone the legacy files are gitignored, so always edit the new-layout
files.

### Weather widget says "couldn't find <city>"

The geocoder (open-meteo) doesn't tolerate over-specific strings.
The bridge already tries progressively-simpler variants ("City,
Region, Country" → "City, Region" → "City") and uses US state
abbreviations to disambiguate among same-name results. If you still
see "couldn't find" for a real city, check
[bridge/tools/_weather.py](bridge/tools/_weather.py) `_geocode_open_meteo` — the
candidate list may need a new alias or the result-scoring needs
tightening. Tail logs to see which variants were tried:
```bash
docker compose logs web | grep geocod
```

### Web UI loads but turns return HTTP 502

- LLM unreachable: `docker compose logs ollama` — model pull may still
  be in progress on first boot.
- TTS unreachable: `docker compose logs sovits`. Each turn's reply
  still lands as text — check the browser for a muted "voice offline"
  note.

### Ollama keeps re-pulling the model on restart

The `./ollama/` directory is bind-mounted from the compose dir. If it's
getting wiped, check file permissions:

```bash
ls -la deploy/ollama/
```

Should be owned by your user / writable.

### Browser mic recording won't stop / "endless listening"

The PTT button is **toggle mode** with browser-side VAD: tap once to
start, tap again to stop, OR auto-stops after 1.5 s of silence
(once at least one second of speech has been detected). If you
press but the mic seems stuck:
- Check the browser console for "PTT recording auto-stopped after
  max duration" — there's a 30 s hard cap as the last safety net.
- VAD thresholds in [web/static/app.js](web/static/app.js) (`VAD_RMS_SPEECH`,
  `VAD_RMS_SILENCE`, `VAD_SILENCE_MS`) — raise/lower if VAD is
  cutting you off mid-thought or never triggering.

### `ImportError: libcusparseLt.so.0: cannot open shared object file`

Torch 2.5 on Jetson needs cuSPARSELt, which JetPack 6.2 doesn't ship.
You installed torch but skipped the cuSPARSELt step. Run Step 2 from
[What you need on the host](#what-you-need-on-the-host-before-docker-compose-up)
§1b (the `cusparselt-local-tegra-repo-ubuntu2204-0.7.1` .deb). Should
clear the import without reinstalling torch.

### `pypi.jetson-ai-lab.dev` DNS failure during torch install

The community wheel index moved from `.dev` to `.io`. Use
`https://pypi.jetson-ai-lab.io/jp6/cu126` instead, or the NVIDIA CDN
direct wheel URL in Step 1b.

### GPU not detected inside a container

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Should print the GPU table. If it errors, NVIDIA Container Toolkit
isn't installed / configured.

## What to commit and what to keep local

| Path | Status | Why |
|---|---|---|
| `*.example.json` | tracked | shipping defaults |
| `bridge/news/config/preferences.json` | gitignored | personal curation |
| `bridge/trackers/config/trackers.json` | gitignored | personal curation |
| `bridge/*/data/`, `data/` | gitignored | runtime artifacts |
| `config/assistant.yaml` | tracked | character / voice / prompt config |
| `characters/*/voices/*.{ckpt,pth,wav}` | gitignored | your trained voice bundles |
| `third_party/GPT-SoVITS/` | gitignored | upstream clone |
| `deploy/ollama/` | gitignored | local model cache |
| `RUNBOOK.md`, `README.md`, `CHANGES.md` | tracked | docs |

## Falling back to the host `setup.sh` path

If the Docker setup fights your specific hardware (JetPack edge cases
are real), the host-install path still works:

```bash
sudo -v
bash deploy/setup.sh
```

This creates venvs on the host and installs systemd units
(`sovits-server.service`, `assistant.service`, `web.service`). See
[docs/deployment.md](docs/deployment.md).
