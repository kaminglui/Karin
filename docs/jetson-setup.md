# Jetson setup — flash JetPack and provision for Karin

Walks you from a brand-new (or wiped) Jetson Orin Nano Developer Kit
all the way to a running Karin stack. Two parts:

1. **[Part 1 — Flash JetPack 6](#part-1--flash-jetpack-6)** — SDK Manager
   + recovery-mode pin procedure. Skip this if JetPack 6 is already on
   the device.
2. **[Part 2 — Provision Karin on the Jetson](#part-2--provision-karin-on-the-jetson)** — Docker, optional desktop strip,
   Ollama, repo + voice weights transfer, first bring-up, HTTPS.

After finishing both, head to [RUNBOOK.md](../RUNBOOK.md) for day-to-day
operations.

Total time, fresh Jetson → working voice assistant: **~75-135 min**
(~45-90 flashing, ~30-45 provisioning — most of it spent waiting on
downloads).

## Contents

- [Part 1 — Flash JetPack 6](#part-1--flash-jetpack-6)
  - [Hardware checklist](#hardware-checklist)
  - [Install SDK Manager on the host](#install-sdk-manager-on-the-host)
  - [Put the Jetson into recovery mode](#put-the-jetson-into-recovery-mode)
  - [Run the flash from SDK Manager](#run-the-flash-from-sdk-manager)
  - [First boot and networking](#first-boot-and-networking)
  - [Install Tailscale for remote SSH](#install-tailscale-for-remote-ssh)
- [Part 2 — Provision Karin on the Jetson](#part-2--provision-karin-on-the-jetson)
  - [Prerequisites from JetPack](#prerequisites-from-jetpack)
  - [Install Docker Engine](#install-docker-engine)
  - [Install PyTorch + cuSPARSELt](#install-pytorch--cusparselt)
  - [Strip the desktop environment (optional)](#strip-the-desktop-environment-optional)
  - [Install and tune Ollama](#install-and-tune-ollama)
  - [Transfer the repo](#transfer-the-repo)
  - [Add a voice (optional)](#add-a-voice-optional)
  - [Bootstrap GPT-SoVITS (local sovits only)](#bootstrap-gpt-sovits-local-sovits-only)
  - [Configure .env for Jetson constraints](#configure-env-for-jetson-constraints)
  - [First-time bring-up](#first-time-bring-up)
- [Troubleshooting](#troubleshooting)

---

# Part 1 — Flash JetPack 6

## Hardware checklist

- **Jetson Orin Nano Developer Kit** (8 GB)
- **Storage** — either:
  - A **microSD card** (32 GB+; 64 GB recommended)
  - OR an **NVMe SSD** mounted on the carrier board (preferred — much faster I/O)
- **USB-C cable** for flashing (host PC ↔ Jetson)
- **Jumper wire** OR a small piece of metal (paperclip works) to short two pins
- **Host computer** running an OS supported by NVIDIA SDK Manager
  (Ubuntu, Windows, CentOS/RHEL, or via Docker — current list on the
  [SDK Manager download page](https://developer.nvidia.com/sdk-manager)).
  Needs **~40 GB free disk** for SDK Manager + the JetPack image.
  macOS is not supported.
- **Internet on the host** (~5-15 GB download for the JetPack components)
- **Optional but recommended**: HDMI monitor + USB keyboard for the
  Jetson's first boot (the Ubuntu setup wizard runs there)

## Install SDK Manager on the host

> <https://developer.nvidia.com/sdk-manager>

Sign in with a (free) NVIDIA Developer account, download the
installer for your host OS, and follow NVIDIA's install instructions
on that page (per-OS commands for Ubuntu / Windows / CentOS / Docker
are listed there and stay in sync with the latest SDK Manager
release).

Once installed, launch the GUI (Ubuntu: `sdkmanager`; Windows: Start
menu → "NVIDIA SDK Manager") and sign in again with the same
NVIDIA Developer account.

## Put the Jetson into recovery mode

The Jetson must be in **Force Recovery Mode** before SDK Manager can
talk to it over USB. This is a hardware procedure — done by shorting
two pins on the carrier board's button header while powering on.

### Find the pins

On the Orin Nano Dev Kit, locate the **button header J14** — a
12-pin single-row block on the carrier board. Pins are labeled in
white silkscreen by *function* (e.g. `FORCE_RECOVERY*`) but **pin
numbers are NOT printed**, so identify pins by their label.

Per the [Carrier Board Specification (PDF)](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf)
(also linked from the [Jetson Download Center](https://developer.nvidia.com/embedded/downloads)),
the relevant pins are:

| Pin # | Label | Notes |
|---|---|---|
| 1 | PC_LED- | one end of the header |
| 2 | PC_LED+ | |
| ... | UART debug, AC OK, SYS_RESET — not needed for recovery | |
| **9** | **GND** | use this GND for the FORCE_RECOVERY short |
| **10** | **FORCE_RECOVERY*** | pull LOW to enter recovery mode |
| 11 | GND | DO NOT use — paired with pin 12 (power-button function) |
| 12 | SLEEP/WAKE* | power button — other end of the header |

> ⚠️ **Always pair FORCE_RECOVERY* (pin 10) with pin 9, never pin 11.**
> Pin 11's GND is reserved for the manual power-button circuit on
> pin 12. With the default Auto-Power-On behavior (pins 5-6 left
> disconnected, factory state) you never need to touch 11 or 12.

### The procedure

Per the spec: *"Connect pins 9 and 10 during power-on to put system
in USB Force Recovery mode."*

1. **Make sure the Jetson is OFF** (no power, or use the power
   button to fully shut down — not suspend).
2. **Plug the USB-C flashing cable** from the host PC into the
   Jetson's front USB-C **data** port (not the power input).
3. **Short pin 9 to pin 10** with a jumper / paperclip / tweezers
   and hold it there.
4. **Power on the Jetson** (plug in the barrel-jack supply or
   press the power button) while still holding the short.
5. **Wait ~3 seconds**, then **remove the jumper**.

The Jetson now appears as an "NVIDIA Corp. APX" device on the host
PC (you can verify with `lsusb | grep -i nvidia`).

If you have a monitor connected, the screen stays blank — recovery
mode does not start the desktop.

## Run the flash from SDK Manager

In SDK Manager:

1. **STEP 01 — DEVELOPMENT ENVIRONMENT**:
   - Product Category: **Jetson**
   - Hardware Configuration: **Target Hardware** → SDK Manager auto-
     detects the recovery-mode device and lists "Jetson Orin Nano".
     If it's stuck on "Refreshing", check that `lsusb` shows the
     APX device. If not, redo the recovery-mode procedure.
   - Target Operating System: **Linux JetPack 6.x** (latest)
   - Click **Continue**.

2. **STEP 02 — DETAILS AND LICENSE**:
   - Tick the components you want — at minimum:
     - **Jetson Linux** (the OS / BSP)
     - **CUDA** (required for our LLM + STT)
     - **cuDNN** (required for STT)
     - **TensorRT** (optional but small; useful for some accel paths)
   - Skip "Jetson SDK Components" if you want a lean install (we
     install our own deps via Docker).
   - Accept the license. Click **Continue**.

3. **STEP 03 — SETUP PROCESS**:
   - Storage device: pick your **microSD** or **NVMe** target.
   - OEM Configuration: choose **Runtime** so the Jetson runs the
     Ubuntu setup wizard on first boot (you create the user account
     interactively). The alternative — "Pre-config" — is faster but
     bakes in default credentials, less secure.
   - Click **Flash** → enter your sudo password when prompted.

The flash takes **~30-60 minutes** depending on host disk speed and
which components you selected. SDK Manager downloads everything to
`~/nvidia/sdkm_downloads/` (cached for next time) then writes the
image to the Jetson over USB.

When it finishes, SDK Manager pops a "Flash succeeded" dialog. The
Jetson reboots into normal mode automatically.

## First boot and networking

If you connected a monitor + keyboard:

1. The **Ubuntu setup wizard** runs on the Jetson's screen — pick
   language, timezone, **create a user account** (this is what
   you'll SSH in as later), set a password.
2. After the wizard, you land at the GNOME desktop or login screen.
3. **Connect to your Wi-Fi** via the system tray's network applet.

If you flashed without a monitor (headless): SDK Manager's "STEP 04"
finishes the OS configuration over USB serial / SSH. Follow the
prompts on the host side.

Either way, after first boot:

```bash
# Verify networking + JetPack
ip addr            # confirm Wi-Fi/Ethernet has an IP
cat /etc/nv_tegra_release    # should print 'R36, REVISION: ...'
nvidia-smi         # if you see the GPU table, CUDA stack is up
```

## Install Tailscale for remote SSH

So you can SSH into the Jetson from anywhere without exposing it
to the open internet.

### 1. Install Tailscale on the Jetson

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# Click the URL it prints to authenticate against your tailnet.
tailscale ip -4    # note the 100.x.y.z IP — use this for SSH from your dev machine
```

### 2. Confirm the SSH server is running

Ubuntu/JetPack ships `openssh-server` pre-installed and enabled, but
verify:

```bash
sudo systemctl status ssh       # look for "active (running)" — Ctrl+C to exit
# If not enabled for some reason:
sudo systemctl enable --now ssh
```

### 3. Authorize your dev-machine SSH key

Pick the block that matches your dev machine.

**Windows PC (PowerShell):** Windows ships OpenSSH but has no
`ssh-copy-id`, so we append the public key manually. On your PC:

```powershell
# If you don't already have a key pair:
ssh-keygen -t ed25519 -C "windows-pc"
# Press Enter to accept the default path; passphrase optional.

# Append your public key to the Jetson's authorized_keys
# (prompts for your Jetson password once — key auth replaces it afterwards):
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <jetson-user>@<tailscale-ip> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

**Linux / macOS dev machine:** use the built-in helper:

```bash
ssh-keygen -t ed25519 -C "dev-machine"   # skip if you already have a key
ssh-copy-id <jetson-user>@<tailscale-ip>
```

### 4. Verify passwordless SSH works

```
ssh <jetson-user>@<tailscale-ip>
```

Should drop straight into a Jetson shell — no password prompt. If
the key has a passphrase, your OS unlocks it instead of asking for
the Jetson password. If it still asks for the Jetson password,
permissions on the Jetson got wrong somewhere; fix them by SSHing
in with the password once and running:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### 5. (Optional but recommended) Disable password SSH on the Jetson

Once key-based login works, lock out password auth entirely — the
Jetson is then only reachable via your authorized key:

```bash
# On the Jetson:
sudo sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

> ⚠️ **Keep your existing SSH session open and test a fresh connection
> from another terminal BEFORE closing it.** If password auth breaks
> and key auth isn't working, your active session is the only way
> back in to fix it.

### 6. (Later) Enable HTTPS fronting for the browser UI

Karin is served **HTTPS-only** over the tailnet. The web container
binds `uvicorn` to `127.0.0.1:80` (not `0.0.0.0`), so plain HTTP is
**not reachable** on the tailnet IP. Tailscale Serve terminates TLS
on :443 with an auto-provisioned Let's Encrypt cert bound to the
device's current MagicDNS name, and proxies to the local container.
This also unblocks `getUserMedia` in the browser (mic button needs
HTTPS).

> ⏭ **Don't run this yet.** The web container has to be up first —
> come back here after you finish
> [First-time bring-up](#first-time-bring-up) in Part 2.

**If you run `bash deploy/setup.sh`**, Tailscale Serve is provisioned
automatically at the tail of the script. You don't need to do anything
manually.

**If you want to provision manually** (e.g. running via docker compose
without `setup.sh`, or refreshing after a device rename):

```bash
# Clear any prior config (e.g. bound to an old device name):
sudo tailscale serve reset

# Bind :443 → 127.0.0.1:80 under the current Tailscale DNSName.
# Cert is auto-managed by tailscaled.
sudo tailscale serve --bg http://127.0.0.1:80
```

Then access at `https://<current-magicdns-name>.<your-tailnet>.ts.net`.
The cert is provisioned on the first TLS handshake (~5 s).

**After a device rename** (admin console or
`tailscale set --hostname=...`), re-run the two commands above — the
serve config sticks to the old name otherwise. See
[RUNBOOK § Tailscale HTTPS](../RUNBOOK.md#tailscale-https-port-443-on-port-80-off)
for the operator-facing details and verification commands.

**If you ever need plain HTTP instead** (LAN deploy without Tailscale,
different reverse proxy in front): edit
[`deploy/docker-compose.yml`](../deploy/docker-compose.yml) and change
the `web` service `command:` from `--host 127.0.0.1` to
`--host 0.0.0.0`, then `docker compose up -d web`.

---

# Part 2 — Provision Karin on the Jetson

Run-once steps for going from a JetPack 6 image → working Karin.
Replace `<jetson-user>` with whatever your Jetson login is.

## Prerequisites from JetPack

- JetPack 6.x (L4T 36.x); verify with `cat /etc/nv_tegra_release` →
  should print `# R36 (release), REVISION: ...`.
- ARM64 / aarch64 (`uname -m` → `aarch64`).
- NVIDIA Container Toolkit (JetPack ships with it).

## Install Docker Engine

JetPack does NOT ship Docker by default; only the NVIDIA Container
Toolkit. Install Docker via the upstream installer:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER       # so docker commands don't need sudo
# Log out + back in for the group change to take effect.
```

Verify GPU passthrough works (Jetson's NCT uses the LEGACY runtime,
NOT `--gpus`; use `--runtime=nvidia` for one-off tests):

```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# Should print the Orin GPU table.
```

## Install PyTorch + cuSPARSELt

`deploy/setup.sh` fails fast if system `python3` can't `import torch`
with CUDA, because the bridge venv is created with
`--system-site-packages` and inherits torch from the host. Install the
NVIDIA-built wheel to `~/.local/` **once** — it survives future
`rm -rf ~/Karin` cleanups and is shared across every venv on the box.

**Verified on JetPack 6.2 (CUDA 12.6, Python 3.10).** JetPack 6.1 works
with the same wheel. For 6.0 (CUDA 12.2) swap the URL — see the RUNBOOK
§ 1b notes.

```bash
# 1. Torch 2.5 wheel from NVIDIA's CDN (~807 MB):
pip3 install --user --no-cache \
    'https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl'

# 2. cuSPARSELt (torch 2.5 needs libcusparseLt.so.0, JetPack doesn't ship it):
cd /tmp
wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y libcusparselt0 libcusparselt-dev

# 3. Verify — should print "2.5.0a0+872d972e41.nv24.08 True":
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A numpy ABI `UserWarning` on first import is harmless — system numpy
is older than torch 2.5 expects, and the bridge venv installs a
matching numpy later.

If `pip3 install` fails on the jetson-ai-lab index, note that the
old community domain `pypi.jetson-ai-lab.dev` no longer resolves —
use `.io` (`https://pypi.jetson-ai-lab.io/jp6/cu126`) or the NVIDIA
CDN direct URL above.

## Strip the desktop environment (optional)

If the Jetson is headless (no monitor / SSH-only), remove GNOME and
the display manager. This frees **1-2 GB of RAM and ~3 GB of disk**,
and on a memory-constrained device that's the difference between an
LLM model fitting on GPU vs spilling to CPU.

### Soft mode (recommended — keeps packages, just stops auto-start)

```bash
sudo systemctl set-default multi-user.target   # boot into CLI by default
sudo systemctl isolate multi-user.target       # apply NOW (closes desktop)
```

After this, GUI is gone but GNOME packages remain installed. If you
ever want it back: `sudo systemctl set-default graphical.target` +
reboot.

### Hard mode (purge the desktop entirely)

> ⚠️ **Two critical rules. Don't skip either one.**
>
> **Rule 1: Pin everything you depend on BEFORE purging.** `ubuntu-desktop`
> has many auto-installed dependencies; `apt autoremove` after the
> purge will cascade-remove anything that no longer has a rdep, and on
> JetPack 6 that famously includes **NetworkManager** — which locks
> you out of SSH. Recovery then requires physical access + another
> internet source to re-install it. Preflight is cheap insurance.
>
> **Rule 2: DO NOT run `apt autoremove --purge` after the purge.** The
> disk savings aren't worth the cascade risk. Leave orphaned deps on
> disk; they don't consume runtime resources. Soft mode is cleaner
> than hard mode done wrong.
>
> **If you skip either rule, you can lose SSH access and need physical
> recovery (monitor + keyboard or UART console).**

**1. Pin critical packages:**

```bash
sudo apt-mark hold \
    network-manager network-manager-config-connectivity-ubuntu \
    systemd-resolved systemd-networkd \
    wpa_supplicant isc-dhcp-client \
    openssh-server openssh-client \
    tailscale \
    docker-ce docker-ce-cli containerd.io \
    nvidia-container-toolkit nvidia-container-runtime \
    ollama \
    apparmor \
    nvidia-l4t-core nvidia-l4t-init nvidia-l4t-firmware nvidia-l4t-kernel \
    nvidia-l4t-bootloader nvidia-l4t-cuda nvidia-l4t-cudnn \
    nvidia-jetpack
```

Some of those packages may not exist on your specific image (e.g.
`ollama` only if you installed it) — `apt-mark hold` silently skips
unknowns, so it's safe to paste the whole block.

**2. Stop and disable the display manager:**

```bash
sudo systemctl stop gdm3 || true
sudo systemctl disable gdm3 || true
```

**3. Purge desktop packages ONLY (no autoremove):**

```bash
sudo apt purge -y \
    ubuntu-desktop ubuntu-desktop-minimal \
    gdm3 gnome-shell gnome-session gnome-control-center \
    nautilus mutter update-manager-gnome
# Notes:
# - do NOT add `network-manager-gnome` — it's a GUI applet but its
#   transitive deps overlap with network-manager itself.
# - do NOT add `xserver-xorg`, `xorg`, `xwayland` — several Jetson
#   SDK tools (CUDA samples, video acceleration paths) link against
#   X libraries even when no X server is running.
# - do NOT run `apt autoremove` afterwards — leave orphans on disk.

sudo apt clean   # safe; only clears the apt download cache
```

**4. Verify BEFORE you log out or reboot:**

```bash
# Networking still alive?
ip addr show | grep -E 'inet |state UP'
systemctl is-active NetworkManager           # should print 'active'
ping -c2 1.1.1.1                             # connectivity
ping -c2 google.com                          # DNS resolution

# SSH still alive? (from another machine):
#   ssh <user>@<jetson-ip> "echo ok"

# Tailscale still alive?
tailscale status | head -3

# Docker still alive?
docker ps
```

If any of those fail, **do not log out** until you fix it. You have a
live SSH session; you can still recover. Reinstall whatever got
hit: `sudo apt install -y <package>`.

### Recovering from a GUI-purge that broke networking

If you skipped the `apt-mark hold` step and lost SSH after the purge,
the Jetson is alive but has no IP. Recovery requires physical access
(monitor + USB keyboard, or UART serial at `/dev/ttyUSB0` 115200 8N1):

1. Log in at the TTY (or connect via USB-TTL serial cable from
   another machine).
2. Reinstall + start NetworkManager:
   ```bash
   sudo apt install -y network-manager
   sudo systemctl enable --now NetworkManager
   ```
3. If Wi-Fi:
   ```bash
   sudo nmcli device wifi connect "<your-ssid>" password "<your-password>"
   ```
4. If apt itself has no internet (offline recovery):
   - Plug Ethernet into any router with internet (`sudo dhclient enP8p1s0`)
   - Or use iPhone Wi-Fi hotspot (standard WPA2-PSK):
     ```bash
     sudo wpa_passphrase "<iPhone-name>" "<hotspot-pwd>" | sudo tee /etc/wpa_tmp.conf
     sudo ip link set wlP1p1s0 up
     sudo wpa_supplicant -B -i wlP1p1s0 -c /etc/wpa_tmp.conf
     sudo dhclient wlP1p1s0
     ```
   - iPhone USB tether needs `usbmuxd` installed (often cascade-removed
     with GUI); skip this path if GUI was purged.
5. Tailscale should auto-reconnect; if not:
   `sudo systemctl restart tailscaled`.
6. Verify with `tailscale status`, then SSH back in from your dev box.

**Strongly prefer soft mode over hard mode** unless you really need
the ~3 GB of disk space back. Soft mode can never lock you out.

### Disable services you don't need on a headless Jetson

Frees ~1 GB of RAM. Required if you want to keep both LLM and sovits
live concurrently, optional otherwise (the [Voice toggle](../RUNBOOK.md#operating-voice-on-demand)
already handles the LLM/sovits coexistence problem on tight memory).

```bash
sudo systemctl disable --now snapd.service snapd.socket snapd.seeded.service
sudo systemctl disable --now bluetooth.service ModemManager.service
sudo systemctl disable --now fwupd.service fwupd-refresh.timer
sudo systemctl disable --now cups.service cups.socket cups-browsed.service
sudo systemctl disable --now snap.cups.cupsd.service snap.cups.cups-browsed.service 2>/dev/null

# Confirm what got freed
free -m | head -2
```

These daemons are useful on a desktop Ubuntu install (printing,
device firmware updates, Bluetooth audio) but a headless voice
assistant doesn't need any of them. Re-enable individually if you
discover you do.

## Install and tune Ollama

We run Ollama natively on the host (not in a container) because the
JetPack-supplied CUDA torch + ollama's GPU detection need direct
access to the L4T NVIDIA driver. Containerizing it adds friction
without benefit on this hardware.

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

**Tune Ollama for VRAM-tight Jetson**: enable flash attention + KV
cache quantization (q8_0). On Orin Nano 8 GB this can be the difference
between fitting the LLM fully on GPU vs OOM-killing it.

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/quantize-kv.conf <<'EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=true"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
# OLLAMA_GPU_OVERHEAD reserves N bytes of shared-VRAM for other
# components (sovits, Whisper). Pick ONE:
#   Text-only default (recommended on Orin Nano): leave UNSET or =0.
#       Any reservation reduces what karin-tuned can place on GPU and
#       increases CPU spill + swap pressure (measured 2026-04-24:
#       9%/91% CPU/GPU split at idle with no reservation).
#   LOCAL_SOVITS=yes (Orin NX 16 GB+ territory): 536870912 (512 MB)
#       prevents the Tegra allocator crash `NvMapMemHandleAlloc
#       error 12` when sovits starts while the LLM is warm.
#Environment="OLLAMA_GPU_OVERHEAD=536870912"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

The `OLLAMA_GPU_OVERHEAD` line above is **commented out by default**
because on Orin Nano 8 GB the text-only production path has no second
model to reserve for. Uncomment it only when flipping on LOCAL_SOVITS.

Pull the model — pick based on hardware:

```bash
# Orin Nano 8 GB (recommended — fits 100% on GPU when sovits is off):
ollama pull qwen2.5:3b

# Orin NX 16 GB or x86 with ≥12 GB free VRAM (can run sovits concurrently):
ollama pull huihui_ai/qwen3.5-abliterated:4b

# Embedding model used by the preference-feedback bandit (~600 MB).
# Optional but recommended; without it the thumbs-up/down learning
# loop stays inert.
ollama pull nomic-embed-text
```

Later in the `.env` step, set `KARIN_LLM_MODEL` to whichever tag you
actually pulled here. The checked-in `assistant.yaml` default is
`llama3.1:8b`, so the env override is what makes Jetson deploys use the
smaller Qwen variants.

Verify:

```bash
ollama ps
# Right after a request, should show "100% GPU" in the PROCESSOR column.
# If you see e.g. "30%/70% CPU/GPU", the model is too big for available
# VRAM and is spilling to CPU — switch to a smaller model.
#
# Two things in `ollama ps` output that look like bugs but aren't:
#   * A small CPU spill (e.g. `9%/91% CPU/GPU`) is normal on Orin Nano
#     with karin-tuned (4.9 GB Q4_K_M) — the whole model doesn't fit
#     100% on GPU so a sliver runs on CPU. Inference still works; it's
#     just slightly slower than 100% GPU would be.
#   * The CONTEXT column (e.g. `512`) shows the *currently allocated*
#     KV cache, not the configured `num_ctx`. Ollama allocates lazily
#     and grows up to num_ctx as turns consume tokens. At idle it sits
#     at the floor (512). Run an actual chat turn, then re-check and
#     you'll see it tick up toward 2048.
```

## Transfer the repo

Either clone from GitHub or rsync from a dev machine. Fresh clone is the
simplest:

```bash
# On the Jetson:
git clone https://github.com/kaminglui/Karin.git
cd Karin
```

If you already have a working copy on a dev machine, you can push it over
Tailscale SSH instead (replace `<jetson-user>` and `<jetson-ip>`):

```bash
# On dev machine — tar repo + push (skips weights, venvs, third_party):
tar --exclude='Karin/.venv' \
    --exclude='Karin/third_party' \
    --exclude='Karin/data' \
    --exclude='Karin/characters/*/voices/*.ckpt' \
    --exclude='Karin/characters/*/voices/*.pth' \
    -czf Karin.tar.gz Karin
scp Karin.tar.gz <jetson-user>@<jetson-ip>:~/

# On Jetson:
tar xzf ~/Karin.tar.gz -C ~/
cd ~/Karin
```

## Add a voice (optional)

Karin ships text-only by default. Voice requires a trained GPT-SoVITS
bundle per character — not shipped with the repo because the weights are
~300 MB per character and voice training is personal (the `ref.wav` is
your own voice sample). Skip this section entirely if text-only is fine
for now — you can add voice later.

### Where will GPT-SoVITS run?

Two choices. Pick based on hardware:

| Your Jetson | Recommended path | Why |
|---|---|---|
| **Orin Nano 8 GB** | PC-TTS offload | Only 7.4 GiB of unified memory is usable. karin-tuned alone holds ~6.8 GiB at idle (measured), so sovits + LLM co-load OOMs or forces a ~3× CPU-spill slowdown. |
| **Orin NX 16 GB or larger** | Either works | You have headroom to co-load. Local is simpler (no second machine). |

- **PC-TTS** — GPT-SoVITS runs on a separate PC over Tailscale. The
  Jetson bridge sends text + receives WAV. See
  [../deploy/pc-tts/README.md](../deploy/pc-tts/README.md) for the
  Windows sidecar setup and the `KARIN_TTS_BASE_URL` pointer.
- **Local sovits** — opt-in via `LOCAL_SOVITS=yes bash deploy/setup.sh`.
  Clones GPT-SoVITS, downloads its ~4 GB upstream pretrained weights,
  installs `sovits-server.service`, and brings up the voice loop. Do
  NOT enable on Orin Nano 8 GB.

### Producing a voice bundle

A complete voice bundle is a triplet under `characters/<name>/voices/`:

- `ref.wav` — reference audio (≤10 s of the target voice)
- `gpt_model_*.ckpt` — trained GPT weights
- `sovits_model_*.pth` — trained SoVITS weights

Plus `characters/<name>/voice.yaml` (copy from `characters/template/voice.example.yaml`).

To produce a bundle, follow [../docs/training-guide.md](training-guide.md)
on a GPU machine (training a 32-epoch GPT + 16-epoch SoVITS typically
takes 45-90 minutes on a consumer RTX-class card). Drop it where sovits
actually runs:

```bash
# Local sovits (Orin NX 16 GB+): push the bundle to the Jetson.
ssh <jetson-user>@<jetson-ip> 'mkdir -p ~/Karin/characters/<name>/voices'
scp characters/<name>/voices/* \
    <jetson-user>@<jetson-ip>:~/Karin/characters/<name>/voices/
scp characters/<name>/voice.yaml \
    <jetson-user>@<jetson-ip>:~/Karin/characters/<name>/

# PC-TTS offload: put the same files on the PC running the sidecar
# instead, under deploy/pc-tts/characters/<name>/ (see its README).
```

See [characters/README.md](../characters/README.md) for the folder layout.

## Bootstrap GPT-SoVITS (local sovits only)

Skip this section if you're in text-only mode OR using PC-TTS offload.
`deploy/setup.sh` handles GPT-SoVITS bootstrap automatically when you
invoke it with `LOCAL_SOVITS=yes` and the voice bundle is present —
clones the repo, pins the commit, downloads the upstream pretrained
weights, creates the venv, filters out the CPU torch wheel. To do it
manually:

```bash
cd ~/Karin
mkdir -p third_party
git clone https://github.com/RVC-Boss/GPT-SoVITS.git third_party/GPT-SoVITS
(cd third_party/GPT-SoVITS && git checkout 2d9193b0)

# Download the ~4 GB upstream weights (one-time, ~15 min over Tailscale):
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

## Configure .env for Jetson constraints

```bash
cd ~/Karin
cp .env.example .env

# Edit .env — uncomment the Jetson COMPOSE_FILE line:
sed -i 's|^# COMPOSE_FILE=deploy/docker-compose.yml:deploy/docker-compose.jetson.yml|COMPOSE_FILE=deploy/docker-compose.yml:deploy/docker-compose.jetson.yml|' .env
sed -i 's|^COMPOSE_FILE=deploy/docker-compose.yml$|# &|' .env

# Set the model + STT to match Jetson constraints:
cat >> .env <<'EOF'
KARIN_LLM_MODEL=karin-tuned:latest
KARIN_LLM_TIMEOUT=300
KARIN_NUM_CTX=2048
KARIN_STT_MODEL=tiny.en
KARIN_STT_DEVICE=cpu
KARIN_STT_COMPUTE_TYPE=int8
EOF
```

### Three voice paths, pick one

Karin ships with TTS + STT **both disabled by default**. Choose one of:

**(a) Text-only (do nothing.)** The setup above is complete. Skip the
rest of this subsection.

**(b) Local STT on the Jetson (tiny.en, CPU).** Run setup once with
the opt-in flag:
```bash
LOCAL_STT=yes bash deploy/setup.sh
```
That writes `KARIN_STT_ENABLED=true` + `KARIN_STT_MODEL=tiny.en` +
`KARIN_STT_DEVICE=cpu` + `KARIN_STT_COMPUTE_TYPE=int8` to `deploy/.env`.
No KARIN_STT_BASE_URL — the bridge loads faster-whisper in-process.
TTS stays off (won't fit alongside the LLM on Orin Nano). The knobs
above from the initial `.env` block are superseded; the LOCAL_STT
path picks the smallest Whisper so headroom stays positive.

**(c) Offload both to a PC over Tailscale (recommended for accuracy).**
Set up the sidecar on your PC per [../deploy/pc-tts/README.md](../deploy/pc-tts/README.md),
then add to `deploy/.env`:
```bash
cat >> .env <<'EOF'
KARIN_TTS_ENABLED=true
KARIN_TTS_BASE_URL=http://<pc-tailscale-ip>:9880
KARIN_STT_ENABLED=true
KARIN_STT_BASE_URL=http://<pc-tailscale-ip>:9880
EOF
```
The Jetson bridge will POST PCM to the sidecar on `/transcribe` rather
than loading faster-whisper in-process. Same URL and port as
`KARIN_TTS_BASE_URL` — the sidecar serves both endpoints.

Re-run `bash deploy/setup.sh` after adding the URLs. It will **probe**
both endpoints and warn if the PC isn't reachable (sidecar down,
Tailscale disconnected, wrong IP). Non-fatal — the web UI will still
start, and the sidebar TTS/STT dots will show a live "PC offline"
state until the sidecar comes back up.

Why these defaults:
- `karin-tuned:latest` — the production model. An iter-3 LoRA merged on
  `mannix/llama3.1-8b-abliterated:tools-q4_k_m` (~4.9 GB). Tool-routing
  was trained for this model — the 135-case eval baseline (93.3%
  routing / 91.9% reply / 59.2% tool-output) was measured against it.
  If karin-tuned isn't built yet, see
  [../sft/colab_sft.ipynb § 9](../sft/colab_sft.ipynb) for the one-time
  Modelfile build. Pre-LoRA fallback for a first-boot smoke test:
  `KARIN_LLM_MODEL=qwen2.5:3b` (lower routing accuracy but ~2 GB, very
  forgiving on memory). Other bases (Qwen3.5 4B, Hermes3 8B, Mistral)
  are listed in `config/models.yaml` with per-model overrides.
- `KARIN_LLM_TIMEOUT=300` — gives the bridge 5 min for cold-load + tool
  turns. First turn includes ~10s model load.
- `KARIN_NUM_CTX=2048` — what karin-tuned ships with in
  `config/models.yaml`. Enough for the system prompt (~3000 chars fits
  after tokenization) plus tool schemas. Multi-turn overflow is handled
  by compaction at ~50% of num_ctx. Raise to 3072 if you swap to a
  Qwen-family alternate whose tokenizer needs more room.
- `KARIN_STT_MODEL=tiny.en` — Orin-Nano-safe default. Uses
  ~400 MB of unified memory on CPU and leaves ~600 MB headroom next
  to karin-tuned (~6.8 GiB). Accuracy is ~10% WER on common English;
  combined with the pre-classifier's routing patterns it still picks
  up most voice-assistant prompts. Flip up to `base.en` (~700 MB,
  ~6-7% WER) only on Orin NX 16 GB or larger — on Orin Nano 8 GB that
  would push total memory past the
  7.4 GiB cap. Drop to `tiny.en` to claw back another ~300 MB at the
  cost of worse proper-noun recognition. For multilingual voice prompts,
  set `KARIN_STT_LANGUAGE=auto KARIN_STT_MODEL=base` and Whisper
  auto-detects per utterance — Karin still replies in English
  (enforced in the system prompt).
- `KARIN_STT_DEVICE=cpu` — PyPI's ARM64 ctranslate2 wheel is CPU-only,
  and this is also what fits: Orin Nano has no VRAM headroom for a
  second model while karin-tuned is loaded (measured 2026-04-24,
  ~600 MiB free at idle). CUDA STT on Jetson requires a from-source
  build of ctranslate2 AND a bigger Jetson (Orin NX 16 GB+) — see
  RUNBOOK's [Performance tuning](../RUNBOOK.md#performance-tuning-latency).
- `KARIN_STT_COMPUTE_TYPE=int8` — required on CPU. `int8_float16` is a
  CUDA-only type; `float16` + CPU produces NaN inference.

## First-time bring-up

```bash
docker compose up -d
docker compose logs -f web    # tail the web container until "Uvicorn running"
```

Then open `http://<jetson-ip>:80` in any browser on the tailnet (or
LAN). For the mic to work in browsers you need HTTPS — jump back to
Part 1's
[Enable HTTPS fronting for the browser UI](#6-later-enable-https-fronting-for-the-browser-ui)
and run the `tailscale serve` command now that the web container
is up.

You now have a running Karin on the Jetson. Ongoing operations —
updating, stopping, health checks, performance tuning — all live in
[RUNBOOK.md](../RUNBOOK.md).

---

# Troubleshooting

## SDK Manager doesn't detect the Jetson

- Verify recovery mode: `lsusb | grep -i nvidia` should show
  `NVIDIA Corp. APX` (vendor 0955).
- The USB-C cable matters — must be a data cable, not charge-only.
  Try the cable that came with the Jetson.
- Plug into the Jetson's front USB-C port, not the rear one (the
  rear port is power on some carriers).
- Re-do recovery mode (see [Put the Jetson into recovery mode](#put-the-jetson-into-recovery-mode)):
  full power off, short pin 9 to pin 10, power on, release after ~3s.
  Sometimes the timing matters.

## Flash fails partway through

- Check host disk space: SDK Manager needs ~40 GB free.
- Check internet — if a download stalled, retry the same flash;
  it resumes from cache.
- If your storage target is microSD and the flash hangs at "Writing
  image", the card may be slow / counterfeit. Try a name-brand card
  (SanDisk, Samsung) rated A2 or higher.

## Jetson doesn't boot after flash

- Disconnect the USB-C cable from the host before powering on the
  Jetson normally — leaving the host connected can confuse the
  boot detection.
- For NVMe targets: the carrier board's BIOS needs to be set to
  "Boot from NVMe". On Orin Nano dev kit this is automatic but
  flashing to NVMe sometimes requires re-flashing the bootloader
  — SDK Manager's "Storage Device Manager" panel can do this.

## Wrong JetPack version

This project is tested on **JetPack 6.x (L4T R36.x)**. Older JetPack 5
will work for some pieces but the Ollama install + dustynv container
images are tuned for JetPack 6. If SDK Manager only offers JetPack 5,
make sure you've updated SDK Manager itself (`sudo apt update && sudo
apt install --only-upgrade sdkmanager`).
