#!/usr/bin/env bash
#
# Push code from PC to Jetson + rebuild + restart.
# Run from the repo root on the PC:
#
#   bash deploy/update.sh                    # default Jetson SSH target
#   bash deploy/update.sh user@1.2.3.4       # override target
#
# Idempotent. Skips the bulk transfers when nothing changed.

set -euo pipefail

# Override with $1 on the command line or KARIN_JETSON_SSH in the environment.
# The fallback below is a placeholder — set KARIN_JETSON_SSH=user@host in your
# shell profile so you don't have to pass it each time.
JETSON="${1:-${KARIN_JETSON_SSH:-jetson@jetson-orin-nano}}"
REPO_NAME="$(basename "$(pwd)")"
REMOTE_DIR="~/${REPO_NAME}"

# Sanity: must run from repo root
[[ -f bridge/main.py && -d deploy ]] || {
    echo "Run this from the Karin repo root, not deploy/." >&2
    exit 1
}

echo "[update] target: ${JETSON}:${REMOTE_DIR}"

# rsync over SSH — fastest delta-only transfer. Excludes match
# .dockerignore so we don't push gigs of weights / venvs / state.
echo "[update] rsync source"
rsync -avz --delete \
    --exclude='.venv/' \
    --exclude='.venv-*/' \
    --exclude='.git/' \
    --exclude='.claude/' \
    --exclude='.pytest_cache/' \
    --exclude='.vscode/' \
    --exclude='__pycache__/' \
    --exclude='**/__pycache__/' \
    --exclude='third_party/' \
    --exclude='helper/' \
    --exclude='tmp/' \
    --exclude='data/' \
    --exclude='characters/*/voices/*.ckpt' \
    --exclude='characters/*/voices/*.pth' \
    --exclude='characters/*/voices/ref.wav' \
    --exclude='voice_training/_OTHER/' \
    --exclude='voice_training/export/OLD/' \
    --exclude='voice_training/export/*.ckpt' \
    --exclude='voice_training/export/*.pth' \
    --exclude='bridge/news/data/' \
    --exclude='bridge/trackers/data/' \
    --exclude='bridge/alerts/data/' \
    --exclude='deploy/ollama/' \
    ./ "${JETSON}:${REMOTE_DIR}/"

echo "[update] remote: rebuild + restart what changed"
# shellcheck disable=SC2087  # we want the heredoc to expand on the local side
ssh "${JETSON}" bash -s <<EOF
set -euo pipefail
cd "${REMOTE_DIR}"

echo "[update] docker compose build (cached layers reused)"
docker compose build 2>&1 | tail -5

echo "[update] docker compose up -d (recreate only changed services)"
docker compose up -d 2>&1 | tail -10

echo "[update] status"
docker compose ps --format 'table {{.Name}}\t{{.Status}}'
EOF

echo "[update] done. tail logs with: ssh ${JETSON} 'cd ${REMOTE_DIR} && docker compose logs -f web'"
