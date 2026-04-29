#!/usr/bin/env bash
#
# Block until the GPT-SoVITS api_v2.py server is accepting HTTP requests.
# Used as ExecStartPre in assistant.service so the bridge doesn't race the
# TTS server during boot.
#
# Usage:
#   wait-for-tts.sh <base-url> <max-seconds>
#   wait-for-tts.sh http://127.0.0.1:9880 120
#
# Probes /docs (FastAPI's built-in Swagger UI endpoint). 200 = ready.

set -eu

URL="${1:-http://127.0.0.1:9880}"
MAX="${2:-120}"

for i in $(seq 1 "$MAX"); do
    if curl -fsS -o /dev/null "$URL/docs"; then
        echo "[wait-for-tts] $URL ready after ${i}s"
        exit 0
    fi
    sleep 1
done

echo "[wait-for-tts] timed out after ${MAX}s waiting for $URL/docs" >&2
exit 1
