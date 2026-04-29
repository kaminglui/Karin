#!/usr/bin/env bash
#
# GPT-SoVITS container entrypoint.
#
# Runs the upstream requirements install and NLTK download on first
# start, then launches api_v2.py. Idempotent via marker files so
# subsequent container starts go straight to api_v2.

set -euo pipefail

cd /gpt-sovits

# Self-healing install check: probe for a sentinel import. The bulk of
# the install lives in the container's filesystem, NOT the bind-mounted
# clone — so a marker in the clone can survive a container recreate
# while the packages themselves have been wiped. Python's own import
# resolver is the honest source of truth.
PIP_BIN="$(command -v pip || command -v pip3)"
PY_BIN="$(command -v python3 || command -v python)"

if ! "${PY_BIN}" -c "import fastapi" 2>/dev/null; then
    echo "[sovits-entrypoint] installing upstream GPT-SoVITS requirements"
    if [[ -f /gpt-sovits/requirements.txt ]]; then
        grep -vE '^(torch|torchaudio|torchvision)(==|>=|\b)' /gpt-sovits/requirements.txt \
            > /tmp/gpt-sovits-reqs.txt || true
        "${PIP_BIN}" install -r /tmp/gpt-sovits-reqs.txt || \
            echo "[sovits-entrypoint] WARN: some requirements failed to install"
    else
        echo "[sovits-entrypoint] WARN: /gpt-sovits/requirements.txt missing — is the clone bind-mounted?"
    fi
else
    echo "[sovits-entrypoint] requirements already present"
fi

# NLTK corpora land in /root/nltk_data inside the container — also
# wiped on recreate. Re-download is fast (small files) and idempotent.
"${PY_BIN}" - <<'PY' || echo "[sovits-entrypoint] NLTK download failed; continuing"
import nltk
for pkg in ("averaged_perceptron_tagger_eng", "cmudict", "punkt"):
    try: nltk.download(pkg, quiet=True)
    except Exception as e: print(f"NLTK {pkg}: {e}")
PY

echo "[sovits-entrypoint] starting api_v2.py on 0.0.0.0:9880"
# api_v2.py defaults to binding 127.0.0.1, which is fine inside the
# container but invisible to docker's port forwarder. We want
# 0.0.0.0 so the published 9880 actually has a service to reach.
exec "${PY_BIN}" api_v2.py -a 0.0.0.0 -p 9880
