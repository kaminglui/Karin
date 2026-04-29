"""Voice (sovits) lifecycle helpers exposed via /api/voice/*.

We talk to the Docker Engine API directly over its unix socket — the
web container mounts ``/var/run/docker.sock`` for this. No docker CLI
inside the image; httpx (already a dep) handles the unix-socket HTTP.

Why this exists: on Orin Nano 8 GB, sovits + the LLM can't both stay
GPU-resident reliably. The recommended pattern is to keep sovits
stopped during normal text use and start it on demand when voice is
wanted. This module + the matching server endpoints let the user
toggle that from the browser instead of SSHing in to run
``docker compose stop sovits``.

All helpers return a status dict with the same shape so the UI can
update from any of them:

    {"running": bool, "health": str|None, "exists": bool, "error": str|None}
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

log = logging.getLogger("bridge.voice")

# Override-able for tests / non-default deploy layouts.
_DOCKER_SOCK = os.environ.get("KARIN_DOCKER_SOCK", "/var/run/docker.sock")
_SOVITS_CONTAINER = os.environ.get("KARIN_SOVITS_CONTAINER", "karin-sovits")
_OLLAMA_URL = os.environ.get("KARIN_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# Time to wait after an LLM unload before starting sovits, giving the
# unified-memory allocator a beat to actually reclaim the freed pages.
_POST_UNLOAD_SETTLE_S = 2.0


def _client() -> httpx.Client:
    """httpx client that speaks Docker Engine API over the bind-mounted
    unix socket. Base URL is irrelevant for unix transports — we only
    care about the path part of subsequent requests."""
    transport = httpx.HTTPTransport(uds=_DOCKER_SOCK)
    return httpx.Client(transport=transport, base_url="http://docker", timeout=15.0)


def status() -> dict[str, Any]:
    """Inspect the sovits container.

    Returns a status dict (always populated, never raises) so callers
    can treat /api/voice/status as a pure read with predictable shape.
    """
    try:
        with _client() as c:
            r = c.get(f"/containers/{_SOVITS_CONTAINER}/json")
            if r.status_code == 404:
                return {"running": False, "health": None, "exists": False, "error": None}
            r.raise_for_status()
            data = r.json()
        state = data.get("State") or {}
        running = bool(state.get("Running"))
        health: str | None = None
        if running:
            health = ((state.get("Health") or {}).get("Status"))
        return {"running": running, "health": health, "exists": True, "error": None}
    except Exception as e:
        log.debug("voice.status: %s", e)
        return {"running": False, "health": None, "exists": False, "error": str(e)}


def _post_action(verb: str, params: dict | None = None) -> dict[str, Any]:
    """Issue a /containers/{id}/{verb} call. 204 = ok, 304 = already
    in target state (e.g. start a running container) — both fine."""
    try:
        with _client() as c:
            r = c.post(
                f"/containers/{_SOVITS_CONTAINER}/{verb}",
                params=params or {},
            )
            if r.status_code not in (204, 304):
                r.raise_for_status()
        log.info("voice: %s requested for %s", verb, _SOVITS_CONTAINER)
    except Exception as e:
        log.warning("voice.%s failed: %s", verb, e)
        # Fall through to status() below so the response is still
        # structured for the UI; the error field carries the cause.
        s = status()
        s["ok"] = False
        s["error"] = str(e)
        return s
    s = status()
    s["ok"] = True
    return s


def _unload_ollama_models(timeout_s: float = 5.0) -> list[str]:
    """Evict every currently-loaded model from Ollama by sending a
    ``keep_alive: 0`` generate request — the documented unload idiom.
    Returns the list of model names we attempted to unload. Silent
    failure: on network / timeout / HTTP error we log and return
    whatever we managed. Caller shouldn't let any of this block sovits
    start if it fails.
    """
    unloaded: list[str] = []
    try:
        with httpx.Client(timeout=timeout_s) as c:
            ps = c.get(f"{_OLLAMA_URL}/api/ps")
            ps.raise_for_status()
            loaded = (ps.json() or {}).get("models") or []
            for m in loaded:
                name = m.get("name") or m.get("model")
                if not name:
                    continue
                try:
                    c.post(
                        f"{_OLLAMA_URL}/api/generate",
                        json={
                            "model": name,
                            "keep_alive": 0,
                            "prompt": "",
                            "stream": False,
                        },
                        timeout=timeout_s,
                    )
                    unloaded.append(name)
                    log.info("voice.start: unloaded ollama model %s", name)
                except Exception as e:
                    log.warning("couldn't unload %s: %s", name, e)
    except Exception as e:
        log.warning("couldn't query ollama /api/ps: %s", e)
    return unloaded


def start() -> dict[str, Any]:
    """Bring sovits up.

    On Orin Nano 8 GB the LLM and sovits can't both be GPU-resident —
    sovits's model-load spike will OOM mid-load-BERT and hang forever
    if the LLM is still holding its ~2.3 GB. So: unload any loaded
    Ollama models first, sleep briefly for the allocator to settle,
    THEN start the container. The LLM reloads naturally on the next
    chat turn because ``keep_alive: -1`` is set in bridge/llm.py.

    Idempotent — already-running returns ok=True. Unload step is
    best-effort (sovits start still fires if /api/ps is unreachable).
    """
    _unload_ollama_models()
    if _POST_UNLOAD_SETTLE_S > 0:
        time.sleep(_POST_UNLOAD_SETTLE_S)
    return _post_action("start")


def stop() -> dict[str, Any]:
    """Bring sovits down. Idempotent — already-stopped returns ok=True.
    ``t=10`` gives the container 10 seconds for graceful shutdown
    before SIGKILL — sovits's NLTK / model unload is fast enough."""
    return _post_action("stop", params={"t": 10})
