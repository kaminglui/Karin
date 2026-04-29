"""FastAPI app for the Karin web voice interface.

Serves a single-page push-to-talk UI and a single ``/api/turn`` endpoint
that:

1. Receives an uploaded audio blob (any format a browser produces)
2. Decodes to 16 kHz mono int16 PCM via ffmpeg
3. Runs it through the shared STT -> LLM -> TTS pipeline
4. Returns a JSON payload containing the transcript and a base64-encoded
   WAV of the spoken reply

Launch (local / PC testing):

    python -m uvicorn web.server:app --host 0.0.0.0 --port 8001

Then visit http://localhost:8001 on the same machine, or
http://<tailscale-ip>:8001 from a device on your tailnet.

The app holds its own ``WhisperSTT`` / ``OllamaLLM`` / ``SoVITSTTS``
instances, separate from ``bridge/main.py``. Models are constructed at
module load (cost: ~5-10 s cold start — most of it is Whisper loading
onto CUDA). A single ``asyncio.Lock`` serializes pipeline calls so a
second turn during processing waits instead of colliding on LLM history
or TTS state.

Runtime prerequisites (mirror bridge/main.py):

- ``ollama.service`` running on http://localhost:11434
- GPT-SoVITS ``api_v2.py`` running on http://localhost:9880
- ``ffmpeg`` on PATH (server-side audio decoding)
"""
from __future__ import annotations

import asyncio
import base64
import io
import json as json_mod
import logging
import os
import queue as thread_queue
import random
import subprocess
import threading
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from bridge import tools as tools_mod
from bridge.history import ConversationStore, maybe_compact
from bridge.jobs import JobStore, TurnWorker
from bridge.llm import OllamaLLM
from bridge.memory import MAX_MEMORY_CHARS, MemoryStore
from bridge.stt import WhisperSTT
from bridge.tts import SoVITSTTS, TTSError
from bridge.utils import REPO_ROOT, load_config, resolve_path

log = logging.getLogger("web")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Scrub secret query params (api_key, token, etc.) from every log
# record in the process. httpx's INFO logs full URLs including query
# strings; without this, an accidentally-logged URL like
# ``?api_key=REALKEY`` would land in container logs. The filter
# rewrites those to ``<redacted>`` at emit time — originals stay
# untouched in memory, only the log line is sanitized.
from bridge.log_filters import install_secret_scrubber  # noqa: E402
install_secret_scrubber()

# Phase H: one-shot legacy->profile migration. MUST run before any
# subsystem store is instantiated, because singletons like feedback_store
# resolve their data path through active_profile() at import time — if
# migration hasn't created data/profiles/default/ and moved the legacy
# files yet, those stores would write to an empty profile dir while the
# user's real data sat orphaned at the old paths.
from bridge.profiles.migration import run_legacy_migration  # noqa: E402
run_legacy_migration()

# --- load config + build pipeline ------------------------------------------

CONFIG_PATH = REPO_ROOT / "config" / "assistant.yaml"
cfg = load_config(CONFIG_PATH)

stt_cfg = cfg["stt"]
llm_cfg = cfg["llm"]
tts_cfg = cfg["tts"]

# Feature registry — single source of truth for subsystem/tool flags.
# See config/features.yaml for defaults; env vars (KARIN_STT_ENABLED,
# KARIN_TTS_ENABLED, KARIN_BANDIT, ...) still override, so existing
# deploy flows work unchanged. RUNBOOK § Feature flags has the full
# list and re-enable procedures.
from bridge import features as _features  # noqa: E402

STT_ENABLED: bool = _features.is_enabled("stt")
TTS_ENABLED: bool = _features.is_enabled("tts")

# Runtime STT toggle. Defaults to STT_ENABLED at boot so it matches the
# features.yaml / KARIN_STT_ENABLED resolution. The `/api/stt/{enable,
# disable}` endpoints flip this at request time without reloading the
# Whisper model — the model stays warm in memory regardless, the flag
# just controls whether transcription actually runs. If STT_ENABLED is
# False at boot, the model was never loaded and the runtime flag can't
# be flipped on (the endpoint returns 503 with that explanation).
_stt_runtime_enabled: bool = STT_ENABLED

log.info("loading models ... (STT=%s, TTS=%s)", STT_ENABLED, TTS_ENABLED)
if STT_ENABLED:
    # Remote vs local STT is picked at boot via KARIN_STT_BASE_URL.
    # Remote: POSTs to the Karin voice sidecar (/transcribe) so the
    #         Whisper memory cost stays on the PC. See
    #         deploy/pc-tts/tts_server.py.
    # Local:  loads faster-whisper in this process (see bridge/stt.py).
    _stt_base_url = os.environ.get("KARIN_STT_BASE_URL", "").strip()
    if _stt_base_url:
        from bridge.stt_remote import RemoteWhisperSTT  # noqa: E402
        stt = RemoteWhisperSTT(  # type: ignore[assignment]
            base_url=_stt_base_url,
            language=stt_cfg["language"],
            beam_size=stt_cfg["beam_size"],
            vad_filter=stt_cfg["vad_filter"],
        )
        log.info("STT: remote sidecar @ %s", _stt_base_url)
    else:
        stt = WhisperSTT(
            model=stt_cfg["model"],
            device=stt_cfg["device"],
            compute_type=stt_cfg["compute_type"],
            language=stt_cfg["language"],
            beam_size=stt_cfg["beam_size"],
            vad_filter=stt_cfg["vad_filter"],
        )
        log.info(
            "STT: local faster-whisper model=%s device=%s",
            stt_cfg["model"], stt_cfg["device"],
        )
else:
    stt = None  # type: ignore[assignment]
    log.info("STT disabled via KARIN_STT_ENABLED — skipping faster-whisper load")
memory_store = MemoryStore()

# Preference-feedback store: append-only JSONL of (prompt, tool_chain,
# reply, rating) so the retrieval bandit can nudge routing on similar
# future prompts. Disabled entirely when KARIN_BANDIT=off — set path
# is still created, just no embeddings computed and no hint injected.
from bridge.feedback import FeedbackStore, ollama_embed  # noqa: E402
from bridge.bandit import preference_hint, retry_hint  # noqa: E402

_BANDIT_ENABLED = _features.is_enabled("bandit")
_EMBED_MODEL = os.environ.get("KARIN_EMBED_MODEL", "nomic-embed-text")


def _embed(text: str) -> list[float] | None:
    if not _BANDIT_ENABLED:
        return None
    return ollama_embed(
        base_url=llm_cfg["base_url"],
        text=text,
        model=_EMBED_MODEL,
    )


# Phase H: feedback rows live under the active profile so thumbs up/down
# history doesn't leak between "work" and "family" etc.
from bridge.profiles import active_profile as _active_profile
feedback_store = FeedbackStore(
    path=_active_profile().feedback_dir / "entries.jsonl",
    embedder=_embed,
)


def _system_prompt_suffix(user_text: str | None = None) -> str:
    """Combined system-prompt suffix: memory + IP location + preference hint.

    Chained so the LLM gets all per-user context on every turn. Each
    source is fail-soft (empty string on error) — a missing signal
    doesn't break the prompt.

    ``user_text`` is the current turn's prompt, used by the bandit
    layer to compute its embedding and retrieve similar past feedback.
    None on non-turn contexts (e.g. summarizer) — the bandit block
    just returns empty then.
    """
    from bridge.location import user_location_context

    parts: list[str] = []
    mem = (memory_store.build_prompt_block() or "").strip()
    if mem:
        parts.append(mem)
    loc = (user_location_context() or "").strip()
    if loc:
        parts.append(loc)
    if _BANDIT_ENABLED and user_text:
        try:
            emb = _embed(user_text)
            hint = preference_hint(emb, feedback_store)
            if hint.strip():
                parts.append(hint)
        except Exception as e:
            log.debug("preference hint computation failed: %s", e)
    return "\n\n".join(parts)


# Per-model tuning overlay. Reads config/models.yaml and returns
# merged defaults + per-tag overrides. Lets us keep model-specific
# knobs (temperature, num_ctx, tool-loop caps, request timeout) in
# one declarative file instead of scattered constants.
from bridge.model_config import resolve_for as _resolve_model_cfg  # noqa: E402
_model_cfg = _resolve_model_cfg(llm_cfg["model"])

# Merge model-specific overrides on top of the assistant.yaml values
# so a tag's entry in models.yaml wins, but assistant.yaml still
# supplies the base (system_prompt, base_url, options-dict defaults).
_options = dict(llm_cfg.get("options", {}))
# `think` lives at request top-level for Ollama, not in options —
# OllamaLLM pops it out. Feed it in through options so the same
# extraction path handles the override.
if "think" in _model_cfg:
    _options["think"] = _model_cfg["think"]

llm = OllamaLLM(
    base_url=llm_cfg["base_url"],
    model=llm_cfg["model"],
    system_prompt=llm_cfg["system_prompt"],
    temperature=_model_cfg.get("temperature", llm_cfg["temperature"]),
    num_ctx=_model_cfg.get("num_ctx", llm_cfg["num_ctx"]),
    options=_options,
    request_timeout=float(_model_cfg.get(
        "request_timeout",
        llm_cfg.get("request_timeout", 300.0),
    )),
    system_prompt_suffix_provider=_system_prompt_suffix,
    max_per_tool=int(_model_cfg.get("max_per_tool", 1)),
    max_tool_iters=int(_model_cfg.get("max_tool_iters", 5)),
    history_pairs=int(_model_cfg.get("history_pairs", 0)),
    backend=llm_cfg.get("backend", "ollama"),
    under_fire_rescue=bool(llm_cfg.get("under_fire_rescue", True)),
    two_phase_compose=bool(llm_cfg.get("two_phase_compose", False)),
    hint_in_user_msg=bool(llm_cfg.get("hint_in_user_msg", False)),
)
log.info(
    "LLM tuning: model=%s per_tool=%d iters=%d history_pairs=%d temp=%s num_ctx=%s timeout=%s",
    llm_cfg["model"], llm.max_per_tool, llm.max_tool_iters, llm.history_pairs,
    _options.get("temperature", _model_cfg.get("temperature")),
    _model_cfg.get("num_ctx"),
    _model_cfg.get("request_timeout"),
)
if TTS_ENABLED:
    # Env var overrides the YAML default — the compose file passes
    # KARIN_TTS_BASE_URL from .env so the remote-PC Tailscale URL
    # takes effect without editing assistant.yaml.
    _tts_base = os.environ.get("KARIN_TTS_BASE_URL") or tts_cfg["base_url"]
    _tts_remote = "localhost" not in _tts_base and "127.0.0.1" not in _tts_base

    # When the TTS server runs REMOTELY (PC over Tailscale), voice
    # file paths must pass through AS-IS — the PC resolves them from
    # its own CWD (the GPT-SoVITS directory). resolve_path() would
    # resolve them against the Jetson's /app/ directory and produce
    # broken Linux absolute paths the PC can't find.
    _rp = (lambda p: str(p)) if _tts_remote else resolve_path

    tts = SoVITSTTS(
        base_url=_tts_base,
        endpoint=tts_cfg["endpoint"],
        ref_audio_path=_rp(tts_cfg["ref_audio_path"]),
        prompt_text=tts_cfg["prompt_text"],
        prompt_lang=tts_cfg["prompt_lang"],
        text_lang=tts_cfg["text_lang"],
        top_k=tts_cfg["top_k"],
        top_p=tts_cfg["top_p"],
        temperature=tts_cfg["temperature"],
        speed_factor=tts_cfg["speed_factor"],
        streaming_mode=tts_cfg["streaming_mode"],
        gpt_weights_path=_rp(tts_cfg.get("gpt_weights_path") or ""),
        sovits_weights_path=_rp(tts_cfg.get("sovits_weights_path") or ""),
        request_timeout=float(tts_cfg.get("request_timeout", 300.0)),
    )
else:
    tts = None  # type: ignore[assignment]
    log.info("TTS disabled via KARIN_TTS_ENABLED — skipping GPT-SoVITS client")
log.info("models loaded")

# --- background turn-job runner --------------------------------------------
#
# A single worker thread runs LLM turns OFF the request lifetime so a
# closed browser doesn't kill an in-progress turn. The HTTP layer just
# enqueues + later streams events from the persisted job log.

job_store = JobStore()
turn_worker = TurnWorker(job_store)


# --- conversation persistence + compaction ---------------------------------

history_store = ConversationStore()
_current_cid, _restored = history_store.load_current_or_new()
if _restored:
    llm.set_history(_restored)
    log.info("restored conversation %s (%d messages)", _current_cid, len(_restored))
else:
    log.info("started fresh conversation %s", _current_cid)

# Pulled from config so compaction scales when you swap to a 32k model.
_NUM_CTX: int = int(_model_cfg.get("num_ctx", llm_cfg.get("num_ctx", 4096)))


def _persist_and_maybe_compact(
    user_text: str = "",
    tools_called: list[str] | None = None,
    reply: str = "",
    audio_id: str | None = None,
) -> None:
    """Save current history, compact if it's outgrown the context budget.

    Called after each successful LLM turn (both streaming routes). Runs
    in the request thread; compaction is a single Ollama summarize call
    so it blocks that one turn but never a concurrent one thanks to
    ``pipeline_lock``.

    When ``user_text`` is provided, also records a turn-level flow note
    for debugging and UI display (what tool was called, what the
    routing classifier guessed, prompt + reply previews).

    ``audio_id`` (optional) pins this turn to a cached TTS WAV under
    ``data/audio_cache/<cid>/<audio_id>.wav``. Set when the streaming
    path is about to synthesize audio; a subsequent missing file (e.g.
    synthesis failed mid-stream or tmpfs was wiped on restart) is a
    recoverable 404 on the ``/api/audio/...`` endpoint.
    """
    # Snapshot once so compaction operates on a stable list rather than
    # re-acquiring the HistorySession lock for every indexed access.
    snapshot = llm._history.snapshot()
    history_store.save(_current_cid, snapshot)
    compacted = maybe_compact(
        snapshot, num_ctx=_NUM_CTX, summarize_fn=llm.summarize,
    )
    if compacted is not snapshot:
        llm.set_history(compacted)
        history_store.save(_current_cid, compacted)

    # Record a turn note for observability. Best-effort — never let
    # a note failure interrupt the chat flow.
    if user_text:
        try:
            from bridge.routing import classify as _routing_classify
            hint = _routing_classify(user_text)
        except Exception:
            hint = None
        try:
            history_store.append_turn_note(
                cid=_current_cid,
                user_text=user_text,
                tools_called=tools_called or [],
                routing_hint=hint,
                reply_preview=reply,
                audio_id=audio_id,
            )
        except Exception as e:
            log.debug("turn note failed (non-fatal): %s", e)

# --- FastAPI app -----------------------------------------------------------

app = FastAPI(title="Karin", docs_url=None, redoc_url=None)


# --- IP whitelist --------------------------------------------------------
# Configurable via security.ip_whitelist in assistant.yaml:
#   "tailscale"  — only 100.64.0.0/10 + localhost (default)
#   "off"        — no filtering (for public internet + auth layer)
#   [list]       — custom CIDR allowlist
import ipaddress as _ipaddress  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402

_TAILSCALE_NET = _ipaddress.ip_network("100.64.0.0/10")
_LOCALHOST_ADDRS = {"127.0.0.1", "::1"}
_PUBLIC_PREFIXES = ("/static/", "/api/health")

_wl_cfg = cfg.get("security", {}).get("ip_whitelist", "tailscale")
if _wl_cfg == "tailscale":
    _ALLOWED_NETS: list[_ipaddress.IPv4Network | _ipaddress.IPv6Network] | None = [_TAILSCALE_NET]
    log.info("IP whitelist: tailscale only (100.64.0.0/10 + localhost)")
elif _wl_cfg == "off" or _wl_cfg is False:
    _ALLOWED_NETS = None
    log.warning("IP whitelist: OFF — all IPs allowed. Add auth before exposing publicly!")
elif isinstance(_wl_cfg, list):
    _ALLOWED_NETS = []
    for cidr in _wl_cfg:
        try:
            _ALLOWED_NETS.append(_ipaddress.ip_network(str(cidr), strict=False))
        except ValueError as e:
            log.warning("ip_whitelist: invalid CIDR %r (%s), skipping", cidr, e)
    log.info("IP whitelist: custom %d network(s)", len(_ALLOWED_NETS))
else:
    _ALLOWED_NETS = [_TAILSCALE_NET]
    log.warning("ip_whitelist: unrecognized value %r, defaulting to tailscale", _wl_cfg)


class _IPWhitelistMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _ALLOWED_NETS is None:
            return await call_next(request)
        path = request.url.path
        if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)
        client_ip = request.client.host if request.client else None
        if client_ip and client_ip not in _LOCALHOST_ADDRS:
            try:
                addr = _ipaddress.ip_address(client_ip)
                if not any(addr in net for net in _ALLOWED_NETS):
                    log.warning("blocked %s -> %s", client_ip, path)
                    return PlainTextResponse(
                        "forbidden: IP not in allowlist", status_code=403,
                    )
            except ValueError:
                pass
        return await call_next(request)


app.add_middleware(_IPWhitelistMiddleware)


# Phase H.d: Tailscale IP routing middleware. On every request, check
# whether the client's IP maps to a profile that differs from the
# currently active one. If so, auto-switch (write active_profile.txt)
# so the NEXT restart serves the right profile. The current process
# keeps running on the old profile — singletons are already bound.
# This is a best-effort convenience, not a hard boundary. The resolved
# profile name is stashed in request.state.resolved_profile so
# endpoints can surface it (e.g. the restart-required banner).


class _ProfileRoutingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else None
        request.state.resolved_profile = None
        if client_ip:
            try:
                from bridge.profiles.routing import resolve_profile_for_ip
                from bridge.profiles import (
                    active_profile, profile_exists, set_active,
                )
                mapped = resolve_profile_for_ip(client_ip)
                if mapped and profile_exists(mapped):
                    request.state.resolved_profile = mapped
                    current = active_profile().name
                    if mapped != current:
                        set_active(mapped)
                        log.info(
                            "IP routing: %s -> profile %r (was %r, restart to apply)",
                            client_ip, mapped, current,
                        )
            except Exception as e:
                log.debug("IP routing middleware failed for %s: %s", client_ip, e)
        return await call_next(request)


app.add_middleware(_ProfileRoutingMiddleware)


# Content Security Policy. Scripts limited to our origin + the two CDNs
# we load (KaTeX + Plotly from cdn.jsdelivr.net). Blocks inline script
# execution by default — if an XSS sink in panels/*.js ever slips
# through an escape helper, the browser still refuses to run it.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    # KaTeX's css references font files (.woff2) at the same CDN — without
    # an explicit font-src directive the browser falls back to default-src
    # ('self') and blocks them, leaving math glyphs unrendered. data: is
    # included so any base64-inlined fonts still resolve.
    "font-src 'self' https://cdn.jsdelivr.net data:; "
    "img-src 'self' data: blob:; "
    "media-src 'self' blob:; "
    "connect-src 'self'; "
    # 'self' (not 'none') so the main UI's popup iframe can load
    # /ui/* same-origin panels. Cross-origin embedding remains blocked.
    "frame-ancestors 'self'"
)


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", _CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response


app.add_middleware(_SecurityHeadersMiddleware)


# Per-IP rate limiter. **Default OFF** — only makes sense when Karin is
# exposed beyond a trusted tailnet. Enable by setting
# ``security.rate_limit.enabled: true`` in config/assistant.yaml.
# Simple sliding-window in process memory; single uvicorn worker only.
_RATE_EXPENSIVE_PREFIXES = ("/api/turn", "/api/web-search")
_RATE_CFG = cfg.get("security", {}).get("rate_limit", {}) or {}
_RATE_ENABLED = bool(_RATE_CFG.get("enabled", False))
_RATE_DEFAULT_PER_MIN = int(_RATE_CFG.get("default_per_minute", 60))
_RATE_EXPENSIVE_PER_MIN = int(_RATE_CFG.get("expensive_per_minute", 20))
_RATE_WINDOW_S = 60.0
from collections import deque as _deque  # noqa: E402
_rate_buckets: dict[tuple[str, str], _deque] = {}
_rate_lock = threading.Lock()


class _RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)
        client_ip = request.client.host if request.client else ""
        if not client_ip or client_ip in _LOCALHOST_ADDRS:
            return await call_next(request)
        is_expensive = any(path.startswith(p) for p in _RATE_EXPENSIVE_PREFIXES)
        bucket = "expensive" if is_expensive else "default"
        cap = _RATE_EXPENSIVE_PER_MIN if is_expensive else _RATE_DEFAULT_PER_MIN
        import time as _t
        now = _t.monotonic()
        cutoff = now - _RATE_WINDOW_S
        key = (client_ip, bucket)
        with _rate_lock:
            q = _rate_buckets.get(key)
            if q is None:
                q = _deque()
                _rate_buckets[key] = q
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= cap:
                retry_after = max(1, int(q[0] + _RATE_WINDOW_S - now))
                log.warning(
                    "rate limit: %s on %s bucket (%d/%d/min)",
                    client_ip, bucket, len(q), cap,
                )
                return PlainTextResponse(
                    "rate limit exceeded", status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            q.append(now)
        return await call_next(request)


if _RATE_ENABLED:
    app.add_middleware(_RateLimitMiddleware)
    log.info("rate limiter: enabled (default %d/min, expensive %d/min)",
             _RATE_DEFAULT_PER_MIN, _RATE_EXPENSIVE_PER_MIN)
else:
    log.info("rate limiter: disabled (set security.rate_limit.enabled=true to opt in)")


STATIC_DIR = REPO_ROOT / "web" / "static"
CHARACTERS_DIR = REPO_ROOT / "characters"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/characters/{char_name}/expressions/{filename}")
async def character_expression(char_name: str, filename: str):
    """Serve per-character expression PNGs for the avatar animation.

    Fallback chain (so incomplete character bundles still render):
      1. characters/<char>/expressions/<filename>
      2. characters/<char>/expressions/default.png
      3. web/static/faces/<filename>   (legacy shared faces)
      4. web/static/faces/default.png
      5. 404

    Each hit is normalized to a fixed square (512×512 center-crop
    cover) before serving, so user-supplied images of any size/aspect
    swap cleanly in the avatar button. Normalized bytes are cached
    under data/expressions_cache/ keyed by source mtime.
    """
    import re as _re
    from fastapi.responses import Response

    if not _re.match(r"^[a-zA-Z0-9_-]+$", char_name):
        raise HTTPException(400, "invalid character name")
    if not _re.match(r"^[a-zA-Z0-9_.-]+\.png$", filename):
        raise HTTPException(400, "invalid filename")

    from bridge.expressions import resolve_expression
    result = await run_in_threadpool(
        resolve_expression,
        CHARACTERS_DIR / char_name,
        filename,
        STATIC_DIR / "faces",
    )
    if result is None:
        raise HTTPException(404, "expression not found")
    data, media_type = result
    return Response(content=data, media_type=media_type)


@app.get("/api/audio/{conversation_id}/{audio_id}")
async def cached_audio(conversation_id: str, audio_id: str):
    """Serve a per-turn cached WAV from the audio cache.

    Pairs with the ``/api/text-turn-stream`` done event's ``audio_id``
    field. Returns 404 if the file isn't there — legitimate states
    include: the container restarted (audio_cache is tmpfs), synthesis
    failed for that turn, or the conversation was deleted. The browser
    handles 404 by hiding the replay button for that turn.
    """
    import re as _re
    from fastapi.responses import FileResponse
    if not _re.match(r"^[a-zA-Z0-9_-]+$", conversation_id):
        raise HTTPException(400, "invalid conversation id")
    if not _re.match(r"^[a-zA-Z0-9]+$", audio_id):
        raise HTTPException(400, "invalid audio id")

    from bridge.audio_cache import resolve_turn_audio
    path = resolve_turn_audio(conversation_id, audio_id)
    if path is None:
        raise HTTPException(404, "audio not cached")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/active-character")
async def active_character():
    """Return the active character name for frontend configuration."""
    import os
    char = os.environ.get("KARIN_CHARACTER") or cfg.get("character", "karin")
    return JSONResponse({"character": char})


# Merge the panels subsystem in. Both the JSON read-layer (/api/alerts,
# /api/news, /api/trackers, /api/chat/stream) and the standalone panel
# HTML pages (/ui/alerts, /ui/news, /ui/trackers) live in web/panels_api
# and attach here. The main page at "/" stays the unified voice + chat +
# panels UI; /ui/* routes are drill-down standalone views.
from web.panels_api import router as _panels_api_router
from web.panels_api import ui_router as _panels_ui_router
app.include_router(_panels_api_router)
app.include_router(_panels_ui_router)

# Background pollers (news ingest + alerts scan). Start at FastAPI
# startup via the lifespan-style hook so the threads only spin up
# once the app is actually ready to serve, and stop cleanly on exit
# so tests / reloads don't leave orphaned threads.
from bridge import pollers as _pollers  # noqa: E402
from bridge import startup_probes as _startup_probes  # noqa: E402

@app.on_event("startup")
async def _start_background_pollers() -> None:
    _pollers.start_pollers([
        _pollers.news_poller(),
        _pollers.alerts_poller(),
        _pollers.trackers_poller(),
        _pollers.reminders_poller(),
        _pollers.calendar_poller(),
        _pollers.digest_poller(),
    ])
    # Sanity-check external API reachability at boot so we don't wait
    # 20 min to discover that news fetches are failing. Probes are
    # read-only, ~8 s each (in parallel), and failures only log.
    await asyncio.get_running_loop().run_in_executor(None, _startup_probes.run_probes)

@app.on_event("shutdown")
async def _stop_background_pollers() -> None:
    _pollers.stop_pollers()
    await _close_probe_client()


# Reused across the /api/voice/status + /api/stt/status remote-sidecar
# probes (each polled every 2 s by the browser). One pooled client beats
# allocating a fresh AsyncClient per poll — that pattern was opening a
# new TCP connection to the PC sidecar on every status tick.
_PROBE_CLIENT = None  # type: ignore[var-annotated]


def _get_probe_client():
    """Lazy module-level httpx.AsyncClient for sidecar reachability probes.

    Initialised on first use rather than at import so importing server.py
    in non-async contexts (tests, scripts) doesn't try to bind to the
    running event loop too early.
    """
    global _PROBE_CLIENT
    if _PROBE_CLIENT is None:
        import httpx
        _PROBE_CLIENT = httpx.AsyncClient(timeout=2.0)
    return _PROBE_CLIENT


async def _close_probe_client() -> None:
    global _PROBE_CLIENT
    if _PROBE_CLIENT is not None:
        try:
            await _PROBE_CLIENT.aclose()
        except Exception as e:
            log.warning("probe client close raised: %s", e)
        _PROBE_CLIENT = None

# Serializes pipeline access. Belt-and-suspenders alongside the client's
# button disable: if a second turn manages to arrive, it waits instead of
# clobbering LLM history or racing TTS.
pipeline_lock = asyncio.Lock()


# --- Repeat-prompt dedup ---------------------------------------------------
#
# Catches the rapid-fire-resend pattern (double-clicked send button, page-
# reload retries, accidental enter-key repeats) that would otherwise queue
# behind ``pipeline_lock`` and run identical 15-25 s LLM turns over again.
# Bounded LRU keyed on the normalized user text; stale entries are dropped
# on access; entries past TTL are ignored. Sized for a single-user device:
# 32 distinct prompts × ~30 s window covers the realistic spam shapes
# without exposing a memory-growth surface.
#
# Skipped for prompts that look time-sensitive — replaying yesterday's
# weather or stock price would be misleading. The check is conservative
# (substring match) — false positives just bypass the cache, never serve
# stale data.

import collections as _collections
import re as _re
import time as _time

# Regex: a hit means "result depends on wall-clock time" → don't cache.
# Word-boundary anchored so "tomorrow"/"today"/"now" don't false-positive
# on prefixes (e.g. "downpour" does not contain a "now" boundary token).
_TIME_SENSITIVE = _re.compile(
    r"\b("
    r"time|now|today|tonight|tomorrow|yesterday|"
    r"weather|forecast|raining|temperature|hot|cold|"
    r"news|headlines?|breaking|happening|"
    r"alerts?|warnings?|advisor(?:y|ies)|"
    r"digest|brief|"
    r"price|trading|rate|exchange|quote|gold|silver|oil"
    r")\b",
    _re.IGNORECASE,
)

_DEDUP_TTL_S = 30.0
_DEDUP_MAX = 32
_dedup_lock = asyncio.Lock()
_dedup_cache: "_collections.OrderedDict[str, tuple[float, dict]]" = _collections.OrderedDict()


def _dedup_normalize(text: str) -> str:
    """Lower-case + collapse whitespace. Keys are equivalence classes —
    'What time is it?' and 'what  time is it?' map to the same key."""
    return " ".join(text.lower().strip().split())


def _dedup_is_time_sensitive(text: str) -> bool:
    return bool(_TIME_SENSITIVE.search(text))


async def _dedup_lookup(text: str) -> dict | None:
    """Return a cached response dict if the normalized prompt was seen
    within the TTL window. Side effect: prunes the matched entry if
    expired, and bumps recent hits to the LRU front."""
    if _dedup_is_time_sensitive(text):
        return None
    key = _dedup_normalize(text)
    async with _dedup_lock:
        entry = _dedup_cache.get(key)
        if entry is None:
            return None
        ts, response = entry
        if _time.monotonic() - ts > _DEDUP_TTL_S:
            _dedup_cache.pop(key, None)
            return None
        _dedup_cache.move_to_end(key)
        return response


async def _dedup_store(text: str, response: dict) -> None:
    """Cache a fresh response for ``text`` if it isn't time-sensitive.
    Bounded — oldest entries are evicted past ``_DEDUP_MAX``."""
    if _dedup_is_time_sensitive(text):
        return
    key = _dedup_normalize(text)
    async with _dedup_lock:
        _dedup_cache[key] = (_time.monotonic(), response)
        _dedup_cache.move_to_end(key)
        while len(_dedup_cache) > _DEDUP_MAX:
            _dedup_cache.popitem(last=False)


def _dedup_clear() -> None:
    """Test hook: drop the cache. Synchronous because tests run outside
    an event loop. Safe — the `OrderedDict.clear()` is atomic."""
    _dedup_cache.clear()


# Used by job runners (synchronous threads). ``pipeline_lock`` is
# ``asyncio.Lock`` and can't be acquired from a plain thread, so the
# worker holds this one while it runs ``chat_stream`` + persist for a
# turn.
import threading as _threading
_job_pipeline_lock = _threading.Lock()


@asynccontextmanager
async def _pipeline_and_job_lock():
    """Hold both pipeline locks for a critical section that mutates
    ``llm._history`` or ``_current_cid`` from an async route.

    Acquires ``pipeline_lock`` (serializing with other async routes)
    AND ``_job_pipeline_lock`` (serializing with any in-flight worker-
    thread turn). This is what prevents a DELETE / new-chat / switch
    from racing a streaming-job turn that's still appending the
    assistant message — without the worker lock, the worker would
    finish its turn and persist the new history under the *post-DELETE*
    conversation id, silently corrupting the wrong file.

    The threading lock is acquired via ``run_in_threadpool`` so the
    asyncio event loop isn't blocked waiting for a slow worker turn
    to finish; other unrelated async routes keep responding.
    """
    async with pipeline_lock:
        await run_in_threadpool(_job_pipeline_lock.acquire)
        try:
            yield
        finally:
            _job_pipeline_lock.release()


async def _chat_with_retry(
    user_text: str,
    max_attempts: int = 3,
    delay: float = 2.0,
    tool_calls_sink: list | None = None,
    commit_history: bool = True,
) -> str:
    """Call ``llm.chat()`` with automatic retry on empty responses.

    Passes the registered tool schemas so the LLM can call get_weather /
    get_time / etc. Tool execution happens inside ``llm.chat`` transparently.

    If ``tool_calls_sink`` is provided, each tool invocation during the
    turn appends ``{"name": ..., "arguments": ...}`` to the list. The
    unified voice/text/panels UI uses this to drive panel mounting from
    the same NDJSON stream that carries transcript + audio.
    """
    def _on_tool(name: str, args: dict, result: str) -> None:
        if tool_calls_sink is not None:
            # Truncate long results so the NDJSON line stays small and
            # the tool-trace stays readable in the UI. The full result
            # is still logged server-side.
            preview = (result or "").strip()
            if len(preview) > 600:
                preview = preview[:600].rstrip() + "\u2026"
            tool_calls_sink.append({
                "name": name,
                "arguments": args,
                "result": preview,
            })
    for attempt in range(max_attempts):
        reply = await run_in_threadpool(
            llm.chat, user_text, tools_mod.active_tool_schemas(), _on_tool, commit_history,
        )
        reply = reply.strip()
        if reply:
            return reply
        log.warning(
            "LLM returned empty, retrying in %.0fs (%d/%d)",
            delay, attempt + 1, max_attempts,
        )
        await asyncio.sleep(delay)
    raise HTTPException(
        status_code=502,
        detail=f"LLM returned empty after {max_attempts} attempts",
    )

# --- STT silence / hallucination handling ---------------------------------

# Whisper (and faster-whisper) frequently hallucinate one of a short list of
# filler phrases when the input is silent or just noise. We treat any of
# these — on their own — as "no real speech" and let the LLM handle it
# instead of returning a 422 that renders as a red error.
_WHISPER_SILENCE_HALLUCINATIONS: frozenset[str] = frozenset({
    "", ".", "...", "?", "!",
    "thank you.", "thanks.", "thank you", "thanks",
    "bye.", "bye", "okay.", "ok.", "yeah.", "yeah",
    "you", "you.",
    "thanks for watching.", "thanks for watching",
    "please subscribe.", "please subscribe",
})


def _is_likely_silence(text: str) -> bool:
    """True if the STT output is empty or a known silence hallucination.

    Whisper returns these on pure noise / silence often enough that we
    can't trust a short match — cheaper to route the turn through the
    LLM with an "unclear audio" sentinel than to answer "thank you."
    with a real response.
    """
    norm = (text or "").strip().lower()
    if not norm:
        return True
    if norm in _WHISPER_SILENCE_HALLUCINATIONS:
        return True
    return False


# Canned replies used when STT returns nothing usable. We skip the LLM
# call entirely here — a 15-second wait for the model to say "could you
# repeat?" is worse UX than an instant deterministic phrase. The server
# rotates through these so it doesn't sound like a recording.
_SILENCE_REPLIES: tuple[str, ...] = (
    "Sorry, I didn't catch that. Could you say it again?",
    "I didn't quite hear you — could you repeat that?",
    "Hmm, I missed that. One more time?",
    "Could you repeat that? I didn't catch it.",
)


# Upload size cap (~10 MB). A ~30 s Opus clip is about 400 KB; 10 MB gives
# plenty of headroom while protecting against a misbehaving client.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def decode_to_pcm16k(audio_bytes: bytes) -> np.ndarray:
    """Decode any browser-supplied audio blob to 16 kHz mono int16 PCM.

    Uses ffmpeg as a subprocess so we get codec coverage for every format
    browsers produce (WebM/Opus on Android Chrome, MP4/AAC on iOS Safari,
    etc.) without having to parse any of them ourselves.

    Args:
        audio_bytes: Raw bytes of the uploaded audio file.

    Returns:
        1-D int16 numpy array of 16 kHz mono PCM.

    Raises:
        RuntimeError: If ffmpeg isn't on PATH, or exits non-zero.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "s16le",
                "-ac", "1",
                "-ar", "16000",
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Install ffmpeg and "
            "restart the server in a fresh shell so it picks up the new PATH."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            "ffmpeg decode timed out after 30s — likely a malformed or "
            "truncated upload."
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed [{proc.returncode}]: "
            f"{proc.stderr.decode(errors='replace')[:500]}"
        )
    return np.frombuffer(proc.stdout, dtype=np.int16)


def wrap_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Wrap int16 mono PCM in a WAV container for browser playback.

    Args:
        pcm: 1-D int16 numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        WAV file bytes (header + data).
    """
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


# --- routes ----------------------------------------------------------------

def _character_has_voice(char_dir: Path) -> bool:
    """Return True if `char_dir/voices/` holds a complete GPT-SoVITS triplet.

    A usable voice bundle needs all three:
      - ref.wav (or *_ref.wav) — reference clip
      - *.ckpt                  — GPT weights
      - *.pth                   — SoVITS weights

    Used to surface a "(text only)" indicator in the sidebar dropdown and
    to short-circuit voice-swap POSTs that would otherwise 400. Purely
    filesystem-driven — no character names are hard-coded.
    """
    voices_dir = char_dir / "voices"
    if not voices_dir.is_dir():
        return False
    has_ref = any(voices_dir.glob("*ref.wav"))
    has_ckpt = any(voices_dir.glob("*.ckpt"))
    has_pth = any(voices_dir.glob("*.pth"))
    return has_ref and has_ckpt and has_pth


def _character_is_activatable(char_dir: Path) -> bool:
    """True iff `char_dir` has either persona metadata or a voice bundle.

    The scanner uses this to filter the dropdown; the switch handler
    in `web.panels_api.tts_voice_switch` uses the same predicate to 400
    on un-activatable targets. Keeping both behind one helper means the
    dropdown can never show a choice the switch handler would reject.
    """
    return (char_dir / "voice.yaml").is_file() or _character_has_voice(char_dir)


def _scan_available_characters() -> list[dict]:
    """Walk `characters/<name>/` looking for face renderers.

    Each subdirectory is a character. A character is "available"
    (shows up in the sidebar face picker) if either:
      - `face.json` exists → procedural renderer (preferred when
        both are present).
      - `expressions/default.png` exists → bitmap renderer.

    Each entry also carries `has_voice: bool` so the frontend can:
      - annotate the dropdown label ("(text only)" for bundle-less chars)
      - skip the `/api/tts/voice` POST when the target has no weights

    Returns a list of `{name, type, face_config?, label, has_voice}` dicts,
    sorted by name. Dirs with neither face nor expressions are skipped so
    the dropdown only lists characters that at least render. Called fresh
    on every index render so adding a character folder is a zero-restart
    operation.
    """
    import json as _json
    out: list[dict] = []
    repo_root = STATIC_DIR.parent.parent
    chars_root = repo_root / "characters"
    if not chars_root.is_dir():
        return out
    for entry in sorted(chars_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        # Skip the "template" scaffold and any hidden dirs.
        if name.startswith(".") or name == "template":
            continue
        if not _character_is_activatable(entry):
            continue
        face_json = entry / "face.json"
        default_png = entry / "expressions" / "default.png"
        has_voice = _character_has_voice(entry)
        if face_json.is_file():
            try:
                cfg_blob = _json.loads(face_json.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("face.json for %r is invalid: %s", name, e)
                continue
            out.append({
                "name": name,
                "type": cfg_blob.get("type", "procedural-sun"),
                "face_config": cfg_blob,
                "label": cfg_blob.get("label") or name.replace("_", " ").title(),
                "has_voice": has_voice,
            })
        elif default_png.is_file():
            out.append({
                "name": name,
                "type": "bitmap",
                "face_config": None,
                "label": name.replace("_", " ").title(),
                "has_voice": has_voice,
            })
    return out


@app.get("/api/characters/available")
async def available_characters() -> JSONResponse:
    """List characters the sidebar can pick between.

    Hot-reloaded: the file system is scanned on every request, so
    adding or removing a character folder takes effect immediately
    (no bridge restart). Used by the sidebar face-picker dropdown
    and for refresh after an upload from the UI.
    """
    return JSONResponse({"characters": _scan_available_characters()})


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page UI.

    Placeholders read synchronously by the face picker (an async
    fetch would race with app.js's module-load getElementById
    captures, so we inject the values directly):

      - ACTIVE_CHARACTER: persona name (config.character); exposed
        as `window.KARIN_ACTIVE_CHARACTER`.
      - ACTIVE_VOICE: voice / face folder (config.tts.voice). Auto
        default for the face picker when no explicit override is
        set in localStorage.
      - AVAILABLE_FACES_JSON: JSON list of characters with a
        renderable face, populated from `_scan_available_characters()`.
        Drives the sidebar dropdown and embeds each character's
        face.json inline so we don't need a second HTTP fetch when
        the user switches.
    """
    import os as _os
    import json as _json
    # Resolution order: env override > config > "default" (the shipped
    # neutral fallback character). The runtime voice-switch endpoint
    # (web/panels_api.py::tts_voice_switch) sets KARIN_CHARACTER on
    # successful swap so reloads after a dropdown change see the new
    # selection without needing a server restart.
    active_character = (
        _os.environ.get("KARIN_CHARACTER")
        or cfg.get("character")
        or "default"
    )
    active_voice = (cfg.get("tts", {}) or {}).get("voice") or active_character
    available = _scan_available_characters()

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{ACTIVE_CHARACTER}}", active_character)
    html = html.replace("{{ACTIVE_VOICE}}", active_voice)
    html = html.replace(
        "{{AVAILABLE_FACES_JSON}}",
        _json.dumps(available),
    )
    # Cache-bust the static CSS/JS hashes per server start. Browsers
    # otherwise hold onto an old style.css across deploys, which made
    # theme tweaks ghost for users on hard refresh.
    import time as _time
    html = html.replace("{{ASSET_VERSION}}", str(int(_time.time())))
    return HTMLResponse(html)


@app.get("/api/graph")
async def graph_api(
    expression: str,
    variable: str = "x",
    x_min: float = -10.0,
    x_max: float = 10.0,
) -> JSONResponse:
    """Structured plot data — x array + per-series y arrays."""
    data = await run_in_threadpool(
        tools_mod.graph_data, expression, variable, x_min, x_max,
    )
    return JSONResponse(data)


@app.get("/api/circuit")
async def circuit_api(request: Request) -> JSONResponse:
    """Structured circuit-op result for the inline widget.

    Accepts all possible kwargs as query params; unused ones are
    ignored by `circuit_data`.
    """
    params = dict(request.query_params)
    op = params.pop("op", "")
    data = await run_in_threadpool(tools_mod.circuit_data, op, **params)
    return JSONResponse(data)


@app.get("/api/math")
async def math_api(
    op: str,
    expression: str,
    variable: str = "x",
    transform_var: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
) -> JSONResponse:
    """Structured math result for the KaTeX widget.

    Mirrors the `math` LLM tool but returns the full dict (input_latex,
    result_latex, plain) so the widget can render the equation properly.
    """
    data = await run_in_threadpool(
        tools_mod.math_data, op, expression, variable, transform_var, lower, upper,
    )
    return JSONResponse(data)


@app.get("/api/diagnostics/data-sources")
async def diagnostics_data_sources() -> JSONResponse:
    """Aggregate _last_fetch_status.json from every bridge/data/<tool>/
    subdir into a single status report. Used by the (future) Settings
    diagnostics panel — surfaces any source that's stale or broken so
    the user can see what to fix.

    Schema:
        {
          "sources": {
            "<tool>:<key>": {ok, ran_at, error?, ...},
            ...
          },
          "summary": {"total": N, "ok": O, "failed": F, "any_failed": bool}
        }
    """
    from pathlib import Path as _Path
    import json as _j
    base = _Path(__file__).resolve().parent.parent / "bridge" / "data"
    sources: dict = {}
    if base.is_dir():
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            status_file = sub / "_last_fetch_status.json"
            if not status_file.is_file():
                continue
            try:
                blob = _j.loads(status_file.read_text(encoding="utf-8"))
                if isinstance(blob, dict):
                    sources.update(blob)
            except Exception as e:
                sources[f"_read_error:{sub.name}"] = {
                    "ok": False, "error": f"could not read status file: {e}",
                }
    total = len(sources)
    failed = sum(1 for v in sources.values() if not v.get("ok"))
    return JSONResponse({
        "sources": sources,
        "summary": {
            "total": total,
            "ok": total - failed,
            "failed": failed,
            "any_failed": failed > 0,
        },
    })


@app.get("/api/population")
async def population_api(
    region: str = "world",
    year: int | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
    metric: str = "value",
    top: int = 10,
) -> JSONResponse:
    """Population calc + full year-by-year series for the widget chart.

    metric='rank' returns top-N most populous countries for ``year``
    (loaded from the all-countries cache) and skips the per-region
    series block."""
    if metric == "rank":
        from bridge.tools._population import _population
        import json as _j
        raw = await run_in_threadpool(
            _population, region, year, None, None, "rank", top,
        )
        return JSONResponse(_j.loads(raw))
    from bridge.tools._population import population_widget_data
    data = await run_in_threadpool(
        population_widget_data, region, year, from_year, to_year,
    )
    return JSONResponse(data)


@app.get("/api/facts")
async def facts_api(year: int, region: str | None = None) -> JSONResponse:
    """Year-card aggregate (population + inflation sections) for the widget."""
    from bridge.tools._facts import _facts
    raw = await run_in_threadpool(_facts, year, region)
    import json as _j
    return JSONResponse(_j.loads(raw))


@app.get("/api/map")
async def map_api(
    year: int | None = None,
    metric: str = "median_2br_rent",
) -> JSONResponse:
    """Choropleth-friendly state-level data for the /ui/map panel.
    Wraps `bridge.tools._map.map_state_rents`."""
    from bridge.tools._map import map_state_rents
    data = await run_in_threadpool(map_state_rents, year, metric)
    return JSONResponse(data)


@app.get("/api/county/metrics")
async def county_metrics_list_api() -> JSONResponse:
    """List of registered county-level metrics + which have data on
    disk. The panel uses this to populate the metric-selector dropdown
    + grey out the not-yet-fetched ones."""
    from bridge.tools._county_metrics import list_available_metrics
    data = await run_in_threadpool(list_available_metrics)
    return JSONResponse(data)


@app.get("/api/county")
async def county_metric_api(
    metric: str = "rent",
    year: int | None = None,
) -> JSONResponse:
    """Per-county values for one metric/year (county-overlay choropleth)."""
    from bridge.tools._county_metrics import county_metric
    data = await run_in_threadpool(county_metric, metric, year)
    return JSONResponse(data)


@app.get("/api/county/compare")
async def county_compare_api(
    a: str,
    b: str,
    year: int | None = None,
) -> JSONResponse:
    """Pairwise Pearson r between two metrics across all counties +
    scatter-friendly pairs."""
    from bridge.tools._county_metrics import county_compare
    data = await run_in_threadpool(county_compare, a, b, year)
    return JSONResponse(data)


@app.get("/api/county/drill/{fips}")
async def county_drill_api(fips: str) -> JSONResponse:
    """All metrics for a single county across all available years +
    national-distribution percentile per metric."""
    from bridge.tools._county_metrics import county_drill
    data = await run_in_threadpool(county_drill, fips)
    return JSONResponse(data)


@app.get("/api/map/regions")
async def map_regions_api() -> JSONResponse:
    """Curated economic-region overlays for the map panel."""
    from bridge.tools._map import map_regions
    data = await run_in_threadpool(map_regions)
    return JSONResponse(data)


@app.get("/api/map/state/{code}")
async def map_state_profile_api(code: str) -> JSONResponse:
    """Per-state profile (industries, resources, narrative, region
    memberships, latest rent) for the click-to-drill panel."""
    from bridge.tools._map import map_state_profile
    data = await run_in_threadpool(map_state_profile, code)
    return JSONResponse(data)


@app.get("/api/alice")
async def alice_api(
    year: int | None = None,
    composition: str | None = None,
    household_size: int | None = None,
) -> JSONResponse:
    """Structured ALICE estimator output for the standalone /ui/alice page.

    `composition` (1A0K / 2A0K / 1A1K / 2A1K / 2A2K / 2A3K) is preferred;
    `household_size` is a backward-compat alternative that maps to a
    composition via _SIZE_TO_COMPOSITION. When neither is provided, the
    tool defaults to the 4-person canonical (2A2K).

    Wraps `bridge.tools._alice._alice` directly so the panel sees the
    same payload an LLM tool call would. The tool already returns a
    fully-formed JSON; we just decode + re-emit it through FastAPI."""
    from bridge.tools._alice import _alice
    raw = await run_in_threadpool(
        _alice, year, composition, household_size,
    )
    import json as _j
    return JSONResponse(_j.loads(raw))


@app.get("/api/inflation")
async def inflation_api(
    amount: float = 1.0,
    from_year: int = 1970,
    to_year: int | None = None,
    measure: str = "cpi",
    item: str | None = None,
    region: str = "us",
    regions: str | None = None,
) -> JSONResponse:
    """Structured inflation/wages/item result + time series for the widget.

    region='X' for single-region (default 'us'). regions='us,japan,hk_sar'
    for cross-region comparison mode (returns comparison + per-region series).
    """
    from bridge.tools._inflation import inflation_widget_data
    data = await run_in_threadpool(
        inflation_widget_data, amount, from_year, to_year,
        measure, item, region, regions,
    )
    return JSONResponse(data)


@app.get("/api/find-places")
async def find_places(q: str | None = None, location: str | None = None) -> JSONResponse:
    """Structured place suggestions for the inline widget.

    Mirrors the ``find_places`` LLM tool via ``find_places_data`` so
    the widget sees the same location resolution and DDG hits the LLM
    used in its reply.
    """
    query = (q or "").strip()
    if not query:
        return JSONResponse({"error": "empty query", "results": []})
    data = await run_in_threadpool(tools_mod.find_places_data, query, location)
    return JSONResponse(data)


@app.get("/api/web-search")
async def web_search(q: str | None = None) -> JSONResponse:
    """Structured DDG search results for the inline widget.

    Wraps the same ``ddgs`` library the LLM tool uses so the widget
    list and the LLM's reply paraphrase the same data.
    """
    query = (q or "").strip()
    if not query:
        return JSONResponse({"error": "empty query", "results": []})
    try:
        from ddgs import DDGS
    except ImportError:
        return JSONResponse({"error": "ddgs not installed", "results": []})

    def _run() -> list[dict]:
        try:
            with DDGS() as d:
                return list(d.text(query, max_results=5))
        except Exception as e:
            log.error("web-search widget failed: %s", e)
            return []

    results = await run_in_threadpool(_run)
    # Project to a stable shape so the widget doesn't depend on ddgs internals.
    return JSONResponse({
        "query": query,
        "results": [
            {
                "title": (r.get("title") or "").strip(),
                "body": (r.get("body") or "").strip(),
                "href": (r.get("href") or "").strip(),
            }
            for r in results
        ],
    })


@app.get("/api/weather")
async def weather(location: str | None = None) -> JSONResponse:
    """Structured weather for the inline widget.

    Mirrors the ``get_weather`` tool's resolution logic but returns the
    raw dict instead of a formatted sentence. Callable directly from
    the browser so the widget stays independent of any in-flight LLM
    turn (e.g. the user can re-open the widget later and it refetches).
    """
    data = await run_in_threadpool(tools_mod.fetch_weather, location)
    return JSONResponse(data)


@app.get("/api/convert")
async def convert(value: str, from_unit: str, to_unit: str) -> JSONResponse:
    """Structured unit conversion for the inline widget.

    Wraps ``bridge.tools.convert_data`` so the browser widget can reuse
    the pint-backed conversion engine without going through the LLM.
    """
    data = await run_in_threadpool(tools_mod.convert_data, value, from_unit, to_unit)
    return JSONResponse(data)


@app.get("/api/health")
async def health() -> JSONResponse:
    """Lightweight readiness check the client pings on page load."""
    return JSONResponse({
        "ok": True,
        "stt_model": stt_cfg["model"] if STT_ENABLED else None,
        "llm_model": llm_cfg["model"],
        "text_lang": tts_cfg["text_lang"] if TTS_ENABLED else None,
    })


@app.get("/api/capabilities")
async def capabilities() -> JSONResponse:
    """Legacy endpoint kept for old clients. New clients should read
    /api/features which also exposes bandit, holidays, and tool
    denylist. See RUNBOOK § Feature flags."""
    return JSONResponse({"stt": STT_ENABLED, "tts": TTS_ENABLED})


@app.get("/api/features")
async def features_endpoint() -> JSONResponse:
    """Full feature registry snapshot — what's on, what's off, which
    tools are denylisted. Clients use this to decide UI rendering
    (hide mic when stt off, hide voice dot when tts off, etc.)."""
    snap = dict(_features.snapshot())
    # Runtime-toggleable flags from the LLM layer piggyback on the same
    # snapshot so the UI can read them in one call. Quality vs speed
    # (two-phase compose) shown in the settings panel.
    snap["two_phase_compose"] = bool(getattr(llm, "two_phase_compose", False))
    return JSONResponse(snap)


class _SettingsBody(BaseModel):
    """Runtime-mutable knobs the UI can flip without a restart."""
    # Two-phase compose: extra LLM call on every turn to re-write the
    # final reply with no tool schema in context. Big reply-quality win
    # at +5-10s latency per tool turn. Users can turn this off when
    # speed matters more than polish.
    two_phase_compose: bool | None = None


@app.post("/api/settings")
async def settings_endpoint(body: _SettingsBody) -> JSONResponse:
    """Patch runtime-mutable knobs. Only the fields present in the body
    are applied; others keep their current values. Returns the post-patch
    state for the fields the UI tracks."""
    applied: dict[str, object] = {}
    if body.two_phase_compose is not None:
        llm.set_two_phase_compose(body.two_phase_compose)
        applied["two_phase_compose"] = bool(llm.two_phase_compose)
        log.info("settings: two_phase_compose=%s", llm.two_phase_compose)
    return JSONResponse({"applied": applied})


@app.post("/api/turn")
async def turn(audio: UploadFile = File(...)) -> JSONResponse:
    """Run one push-to-talk turn through the pipeline.

    Flow:
        upload -> ffmpeg decode -> STT -> LLM -> TTS -> base64 WAV response

    Returns JSON with:
        - ``user``: the transcribed user text
        - ``assistant``: the LLM reply
        - ``audio_b64``: base64-encoded WAV of the spoken reply
        - ``audio_mime``: ``audio/wav``
    """
    if not STT_ENABLED:
        raise HTTPException(status_code=503, detail="STT disabled on this server")
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload too large ({len(audio_bytes)} bytes, cap is {MAX_UPLOAD_BYTES})",
        )

    log.info(
        "turn: received %d bytes (%s)",
        len(audio_bytes),
        audio.content_type or "unknown content-type",
    )

    # Decode on the threadpool so we don't block the event loop.
    try:
        pcm = await run_in_threadpool(decode_to_pcm16k, audio_bytes)
    except RuntimeError as e:
        log.error("decode failed: %s", e)
        raise HTTPException(status_code=415, detail=f"audio decode failed: {e}")

    if pcm.size == 0:
        raise HTTPException(status_code=400, detail="decoded audio is empty")
    duration_s = pcm.size / 16000
    log.info("turn: decoded to %d samples (%.2fs)", pcm.size, duration_s)

    async with pipeline_lock:
        # --- STT ---
        if not _stt_runtime_enabled or stt is None:
            raise HTTPException(
                status_code=503,
                detail="STT is disabled on this server (flip it on via the sidebar STT indicator or POST /api/stt/enable).",
            )
        try:
            user_text = await run_in_threadpool(stt.transcribe, pcm)
        except Exception as e:
            log.error("STT failed: %s", e)
            raise HTTPException(status_code=500, detail=f"STT failed: {e}")
        user_text = user_text.strip()
        if not user_text:
            raise HTTPException(status_code=422, detail="no speech detected")
        log.info("user: %s", user_text)

        # --- LLM ---
        try:
            reply = await run_in_threadpool(llm.chat, user_text)
        except Exception as e:
            log.error("LLM failed: %s", e)
            raise HTTPException(status_code=502, detail=f"LLM failed: {e}")
        reply = reply.strip()
        if not reply:
            raise HTTPException(status_code=502, detail="empty reply from LLM")
        log.info("assistant: %s", reply)

        # Persist + maybe compact — match the streaming endpoints so
        # turns made via /api/turn also survive a page reload.
        await run_in_threadpool(
            _persist_and_maybe_compact, user_text=user_text, reply=reply,
        )

        # --- TTS ---
        if not TTS_ENABLED:
            reply_pcm, sr = np.zeros(0, dtype=np.int16), 16000
        else:
            try:
                reply_pcm, sr = await run_in_threadpool(tts.synthesize, reply)
            except TTSError as e:
                log.error("TTS rejected: %s", e)
                raise HTTPException(status_code=502, detail=f"TTS failed: {e}")
            except Exception as e:
                log.error("TTS unexpected error: %s", e)
                raise HTTPException(status_code=502, detail=f"TTS failed: {e}")

    wav_bytes = wrap_wav(reply_pcm, sr) if reply_pcm.size else b""
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    log.info(
        "turn: ok (%.2fs audio out, %d bytes WAV, %d bytes base64)",
        len(reply_pcm) / sr,
        len(wav_bytes),
        len(audio_b64),
    )

    return JSONResponse({
        "user": user_text,
        "assistant": reply,
        "audio_b64": audio_b64,
        "audio_mime": "audio/wav",
    })


# ---- streaming turns (NDJSON: transcript line, then audio chunks) ----------

def _make_stream_generator(
    user_text: str,
    reply: str,
    tool_calls: list | None = None,
    conversation_id: str | None = None,
    audio_id: str | None = None,
):
    """Build an async generator that yields NDJSON lines.

    Order of events:
      1. tool_call (0..N, in the order the LLM invoked them)
      2. transcript (always one)
      3. audio (0..N base64 WAV chunks from GPT-SoVITS streaming)
      4. error (optional, replaces further audio)
      5. done (always last — carries audio_id if a WAV was cached)

    The tool_call events come first so the browser can mount panels as
    soon as the stream starts, without waiting for audio synthesis.

    If ``conversation_id`` + ``audio_id`` are provided AND synthesis
    actually produces chunks, the assembled WAV is persisted to
    ``data/audio_cache/<cid>/<audio_id>.wav`` so the browser can
    replay it after a refresh via ``/api/audio/<cid>/<audio_id>``.
    """
    q: thread_queue.Queue = thread_queue.Queue()
    # Server-side buffer of raw int16 PCM — separate from the base64
    # stream so we can write the canonical WAV after the TTS thread
    # finishes without paying base64 decode cost.
    pcm_buffer: list = []
    sample_rate_seen: list[int] = []

    def _run_tts():
        try:
            for pcm_chunk, sr in tts.synthesize_stream(reply):
                # Keep both: one path streams to the browser, one is
                # saved for cross-session replay.
                pcm_buffer.append(pcm_chunk)
                if not sample_rate_seen:
                    sample_rate_seen.append(sr)
                b64 = base64.b64encode(pcm_chunk.tobytes()).decode("ascii")
                q.put(("audio", b64, sr))
        except Exception as e:
            log.error("TTS stream error: %s", e)
            q.put(("error", str(e), 0))
        finally:
            q.put(("done", "", 0))

    async def generate():
        # 0. Tool calls observed during the LLM turn. Emitted BEFORE the
        # transcript so the browser can start mounting the right panel
        # immediately, in parallel with TTS audio synthesis.
        for tc in (tool_calls or []):
            yield json_mod.dumps({
                "type": "tool_call",
                "name": tc.get("name", ""),
                "arguments": tc.get("arguments", {}),
                "result": tc.get("result", ""),
            }) + "\n"

        # 1. Transcript (arrives before any audio)
        yield json_mod.dumps({
            "type": "transcript", "user": user_text, "assistant": reply,
        }) + "\n"

        # 2. Stream TTS audio from a background thread — skipped entirely
        # when TTS is disabled. No audio events, stream closes with done.
        if not TTS_ENABLED:
            yield json_mod.dumps({"type": "done"}) + "\n"
            return
        tts_thread = threading.Thread(target=_run_tts, daemon=True)
        tts_thread.start()

        while True:
            # asyncio.to_thread runs the blocking q.get in a thread-pool
            # worker, keeping the event loop responsive.
            try:
                kind, payload, sr = await asyncio.to_thread(q.get, timeout=120)
            except Exception:
                break
            if kind == "done":
                break
            if kind == "error":
                yield json_mod.dumps({"type": "error", "detail": payload}) + "\n"
                break
            yield json_mod.dumps({"type": "audio", "b64": payload, "sr": sr}) + "\n"

        tts_thread.join(timeout=10)

        # 3. Persist the assembled WAV so the browser can replay after a
        # refresh. Best-effort — a failure here never blocks the 'done'
        # event or the chat flow.
        saved_audio_id: str | None = None
        if (conversation_id and audio_id and pcm_buffer and sample_rate_seen):
            try:
                from bridge.audio_cache import save_turn_audio
                n = await asyncio.to_thread(
                    save_turn_audio, conversation_id, audio_id,
                    pcm_buffer, sample_rate_seen[0],
                )
                if n > 0:
                    saved_audio_id = audio_id
            except Exception as e:
                log.debug("audio cache save failed (non-fatal): %s", e)

        # 4. Done — carries the audio_id iff a WAV is actually cached.
        done_event: dict = {"type": "done"}
        if saved_audio_id:
            done_event["audio_id"] = saved_audio_id
        yield json_mod.dumps(done_event) + "\n"

    return generate()


@app.post("/api/turn-stream")
async def turn_stream(audio: UploadFile = File(...)) -> StreamingResponse:
    """Streaming voice turn: STT + LLM, then NDJSON stream of TTS chunks."""
    if not STT_ENABLED:
        raise HTTPException(status_code=503, detail="STT disabled on this server")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty upload")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "upload too large")

    # Decode to PCM. A too-short / malformed click-and-release upload often
    # makes ffmpeg exit non-zero; rather than bubbling a 500 we treat it as
    # silence and let the LLM reply "sorry, didn't catch that."
    try:
        pcm = await run_in_threadpool(decode_to_pcm16k, audio_bytes)
    except Exception as e:
        log.info("decode failed, routing as silence: %s", e)
        pcm = None

    # Minimum usable length is ~0.25 s of 16 kHz PCM. Below that Whisper
    # reliably hallucinates anyway; fast-path to the sentinel.
    _MIN_PCM_SAMPLES = 4000
    too_short = pcm is None or pcm.size < _MIN_PCM_SAMPLES

    async with pipeline_lock:
        if too_short:
            user_text = ""
        elif not _stt_runtime_enabled or stt is None:
            log.info("STT runtime-disabled — routing as silence")
            user_text = ""
        else:
            try:
                user_text = await run_in_threadpool(stt.transcribe, pcm)
            except Exception as e:
                log.info("STT failed, routing as silence: %s", e)
                user_text = ""
        user_text = user_text.strip()

        # STT miss / silence hallucination: route through the LLM with a
        # sentinel so it produces a natural "could you repeat that?" reply
        # (which still runs through TTS). The user bubble shows a muted
        # "(couldn't hear you clearly)" placeholder instead of the raw
        # hallucinated word.
        is_silence = _is_likely_silence(user_text)
        tool_calls: list[dict] = []
        if is_silence:
            # Skip the LLM entirely — instant canned reply, TTS only.
            # The user bubble stays empty (client-side hides it).
            log.info("STT miss (raw=%r) — canned silence reply", user_text)
            reply = random.choice(_SILENCE_REPLIES)
            display_user = ""
        else:
            log.info("user: %s", user_text)
            display_user = user_text
            try:
                reply = await _chat_with_retry(
                    user_text, tool_calls_sink=tool_calls,
                )
            except HTTPException:
                raise
            except Exception as e:
                log.exception("LLM/tool call failed")
                raise HTTPException(502, f"LLM failed: {e}")
            log.info("assistant: %s  (tools=%d)", reply, len(tool_calls))
            _tc_names = [c.get("name", "") for c in tool_calls]
            # Allocate audio_id up-front so the turn note and the WAV
            # filename agree, even if synthesis fails mid-stream.
            audio_id = None
            if TTS_ENABLED:
                from bridge.audio_cache import new_turn_id
                audio_id = new_turn_id()
            await run_in_threadpool(
                _persist_and_maybe_compact,
                user_text=user_text, tools_called=_tc_names, reply=reply,
                audio_id=audio_id,
            )

    log.info("streaming TTS for: %s", reply[:80])
    # audio_id may be None if we hit the canned-silence fast path —
    # that branch doesn't define `audio_id`. Default it so the
    # generator's "persist WAV" block is a no-op.
    _audio_id = locals().get("audio_id")
    return StreamingResponse(
        _make_stream_generator(
            display_user, reply, tool_calls=tool_calls,
            conversation_id=_current_cid, audio_id=_audio_id,
        ),
        media_type="application/x-ndjson",
    )


@app.post("/api/text-turn-stream")
async def text_turn_stream(request: Request) -> StreamingResponse:
    """Streaming text turn: LLM, then NDJSON stream of TTS chunks."""
    body = await request.json()
    user_text = (body.get("text") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="empty text")
    log.info("text-turn-stream: %s", user_text)

    async with pipeline_lock:
        tool_calls: list[dict] = []
        try:
            reply = await _chat_with_retry(user_text, tool_calls_sink=tool_calls)
        except HTTPException:
            raise
        except Exception as e:
            log.exception("LLM/tool call failed")
            raise HTTPException(502, f"LLM failed: {e}")
        log.info("assistant: %s  (tools=%d)", reply, len(tool_calls))

        # Allocate a turn-scoped id up front (only when TTS is on, since
        # the audio cache is useless without synthesis). The id flows
        # into the turn note NOW and into the WAV filename AT DONE; if
        # synthesis fails, the turn note still has the id but the
        # endpoint 404s gracefully.
        audio_id = None
        if TTS_ENABLED:
            from bridge.audio_cache import new_turn_id
            audio_id = new_turn_id()

        _tc_names = [c.get("name", "") for c in tool_calls]
        await run_in_threadpool(
            _persist_and_maybe_compact,
            user_text=user_text, tools_called=_tc_names, reply=reply,
            audio_id=audio_id,
        )

    return StreamingResponse(
        _make_stream_generator(
            user_text, reply, tool_calls=tool_calls,
            conversation_id=_current_cid, audio_id=audio_id,
        ),
        media_type="application/x-ndjson",
    )


# ---- background-job turn endpoints ----------------------------------------
#
# Browser-disconnect-safe variant of /api/text-turn-stream and
# /api/turn-stream. Flow:
#   POST /api/turn/start     -> {"job_id": "..."}  (returns immediately)
#   GET  /api/turn/<id>/stream  ->  NDJSON stream of events (live + replay)
#
# The pipeline runs on a worker thread regardless of whether anyone is
# listening, so a 60-second turn won't get cancelled by phone-sleep
# or tab-close. The browser can reconnect to /stream and receive
# whatever events were emitted while it was away.


def _run_text_turn_job(
    job,
    user_text: str,
    *,
    do_tts: bool = True,
    extra_system_suffix: str = "",
    excluded_tool_names: list[str] | None = None,
) -> None:
    """Worker-thread runner: full text turn, persisted as job events.

    Same event order as the streaming endpoints (tool_call×N, transcript,
    audio×N, error?, done) so the client renderer is identical."""
    import base64
    # Pre-LLM reminder detection. Pure regex + dateparser; runs in
    # microseconds and never raises (the helper catches its own
    # errors). When a hit fires, the reminder is persisted via the
    # public api.create_reminder which logs the id; the LLM still
    # processes the original user_text so Karin can acknowledge in
    # her own voice. The poller fires the push at trigger_at.
    try:
        from bridge.reminders.detect import detect_reminder
        from bridge.reminders.api import create_reminder as _create_reminder
        _detected = detect_reminder(user_text)
        if _detected is not None:
            _rem = _create_reminder(
                trigger_at=_detected.trigger_at,
                message=_detected.message,
                source="chat_detect",
            )
            log.info(
                "chat-detected reminder: id=%s phrase=%s trigger=%s",
                _rem.id, _detected.matched_phrase,
                _detected.trigger_at.isoformat(),
            )
            # Surface the detection so the UI can show a small "set"
            # confirmation card later (Phase 2 — for now this just
            # lands in the job event stream as informational).
            job.append_event({
                "type": "reminder_set",
                "id": _rem.id,
                "trigger_at": _detected.trigger_at.isoformat(),
                "message": _detected.message,
                "matched_phrase": _detected.matched_phrase,
            })
    except Exception as e:
        # Detection must NEVER block the chat turn. Log + move on.
        log.warning("reminder detection failed (continuing without): %s", e)

    # Acquire the pipeline lock synchronously since we're on a worker
    # thread, not the asyncio loop. asyncio.Lock can't be used from a
    # plain thread; using a threading.Lock around the same scope.
    with _job_pipeline_lock:
        try:
            tool_calls: list[dict] = []
            def _on_tool(name: str, args: dict, result: str) -> None:
                preview = (result or "").strip()
                if len(preview) > 600:
                    preview = preview[:600].rstrip() + "\u2026"
                tool_calls.append({"name": name, "arguments": args, "result": preview})
                # Emit tool_call events as they happen so the widget mounts
                # while the LLM is still composing its final reply.
                job.append_event({
                    "type": "tool_call",
                    "name": name,
                    "arguments": args,
                    "result": preview,
                })

            # Emit a transcript-start so the client can flip the user
            # bubble from "listening…" / "…" to the actual text BEFORE
            # the assistant tokens start flowing.
            job.append_event({"type": "user_text", "text": user_text})

            def _on_token(delta: str) -> None:
                job.append_event({"type": "token_delta", "delta": delta})

            # On a retry turn, drop the tools the prior attempt used so
            # the model LITERALLY CAN'T pick them again. Prompt-level
            # "don't repeat" hints weren't enough on a 3B — this makes
            # it a hard constraint at the schema level.
            if excluded_tool_names:
                excluded = set(excluded_tool_names)
                turn_schemas = [
                    s for s in tools_mod.active_tool_schemas()
                    if (s.get("function") or {}).get("name") not in excluded
                ]
                log.info("job %s: retry — excluding %s from schemas",
                         job.id, sorted(excluded))
            else:
                turn_schemas = tools_mod.active_tool_schemas()

            try:
                reply = llm.chat_stream(
                    user_text, turn_schemas, _on_tool, _on_token,
                    extra_system_suffix=extra_system_suffix,
                )
            except Exception as e:
                log.exception("job %s: LLM failed", job.id)
                job.append_event({"type": "error", "detail": f"LLM failed: {e}"})
                job.append_event({"type": "done"})
                job.mark_status("failed", error=str(e))
                return
            reply = (reply or "").strip()
            if not reply:
                job.append_event({"type": "error", "detail": "empty reply from LLM"})
                job.append_event({"type": "done"})
                job.mark_status("failed", error="empty reply")
                return
            log.info("job %s: assistant: %s (tools=%d)", job.id, reply, len(tool_calls))
            _tc_names = [c.get("name", "") for c in tool_calls]
            _persist_and_maybe_compact(
                user_text=user_text, tools_called=_tc_names, reply=reply,
            )

            # Record this turn in the preference store (unrated). The
            # UI's thumbs-up/down buttons patch the rating later via
            # /api/feedback. Embedding happens inside append() and is
            # fail-soft — if Ollama's embedding model isn't pulled yet,
            # the entry is still logged with embedding=None.
            if _BANDIT_ENABLED:
                try:
                    feedback_store.append(
                        turn_id=job.id,
                        prompt=user_text,
                        tool_chain=tool_calls,
                        reply=reply,
                        conversation_id=_current_cid,
                    )
                except Exception as e:
                    log.warning("job %s: feedback append failed: %s", job.id, e)

            # Final transcript event (full text). Clients that missed
            # token_deltas (reconnect after streaming) use this to render
            # the complete reply in one shot.
            job.append_event({
                "type": "transcript", "user": user_text, "assistant": reply,
            })

            if do_tts and TTS_ENABLED:
                try:
                    pcm, sr = tts.synthesize(reply)
                    b64 = base64.b64encode(pcm.tobytes()).decode("ascii")
                    job.append_event({"type": "audio", "b64": b64, "sr": sr})
                except Exception as e:
                    log.warning("job %s: TTS failed: %s", job.id, e)
                    job.append_event({"type": "error", "detail": str(e)})

            job.append_event({"type": "done"})
            job.mark_status("done")
        finally:
            job_store.persist(job)


# _job_pipeline_lock + _pipeline_and_job_lock are defined alongside
# pipeline_lock at module load time so the worker-thread code path and
# the async routes that need to coordinate with it both reference the
# same object.


class _TurnStartBody(BaseModel):
    text: str | None = None
    # Optional: turn_id of a previous, thumbs-down-rated turn. When set,
    # we look up that turn's tool_chain and inject a retry hint so the
    # LLM knows to take a different approach this time.
    retry_of: str | None = None


@app.post("/api/turn/start")
async def turn_start(body: _TurnStartBody) -> JSONResponse:
    """Enqueue a text turn for background processing. Returns job_id
    immediately so the browser can disconnect / reconnect freely."""
    user_text = (body.text or "").strip()
    if not user_text:
        raise HTTPException(400, "empty text")

    retry_suffix = ""
    excluded_tools: list[str] = []
    if body.retry_of and _BANDIT_ENABLED:
        prev = feedback_store.get(body.retry_of)
        if prev is not None:
            retry_suffix = retry_hint(prev.tool_chain)
            # Extract unique tool names from the prior chain; these get
            # stripped from the schema list so the retry can't repeat
            # the same (failed) routing. Deduplicated since a chain can
            # call the same tool twice — we want the SET not the sequence.
            seen: set[str] = set()
            for tc in prev.tool_chain:
                name = (tc.get("name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    excluded_tools.append(name)

    job = job_store.create()
    log.info("job %s: enqueued (text=%r%s)", job.id, user_text[:60],
             " [retry]" if retry_suffix else "")
    turn_worker.submit(
        job,
        lambda j: _run_text_turn_job(
            j, user_text,
            extra_system_suffix=retry_suffix,
            excluded_tool_names=excluded_tools or None,
        ),
    )
    return JSONResponse({"job_id": job.id})


class _FeedbackBody(BaseModel):
    turn_id: str
    rating: int   # +1 or -1


@app.post("/api/feedback")
async def feedback(body: _FeedbackBody) -> JSONResponse:
    """Patch a past turn's rating (thumbs-up/down from the UI).

    Safe to call even when bandit is disabled — the store still records
    ratings, they just don't influence routing until KARIN_BANDIT is on.
    """
    if body.rating not in (-1, 1):
        raise HTTPException(400, "rating must be +1 or -1")
    ok = feedback_store.update_rating(body.turn_id, body.rating)
    if not ok:
        raise HTTPException(404, f"no feedback entry for turn_id={body.turn_id}")
    log.info("feedback: turn=%s rating=%+d", body.turn_id, body.rating)
    return JSONResponse({"ok": True})


@app.get("/api/feedback/stats")
async def feedback_stats() -> JSONResponse:
    """Debug endpoint: per-tool rating aggregates + total entries."""
    entries = feedback_store.all_entries()
    rated = [e for e in entries if e.rating is not None]
    return JSONResponse({
        "total_entries": len(entries),
        "rated_entries": len(rated),
        "positive": sum(1 for e in rated if e.rating > 0),
        "negative": sum(1 for e in rated if e.rating < 0),
        "per_tool": feedback_store.tool_stats(),
    })


@app.post("/api/feedback/reset")
async def feedback_reset() -> JSONResponse:
    """Wipe the feedback log. Parallels /api/history/new for routing signal."""
    feedback_store.reset()
    log.info("feedback log reset")
    return JSONResponse({"ok": True})


# ---- voice (sovits) lifecycle --------------------------------------------
#
# Lets the browser toggle the sovits container without SSH. On Orin Nano
# 8 GB the LLM and sovits can't both stay loaded reliably, so the
# expected workflow is: leave sovits stopped for fast text turns, start
# it on demand when voice is wanted, stop again when done. See
# bridge/voice.py for the docker-socket plumbing.

@app.get("/api/voice/status")
async def voice_status_route() -> JSONResponse:
    if not TTS_ENABLED:
        return JSONResponse({"status": "disabled", "reason": "KARIN_TTS_ENABLED=false"})
    # If KARIN_TTS_BASE_URL points at a remote sidecar, skip the local
    # Docker inspection (there's no container to inspect) and probe the
    # remote host instead. Prefer /health on current sidecars; fall back
    # to /stt/status so older PC installs still report state correctly.
    remote_url = os.environ.get("KARIN_TTS_BASE_URL", "").strip()
    if remote_url and "127.0.0.1" not in remote_url and "localhost" not in remote_url:
        try:
            probe = _get_probe_client()
            base = remote_url.rstrip("/")
            r = await probe.get(f"{base}/health")
            if r.status_code == 404:
                r = await probe.get(f"{base}/stt/status")
            return JSONResponse({
                "status": "remote",
                "backend": "remote",
                "remote_url": remote_url,
                "remote_reachable": (r.status_code < 500),
                "running": (r.status_code < 500),
            })
        except Exception as e:
            return JSONResponse({
                "status": "remote",
                "backend": "remote",
                "remote_url": remote_url,
                "remote_reachable": False,
                "running": False,
                "error": str(e)[:120],
            })
    from bridge import voice as voice_mod
    return JSONResponse(await run_in_threadpool(voice_mod.status))


@app.post("/api/voice/start")
async def voice_start_route() -> JSONResponse:
    if not TTS_ENABLED:
        raise HTTPException(status_code=503, detail="TTS disabled on this server")
    from bridge import voice as voice_mod
    return JSONResponse(await run_in_threadpool(voice_mod.start))


@app.post("/api/voice/stop")
async def voice_stop_route() -> JSONResponse:
    if not TTS_ENABLED:
        raise HTTPException(status_code=503, detail="TTS disabled on this server")
    from bridge import voice as voice_mod
    return JSONResponse(await run_in_threadpool(voice_mod.stop))


# ---- STT runtime enable/disable ------------------------------------------
#
# Unlike TTS (which toggles a Docker container via /api/voice/*), STT runs
# in-process with faster-whisper. Model load is expensive (~3-5 s on the
# Jetson) so we don't unload on "disable" — the runtime flag just gates
# whether stt.transcribe is called on incoming audio. Turning STT off
# therefore skips transcription but keeps the model warm for fast re-enable.
#
# STT_ENABLED (the boot-time flag from features.yaml / KARIN_STT_ENABLED)
# is a hard ceiling: if it's False at boot, the Whisper model wasn't
# loaded and enable() returns 503. Flip the env var + restart the
# service to change that.

@app.get("/api/stt/status")
async def stt_status_route() -> JSONResponse:
    cfg = (load_config(CONFIG_PATH).get("stt") or {})
    remote_url = os.environ.get("KARIN_STT_BASE_URL", "").strip() or None
    backend = "remote" if remote_url else "local"

    # Live reachability probe for the remote path so the frontend can
    # show an "unreachable" indicator without waiting for a real
    # transcribe call to fail. `None` = not applicable (local backend);
    # `True`/`False` = probed result. Short 2s timeout to keep the /status
    # hit cheap — the polling loop runs every 20s in the browser.
    remote_reachable: bool | None = None
    if remote_url:
        try:
            probe = _get_probe_client()
            r = await probe.get(f"{remote_url.rstrip('/')}/stt/status")
            remote_reachable = (r.status_code == 200)
        except Exception:
            remote_reachable = False

    return JSONResponse({
        "boot_enabled": STT_ENABLED,
        "runtime_enabled": _stt_runtime_enabled,
        "backend": backend,
        "remote_url": remote_url,
        "remote_reachable": remote_reachable,
        "model": cfg.get("model") if STT_ENABLED else None,
        "device": cfg.get("device") if STT_ENABLED else None,
        "compute_type": cfg.get("compute_type") if STT_ENABLED else None,
        "reason": None if STT_ENABLED else "KARIN_STT_ENABLED=false at boot",
    })


@app.post("/api/stt/enable")
async def stt_enable_route() -> JSONResponse:
    global _stt_runtime_enabled
    if not STT_ENABLED or stt is None:
        raise HTTPException(
            status_code=503,
            detail="STT was disabled at boot (KARIN_STT_ENABLED=false). "
            "Set KARIN_STT_ENABLED=true in deploy/.env and restart the service "
            "to load the Whisper model, then try again.",
        )
    _stt_runtime_enabled = True
    log.info("STT runtime flag: enabled")
    return JSONResponse({"runtime_enabled": True})


@app.post("/api/stt/disable")
async def stt_disable_route() -> JSONResponse:
    global _stt_runtime_enabled
    _stt_runtime_enabled = False
    log.info("STT runtime flag: disabled")
    return JSONResponse({"runtime_enabled": False})


# ---- Holiday banner + voiced greeting -------------------------------------

@app.get("/api/holiday/today")
async def holiday_today_route() -> JSONResponse:
    from bridge import holidays as holidays_mod
    rec = await run_in_threadpool(holidays_mod.today_holiday)
    return JSONResponse({"holiday": rec})


@app.get("/api/holiday/today/audio", response_model=None)
async def holiday_audio_route():
    """Karin-voiced greeting for today's holiday, lazily synthesized.

    Synthesis needs sovits running — returns 503 when it isn't so the
    UI can show a "voice unavailable" hint instead of a silent failure.
    Also runs stale-audio cleanup on every call, so yesterday's cached
    greeting can't linger.
    """
    if not TTS_ENABLED:
        raise HTTPException(status_code=503, detail="TTS disabled on this server")
    from bridge import holidays as holidays_mod

    def _compute() -> "Path | None":
        return holidays_mod.get_or_synth_greeting_audio(tts)

    path = await run_in_threadpool(_compute)
    if path is None:
        # Distinguish "no holiday today" (204) from "sovits down" (503)
        # by checking what today_holiday would have said.
        rec = await run_in_threadpool(holidays_mod.today_holiday)
        if rec is None:
            return JSONResponse({"error": "no holiday today"}, status_code=204)
        return JSONResponse(
            {"error": "voice backend unavailable — start sovits via the toggle"},
            status_code=503,
        )
    return StreamingResponse(
        iter([path.read_bytes()]),
        media_type="audio/wav",
    )


@app.get("/api/turn/{job_id}/stream")
async def turn_stream_job(job_id: str) -> StreamingResponse:
    """Stream a job's events (replay history + live tail). Safe to
    reconnect if the browser dropped — events seen so far are persisted."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")

    async def generate():
        cursor = 0
        while True:
            with job._lock:
                events = list(job.events[cursor:])
                cursor = len(job.events)
                done = job.status in ("done", "failed", "cancelled")
            for evt in events:
                yield json_mod.dumps(evt) + "\n"
            if done:
                break
            # Wait for new events (or job completion).
            await asyncio.to_thread(job._new_event.wait, 30.0)
            with job._lock:
                job._new_event.clear()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/api/turn/{job_id}/status")
async def turn_status_job(job_id: str) -> JSONResponse:
    """Lightweight job status check (no streaming)."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    with job._lock:
        return JSONResponse({
            "job_id": job.id,
            "status": job.status,
            "event_count": len(job.events),
            "error": job.error,
            "finished_at": job.finished_at,
        })


# ---- conversation management ----------------------------------------------


@app.get("/api/history")
async def get_history() -> JSONResponse:
    """Return the current conversation so the client can restore on load.

    Only user/assistant turns are shown — system prompt stays server-side,
    tool-call internals stay server-side. The compaction summary (if any)
    surfaces as a synthetic assistant note so the user sees there's
    earlier context even after compaction.
    """
    turns: list[dict] = []
    i = 0
    # Snapshot up front so the index walk below is stable even if
    # another thread appends to the live history mid-iteration.
    msgs = llm._history.snapshot()
    while i < len(msgs):
        m = msgs[i]
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system" and isinstance(content, str) and content.startswith("[Prior conversation summary]"):
            turns.append({"role": "summary", "content": content})
            i += 1
            continue
        if role == "user" and isinstance(content, str):
            # Walk forward to the next final assistant message, collecting
            # every tool_call / role=tool pair along the way so the browser
            # can replay the corresponding widgets on reload.
            assistant = ""
            tool_events: list[dict] = []
            j = i + 1
            while j < len(msgs):
                nxt = msgs[j]
                nxt_role = nxt.get("role")
                if nxt_role == "assistant":
                    # Intermediate assistant carrier: no content, just tool_calls.
                    # Seed pending tool_events from its tool_calls list; the
                    # role=tool responses that follow will fill in the result.
                    ncontent = nxt.get("content") or ""
                    if not ncontent.strip():
                        for tc in (nxt.get("tool_calls") or []):
                            fn = tc.get("function") or {}
                            tool_events.append({
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments") or {},
                                "result": None,
                            })
                        j += 1
                        continue
                    # Final assistant with user-visible content — this turn's reply.
                    assistant = ncontent
                    break
                if nxt_role == "tool":
                    tname = nxt.get("name", "")
                    tresult = nxt.get("content") or ""
                    # Fill the most recent pending event matching this tool name.
                    for te in reversed(tool_events):
                        if te["name"] == tname and te["result"] is None:
                            te["result"] = tresult
                            break
                    else:
                        # No pending carrier — still surface the result so
                        # UI can render something.
                        tool_events.append({
                            "name": tname, "arguments": {}, "result": tresult,
                        })
                    j += 1
                    continue
                j += 1
            turns.append({
                "role": "turn",
                "user": content,
                "assistant": assistant,
                "tool_events": [te for te in tool_events if te.get("result") is not None],
            })
            i = j + 1 if assistant else i + 1
        else:
            i += 1

    # Attach audio_id to surviving user turns so the browser can
    # rehydrate replay buttons on restore. Turn notes are preserved
    # across history compaction (llm._history gets truncated, notes
    # don't) — so len(notes) >= count(user_turns_in_view). We take
    # the LAST N notes to match the surviving tail.
    try:
        all_notes = history_store.get_turn_notes(_current_cid) if _current_cid else []
    except Exception as e:
        log.debug("turn notes load failed (non-fatal): %s", e)
        all_notes = []
    user_turn_positions = [idx for idx, t in enumerate(turns) if t.get("role") == "turn"]
    if all_notes and user_turn_positions:
        from bridge.audio_cache import resolve_turn_audio
        relevant_notes = all_notes[-len(user_turn_positions):]
        for pos, note in zip(user_turn_positions, relevant_notes):
            aid = note.get("audio_id")
            if not aid:
                continue
            # Only advertise the audio_id when the WAV actually exists.
            # The audio_cache lives on tmpfs (cleared on container restart),
            # so most rehydrated turns won't have a backing file. Probing
            # here keeps the client from issuing /api/audio/... fetches
            # that 404 — those used to clutter the dev console with red
            # entries on every history restore.
            if resolve_turn_audio(_current_cid, aid) is not None:
                turns[pos]["audio_id"] = aid

    return JSONResponse({
        "conversation_id": _current_cid,
        "turns": turns,
    })


class _NewChatBody(BaseModel):
    pass


@app.post("/api/history/new")
async def new_chat() -> JSONResponse:
    """Archive the current conversation and start a fresh one.

    If the current conversation is already empty (no turns yet), reuse
    it instead of spawning another — otherwise repeated taps on the
    'new chat' button pile up unlimited empty conversations in the
    sidebar. ``reused`` in the response distinguishes the two cases
    so the client can avoid an unnecessary transcript clear.
    """
    global _current_cid
    # Use the combined pipeline+job lock so a streaming worker turn in
    # flight finishes (and persists to its captured _current_cid)
    # before we swap to a fresh conversation.
    async with _pipeline_and_job_lock():
        # If nothing's been said in the current conversation, there's
        # nothing to archive AND nothing to meaningfully create — just
        # hand the same id back.
        outgoing = llm._history.snapshot()
        if not outgoing and _current_cid is not None:
            log.info("new chat: reusing empty %s", _current_cid)
            return JSONResponse({"conversation_id": _current_cid, "reused": True})
        # Flush whatever's in memory before cycling.
        if outgoing:
            await run_in_threadpool(history_store.save, _current_cid, outgoing)
        new_cid, _ = await run_in_threadpool(history_store.new_conversation)
        _current_cid = new_cid
        llm.set_history([])
    log.info("new chat: %s", _current_cid)
    return JSONResponse({"conversation_id": _current_cid, "reused": False})


@app.delete("/api/history/{cid}")
async def delete_history(cid: str) -> JSONResponse:
    """Delete a conversation by id.

    If the deleted conversation was the current one, the store picks the
    next most recent (or creates a fresh one if none remain). The LLM's
    in-memory history is re-synced to whatever ends up current so the
    next turn starts from the right context.
    """
    global _current_cid
    # Combined pipeline+job lock: any in-flight worker turn finishes
    # and persists to its captured _current_cid BEFORE this DELETE
    # potentially cycles current.
    async with _pipeline_and_job_lock():
        # Flush any in-memory edits before we potentially cycle current.
        outgoing = llm._history.snapshot()
        if outgoing and _current_cid != cid:
            await run_in_threadpool(history_store.save, _current_cid, outgoing)

        was_current = _current_cid == cid
        new_current = await run_in_threadpool(history_store.delete_conversation, cid)
        if new_current is None:
            raise HTTPException(404, f"conversation not found: {cid}")

        _current_cid = new_current
        if was_current:
            loaded = await run_in_threadpool(history_store.load_conversation, new_current)
            llm.set_history(loaded or [])
    log.info("deleted conversation %s (current now %s)", cid, _current_cid)
    return JSONResponse({"current": _current_cid, "was_current": was_current})


@app.get("/api/history/list")
async def list_history() -> JSONResponse:
    """Return the conversation index + the current conversation id."""
    convs = await run_in_threadpool(history_store.list_conversations)
    return JSONResponse({
        "current": _current_cid,
        "conversations": convs,
    })


@app.get("/api/history/{cid}/notes")
async def get_turn_notes(cid: str) -> JSONResponse:
    """Return the turn-level flow notes for a conversation.

    Each note records: timestamp, user prompt preview, tools called,
    routing classifier hint, and reply preview. Used for debugging
    tool-routing accuracy and conversation flow analysis.
    """
    notes = await run_in_threadpool(history_store.get_turn_notes, cid)
    return JSONResponse({"conversation_id": cid, "notes": notes})


class _SwitchBody(BaseModel):
    conversation_id: str


@app.post("/api/history/switch")
async def switch_chat(body: _SwitchBody) -> JSONResponse:
    """Switch the LLM's in-memory history to a different past conversation."""
    global _current_cid
    # Combined pipeline+job lock so a worker turn that's still
    # composing a reply persists to its captured _current_cid before
    # the switch swaps in a different conversation's history.
    async with _pipeline_and_job_lock():
        # Save the outgoing conversation before cycling so the kept
        # compaction/summary lands on disk for next time.
        outgoing = llm._history.snapshot()
        if outgoing:
            await run_in_threadpool(history_store.save, _current_cid, outgoing)

        target = body.conversation_id
        loaded = await run_in_threadpool(history_store.load_conversation, target)
        if loaded is None:
            raise HTTPException(404, f"conversation not found: {target}")
        ok = await run_in_threadpool(history_store.switch_to, target)
        if not ok:
            raise HTTPException(404, f"conversation not found: {target}")
        _current_cid = target
        llm.set_history(loaded)
    log.info("switched to conversation %s (%d messages)", _current_cid, len(loaded))
    return JSONResponse({"conversation_id": _current_cid})


# ---- memory (user + agent long-term context) -------------------------------


@app.get("/api/memory")
async def get_memory() -> JSONResponse:
    """Return current user + agent memory plus the per-file char cap."""
    return JSONResponse({
        "user": memory_store.get_user(),
        "agent": memory_store.get_agent(),
        "max_chars": MAX_MEMORY_CHARS,
    })


class _MemoryBody(BaseModel):
    user: str | None = None
    agent: str | None = None


@app.put("/api/memory")
async def put_memory(body: _MemoryBody) -> JSONResponse:
    """Overwrite one or both memory files. Truncates at MAX_MEMORY_CHARS."""
    truncated: dict[str, bool] = {}
    if body.user is not None:
        _, truncated["user"] = await run_in_threadpool(memory_store.set_user, body.user)
    if body.agent is not None:
        _, truncated["agent"] = await run_in_threadpool(memory_store.set_agent, body.agent)
    return JSONResponse({
        "user": memory_store.get_user(),
        "agent": memory_store.get_agent(),
        "max_chars": MAX_MEMORY_CHARS,
        "truncated": truncated,
    })


# ---- text-only turn (skips STT) -------------------------------------------

class TextTurnRequest(BaseModel):
    text: str


@app.post("/api/text-turn")
async def text_turn(req: TextTurnRequest) -> JSONResponse:
    """Run one text turn: LLM + TTS only, no audio upload / STT.

    Accepts a JSON body ``{"text": "..."}``, runs it through the LLM and TTS
    pipeline, and returns the same JSON shape as ``/api/turn``.
    """
    user_text = req.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="empty text")

    # Repeat-prompt dedup: if the same normalized text arrived in the
    # last _DEDUP_TTL_S seconds AND it isn't a time-sensitive query
    # (weather/news/time/etc.), short-circuit to the cached response
    # without re-running the LLM or TTS. Catches the double-click +
    # page-reload + accidental-enter pattern. Cheap (one dict lookup)
    # and never serves stale data — time-sensitive prompts always
    # re-run.
    cached = await _dedup_lookup(user_text)
    if cached is not None:
        log.info("text-turn dedup hit: %r", user_text[:60])
        return JSONResponse(cached)

    log.info("text-turn: %s", user_text)

    async with pipeline_lock:
        try:
            reply = await run_in_threadpool(llm.chat, user_text)
        except Exception as e:
            log.error("LLM failed: %s", e)
            raise HTTPException(status_code=502, detail=f"LLM failed: {e}")
        reply = reply.strip()
        if not reply:
            raise HTTPException(status_code=502, detail="empty reply from LLM")
        log.info("assistant: %s", reply)

        # Persist + maybe compact — otherwise text-only turns done via
        # the non-streaming endpoint silently drop off disk.
        await run_in_threadpool(
            _persist_and_maybe_compact, user_text=user_text, reply=reply,
        )

        if not TTS_ENABLED:
            reply_pcm, sr = np.zeros(0, dtype=np.int16), 16000
        else:
            try:
                reply_pcm, sr = await run_in_threadpool(tts.synthesize, reply)
            except TTSError as e:
                log.error("TTS rejected: %s", e)
                raise HTTPException(status_code=502, detail=f"TTS failed: {e}")
            except Exception as e:
                log.error("TTS unexpected error: %s", e)
                raise HTTPException(status_code=502, detail=f"TTS failed: {e}")

    wav_bytes = wrap_wav(reply_pcm, sr) if reply_pcm.size else b""
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    log.info(
        "text-turn: ok (%.2fs audio out, %d bytes base64)",
        (len(reply_pcm) / sr) if reply_pcm.size else 0.0,
        len(audio_b64),
    )

    response = {
        "user": user_text,
        "assistant": reply,
        "audio_b64": audio_b64,
        "audio_mime": "audio/wav" if audio_b64 else None,
    }
    await _dedup_store(user_text, response)
    return JSONResponse(response)
