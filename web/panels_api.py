"""Thin JSON API for the UI panels (Phase UI-1).

Exposes read-only endpoints that return the same structured dataclasses
the tool dispatcher already consumes — so the UI never has to scrape
formatted voice strings.

Routes:
    GET /api/alerts/active?max_results=5
    GET /api/news/briefs?topic=<str>&max_results=3
    GET /api/trackers/snapshots
    GET /api/trackers/snapshot?id=<tracker_id_or_alias>

Each route:
  1. Gets the default subsystem singleton (same one the tools use).
  2. Calls the same service method the tools call, so behavior and
     TTL/cooldown semantics are identical to the voice path.
  3. Serializes dataclass results to JSON via bridge.utils.json_default.

Run standalone for UI development without the full voice stack:

    python -m uvicorn web.panels_api:app --host 0.0.0.0 --port 8002

Or later, include the router into web/server.py alongside the PTT UI:

    from web.panels_api import router as panels_router
    app.include_router(panels_router)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue as thread_queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bridge.utils import REPO_ROOT, json_default

log = logging.getLogger("web.panels_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def _server_error(op: str, exc: Exception) -> HTTPException:
    """Log the real exception, return a sanitized 500 to the client.

    Prior pattern leaked exception messages (and often stack-wrapped
    paths) via ``detail=f"X failed: {e}"``. Now the traceback stays in
    server logs and the client sees only the operation name.
    """
    log.exception("%s failed", op)
    return HTTPException(status_code=500, detail=f"{op} failed")


router = APIRouter(prefix="/api", tags=["panels"])

# Separate router for the /ui/* HTML pages. Kept prefix-free so the paths
# map 1:1 (/ui/alerts, /ui/news, etc.) and so web/server.py can include
# both routers without the /api prefix leaking into page URLs.
ui_router = APIRouter(tags=["panels-ui"])


# --- serialization helper -------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    """Round-trip through json to get a plain-Python tree that FastAPI's
    jsonable encoder will accept. Handles our Enum + datetime types via
    bridge.utils.json_default."""
    return json.loads(json.dumps(obj, default=json_default))


def _dataclass_list_to_json(items) -> list[dict]:
    return [_to_jsonable(asdict(item)) for item in items]


# --- alerts ---------------------------------------------------------------

@router.get("/alerts/active")
def alerts_active(max_results: int = Query(5, ge=1, le=50)) -> JSONResponse:
    """Active alerts (cooldown_until > now), highest level first.

    Triggers a TTL-gated scan first (same as the get_alerts tool), so
    opening the panel will refresh alerts when the scan TTL has expired.

    Phase G.c enrichment:
      - each alert gets a top-level ``threat_score`` (max over the 0-4
        scores stamped on its triggered signals; None when no signal
        has a score yet — typically because the user hasn't set a
        location and everything scored 0).
      - top-level ``location_configured`` tells the UI whether to
        show the "set your location" hint. False when the active
        profile has no user_location block.
    """
    from bridge.alerts.service import get_default_alert_service
    try:
        svc = get_default_alert_service()
        svc.scan(force=False)
        alerts = svc.get_active_alerts(max_results=max_results)
    except Exception as e:
        raise _server_error("alerts_active", e)

    alert_dicts = _dataclass_list_to_json(alerts)
    for entry in alert_dicts:
        entry["threat_score"] = _max_threat_score(entry.get("triggered_by_signals"))

    return JSONResponse({
        "count": len(alert_dicts),
        "alerts": alert_dicts,
        "location_configured": _location_configured(),
    })


def _max_threat_score(signals) -> int | None:
    """Return the highest 0-4 threat_score across the signals list, or
    None if no signal has one. Guards against a signal list without
    payload.threat_score (old persisted alerts, mocks)."""
    if not isinstance(signals, list):
        return None
    best: int | None = None
    for s in signals:
        payload = (s or {}).get("payload") or {}
        score = payload.get("threat_score")
        if isinstance(score, (int, float)):
            iv = int(score)
            if best is None or iv > best:
                best = iv
    return best


def _location_configured() -> bool:
    """True when the active profile (or the yaml fallback) has at least
    one non-empty field under user_location. Mirrors the precedence the
    threat assessor uses so the UI hint stays honest."""
    try:
        from bridge.alerts.user_context import load_user_context
        ctx = load_user_context()
    except Exception as e:
        log.debug("location_configured check raised: %s", e)
        return False
    return bool(
        ctx.city or ctx.region or ctx.country
        or ctx.latitude is not None or ctx.longitude is not None
    )


# --- news -----------------------------------------------------------------

@router.get("/news/briefs")
def news_briefs(
    topic: str | None = Query(None),
    max_results: int = Query(3, ge=1, le=100),
) -> JSONResponse:
    """Top news story briefs, ranked by state then preference then recency.

    Calls NewsService.get_news() — same path the get_news tool uses,
    including TTL-gated ingest and topic-filter-with-fallback.

    UI enrichment: each brief in the response carries an extra
    `watchlist_matches` list (not a field on StoryBrief itself). This
    lets the NewsPanel render "why this matched my preferences" chips
    without a second round trip. StoryBrief stays unchanged as the
    backend contract; enrichment is API-layer-only.

    `fetch=False` — the UI must never block on a full ingest. When the
    TTL window expires, get_news(fetch=True) would synchronously run
    RSS fetching + trafilatura extraction + keyword-learn LLM calls
    (2+ minutes end-to-end on the Jetson with an 8B model). The
    background poller (every 20 min) is the authoritative refresh
    path; user-facing endpoints read from cache only. Matches how the
    get_news chat tool passes fetch=False for the same reason.
    """
    from bridge.news.preferences import match_watchlist_items
    from bridge.news.service import get_default_service
    try:
        svc = get_default_service()
        briefs = svc.get_news(topic=topic, max_results=max_results, fetch=False)
        articles = svc.load_all_articles()
        clusters = svc.load_all_clusters()
        prefs = svc.get_preferences()
    except Exception as e:
        raise _server_error("news_briefs", e)

    enriched: list[dict] = []
    for brief in briefs:
        brief_dict = _to_jsonable(asdict(brief))
        cluster = clusters.get(brief.cluster_id)
        matches = match_watchlist_items(cluster, articles, prefs) if cluster else []
        brief_dict["watchlist_matches"] = [_to_jsonable(asdict(m)) for m in matches]
        enriched.append(brief_dict)

    return JSONResponse({
        "topic": topic,
        "count": len(enriched),
        "briefs": enriched,
    })


@router.get("/news/cluster/{cluster_id}")
def news_cluster(cluster_id: str) -> JSONResponse:
    """Return one cluster's metadata + all member articles, each with
    (when available) extracted full-article text.

    Powers the news-detail modal: the cluster is the "story", the
    articles are the different outlets' coverage of it, and
    extracted_text is what trafilatura pulled from each page. Articles
    without extraction show an empty text + the publisher URL as a
    fallback "read on <source>" link.
    """
    from bridge.news.service import get_default_service
    try:
        svc = get_default_service()
        articles = svc.load_all_articles()
        clusters = svc.load_all_clusters()
        extracted = svc._extract_store.load()
    except Exception as e:
        raise _server_error("news_cluster", e)

    cluster = clusters.get(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"cluster {cluster_id} not found")

    # Attach source display names + extracted text per member article.
    sources = svc._sources
    members = []
    for aid in cluster.article_ids:
        art = articles.get(aid)
        if art is None:
            continue
        src = sources.get(art.source_id)
        ex = extracted.get(aid)
        members.append({
            "article_id": aid,
            "title": art.display_title,
            "url": art.url,
            "published_at": art.published_at.isoformat(),
            "source_id": art.source_id,
            "source_name": src.name if src else art.source_id,
            "wire_attribution": art.wire_attribution,
            "extracted_text": ex.text if (ex and ex.ok) else "",
            "extracted_author": ex.author if (ex and ex.ok) else "",
            "extraction_error": ex.error if (ex and not ex.ok) else "",
        })
    # Put extraction-successful members first — users prefer readable
    # rows at the top of the modal.
    members.sort(key=lambda m: (not bool(m["extracted_text"]), m["published_at"]))

    return JSONResponse({
        "cluster_id": cluster.cluster_id,
        "headline": cluster.centroid_display_title,
        "state": cluster.state.value if hasattr(cluster.state, "value") else str(cluster.state),
        "is_stale": cluster.is_stale,
        "independent_confirmation_count": cluster.independent_confirmation_count,
        "article_count": cluster.article_count,
        "latest_update_at": cluster.latest_update_at.isoformat(),
        "members": members,
    })


# --- digest --------------------------------------------------------------

@router.get("/digest/today")
def digest_today() -> JSONResponse:
    """Today's pre-computed digest. Pulls the most recent cached
    snapshot (hourly poller writes it) — never triggers LLM calls or
    upstream fetches on the request path. If no snapshot exists yet
    (first boot before the poller fired), builds one on the spot
    from current ledger state so the UI always gets something."""
    from bridge.digest.service import get_default_digest_service, _snap_to_dict
    svc = get_default_digest_service()
    try:
        snap = svc.latest() or svc.build_and_persist()
    except Exception as e:
        raise _server_error("digest", e)
    return JSONResponse(_snap_to_dict(snap))


# --- trackers -------------------------------------------------------------

@router.get("/trackers/snapshots")
def trackers_snapshots() -> JSONResponse:
    """All enabled trackers as TrackerSnapshot list.

    TTL-gated refresh per tracker (via TrackerService.get_trackers).
    """
    from bridge.trackers.service import get_default_tracker_service
    try:
        snapshots = get_default_tracker_service().get_trackers()
    except Exception as e:
        raise _server_error("trackers_snapshots", e)
    return JSONResponse({
        "count": len(snapshots),
        "snapshots": _dataclass_list_to_json(snapshots),
    })


@router.get("/news/graph")
def news_graph() -> JSONResponse:
    """Phase F.a: relation-graph view of the current cluster set.

    Nodes = user's enabled watchlist items (regions / topics / events).
    Edges = count of clusters that match both endpoints. Used by the
    future graph viz panel; hitting the endpoint directly surfaces the
    raw numbers so we can judge whether the data is worth drawing
    before investing in layout code.

    Returns ``{"nodes": [], "edges": [], "counts": {...}}`` when the
    user hasn't configured watchlists (preferences disabled) or when
    no current cluster matches any watchlist item.
    """
    from bridge.news.service import get_default_service
    try:
        payload = get_default_service().build_graph_payload()
    except Exception as e:
        raise _server_error("news_graph", e)
    return JSONResponse(payload)


@router.get("/news/learned-keywords")
def news_learned_keywords() -> JSONResponse:
    """Phase E: entities the LLM has learned to associate with each
    configured watchlist bucket.

    Shape: ``{regions: [{watchlist_label, entities: [...]}, ...],
    topics: [...], events: [...]}`` — empty arrays when the feature
    flag is off or nothing has been learned yet.
    """
    from bridge.news.service import get_default_service
    try:
        payload = get_default_service().learned_keywords()
    except Exception as e:
        raise _server_error("news_learned_keywords", e)
    return JSONResponse(payload)


@router.get("/trackers/catalog")
def trackers_catalog() -> JSONResponse:
    """Every configured tracker, grouped by category.

    Used by the Settings panel so the user can reorder trackers within
    each category (the `tracker_order` field in tracker_preferences.json).
    Returns all trackers — enabled AND disabled — so the UI can show
    the full layout; whether a tracker actually surfaces in the Trackers
    panel is governed separately by its `enabled` flag in trackers.json.
    """
    from bridge.trackers.service import get_default_tracker_service
    try:
        svc = get_default_tracker_service()
    except Exception as e:
        raise _server_error("trackers_catalog", e)
    from bridge.trackers.preferences import is_tracker_visible
    prefs = svc._preferences
    by_category: dict[str, list[dict]] = {}
    for cfg in svc._configs:
        by_category.setdefault(cfg.category, []).append({
            "id": cfg.id,
            "label": cfg.label,
            "category": cfg.category,
            # `enabled` = config default (from trackers.json).
            # `visible` = effective state after user prefs apply — the
            # UI uses this to render the enable toggle's current value.
            "enabled": cfg.enabled,
            "visible": is_tracker_visible(cfg.id, cfg.category, cfg.enabled, prefs),
        })
    return JSONResponse({
        "categories": by_category,
        "disabled_categories": list(prefs.disabled_categories),
    })


@router.get("/trackers/snapshot")
def tracker_snapshot(id: str = Query(..., description="Tracker id or alias")) -> JSONResponse:
    """Single tracker snapshot. Resolves aliases (e.g. 'gold' -> 'gold_usd')
    via the same alias map the get_tracker tool uses.
    """
    # Alias resolution lives in bridge.tools; import lazily to avoid
    # dragging the full tool module into the panels API at startup.
    from bridge.tools import _resolve_tracker_alias
    from bridge.trackers.service import get_default_tracker_service
    canonical = _resolve_tracker_alias(id)
    try:
        snap = get_default_tracker_service().get_tracker(canonical)
    except Exception as e:
        raise _server_error("tracker_snapshot", e)
    if snap is None:
        raise HTTPException(
            status_code=404,
            detail=f"tracker '{id}' not found (resolved to '{canonical}')",
        )
    return JSONResponse({
        "id": canonical,
        "snapshot": _to_jsonable(asdict(snap)),
    })


# --- chat streaming (UI-5) ------------------------------------------------
#
# Provides a thin NDJSON stream over the LLM's chat loop so the integrated
# chat+panels page can observe tool invocations in real time and mount the
# right panel as soon as Karin decides to call get_news / get_alerts / etc.
#
# The LLM instance is lazy-singleton: first request builds it from the
# same assistant.yaml the PTT server reads. Subsequent requests reuse it
# so the rolling conversation history persists across chat turns.
#
# Event shape (one JSON object per line):
#   {"type": "tool_call", "name": "get_news", "arguments": {...}}
#   {"type": "reply",     "text":  "Karin's paraphrased response"}
#   {"type": "done"}
#   {"type": "error",     "detail":"why"}

_llm = None
_llm_lock = threading.Lock()


def _get_llm():
    """Lazy singleton used by the standalone chat stream.

    Instantiated inside this module rather than imported from web/server.py
    because server.py loads heavy audio models (Whisper + SoVITS) at
    import time. The chat-only UI doesn't need any of that.
    """
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is not None:
            return _llm
        from bridge.llm import OllamaLLM
        from bridge.utils import load_config
        cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
        lcfg = cfg["llm"]
        _llm = OllamaLLM(
            base_url=lcfg["base_url"],
            model=lcfg["model"],
            system_prompt=lcfg["system_prompt"],
            temperature=lcfg["temperature"],
            num_ctx=lcfg["num_ctx"],
            options=lcfg.get("options", {}),
            backend=lcfg.get("backend", "ollama"),
            under_fire_rescue=bool(lcfg.get("under_fire_rescue", True)),
        )
        log.info("chat LLM ready: model=%s", lcfg["model"])
    return _llm


class _ChatRequest(BaseModel):
    text: str


@router.post("/chat/stream")
async def chat_stream(req: _ChatRequest) -> StreamingResponse:
    """Text chat turn, streamed as NDJSON.

    The LLM call runs in a background thread; tool invocations are pushed
    onto a thread-safe queue via the on_tool_call callback and relayed to
    the HTTP client through an async generator. This mirrors the same
    pattern web/server.py uses for streaming TTS chunks.
    """
    user_text = (req.text or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="empty text")

    from bridge import tools as tools_mod
    llm = _get_llm()
    q: thread_queue.Queue = thread_queue.Queue()

    def _run_chat() -> None:
        def _on_tool(name: str, arguments: dict, result: str) -> None:
            q.put({"type": "tool_call", "name": name, "arguments": arguments})
        try:
            reply = llm.chat(
                user_text,
                tools=tools_mod.active_tool_schemas(),
                on_tool_call=_on_tool,
            )
            q.put({"type": "reply", "text": reply or ""})
        except Exception as e:
            log.exception("chat turn failed: %s", e)
            q.put({"type": "error", "detail": str(e)})
        finally:
            q.put({"type": "done"})

    async def generate():
        thread = threading.Thread(target=_run_chat, daemon=True)
        thread.start()
        while True:
            try:
                event = await asyncio.to_thread(q.get, True, 120)
            except thread_queue.Empty:
                yield json.dumps({"type": "error", "detail": "LLM turn timed out"}) + "\n"
                break
            yield json.dumps(event) + "\n"
            if event.get("type") == "done":
                break
        thread.join(timeout=5)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# --- standalone app -------------------------------------------------------

app = FastAPI(
    title="Karin panels API",
    description=(
        "Read-only JSON API for the UI panels. Same service calls as the "
        "LLM tool dispatcher, returning typed dataclasses as JSON. "
        "Designed to be mounted into web/server.py OR run standalone on "
        "its own port for UI development."
    ),
    docs_url="/api/docs",
    redoc_url=None,
)
app.include_router(router)
app.include_router(ui_router)

# Serve panel CSS/JS assets. Paths resolve to web/static/* via REPO_ROOT
# so the app can be launched from any working directory.
_STATIC_DIR = REPO_ROOT / "web" / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

_PANELS_DIR = _STATIC_DIR / "panels"


@app.get("/")
def index() -> dict:
    """Bare landing page. Lists the routes for a human browsing directly."""
    return {
        "service": "Karin panels API",
        "routes": [
            "GET  /api/alerts/active?max_results=5",
            "GET  /api/news/briefs?topic=<str>&max_results=3",
            "GET  /api/trackers/snapshots",
            "GET  /api/trackers/snapshot?id=<tracker_id_or_alias>",
            "POST /api/chat/stream    (NDJSON tool-call + reply events)",
            "GET  /api/docs            (interactive OpenAPI)",
            "GET  /ui/                 (integrated chat + panels, UI-5)",
            "GET  /ui/alerts           (AlertsPanel standalone, UI-2)",
            "GET  /ui/news             (NewsPanel standalone, UI-3)",
            "GET  /ui/trackers         (TrackersPanel standalone, UI-4)",
        ],
    }


# --- UI pages -------------------------------------------------------------
#
# Each /ui/<panel> route serves the standalone HTML shell. The shell loads
# panels.css + panels.js (shared) and the per-panel script, which mounts
# the panel into its own container. Future UI-5 integration can reuse the
# same mount functions against a different container inside Karin's chat.

# UI pages are registered on ui_router so web/server.py can pull them
# in directly. Attached to the standalone app via the include_router()
# call at the bottom of this file.


def _render_panel(name: str) -> HTMLResponse:
    """Read a panel HTML and substitute `{{ASSET_VERSION}}` so the
    embedded `<script src=...?v=...>` and `<link href=...?v=...>` tags
    bypass browser caches on each server start. Mirrors the same token
    the parent index page uses (web/server.py).

    Also sets `Cache-Control: no-store` so the iframe always re-fetches
    the panel HTML when reopened. Without this, browsers heuristically
    cache the response (FastAPI doesn't set cache headers by default)
    and a stale HTML keeps loading old asset URLs even after a server
    restart bumped the ASSET_VERSION."""
    import time as _time
    html = (_PANELS_DIR / f"{name}.html").read_text(encoding="utf-8")
    html = html.replace("{{ASSET_VERSION}}", str(int(_time.time())))
    return HTMLResponse(
        html,
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@ui_router.get("/ui/alerts", response_class=HTMLResponse)
def ui_alerts() -> HTMLResponse:
    return _render_panel("alerts")


@ui_router.get("/ui/news", response_class=HTMLResponse)
def ui_news() -> HTMLResponse:
    return _render_panel("news")


@ui_router.get("/ui/trackers", response_class=HTMLResponse)
def ui_trackers() -> HTMLResponse:
    return _render_panel("trackers")


@ui_router.get("/ui/digest", response_class=HTMLResponse)
def ui_digest() -> HTMLResponse:
    return _render_panel("digest")


@ui_router.get("/ui/settings", response_class=HTMLResponse)
def ui_settings() -> HTMLResponse:
    return _render_panel("settings")


@ui_router.get("/ui/facts", response_class=HTMLResponse)
def ui_facts() -> HTMLResponse:
    """Standalone year-card page — type a year, get the curated facts
    aggregator output (population, inflation, cohort age, wages, items,
    Wikipedia events) without going through the chat."""
    return _render_panel("facts")


@ui_router.get("/ui/graph", response_class=HTMLResponse)
def ui_graph() -> HTMLResponse:
    """Phase F.b: D3 force-directed view of /api/news/graph. Vendors
    d3.min.js from web/static/vendor — no CDN, fully local."""
    return _render_panel("graph")


@ui_router.get("/ui/alice", response_class=HTMLResponse)
def ui_alice() -> HTMLResponse:
    """ALICE (Asset Limited, Income Constrained, Employed) estimator —
    survival budget breakdown for a US 4-person household + cross-
    validation against United for ALICE published thresholds."""
    return _render_panel("alice")


@ui_router.get("/ui/map", response_class=HTMLResponse)
def ui_map() -> HTMLResponse:
    """US choropleth map — state-level housing/rent data, with
    click-to-emit `karin:focus-region` postMessage for future
    cross-panel navigation."""
    return _render_panel("map")


# --- preferences read / write --------------------------------------------

# Preferences are split between two locations:
#
# * `data/<sub>/preferences.json` — written by the Settings panel.
#   Lives under data/ because that's the bind-mounted writable
#   volume in the Jetson docker compose; config/ is mounted ro.
# * `bridge/<sub>/config/preferences.json` — legacy hand-edited
#   path. Loader still reads from here as a fallback so existing
#   setups don't break, but the API never writes here.
#
# Reads always check the writable path first, then fall back to the
# legacy path. Writes always go to the writable path.

# Phase H: preference files move under the active profile. The legacy
# paths are still consulted on read so the UI keeps working during the
# migration window; writes always go to the profile. Resolution is
# deferred to call time (functions, not module-level paths) because
# active_profile() can change between requests if the user switches.
_NEWS_PREFS_LEGACY_WRITABLE = REPO_ROOT / "data" / "news" / "preferences.json"
_NEWS_PREFS_LEGACY_CONFIG = REPO_ROOT / "bridge" / "news" / "config" / "preferences.json"
_TRACKER_PREFS_LEGACY_WRITABLE = REPO_ROOT / "data" / "trackers" / "tracker_preferences.json"
_TRACKER_PREFS_LEGACY_CONFIG = REPO_ROOT / "bridge" / "trackers" / "config" / "tracker_preferences.json"


def _news_prefs_write_path() -> Path:
    from bridge.profiles import active_profile
    return active_profile().news_dir / "preferences.json"


def _tracker_prefs_write_path() -> Path:
    from bridge.profiles import active_profile
    return active_profile().trackers_dir / "tracker_preferences.json"


class _PrefsBody(BaseModel):
    """POST body for the preferences endpoints. The caller sends the
    full file contents as a parsed JSON object — we validate, write,
    and ask the subsystem to reload its singleton."""
    data: dict


def _read_prefs_file(*paths: Path) -> dict:
    """Read the first existing path from ``paths`` (preference order:
    writable, then legacy). Empty dict if none exist."""
    for path in paths:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"{path.name} is not valid JSON: {e}",
            )
    return {}


def _write_prefs_file(path: Path, data: dict) -> None:
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="prefs body must be a JSON object")
    # Pretty-print so a human inspecting the file on disk gets a
    # reasonable layout. trailing newline keeps UNIX tools happy.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@router.get("/preferences/news")
def preferences_news_get() -> JSONResponse:
    # Read: profile first, then legacy writable, then legacy config.
    return JSONResponse({"data": _read_prefs_file(
        _news_prefs_write_path(),
        _NEWS_PREFS_LEGACY_WRITABLE,
        _NEWS_PREFS_LEGACY_CONFIG,
    )})


@router.post("/preferences/news")
def preferences_news_post(body: _PrefsBody) -> JSONResponse:
    target = _news_prefs_write_path()
    _write_prefs_file(target, body.data)
    # Force a re-load on the next get_news call so the edit takes
    # effect immediately without a container restart.
    try:
        from bridge.news import service as _news_service
        _news_service.reset_default_service()
    except Exception as e:
        log.warning("news preferences save: singleton reset failed: %s", e)
    return JSONResponse({"ok": True, "path": str(target)})


@router.get("/preferences/trackers")
def preferences_trackers_get() -> JSONResponse:
    return JSONResponse({"data": _read_prefs_file(
        _tracker_prefs_write_path(),
        _TRACKER_PREFS_LEGACY_WRITABLE,
        _TRACKER_PREFS_LEGACY_CONFIG,
    )})


@router.post("/preferences/trackers")
def preferences_trackers_post(body: _PrefsBody) -> JSONResponse:
    target = _tracker_prefs_write_path()
    _write_prefs_file(target, body.data)
    try:
        from bridge.trackers import service as _tracker_service
        _tracker_service.reset_default_tracker_service()
    except Exception as e:
        log.warning("tracker preferences save: singleton reset failed: %s", e)
    return JSONResponse({"ok": True, "path": str(target)})


# Notification channel URLs (Discord webhook, ntfy topic). Same
# round-trip pattern as the prefs above but routes through the
# notify.secrets module so the file location, mode 0600 chmod, and
# env-var fallback all live in one place. Saving resets the
# dispatcher so the next push uses the new URLs without a restart.

@router.get("/preferences/location")
def preferences_location_get() -> JSONResponse:
    """Return the current user_location block from the active profile's
    preferences.json. Empty {} when nothing has been set yet — that's
    the state the alerts hint banner triggers on."""
    from bridge.profiles import load_profile_preferences
    prefs = load_profile_preferences()
    block = prefs.get("user_location") or {}
    return JSONResponse({
        "data": block if isinstance(block, dict) else {},
    })


@router.post("/preferences/location")
def preferences_location_post(body: _PrefsBody) -> JSONResponse:
    """Merge the posted user_location into the active profile's
    preferences.json. Other top-level keys (memory, watchlists,
    cooldowns, …) are preserved. Empty-string fields are stripped so
    the file stays clean and `_location_configured()` returns false
    again if the user clears all fields."""
    from bridge.profiles import (
        load_profile_preferences, save_profile_preferences,
    )
    block = body.data or {}
    if not isinstance(block, dict):
        raise HTTPException(
            status_code=400,
            detail="user_location must be a JSON object",
        )
    # Strip empty-string fields; coerce numerics for lat/lon so an
    # accidental string like "37.7" still saves as a float.
    cleaned: dict[str, Any] = {}
    for k in ("city", "region", "country"):
        v = block.get(k)
        if isinstance(v, str) and v.strip():
            cleaned[k] = v.strip()
    for k in ("latitude", "longitude"):
        v = block.get(k)
        if v in (None, ""):
            continue
        try:
            cleaned[k] = float(v)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=f"{k} must be numeric, got {v!r}",
            )
    prefs = load_profile_preferences()
    if cleaned:
        prefs["user_location"] = cleaned
    else:
        # All-empty save means "clear my location" — drop the key
        # entirely so `_location_configured()` returns false and the
        # alerts hint reappears.
        prefs.pop("user_location", None)
    path = save_profile_preferences(prefs)
    return JSONResponse({"ok": True, "path": str(path), "data": cleaned})


# --- Optional API keys ---
# Some upstream sources (CDC Socrata, NewsAPI, etc.) work without a
# key but raise rate limits / unlock features when one is provided.
# Stored under the active profile's preferences.json `api_keys` block.
# GET returns metadata (set/not, masked hint) — never echoes the full
# secret. POST accepts plaintext to write; empty string clears the
# entry. Env vars (KARIN_*) still win at runtime, mirroring how
# NewsAPI resolves its key, so deploy/.env continues to work unchanged.
#
# The per-key registry lives in `bridge.profiles.PROFILE_API_KEY_FIELDS`
# so the bridge resolver and the UI metadata can't drift apart — a new
# key needs exactly one entry there.

# Defense-in-depth: cap an individual saved key at 4 KB so a malformed
# UI submission can't write an arbitrary blob into preferences.json.
# Real keys are <100 chars; 4 KB is generous headroom.
_MAX_API_KEY_LEN = 4096


def _mask_secret(value: object) -> str:
    """Show only the last 4 characters of a secret for the hint field.
    Long enough to confirm "yes, that's the one I pasted"; short enough
    that a screenshot doesn't leak the secret. Defensive type-check
    keeps a future registry-shape change from raising."""
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if not s:
        return ""
    if len(s) <= 4:
        return "····"
    return "····" + s[-4:]


@router.get("/preferences/api-keys")
def preferences_api_keys_get() -> JSONResponse:
    """Return per-key metadata for the Settings UI.

    Never returns the plaintext secret. ``set`` reflects whether ANY
    source (env or preferences) currently resolves to a non-empty
    value, so the UI can show "✓ in use" even when the secret lives
    in deploy/.env. ``hint`` shows the last 4 chars of the
    preferences-stored value (env-only secrets show empty hint
    because they're not ours to echo)."""
    from bridge.profiles import (
        PROFILE_API_KEY_FIELDS, load_profile_preferences,
    )
    prefs = load_profile_preferences()
    block = prefs.get("api_keys") or {}
    if not isinstance(block, dict):
        block = {}
    out: dict[str, dict[str, Any]] = {}
    for name, meta in PROFILE_API_KEY_FIELDS.items():
        raw = block.get(name)
        prefs_val = raw if isinstance(raw, str) else ""
        env_val = (os.environ.get(meta["env"]) or "").strip()
        in_prefs = bool(prefs_val.strip())
        out[name] = {
            "label": meta["label"],
            "purpose": meta["purpose"],
            "register_url": meta["register_url"],
            "env_var": meta["env"],
            "set": bool(env_val) or in_prefs,
            "source": (
                "environment" if env_val else
                "preferences" if in_prefs else
                "none"
            ),
            "hint": _mask_secret(prefs_val) if in_prefs else "",
        }
    return JSONResponse({"data": out})


@router.post("/preferences/api-keys")
def preferences_api_keys_post(body: _PrefsBody) -> JSONResponse:
    """Save plaintext values into the profile's preferences.json
    `api_keys` block. Only known names are accepted. Empty-string or
    whitespace clears the entry. Round-trip returns the same masked
    metadata as GET so the UI can refresh status in one shot."""
    from bridge.profiles import (
        PROFILE_API_KEY_FIELDS,
        load_profile_preferences,
        save_profile_preferences,
    )
    block = body.data or {}
    if not isinstance(block, dict):
        raise HTTPException(
            status_code=400,
            detail="api-keys body must be a JSON object",
        )
    unknown = sorted(set(block) - set(PROFILE_API_KEY_FIELDS))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown api-key name(s): {unknown}",
        )
    prefs = load_profile_preferences()
    existing = prefs.get("api_keys") or {}
    if not isinstance(existing, dict):
        existing = {}
    for name, value in block.items():
        if not isinstance(value, str):
            raise HTTPException(
                status_code=400,
                detail=f"{name} value must be a string",
            )
        if len(value) > _MAX_API_KEY_LEN:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name} value exceeds {_MAX_API_KEY_LEN}-char "
                    f"limit (got {len(value)})"
                ),
            )
        cleaned = value.strip()
        if cleaned:
            existing[name] = cleaned
        else:
            existing.pop(name, None)
    if existing:
        prefs["api_keys"] = existing
    else:
        prefs.pop("api_keys", None)
    save_profile_preferences(prefs)
    return preferences_api_keys_get()


@router.get("/preferences/notify")
def preferences_notify_get() -> JSONResponse:
    from bridge.notify.secrets import read_secrets
    return JSONResponse({"data": read_secrets()})


@router.post("/preferences/notify")
def preferences_notify_post(body: _PrefsBody) -> JSONResponse:
    from bridge.notify.secrets import write_secrets
    write_secrets(body.data)
    try:
        from bridge.notify.dispatcher import reset_dispatcher
        reset_dispatcher()
    except Exception as e:
        log.warning("notify preferences save: dispatcher reset failed: %s", e)
    return JSONResponse({"ok": True})


# --- reminders ----------------------------------------------------------
#
# The inline chat card calls this when the user taps "Undo". No body
# required — the id is in the URL. Returns 404 if the reminder was
# already delivered / never existed / already cancelled, which the
# UI treats as "card is stale, just remove it anyway".

@router.post("/reminders/{reminder_id}/cancel")
def reminders_cancel(reminder_id: str) -> JSONResponse:
    from bridge.reminders import cancel_reminder
    ok = cancel_reminder(reminder_id)
    if not ok:
        raise HTTPException(status_code=404, detail="reminder not found")
    return JSONResponse({"ok": True, "id": reminder_id})


@router.get("/reminders/upcoming")
def reminders_upcoming(limit: int = Query(50, ge=1, le=200)) -> JSONResponse:
    """Enumerate pending (undelivered) reminders soonest-first. The
    Settings page can use this later to build a full reminder list
    view; for now it just powers debug / manual review."""
    from bridge.reminders import list_upcoming
    rems = list_upcoming(limit=limit)
    return JSONResponse({
        "count": len(rems),
        "reminders": [
            {
                "id": r.id,
                "trigger_at": r.trigger_at.isoformat(),
                "message": r.message,
                "source": r.source,
            }
            for r in rems
        ],
    })


# ===========================================================================
# Phase H.c: profile management
#
# Every subsystem that holds user-specific state (reminders, news prefs,
# alerts cooldowns, user_location, ...) now lives under
# ``data/profiles/<active>/``. These endpoints expose profile listing,
# creation, and switching to the UI.
#
# Switching is persistent (writes data/active_profile.txt) but does NOT
# hot-swap in-process singletons — the client must reload / restart the
# bridge for the new profile to take effect. We return that requirement
# explicitly so the UI can show a banner rather than leaving the user
# wondering why their old data is still showing.
# ===========================================================================


class _ProfileCreateBody(BaseModel):
    name: str


class _ProfileActiveBody(BaseModel):
    name: str


@router.get("/profiles")
def profiles_list() -> JSONResponse:
    """Return the list of profile names and which one is currently
    active. Never raises; an error reading the registry is surfaced as
    an empty list rather than a 500, so the UI picker always renders."""
    from bridge import profiles
    try:
        # Resolve active first — it auto-creates the default profile
        # on disk if nothing exists yet. list_profiles() then sees it
        # and the UI renders a correct "default" entry on fresh boot.
        active = profiles.active_profile().name
        names = profiles.list_profiles()
    except Exception as e:
        log.warning("profiles_list failed: %s", e)
        return JSONResponse({"profiles": [], "active": None, "error": str(e)})
    return JSONResponse({"profiles": names, "active": active})


@router.post("/profiles")
def profiles_create(body: _ProfileCreateBody) -> JSONResponse:
    """Create a new profile. Idempotent (returns existing on collision)
    so the UI doesn't need to race list-vs-create."""
    from bridge import profiles
    try:
        p = profiles.create_profile(body.name)
    except profiles.ProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.warning("profiles_create failed for %r: %s", body.name, e)
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"ok": True, "name": p.name, "path": str(p.root)})


@router.post("/profiles/active")
def profiles_set_active(body: _ProfileActiveBody) -> JSONResponse:
    """Persist the active profile choice. The caller should show a
    "restart required to fully apply" banner — singletons built around
    the previous profile continue to use the old paths until the bridge
    restarts. Data is safe (each profile has isolated directories), but
    e.g. the feedback_store instance is already bound to the prior path."""
    from bridge import profiles
    try:
        p = profiles.set_active(body.name)
    except profiles.ProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.warning("profiles_set_active failed for %r: %s", body.name, e)
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({
        "ok": True,
        "active": p.name,
        "restart_required": True,
    })


# --- Phase H.d: IP-to-profile routing config ----------------------------
#
# Tailscale IP → profile mapping. Lets devices auto-select their
# profile by IP rather than relying on cookies or manual switching.
# The middleware in server.py reads this per-request and auto-switches
# active_profile.txt when the mapped profile differs. These endpoints
# expose the mapping to the Settings UI for editing.


# ===========================================================================
# TTS voice management — auto-discover + runtime switching
# ===========================================================================


@router.get("/tts/voices")
def tts_voices_list() -> JSONResponse:
    """Return discovered voice models + which one is currently active."""
    from bridge.tts_voices import discover_voices
    from bridge.utils import REPO_ROOT, load_config
    try:
        cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
        # voice_dir is optional — when unset, discover_voices scans
        # characters/*/voices/ by default (the new per-character layout).
        voice_dir = cfg.get("tts", {}).get("voice_dir")
        active = cfg.get("tts", {}).get("voice", "")
    except Exception as e:
        log.warning("tts_voices_list config error: %s", e)
        return JSONResponse({"voices": [], "active": "", "error": str(e)})
    voices = discover_voices(voice_dir)
    return JSONResponse({
        "voices": [
            {
                "name": v.name,
                "prompt_lang": v.prompt_lang,
                "text_lang": v.text_lang,
                "description": v.description,
            }
            for v in voices.values()
        ],
        "active": active,
    })


class _VoiceSwitchBody(BaseModel):
    name: str


@router.post("/tts/voice")
def tts_voice_switch(body: _VoiceSwitchBody) -> JSONResponse:
    """Hot-swap the active TTS voice without restarting the bridge.

    Pushes new weights to GPT-SoVITS and updates the ref audio path.
    Also updates prompt_lang if the voice has different metadata.
    """
    from bridge.tts_voices import discover_voices
    from bridge.utils import REPO_ROOT, load_config, resolve_path
    try:
        cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
        tts_cfg = cfg.get("tts", {})
        voice_dir = tts_cfg.get(
            "voice_dir", str(REPO_ROOT / "voice_training"),
        )
    except Exception as e:
        raise _server_error("config load", e)

    voices = discover_voices(voice_dir)
    target = body.name.strip().lower()

    # A character folder may exist without a voice bundle (e.g. the
    # shipped "default" neutral character, or a newly-added one where
    # the operator hasn't dropped in voice weights yet). In that case
    # we still accept the POST so the persona + face can swap — just
    # skip the weight-load step below.
    target_char_dir = REPO_ROOT / "characters" / target
    target_has_voice_yaml = (target_char_dir / "voice.yaml").is_file()
    target_has_voice_bundle = target in voices

    # Same predicate the dropdown scanner uses (web.server._character_is_activatable);
    # keeping them in sync means clients can never POST a target the
    # scanner would have hidden.
    if not (target_has_voice_yaml or target_has_voice_bundle):
        available = ", ".join(voices.keys()) or "(none)"
        raise HTTPException(
            status_code=400,
            detail=f"character {body.name!r} not found. Voices available: {available}",
        )

    voice = voices.get(target)
    import web.server as _srv
    if target_has_voice_bundle and _srv.tts is None:
        raise HTTPException(status_code=400, detail="TTS is disabled")

    _tts_base = os.environ.get("KARIN_TTS_BASE_URL") or tts_cfg.get("base_url", "")
    _tts_remote = "localhost" not in _tts_base and "127.0.0.1" not in _tts_base

    # Path resolution strategy:
    #   local TTS  → absolute container path (resolve_path).
    #   remote TTS → path that the PC-side TTS server can open from
    #                its own CWD. The convention baked into
    #                config/assistant.yaml uses a "../../" prefix
    #                (PC-TTS CWD is typically the GPT-SoVITS install
    #                dir, two levels below the Karin repo). We derive
    #                the prefix from the already-configured startup
    #                paths so voice-switching matches the convention
    #                the user wired in — no hardcoded assumption
    #                about depth.
    from pathlib import Path as _Path
    import re as _re_prefix

    def _derive_remote_prefix() -> str:
        for key in ("gpt_weights_path", "ref_audio_path", "sovits_weights_path"):
            sample = str(tts_cfg.get(key) or "")
            m = _re_prefix.match(r"^((?:\.\./)+)characters/", sample)
            if m:
                return m.group(1)
        # No config sample → default to the standard two-level hop
        # used by the current deploy/pc-tts setup.
        return "../../"

    _remote_prefix = _derive_remote_prefix() if _tts_remote else ""

    def _rp_local(p) -> str:
        return str(resolve_path(str(p)))
    def _rp_remote(p) -> str:
        p = _Path(str(p))
        try:
            rel = p.relative_to(REPO_ROOT)
            return _remote_prefix + str(rel).replace("\\", "/")
        except ValueError:
            # Already relative or lives outside the repo; pass through.
            return str(p).replace("\\", "/")
    _rp = _rp_remote if _tts_remote else _rp_local

    if target_has_voice_bundle:
        try:
            _srv.tts.switch_voice(
                ref_audio_path=_rp(voice.ref_wav),
                gpt_weights_path=_rp(voice.gpt_ckpt),
                sovits_weights_path=_rp(voice.sovits_pth),
            )
            if voice.prompt_lang:
                _srv.tts.prompt_lang = voice.prompt_lang
            if voice.text_lang:
                _srv.tts.text_lang = voice.text_lang
        except Exception as e:
            raise _server_error("voice switch", e)
    else:
        # Voice-less character (e.g. "default") — persona still needs to
        # swap, TTS weights have nothing to load. Continue to the
        # persona-swap block below.
        log.info(
            "character switch without voice weights: target=%r (voices/ empty)",
            target,
        )

    # Swap the LLM persona to match the new voice. Re-read the
    # character template and fill {persona} + {language_note} from
    # the new character's voice.yaml. This lets the LLM's personality
    # match whoever is "speaking."
    #
    # IMPORTANT: _fill_voice_persona reads `cfg["character"]`, NOT
    # `cfg["tts"]["voice"]`, to locate the character's voice.yaml. So
    # we MUST set both: character (drives persona) + tts.voice (drives
    # the voice weight lookup). Previous code set only tts.voice and
    # the persona silently fell back to the config default — that's
    # the "karin's prompt leaks when I switch to general" bug.
    #
    # Also set os.environ["KARIN_CHARACTER"] so the next /index render
    # picks up the new selection. The server's module-level `cfg` was
    # loaded once at startup, but load_config() + the /index handler
    # both honor the env var override at read time.
    try:
        from bridge.utils import REPO_ROOT, load_config, _fill_voice_persona
        fresh_cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
        if "tts" not in fresh_cfg:
            fresh_cfg["tts"] = {}
        fresh_cfg["character"] = target        # drives persona lookup
        fresh_cfg["tts"]["voice"] = target     # drives voice weights
        _fill_voice_persona(fresh_cfg)
        new_prompt = fresh_cfg.get("llm", {}).get("system_prompt", "")
        if new_prompt and hasattr(_srv, "llm") and _srv.llm is not None:
            _srv.llm.system_prompt = new_prompt
            # Persist selection for this process so reloads see it.
            os.environ["KARIN_CHARACTER"] = target
            log.info(
                "swapped character to %r — persona + voice + env override updated",
                target,
            )
    except Exception as e:
        log.warning("persona swap failed (non-fatal, TTS still switched): %s", e)

    return JSONResponse({
        "ok": True,
        "voice": target,
        # voice can be None for voice-less characters (e.g. shipped
        # "default") — the persona still swapped above, just no weights.
        "prompt_lang": voice.prompt_lang if voice else None,
        "text_lang": voice.text_lang if voice else None,
        "description": voice.description if voice else None,
    })


class _RoutingEntry(BaseModel):
    ip: str
    profile: str
    nickname: str = ""


class _RoutingBody(BaseModel):
    entries: list[_RoutingEntry]


@router.get("/profiles/routing")
def profiles_routing_get() -> JSONResponse:
    from bridge.profiles.routing import get_routing_dict
    return JSONResponse({"entries": get_routing_dict()})


@router.get("/profiles/routing/peers")
def profiles_routing_peers() -> JSONResponse:
    """Discover devices on the Tailscale tailnet via the local daemon."""
    from bridge.profiles.routing import discover_tailscale_peers
    peers = discover_tailscale_peers()
    return JSONResponse({"peers": peers})


@router.post("/profiles/routing")
def profiles_routing_post(body: _RoutingBody) -> JSONResponse:
    from bridge.profiles.routing import set_routing_list
    from bridge.profiles import ProfileNameError
    try:
        path = set_routing_list([e.model_dump() for e in body.entries])
    except ProfileNameError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.warning("profiles_routing_post failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"ok": True, "path": str(path)})
