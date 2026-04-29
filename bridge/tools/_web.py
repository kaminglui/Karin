"""Web search and place finder tools."""
from __future__ import annotations

import concurrent.futures
import logging

from bridge._http import make_client
from bridge.tools._weather import _is_vague_location, _ip_geolocate

log = logging.getLogger("bridge.tools")


# Outer timeout for ``ddgs.DDGS().text(...)`` calls. The library has no
# user-facing timeout knob — DDG occasionally hangs on a slow upstream
# and the call sits forever, blocking the chat turn. Wrap in a thread
# pool with an overall deadline so the tool returns a clean error
# instead. 8s is generous for a healthy DDG; pathological hangs return
# in time for the chat loop's other safety nets to take over.
_DDG_TIMEOUT_S = 8.0


def _ddg_text_with_timeout(query: str, max_results: int) -> list[dict]:
    """Run ``DDGS().text()`` under an outer deadline. Raises
    ``TimeoutError`` on hang, the underlying ``ddgs`` exception on
    upstream errors. Caller decides how to translate either."""
    from ddgs import DDGS

    def _go() -> list[dict]:
        with DDGS() as d:
            return list(d.text(query, max_results=max_results))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_go)
        return fut.result(timeout=_DDG_TIMEOUT_S)


def _web_search(query: str) -> str:
    """LLM tool: top 3 web results via DuckDuckGo (no API key).

    Uses the ``ddgs`` package which scrapes DDG's HTML — fragile but
    free. Returns a compact text block the model can paraphrase or
    quote. Each line: index, title, short snippet, URL on the next line.
    """
    q = (query or "").strip()
    if not q:
        return "Error: empty query"
    if len(q) > 200:
        return "Error: query too long"
    try:
        import ddgs  # noqa: F401  — re-checks availability before fetch
    except ImportError:
        return "Error: ddgs not installed (pip install ddgs)"
    try:
        results = _ddg_text_with_timeout(q, max_results=3)
    except concurrent.futures.TimeoutError:
        log.warning("web_search timed out after %.1fs (query=%r)", _DDG_TIMEOUT_S, q[:80])
        return "Error: web search timed out — try again shortly"
    except Exception as e:
        log.error("web_search failed: %s", e)
        return f"Error fetching search: {e}"
    if not results:
        return f"No web results found for '{q}'."
    lines: list[str] = [f"Top results for '{q}':"]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip().replace("\n", " ")
        url = (r.get("href") or "").strip()
        if len(body) > 200:
            body = body[:200].rstrip() + "\u2026"
        lines.append(f"{i}. {title} — {body} ({url})")
    return "\n".join(lines)


# ---- Place finder (location-aware web search) ----------------------------

def _resolve_user_location(explicit: str | None) -> str | None:
    """Pick a location string for `find_places`.

    Priority:
      1. Explicit non-vague string the LLM already extracted.
      2. ipapi.co lookup ("city, region" if available).
      3. None (caller surfaces a "couldn't detect location" message).
    """
    if explicit and not _is_vague_location(explicit):
        return explicit.strip()
    try:
        with make_client(timeout=8.0) as client:
            coords = _ip_geolocate(client)
    except Exception:
        return None
    if coords is None:
        return None
    _, _, place_name = coords
    return place_name


def find_places_data(query: str, location: str | None = None) -> dict:
    """Public: location-aware DDG search returning structured results.

    Used both by the LLM tool (text-formatted) and the place widget
    (renders the dict directly). Keeps the tool result and the panel
    in sync — same query, same location, same hits.
    """
    q = (query or "").strip()
    if not q:
        return {"error": "empty query", "results": []}
    resolved = _resolve_user_location(location)
    if resolved is None:
        return {
            "error": "couldn't detect your location — name a city",
            "results": [],
            "query": q,
            "location": None,
        }

    # DDG search query: "best <query> in <location>". The "best" prefix
    # nudges the engine toward review/listicle pages over chain
    # restaurant homepages, which gives the LLM more context to
    # paraphrase from.
    search_q = f"best {q} in {resolved}"

    try:
        import ddgs  # noqa: F401
    except ImportError:
        return {"error": "ddgs not installed", "results": [], "query": q, "location": resolved}
    try:
        raw = _ddg_text_with_timeout(search_q, max_results=5)
    except concurrent.futures.TimeoutError:
        log.warning("find_places timed out after %.1fs (q=%r)", _DDG_TIMEOUT_S, search_q[:80])
        return {"error": "search timed out — try again shortly", "results": [], "query": q, "location": resolved}
    except Exception as e:
        log.error("find_places search failed: %s", e)
        return {"error": str(e), "results": [], "query": q, "location": resolved}

    results = [
        {
            "title": (r.get("title") or "").strip(),
            "body": (r.get("body") or "").strip(),
            "href": (r.get("href") or "").strip(),
        }
        for r in raw
    ]
    return {"query": q, "location": resolved, "search_query": search_q, "results": results}


def _find_places(query: str, location: str | None = None) -> str:
    """LLM tool: short text summary of nearby place suggestions."""
    data = find_places_data(query, location)
    if data.get("error"):
        return f"Error: {data['error']}"
    if not data["results"]:
        return f"No places found for '{query}' in {data['location']}."
    lines = [f"Top {len(data['results'])} {query} places in {data['location']}:"]
    for i, r in enumerate(data["results"][:3], 1):
        title = r["title"]
        body = r["body"].replace("\n", " ")
        if len(body) > 160:
            body = body[:160].rstrip() + "\u2026"
        lines.append(f"{i}. {title} — {body}")
    return "\n".join(lines)


