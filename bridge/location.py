"""User-location context for the LLM system prompt.

Injects the user's approximate IP-derived location as a tiny suffix
block so the model can disambiguate ambiguous place tokens (e.g. "PA"
→ Pennsylvania if the user is in the US). The same ipapi.co path used
by ``bridge.tools._ip_geolocate`` for weather fallback; cached here
so we don't re-query per turn.

Intentionally fail-soft: if geolocation fails (network hiccup, ipapi
rate-limit) the suffix is empty string and the rest of the prompt is
unaffected.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import httpx

log = logging.getLogger("bridge.location")

# Location rarely changes mid-session; an hour is a reasonable TTL
# that avoids hammering ipapi.co while still picking up a move or a
# VPN flip within a workday.
_CACHE_TTL_SEC: float = 3600.0

_cache: dict = {
    "when": 0.0, "block": "", "timezone": "", "utc_offset": "",
    "lat": None, "lon": None,
}
_cache_lock = threading.Lock()


def _load_user_override() -> dict:
    """Read the user_location from profile + yaml + env overrides.

    Precedence (highest wins, per field):
        1. KARIN_USER_* env vars (operational override for containers)
        2. Active profile's preferences.json → user_location
        3. config/assistant.yaml → user_location (legacy, migration path)
        4. Empty defaults

    Per-field precedence (not whole-block) lets a partially-configured
    profile inherit missing fields from the yaml without the user
    having to duplicate everything.
    """
    out = {
        "city": "", "region": "", "country": "",
        "timezone": "", "latitude": None, "longitude": None,
    }

    def _overlay(block: dict) -> None:
        """Overlay non-empty fields from ``block`` onto ``out``.
        Used twice: first with yaml as the base, then profile over that."""
        if not isinstance(block, dict):
            return
        for k in ("city", "region", "country", "timezone"):
            v = block.get(k)
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()
        for k in ("latitude", "longitude"):
            v = block.get(k)
            if isinstance(v, (int, float)):
                out[k] = float(v)

    # Base: yaml (lowest priority of the persisted sources).
    try:
        from bridge.utils import REPO_ROOT, load_config
        cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
        _overlay(cfg.get("user_location") or {})
    except Exception as e:
        log.debug("user_location yaml load failed: %s", e)

    # Profile overrides yaml (user's per-profile choice wins over
    # the system-wide legacy default).
    try:
        from bridge.profiles import load_profile_preferences
        _overlay(load_profile_preferences().get("user_location") or {})
    except Exception as e:
        log.debug("profile user_location load failed: %s", e)

    # Env-var overrides sit on top of everything. Loaded fresh each call so a
    # config reload via restart picks them up without an extra hook.
    for env_key, out_key in (
        ("KARIN_USER_CITY", "city"),
        ("KARIN_USER_REGION", "region"),
        ("KARIN_USER_COUNTRY", "country"),
        ("KARIN_USER_TZ", "timezone"),
    ):
        v = os.environ.get(env_key)
        if v and v.strip():
            out[out_key] = v.strip()
    for env_key, out_key in (("KARIN_USER_LAT", "latitude"), ("KARIN_USER_LON", "longitude")):
        v = os.environ.get(env_key)
        if v:
            try:
                out[out_key] = float(v)
            except ValueError:
                pass
    return out


def _apply_override(cache: dict) -> None:
    """Overlay the user-override on top of whatever the IP lookup
    already wrote into the cache. Override wins on every field the
    user actually set — empty / None fields fall through to the IP
    result (which itself may be empty when the provider rate-limits)."""
    override = _load_user_override()
    parts = [p for p in (override["city"], override["region"], override["country"]) if p]
    place_name = ", ".join(parts)
    if place_name:
        cache["block"] = (
            f"[User location context]\n"
            f"The user is currently in: {place_name}.\n"
            f"Use this when an ambiguous place token appears in a request "
            f"(e.g. state abbreviations like 'PA', 'NY', 'CA', 'TX'). "
            f"Prefer a local interpretation unless the user says otherwise."
        )
    if override["timezone"]:
        cache["timezone"] = override["timezone"]
        # Leave utc_offset alone — it's computed by the IP provider and
        # we'd need ZoneInfo to derive it from the IANA name, which
        # adds dep weight for a cosmetic field.
    if override["latitude"] is not None:
        cache["lat"] = override["latitude"]
    if override["longitude"] is not None:
        cache["lon"] = override["longitude"]


def _refresh_cache() -> None:
    """Populate the cache with a fresh IP-geolocation lookup.

    Thread-safe: only one refresh runs at a time. Concurrent callers
    that arrive while a refresh is in progress wait for it to finish.

    Uses ipwho.is (ipwhois.app) — HTTPS, no auth, 10k/month free tier
    which is two orders of magnitude over our hourly-refresh budget.
    Previously pointed at ipapi.co but that provider started paywalling
    residential / shared IPs with an HTML response that broke
    resp.json() — the fallback branch kicked in every hour and the LLM
    prompt had no location context.

    Response shape reference:
      {"success": true, "city": ..., "region": ..., "country": ...,
       "latitude": ..., "longitude": ...,
       "timezone": {"id": "America/New_York", "utc": "-04:00"}}
    """
    with _cache_lock:
        _refresh_cache_locked()


def _refresh_cache_locked() -> None:
    now = time.time()
    try:
        with httpx.Client() as client:
            resp = client.get("https://ipwho.is/", timeout=8.0)
            if resp.status_code != 200:
                raise httpx.HTTPError(f"status {resp.status_code}")
            data = resp.json()
        # ipwho.is returns {"success": false, "message": "..."} on
        # rate-limit / bad IP. Treat that as a failure, fall through
        # to the override-only path.
        if not isinstance(data, dict) or not data.get("success", False):
            reason = data.get("message", "unknown") if isinstance(data, dict) else "non-dict response"
            raise httpx.HTTPError(f"ipwho.is: {reason}")
    except Exception as e:
        log.debug("ip geolocation failed: %s", e)
        _cache["when"] = now
        _cache["block"] = ""
        _cache["timezone"] = ""
        _cache["utc_offset"] = ""
        _cache["lat"] = None
        _cache["lon"] = None
        # Apply user override even on lookup failure — that's the whole
        # point of the override. Lets the prompt still carry a location
        # when the provider is rate-limiting the Jetson.
        _apply_override(_cache)
        return

    city = data.get("city") or ""
    region = data.get("region") or ""
    country = data.get("country") or ""
    parts = [p for p in (city, region, country) if p]
    place_name = ", ".join(parts)
    if place_name:
        block = (
            f"[User location context]\n"
            f"The user is currently in: {place_name}.\n"
            f"Use this when an ambiguous place token appears in a request "
            f"(e.g. state abbreviations like 'PA', 'NY', 'CA', 'TX'). "
            f"Prefer a local interpretation unless the user says otherwise."
        )
    else:
        block = ""

    # timezone is a nested object on ipwho.is.
    tz_obj = data.get("timezone") or {}
    if isinstance(tz_obj, dict):
        tz = str(tz_obj.get("id") or "")
        offset = str(tz_obj.get("utc") or "").replace(":", "")
    else:
        tz, offset = "", ""

    try:
        lat = float(data.get("latitude")) if data.get("latitude") is not None else None
        lon = float(data.get("longitude")) if data.get("longitude") is not None else None
    except (TypeError, ValueError):
        lat, lon = None, None

    _cache["when"] = now
    _cache["block"] = block
    _cache["timezone"] = tz
    _cache["utc_offset"] = offset
    _cache["lat"] = lat
    _cache["lon"] = lon
    # User override wins on every field the user actually set. Applied
    # last so the IP-derived values above are the fallback, not the
    # authority — crucial when ipapi is rate-limiting and we only got
    # an error block back.
    _apply_override(_cache)


def user_location_context() -> str:
    """Return a short system-prompt block describing the user's location.

    Cached for ``_CACHE_TTL_SEC`` seconds. Returns empty string if IP
    geolocation is unavailable — callers should treat that as a no-op
    suffix, not an error.
    """
    now = time.time()
    if not (_cache["when"] and (now - _cache["when"] < _CACHE_TTL_SEC)):
        _refresh_cache()
    return _cache["block"]


def user_timezone() -> tuple[str, str]:
    """Return ``(iana_timezone, utc_offset)`` for the user's IP.

    Both strings are empty on geolocation failure. Example return:
    ``("America/New_York", "-0400")``. Callers use the IANA name to
    construct a ``ZoneInfo``, and the offset as a display hint.
    """
    now = time.time()
    if not (_cache["when"] and (now - _cache["when"] < _CACHE_TTL_SEC)):
        _refresh_cache()
    return _cache["timezone"], _cache["utc_offset"]


def user_coords() -> tuple[float, float] | None:
    """Return ``(lat, lon)`` for the user's IP, or None on failure."""
    now = time.time()
    if not (_cache["when"] and (now - _cache["when"] < _CACHE_TTL_SEC)):
        _refresh_cache()
    lat, lon = _cache.get("lat"), _cache.get("lon")
    if lat is None or lon is None:
        return None
    return (lat, lon)


def invalidate_cache() -> None:
    """Force the next call to re-query geolocation. Used by tests."""
    _cache["when"] = 0.0
    _cache["block"] = ""
    _cache["timezone"] = ""
    _cache["utc_offset"] = ""
    _cache["lat"] = None
    _cache["lon"] = None
