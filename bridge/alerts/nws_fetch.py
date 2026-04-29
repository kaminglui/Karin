"""NWS (National Weather Service) active-alerts fetcher.

Pulls severe-weather, flood, wind, tornado, winter-storm, heat, and
other CAP-formatted alerts from the free, keyless api.weather.gov
endpoint, filtered to the user's IP-detected coordinates.

Why this is its own module (not reusing advisory_fetch):
  - NWS uses JSON (GeoJSON), not RSS — parsing lives here
  - Active alerts carry a stable ``id`` from the provider, so we
    de-dupe by that id rather than by a country code like advisories
  - Severity mapping is domain-specific (NWS terms: Minor / Moderate /
    Severe / Extreme) and translates differently to AlertLevel than
    State Dept's numeric 1-4 scale

Design:
  - ``fetch_nws_alerts(lat, lon)`` returns a list of small dicts —
    the subset of each alert we actually need (id, event, severity,
    headline, onset/expires, areaDesc). Keeps the NWS JSON schema from
    leaking into the rest of the alerts subsystem.
  - Silently returns [] on any failure (network error, non-200 status,
    parse error). Callers treat missing data as "no alerts this poll",
    not as an exception — an NWS outage shouldn't break the assistant.
  - No state persistence here. Each call is a snapshot of currently-
    active alerts; re-firing is prevented by the alerts-engine cooldown
    (keyed on the NWS alert id via scope_key), not by diffing here.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger("bridge.alerts.nws_fetch")

# api.weather.gov is free + keyless. They DO require a descriptive
# User-Agent with contact info per their TOS, which we supply here.
# Keep this string stable so rate limiting can trace it to this app.
NWS_USER_AGENT = "karin (self-hosted Jetson deploy)"
NWS_BASE = "https://api.weather.gov"
# 15s is conservative. api.weather.gov answers most requests in <2s but
# can briefly stall to 10+ s during active weather events — the last
# thing we want is to drop warnings exactly when they matter most.
# Tunable via config/tuning.yaml → alerts.nws.fetch_timeout_s.
from bridge import tuning as _tuning
FETCH_TIMEOUT = _tuning.get("alerts.nws.fetch_timeout_s", 15.0)


def fetch_nws_alerts(
    lat: float,
    lon: float,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Return the list of currently-active NWS alerts for a point.

    Each returned dict has the shape::

        {
          "id": "urn:oid:...",
          "event": "Winter Storm Warning",
          "severity": "Severe",
          "certainty": "Observed",
          "urgency": "Expected",
          "headline": "Winter Storm Warning issued ...",
          "description": "Total snow accumulations of ...",
          "area_desc": "Centre County, PA",
          "onset": "2026-04-16T02:00:00-04:00",
          "expires": "2026-04-16T14:00:00-04:00",
          "sender": "NWS State College PA",
        }

    Empty list on any failure — callers don't need to distinguish
    "no active alerts" from "NWS was briefly down".
    """
    own_client = client is None
    if own_client:
        # api.weather.gov requires a descriptive User-Agent per their
        # TOS and specifically expects an Accept: application/geo+json
        # header — without it the endpoint has been observed to
        # silently return zero features on some queries.
        client = httpx.Client(headers={
            "User-Agent": NWS_USER_AGENT,
            "Accept": "application/geo+json",
        })
    else:
        # If the caller passed a client, trust them to set headers.
        # Most callers do want the User-Agent added; the auto-created
        # path above handles them. Tests injecting mocks set whatever
        # they like.
        pass

    url = f"{NWS_BASE}/alerts/active"
    params = {"point": f"{lat:.4f},{lon:.4f}"}
    # Always send the geo+json Accept for point-queries. Harmless on
    # caller-supplied clients that already set it.
    request_headers = {"Accept": "application/geo+json"}
    try:
        resp = client.get(
            url, params=params, headers=request_headers,
            timeout=FETCH_TIMEOUT,
        )
    except httpx.HTTPError as e:
        log.warning("NWS fetch failed: %s", e)
        return []
    finally:
        if own_client:
            client.close()

    if resp.status_code != 200:
        log.warning("NWS returned %d: %s", resp.status_code, resp.text[:200])
        return []
    try:
        data = resp.json()
    except ValueError as e:
        log.warning("NWS returned non-JSON: %s", e)
        return []

    features = data.get("features") or []
    out: list[dict[str, Any]] = []
    for feat in features:
        flat = _flatten_feature(feat)
        if flat is not None:
            out.append(flat)
    log.debug("NWS: %d active alerts at (%.4f, %.4f)", len(out), lat, lon)
    return out


def _flatten_feature(feat: dict) -> dict[str, Any] | None:
    """Pull the fields we care about out of a GeoJSON alert feature.

    Drops the geometry and the rest of the CAP metadata — we don't
    need polygon masks or sender_name variants downstream."""
    props = feat.get("properties") or {}
    alert_id = props.get("id") or feat.get("id")
    event = props.get("event") or ""
    if not alert_id or not event:
        return None
    return {
        "id": alert_id,
        "event": event,
        "severity": props.get("severity") or "Unknown",
        "certainty": props.get("certainty") or "Unknown",
        "urgency": props.get("urgency") or "Unknown",
        "headline": props.get("headline") or "",
        "description": props.get("description") or "",
        "area_desc": props.get("areaDesc") or "",
        "onset": props.get("onset") or "",
        "expires": props.get("expires") or "",
        "sender": props.get("senderName") or "",
    }


# Severity values NWS emits (in practice): Unknown, Minor, Moderate,
# Severe, Extreme. We only care about Moderate and above — Minor
# floods us with routine SPS bulletins nobody needs to hear about.
SIGNIFICANT_SEVERITIES: frozenset[str] = frozenset({"Moderate", "Severe", "Extreme"})


def is_significant(alert: dict[str, Any]) -> bool:
    """True if this alert merits firing an AlertLevel.

    Significance = severity of Moderate/Severe/Extreme. Minor weather
    statements (the routine daily SPS feed) are filtered out here so
    they don't generate noise in the alerts panel.
    """
    return (alert.get("severity") or "").strip() in SIGNIFICANT_SEVERITIES
