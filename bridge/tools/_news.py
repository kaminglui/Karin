"""News, alerts, and digest tools."""
from __future__ import annotations

import logging

log = logging.getLogger("bridge.tools")


def _get_news(topic: str | None = None) -> str:
    """Return the top news story: voice line plus detail blurb when
    available.

    Lazy import of bridge.news.service so the news subsystem doesn't
    load (and hit its config files) on every Karin startup — only when
    the LLM first calls this tool.

    ``fetch=False`` keeps the tool read-only: user queries never
    trigger upstream feed fetches. The background news poller (every
    20 min; see bridge/pollers.py) is the authoritative source of
    refreshes so the NewsAPI + RSS call budget stays bounded by the
    poller's schedule rather than scaling with user activity.

    Output shape:
      <voice_line>

      <detail> (if present — extracted-body or RSS-summary blurb,
      translated if the translate feature is on)
    """
    try:
        from bridge.news.service import get_default_service
        briefs = get_default_service().get_news(
            topic=topic, max_results=1, fetch=False,
        )
    except Exception as e:
        log.error("get_news failed: %s", e)
        return f"Error fetching news: {e}"
    if not briefs:
        suffix = f" for '{topic}'" if topic else ""
        return f"No news found{suffix}."
    brief = briefs[0]
    if brief.detail:
        return f"{brief.voice_line}\n\n{brief.detail}"
    return brief.voice_line




def _get_alerts(max_results: int = 5) -> str:
    """Return a concise listing of currently-active system alerts.

    Triggers a TTL-gated scan first, so repeated calls inside the cache
    window don't hammer upstream feeds. Lazy-imports AlertService so the
    alerts subsystem doesn't load on Karin startup.
    """
    try:
        from bridge.alerts.formatting import format_alerts_voice
        from bridge.alerts.service import get_default_alert_service
        svc = get_default_alert_service()
        svc.scan(force=False)  # TTL-gated
        alerts = svc.get_active_alerts(max_results=max_results)
    except Exception as e:
        log.error("get_alerts failed: %s", e)
        return f"Error fetching alerts: {e}"
    return format_alerts_voice(alerts)




def _get_digest() -> str:
    """Return the day's digest as a short voice line the LLM can
    paraphrase. Deterministic — reads the cached snapshot produced
    by the hourly digest poller. No LLM involved in composition."""
    try:
        from bridge.digest.service import get_default_digest_service
        svc = get_default_digest_service()
        snap = svc.latest()
        # Rebuild-on-read if the cache is empty — gives us something
        # to show on first boot before the poller has fired.
        if snap is None:
            snap = svc.build_and_persist()
    except Exception as e:
        log.error("get_digest failed: %s", e)
        return f"Error building digest: {e}"
    if snap.is_empty:
        return "Quiet day — nothing flagged today."
    # Return the pre-composed headline; the LLM will paraphrase it
    # per karin.yaml voice rules. We intentionally don't dump every
    # item — the UI page surfaces those; the voice line is the
    # one-sentence overview.
    return snap.headline


