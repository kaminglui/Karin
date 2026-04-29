"""Tracker tools (market/FX/commodity)."""
from __future__ import annotations

import datetime
import logging

log = logging.getLogger("bridge.tools")


_TRACKER_PERIOD_DAYS: dict[str, int] = {
    "1w": 7, "1week": 7, "week": 7,
    "1m": 30, "1month": 30, "month": 30,
    "3m": 90, "3month": 90, "quarter": 90,
    "6m": 180, "6month": 180, "halfyear": 180,
    "1y": 365, "1year": 365, "year": 365,
}


# ---- tracker tool (Phase 5.1) --------------------------------------------

# Tool-layer alias map. Normalized keys (lowercased, stripped). Lives in
# the tool layer per spec — the tracker store itself only knows
# canonical ids. Unknown inputs pass through unchanged so the service's
# own "not found" handling kicks in.
_TRACKER_ALIASES: dict[str, str] = {
    "gold": "gold_usd",
    "xau": "gold_usd",
    "xauusd": "gold_usd",
    "usd/cny": "usd_cny",
    "usdcny": "usd_cny",
    "usd-cny": "usd_cny",
    "usd/hkd": "usd_hkd",
    "usdhkd": "usd_hkd",
    "usd-hkd": "usd_hkd",
    "usd/jpy": "usd_jpy",
    "usdjpy": "usd_jpy",
    "usd-jpy": "usd_jpy",
    "food": "us_cpi_food",
    "cpi_food": "us_cpi_food",
    "food_at_home": "us_cpi_food_at_home",
    "cpi_food_at_home": "us_cpi_food_at_home",
    "groceries": "us_cpi_food_at_home",
    # Gasoline — retail pump price only. Wholesale/futures (RBOB) was
    # removed because it confused users who just wanted to know what
    # they'd pay at the station. Default region is PADD 1B (Central
    # Atlantic: PA/NJ/NY/MD) via trackers config; override via duoarea.
    "gas":        "gas_retail",
    "gasoline":   "gas_retail",
    "pump":       "gas_retail",
    "pump_gas":   "gas_retail",
    "retail_gas": "gas_retail",
    # Crypto — placeholders only. Map common aliases so if the user
    # flips the tracker on, LLM tool calls resolve correctly. Until
    # that flip happens the service returns "not found" cleanly.
    "btc":      "btc_usd",
    "bitcoin":  "btc_usd",
    "eth":      "eth_usd",
    "ether":    "eth_usd",
    "ethereum": "eth_usd",
}


def _resolve_tracker_alias(raw: str) -> str:
    """Map a user/LLM-supplied id to the canonical tracker id.

    Case-insensitive, whitespace-tolerant. Unknown inputs are returned
    unchanged — the service's 'unknown id -> None' path produces the
    user-facing "not found" message rather than this function raising.
    """
    if not raw:
        return raw
    return _TRACKER_ALIASES.get(raw.strip().lower(), raw.strip().lower())


def _get_tracker(tracker_id: str) -> str:
    """Return a one-sentence summary for a single tracker.

    Wraps TrackerService.get_tracker. Lazy-imports the service so the
    tools module doesn't pay tracker-subsystem load cost at startup.

    ``fetch=False`` keeps the tool read-only: only the background
    tracker poller (bridge/pollers.py) is allowed to hit BLS / EIA /
    Stooq / Frankfurter, so the free-tier API budgets aren't spent
    per user query.
    """
    try:
        from bridge.trackers.formatting import format_tracker_voice
        from bridge.trackers.service import get_default_tracker_service
        canonical = _resolve_tracker_alias(tracker_id)
        snapshot = get_default_tracker_service().get_tracker(
            canonical, fetch=False,
        )
    except Exception as e:
        log.error("get_tracker failed: %s", e)
        return f"Error fetching tracker: {e}"
    if snapshot is None:
        return f"Tracker '{tracker_id}' not found."
    return format_tracker_voice(snapshot)


def _get_trackers(tracker_ids: list[str] | None = None) -> str:
    """Return a concise multi-line listing of trackers.

    `tracker_ids=None` (or omitted) returns every enabled tracker in
    config order. A non-empty list filters to just those ids.

    Read-only: see :func:`_get_tracker` for the rationale.
    """
    try:
        from bridge.trackers.formatting import format_trackers_voice
        from bridge.trackers.service import get_default_tracker_service
        svc = get_default_tracker_service()
        if tracker_ids:
            canonical_ids = [_resolve_tracker_alias(i) for i in tracker_ids]
            snapshots = svc.get_trackers(ids=canonical_ids, fetch=False)
        else:
            snapshots = svc.get_trackers(fetch=False)
    except Exception as e:
        log.error("get_trackers failed: %s", e)
        return f"Error fetching trackers: {e}"
    if not snapshots:
        if tracker_ids:
            return f"No trackers found for: {', '.join(tracker_ids)}."
        return "No trackers available."
    return format_trackers_voice(snapshots)




def _tracker_history_summary(tracker_id: str, period: str) -> str:
    """Return a long-window stats line for one tracker.

    Reads persisted history from the tracker store, windowed to the
    requested period, and computes high / low / mean / net change. No
    network call — history was already filled by normal refresh paths.
    """
    period_key = (period or "").strip().lower()
    days = _TRACKER_PERIOD_DAYS.get(period_key)
    if days is None:
        valid = sorted({k for k in _TRACKER_PERIOD_DAYS if len(k) <= 3})
        return f"Unknown period '{period}'. Try one of: {', '.join(valid)}."
    try:
        from bridge.trackers.service import get_default_tracker_service
        svc = get_default_tracker_service()
        canonical = _resolve_tracker_alias(tracker_id)
        # Trigger normal TTL-gated refresh so we don't analyze stale data.
        svc.refresh_one(canonical, force=False)
        records = svc._store.load()
        record = records.get(canonical)
        cfg = svc._configs_by_id.get(canonical)
    except Exception as e:
        log.error("tracker history fetch failed: %s", e)
        return f"Error fetching tracker history: {e}"
    if record is None or cfg is None:
        return f"Tracker '{tracker_id}' not found."
    if not record.history:
        return f"No history yet for {cfg.label}."

    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=days)
    points = [r for r in record.history if r.timestamp >= cutoff]
    if len(points) < 2:
        return (
            f"Only {len(points)} data point for {cfg.label} in the last "
            f"{period_key} — not enough to analyze. Configured retention "
            f"is {cfg.history_days} days."
        )
    values = [p.value for p in points]
    hi, lo = max(values), min(values)
    first, last = points[0].value, points[-1].value
    net_abs = last - first
    net_pct = (net_abs / first * 100.0) if first else 0.0
    mean = sum(values) / len(values)
    first_date = points[0].timestamp.strftime("%Y-%m-%d")
    last_date = points[-1].timestamp.strftime("%Y-%m-%d")
    arrow = "↑" if net_abs > 0 else "↓" if net_abs < 0 else "→"
    return (
        f"{cfg.label} over ~{period_key} ({first_date} → {last_date}, "
        f"{len(points)} points): high {hi:.2f}, low {lo:.2f}, "
        f"avg {mean:.2f}, net {arrow} {abs(net_abs):.2f} ({net_pct:+.2f}%)."
    )



def _tracker(id: str | None = None, period: str | None = None) -> str:
    """Unified tracker tool.

    Dispatch matrix:
      id empty, period empty  →  grid of all trackers (latest values)
      id set,   period empty  →  single latest snapshot + 1d/1w/1m deltas
      id set,   period set    →  long-window stats (high/low/avg/net) for
                                  the requested span (1w/1m/3m/6m/1y)
      id empty, period set    →  rejected (ask for a specific tracker)

    The period path consumes only persisted history — no new network
    calls — so it's fast and safe to call alongside the latest snapshot.
    """
    tid = (id or "").strip()
    p = (period or "").strip()
    if p and not tid:
        return (
            "A period was given without an id — please say which "
            "tracker you want history for (gold, usd_jpy, food, etc.)."
        )
    if tid and p:
        return _tracker_history_summary(tid, p)
    if tid:
        return _get_tracker(tid)
    return _get_trackers()


# ---- update_memory -----------------------------------------------------------
