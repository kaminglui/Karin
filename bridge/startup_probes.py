"""Startup sanity probes for external API dependencies.

Runs at FastAPI boot to answer the question "can the data-fetching
pipelines reach their upstreams?" without waiting for the next poller
tick (20 min for news, 2 min for alerts, 6 hr for trackers). Probes
are read-only, tight-timeout, parallel.

Semantics:
- One probe per service. Either the service's data source is up or
  it isn't — we don't enumerate every feed/tracker, since the goal is
  a boot-time smoke test, not an upstream audit.
- Probes never raise. Any error is captured + logged as WARNING.
- Periodic pollers run unchanged; probes do not reset cadences.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# Tight — boot path must stay responsive. If an upstream needs more
# than 8 s just to respond to a sanity probe, the periodic poller's
# larger timeout (e.g. alerts.nws.fetch_timeout_s=60 via tuning.yaml)
# still gets a chance; the probe just flags the slow upstream.
PROBE_TIMEOUT_S = 8.0

# Stable, keyless, tiny-response canaries per service. Kept here
# (not fetched from service configs) so the probe is decoupled from
# runtime config and stays deterministic.
NWS_PROBE_URL = "https://api.weather.gov/"
FRANKFURTER_PROBE_URL = "https://api.frankfurter.app/latest?from=USD&to=EUR"


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    detail: str
    latency_s: float


def _probe_alerts() -> ProbeResult:
    """NWS root returns a small JSON API index. If it 200s, NWS is up."""
    t0 = time.monotonic()
    try:
        r = httpx.get(
            NWS_PROBE_URL,
            timeout=PROBE_TIMEOUT_S,
            headers={"User-Agent": "Karin (startup-probe)"},
        )
        ok = r.status_code == 200
        return ProbeResult("alerts", ok, f"HTTP {r.status_code}", time.monotonic() - t0)
    except Exception as e:
        return ProbeResult("alerts", False, f"{type(e).__name__}: {e}", time.monotonic() - t0)


def _probe_news() -> ProbeResult:
    """Probe the first feed URL configured in the news service.

    We intentionally don't probe all 20 feeds — that's the job of
    the periodic poller. One representative probe answers 'can we
    reach the internet + parse RSS in principle?'.
    """
    t0 = time.monotonic()
    try:
        from bridge.news.service import get_default_service

        feeds = get_default_service()._feeds
        if not feeds:
            return ProbeResult("news", False, "no feeds configured", time.monotonic() - t0)
        url = feeds[0].url
        r = httpx.head(url, timeout=PROBE_TIMEOUT_S, follow_redirects=True)
        ok = r.status_code < 400
        return ProbeResult(
            "news", ok, f"{url.split('//', 1)[-1].split('/', 1)[0]} HTTP {r.status_code}",
            time.monotonic() - t0,
        )
    except Exception as e:
        return ProbeResult("news", False, f"{type(e).__name__}: {e}", time.monotonic() - t0)


def _probe_trackers() -> ProbeResult:
    """Frankfurter is the most common (keyless) tracker backend, used
    for FX + commodity-priced-in-currency trackers. If it resolves,
    the tracker pipeline's base case is reachable."""
    t0 = time.monotonic()
    try:
        r = httpx.get(FRANKFURTER_PROBE_URL, timeout=PROBE_TIMEOUT_S, follow_redirects=True)
        ok = r.status_code == 200 and "rates" in r.text
        return ProbeResult("trackers", ok, f"HTTP {r.status_code}", time.monotonic() - t0)
    except Exception as e:
        return ProbeResult("trackers", False, f"{type(e).__name__}: {e}", time.monotonic() - t0)


_PROBES = (_probe_alerts, _probe_news, _probe_trackers)


def run_probes() -> list[ProbeResult]:
    """Run all probes in parallel, log results, return them.

    Total wallclock is bounded by max(probe) since they run
    concurrently — in practice 1-3 s when everything's healthy.
    Safe to call from the FastAPI startup hook.
    """
    log.info("startup probes: firing %d sanity checks", len(_PROBES))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_PROBES)) as pool:
        futures = [pool.submit(p) for p in _PROBES]
        results = [f.result() for f in futures]

    ok_count = sum(1 for r in results if r.ok)
    for r in results:
        level = logging.INFO if r.ok else logging.WARNING
        log.log(
            level,
            "startup probe %s: %s (%s, %.2fs)",
            r.name, "OK" if r.ok else "FAIL", r.detail, r.latency_s,
        )
    log.info("startup probes: %d/%d ok", ok_count, len(results))
    return results
