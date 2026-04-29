"""Background pollers for news + alerts.

One daemon thread per poller, started at web server boot, shut down
cleanly on server exit via threading.Event signals. Each poller is a
``(name, interval_seconds, callable)`` tuple; the runner just fires
the callable on its cadence and logs outcomes.

Why a thread, not asyncio: the underlying NewsService and AlertService
use synchronous httpx clients + blocking JSON ledger I/O. Running them
on the event loop would require converting those too, which isn't
worth it for two pollers. Threads cost ~50 KB each and block the GIL
only during the actual network/disk work.

Lifecycle:
  - :func:`start_pollers` is called once at web startup. Idempotent:
    calling twice no-ops the second time.
  - Each poller runs on its own thread. Threads are daemons so process
    exit doesn't wait on them.
  - :func:`stop_pollers` signals every poller's stop event; threads
    exit at the next interval check (<= their interval seconds).

Failure mode: if a poll raises, it's logged and the runner continues.
A broken poller doesn't wedge the others or crash the server.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("bridge.pollers")


@dataclass
class Poller:
    """One background job spec — name, cadence, and the work function.

    `fn` takes no args and returns nothing; the runner wraps it in
    exception handling so no single poll can crash the thread."""
    name: str
    interval_sec: int
    fn: Callable[[], None]


# Module-level registry + lifecycle events. Using module state keeps
# start_pollers idempotent (relevant when uvicorn --reload restarts
# the process without killing the import cache) and makes the set of
# running pollers inspectable from a REPL / health endpoint.
_lock = threading.Lock()
_stop_event = threading.Event()
_threads: list[threading.Thread] = []


def _runner(poller: Poller, stop: threading.Event) -> None:
    """Run one poller on its cadence until stop is set.

    We fire the first poll AFTER the first interval, not at t=0 —
    server startup is already expensive (model load, ledger open)
    and back-to-back bootstraps don't need immediate polling on top.
    Use stop.wait() as the sleep primitive so shutdown is prompt.
    """
    log.info("poller %s started (every %ds)", poller.name, poller.interval_sec)
    while not stop.wait(poller.interval_sec):
        start = time.time()
        try:
            poller.fn()
        except Exception as e:
            log.warning("poller %s raised (continuing): %s", poller.name, e)
        else:
            dur = time.time() - start
            log.debug("poller %s ok in %.2fs", poller.name, dur)
    log.info("poller %s stopped", poller.name)


def start_pollers(pollers: list[Poller]) -> None:
    """Spawn a daemon thread for each poller. Idempotent."""
    with _lock:
        if _threads:
            log.debug("pollers already running; start_pollers is a no-op")
            return
        _stop_event.clear()
        for p in pollers:
            t = threading.Thread(
                target=_runner,
                args=(p, _stop_event),
                daemon=True,
                name=f"poller-{p.name}",
            )
            t.start()
            _threads.append(t)
        log.info("started %d background pollers", len(_threads))


def stop_pollers(timeout: float = 5.0) -> None:
    """Signal every poller to stop and wait up to `timeout` seconds.

    Called from the web server's shutdown hook. Idempotent — safe to
    call when no pollers are running."""
    with _lock:
        if not _threads:
            return
        _stop_event.set()
        for t in _threads:
            t.join(timeout=timeout)
        _threads.clear()
        log.info("pollers stopped")


# --- concrete poller factories ----------------------------------------------
#
# These are thin adapters over the service layers; they live here
# (rather than inside each service module) so the web server can wire
# them up in one place without each service knowing about the others.


def news_poller() -> Poller:
    """Refresh the news ledger (RSS + NewsAPI if keyed). 20-min cadence
    keeps us well under NewsAPI's 100/day free quota (72 calls/day at
    20m) and keeps the cluster-state machine responsive to breaking
    stories without being a polling abuser."""
    def _tick() -> None:
        from bridge.news.service import get_default_service
        result = get_default_service().ingest_latest(force=True)
        log.info(
            "news poll: feeds_ok=%d feeds_fail=%d new_articles=%d",
            result.fetched_feeds, result.failed_feeds, result.new_articles,
        )
    return Poller(name="news", interval_sec=20 * 60, fn=_tick)


def alerts_poller() -> Poller:
    """Refresh the alerts ledger (travel advisories + NWS weather).

    2-min cadence for *emergency responsiveness* — NWS publishes new
    CAP entries about once a minute during severe weather, so any
    interval slower than that meaningfully delays the user's first
    look at a tornado/flash-flood warning. The endpoints we hit
    (state.gov travel advisory XML; NWS api.weather.gov) do not
    publish tight rate limits and a 2-min poll of a handful of URLs
    is well within their reasonable-use guidance.

    For true push delivery (server-sent events or webhooks) see the
    notification/SSE idea in docs/ideas.md.
    """
    def _tick() -> None:
        from bridge.alerts.service import get_default_alert_service
        # force=True bypasses the TTL gate so each poll actually
        # hits upstream — the point of the poller is keeping the
        # ledger fresh between user queries, not riding their cache.
        get_default_alert_service().scan(force=True)
    return Poller(name="alerts", interval_sec=2 * 60, fn=_tick)


def trackers_poller() -> Poller:
    """Refresh every enabled tracker on a slow cadence.

    6-hour cadence is a safe fit for the mix:
      * daily series (FX, gold, RBOB) have a 12h TTL at the service
        layer, so a 6h poll gives two in-flight chances per TTL window
        while still being well under the Frankfurter / Stooq rate caps.
      * weekly (EIA retail gas) and monthly (BLS CPI food) series have
        24h TTLs and release cadences much slower than 6h — the
        service-layer TTL short-circuits most polls to a no-op.

    force=True bypasses the per-cadence TTL so the poller actually
    attempts refresh on schedule. API keys (BLS, EIA) are optional
    per-tracker; missing-key cases fail gracefully inside refresh_one.
    """
    def _tick() -> None:
        from bridge.trackers.service import get_default_tracker_service
        results = get_default_tracker_service().refresh_all(force=True)
        ok = sum(1 for v in results.values() if v)
        log.info("trackers poll: refreshed %d/%d", ok, len(results))
    return Poller(name="trackers", interval_sec=6 * 60 * 60, fn=_tick)


def calendar_poller() -> Poller:
    """Fetch configured .ics feeds on a 10-minute cadence and fire
    notifications for events crossing their lead-time window. No-op
    when the ``calendar`` feature flag is off.

    10 min is a good balance for "15-minute heads-up" notifications
    — worst case you get the ping 10 min before the event instead
    of exactly 15, which is still useful. Tighter cadence would be
    wasted work; looser risks silent misses."""
    def _tick() -> None:
        from bridge import features
        if not features.is_enabled("calendar", default=False):
            return
        from bridge.calendar.poll import tick as calendar_tick
        fired = calendar_tick()
        if fired:
            log.info("calendar poll: fired %d notification(s)", fired)
    return Poller(name="calendar", interval_sec=10 * 60, fn=_tick)


def reminders_poller() -> Poller:
    """Tick the reminder scheduler. 60-second cadence — the smallest
    interval where wall-clock drift on a busy system is invisible to
    the user ("set a reminder for 2 minutes from now" feels prompt
    even if the tick happens 30 s after the trigger time).

    The scheduler call is idempotent: the store's atomic
    mark_delivered short-circuits any double-fire if two ticks ever
    overlap. Failures inside fire_due_reminders are logged there;
    this poller wrapper only logs the per-tick summary count."""
    def _tick() -> None:
        from bridge.reminders.api import fire_due_reminders
        fired = fire_due_reminders()
        if fired:
            log.info("reminders poll: fired %d due reminder(s)", len(fired))
    return Poller(name="reminders", interval_sec=60, fn=_tick)


def digest_poller() -> Poller:
    """Rebuild today's digest from current news/alerts/trackers state.
    Hourly cadence is plenty — the digest is a ROLLUP of the faster
    pollers; no point refreshing more often than the underlying data
    could have changed. Always overwrites today's JSON, so the
    stored version is always current-state."""
    def _tick() -> None:
        from bridge.digest.service import get_default_digest_service
        snap = get_default_digest_service().build_and_persist()
        log.info(
            "digest built: news=%d alerts=%d trackers=%d",
            len(snap.news), len(snap.alerts), len(snap.trackers),
        )
    return Poller(name="digest", interval_sec=60 * 60, fn=_tick)
