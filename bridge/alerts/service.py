"""AlertService: the orchestrator that Phase 6's tool layer calls into.

Responsibilities:
  - gather signals from detectors (tracker + news + travel advisory)
  - invoke AlertEngine to apply rules
  - TTL-gate the full scan so the tool path doesn't rescan constantly
  - expose read accessors: get_active_alerts() for the tool layer

Design choices:
  - scan is ON-DEMAND only (no background thread in V1). The tool path
    calls scan(force=False), TTL decides whether to actually re-gather.
  - advisory baseline is seeded silently on first run; no initial flood.
  - each external dependency is injected, so tests can pass mocks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from bridge.alerts.advisory_fetch import (
    AdvisoryFetchError,
    diff_advisories,
    fetch_advisories,
    snapshot_to_state,
)
from bridge.alerts.cooldown import CooldownLedger
from bridge.alerts.detectors import (
    signals_from_advisory_changes,
    signals_from_news,
    signals_from_nws,
    signals_from_trackers,
)
from bridge.alerts.nws_fetch import fetch_nws_alerts, is_significant
from bridge.alerts.engine import AlertEngine
from bridge.alerts.models import (
    Alert,
    AlertCategory,
    AlertLevel,
    ScanResult,
    Signal,
    SignalKind,
)
from bridge.alerts.rules import DEFAULT_RULES, AlertRule
from bridge.alerts.store import AlertStore

log = logging.getLogger("bridge.alerts.service")

# Tunable via config/tuning.yaml → alerts.service.scan_ttl_minutes.
from bridge import tuning as _tuning
DEFAULT_SCAN_TTL_MINUTES = _tuning.get("alerts.service.scan_ttl_minutes", 15)


class AlertService:
    def __init__(
        self,
        *,
        store: AlertStore,
        cooldown_ledger: CooldownLedger,
        rules: list[AlertRule] | None = None,
        tracker_service=None,
        news_service=None,
        advisory_fetcher=None,
        scan_ttl_minutes: int = DEFAULT_SCAN_TTL_MINUTES,
        user_context=None,
    ) -> None:
        self._store = store
        self._engine = AlertEngine(
            rules=rules if rules is not None else DEFAULT_RULES,
            cooldown_ledger=cooldown_ledger,
            store=store,
        )
        self._tracker_service = tracker_service
        self._news_service = news_service
        # advisory_fetcher is an optional callable returning dict[country, {level, title}].
        # Allows tests to inject fakes without patching httpx.
        self._advisory_fetcher = advisory_fetcher
        self._scan_ttl_minutes = scan_ttl_minutes
        # Phase G.a: threat-assessor context. Loaded once here; rules
        # read the resulting score off each signal's payload. Empty
        # context (no configured location) makes every news-proximity
        # check score 0 — effectively "no life-threat signals from
        # news" until the user fills in assistant.yaml:user_location.
        # Tests inject a matching UserContext via the kwarg.
        if user_context is None:
            from bridge.alerts.user_context import load_user_context
            user_context = load_user_context()
        self._user_context = user_context
        # Phase G.b: optional LLM verifier for borderline (score 2-3)
        # news signals. Feature-flag gated so the rule-based path
        # alone is the default and shipping this is a no-op until
        # `alerts_threat_llm` is flipped on.
        self._threat_verifier = _build_threat_verifier_if_enabled(
            store.data_dir,
        )

    # --- scan ----------------------------------------------------------

    def scan(self, *, force: bool = False) -> ScanResult:
        """Gather signals, run rules, persist outcomes.

        TTL-gated: if the last successful scan event is within the TTL
        window, returns early with skipped_due_to_cache=True.
        """
        if not force and self._within_ttl():
            return ScanResult(skipped_due_to_cache=True)

        now = datetime.now(timezone.utc)
        signals: list[Signal] = []
        signals.extend(self._collect_tracker_signals(now))
        signals.extend(self._collect_news_signals(now))
        # State Dept travel advisories are gated — off by default (the
        # per-country level changes are too coarse to be actionable
        # for most users). NWS weather alerts have their own path and
        # stay always-on.
        try:
            from bridge import features as _features
            travel_enabled = _features.is_enabled(
                "alerts_travel_advisory", default=False,
            )
        except Exception as e:
            # Fail-soft: if the feature registry import or read raises
            # for any reason, leave travel-advisory off — but log so
            # the silent disable is visible during debugging instead
            # of looking like a config issue.
            log.warning("alerts_travel_advisory flag check raised: %s", e)
            travel_enabled = False
        if travel_enabled:
            signals.extend(self._collect_advisory_signals(now))
        signals.extend(self._collect_nws_signals(now))

        result = self._engine.run(signals, now)
        self._store.append_event(
            "scan_ok",
            {
                "signals": len(signals),
                "alerts_fired": len(result.alerts),
                "alerts_suppressed": len(result.suppressed),
            },
        )
        # Push every freshly-fired alert through the notify dispatcher.
        # The dispatcher itself enforces severity threshold + cooldown
        # per its rules; we just hand off the event with a stable
        # dedupe_key so its own ledger can suppress repeats. Notify
        # failure is logged inside notify() and never propagates —
        # the scan result is unaffected by a flaky webhook.
        if result.alerts:
            self._emit_notify_events(result.alerts)
        return result

    def _emit_notify_events(self, alerts: list[Alert]) -> None:
        """Convert each newly-fired Alert to a NotifyEvent and dispatch.

        Severity mapping mirrors the alert level: ADVISORY/CRITICAL
        flow as WARNING/CRITICAL notifications; lower levels go as
        INFO so the dispatcher's per-rule min_severity filter can
        drop them. ``dedupe_key`` is the alert_id so successive scans
        within the cooldown window collapse to one push."""
        try:
            from bridge.notify import NotifyEvent, notify
            from bridge.notify.events import Severity
        except Exception as e:
            # If the notify module fails to import for any reason,
            # the scan path is still authoritative — we just lose
            # outbound push for this batch.
            log.warning("notify import failed: %s", e)
            return

        level_to_sev = {
            AlertLevel.INFO:     Severity.INFO,
            AlertLevel.WATCH:    Severity.INFO,
            AlertLevel.ADVISORY: Severity.WARNING,
            AlertLevel.CRITICAL: Severity.CRITICAL,
        }
        for alert in alerts:
            sev = level_to_sev.get(alert.level, Severity.INFO)
            title = f"{alert.level.name}: {alert.title}"[:200]
            bullets = " · ".join(alert.reasoning_bullets[:3])
            body = bullets if bullets else alert.title
            try:
                notify(NotifyEvent(
                    kind="alerts.fired",
                    title=title,
                    body=body,
                    severity=sev,
                    source="alerts",
                    payload={
                        "dedupe_key": alert.alert_id,
                        "category": alert.category.value,
                        "rule_id": alert.rule_id,
                    },
                    timestamp=alert.created_at,
                ))
            except Exception as e:
                log.warning("notify dispatch raised on %s: %s", alert.alert_id, e)

    # --- read accessors ------------------------------------------------

    def get_active_alerts(self, max_results: int = 10) -> list[Alert]:
        """Return alerts whose cooldown_until is still in the future.

        "Active" == within cooldown window. Once cooldown expires, an
        alert silently drops off this list (though it remains in the
        append-only log for audit). Most recent first.

        Travel-advisory alerts are *also* hidden when the
        ``alerts_travel_advisory`` feature flag is off — this keeps
        previously-fired State-Dept entries from lingering on the
        Alerts panel after the user disables that feed. The append-
        only log is untouched, so flipping the flag back on restores
        the same rows immediately (no replay needed).
        """
        try:
            from bridge import features as _features
            travel_enabled = _features.is_enabled(
                "alerts_travel_advisory", default=False,
            )
        except Exception as e:
            log.warning("alerts_travel_advisory flag check raised: %s", e)
            travel_enabled = False

        now = datetime.now(timezone.utc)
        active: list[Alert] = []
        for raw in self._store.iter_fired_alerts():
            alert = _alert_from_dict(raw)
            if alert is None:
                continue
            if alert.cooldown_until <= now:
                continue
            if (
                not travel_enabled
                and alert.category == AlertCategory.TRAVEL
            ):
                continue
            # News-watchlist alerts are scoped to EVENT matches only
            # after the R4/R5 narrowing. Filter out previously-fired
            # region/topic matches so the panel stops lingering on
            # old news-as-alerts until their cooldowns expire. Same
            # flag-flip pattern as the travel filter above: append-
            # only log is untouched, flipping behavior is read-time.
            if alert.rule_id in ("news_confirmed_watchlist", "news_provisional_watchlist"):
                wl_types = {
                    (s.payload or {}).get("watchlist_type")
                    for s in (alert.triggered_by_signals or [])
                }
                if "event" not in wl_types:
                    continue
            active.append(alert)
        active.sort(key=lambda a: (a.level, a.created_at), reverse=True)
        return active[:max_results]

    # --- signal gathering (internal) -----------------------------------

    def _collect_tracker_signals(self, now: datetime) -> list[Signal]:
        if self._tracker_service is None:
            return []
        try:
            snaps = self._tracker_service.get_trackers()
        except Exception as e:
            log.warning("tracker signal collection failed: %s", e)
            self._store.append_event(
                "collect_error", {"source": "trackers", "error": str(e)},
            )
            return []
        signals = signals_from_trackers(snaps, now=now)
        return self._annotate_threat(signals)

    def _collect_news_signals(self, now: datetime) -> list[Signal]:
        if self._news_service is None:
            return []
        try:
            articles = self._news_service.load_all_articles()
            clusters = self._news_service.load_all_clusters()
            prefs = self._news_service.get_preferences()
        except Exception as e:
            log.warning("news signal collection failed: %s", e)
            self._store.append_event(
                "collect_error", {"source": "news", "error": str(e)},
            )
            return []
        signals = signals_from_news(clusters, articles, prefs, now=now)
        return self._annotate_threat(signals)

    def _collect_advisory_signals(self, now: datetime) -> list[Signal]:
        fetcher = self._advisory_fetcher if self._advisory_fetcher is not None else self._default_advisory_fetch
        try:
            current = fetcher()
        except AdvisoryFetchError as e:
            log.warning("travel advisory fetch failed: %s", e)
            self._store.append_event(
                "collect_error", {"source": "travel_advisory", "error": str(e)},
            )
            return []
        except Exception as e:
            log.warning("travel advisory fetch crashed: %s", e)
            self._store.append_event(
                "collect_error", {"source": "travel_advisory", "error": f"unexpected: {e}"},
            )
            return []

        previous = self._store.load_advisory_state()

        # First-poll rule: no previous state -> silently baseline, emit nothing.
        if not previous:
            self._store.save_advisory_state(snapshot_to_state(current))
            self._store.append_event(
                "advisory_baseline_seeded", {"countries": len(current)},
            )
            return []

        changes = diff_advisories(current, previous)
        # Persist the updated state before returning signals so we don't
        # emit the same change on the next scan if anything downstream fails.
        self._store.save_advisory_state(snapshot_to_state(current))
        signals = signals_from_advisory_changes(changes, now=now)
        return self._annotate_threat(signals)

    def _default_advisory_fetch(self):
        """Bound-method wrapper around fetch_advisories().

        Exists as a method (rather than inlined) so tests can override
        the fetcher via the constructor's `advisory_fetcher` parameter
        without monkey-patching httpx or the module-level function.
        """
        return fetch_advisories()

    def _collect_nws_signals(self, now: datetime) -> list[Signal]:
        """Poll NWS for active alerts at the user's IP-derived coords
        and emit NWS_WEATHER_ALERT signals for any above-Minor severity.

        Returns [] silently when coords aren't available (no network
        geolocation, rate-limited ipapi, etc.) or when NWS is down —
        weather alerts are nice-to-have, not a hard dependency.
        """
        try:
            from bridge.location import user_coords
            coords = user_coords()
        except Exception as e:
            log.debug("nws: couldn't resolve user coords: %s", e)
            return []
        if coords is None:
            return []
        lat, lon = coords
        try:
            raw = fetch_nws_alerts(lat, lon)
        except Exception as e:
            log.warning("nws alerts fetch crashed: %s", e)
            self._store.append_event(
                "collect_error", {"source": "nws", "error": f"unexpected: {e}"},
            )
            return []
        significant = [a for a in raw if is_significant(a)]
        if not significant and raw:
            log.debug(
                "nws: %d alerts returned, none significant (Moderate+)",
                len(raw),
            )
        signals = signals_from_nws(significant, now=now)
        return self._annotate_threat(signals)

    # --- Phase G.a: threat annotation ----------------------------------

    def _annotate_threat(self, signals: "list[Signal]") -> "list[Signal]":
        """Stamp each signal's payload with a ``threat_score`` 0-4.

        Called from the per-kind collectors so scoring is inline with
        existing signal construction — no parallel pass over the data.
        Rules downstream read the score and decide whether to fire.
        Empty user_context produces score 0 on location-dependent
        signals; scoring is backwards-compatible (rules that ignore
        the field work unchanged).

        When the Phase G.b LLM verifier is enabled, borderline news
        scores (2 or 3) get a second opinion from Qwen with strict
        ±1 clamping and citation guards (see threat_llm.py). Other
        kinds / scores skip the verifier — rule-based scoring is
        confident enough outside the borderline band.
        """
        from bridge.alerts.proximity import compute_threat_score
        ctx = self._user_context
        for s in signals:
            try:
                score = compute_threat_score(s.payload, s.kind, ctx)
            except Exception as e:
                log.debug("threat_score failed for %s: %s", s.kind, e)
                score = 1
            s.payload["threat_score"] = score
            if (
                self._threat_verifier is not None
                and s.kind == SignalKind.NEWS_WATCHLIST_MATCH
                and 2 <= score <= 3
            ):
                try:
                    adjusted = self._threat_verifier.verify(s.payload, score, ctx)
                    s.payload["threat_score"] = adjusted
                except Exception as e:
                    # Verifier errors must never block a scan. Fall
                    # back to the rule-based score.
                    log.warning("threat verifier failed (non-fatal): %s", e)
        return signals

    # --- TTL gate ------------------------------------------------------

    def _within_ttl(self) -> bool:
        last = self._store.last_event("scan_ok")
        if not last:
            return False
        ts = last.get("ts")
        if not ts:
            return False
        try:
            last_ts = datetime.fromisoformat(ts)
        except ValueError:
            return False
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_ts
        return age < timedelta(minutes=self._scan_ttl_minutes)


# --- Alert reconstruction from dict ---------------------------------------

def _alert_from_dict(d: dict) -> Alert | None:
    """Rehydrate an Alert from its persisted JSON shape.

    Recomputes `is_active` from `cooldown_until > now()` so the field on
    the returned Alert reflects current truth (not the stale on-disk
    value from when the alert was originally written). Returns None if
    the entry is malformed (e.g. log was corrupted mid-write).
    """
    try:
        cooldown_until = _parse_dt(d["cooldown_until"])
        now = datetime.now(timezone.utc)
        return Alert(
            alert_id=d["alert_id"],
            level=AlertLevel(int(d["level"])),
            category=AlertCategory(d["category"]),
            title=d["title"],
            reasoning_bullets=list(d.get("reasoning_bullets", [])),
            triggered_by_signals=[
                Signal(
                    kind=SignalKind(s["kind"]),
                    source=s["source"],
                    payload=dict(s.get("payload", {})),
                    observed_at=_parse_dt(s["observed_at"]),
                )
                for s in d.get("triggered_by_signals", [])
            ],
            source_attribution=list(d.get("source_attribution", [])),
            affected_domains=list(d.get("affected_domains", [])),
            rule_id=d["rule_id"],
            scope_key=d["scope_key"],
            created_at=_parse_dt(d["created_at"]),
            cooldown_until=cooldown_until,
            is_active=cooldown_until > now,
        )
    except (KeyError, ValueError, TypeError) as e:
        log.debug("malformed alert entry in log: %s (%s)", d, e)
        return None


from bridge.utils import parse_iso_utc as _parse_dt  # noqa: E402


# --- default singleton ----------------------------------------------------

_default_service: AlertService | None = None


def get_default_alert_service() -> AlertService:
    global _default_service
    if _default_service is None:
        _default_service = _build_default_service()
    return _default_service


def reset_default_alert_service() -> None:
    global _default_service
    _default_service = None


def _build_threat_verifier_if_enabled(data_dir):
    """Construct the Phase G.b LLM verifier, or return None.

    Mirrors the `translator` / `keyword_learn_cfg` fail-soft pattern in
    bridge/news/service._build_default_service: feature flag gate first,
    then any missing config or import failure just leaves the verifier
    off. The rule-based score stands when this returns None.
    """
    try:
        from bridge import features as _features
        if not _features.is_enabled("alerts_threat_llm", default=False):
            return None
        from bridge.alerts.threat_llm import ThreatVerifier
        from bridge.utils import REPO_ROOT, load_config
        cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
        lcfg = cfg["llm"]
        verifier = ThreatVerifier(
            base_url=lcfg["base_url"],
            model=lcfg["model"],
            cache_path=data_dir / "threat_decisions.json",
            request_timeout=float(lcfg.get("request_timeout", 60.0)),
        )
        log.info("threat verifier enabled: model=%s", lcfg["model"])
        return verifier
    except Exception as e:
        log.warning("threat verifier disabled (build failed): %s", e)
        return None


def _build_default_service() -> AlertService:
    from bridge.news.service import get_default_service as get_news_service
    from bridge.profiles import active_profile
    from bridge.trackers.service import get_default_tracker_service

    # Phase H: alerts state (cooldowns, advisory_state, fired alerts,
    # threat_decisions) is per-profile. Two profiles on the same box
    # fire independently and carry independent cooldown timers.
    data_dir = active_profile().alerts_dir
    store = AlertStore(data_dir)
    cooldown_ledger = CooldownLedger(data_dir / "cooldowns.json")

    # Tracker + news services are the established singletons; alerts
    # just consumes them read-only.
    tracker_service = get_default_tracker_service()
    news_service = get_news_service()

    log.info(
        "built default AlertService: %d rules, tracker+news subsystems attached",
        len(DEFAULT_RULES),
    )
    return AlertService(
        store=store,
        cooldown_ledger=cooldown_ledger,
        rules=DEFAULT_RULES,
        tracker_service=tracker_service,
        news_service=news_service,
    )
