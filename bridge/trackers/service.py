"""TrackerService: public API for the tracker subsystem.

Four entry points:
  - get_tracker(id)          one snapshot (TTL-gated refresh first)
  - get_trackers(ids=None)   many snapshots (TTL-gated refresh per id)
  - refresh_one(id, force)   explicit fetch; returns True if HTTP ran
  - refresh_all(force)       explicit fetch-all

TTL gate is per-cadence: daily series are rechecked at most every 12h,
monthly at most every 24h. This is separate from each series'
`stale_after_hours` — which drives the "data looks too old" flag on the
read-side snapshot. They answer different questions:
  - TTL: "should we even hit the network right now?"
  - stale_after_hours: "is the data we already have too old to trust?"

Deltas (1d/1w/1m) are computed at read time against persisted history
with tolerance windows. Monthly cadence forces 1d delta to None even
when a prior reading is coincidentally nearby (spec decision).

The service is stateless between calls — all truth lives in the ledger,
so restart is safe.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import httpx

from bridge._http import make_client

from bridge.trackers.fetch import FETCH_TIMEOUT, FetchError
from bridge.trackers.fetch import fetch as run_fetch
from bridge.trackers.fetch import fetch_history as run_fetch_history
from bridge.trackers.models import (
    TrackerConfig,
    TrackerReading,
    TrackerRecord,
    TrackerSnapshot,
)
from bridge.trackers.store import (
    TrackerStore,
    add_or_replace_reading,
    prune_history,
)

log = logging.getLogger("bridge.trackers.service")

# Per-cadence TTL (minutes). Used by refresh_one/refresh_all to decide
# whether to hit the network. Defaults assume that neither FX nor gold
# move faster than once per day at the source (both are daily-close).
_TTL_MINUTES_BY_CADENCE: dict[str, int] = {
    "daily": 12 * 60,     # 12h — refetch at most twice per day
    "weekly": 24 * 60,    # 24h — EIA retail gas posts once a week (Monday PM)
    "monthly": 24 * 60,   # 24h — CPI releases at most monthly anyway
}
_DEFAULT_TTL_MINUTES = 12 * 60

# --- Phase 5.2 derived-label calculation constants -------------------------
#
# All label computations are pure: they take a list[TrackerReading] (or
# derived pct-changes) and return a string or None. No side effects, no
# clock dependency, no service state. Unit-testable in isolation.

# Direction: label a change as up/down/flat using a volatility-aware
# threshold. Need ≥10 observed daily changes to estimate sigma meaningfully;
# below that, fall back to a fixed 0.25% threshold (conservative floor).
_DIRECTION_MIN_POINTS_FOR_SIGMA = 10
_DIRECTION_FALLBACK_PCT = 0.25
_DIRECTION_K = 1.0
_DIRECTION_VOLATILITY_LOOKBACK = 20   # last N pct-changes feeding sigma

# Movement: compare recent volatility to a lagged baseline. The two
# windows DO NOT OVERLAP — recent is the tail; baseline is the chunk
# immediately before it. This keeps today's moves from being compared
# against themselves.
_MOVEMENT_RECENT_WINDOW = 5
_MOVEMENT_BASELINE_WINDOW = 20
_MOVEMENT_STABLE_RATIO = 0.75
_MOVEMENT_VOLATILE_RATIO = 1.5

# Shock: a single-day move past K sigmas of a baseline that EXCLUDES
# the last few observations. The lag prevents a new shock from being
# suppressed because yesterday's shock already inflated the stdev.
_SHOCK_K = 3.0
_SHOCK_EXCLUDE_RECENT = 5
_SHOCK_MIN_BASELINE_POINTS = 10


def _pct_changes(history: list[TrackerReading]) -> list[float]:
    """Consecutive percent changes between adjacent readings, in pp units.

    For N readings, returns up to N-1 values. Pairs where the earlier
    value is zero are skipped (avoids div-by-zero; zero-baselined
    "infinite % change" is not meaningful for any of our series).
    """
    out: list[float] = []
    for i in range(1, len(history)):
        prev = history[i - 1].value
        curr = history[i].value
        if prev == 0:
            continue
        out.append((curr - prev) / prev * 100.0)
    return out


def _stddev(values: list[float]) -> float | None:
    """Sample standard deviation. None when n < 2 (undefined)."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return var ** 0.5


def _compute_direction(
    change_pct: float | None,
    volatility_changes: list[float],
    lookback_days: int,
) -> str | None:
    """Label a change up/down/flat using a volatility-aware threshold.

    threshold = K * sigma(volatility_changes) * sqrt(lookback_days)

    The sqrt scaling assumes daily changes are ~i.i.d. so a 7-day total
    has ~sqrt(7) the stdev of a single day — approximate but good
    enough to keep 1w direction from flipping on noise.

    Falls back to a fixed 0.25% threshold when history is too short
    to estimate sigma. Returns None if change_pct is None.
    """
    if change_pct is None:
        return None
    if len(volatility_changes) >= _DIRECTION_MIN_POINTS_FOR_SIGMA:
        sigma = _stddev(volatility_changes)
    else:
        sigma = None
    if sigma is None or sigma == 0.0:
        threshold = _DIRECTION_FALLBACK_PCT
    else:
        threshold = _DIRECTION_K * sigma * (lookback_days ** 0.5)
    if abs(change_pct) <= threshold:
        return "flat"
    return "up" if change_pct > 0 else "down"


def _compute_movement(history: list[TrackerReading]) -> str | None:
    """Label the series as stable / moving / volatile.

    Compares sigma(recent 5 changes) to sigma(baseline 20 changes just
    before those). Non-overlapping by construction. Returns None when
    we don't have at least 25 consecutive pct-changes (~26 readings).
    """
    changes = _pct_changes(history)
    need = _MOVEMENT_RECENT_WINDOW + _MOVEMENT_BASELINE_WINDOW
    if len(changes) < need:
        return None
    recent = changes[-_MOVEMENT_RECENT_WINDOW:]
    baseline = changes[-need:-_MOVEMENT_RECENT_WINDOW]
    sig_recent = _stddev(recent)
    sig_baseline = _stddev(baseline)
    if sig_recent is None or sig_baseline is None or sig_baseline == 0.0:
        return None
    ratio = sig_recent / sig_baseline
    if ratio < _MOVEMENT_STABLE_RATIO:
        return "stable"
    if ratio >= _MOVEMENT_VOLATILE_RATIO:
        return "volatile"
    return "moving"


def _compute_shock(
    change_1d_pct: float | None,
    history: list[TrackerReading],
) -> str | None:
    """Label a 1d move as a shock when it exceeds K sigmas of a lagged baseline.

    Baseline excludes the last _SHOCK_EXCLUDE_RECENT changes so a shock
    yesterday doesn't artificially raise today's detection threshold.
    Returns "surging" (positive) / "plunging" (negative) / None.
    """
    if change_1d_pct is None:
        return None
    changes = _pct_changes(history)
    if len(changes) > _SHOCK_EXCLUDE_RECENT:
        baseline = changes[:-_SHOCK_EXCLUDE_RECENT]
    else:
        baseline = []
    if len(baseline) < _SHOCK_MIN_BASELINE_POINTS:
        return None
    sigma = _stddev(baseline)
    if sigma is None or sigma == 0.0:
        return None
    if abs(change_1d_pct) <= _SHOCK_K * sigma:
        return None
    return "surging" if change_1d_pct > 0 else "plunging"


class TrackerService:
    def __init__(
        self,
        *,
        store: TrackerStore,
        configs: list[TrackerConfig],
        preferences: "TrackerPreferences | None" = None,
    ) -> None:
        from bridge.trackers.preferences import (
            TrackerPreferences,
            resolve_label,
            resolve_params,
        )
        self._store = store
        # User-facing ordering. Disabled preferences short-circuit the
        # sort to config-file order (Phase-5 behavior).
        self._preferences = preferences or TrackerPreferences(enabled=False)
        # Resolve ${home_state_padd} / ${home_state} placeholders in
        # config params + labels once at construction time. Non-
        # templated configs pass through unchanged. Resolving here
        # (rather than per-fetch) keeps the fetcher layer unchanged —
        # it only sees concrete literal values.
        resolved: list[TrackerConfig] = []
        for cfg in configs:
            resolved.append(TrackerConfig(
                id=cfg.id,
                label=resolve_label(cfg.label, self._preferences),
                category=cfg.category,
                source=cfg.source,
                params=resolve_params(cfg.params, self._preferences),
                cadence=cfg.cadence,
                stale_after_hours=cfg.stale_after_hours,
                history_days=cfg.history_days,
                enabled=cfg.enabled,
            ))
        self._configs = resolved
        self._configs_by_id = {c.id: c for c in resolved}

    # --- read paths ------------------------------------------------------

    def get_tracker(
        self, tracker_id: str, *, fetch: bool = True,
    ) -> TrackerSnapshot | None:
        """Return a snapshot for one tracker. TTL-gated refresh first.

        Returns None if the id is unknown or disabled. Returns a snapshot
        with latest_value=None and note="no history" if we've never been
        able to fetch a value (e.g. first call failed).

        ``fetch`` gates whether this call may hit upstream data sources.
        User-facing paths (chat tools, panel APIs) pass ``fetch=False``
        so the hourly tracker poller is the sole origin of HTTP — the
        BLS / EIA / Stooq / Frankfurter keys stay under their free-tier
        daily ceilings regardless of how often the user pokes the UI.
        """
        from bridge.trackers.preferences import is_tracker_visible

        cfg = self._configs_by_id.get(tracker_id)
        if cfg is None:
            return None
        if not is_tracker_visible(cfg.id, cfg.category, cfg.enabled, self._preferences):
            return None
        if fetch:
            self.refresh_one(tracker_id, force=False)
        records = self._store.load()
        record = records.get(tracker_id)
        if record is None:
            return self._empty_snapshot(cfg, note="no history")
        return self._build_snapshot(cfg, record)

    def get_trackers(
        self, ids: list[str] | None = None, *, fetch: bool = True,
    ) -> list[TrackerSnapshot]:
        """Return snapshots for all (or a subset of) enabled trackers.

        Order: preserves config order when ids is None; preserves the
        caller's order when ids is provided. Unknown/disabled ids are
        skipped silently — callers wanting strict matching should use
        get_tracker() per id and check for None.

        ``fetch`` — same semantics as :meth:`get_tracker`: user-facing
        callers should pass False so only the background poller hits
        the network.
        """
        from bridge.trackers.preferences import is_tracker_visible

        if ids is None:
            # Visibility = config.enabled overlaid with user prefs (per-
            # tracker override + disabled_categories). Lets users flip
            # e.g. crypto on without editing trackers.json.
            target_ids = [
                c.id for c in self._configs
                if is_tracker_visible(c.id, c.category, c.enabled, self._preferences)
            ]
        else:
            target_ids = list(ids)
        if fetch:
            for tid in target_ids:
                self.refresh_one(tid, force=False)
        records = self._store.load()
        out: list[TrackerSnapshot] = []
        for tid in target_ids:
            cfg = self._configs_by_id.get(tid)
            if cfg is None:
                continue
            if not is_tracker_visible(cfg.id, cfg.category, cfg.enabled, self._preferences):
                continue
            record = records.get(tid)
            if record is None:
                out.append(self._empty_snapshot(cfg, note="no history"))
            else:
                out.append(self._build_snapshot(cfg, record))

        # Apply user-preference ordering when enabled. The sort is
        # (category-priority, in-category-priority, original-index) so
        # ties inside a preference bucket fall back to config order —
        # users never see "the same list shuffled" between refreshes.
        if self._preferences.enabled:
            from bridge.trackers.preferences import (
                category_sort_key, tracker_sort_key,
            )
            original_idx = {s.id: i for i, s in enumerate(out)}
            prefs = self._preferences
            out.sort(key=lambda s: (
                category_sort_key(s.category, prefs),
                tracker_sort_key(s.id, s.category, prefs),
                original_idx[s.id],
            ))
        return out

    # --- refresh paths ---------------------------------------------------

    def refresh_one(self, tracker_id: str, *, force: bool = False) -> bool:
        """Fetch one tracker's latest reading. TTL-gated unless force=True.

        Returns True if the network was actually hit. Errors are recorded
        in TrackerRecord.last_fetch_error and persisted; they don't
        propagate to the caller. Rationale: get_tracker should not raise
        just because an upstream feed is temporarily down.
        """
        from bridge.trackers.preferences import is_tracker_visible

        cfg = self._configs_by_id.get(tracker_id)
        if cfg is None:
            return False
        # Respect both config + user prefs so a user-disabled tracker
        # doesn't keep hitting the network via the background poller.
        if not is_tracker_visible(cfg.id, cfg.category, cfg.enabled, self._preferences):
            return False

        records = self._store.load()
        record = records.get(tracker_id)

        if not force and record is not None and self._within_ttl(cfg, record):
            return False

        # Backfill path: if the record is brand-new (no history) or has
        # only the very first reading, try to pull a block of history so
        # deltas and Phase 5.2 labels become usable immediately. If the
        # source doesn't support history (returns []) or the history
        # fetch itself errors, silently fall through to the single-reading
        # fetch below — that preserves original behavior as a safety net.
        needs_backfill = record is None or len(record.history) <= 1
        readings: list = []
        if needs_backfill:
            try:
                with make_client(timeout=FETCH_TIMEOUT) as client:
                    readings = run_fetch_history(cfg, client)
            except FetchError as e:
                log.warning("tracker %s history backfill failed: %s", cfg.id, e)
            except Exception as e:
                log.warning("tracker %s history backfill crashed: %s", cfg.id, e)

        if not readings:
            # Normal single-reading path (no backfill attempted, backfill
            # returned empty, or backfill failed). Preserves original flow.
            try:
                with make_client(timeout=FETCH_TIMEOUT) as client:
                    readings = [run_fetch(cfg, client)]
            except FetchError as e:
                self._record_fetch_error(records, cfg, record, str(e))
                return False
            except Exception as e:
                log.error("tracker %s unexpected fetcher error: %s", cfg.id, e)
                self._record_fetch_error(records, cfg, record, f"unexpected: {e}")
                return False

        if record is None:
            record = TrackerRecord(
                id=cfg.id, label=cfg.label, category=cfg.category, history=[],
            )
        for reading in readings:
            add_or_replace_reading(record, reading)
        prune_history(record, cfg.history_days)
        record.last_fetched_at = datetime.now(timezone.utc)
        record.last_fetch_error = None
        records[cfg.id] = record
        self._store.save(records)
        latest = readings[-1]
        self._store.append_event(
            "tracker_fetch_ok",
            {
                "id": cfg.id,
                "value": latest.value,
                # reading_ts (not "ts") to avoid clobbering the event's
                # own ts — append_event already stamps entry["ts"] with
                # wall-clock time, and entry.update(data) would overwrite
                # it if we reused the key.
                "reading_ts": latest.timestamp.isoformat(),
                "readings_added": len(readings),
            },
        )
        # Push a notification when this refresh produced a shock label.
        # Build the snapshot so we get the same shock_label / 1d-pct
        # the user sees in the panel — keeping the notify text in
        # sync with the displayed value. The dispatcher's per-rule
        # cooldown collapses repeated shocks on the same id.
        try:
            snap = self._build_snapshot(cfg, record)
            if snap.shock_label in ("surging", "plunging"):
                self._emit_shock_notification(snap)
        except Exception as e:
            log.warning("tracker shock notify check failed for %s: %s", cfg.id, e)
        return True

    def _emit_shock_notification(self, snap: TrackerSnapshot) -> None:
        """Send a `trackers.shock` NotifyEvent for a tracker whose
        latest reading triggered the shock label. Severity is WARNING
        — a 3-sigma 1d move is interesting but not safety-critical."""
        try:
            from bridge.notify import NotifyEvent, notify
            from bridge.notify.events import Severity
        except Exception as e:
            log.warning("notify import failed: %s", e)
            return
        direction = "up" if snap.shock_label == "surging" else "down"
        pct = snap.change_1d_pct
        pct_str = f"{pct:+.2f}%" if pct is not None else "n/a"
        try:
            notify(NotifyEvent(
                kind="trackers.shock",
                title=f"{snap.label}: {snap.shock_label} ({pct_str})",
                body=(
                    f"{snap.label} 1-day move: {pct_str} ({direction}). "
                    f"Latest value: {snap.latest_value}."
                ),
                severity=Severity.WARNING,
                source="trackers",
                payload={
                    "dedupe_key": snap.id,
                    "shock_label": snap.shock_label,
                    "change_1d_pct": pct,
                },
            ))
        except Exception as e:
            log.warning("notify dispatch raised on %s shock: %s", snap.id, e)

    def refresh_all(self, *, force: bool = False) -> dict[str, bool]:
        """Refresh every enabled tracker. Returns {id: ran_network}.

        One tracker failing does not stop others — each fetch is
        isolated by per-call error handling in refresh_one.
        """
        from bridge.trackers.preferences import is_tracker_visible
        return {
            c.id: self.refresh_one(c.id, force=force)
            for c in self._configs
            if is_tracker_visible(c.id, c.category, c.enabled, self._preferences)
        }

    # --- internals -------------------------------------------------------

    def _within_ttl(self, cfg: TrackerConfig, record: TrackerRecord) -> bool:
        if record.last_fetched_at is None:
            return False
        ttl_min = _TTL_MINUTES_BY_CADENCE.get(cfg.cadence, _DEFAULT_TTL_MINUTES)
        age = datetime.now(timezone.utc) - record.last_fetched_at
        return age < timedelta(minutes=ttl_min)

    def _record_fetch_error(
        self,
        records: dict[str, TrackerRecord],
        cfg: TrackerConfig,
        record: TrackerRecord | None,
        message: str,
    ) -> None:
        log.warning("tracker %s fetch error: %s", cfg.id, message)
        if record is None:
            record = TrackerRecord(
                id=cfg.id, label=cfg.label, category=cfg.category, history=[],
            )
        record.last_fetched_at = datetime.now(timezone.utc)
        record.last_fetch_error = message
        records[cfg.id] = record
        self._store.save(records)
        self._store.append_event(
            "tracker_fetch_error", {"id": cfg.id, "error": message},
        )

    def _build_snapshot(
        self, cfg: TrackerConfig, record: TrackerRecord,
    ) -> TrackerSnapshot:
        if not record.history:
            return self._empty_snapshot(
                cfg, note=record.last_fetch_error or "no history",
            )
        latest = record.history[-1]  # history is sorted ascending
        now = datetime.now(timezone.utc)
        is_stale = (now - latest.timestamp) > timedelta(hours=cfg.stale_after_hours)

        d1, d1p = self._delta(record.history, latest, days=1, tolerance_hours=36)
        d7, d7p = self._delta(record.history, latest, days=7, tolerance_hours=48)
        d30, d30p = self._delta(record.history, latest, days=30, tolerance_hours=24 * 7)

        # Monthly cadence: 1d delta is not meaningful even if a coincidental
        # prior reading falls in the window (e.g. mid-month refresh).
        if cfg.cadence == "monthly":
            d1 = None
            d1p = None

        # Phase 5.2 derived labels. Daily-cadence only — monthly CPI
        # has no business carrying direction_1d / shock_label.
        direction_1d: str | None = None
        direction_1w: str | None = None
        movement_label: str | None = None
        shock_label: str | None = None
        if cfg.cadence != "monthly":
            all_changes = _pct_changes(record.history)
            volatility_window = all_changes[-_DIRECTION_VOLATILITY_LOOKBACK:]
            direction_1d = _compute_direction(d1p, volatility_window, lookback_days=1)
            direction_1w = _compute_direction(d7p, volatility_window, lookback_days=7)
            movement_label = _compute_movement(record.history)
            shock_label = _compute_shock(d1p, record.history)

        return TrackerSnapshot(
            id=cfg.id,
            label=cfg.label,
            category=cfg.category,
            latest_value=latest.value,
            latest_timestamp=latest.timestamp,
            change_1d=d1,
            change_1d_pct=d1p,
            change_1w=d7,
            change_1w_pct=d7p,
            change_1m=d30,
            change_1m_pct=d30p,
            is_stale=is_stale,
            note=record.last_fetch_error or "",
            direction_1d=direction_1d,
            direction_1w=direction_1w,
            movement_label=movement_label,
            shock_label=shock_label,
        )

    def _empty_snapshot(self, cfg: TrackerConfig, *, note: str) -> TrackerSnapshot:
        return TrackerSnapshot(
            id=cfg.id, label=cfg.label, category=cfg.category,
            latest_value=None, latest_timestamp=None,
            change_1d=None, change_1d_pct=None,
            change_1w=None, change_1w_pct=None,
            change_1m=None, change_1m_pct=None,
            is_stale=True, note=note,
        )

    @staticmethod
    def _delta(
        history: list[TrackerReading],
        latest: TrackerReading,
        *,
        days: int,
        tolerance_hours: int,
    ) -> tuple[float | None, float | None]:
        """Find reading closest to (latest - days) within ±tolerance_hours.

        Returns (absolute_delta, percentage_delta). Both None if no
        history point falls inside the window.

        Rounding: absolute to 6 decimal places (preserves FX precision
        without drowning in noise), percentage to 2 (suitable for voice
        output like "down 0.34 percent").
        """
        target = latest.timestamp - timedelta(days=days)
        tolerance = timedelta(hours=tolerance_hours)
        best: TrackerReading | None = None
        best_dist: timedelta | None = None
        for r in history:
            if r.timestamp == latest.timestamp:
                continue
            dist = abs(r.timestamp - target)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best = r
                best_dist = dist
        if best is None:
            return None, None
        abs_delta = latest.value - best.value
        pct_delta = (abs_delta / best.value) * 100.0 if best.value != 0 else None
        pct_rounded = round(pct_delta, 2) if pct_delta is not None else None
        return round(abs_delta, 6), pct_rounded


# --- default service singleton --------------------------------------------

_default_service: TrackerService | None = None


def get_default_tracker_service() -> TrackerService:
    """Lazily construct a TrackerService from the repo's config files.

    Prefers bridge/trackers/config/trackers.json (user-curated) and
    falls back to trackers.example.json for first-run development.
    Caches on first call.
    """
    global _default_service
    if _default_service is None:
        _default_service = _build_default_service()
    return _default_service


def reset_default_tracker_service() -> None:
    """Clear the singleton. Tests only."""
    global _default_service
    _default_service = None


def _build_default_service() -> TrackerService:
    from bridge.utils import REPO_ROOT
    base = REPO_ROOT / "bridge" / "trackers"
    cfg_path = base / "config" / "trackers.json"
    if not cfg_path.exists():
        cfg_path = base / "config" / "trackers.example.json"
        log.info("trackers.json not found; using example config at %s", cfg_path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    configs: list[TrackerConfig] = []
    for t in raw.get("trackers", []):
        configs.append(TrackerConfig(
            id=t["id"],
            label=t["label"],
            category=t["category"],
            source=t["source"],
            params=t.get("params", {}),
            cadence=t["cadence"],
            stale_after_hours=int(t["stale_after_hours"]),
            history_days=int(t["history_days"]),
            enabled=bool(t.get("enabled", True)),
        ))
    # Market snapshots (trackers.json, events.jsonl) are GLOBAL — the
    # USD/CNY rate is the same for every profile, no point duplicating.
    store = TrackerStore(base / "data")
    # Phase H: tracker preferences (ordering, enable/disable) are
    # per-profile. Resolution order:
    #   1. active profile's trackers/tracker_preferences.json
    #   2. legacy data/trackers/tracker_preferences.json (pre-H layout)
    #   3. bridge/trackers/config/tracker_preferences.json (hand-edited)
    # The legacy fallback keeps the UI usable during the migration
    # window before the boot-time runner relocates files.
    from bridge.profiles import active_profile
    from bridge.trackers.preferences import load_tracker_preferences
    from bridge.utils import REPO_ROOT as _REPO_ROOT
    _profile_prefs = active_profile().trackers_dir / "tracker_preferences.json"
    _legacy_writable = _REPO_ROOT / "data" / "trackers" / "tracker_preferences.json"
    _legacy_cfg = base / "config" / "tracker_preferences.json"
    if _profile_prefs.exists():
        prefs_path = _profile_prefs
    elif _legacy_writable.exists():
        prefs_path = _legacy_writable
    else:
        prefs_path = _legacy_cfg
    try:
        preferences = load_tracker_preferences(prefs_path)
    except Exception as e:
        log.warning("tracker_preferences.json failed to load: %s — ignoring", e)
        from bridge.trackers.preferences import TrackerPreferences
        preferences = TrackerPreferences(enabled=False)
    log.info(
        "built default TrackerService: %d trackers, prefs=%s",
        len(configs), "on" if preferences.enabled else "off",
    )
    return TrackerService(
        store=store, configs=configs, preferences=preferences,
    )
