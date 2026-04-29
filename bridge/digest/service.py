"""DigestService — pure composition over news / alerts / trackers.

Design:
  - Zero LLM in the build path. The LLM only READS the finished
    `voice_line` at query time and paraphrases. All filtering, ranking,
    and state computation happens in the source subsystems' existing
    deterministic code.
  - Build is cheap: the underlying subsystems are already polled in
    the background, so DigestService.build() just READS their current
    state (no fetch), filters, and composes.
  - Persistence is one JSON file per day at
    ``data/digest/YYYY-MM-DD.json``. Re-building the same day
    overwrites the file — we always display the newest snapshot.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bridge.digest.models import (
    DigestAlertItem,
    DigestNewsItem,
    DigestSnapshot,
    DigestTrackerItem,
)
from bridge.utils import REPO_ROOT, atomic_write_text

log = logging.getLogger("bridge.digest.service")

# How much a tracker must move (abs percent) to show up in the digest.
# Below this we consider the move noise — no point cluttering the
# digest with "USD/CNY changed 0.03% today."
# Caps per section — keeps the digest scannable in one screen.
# All four below are tunable via config/tuning.yaml under `digest.*`;
# the literals are the fallback defaults.
from bridge import tuning as _tuning

_TRACKER_MIN_CHANGE_PCT = _tuning.get("digest.tracker_min_change_pct", 1.0)
_MAX_NEWS = _tuning.get("digest.max_news_items", 5)
_MAX_ALERTS = _tuning.get("digest.max_alert_items", 5)
_MAX_TRACKERS = _tuning.get("digest.max_tracker_items", 6)

# Directory where daily snapshots land. Created on first write.
_DIGEST_DIR = REPO_ROOT / "data" / "digest"


class DigestService:
    """Builds + persists daily digests. Stateless between calls."""

    def __init__(
        self,
        *,
        news_service=None,
        alert_service=None,
        tracker_service=None,
        digest_dir: Path | None = None,
    ) -> None:
        self._news_service = news_service
        self._alert_service = alert_service
        self._tracker_service = tracker_service
        self._digest_dir = digest_dir or _DIGEST_DIR

    # --- build path ------------------------------------------------------

    def build(self, *, now: datetime | None = None) -> DigestSnapshot:
        """Read current state from news/alerts/trackers, filter, rank,
        and return a DigestSnapshot. Does NOT persist — use
        :meth:`build_and_persist` for that path."""
        now = now or datetime.now(timezone.utc)
        news_items = self._collect_news(now)
        alert_items = self._collect_alerts(now)
        tracker_items = self._collect_trackers(now)
        headline = _compose_headline(news_items, alert_items, tracker_items)
        return DigestSnapshot(
            generated_at=now,
            date_key=now.strftime("%Y-%m-%d"),
            headline=headline,
            news=news_items,
            alerts=alert_items,
            trackers=tracker_items,
        )

    def build_and_persist(self, *, now: datetime | None = None) -> DigestSnapshot:
        """Build and write to ``data/digest/<date>.json``. Returns the
        same snapshot for callers that want both."""
        snap = self.build(now=now)
        self._persist(snap)
        return snap

    # --- read path -------------------------------------------------------

    def latest(self) -> DigestSnapshot | None:
        """Return the most recent on-disk digest, or None if none exist.

        Caller is responsible for deciding whether it's fresh enough
        (snapshot has `generated_at`). This path is what the HTTP
        surface + LLM tool hit — cheap and never touches the LLM.
        """
        if not self._digest_dir.exists():
            return None
        files = sorted(self._digest_dir.glob("*.json"))
        if not files:
            return None
        return self._load(files[-1])

    def for_date(self, date_key: str) -> DigestSnapshot | None:
        """Load the digest for a specific YYYY-MM-DD, or None."""
        p = self._digest_dir / f"{date_key}.json"
        if not p.exists():
            return None
        return self._load(p)

    # --- internal: collect each section ----------------------------------

    def _collect_news(self, now: datetime) -> list[DigestNewsItem]:
        if self._news_service is None:
            return []
        # Ask the existing NewsService for its top N briefs. It already
        # handles ranking (state priority + recency) and returns
        # StoryBrief objects with a ready-to-use voice_line. We just
        # filter to confirmed / provisionally confirmed and cap.
        try:
            briefs = self._news_service.get_news(
                topic=None, max_results=_MAX_NEWS * 2,
            )
        except Exception as e:
            log.warning("digest: news collection failed: %s", e)
            return []
        out: list[DigestNewsItem] = []
        for b in briefs:
            state = getattr(b, "state", "") or ""
            # Skip bare "developing" stories — too noisy for digest.
            if state not in ("confirmed", "provisionally_confirmed"):
                continue
            # StoryBrief exposes the display-safe title as `headline`
            # (see bridge/news/models.py). Historical code read `.title`
            # which silently returned "" — producing digest cards with
            # only a state pill and no text at all. Keep the old
            # attribute name as a fallback in case a model variant
            # still uses `.title`.
            headline = (
                getattr(b, "headline", None)
                or getattr(b, "title", None)
                or getattr(b, "voice_line", "")
                or ""
            )
            # StoryBrief's canonical field names (see
            # bridge/news/models.py) are `top_sources` and
            # `independent_confirmation_count`. Older digest code read
            # the plural "source_display_names" / "independent_confirmations"
            # and silently got the empty-list / zero fallback — producing
            # digest cards that claimed "0 independent sources" from
            # zero outlets. Try both shapes for forward-compat with
            # any model refactor; first present wins.
            sources = (
                list(getattr(b, "top_sources", None) or [])
                or list(getattr(b, "source_display_names", None) or [])
            )
            indep = int(
                getattr(b, "independent_confirmation_count", None)
                if getattr(b, "independent_confirmation_count", None) is not None
                else getattr(b, "independent_confirmations", 0) or 0
            )
            # latest_update_at isn't on StoryBrief — map it from the
            # underlying cluster when available, else use `now` so the
            # digest card at least has a stable timestamp.
            latest = (
                getattr(b, "latest_update_at", None)
                or getattr(b, "updated_at", None)
                or now
            )
            out.append(DigestNewsItem(
                cluster_id=getattr(b, "cluster_id", "") or "",
                state=state,
                title=headline,
                voice_line=getattr(b, "voice_line", "") or "",
                sources=sources,
                independent_confirmations=indep,
                latest_update_at=latest,
            ))
            if len(out) >= _MAX_NEWS:
                break
        return out

    def _collect_alerts(self, now: datetime) -> list[DigestAlertItem]:
        if self._alert_service is None:
            return []
        try:
            alerts = self._alert_service.get_active_alerts(max_results=_MAX_ALERTS)
        except Exception as e:
            log.warning("digest: alert collection failed: %s", e)
            return []
        out: list[DigestAlertItem] = []
        for a in alerts:
            out.append(DigestAlertItem(
                alert_id=a.alert_id,
                level=a.level.name,           # "WATCH" / "ADVISORY" / "CRITICAL"
                category=a.category.value,
                title=a.title,
                reasoning_bullets=list(a.reasoning_bullets),
                created_at=a.created_at,
                cooldown_until=a.cooldown_until,
            ))
        return out

    def _collect_trackers(self, now: datetime) -> list[DigestTrackerItem]:
        if self._tracker_service is None:
            return []
        try:
            snaps = self._tracker_service.get_trackers()
        except Exception as e:
            log.warning("digest: tracker collection failed: %s", e)
            return []
        out: list[DigestTrackerItem] = []
        for s in snaps:
            # Only surface movers — don't flood the digest with flats.
            pct = s.change_1d_pct if s.change_1d_pct is not None else None
            direction = getattr(s, "direction_label", None)
            shock = getattr(s, "shock_label", None)
            has_move = (pct is not None and abs(pct) >= _TRACKER_MIN_CHANGE_PCT)
            if not (has_move or shock):
                continue
            out.append(DigestTrackerItem(
                tracker_id=s.id,
                label=s.label,
                latest_value=s.latest_value if s.latest_value is not None else 0.0,
                change_1d_pct=pct,
                change_1w_pct=s.change_1w_pct,
                direction_label=direction,
                shock_label=shock,
            ))
        # Sort biggest-move first so the top of the tracker section is
        # the most attention-worthy row.
        out.sort(key=lambda t: abs(t.change_1d_pct or 0.0), reverse=True)
        return out[:_MAX_TRACKERS]

    # --- internal: persistence -------------------------------------------

    def _persist(self, snap: DigestSnapshot) -> None:
        self._digest_dir.mkdir(parents=True, exist_ok=True)
        p = self._digest_dir / f"{snap.date_key}.json"
        payload = _snap_to_dict(snap)
        atomic_write_text(p, json.dumps(payload, indent=2))

    def _load(self, p: Path) -> DigestSnapshot | None:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("digest: failed to load %s: %s", p, e)
            return None
        return _snap_from_dict(raw)


# --- headline composition ---------------------------------------------------


def _compose_headline(
    news: list[DigestNewsItem],
    alerts: list[DigestAlertItem],
    trackers: list[DigestTrackerItem],
) -> str:
    """Build the one-sentence summary. Deterministic, no LLM.

    Strategy: count what's in each section, pick the most
    attention-grabbing item from each, and compose a short sentence.
    Avoid numbers that the reader can already see — just headline the
    *kinds* of thing that moved.
    """
    bits: list[str] = []
    if news:
        confirmed = sum(1 for n in news if n.state == "confirmed")
        if confirmed:
            bits.append(f"{confirmed} confirmed stor{'y' if confirmed == 1 else 'ies'}")
        else:
            bits.append(f"{len(news)} developing stor{'y' if len(news) == 1 else 'ies'}")
    if alerts:
        critical = sum(1 for a in alerts if a.level == "CRITICAL")
        if critical:
            bits.append(
                f"{critical} critical alert{'s' if critical != 1 else ''}"
            )
        else:
            bits.append(
                f"{len(alerts)} active alert{'s' if len(alerts) != 1 else ''}"
            )
    if trackers:
        t = trackers[0]  # already sorted biggest-move-first
        pct = t.change_1d_pct
        if pct is not None:
            arrow = "up" if pct > 0 else "down"
            bits.append(f"{t.label} {arrow} {abs(pct):.1f}%")
    if not bits:
        return "Quiet day — nothing flagged across news, alerts, or markets."
    return "Today: " + ", ".join(bits) + "."


# --- dict <-> snapshot helpers ---------------------------------------------


def _snap_to_dict(snap: DigestSnapshot) -> dict:
    return {
        "generated_at": snap.generated_at.isoformat(),
        "date_key": snap.date_key,
        "headline": snap.headline,
        "news": [
            {
                "cluster_id": n.cluster_id,
                "state": n.state,
                "title": n.title,
                "voice_line": n.voice_line,
                "sources": n.sources,
                "independent_confirmations": n.independent_confirmations,
                "latest_update_at": n.latest_update_at.isoformat(),
            }
            for n in snap.news
        ],
        "alerts": [
            {
                "alert_id": a.alert_id,
                "level": a.level,
                "category": a.category,
                "title": a.title,
                "reasoning_bullets": a.reasoning_bullets,
                "created_at": a.created_at.isoformat(),
                "cooldown_until": a.cooldown_until.isoformat(),
            }
            for a in snap.alerts
        ],
        "trackers": [
            {
                "tracker_id": t.tracker_id,
                "label": t.label,
                "latest_value": t.latest_value,
                "change_1d_pct": t.change_1d_pct,
                "change_1w_pct": t.change_1w_pct,
                "direction_label": t.direction_label,
                "shock_label": t.shock_label,
            }
            for t in snap.trackers
        ],
    }


def _snap_from_dict(raw: dict) -> DigestSnapshot:
    def _dt(s: str) -> datetime:
        return datetime.fromisoformat(s)
    return DigestSnapshot(
        generated_at=_dt(raw["generated_at"]),
        date_key=raw["date_key"],
        headline=raw.get("headline", ""),
        news=[
            DigestNewsItem(
                cluster_id=n["cluster_id"], state=n["state"],
                title=n["title"], voice_line=n["voice_line"],
                sources=list(n.get("sources", [])),
                independent_confirmations=int(n.get("independent_confirmations", 0)),
                latest_update_at=_dt(n["latest_update_at"]),
            )
            for n in raw.get("news", [])
        ],
        alerts=[
            DigestAlertItem(
                alert_id=a["alert_id"], level=a["level"], category=a["category"],
                title=a["title"],
                reasoning_bullets=list(a.get("reasoning_bullets", [])),
                created_at=_dt(a["created_at"]),
                cooldown_until=_dt(a["cooldown_until"]),
            )
            for a in raw.get("alerts", [])
        ],
        trackers=[
            DigestTrackerItem(
                tracker_id=t["tracker_id"], label=t["label"],
                latest_value=float(t.get("latest_value") or 0.0),
                change_1d_pct=t.get("change_1d_pct"),
                change_1w_pct=t.get("change_1w_pct"),
                direction_label=t.get("direction_label"),
                shock_label=t.get("shock_label"),
            )
            for t in raw.get("trackers", [])
        ],
    )


# --- default service wiring -------------------------------------------------


_default: DigestService | None = None


def get_default_digest_service() -> DigestService:
    """Lazy-construct a DigestService wired to the default subsystem
    services. Used by the tool + API + poller."""
    global _default
    if _default is not None:
        return _default
    from bridge.alerts.service import get_default_alert_service
    from bridge.news.service import get_default_service as get_default_news_service
    from bridge.trackers.service import get_default_tracker_service
    _default = DigestService(
        news_service=get_default_news_service(),
        alert_service=get_default_alert_service(),
        tracker_service=get_default_tracker_service(),
    )
    return _default
