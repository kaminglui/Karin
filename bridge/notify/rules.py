"""Routing rules: which events go to channels, and at what severity.

The dispatcher applies these in order — first matching rule wins —
to decide whether to send an event. A rule has:

* ``kinds`` : tuple of event-kind prefixes ("alerts.", "trackers.shock", "*")
* ``min_severity`` : minimum :class:`Severity` to fire ("warning"
  drops INFO events of this kind)
* ``channels`` : list of channel names that should receive matches
  (subset of the dispatcher's enabled channels)
* ``cooldown_s`` : optional dedupe window; a second event with the
  same ``(kind, dedupe_key)`` within this many seconds is suppressed.
  ``dedupe_key`` defaults to ``payload.get("dedupe_key")`` then
  ``payload.get("id")`` — see :class:`Rule.match` below.

Defaults (when no config is present): ``alerts.fired`` at WARNING+
goes to every enabled channel, with a 30-min cooldown per
``(kind, alert_id)``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from bridge.notify.events import NotifyEvent, Severity

log = logging.getLogger("bridge.notify.rules")


@dataclass(frozen=True)
class Rule:
    """One routing rule."""

    kinds: tuple[str, ...]      # ("alerts.fired",) or ("alerts.",) for prefix
    min_severity: Severity
    channels: tuple[str, ...]
    cooldown_s: int = 0

    def matches_kind(self, kind: str) -> bool:
        for k in self.kinds:
            if k == "*" or k == kind:
                return True
            if k.endswith(".") and kind.startswith(k):
                return True
        return False


# Built-in defaults applied when notify.yaml has no `rules:` list.
# Keep narrow: only WARNING-and-up alerts go out by default; tracker
# shocks go out at INFO (any shock is itself a high-signal event);
# news transitions stay quiet by default since they're frequent.
# Default channels: Discord only. ntfy has been demoted to legacy
# (the NtfyChannel class is still in channels.py — flip it back on
# by listing "ntfy" in your config/notify.yaml channels: array).
DEFAULT_RULES: tuple[Rule, ...] = (
    Rule(
        kinds=("alerts.fired",),
        min_severity=Severity.WARNING,
        channels=("discord",),
        cooldown_s=30 * 60,
    ),
    Rule(
        kinds=("trackers.shock",),
        min_severity=Severity.INFO,
        channels=("discord",),
        cooldown_s=60 * 60,
    ),
    # User-set reminders. INFO threshold so any reminder fires; the
    # tight 60 s cooldown is just a guard against the scheduler
    # somehow ticking twice with the same row marked due (the
    # store's atomic mark_delivered already prevents this — the
    # cooldown is belt-and-braces).
    Rule(
        kinds=("reminders.fired",),
        min_severity=Severity.INFO,
        channels=("discord",),
        cooldown_s=60,
    ),
    # Calendar "meeting in N min" pings. INFO threshold to match
    # reminders (these are on-purpose heads-ups, not alarms). The
    # dedup ledger in bridge/calendar/store.py prevents re-fires
    # across polls; cooldown_s=60 is belt-and-braces.
    Rule(
        kinds=("calendar.upcoming",),
        min_severity=Severity.INFO,
        channels=("discord",),
        cooldown_s=60,
    ),
)


@dataclass
class RuleEngine:
    """Holds the rule list plus a small in-memory cooldown ledger.

    Cooldowns are intentionally per-process: a container restart
    forgets them, which is fine — we'd rather re-notify on a
    crash-loop than miss the same event forever. For longer-window
    dedupe (cross-restart), the originating subsystem already has
    its own cooldown ledger (alerts/cooldown.py, trackers' shock
    detection) and won't re-emit the same event in the first place.
    """

    rules: tuple[Rule, ...] = field(default_factory=lambda: DEFAULT_RULES)
    _last_sent: dict[tuple[str, str], float] = field(default_factory=dict)

    def routes_for(self, event: NotifyEvent) -> list[str]:
        """Return the union of channels from every matching rule that
        passes severity + cooldown. Returns ``[]`` when the event
        should be dropped entirely."""
        now = time.monotonic()
        channels: list[str] = []
        for rule in self.rules:
            if not rule.matches_kind(event.kind):
                continue
            if int(event.severity) < int(rule.min_severity):
                continue
            if rule.cooldown_s > 0:
                key = (event.kind, _dedupe_key(event))
                last = self._last_sent.get(key)
                if last is not None and (now - last) < rule.cooldown_s:
                    log.debug(
                        "notify cooldown: %s for %ds (key=%s)",
                        event.kind, rule.cooldown_s, key[1],
                    )
                    continue
                self._last_sent[key] = now
            for c in rule.channels:
                if c not in channels:
                    channels.append(c)
        return channels


def _dedupe_key(event: NotifyEvent) -> str:
    """Stable string used to debounce repeated events. Callers can
    set ``payload['dedupe_key']`` explicitly; otherwise we fall back
    to ``payload['id']`` then to a hash of the title (so two distinct
    titles never collapse, but two identical titles within the
    cooldown window do)."""
    p = event.payload or {}
    if isinstance(p.get("dedupe_key"), str):
        return p["dedupe_key"]
    if isinstance(p.get("id"), str):
        return p["id"]
    return event.title
