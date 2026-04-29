"""Singleton dispatcher: hold the channel list + rule engine, route
events through both. Subsystems call :func:`notify(event)` and don't
think about channels.

Lifetime: the dispatcher is built once on first :func:`notify` call,
caching the snapshot of features + notify.yaml at that moment. Edits
to either file take effect after :func:`reset_dispatcher` (or a
process restart).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from bridge.notify.channels import Channel, build_enabled_channels
from bridge.notify.events import NotifyEvent, Severity
from bridge.notify.rules import DEFAULT_RULES, Rule, RuleEngine

log = logging.getLogger("bridge.notify.dispatcher")


@dataclass
class Dispatcher:
    """Holds enabled channels + the rule engine. Stateful only for
    the cooldown ledger inside the engine."""

    channels: list[Channel]
    engine: RuleEngine

    def send(self, event: NotifyEvent) -> list[str]:
        """Route the event through rules → channels. Returns the
        list of channel names that ACK'd; failures are logged
        inside the channel's send() and excluded."""
        if not self.channels:
            return []
        target_names = self.engine.routes_for(event)
        if not target_names:
            return []
        delivered: list[str] = []
        for ch in self.channels:
            if ch.name not in target_names:
                continue
            try:
                ok = ch.send(event)
            except Exception as e:
                log.warning(
                    "channel %s raised on send (event=%s): %s",
                    ch.name, event.kind, e,
                )
                ok = False
            if ok:
                delivered.append(ch.name)
        log.info(
            "notify dispatched: kind=%s sev=%s delivered=%s targeted=%s",
            event.kind, event.severity.name,
            delivered, target_names,
        )
        return delivered


# --- singleton ----------------------------------------------------------

_default: Dispatcher | None = None


def _build() -> Dispatcher:
    """Construct the singleton from features.yaml + notify.yaml.

    When the ``notifications`` feature is off (default), we still
    return a Dispatcher — but with empty channels so every send is
    a no-op. Caller code stays branch-free.
    """
    from bridge import features
    if not features.is_enabled("notifications", default=False):
        log.info("notify: notifications feature OFF — every send is a no-op")
        return Dispatcher(channels=[], engine=RuleEngine())

    cfg = _load_notify_yaml()
    # Default channels: Discord only. ntfy is legacy — re-enable by
    # listing it explicitly in config/notify.yaml channels: array.
    channel_names = list(cfg.get("channels") or ["discord"])
    channels = build_enabled_channels(channel_names)
    rules = _parse_rules(cfg.get("rules") or [])
    engine = RuleEngine(rules=rules or DEFAULT_RULES)
    log.info(
        "notify dispatcher ready: %d channel(s), %d rule(s)",
        len(channels), len(engine.rules),
    )
    return Dispatcher(channels=channels, engine=engine)


def _load_notify_yaml() -> dict:
    from bridge.utils import REPO_ROOT, load_config
    path = REPO_ROOT / "config" / "notify.yaml"
    if not path.exists():
        log.debug("no notify.yaml at %s — using defaults", path)
        return {}
    try:
        data = load_config(path) or {}
        if not isinstance(data, dict):
            log.warning("%s root must be a YAML mapping; ignoring", path)
            return {}
        return data
    except Exception as e:
        log.warning("failed to parse %s: %s — using defaults", path, e)
        return {}


def _parse_rules(raw: list) -> tuple[Rule, ...]:
    """Convert YAML-side rule dicts to typed :class:`Rule` tuples.
    Malformed entries are logged and skipped (one bad rule shouldn't
    silence the whole dispatcher)."""
    parsed: list[Rule] = []
    for r in raw:
        if not isinstance(r, dict):
            log.warning("notify rule must be a mapping, got %r", type(r))
            continue
        kinds = r.get("kinds") or []
        if isinstance(kinds, str):
            kinds = [kinds]
        if not kinds:
            log.warning("notify rule missing 'kinds'; skipping: %r", r)
            continue
        sev_name = str(r.get("min_severity", "info")).upper()
        try:
            sev = Severity[sev_name]
        except KeyError:
            log.warning("notify rule unknown severity %r; defaulting to INFO", sev_name)
            sev = Severity.INFO
        chans = r.get("channels") or []
        if isinstance(chans, str):
            chans = [chans]
        cooldown = int(r.get("cooldown_s", 0))
        parsed.append(Rule(
            kinds=tuple(str(k) for k in kinds),
            min_severity=sev,
            channels=tuple(str(c) for c in chans),
            cooldown_s=cooldown,
        ))
    return tuple(parsed)


def notify(event: NotifyEvent) -> list[str]:
    """Public entry point. Send ``event`` through the dispatcher.

    Returns the list of channel names that successfully accepted
    the event (empty list when notifications are off, no rules
    matched, or every channel failed). Never raises.
    """
    global _default
    try:
        if _default is None:
            _default = _build()
        return _default.send(event)
    except Exception as e:
        # Belt-and-braces: even a build-time crash must never
        # propagate into the caller (alerts service, trackers, etc).
        log.warning("notify dispatcher crashed: %s", e)
        return []


def reset_dispatcher() -> None:
    """Drop the cached dispatcher so the next :func:`notify` call
    rebuilds from current YAML + env. Tests and a future settings-
    page reload endpoint use this."""
    global _default
    _default = None
