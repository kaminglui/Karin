"""Alert -> short human-readable string. Pure presentation.

Deliberately minimal: one line per alert with a level tag + title.
Reasoning bullets are available on the structured Alert for deeper
inspection, but the voice-path output stays compact.

The level-tag convention ([INFO] / [WATCH] / [ADVISORY] / [CRITICAL])
is stable and deterministic so Karin's LLM can pattern-match on it
if we ever want different prosody per level.
"""
from __future__ import annotations

from bridge.alerts.models import Alert, AlertLevel

_LEVEL_TAG = {
    AlertLevel.INFO: "INFO",
    AlertLevel.WATCH: "WATCH",
    AlertLevel.ADVISORY: "ADVISORY",
    AlertLevel.CRITICAL: "CRITICAL",
}


def format_alert_line(alert: Alert) -> str:
    """Short one-liner: "[LEVEL] Title"."""
    tag = _LEVEL_TAG.get(alert.level, str(alert.level.name))
    return f"[{tag}] {alert.title}"


def format_alerts_voice(alerts: list[Alert]) -> str:
    """Multi-alert output, one per line. Neutral message when empty."""
    if not alerts:
        return "No active alerts."
    return "\n".join(format_alert_line(a) for a in alerts)


def format_alert_detail(alert: Alert) -> str:
    """Verbose form with reasoning bullets. Used for inspection, not voice."""
    lines = [format_alert_line(alert)]
    for bullet in alert.reasoning_bullets:
        lines.append(f"  - {bullet}")
    return "\n".join(lines)
