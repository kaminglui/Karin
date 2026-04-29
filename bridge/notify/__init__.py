"""Outbound notifications: alerts / tracker shocks / news transitions
get pushed to Discord / ntfy.sh / any configured channel.

Public API:

* :class:`NotifyEvent` — one event worth notifying about.
* :func:`notify` — dispatch one event to every enabled channel,
  filtered by per-source rules. Always fail-soft: a channel
  failure is logged, never re-raised. Returns the list of
  channel names that successfully accepted the event.

Subsystems call ``notify(NotifyEvent(...))`` after persisting the
underlying state — never inside a critical section. The dispatcher
is dispatch-fire-forget from the caller's perspective; if the
channel is slow / down, the originating action still completes.

Configuration lives in ``config/notify.yaml`` (optional; no file =
no notifications) plus ``config/features.yaml`` (top-level
``notifications`` flag, default off). Channel URLs / tokens load
from environment variables — never persisted in JSON or YAML.
"""
from __future__ import annotations

from .dispatcher import notify, reset_dispatcher
from .events import NotifyEvent

__all__ = ["NotifyEvent", "notify", "reset_dispatcher"]
