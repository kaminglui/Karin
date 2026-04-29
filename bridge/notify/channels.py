"""Channel implementations: Discord webhook + ntfy.sh push.

Both follow a tiny :class:`Channel` protocol: ``name`` (for logs) and
``send(event)``. Channels:

* Read their endpoint URL / token from environment variables only —
  never from a checked-in config. Missing env → channel is "not
  configured" and the dispatcher skips it on load.
* Treat every send as best-effort: on HTTP error, log a warning and
  swallow. The originating subsystem (alerts / trackers / news)
  cannot have its primary action blocked by a notification glitch.
* Time out fast (5 s default). A wedged channel must not stall a
  poller tick.

Adding a new channel is a three-step contract:

1. Implement ``name`` and ``send(event)`` (see :class:`Channel`).
2. Register a factory in :func:`build_enabled_channels` keyed by
   the channel name in ``config/notify.yaml``.
3. Document the env-var(s) it reads in ``config/notify.example.yaml``.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol

import httpx

from bridge.notify.events import NotifyEvent, Severity

log = logging.getLogger("bridge.notify.channels")

DEFAULT_TIMEOUT_S = 5.0


class Channel(Protocol):
    """One outbound destination."""

    name: str

    def send(self, event: NotifyEvent) -> bool:
        """Push the event. Return True on success, False on any failure
        (logged internally). Never raise."""


# --- Discord webhook -----------------------------------------------------

# Map our 3-step severity to a Discord embed color (decimal int).
_DISCORD_COLOR = {
    Severity.INFO:     0x4F46E5,   # indigo
    Severity.WARNING:  0xF59E0B,   # amber
    Severity.CRITICAL: 0xDC2626,   # red
}


class DiscordChannel:
    """Post one embed per event to a Discord channel via webhook URL.

    Env: ``KARIN_NOTIFY_DISCORD_WEBHOOK``. Webhook URL format is
    ``https://discord.com/api/webhooks/<id>/<token>`` — generate from
    Server Settings → Integrations → Webhooks. URL contains a secret;
    keep it out of source / logs (the SecretScrubFilter masks it
    automatically since the path includes ``token`` semantics).
    """

    name = "discord"
    _ENV_KEY = "KARIN_NOTIFY_DISCORD_WEBHOOK"

    def __init__(self, webhook_url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._url = webhook_url
        self._timeout = timeout_s

    @classmethod
    def from_env(cls) -> "DiscordChannel | None":
        """Construct from the secrets store (file-first, env-fallback);
        return None if neither source has the URL so the dispatcher
        can quietly skip this channel."""
        from bridge.notify.secrets import get_secret
        url = get_secret("discord_webhook")
        if not url:
            return None
        return cls(url)

    def send(self, event: NotifyEvent) -> bool:
        body = {
            "embeds": [{
                "title":  event.title[:256],
                "description": event.body[:2000],
                "color":  _DISCORD_COLOR.get(event.severity, _DISCORD_COLOR[Severity.INFO]),
                "footer": {"text": f"{event.source} · {event.kind}"[:2048]},
                "timestamp": event.timestamp.isoformat(),
            }]
        }
        try:
            r = httpx.post(self._url, json=body, timeout=self._timeout)
            if r.status_code >= 400:
                log.warning("discord notify %d: %s", r.status_code, r.text[:200])
                return False
            return True
        except httpx.HTTPError as e:
            log.warning("discord notify failed: %s", e)
            return False


# --- ntfy.sh -------------------------------------------------------------

# ntfy supports priority 1 (min) - 5 (max). Map our severity onto
# something readers actually want pushed: WARNING goes default-priority
# so it lands on the lock screen; CRITICAL goes urgent so it bypasses
# Do Not Disturb on most platforms.
_NTFY_PRIORITY = {
    Severity.INFO:     "2",   # low — silent
    Severity.WARNING:  "3",   # default — visible
    Severity.CRITICAL: "5",   # urgent — bypasses DND
}


class NtfyChannel:
    """LEGACY: Post one message per event to a ntfy.sh topic.

    Demoted from default channels because Discord covers the
    common use case and ntfy added phone-app subscription friction
    (separate app install, topic-name typing, public-topic guess
    risk). The class stays here so users who want it can re-enable
    by adding ``ntfy`` to their ``config/notify.yaml`` ``channels:``
    array — no code change needed.

    Env: ``KARIN_NOTIFY_NTFY_TOPIC`` — the FULL topic URL, e.g.
    ``https://ntfy.sh/karin-alerts-abc123``. Pick a hard-to-guess
    suffix; ntfy public topics are unauthenticated. For private
    use, self-host or use a subscription with auth token (see ntfy
    docs); this client honors the URL as-is.
    """

    name = "ntfy"
    _ENV_KEY = "KARIN_NOTIFY_NTFY_TOPIC"

    def __init__(self, topic_url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._url = topic_url
        self._timeout = timeout_s

    @classmethod
    def from_env(cls) -> "NtfyChannel | None":
        from bridge.notify.secrets import get_secret
        url = get_secret("ntfy_topic")
        if not url:
            return None
        return cls(url)

    def send(self, event: NotifyEvent) -> bool:
        # ntfy uses HTTP headers for metadata; body is plain text.
        # HTTP headers MUST be ASCII (RFC 7230) — emoji like ⏰ in
        # NotifyEvent.title would crash httpx with a UnicodeEncodeError.
        # Strip non-ASCII from the header but keep it in the body
        # (which is sent UTF-8 and renders fine on the phone).
        ascii_title = event.title.encode("ascii", "ignore").decode("ascii").strip()
        if not ascii_title:
            ascii_title = "Karin notification"
        headers = {
            "Title":    ascii_title[:200],
            "Priority": _NTFY_PRIORITY.get(event.severity, "3"),
            "Tags":     (event.source or "karin").encode("ascii", "ignore").decode("ascii") or "karin",
        }
        try:
            r = httpx.post(
                self._url,
                content=event.body.encode("utf-8"),
                headers=headers,
                timeout=self._timeout,
            )
            if r.status_code >= 400:
                log.warning("ntfy notify %d: %s", r.status_code, r.text[:200])
                return False
            return True
        except httpx.HTTPError as e:
            log.warning("ntfy notify failed: %s", e)
            return False


# --- factory registry ----------------------------------------------------

# Maps the channel name in config/notify.yaml's `channels:` list to
# the from_env factory. Keep in sync with config/notify.example.yaml.
_CHANNEL_FACTORIES: dict[str, type] = {
    "discord": DiscordChannel,
    "ntfy":    NtfyChannel,
}


def build_enabled_channels(channel_names: list[str]) -> list[Channel]:
    """Instantiate channels named in ``channel_names`` whose env vars
    are present. Channels with missing config are silently skipped
    (logged at INFO so the operator can tell).

    Unknown channel names log a warning (likely a config typo) and
    are skipped. Never raises.
    """
    out: list[Channel] = []
    for name in channel_names:
        factory_cls = _CHANNEL_FACTORIES.get(name)
        if factory_cls is None:
            log.warning("unknown notify channel %r; ignoring", name)
            continue
        instance = factory_cls.from_env()  # type: ignore[attr-defined]
        if instance is None:
            log.info(
                "notify channel %r not configured (env var unset); skipping",
                name,
            )
            continue
        out.append(instance)
    return out
