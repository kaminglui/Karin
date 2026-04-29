"""Shared HTTP client factory for outbound calls.

Most subsystems need an ``httpx.Client`` configured with a sensible
timeout and a Karin-identifying User-Agent. Building one inline at
each call site led to drift: every site picked its own timeout,
some forgot the User-Agent entirely, and a few used the bare
``httpx`` default (which omits a UA — some upstreams treat that as
abuse).

This module provides one factory so all "generic outbound HTTP" calls
share defaults. Specialized callers that need their own User-Agent
(NWS API requires a TOS-conformant UA; news/fetch has feed-specific
header negotiation) keep building their own clients — they pass
``user_agent=`` to override.

Example::

    from bridge._http import make_client

    with make_client(timeout=8.0) as client:
        r = client.get("https://api.example.com/v1/resource")
        r.raise_for_status()
        return r.json()

Caller still owns error handling — every site has different needs
(some return None on failure, some raise, some log and continue), so
the factory deliberately stops short of a "do the request for me"
wrapper.
"""
from __future__ import annotations

from typing import Any

import httpx


KARIN_USER_AGENT = (
    "Karin/1.0 (https://github.com/kaminglui/Karin; "
    "kaminglui+karin@users.noreply.github.com)"
)


def make_client(
    *,
    timeout: float = 10.0,
    user_agent: str | None = None,
    follow_redirects: bool = True,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> httpx.Client:
    """Return a configured ``httpx.Client`` ready for use as a context
    manager.

    Args:
        timeout: Per-request timeout in seconds. Defaults to 10s — long
            enough for slow public APIs (BLS, World Bank) on a Tailscale
            link, short enough to not pile up on a hung remote.
        user_agent: Override the default Karin UA. Pass ``None`` to use
            the package default; pass a different string for upstreams
            that require a specific UA (e.g. NWS, scraping endpoints).
            Pass an empty string to omit the header entirely.
        follow_redirects: 30x handling. Default on; httpx defaults off.
        headers: Extra headers to merge in. ``User-Agent`` here loses to
            ``user_agent``.
        **extra: Forwarded to ``httpx.Client``. Use sparingly — most
            knobs the caller wants (verify, mounts, transport) flow
            through here.
    """
    final_headers: dict[str, str] = {}
    if user_agent is None:
        final_headers["User-Agent"] = KARIN_USER_AGENT
    elif user_agent != "":
        final_headers["User-Agent"] = user_agent
    if headers:
        for k, v in headers.items():
            if k.lower() != "user-agent" or "User-Agent" not in final_headers:
                final_headers[k] = v
    return httpx.Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers=final_headers,
        **extra,
    )
