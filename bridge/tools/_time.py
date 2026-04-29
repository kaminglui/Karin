"""Time tool."""
from __future__ import annotations

import datetime
import logging

log = logging.getLogger("bridge.tools")


def _get_time(timezone: str | None = None) -> str:
    """Return the current local date/time.

    Default behavior: resolve the user's timezone from their IP via
    ipapi.co (cached in bridge.location) so the reply reads in their
    local clock. When an explicit ``timezone`` arg is passed, format
    in THAT zone and append ``(UTC±HH:MM)`` so the offset is obvious.

    Fails soft: if IP lookup can't find a zone, fall back to server
    local time (historical behavior).
    """
    from zoneinfo import ZoneInfo

    try:
        explicit = bool(timezone and timezone.strip())
        if explicit:
            zone = ZoneInfo(timezone.strip())
            now = datetime.datetime.now(zone)
            # When the user asked for a specific zone, prepending the
            # UTC offset removes ambiguity (EST vs EDT, JST vs KST etc.)
            offset_hm = now.strftime("%z")  # e.g. "-0400"
            if offset_hm:
                bracket = f" (UTC{offset_hm[:3]}:{offset_hm[3:]})"
            else:
                bracket = ""
            return now.strftime("%A, %B %d, %Y at %I:%M %p %Z") + bracket

        # Default path: try the user's IP-derived zone first, server
        # local time as last resort.
        from bridge.location import user_timezone
        iana, _offset = user_timezone()
        if iana:
            try:
                zone = ZoneInfo(iana)
                now = datetime.datetime.now(zone)
                return now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
            except Exception as e:
                log.debug("ZoneInfo(%r) failed: %s", iana, e)
        # Fallback: server-local time
        now = datetime.datetime.now().astimezone()
        return now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception as e:
        return f"Error getting time: {e}"
