"""Minimal iCalendar (.ics) fetch + parse.

Rolling our own tiny parser instead of pulling the ``icalendar``
library as a dep — the 90% case (DTSTART + SUMMARY + UID on
non-recurring VEVENTs) fits in ~60 lines.

Supported:
* UTC datetimes (``DTSTART:20260415T170000Z``).
* Local datetimes with a TZID parameter. TZID resolution uses
  :mod:`zoneinfo` (stdlib; falls back to UTC on missing zone).
* All-day events (``DTSTART;VALUE=DATE:20260415``) → 00:00 UTC on
  the given date.

Not supported (yet — flagged but not fatal):
* RRULE recurrence expansion. VEVENTs with RRULE are skipped with
  a one-time log.
* EXDATE / RDATE overrides, VTODO, VJOURNAL, VTIMEZONE block
  parsing (we use zoneinfo's IANA names instead of defining our
  own VTIMEZONE→rule mapping).
* Encrypted feeds / HTTP auth (plain GET only — a secret URL is
  auth-by-obscurity, matching the ntfy pattern).

Failure modes all return an empty list with a logged warning,
never raise. The poller's fail-soft contract depends on this.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterator

import httpx

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:   # pragma: no cover — Python 3.8 fallback, unlikely here
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment,misc]

log = logging.getLogger("bridge.calendar.fetch")

DEFAULT_TIMEOUT_S = 10.0

# Logged once per process for RRULE events we skip, so a recurring
# meeting doesn't spam the log every poll cycle.
_rrule_warned_uids: set[str] = set()


@dataclass(frozen=True)
class CalendarEvent:
    """One non-recurring VEVENT we'll notify on."""

    uid: str                 # stable event identifier from the feed
    start_utc: datetime      # timezone-aware UTC
    summary: str             # event title
    calendar_label: str = "" # human-readable calendar nickname (from config)

    def dedupe_key(self) -> str:
        """Store/dispatcher key: uid + start keeps a moved instance
        of the same meeting from being treated as a dup of the old
        slot."""
        return f"{self.uid}|{self.start_utc.isoformat()}"


# --- public fetch helper -------------------------------------------------

def fetch_events(
    url: str,
    *,
    label: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> list[CalendarEvent]:
    """Download + parse ``url``. Returns the list of parseable
    non-recurring VEVENTs. All failures (network, non-200, malformed
    body) return ``[]`` + a logged warning — never raises."""
    try:
        if client is None:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.get(url, follow_redirects=True)
        else:
            resp = client.get(url, follow_redirects=True)
    except httpx.HTTPError as e:
        log.warning("calendar fetch %r failed: %s", _mask(url), e)
        return []

    if resp.status_code != 200:
        log.warning(
            "calendar %r returned HTTP %d", _mask(url), resp.status_code,
        )
        return []

    try:
        return list(_parse_ics(resp.text, label=label))
    except Exception as e:
        log.warning("calendar %r parse raised: %s", _mask(url), e)
        return []


# --- parser ---------------------------------------------------------------

_LINE_RE = re.compile(r"^([A-Za-z][\w-]*)(;[^:]*)?:(.*)$")


def _unfold(body: str) -> list[str]:
    """RFC 5545 line unfolding: lines starting with a space/tab are
    continuations of the previous logical line. Returns a list of
    logical lines with folding undone."""
    out: list[str] = []
    for raw in body.splitlines():
        if not raw:
            continue
        if raw[0] in (" ", "\t") and out:
            out[-1] += raw[1:]
        else:
            out.append(raw)
    return out


def _parse_ics(body: str, *, label: str) -> Iterator[CalendarEvent]:
    """Walk unfolded lines, yield CalendarEvent for each VEVENT we
    can resolve to a concrete start_utc + summary."""
    in_event = False
    current: dict[str, tuple[str, str]] = {}
    for line in _unfold(body):
        stripped = line.strip()
        if stripped == "BEGIN:VEVENT":
            in_event = True
            current = {}
            continue
        if stripped == "END:VEVENT":
            in_event = False
            ev = _record_to_event(current, label=label)
            if ev is not None:
                yield ev
            current = {}
            continue
        if not in_event:
            continue
        m = _LINE_RE.match(stripped)
        if not m:
            continue
        key = m.group(1).upper()
        params_raw = (m.group(2) or "")[1:]   # strip leading ";"
        value = m.group(3).strip()
        # Keep only the first occurrence of each key in a record —
        # duplicate DTSTART rows are malformed input; take the first.
        current.setdefault(key, (params_raw, value))


def _record_to_event(
    rec: dict[str, tuple[str, str]], *, label: str,
) -> CalendarEvent | None:
    """Convert one parsed VEVENT dict to a CalendarEvent, or None
    when we can't resolve a usable start time."""
    if "RRULE" in rec:
        uid = rec.get("UID", ("", ""))[1] or "<no-uid>"
        if uid not in _rrule_warned_uids:
            log.info(
                "skipping recurring event %s (RRULE not supported); "
                "add the icalendar lib for recurrence", uid,
            )
            _rrule_warned_uids.add(uid)
        return None
    if "DTSTART" not in rec:
        return None
    params, value = rec["DTSTART"]
    start_utc = _parse_dtstart(params, value)
    if start_utc is None:
        return None
    summary = rec.get("SUMMARY", ("", ""))[1].strip() or "(no title)"
    # Un-escape minimal ICS escapes: \\ → \, \, → ,, \; → ;, \n → newline.
    summary = (summary
               .replace("\\N", "\n").replace("\\n", "\n")
               .replace("\\,", ",").replace("\\;", ";")
               .replace("\\\\", "\\"))
    uid = rec.get("UID", ("", ""))[1].strip()
    if not uid:
        # Synthesize a stable key from summary+start. Good enough for
        # dedup — users rarely create two separate events at the
        # same instant with the same title.
        uid = f"synth:{summary}:{start_utc.isoformat()}"
    return CalendarEvent(
        uid=uid,
        start_utc=start_utc,
        summary=summary,
        calendar_label=label,
    )


def _parse_dtstart(params_raw: str, value: str) -> datetime | None:
    """Resolve a DTSTART value to UTC datetime. Understands:

    * ``20260415T170000Z``            — UTC datetime
    * ``20260415T170000`` + TZID=X    — local → UTC via zoneinfo
    * ``20260415``                    — all-day → 00:00 UTC
    * ``20260415T170000`` (no TZID)   — floating; assume UTC (spec
      says "local wall clock regardless of TZ" but UTC is a
      practical default and ntfy users see correct relative time).
    """
    params = _parse_params(params_raw)
    tzid = params.get("TZID")
    value_type = params.get("VALUE", "").upper()

    # All-day event.
    if value_type == "DATE" or (len(value) == 8 and "T" not in value):
        try:
            d = date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
        except ValueError:
            return None
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    # Datetime form: YYYYMMDDTHHMMSS(Z)?
    m = re.match(r"^(\d{8})T(\d{6})(Z)?$", value)
    if m is None:
        return None
    ymd, hms, z = m.group(1), m.group(2), m.group(3)
    try:
        naive = datetime(
            int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]),
            int(hms[0:2]), int(hms[2:4]), int(hms[4:6]),
        )
    except ValueError:
        return None
    if z == "Z":
        return naive.replace(tzinfo=timezone.utc)
    if tzid and ZoneInfo is not None:
        try:
            tz = ZoneInfo(tzid)
            return naive.replace(tzinfo=tz).astimezone(timezone.utc)
        except ZoneInfoNotFoundError:
            log.debug("unknown TZID %r — falling back to UTC", tzid)
    # Floating time or unknown TZID → treat as UTC.
    return naive.replace(tzinfo=timezone.utc)


def _parse_params(params_raw: str) -> dict[str, str]:
    """Parse DTSTART param clause like ``VALUE=DATE;TZID=America/New_York``."""
    out: dict[str, str] = {}
    if not params_raw:
        return out
    for chunk in params_raw.split(";"):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        out[k.strip().upper()] = v.strip().strip('"')
    return out


def _mask(url: str) -> str:
    """Keep a secret URL out of the log at WARNING/INFO level."""
    if not url:
        return "<empty>"
    if len(url) <= 32:
        return url[:8] + "…"
    return url[:12] + "…" + url[-6:]
