"""Calendar (.ics) polling → notify pipeline.

Hits a list of calendar URLs on the poller's schedule, parses the
ICS payload for upcoming VEVENTs, and pushes "Meeting in N min"
events through :func:`bridge.notify.notify` when the event crosses
its lead-time threshold.

Scope:

* Read-only. No CalDAV / OAuth / write-back. The user's existing
  calendar (Google / Apple iCloud / Outlook / Nextcloud / Fastmail
  / …) exposes a secret ``.ics`` URL; they paste it into config and
  we poll it.
* Non-recurring events only. VEVENTs with an RRULE field are
  skipped (logged once). Most urgent "meeting-in-15-min" reminders
  are for one-off events; recurrence support gated behind adding
  the full ``icalendar`` library as a dep.
* UTC + timezone-aware-local-time events supported. All-day events
  (DATE only, no time) are treated as starting at midnight UTC on
  the given date.
* Dedup keyed on ``(uid, start_utc)`` in a SQLite ledger so the
  same event fires exactly once per calendar poll sweep even if
  the user added it weeks ago.

Public API:

* :func:`bridge.calendar.fetch.fetch_events` — sync: fetch + parse.
* :func:`bridge.calendar.poll.tick` — what the poller calls each
  interval; returns the count of events that fired notifications.
"""
from __future__ import annotations

from .fetch import CalendarEvent, fetch_events
from .poll import tick as tick_calendars

__all__ = ["CalendarEvent", "fetch_events", "tick_calendars"]
