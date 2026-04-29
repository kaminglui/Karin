"""Tests for bridge.calendar — parser, dedup store, and the tick
that fetches + fires notifications.

No real HTTP: fetch_events is bypassed by patching. No real notify
dispatch: bridge.notify.notify is patched. This keeps the tests
focused on our own logic (parse correctness, dedup, lead-time
window), not the transport layer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from bridge.calendar.fetch import CalendarEvent, _parse_ics
from bridge.calendar.poll import CalendarConfig, tick
from bridge.calendar.store import CalendarDedupeStore


_NOW = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)


def _sample_ics(*events: str) -> str:
    """Minimal VCALENDAR wrapper around caller-supplied VEVENT blocks."""
    return (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        + "".join(events)
        + "END:VCALENDAR\r\n"
    )


def _vevent(
    *, uid="u1", dtstart="20260415T170000Z", summary="Meeting",
    extras: list[str] | None = None,
) -> str:
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTART:{dtstart}",
        f"SUMMARY:{summary}",
    ]
    if extras:
        lines.extend(extras)
    lines.append("END:VEVENT")
    return "\r\n".join(lines) + "\r\n"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class TestIcsParser:
    def test_single_utc_event(self):
        ics = _sample_ics(_vevent())
        events = list(_parse_ics(ics, label=""))
        assert len(events) == 1
        ev = events[0]
        assert ev.uid == "u1"
        assert ev.summary == "Meeting"
        assert ev.start_utc == datetime(2026, 4, 15, 17, 0, tzinfo=timezone.utc)

    def test_multiple_events_parsed(self):
        ics = _sample_ics(
            _vevent(uid="u1", dtstart="20260415T170000Z", summary="A"),
            _vevent(uid="u2", dtstart="20260415T180000Z", summary="B"),
        )
        events = list(_parse_ics(ics, label="work"))
        assert [e.uid for e in events] == ["u1", "u2"]
        assert [e.calendar_label for e in events] == ["work", "work"]

    def test_all_day_event(self):
        ics = _sample_ics(_vevent(
            uid="all-day", dtstart="20260415",
            extras=[],   # VALUE=DATE inferred from length
            summary="Birthday",
        ))
        events = list(_parse_ics(ics, label=""))
        assert len(events) == 1
        assert events[0].start_utc == datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)

    def test_tzid_converts_to_utc(self):
        """DTSTART;TZID=America/New_York:20260415T130000 →
        13:00 NY → 17:00 UTC (assuming EDT offset)."""
        ics = _sample_ics(
            "BEGIN:VEVENT\r\n"
            "UID:tz-event\r\n"
            "DTSTART;TZID=America/New_York:20260415T130000\r\n"
            "SUMMARY:NY meeting\r\n"
            "END:VEVENT\r\n"
        )
        events = list(_parse_ics(ics, label=""))
        assert len(events) == 1
        # April 15 2026 in NY is EDT (UTC-4) → 13:00 local = 17:00 UTC.
        assert events[0].start_utc == datetime(2026, 4, 15, 17, 0, tzinfo=timezone.utc)

    def test_rrule_event_skipped(self):
        ics = _sample_ics(_vevent(
            uid="weekly", extras=["RRULE:FREQ=WEEKLY;BYDAY=MO"],
        ))
        events = list(_parse_ics(ics, label=""))
        assert events == []

    def test_summary_escape_sequences(self):
        ics = _sample_ics(_vevent(
            summary="Part A\\, part B\\; part C\\nnext line",
        ))
        events = list(_parse_ics(ics, label=""))
        assert events[0].summary == "Part A, part B; part C\nnext line"

    def test_missing_uid_synthesized(self):
        ics = _sample_ics(
            "BEGIN:VEVENT\r\n"
            "DTSTART:20260415T170000Z\r\n"
            "SUMMARY:No UID\r\n"
            "END:VEVENT\r\n"
        )
        events = list(_parse_ics(ics, label=""))
        assert len(events) == 1
        assert events[0].uid.startswith("synth:")

    def test_folded_lines_unfolded(self):
        """RFC 5545 line folding: a CRLF followed by a single SPACE/HTAB
        is the fold marker — BOTH characters get stripped when
        unfolding. So "very long " + CRLF + " summary" unfolds to
        "very long summary" (the trailing space before the fold is
        content)."""
        ics = (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\n"
            "UID:fold\r\n"
            "DTSTART:20260415T170000Z\r\n"
            "SUMMARY:This is a very long \r\n"
            " summary that got folded\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        events = list(_parse_ics(ics, label=""))
        assert events[0].summary == "This is a very long summary that got folded"


# ---------------------------------------------------------------------------
# Dedup store
# ---------------------------------------------------------------------------
class TestDedupStore:
    def test_first_insert_returns_true(self, tmp_path):
        s = CalendarDedupeStore(tmp_path / "dedup.db")
        assert s.mark_notified(
            "k1", "uid-1", _NOW, "Meeting",
        ) is True
        assert s.already_notified("k1") is True

    def test_second_insert_same_key_returns_false(self, tmp_path):
        s = CalendarDedupeStore(tmp_path / "dedup.db")
        s.mark_notified("k1", "uid-1", _NOW, "Meeting")
        assert s.mark_notified("k1", "uid-1", _NOW, "Meeting") is False

    def test_different_keys_dont_collide(self, tmp_path):
        s = CalendarDedupeStore(tmp_path / "dedup.db")
        assert s.mark_notified("k1", "uid-1", _NOW, "A") is True
        assert s.mark_notified("k2", "uid-1", _NOW, "A") is True


# ---------------------------------------------------------------------------
# Poll tick: lead-time window + dedup + notify dispatch
# ---------------------------------------------------------------------------
@pytest.fixture
def dedup_store(tmp_path) -> CalendarDedupeStore:
    return CalendarDedupeStore(tmp_path / "dedup.db")


class TestPollTick:
    def test_no_configs_returns_zero(self, dedup_store):
        assert tick(now=_NOW, configs=[], store=dedup_store) == 0

    def test_event_in_window_fires(self, dedup_store):
        """Event starting 10 min from now, lead_time=15 → fires."""
        event = CalendarEvent(
            uid="u1", start_utc=_NOW + timedelta(minutes=10),
            summary="Soon", calendar_label="work",
        )
        cfg = CalendarConfig(label="work", url="https://x.test/cal.ics",
                             lead_time_minutes=15)
        with patch("bridge.calendar.poll.fetch_events", return_value=[event]), \
             patch("bridge.notify.notify") as mock_notify:
            fired = tick(now=_NOW, configs=[cfg], store=dedup_store)
        assert fired == 1
        assert mock_notify.call_count == 1
        ev_obj = mock_notify.call_args.args[0]
        assert ev_obj.kind == "calendar.upcoming"
        assert "Soon" in ev_obj.title
        assert ev_obj.payload["dedupe_key"] == event.dedupe_key()

    def test_event_outside_window_skipped(self, dedup_store):
        """Event starting 30 min from now, lead_time=15 → skipped."""
        event = CalendarEvent(
            uid="u1", start_utc=_NOW + timedelta(minutes=30),
            summary="Later",
        )
        cfg = CalendarConfig(label="w", url="x", lead_time_minutes=15)
        with patch("bridge.calendar.poll.fetch_events", return_value=[event]), \
             patch("bridge.notify.notify") as mock_notify:
            fired = tick(now=_NOW, configs=[cfg], store=dedup_store)
        assert fired == 0
        assert mock_notify.call_count == 0

    def test_past_event_skipped(self, dedup_store):
        event = CalendarEvent(
            uid="u1", start_utc=_NOW - timedelta(minutes=5),
            summary="Past",
        )
        cfg = CalendarConfig(label="w", url="x", lead_time_minutes=15)
        with patch("bridge.calendar.poll.fetch_events", return_value=[event]), \
             patch("bridge.notify.notify") as mock_notify:
            assert tick(now=_NOW, configs=[cfg], store=dedup_store) == 0
            assert mock_notify.call_count == 0

    def test_second_tick_dedups(self, dedup_store):
        """Same event seen twice across ticks — should only fire once."""
        event = CalendarEvent(
            uid="u1", start_utc=_NOW + timedelta(minutes=10),
            summary="Same meeting",
        )
        cfg = CalendarConfig(label="w", url="x", lead_time_minutes=15)
        with patch("bridge.calendar.poll.fetch_events", return_value=[event]), \
             patch("bridge.notify.notify"):
            first = tick(now=_NOW, configs=[cfg], store=dedup_store)
            second = tick(now=_NOW, configs=[cfg], store=dedup_store)
        assert first == 1
        assert second == 0

    def test_rescheduled_event_fires_again(self, dedup_store):
        """Same uid, different start_utc (user moved the meeting) →
        must fire again. Dedup key includes the start time."""
        ev_old = CalendarEvent(
            uid="u1", start_utc=_NOW + timedelta(minutes=10),
            summary="Meeting v1",
        )
        ev_new = CalendarEvent(
            uid="u1", start_utc=_NOW + timedelta(minutes=12),
            summary="Meeting v1",
        )
        cfg = CalendarConfig(label="w", url="x", lead_time_minutes=15)
        with patch("bridge.calendar.poll.fetch_events", side_effect=[[ev_old], [ev_new]]), \
             patch("bridge.notify.notify"):
            assert tick(now=_NOW, configs=[cfg], store=dedup_store) == 1
            assert tick(now=_NOW, configs=[cfg], store=dedup_store) == 1

    def test_partial_fetch_failure_does_not_block_other_feeds(self, dedup_store):
        """One feed broken → other feeds still process."""
        good = CalendarEvent(
            uid="g", start_utc=_NOW + timedelta(minutes=5),
            summary="Good feed event",
        )
        cfg_bad = CalendarConfig(label="bad", url="https://x.test/404",
                                 lead_time_minutes=15)
        cfg_good = CalendarConfig(label="good", url="https://x.test/ok",
                                  lead_time_minutes=15)

        def fake_fetch(url, label="", **_kw):
            if url == cfg_bad.url:
                raise RuntimeError("nope")
            return [good]

        with patch("bridge.calendar.poll.fetch_events", side_effect=fake_fetch), \
             patch("bridge.notify.notify"):
            fired = tick(
                now=_NOW, configs=[cfg_bad, cfg_good], store=dedup_store,
            )
        assert fired == 1
