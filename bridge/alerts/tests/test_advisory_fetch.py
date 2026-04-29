"""Tests for bridge.alerts.advisory_fetch.

Covers RSS parsing, country normalization, and diff behavior including
the first-poll baseline semantics.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bridge.alerts.advisory_fetch import (
    AdvisoryFetchError,
    diff_advisories,
    fetch_advisories,
    snapshot_to_state,
)


_RSS_OK = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Travel Advisories</title>
  <item>
    <title>Egypt - Level 3: Reconsider Travel Travel Advisory</title>
    <link>https://travel.state.gov/egypt</link>
    <description>...</description>
  </item>
  <item>
    <title>Japan - Level 1: Exercise Normal Precautions Travel Advisory</title>
    <link>https://travel.state.gov/japan</link>
    <description>...</description>
  </item>
  <item>
    <title>Saint Vincent and the Grenadines - Level 2: Exercise Increased Caution</title>
    <link>https://travel.state.gov/svg</link>
    <description>...</description>
  </item>
</channel></rss>
"""


def _client(*, status=200, content=b""):
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    client.get.return_value = resp
    return client


class TestFetchAdvisories:
    def test_parses_title_into_country_level(self):
        current = fetch_advisories(_client(content=_RSS_OK))
        assert "EGYPT" in current
        assert current["EGYPT"]["level"] == 3
        assert "JAPAN" in current
        assert current["JAPAN"]["level"] == 1

    def test_handles_multi_word_country(self):
        current = fetch_advisories(_client(content=_RSS_OK))
        assert "SAINT_VINCENT_AND_THE_GRENADINES" in current
        assert current["SAINT_VINCENT_AND_THE_GRENADINES"]["level"] == 2

    def test_non_200_raises(self):
        with pytest.raises(AdvisoryFetchError, match="status 503"):
            fetch_advisories(_client(status=503, content=b""))


class TestDiffAdvisories:
    def test_first_poll_semantics_via_service_not_here(self):
        # First-poll baseline lives in AlertService — diff_advisories
        # itself is symmetrical. When previous_state is empty, every
        # current entry diffs as "appeared" (old_level=None).
        current = {"EGY": {"level": 3, "title": "x"}}
        changes = diff_advisories(current, previous_state={})
        assert len(changes) == 1
        assert changes[0]["old_level"] is None
        assert changes[0]["new_level"] == 3

    def test_unchanged_levels_produce_no_diff(self):
        current = {"EGY": {"level": 3, "title": "x"}, "JPN": {"level": 1, "title": "y"}}
        prev = {"EGY": 3, "JPN": 1}
        assert diff_advisories(current, prev) == []

    def test_raised_level_produces_diff(self):
        current = {"EGY": {"level": 4, "title": "Egypt Level 4"}}
        prev = {"EGY": 3}
        changes = diff_advisories(current, prev)
        assert len(changes) == 1
        assert changes[0]["old_level"] == 3
        assert changes[0]["new_level"] == 4

    def test_dropped_country_not_reported(self):
        # If a country drops off the feed entirely, we don't emit a
        # change — "old_level=X, new_level=None" isn't alertable.
        current = {}
        prev = {"EGY": 3}
        assert diff_advisories(current, prev) == []


class TestSnapshotToState:
    def test_reduces_to_country_level_map(self):
        snap = {
            "EGY": {"level": 3, "title": "x"},
            "JPN": {"level": 1, "title": "y"},
        }
        assert snapshot_to_state(snap) == {"EGY": 3, "JPN": 1}
