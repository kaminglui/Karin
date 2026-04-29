"""Tests for per-source fetchers.

Mocks httpx.Client so no network calls happen. Each fetcher is tested
against a plausible-shape success response plus the failure modes we
expect to see in the wild.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from bridge.trackers.fetch import (
    FetchError,
    fetch,
    fetch_bls,
    fetch_frankfurter,
    fetch_stooq,
)
from bridge.trackers.models import TrackerConfig


# --- helpers ---------------------------------------------------------------

def _cfg(**kw) -> TrackerConfig:
    base = dict(
        id="t1", label="Test", category="fx", source="frankfurter",
        params={}, cadence="daily", stale_after_hours=72,
        history_days=90, enabled=True,
    )
    base.update(kw)
    return TrackerConfig(**base)


def _client(*, status=200, json_body=None, text=""):
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_body
    resp.text = text
    client.get.return_value = resp
    client.post.return_value = resp
    return client


# --- Frankfurter -----------------------------------------------------------

class TestFrankfurter:
    def test_parses_rate_and_date(self):
        cfg = _cfg(params={"from": "USD", "to": "CNY"})
        client = _client(json_body={
            "amount": 1.0, "base": "USD", "date": "2026-04-11",
            "rates": {"CNY": 7.2289},
        })
        reading = fetch_frankfurter(cfg, client)
        assert reading.value == 7.2289
        # Stamped at 15:00 UTC of the publish date.
        assert reading.timestamp == datetime(2026, 4, 11, 15, 0, tzinfo=timezone.utc)

    def test_http_5xx_raises(self):
        cfg = _cfg(params={"from": "USD", "to": "CNY"})
        client = _client(status=503, text="upstream down")
        with pytest.raises(FetchError, match="503"):
            fetch_frankfurter(cfg, client)

    def test_missing_to_param_raises(self):
        cfg = _cfg(params={"from": "USD"})
        with pytest.raises(FetchError, match="missing 'to'"):
            fetch_frankfurter(cfg, _client())

    def test_target_currency_missing_from_response_raises(self):
        cfg = _cfg(params={"from": "USD", "to": "ZZZ"})
        client = _client(json_body={
            "amount": 1.0, "base": "USD", "date": "2026-04-11",
            "rates": {"CNY": 7.2},  # no ZZZ
        })
        with pytest.raises(FetchError, match="no rate for ZZZ"):
            fetch_frankfurter(cfg, client)


# --- Stooq -----------------------------------------------------------------

_STOOQ_CSV_OK = (
    "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
    "XAUUSD,2026-04-11,21:30:23,2345.60,2350.10,2341.20,2347.80,12345\n"
)
_STOOQ_CSV_NO_DATA = (
    "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
    "XAUUSD,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n"
)


class TestStooq:
    def test_parses_close_and_timestamp(self):
        cfg = _cfg(category="metal", source="stooq", params={"symbol": "xauusd"})
        client = _client(text=_STOOQ_CSV_OK)
        reading = fetch_stooq(cfg, client)
        assert reading.value == 2347.80
        assert reading.timestamp == datetime(2026, 4, 11, 21, 30, 23, tzinfo=timezone.utc)

    def test_no_data_response_raises(self):
        cfg = _cfg(category="metal", source="stooq", params={"symbol": "xauusd"})
        client = _client(text=_STOOQ_CSV_NO_DATA)
        with pytest.raises(FetchError, match="no data"):
            fetch_stooq(cfg, client)

    def test_missing_symbol_raises(self):
        cfg = _cfg(category="metal", source="stooq", params={})
        with pytest.raises(FetchError, match="missing 'symbol'"):
            fetch_stooq(cfg, _client(text=""))

    def test_short_row_raises(self):
        cfg = _cfg(category="metal", source="stooq", params={"symbol": "xauusd"})
        # Only 3 columns in the data row — malformed.
        client = _client(text="Symbol,Date,Time\nXAUUSD,2026-04-11,21:30:23\n")
        with pytest.raises(FetchError, match="short row"):
            fetch_stooq(cfg, client)


# --- BLS -------------------------------------------------------------------

def _bls_ok(series_id: str, year: int, month: int, value: float) -> dict:
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 12,
        "message": [],
        "Results": {
            "series": [{
                "seriesID": series_id,
                "data": [
                    {"year": str(year), "period": f"M{month:02d}",
                     "periodName": "March", "value": str(value), "footnotes": [{}]},
                ],
            }],
        },
    }


class TestBLS:
    def test_parses_latest_monthly_value(self):
        cfg = _cfg(category="food_index", source="bls", cadence="monthly",
                   params={"series_id": "CUUR0000SAF1"})
        client = _client(json_body=_bls_ok("CUUR0000SAF1", 2026, 3, 312.456))
        reading = fetch_bls(cfg, client)
        assert reading.value == 312.456
        # Stamped noon UTC on the LAST day of the reported month. This ties
        # the reading to the period described, and keeps the default
        # stale_after_hours threshold from tripping on data that's just
        # been released.
        assert reading.timestamp == datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc)

    def test_end_of_month_stamping_handles_february_non_leap(self):
        # Feb 2026 has 28 days (not leap). calendar.monthrange must give
        # us the 28th, not 29th/30th/31st.
        cfg = _cfg(category="food_index", source="bls", cadence="monthly",
                   params={"series_id": "CUUR0000SAF1"})
        client = _client(json_body=_bls_ok("CUUR0000SAF1", 2026, 2, 310.0))
        reading = fetch_bls(cfg, client)
        assert reading.timestamp == datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)

    def test_end_of_month_stamping_handles_february_leap(self):
        # Feb 2028 is a leap year -> 29 days.
        cfg = _cfg(category="food_index", source="bls", cadence="monthly",
                   params={"series_id": "CUUR0000SAF1"})
        client = _client(json_body=_bls_ok("CUUR0000SAF1", 2028, 2, 320.0))
        reading = fetch_bls(cfg, client)
        assert reading.timestamp == datetime(2028, 2, 29, 12, 0, tzinfo=timezone.utc)

    def test_api_error_status_raises(self):
        cfg = _cfg(category="food_index", source="bls", cadence="monthly",
                   params={"series_id": "BAD"})
        client = _client(json_body={
            "status": "REQUEST_NOT_PROCESSED",
            "message": ["Series does not exist for Series BAD"],
            "Results": {},
        })
        with pytest.raises(FetchError, match="Series does not exist"):
            fetch_bls(cfg, client)

    def test_empty_data_raises(self):
        cfg = _cfg(category="food_index", source="bls", cadence="monthly",
                   params={"series_id": "CUUR0000SAF1"})
        client = _client(json_body={
            "status": "REQUEST_SUCCEEDED", "message": [],
            "Results": {"series": [{"seriesID": "CUUR0000SAF1", "data": []}]},
        })
        with pytest.raises(FetchError, match="empty data"):
            fetch_bls(cfg, client)

    def test_annual_period_m13_rejected(self):
        # BLS uses M13 for annual averages; we want monthly only.
        cfg = _cfg(category="food_index", source="bls", cadence="monthly",
                   params={"series_id": "CUUR0000SAF1"})
        client = _client(json_body={
            "status": "REQUEST_SUCCEEDED", "message": [],
            "Results": {"series": [{"seriesID": "CUUR0000SAF1", "data": [
                {"year": "2025", "period": "M13", "periodName": "Annual",
                 "value": "311.0", "footnotes": [{}]},
            ]}]},
        })
        with pytest.raises(FetchError, match="unexpected period M13"):
            fetch_bls(cfg, client)


# --- dispatch --------------------------------------------------------------

class TestDispatch:
    def test_unknown_source_raises(self):
        cfg = _cfg(source="mystery_source")
        with pytest.raises(FetchError, match="unknown source"):
            fetch(cfg, _client())
