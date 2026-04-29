"""Tests for bridge.holidays — nager fetch, supplemental lookup, audio cache."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

import bridge.holidays as holidays


@pytest.fixture(autouse=True)
def _tmp_dirs(tmp_path, monkeypatch):
    """Redirect the module's cache dirs to a tmp path per test so
    real ``data/holidays/`` never gets touched."""
    monkeypatch.setattr(holidays, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(holidays, "_AUDIO_DIR", tmp_path / "audio")
    yield


# ---- today_holiday lookup order ----------------------------------------


class TestLookupOrder:
    def test_dated_supplemental_wins(self, monkeypatch):
        """YYYY-MM-DD supplemental entry beats nager response + MM-DD fallback."""
        # Pretend nager also has an entry for this date; supplemental should still win.
        monkeypatch.setattr(holidays, "_fetch_from_nager", lambda y, cc: [
            {"date": "2026-08-19", "name": "Nager August Thing"},
        ])
        # "2026-08-19" is in _SUPPLEMENTAL as Qixi.
        rec = holidays.today_holiday(datetime.date(2026, 8, 19))
        assert rec["name"] == "Qixi"
        assert rec["country"] == "CN"
        assert rec["date"] == "2026-08-19"

    def test_nager_cn_beats_us(self, monkeypatch):
        """CN is first in _COUNTRIES, so its entries shadow US on collisions."""
        def fake_fetch(y, cc):
            if cc == "CN":
                return [{"date": "2026-05-01", "name": "Labour Day"}]
            return [{"date": "2026-05-01", "name": "US Labor-ish"}]
        monkeypatch.setattr(holidays, "_fetch_from_nager", fake_fetch)
        rec = holidays.today_holiday(datetime.date(2026, 5, 1))
        assert rec["name"] == "Labour Day"
        assert rec["country"] == "CN"

    def test_fixed_mmdd_supplemental_fallback(self, monkeypatch):
        """If nager has nothing, fixed-date supplemental (Halloween) kicks in."""
        monkeypatch.setattr(holidays, "_fetch_from_nager", lambda y, cc: [])
        rec = holidays.today_holiday(datetime.date(2026, 10, 31))
        assert rec["name"] == "Halloween"
        assert rec["emoji"] == "🎃"

    def test_no_match_returns_none(self, monkeypatch):
        monkeypatch.setattr(holidays, "_fetch_from_nager", lambda y, cc: [])
        assert holidays.today_holiday(datetime.date(2026, 4, 14)) is None


# ---- nager caching ------------------------------------------------------


class TestCaching:
    def test_successful_fetch_writes_cache(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_fetch(y, cc):
            calls["n"] += 1
            return [{"date": f"{y}-12-25", "name": "Christmas Day"}]

        monkeypatch.setattr(holidays, "_fetch_from_nager", fake_fetch)
        holidays.today_holiday(datetime.date(2026, 12, 25))
        assert calls["n"] >= 1
        # Second call should come from disk
        holidays.today_holiday(datetime.date(2026, 12, 25))
        # Cache file exists
        assert (tmp_path / "2026-CN.json").exists() or (tmp_path / "2026-US.json").exists()

    def test_fetch_failure_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(holidays, "_fetch_from_nager", lambda y, cc: None)
        # Lands in fixed-date supplemental (Halloween) so returns something.
        rec = holidays.today_holiday(datetime.date(2026, 10, 31))
        assert rec is not None
        # Non-holiday date returns None cleanly.
        assert holidays.today_holiday(datetime.date(2026, 4, 15)) is None

    def test_nager_network_path_is_fail_soft(self, monkeypatch):
        """Simulate an httpx failure — module shouldn't raise."""
        def boom(url, *a, **k):
            raise httpx.ConnectError("network down")
        # Install a failing client globally
        class _FailClient:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url): raise httpx.ConnectError("network down")
        monkeypatch.setattr(holidays.httpx, "Client", _FailClient)
        # Should just return None (no fixed-date supplemental matches 2026-04-15).
        assert holidays.today_holiday(datetime.date(2026, 4, 15)) is None


# ---- voiced-greeting audio ---------------------------------------------


class _FakeTTS:
    """Minimal stand-in for bridge.tts.SoVITSTTS."""
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def synthesize(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("sovits unreachable")
        # 0.5 s of silence at 32 kHz
        return np.zeros(16000, dtype=np.int16), 32000


class TestAudioCache:
    def test_generates_and_caches_on_first_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            holidays, "today_holiday",
            lambda: {"date": "2026-12-25", "name": "Christmas Day",
                     "country": "US", "emoji": "🎄", "greeting": "Merry Christmas!"},
        )
        tts = _FakeTTS()
        p = holidays.get_or_synth_greeting_audio(tts)
        assert p is not None and p.exists()
        assert tts.calls == 1
        # Second call reuses cache.
        p2 = holidays.get_or_synth_greeting_audio(tts)
        assert p2 == p
        assert tts.calls == 1

    def test_stale_audio_purged_on_non_holiday_day(self, monkeypatch, tmp_path):
        monkeypatch.setattr(holidays, "today_holiday", lambda: None)
        # Plant a stale file from a prior holiday.
        (tmp_path / "audio").mkdir(parents=True)
        stale = tmp_path / "audio" / "2025-12-25-christmas.wav"
        stale.write_bytes(b"\x00\x00")
        tts = _FakeTTS()
        result = holidays.get_or_synth_greeting_audio(tts)
        assert result is None
        assert not stale.exists()   # cleanup removed it

    def test_stale_audio_purged_when_date_changes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            holidays, "today_holiday",
            lambda: {"date": "2026-07-04", "name": "Independence Day",
                     "country": "US", "emoji": "🇺🇸", "greeting": "Happy 4th!"},
        )
        (tmp_path / "audio").mkdir(parents=True)
        stale = tmp_path / "audio" / "2025-12-25-christmas.wav"
        stale.write_bytes(b"\x00\x00")
        holidays.get_or_synth_greeting_audio(_FakeTTS())
        assert not stale.exists()

    def test_tts_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            holidays, "today_holiday",
            lambda: {"date": "2026-12-25", "name": "Christmas Day",
                     "country": "US", "emoji": "🎄", "greeting": "Merry Christmas!"},
        )
        p = holidays.get_or_synth_greeting_audio(_FakeTTS(fail=True))
        assert p is None

    def test_no_holiday_returns_none(self, monkeypatch):
        monkeypatch.setattr(holidays, "today_holiday", lambda: None)
        p = holidays.get_or_synth_greeting_audio(_FakeTTS())
        assert p is None


# ---- helpers ------------------------------------------------------------


class TestSlug:
    def test_basic(self):
        assert holidays._audio_slug("Chinese New Year") == "chinese-new-year"

    def test_strips_punctuation(self):
        assert holidays._audio_slug("St. Patrick's Day!") == "st-patrick-s-day"

    def test_empty_falls_back(self):
        assert holidays._audio_slug("___") == "holiday"
