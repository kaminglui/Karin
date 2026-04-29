"""Tests for Phase G.c enrichment on /api/alerts/active.

Covers:
  - ``threat_score`` is the max of ``triggered_by_signals[*].payload.threat_score``
  - ``threat_score`` is None when no signal carries a score (legacy alerts)
  - ``location_configured`` reflects profile user_location presence
  - helpers tolerate malformed signals (old persisted data)

We exercise the helper functions directly rather than spinning up the
full AlertService, because the shape contract is what the UI depends on.
"""
from __future__ import annotations

import pytest

from web.panels_api import _location_configured, _max_threat_score


class TestMaxThreatScore:
    def test_none_for_empty_list(self):
        assert _max_threat_score([]) is None

    def test_none_for_non_list(self):
        assert _max_threat_score(None) is None
        assert _max_threat_score("garbage") is None

    def test_single_signal(self):
        signals = [{"payload": {"threat_score": 2}}]
        assert _max_threat_score(signals) == 2

    def test_picks_max_across_signals(self):
        signals = [
            {"payload": {"threat_score": 1}},
            {"payload": {"threat_score": 3}},
            {"payload": {"threat_score": 2}},
        ]
        assert _max_threat_score(signals) == 3

    def test_skips_signals_without_score(self):
        signals = [
            {"payload": {}},
            {"payload": {"threat_score": 2}},
        ]
        assert _max_threat_score(signals) == 2

    def test_returns_none_when_no_signal_has_score(self):
        signals = [
            {"payload": {}},
            {"payload": {"other": 5}},
        ]
        assert _max_threat_score(signals) is None

    def test_tolerates_missing_payload(self):
        # Old persisted alerts may not have a payload key at all.
        signals = [{}, {"foo": "bar"}]
        assert _max_threat_score(signals) is None

    def test_coerces_float_to_int(self):
        # Shouldn't happen, but defend against a future change that
        # emits floats — UI expects an int tier 0-4.
        assert _max_threat_score([{"payload": {"threat_score": 2.9}}]) == 2


class TestLocationConfigured:
    def test_empty_profile_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
        monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
        monkeypatch.delenv("KARIN_PROFILE", raising=False)
        # Fresh tmp repo + no yaml + no profile prefs -> empty context.
        assert _location_configured() is False

    def test_profile_city_is_enough(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
        monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
        monkeypatch.delenv("KARIN_PROFILE", raising=False)
        from bridge import profiles
        p = profiles.create_profile("default")
        profiles.save_profile_preferences(
            {"user_location": {"city": "PHL"}}, p,
        )
        assert _location_configured() is True

    def test_yaml_fallback_counts(self, tmp_path, monkeypatch):
        """If the profile is empty but the legacy yaml has a location,
        the threat assessor will still use it (per the Phase H fallback
        chain), so the UI should NOT show the hint."""
        monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
        monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
        monkeypatch.delenv("KARIN_PROFILE", raising=False)
        cfg = tmp_path / "config" / "assistant.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(
            "user_location:\n"
            "  country: United States\n",
            encoding="utf-8",
        )
        assert _location_configured() is True

    def test_latlon_only_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
        monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
        monkeypatch.delenv("KARIN_PROFILE", raising=False)
        from bridge import profiles
        p = profiles.create_profile("default")
        profiles.save_profile_preferences(
            {"user_location": {"latitude": 40.0, "longitude": -75.0}}, p,
        )
        assert _location_configured() is True
