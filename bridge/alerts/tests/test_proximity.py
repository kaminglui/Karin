"""Tests for bridge.alerts.proximity — the rule-based threat scorer.

Pins the 0-4 score grading per signal kind so future tuning (weights,
thresholds) doesn't silently regress the firing behaviour of the
rules layer that consumes it.
"""
from __future__ import annotations

import pytest

from bridge.alerts.models import SignalKind
from bridge.alerts.proximity import (
    compute_threat_score,
    haversine_km,
    location_match_score,
)
from bridge.alerts.user_context import UserContext


_UP_PA = UserContext(
    city="University Park",
    region="Pennsylvania",
    country="United States",
    latitude=40.7982,
    longitude=-77.8599,
)


# --- location_match_score ------------------------------------------------

class TestLocationMatchScore:
    def test_city_wins_over_region(self):
        txt = "Something happened in University Park yesterday"
        assert location_match_score(txt, _UP_PA) == 3

    def test_region_match(self):
        txt = "Pennsylvania governor speaks on new policy"
        assert location_match_score(txt, _UP_PA) == 2

    def test_country_only(self):
        txt = "United States trade delegation returns"
        assert location_match_score(txt, _UP_PA) == 1

    def test_no_match(self):
        assert location_match_score("Tokyo subway expansion", _UP_PA) == 0

    def test_empty_context(self):
        assert location_match_score("Pennsylvania", UserContext()) == 0


# --- haversine ----------------------------------------------------------

class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km(40, -77, 40, -77) == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self):
        # State College, PA ↔ Philadelphia, PA: ~246 km as the crow flies.
        km = haversine_km(40.79, -77.86, 39.95, -75.17)
        assert 230 < km < 260


# --- compute_threat_score per signal kind -------------------------------

def _p(**kwargs):
    return kwargs


class TestNwsScoring:
    def test_extreme_is_4(self):
        s = compute_threat_score(
            _p(severity="extreme", affected_zones="Pennsylvania"),
            SignalKind.NWS_WEATHER_ALERT, _UP_PA,
        )
        assert s == 4

    def test_minor_local(self):
        s = compute_threat_score(
            _p(severity="minor", affected_zones="Pennsylvania"),
            SignalKind.NWS_WEATHER_ALERT, _UP_PA,
        )
        assert s == 1

    def test_distant_demoted(self):
        # Severe alert in a different region → demoted by one tier.
        s = compute_threat_score(
            _p(severity="severe", affected_zones="California"),
            SignalKind.NWS_WEATHER_ALERT, _UP_PA,
        )
        assert s == 2   # severe (3) - 1 = 2

    def test_latlon_demotes_when_far(self):
        # Alert 1500km away demotes by 2 tiers.
        s = compute_threat_score(
            _p(severity="severe", lat=25.7, lon=-80.1),  # Miami
            SignalKind.NWS_WEATHER_ALERT, _UP_PA,
        )
        assert s == 1


class TestTravelScoring:
    def test_user_in_affected_country(self):
        s = compute_threat_score(
            _p(new_level=3, country="United States"),
            SignalKind.TRAVEL_ADVISORY_CHANGED, _UP_PA,
        )
        assert s == 3

    def test_elsewhere_demoted(self):
        s = compute_threat_score(
            _p(new_level=4, country="Pakistan"),
            SignalKind.TRAVEL_ADVISORY_CHANGED, _UP_PA,
        )
        # 4 - 2 = 2 (elsewhere demotion)
        assert s == 2


class TestNewsScoring:
    def test_no_location_is_zero(self):
        s = compute_threat_score(
            _p(headline="ASML earnings beat",
               watchlist_label="AI / Tech",
               watchlist_type="topic",
               cluster_state="confirmed"),
            SignalKind.NEWS_WATCHLIST_MATCH, _UP_PA,
        )
        assert s == 0

    def test_local_event_confirmed_fires(self):
        # Event in the user's state + confirmed -> should be >= 2.
        s = compute_threat_score(
            _p(headline="Pennsylvania flash flood warning upgraded",
               watchlist_label="Disasters",
               watchlist_type="event",
               cluster_state="confirmed"),
            SignalKind.NEWS_WATCHLIST_MATCH, _UP_PA,
        )
        assert s >= 2

    def test_topic_weight_softens_score(self):
        # Same proximity, topic instead of event → lower score.
        event_s = compute_threat_score(
            _p(headline="Pennsylvania state of emergency",
               watchlist_type="event",
               watchlist_label="Disasters",
               cluster_state="confirmed"),
            SignalKind.NEWS_WATCHLIST_MATCH, _UP_PA,
        )
        topic_s = compute_threat_score(
            _p(headline="Pennsylvania economy beats expectations",
               watchlist_type="topic",
               watchlist_label="Economy",
               cluster_state="confirmed"),
            SignalKind.NEWS_WATCHLIST_MATCH, _UP_PA,
        )
        assert event_s > topic_s


class TestTrackerScoring:
    def test_gas_spike_is_life_relevant(self):
        s = compute_threat_score(
            _p(category="energy", change_pct_abs=12.0),
            SignalKind.TRACKER_SHOCK, _UP_PA,
        )
        assert s == 3

    def test_fx_is_awareness_only(self):
        # Even a big FX move caps at 1 — markets aren't life-safety.
        s = compute_threat_score(
            _p(category="fx", change_pct_abs=20.0),
            SignalKind.TRACKER_SHOCK, _UP_PA,
        )
        assert s == 1


# --- UserContext loader --------------------------------------------------

class TestUserContextLoader:
    def test_empty_yaml_returns_empty_context(self, tmp_path, monkeypatch):
        # load_user_context() reuses bridge.utils.load_config. Point
        # REPO_ROOT at a tmp dir with a minimal yaml to confirm the
        # empty-path returns an empty UserContext without raising.
        from bridge.alerts import user_context as mod
        fake = tmp_path / "config" / "assistant.yaml"
        fake.parent.mkdir()
        fake.write_text("user_location:\n  city: ''\n  region: ''\n", encoding="utf-8")
        monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
        monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
        monkeypatch.delenv("KARIN_PROFILE", raising=False)
        ctx = mod.load_user_context()
        assert ctx.city == ""
        assert ctx.region == ""
        assert ctx.latitude is None

    def test_profile_overrides_yaml(self, tmp_path, monkeypatch):
        """Phase H: profile's user_location wins over the legacy
        assistant.yaml value. Confirms the migration path: once a user
        has set location in their profile, the yaml default is
        ignored."""
        from bridge.alerts import user_context as mod
        from bridge import profiles
        fake = tmp_path / "config" / "assistant.yaml"
        fake.parent.mkdir()
        fake.write_text(
            "user_location:\n"
            "  city: 'YAML City'\n"
            "  region: 'YAML State'\n"
            "  country: 'YAML Country'\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
        monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
        monkeypatch.delenv("KARIN_PROFILE", raising=False)
        p = profiles.create_profile("default")
        profiles.save_profile_preferences({
            "user_location": {
                "city": "Profile City",
                "latitude": 40.0,
                "longitude": -75.0,
            },
        }, p)
        ctx = mod.load_user_context()
        # Profile fields win; yaml fills in gaps where profile is empty.
        assert ctx.city == "Profile City"
        assert ctx.region == "YAML State"        # inherited from yaml
        assert ctx.country == "YAML Country"     # inherited from yaml
        assert ctx.latitude == 40.0              # profile-only
        assert ctx.longitude == -75.0            # profile-only

    def test_two_profiles_see_different_locations(self, tmp_path, monkeypatch):
        """The whole point of Phase H: two profiles on one box report
        different UserContexts."""
        from bridge.alerts import user_context as mod
        from bridge import profiles
        monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
        monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
        monkeypatch.delenv("KARIN_PROFILE", raising=False)
        work = profiles.create_profile("work")
        family = profiles.create_profile("family")
        profiles.save_profile_preferences(
            {"user_location": {"city": "PHL", "country": "US"}}, work,
        )
        profiles.save_profile_preferences(
            {"user_location": {"city": "TYO", "country": "JP"}}, family,
        )
        profiles.set_active("work")
        assert mod.load_user_context().city == "PHL"
        profiles.set_active("family")
        assert mod.load_user_context().city == "TYO"
