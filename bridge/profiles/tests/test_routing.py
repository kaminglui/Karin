"""Tests for Phase H.d — Tailscale IP-to-profile routing.

Covers:
  - load/save round-trip
  - resolve_profile_for_ip (hit / miss / empty)
  - profile name validation in set_routing_dict
  - missing / corrupt yaml graceful fallback
  - get/set convenience helpers for the API layer
"""
from __future__ import annotations

import pytest

from bridge.profiles import routing
from bridge.profiles.routing import DeviceRoute


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    monkeypatch.setattr("bridge.profiles.routing.REPO_ROOT", tmp_path)
    monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
    monkeypatch.delenv("KARIN_PROFILE", raising=False)
    yield tmp_path


class TestLoadSaveRoundTrip:
    def test_empty_when_no_file(self, tmp_repo):
        cfg = routing.load_routing()
        assert cfg.tailscale_ip == {}

    def test_save_then_load(self, tmp_repo):
        cfg = routing.RoutingConfig(tailscale_ip={
            "100.64.0.10": DeviceRoute(profile="work", nickname="Work PC"),
            "100.64.0.20": DeviceRoute(profile="family"),
        })
        routing.save_routing(cfg)
        loaded = routing.load_routing()
        assert loaded.tailscale_ip["100.64.0.10"].profile == "work"
        assert loaded.tailscale_ip["100.64.0.10"].nickname == "Work PC"
        assert loaded.tailscale_ip["100.64.0.20"].profile == "family"
        assert loaded.tailscale_ip["100.64.0.20"].nickname == ""

    def test_backward_compat_string_values(self, tmp_repo):
        """Old format {ip: "profile_name"} still parses."""
        path = tmp_repo / "data" / "profile_routing.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "tailscale_ip:\n  100.1.2.3: work\n",
            encoding="utf-8",
        )
        cfg = routing.load_routing()
        assert cfg.tailscale_ip["100.1.2.3"].profile == "work"
        assert cfg.tailscale_ip["100.1.2.3"].nickname == ""

    def test_corrupt_yaml_returns_empty(self, tmp_repo):
        path = tmp_repo / "data" / "profile_routing.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not: [valid: yaml: {{", encoding="utf-8")
        cfg = routing.load_routing()
        assert cfg.tailscale_ip == {}

    def test_non_dict_yaml_returns_empty(self, tmp_repo):
        path = tmp_repo / "data" / "profile_routing.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"just a string"', encoding="utf-8")
        cfg = routing.load_routing()
        assert cfg.tailscale_ip == {}


class TestResolveProfileForIp:
    def test_hit(self, tmp_repo):
        routing.save_routing(routing.RoutingConfig(
            tailscale_ip={"100.1.2.3": DeviceRoute(profile="work")},
        ))
        assert routing.resolve_profile_for_ip("100.1.2.3") == "work"

    def test_miss(self, tmp_repo):
        routing.save_routing(routing.RoutingConfig(
            tailscale_ip={"100.1.2.3": DeviceRoute(profile="work")},
        ))
        assert routing.resolve_profile_for_ip("200.0.0.1") is None

    def test_empty_ip_returns_none(self, tmp_repo):
        assert routing.resolve_profile_for_ip("") is None
        assert routing.resolve_profile_for_ip(None) is None

    def test_no_config_file_returns_none(self, tmp_repo):
        assert routing.resolve_profile_for_ip("100.1.2.3") is None


class TestConvenienceHelpers:
    def test_get_empty(self, tmp_repo):
        assert routing.get_routing_dict() == []

    def test_set_then_get(self, tmp_repo):
        from bridge.profiles import create_profile
        create_profile("work")
        routing.set_routing_list([
            {"ip": "100.1.2.3", "profile": "work", "nickname": "Laptop"},
        ])
        result = routing.get_routing_dict()
        assert len(result) == 1
        assert result[0]["ip"] == "100.1.2.3"
        assert result[0]["profile"] == "work"
        assert result[0]["nickname"] == "Laptop"

    def test_set_validates_profile_names(self, tmp_repo):
        from bridge.profiles import ProfileNameError
        with pytest.raises(ProfileNameError):
            routing.set_routing_list([
                {"ip": "100.1.2.3", "profile": "../evil"},
            ])

    def test_set_strips_whitespace(self, tmp_repo):
        from bridge.profiles import create_profile
        create_profile("work")
        routing.set_routing_list([
            {"ip": "  100.1.2.3  ", "profile": "  Work  ", "nickname": " My PC "},
        ])
        result = routing.get_routing_dict()
        assert result[0]["ip"] == "100.1.2.3"
        assert result[0]["profile"] == "work"
        assert result[0]["nickname"] == "My PC"

    def test_nickname_optional(self, tmp_repo):
        from bridge.profiles import create_profile
        create_profile("work")
        routing.set_routing_list([
            {"ip": "100.1.2.3", "profile": "work"},
        ])
        result = routing.get_routing_dict()
        assert result[0]["nickname"] == ""
