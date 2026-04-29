"""Tests for the feature registry."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bridge import features


@pytest.fixture(autouse=True)
def _clean_cache():
    """Every test gets a fresh registry load. Prevents cross-test
    bleed from the lru_cache on _load()."""
    features.reload()
    yield
    features.reload()


@pytest.fixture
def fake_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "features.yaml"
    p.write_text(
        """
subsystems:
  stt:
    enabled: false
    env: FAKE_STT
  tts:
    enabled: true
    env: FAKE_TTS
  bandit:
    enabled: false
  holidays:
    enabled: true

tools:
  disabled:
    - tell_story
    - wiki_random
""",
        encoding="utf-8",
    )
    return p


class TestIsEnabled:
    def test_yaml_default_is_used_when_env_unset(self, fake_yaml):
        """If no env override is set, the yaml value wins."""
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            assert features.is_enabled("tts") is True
            assert features.is_enabled("stt") is False
            assert features.is_enabled("bandit") is False
            assert features.is_enabled("holidays") is True

    def test_env_var_can_enable_a_yaml_disabled_subsystem(
        self, fake_yaml, monkeypatch,
    ):
        """Setting the env var to a truthy value flips a yaml-off flag on."""
        monkeypatch.setenv("FAKE_STT", "true")
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            assert features.is_enabled("stt") is True

    def test_env_var_can_disable_a_yaml_enabled_subsystem(
        self, fake_yaml, monkeypatch,
    ):
        """Deploy-time kill switch: env=false beats yaml=true."""
        monkeypatch.setenv("FAKE_TTS", "false")
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            assert features.is_enabled("tts") is False

    @pytest.mark.parametrize("disable_token", ["0", "false", "off", "no", ""])
    def test_all_disable_tokens_are_recognized(
        self, fake_yaml, monkeypatch, disable_token,
    ):
        monkeypatch.setenv("FAKE_TTS", disable_token)
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            assert features.is_enabled("tts") is False

    def test_unknown_subsystem_uses_default(self, fake_yaml):
        """Adding a new feature in code before yaml must not crash."""
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            assert features.is_enabled("brand_new") is True
            assert features.is_enabled("brand_new", default=False) is False

    def test_missing_yaml_defaults_to_all_enabled(self, tmp_path):
        """Graceful degradation: a missing yaml must not wedge startup —
        every feature reads as ENABLED so the server still boots."""
        missing = tmp_path / "nowhere.yaml"
        with patch.object(features, "FEATURES_YAML", missing):
            features.reload()
            assert features.is_enabled("whatever") is True

    def test_malformed_yaml_defaults_to_all_enabled(self, tmp_path):
        """A corrupted yaml is treated like a missing one — we log and
        fall back so ops can fix the file without downtime."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("not: [valid: yaml: here", encoding="utf-8")
        with patch.object(features, "FEATURES_YAML", bad):
            features.reload()
            assert features.is_enabled("stt") is True


class TestToolEnabled:
    def test_tool_not_in_denylist_is_enabled(self, fake_yaml):
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            assert features.tool_enabled("get_weather") is True
            assert features.tool_enabled("math") is True

    def test_tool_in_denylist_is_disabled(self, fake_yaml):
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            assert features.tool_enabled("tell_story") is False
            assert features.tool_enabled("wiki_random") is False

    def test_empty_denylist_means_all_tools_enabled(self, tmp_path):
        p = tmp_path / "features.yaml"
        p.write_text("tools:\n  disabled: []\n", encoding="utf-8")
        with patch.object(features, "FEATURES_YAML", p):
            features.reload()
            assert features.tool_enabled("tell_story") is True


class TestSnapshot:
    def test_snapshot_returns_resolved_flags(self, fake_yaml, monkeypatch):
        """/api/features serializes this — make sure the shape matches
        what the client expects (subsystems dict + tools.disabled list)."""
        monkeypatch.setenv("FAKE_STT", "true")   # env wins over yaml
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            snap = features.snapshot()
        assert snap["subsystems"]["stt"] is True       # env override
        assert snap["subsystems"]["tts"] is True       # yaml
        assert snap["subsystems"]["bandit"] is False   # yaml
        assert sorted(snap["tools"]["disabled"]) == ["tell_story", "wiki_random"]


class TestActiveToolSchemas:
    def test_active_tool_schemas_filters_the_denylist(
        self, fake_yaml, monkeypatch,
    ):
        """The LLM should only see tools it's allowed to call."""
        from bridge import tools as tools_mod
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            active = tools_mod.active_tool_schemas()
        names = [
            (s.get("function") or {}).get("name") for s in active
        ]
        assert "tell_story" not in names
        assert "wiki_random" not in names
        # Other tools come through
        assert "get_weather" in names
        assert "math" in names

    def test_execute_of_disabled_tool_returns_friendly_error(
        self, fake_yaml,
    ):
        """Defense in depth — even if a disabled tool's name leaks into
        the tool-call loop via stale history, dispatch refuses it."""
        from bridge import tools as tools_mod
        with patch.object(features, "FEATURES_YAML", fake_yaml):
            features.reload()
            result = tools_mod.execute("tell_story", {"kind": "joke"})
        assert "disabled on this server" in result
        assert result.startswith("[")
