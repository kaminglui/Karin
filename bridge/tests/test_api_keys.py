"""Tests for bridge.api_keys — third-party key loader.

Covers the contract that tools rely on:
  - Missing file is silent (returns None, doesn't crash)
  - Empty / whitespace-only keys count as "not set"
  - Env var `KARIN_API_KEY_<UPPER>` wins over JSON
  - Cache reload works when operator edits the file live
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bridge import api_keys


@pytest.fixture
def isolated_keys_file(tmp_path, monkeypatch):
    """Point api_keys at a tmp file + clear the module cache between tests."""
    fake_path = tmp_path / "api_keys.json"
    monkeypatch.setattr(api_keys, "_KEYS_PATH", fake_path)
    monkeypatch.setattr(api_keys, "_cache", None)
    # Also wipe any env overrides the test env might have leaked in.
    for var in list(os.environ):
        if var.startswith("KARIN_API_KEY_"):
            monkeypatch.delenv(var, raising=False)
    yield fake_path
    monkeypatch.setattr(api_keys, "_cache", None)


class TestMissingFile:
    def test_get_returns_none(self, isolated_keys_file):
        assert not isolated_keys_file.exists()
        assert api_keys.get_api_key("brave_search") is None
        assert api_keys.has_api_key("brave_search") is False

    def test_does_not_crash_on_any_name(self, isolated_keys_file):
        # Any slot name should return None silently.
        for name in ["brave_search", "nonexistent", "", "foo_bar"]:
            assert api_keys.get_api_key(name) is None


class TestConfiguredFile:
    def _write(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_get_returns_configured_key(self, isolated_keys_file):
        self._write(isolated_keys_file, {
            "brave_search": {"api_key": "secret-brave-key"},
        })
        assert api_keys.get_api_key("brave_search") == "secret-brave-key"
        assert api_keys.has_api_key("brave_search") is True

    def test_empty_string_treated_as_unset(self, isolated_keys_file):
        self._write(isolated_keys_file, {"brave_search": {"api_key": ""}})
        assert api_keys.get_api_key("brave_search") is None

    def test_whitespace_only_treated_as_unset(self, isolated_keys_file):
        self._write(isolated_keys_file, {"brave_search": {"api_key": "   "}})
        assert api_keys.get_api_key("brave_search") is None

    def test_strips_surrounding_whitespace(self, isolated_keys_file):
        self._write(isolated_keys_file, {"brave_search": {"api_key": "  key  "}})
        assert api_keys.get_api_key("brave_search") == "key"

    def test_unknown_slot_returns_none(self, isolated_keys_file):
        self._write(isolated_keys_file, {"brave_search": {"api_key": "k"}})
        assert api_keys.get_api_key("openai") is None

    def test_malformed_slot_returns_none(self, isolated_keys_file):
        # Old callers might write the key as a top-level string by mistake;
        # the loader is defensive and treats it as unset.
        self._write(isolated_keys_file, {"brave_search": "just-a-string"})
        assert api_keys.get_api_key("brave_search") is None

    def test_malformed_json_silently_empty(self, isolated_keys_file):
        isolated_keys_file.write_text("{ not valid json", encoding="utf-8")
        assert api_keys.get_api_key("brave_search") is None


class TestEnvOverride:
    def test_env_beats_file(self, isolated_keys_file, monkeypatch):
        isolated_keys_file.write_text(
            json.dumps({"brave_search": {"api_key": "file-key"}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("KARIN_API_KEY_BRAVE_SEARCH", "env-key")
        assert api_keys.get_api_key("brave_search") == "env-key"

    def test_env_alone_works_without_file(self, isolated_keys_file, monkeypatch):
        monkeypatch.setenv("KARIN_API_KEY_OPENAI", "from-env")
        assert api_keys.get_api_key("openai") == "from-env"

    def test_env_empty_falls_back_to_file(self, isolated_keys_file, monkeypatch):
        # Env var set but empty shouldn't mask the file.
        isolated_keys_file.write_text(
            json.dumps({"brave_search": {"api_key": "file-key"}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("KARIN_API_KEY_BRAVE_SEARCH", "")
        assert api_keys.get_api_key("brave_search") == "file-key"


class TestReload:
    def test_reload_picks_up_file_change(self, isolated_keys_file):
        isolated_keys_file.write_text(
            json.dumps({"brave_search": {"api_key": "v1"}}),
            encoding="utf-8",
        )
        assert api_keys.get_api_key("brave_search") == "v1"
        # Operator edits the file in place.
        isolated_keys_file.write_text(
            json.dumps({"brave_search": {"api_key": "v2"}}),
            encoding="utf-8",
        )
        # Cached — still returns v1 until reload.
        assert api_keys.get_api_key("brave_search") == "v1"
        api_keys.reload()
        assert api_keys.get_api_key("brave_search") == "v2"
