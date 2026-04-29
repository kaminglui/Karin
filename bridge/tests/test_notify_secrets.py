"""Tests for bridge.notify.secrets — file-first / env-fallback resolution
+ the round-trip the Settings panel relies on."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge.notify import secrets as notify_secrets


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Point the secrets module at a tmp file per test so a real
    repo's notify_secrets.json (if any) can't leak in or be touched."""
    p = tmp_path / "notify_secrets.json"
    monkeypatch.setattr(notify_secrets, "SECRETS_PATH", p)
    notify_secrets.reload()
    yield p
    notify_secrets.reload()


# --- file precedence over env ----------------------------------------------
class TestPrecedence:
    def test_env_used_when_file_missing(self, monkeypatch):
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://env.test/wh")
        assert notify_secrets.get_secret("discord_webhook") == "https://env.test/wh"

    def test_file_overrides_env(self, _isolated, monkeypatch):
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://env.test/wh")
        notify_secrets.write_secrets({"discord_webhook": "https://file.test/wh"})
        assert notify_secrets.get_secret("discord_webhook") == "https://file.test/wh"

    def test_empty_file_value_falls_back_to_env(self, _isolated, monkeypatch):
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://env.test/wh")
        notify_secrets.write_secrets({"discord_webhook": "   "})  # whitespace only
        assert notify_secrets.get_secret("discord_webhook") == "https://env.test/wh"

    def test_unknown_key_returns_empty_string(self):
        assert notify_secrets.get_secret("not_a_real_key") == ""

    def test_returns_empty_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("KARIN_NOTIFY_DISCORD_WEBHOOK", raising=False)
        assert notify_secrets.get_secret("discord_webhook") == ""


# --- write_secrets + reload --------------------------------------------------
class TestWriteAndReload:
    def test_write_persists_pretty_json(self, _isolated):
        notify_secrets.write_secrets({"discord_webhook": "https://x.test/wh"})
        on_disk = json.loads(_isolated.read_text(encoding="utf-8"))
        assert on_disk == {"discord_webhook": "https://x.test/wh"}
        # Pretty-printed (indent=2 → newline somewhere in the body).
        assert "\n" in _isolated.read_text(encoding="utf-8")

    def test_write_replaces_full_object(self, _isolated):
        notify_secrets.write_secrets({"discord_webhook": "a", "ntfy_topic": "b"})
        notify_secrets.write_secrets({"discord_webhook": "c"})
        out = json.loads(_isolated.read_text(encoding="utf-8"))
        assert out == {"discord_webhook": "c"}   # ntfy_topic dropped

    def test_reload_picks_up_external_edit(self, _isolated):
        _isolated.write_text(json.dumps({"ntfy_topic": "first"}), encoding="utf-8")
        assert notify_secrets.get_secret("ntfy_topic") == "first"
        _isolated.write_text(json.dumps({"ntfy_topic": "second"}), encoding="utf-8")
        # Without reload, lru_cache still returns "first".
        assert notify_secrets.get_secret("ntfy_topic") == "first"
        notify_secrets.reload()
        assert notify_secrets.get_secret("ntfy_topic") == "second"


# --- malformed file fallbacks ----------------------------------------------
class TestMalformedFile:
    def test_invalid_json_falls_back(self, _isolated, monkeypatch, caplog):
        _isolated.write_text("{ not valid", encoding="utf-8")
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://env.test/wh")
        notify_secrets.reload()
        with caplog.at_level("WARNING"):
            v = notify_secrets.get_secret("discord_webhook")
        # Env still works.
        assert v == "https://env.test/wh"
        assert any("invalid" in r.message for r in caplog.records)

    def test_non_object_root_falls_back(self, _isolated, caplog):
        _isolated.write_text("[\"not\", \"an\", \"object\"]", encoding="utf-8")
        notify_secrets.reload()
        with caplog.at_level("WARNING"):
            assert notify_secrets.read_secrets() == {}


# --- channel factory uses the new resolver ---------------------------------
class TestChannelFactory:
    def test_discord_from_env_picks_file_value(self, _isolated, monkeypatch):
        from bridge.notify.channels import DiscordChannel
        monkeypatch.delenv("KARIN_NOTIFY_DISCORD_WEBHOOK", raising=False)
        notify_secrets.write_secrets({"discord_webhook": "https://only-on-disk.test/wh"})
        ch = DiscordChannel.from_env()
        assert ch is not None

    def test_ntfy_from_env_picks_file_value(self, _isolated, monkeypatch):
        from bridge.notify.channels import NtfyChannel
        monkeypatch.delenv("KARIN_NOTIFY_NTFY_TOPIC", raising=False)
        notify_secrets.write_secrets({"ntfy_topic": "https://ntfy.test/x"})
        ch = NtfyChannel.from_env()
        assert ch is not None
