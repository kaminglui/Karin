"""Tests for bridge.notify — channel send paths, dispatcher routing,
rule engine cooldowns, and the alerts/trackers integration hooks.

No real HTTP — every channel test patches httpx.post to a lambda
that records the call. Network failures are simulated by raising
httpx.HTTPError from the patched post.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest

from bridge.notify.channels import (
    DiscordChannel,
    NtfyChannel,
    build_enabled_channels,
)
from bridge.notify.dispatcher import (
    Dispatcher,
    notify,
    reset_dispatcher,
)
from bridge.notify.events import NotifyEvent, Severity
from bridge.notify.rules import DEFAULT_RULES, Rule, RuleEngine


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
class TestDiscordChannel:
    def test_send_posts_embed_to_webhook(self):
        ch = DiscordChannel("https://example.test/wh")
        captured = {}

        class _R:
            status_code = 204
            text = ""

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["body"] = json
            return _R()

        with patch("bridge.notify.channels.httpx.post", side_effect=fake_post):
            ok = ch.send(NotifyEvent(
                kind="alerts.fired",
                title="Tornado warning",
                body="Until 2:30am",
                severity=Severity.CRITICAL,
                source="alerts",
            ))
        assert ok is True
        assert captured["url"] == "https://example.test/wh"
        assert captured["body"]["embeds"][0]["title"] == "Tornado warning"
        # Critical maps to red.
        assert captured["body"]["embeds"][0]["color"] == 0xDC2626

    def test_send_returns_false_on_http_4xx(self):
        ch = DiscordChannel("https://example.test/wh")

        class _R:
            status_code = 404
            text = "not found"

        with patch("bridge.notify.channels.httpx.post", return_value=_R()):
            ok = ch.send(NotifyEvent(kind="x", title="t", body="b"))
        assert ok is False

    def test_send_returns_false_on_network_error(self):
        ch = DiscordChannel("https://example.test/wh")
        with patch(
            "bridge.notify.channels.httpx.post",
            side_effect=httpx.ConnectError("dns down"),
        ):
            ok = ch.send(NotifyEvent(kind="x", title="t", body="b"))
        assert ok is False

    def test_from_env_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("KARIN_NOTIFY_DISCORD_WEBHOOK", raising=False)
        assert DiscordChannel.from_env() is None

    def test_from_env_builds_when_set(self, monkeypatch):
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://x.test/wh")
        ch = DiscordChannel.from_env()
        assert ch is not None
        assert isinstance(ch, DiscordChannel)


class TestNtfyChannel:
    def test_unicode_title_does_not_crash_header(self):
        """HTTP headers are ASCII-only — emoji + non-Latin chars in
        the title would crash httpx with UnicodeEncodeError. The
        channel must strip non-ASCII from the header (Title) but
        keep the full unicode body intact (renders fine on the
        phone). Regression test for the ⏰ reminder bug."""
        ch = NtfyChannel("https://ntfy.sh/test")
        captured = {}

        class _R:
            status_code = 200
            text = ""

        def fake_post(url, content=None, headers=None, timeout=None):
            captured["headers"] = headers
            captured["content"] = content
            return _R()

        with patch("bridge.notify.channels.httpx.post", side_effect=fake_post):
            ok = ch.send(NotifyEvent(
                kind="reminders.fired",
                title="⏰ call mom — 紅 emoji 🔥",
                body="Body keeps the unicode: 🔥 紅",
                severity=Severity.WARNING,
            ))
        assert ok is True
        # Header is pure ASCII (no encoding crash).
        captured["headers"]["Title"].encode("ascii")
        # Body preserved — UTF-8 bytes through.
        assert "🔥".encode("utf-8") in captured["content"]
        assert "紅".encode("utf-8") in captured["content"]

    def test_send_posts_with_priority_header(self):
        ch = NtfyChannel("https://ntfy.sh/karin-test")
        captured = {}

        class _R:
            status_code = 200
            text = ""

        def fake_post(url, content=None, headers=None, timeout=None):
            captured["url"] = url
            captured["content"] = content
            captured["headers"] = headers
            return _R()

        with patch("bridge.notify.channels.httpx.post", side_effect=fake_post):
            ok = ch.send(NotifyEvent(
                kind="alerts.fired",
                title="Severe alert",
                body="Take cover",
                severity=Severity.CRITICAL,
                source="alerts",
            ))
        assert ok is True
        assert captured["url"] == "https://ntfy.sh/karin-test"
        assert captured["content"] == b"Take cover"
        # CRITICAL → priority 5 (urgent, bypasses DND).
        assert captured["headers"]["Priority"] == "5"
        assert captured["headers"]["Title"] == "Severe alert"


# ---------------------------------------------------------------------------
# build_enabled_channels — env-keyed factory
# ---------------------------------------------------------------------------
class TestChannelFactory:
    def test_skips_unconfigured_channels(self, monkeypatch):
        monkeypatch.delenv("KARIN_NOTIFY_DISCORD_WEBHOOK", raising=False)
        monkeypatch.delenv("KARIN_NOTIFY_NTFY_TOPIC", raising=False)
        out = build_enabled_channels(["discord", "ntfy"])
        assert out == []

    def test_includes_only_configured(self, monkeypatch):
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://x.test/wh")
        monkeypatch.delenv("KARIN_NOTIFY_NTFY_TOPIC", raising=False)
        out = build_enabled_channels(["discord", "ntfy"])
        assert [c.name for c in out] == ["discord"]

    def test_unknown_channel_name_logged_and_skipped(self, monkeypatch, caplog):
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://x.test/wh")
        with caplog.at_level("WARNING"):
            out = build_enabled_channels(["discord", "telegram"])
        assert [c.name for c in out] == ["discord"]
        assert any("unknown notify channel" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Rule engine — severity threshold + cooldown dedupe
# ---------------------------------------------------------------------------
class TestRuleEngine:
    def test_no_match_returns_empty(self):
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.fired",), min_severity=Severity.WARNING,
                 channels=("discord",)),
        ))
        out = engine.routes_for(NotifyEvent(
            kind="trackers.shock", title="t", body="b", severity=Severity.WARNING,
        ))
        assert out == []

    def test_severity_below_threshold_filtered(self):
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.fired",), min_severity=Severity.WARNING,
                 channels=("discord",)),
        ))
        out = engine.routes_for(NotifyEvent(
            kind="alerts.fired", title="t", body="b", severity=Severity.INFO,
        ))
        assert out == []

    def test_at_or_above_threshold_passes(self):
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.fired",), min_severity=Severity.WARNING,
                 channels=("discord", "ntfy")),
        ))
        out = engine.routes_for(NotifyEvent(
            kind="alerts.fired", title="t", body="b", severity=Severity.CRITICAL,
        ))
        assert sorted(out) == ["discord", "ntfy"]

    def test_kind_prefix_matches(self):
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.",), min_severity=Severity.INFO,
                 channels=("discord",)),
        ))
        for kind in ("alerts.fired", "alerts.suppressed", "alerts.cleared"):
            out = engine.routes_for(NotifyEvent(kind=kind, title="t", body="b"))
            assert out == ["discord"], kind

    def test_cooldown_suppresses_repeat(self):
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.fired",), min_severity=Severity.INFO,
                 channels=("discord",), cooldown_s=60),
        ))
        e = NotifyEvent(
            kind="alerts.fired", title="t", body="b",
            payload={"dedupe_key": "alert-1"},
        )
        first = engine.routes_for(e)
        second = engine.routes_for(e)
        assert first == ["discord"]
        assert second == [], "second send within cooldown must be suppressed"

    def test_cooldown_distinct_keys_dont_collide(self):
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.fired",), min_severity=Severity.INFO,
                 channels=("discord",), cooldown_s=60),
        ))
        e1 = NotifyEvent(kind="alerts.fired", title="t", body="b",
                         payload={"dedupe_key": "a1"})
        e2 = NotifyEvent(kind="alerts.fired", title="t", body="b",
                         payload={"dedupe_key": "a2"})
        assert engine.routes_for(e1) == ["discord"]
        assert engine.routes_for(e2) == ["discord"]   # different key, fires


# ---------------------------------------------------------------------------
# Dispatcher — fan-out, swallow per-channel failures
# ---------------------------------------------------------------------------
class _StubChannel:
    def __init__(self, name, returns=True, raises=None):
        self.name = name
        self._returns = returns
        self._raises = raises
        self.calls: list[NotifyEvent] = []

    def send(self, event):
        self.calls.append(event)
        if self._raises:
            raise self._raises
        return self._returns


class TestDispatcher:
    def test_routes_to_matching_channels_only(self):
        ch_d = _StubChannel("discord")
        ch_n = _StubChannel("ntfy")
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.fired",), min_severity=Severity.INFO,
                 channels=("discord",)),
        ))
        d = Dispatcher(channels=[ch_d, ch_n], engine=engine)
        delivered = d.send(NotifyEvent(kind="alerts.fired", title="t", body="b"))
        assert delivered == ["discord"]
        assert len(ch_d.calls) == 1
        assert ch_n.calls == []

    def test_one_channel_failure_does_not_block_others(self):
        ch_bad = _StubChannel("discord", returns=False)
        ch_good = _StubChannel("ntfy")
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.fired",), min_severity=Severity.INFO,
                 channels=("discord", "ntfy")),
        ))
        d = Dispatcher(channels=[ch_bad, ch_good], engine=engine)
        delivered = d.send(NotifyEvent(kind="alerts.fired", title="t", body="b"))
        assert delivered == ["ntfy"]   # bad one was tried, returned False

    def test_channel_exception_swallowed(self):
        ch_throw = _StubChannel("discord", raises=RuntimeError("boom"))
        ch_ok = _StubChannel("ntfy")
        engine = RuleEngine(rules=(
            Rule(kinds=("alerts.fired",), min_severity=Severity.INFO,
                 channels=("discord", "ntfy")),
        ))
        d = Dispatcher(channels=[ch_throw, ch_ok], engine=engine)
        # Must not raise, and the good channel still fires.
        delivered = d.send(NotifyEvent(kind="alerts.fired", title="t", body="b"))
        assert delivered == ["ntfy"]


# ---------------------------------------------------------------------------
# Public notify() — feature-flag gating + singleton lifecycle
# ---------------------------------------------------------------------------
class TestPublicNotify:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_dispatcher()
        yield
        reset_dispatcher()

    def test_off_by_default_returns_empty(self, monkeypatch):
        monkeypatch.delenv("KARIN_NOTIFICATIONS", raising=False)
        # Force feature registry to re-read with no env override.
        from bridge import features
        features.reload()
        out = notify(NotifyEvent(kind="alerts.fired", title="t", body="b",
                                 severity=Severity.CRITICAL))
        assert out == []

    def test_on_with_no_channels_returns_empty(self, monkeypatch):
        from bridge import features
        monkeypatch.setenv("KARIN_NOTIFICATIONS", "true")
        monkeypatch.delenv("KARIN_NOTIFY_DISCORD_WEBHOOK", raising=False)
        monkeypatch.delenv("KARIN_NOTIFY_NTFY_TOPIC", raising=False)
        features.reload()
        out = notify(NotifyEvent(kind="alerts.fired", title="t", body="b",
                                 severity=Severity.CRITICAL))
        assert out == []

    def test_on_with_channel_routes(self, monkeypatch):
        from bridge import features
        monkeypatch.setenv("KARIN_NOTIFICATIONS", "true")
        monkeypatch.setenv("KARIN_NOTIFY_DISCORD_WEBHOOK", "https://x.test/wh")
        monkeypatch.delenv("KARIN_NOTIFY_NTFY_TOPIC", raising=False)
        features.reload()

        # Patch the underlying httpx.post so we don't actually go out.
        class _R:
            status_code = 204
            text = ""
        with patch("bridge.notify.channels.httpx.post", return_value=_R()):
            out = notify(NotifyEvent(
                kind="alerts.fired", title="t", body="b",
                severity=Severity.CRITICAL,
            ))
        assert out == ["discord"]
