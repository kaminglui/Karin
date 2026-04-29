"""Tests for bridge.voice — sovits container lifecycle helpers.

The real module talks to /var/run/docker.sock via httpx's unix-socket
transport. We swap the transport for an httpx.MockTransport here so
the tests run anywhere without a docker daemon.
"""
from __future__ import annotations

import json

import httpx
import pytest

import bridge.voice as voice_mod


def _install_mock(monkeypatch, responder):
    """Patch _client() to return a client wired to a MockTransport."""
    def _make():
        return httpx.Client(
            transport=httpx.MockTransport(responder),
            base_url="http://docker",
            timeout=5.0,
        )
    monkeypatch.setattr(voice_mod, "_client", _make)


def _patch_ollama_unload(monkeypatch, loaded_names: list[str] | None = None, raise_on_ps: bool = False):
    """Install a fake Ollama client so start() doesn't hit the network.
    Returns a dict that records which model names got an unload POST.
    """
    recorded = {"unloaded": []}
    def _unloader(timeout_s: float = 5.0):
        if raise_on_ps:
            return []
        for n in (loaded_names or []):
            recorded["unloaded"].append(n)
        return list(recorded["unloaded"])
    monkeypatch.setattr(voice_mod, "_unload_ollama_models", _unloader)
    # Skip the real sleep so the test doesn't block.
    monkeypatch.setattr(voice_mod, "_POST_UNLOAD_SETTLE_S", 0.0)
    return recorded


# ---- status() ------------------------------------------------------------


class TestStatus:
    def test_running_healthy(self, monkeypatch):
        def responder(req):
            return httpx.Response(200, json={
                "State": {"Running": True, "Health": {"Status": "healthy"}},
            })
        _install_mock(monkeypatch, responder)
        s = voice_mod.status()
        assert s == {"running": True, "health": "healthy", "exists": True, "error": None}

    def test_running_starting(self, monkeypatch):
        def responder(req):
            return httpx.Response(200, json={
                "State": {"Running": True, "Health": {"Status": "starting"}},
            })
        _install_mock(monkeypatch, responder)
        assert voice_mod.status()["health"] == "starting"

    def test_stopped(self, monkeypatch):
        def responder(req):
            return httpx.Response(200, json={"State": {"Running": False}})
        _install_mock(monkeypatch, responder)
        s = voice_mod.status()
        assert s["running"] is False
        assert s["health"] is None
        assert s["exists"] is True

    def test_container_not_found(self, monkeypatch):
        def responder(req):
            return httpx.Response(404, json={"message": "no such container"})
        _install_mock(monkeypatch, responder)
        s = voice_mod.status()
        assert s["exists"] is False
        assert s["running"] is False

    def test_docker_unreachable(self, monkeypatch):
        def responder(req):
            raise httpx.ConnectError("socket missing")
        _install_mock(monkeypatch, responder)
        s = voice_mod.status()
        assert s["error"] is not None
        assert s["running"] is False


# ---- start() / stop() ----------------------------------------------------


class TestStartStop:
    def _stub_running_container(self, monkeypatch, *, running_after_action: bool):
        """Mock that handles both POST start/stop AND the follow-up GET."""
        calls: list[str] = []

        def responder(req):
            calls.append(f"{req.method} {req.url.path}")
            if req.method == "POST":
                # Docker returns 204 on success
                return httpx.Response(204)
            # GET status follow-up
            return httpx.Response(200, json={
                "State": {
                    "Running": running_after_action,
                    "Health": {"Status": "healthy" if running_after_action else "exited"},
                },
            })
        _install_mock(monkeypatch, responder)
        return calls

    def test_start_success(self, monkeypatch):
        _patch_ollama_unload(monkeypatch, loaded_names=["karin-tuned:latest"])
        calls = self._stub_running_container(monkeypatch, running_after_action=True)
        s = voice_mod.start()
        assert s["ok"] is True
        assert s["running"] is True
        assert any(c.endswith("/start") for c in calls)

    def test_start_unloads_ollama_before_starting_sovits(self, monkeypatch):
        """Regression guard: the LLM-eviction step must run before the
        docker /start POST, otherwise sovits OOMs mid-load on Jetson."""
        rec = _patch_ollama_unload(monkeypatch, loaded_names=["karin-tuned:latest", "nomic-embed-text:latest"])
        self._stub_running_container(monkeypatch, running_after_action=True)
        voice_mod.start()
        assert "karin-tuned:latest" in rec["unloaded"]
        assert "nomic-embed-text:latest" in rec["unloaded"]

    def test_start_succeeds_when_ollama_unreachable(self, monkeypatch):
        """Unload is best-effort — ollama being down shouldn't block
        sovits start (sovits might still fit on a fresh boot)."""
        _patch_ollama_unload(monkeypatch, raise_on_ps=True)
        self._stub_running_container(monkeypatch, running_after_action=True)
        s = voice_mod.start()
        assert s["ok"] is True

    def test_start_already_running_idempotent(self, monkeypatch):
        """Docker returns 304 if you start an already-running container."""
        _patch_ollama_unload(monkeypatch)
        def responder(req):
            if req.method == "POST":
                return httpx.Response(304)
            return httpx.Response(200, json={
                "State": {"Running": True, "Health": {"Status": "healthy"}},
            })
        _install_mock(monkeypatch, responder)
        s = voice_mod.start()
        assert s["ok"] is True
        assert s["running"] is True

    def test_stop_success(self, monkeypatch):
        calls = self._stub_running_container(monkeypatch, running_after_action=False)
        s = voice_mod.stop()
        assert s["ok"] is True
        assert s["running"] is False
        assert any(c.endswith("/stop") for c in calls)

    def test_action_failure_propagates_error(self, monkeypatch):
        _patch_ollama_unload(monkeypatch)
        def responder(req):
            if req.method == "POST":
                return httpx.Response(500, json={"message": "engine error"})
            return httpx.Response(200, json={"State": {"Running": False}})
        _install_mock(monkeypatch, responder)
        s = voice_mod.start()
        assert s["ok"] is False
        assert s["error"]

    def test_socket_missing_returns_structured_error(self, monkeypatch):
        _patch_ollama_unload(monkeypatch)
        def responder(req):
            raise httpx.ConnectError("/var/run/docker.sock missing")
        _install_mock(monkeypatch, responder)
        s = voice_mod.start()
        assert s["ok"] is False
        assert "docker.sock" in s["error"] or "missing" in s["error"]
