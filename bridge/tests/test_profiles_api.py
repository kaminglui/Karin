"""Tests for the /api/profiles endpoints (Phase H.c).

Covers:
  - GET /api/profiles returns list + active
  - POST /api/profiles creates + is idempotent
  - POST /api/profiles rejects unsafe names with 400
  - POST /api/profiles/active persists the switch
  - POST /api/profiles/active rejects unknown / invalid names
  - restart_required flag is always True on switch (no hot-swap in V1)
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """Fresh FastAPI app mounted on the panels_api router, with a
    tmp-rooted REPO_ROOT so profile writes never leave the tmp dir."""
    from web import panels_api

    monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
    monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
    monkeypatch.delenv("KARIN_PROFILE", raising=False)

    app = FastAPI()
    app.include_router(panels_api.router)
    return TestClient(app), tmp_path


class TestProfilesList:
    def test_empty_bootstraps_default(self, client):
        c, tmp = client
        r = c.get("/api/profiles")
        assert r.status_code == 200
        data = r.json()
        # active_profile() auto-creates default on first call, so it
        # shows up in both the list and the active field.
        assert data["active"] == "default"
        assert "default" in data["profiles"]

    def test_multiple_profiles_are_sorted(self, client):
        from bridge import profiles
        c, _ = client
        profiles.create_profile("work")
        profiles.create_profile("family")
        r = c.get("/api/profiles")
        names = r.json()["profiles"]
        # Alphabetical — matches the registry contract.
        assert names == sorted(names)
        assert {"default", "family", "work"}.issubset(set(names))


class TestProfilesCreate:
    def test_creates_new_profile(self, client):
        c, tmp = client
        r = c.post("/api/profiles", json={"name": "work"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["name"] == "work"
        # Directory exists on disk.
        assert (tmp / "data" / "profiles" / "work").is_dir()

    def test_idempotent(self, client):
        c, _ = client
        c.post("/api/profiles", json={"name": "work"})
        r = c.post("/api/profiles", json={"name": "work"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_case_folded(self, client):
        c, _ = client
        r = c.post("/api/profiles", json={"name": "Work"})
        assert r.status_code == 200
        # "Work" gets folded to "work" — same canonical name on disk.
        assert r.json()["name"] == "work"

    @pytest.mark.parametrize("bad", [
        "../etc",
        "a b",
        "",
        "-leading",
        "has/slash",
    ])
    def test_rejects_unsafe_name(self, client, bad):
        c, _ = client
        r = c.post("/api/profiles", json={"name": bad})
        assert r.status_code == 400


class TestProfilesSetActive:
    def test_switch_persists(self, client):
        from bridge import profiles
        c, _ = client
        profiles.create_profile("work")
        r = c.post("/api/profiles/active", json={"name": "work"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["active"] == "work"
        # Every switch is "restart required" in V1 — no hot-swap.
        assert body["restart_required"] is True
        # And the active-profile resolver agrees.
        assert profiles.active_profile().name == "work"

    def test_switch_to_unknown_404_ish(self, client):
        c, _ = client
        r = c.post("/api/profiles/active", json={"name": "ghost"})
        # Mapped via ProfileNameError -> 400. The UI treats 4xx as
        # "nothing happened, show an error."
        assert r.status_code == 400

    def test_switch_to_invalid_name_400(self, client):
        c, _ = client
        r = c.post("/api/profiles/active", json={"name": "../etc"})
        assert r.status_code == 400

    def test_case_folded_switch(self, client):
        from bridge import profiles
        c, _ = client
        profiles.create_profile("work")
        r = c.post("/api/profiles/active", json={"name": "WORK"})
        assert r.status_code == 200
        assert r.json()["active"] == "work"
