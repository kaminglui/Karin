"""Smoke tests for the /api/preferences/{news,trackers} endpoints.

Covers the round-trip the Settings panel relies on:

* GET → returns current contents (or {} when no file yet)
* POST → writes the JSON to disk, pretty-printed, with a trailing
  newline; reset_default_service is called so subsequent reads
  reflect the new state without a process restart.

The endpoints write to filesystem paths inside `bridge/<sub>/config/`.
We monkey-patch those module-level paths to point at a tmp directory
so tests don't pollute the repo's real config files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """Mount the panels_api router on a fresh FastAPI app and redirect
    the writable + legacy prefs paths into tmp_path so a test write
    doesn't leak into the repo.

    Phase H: the write path resolves through ``active_profile()`` at
    request time, so we monkey-patch the resolver functions to return
    fixed tmp paths. Legacy paths are still module-level constants so
    we patch those directly."""
    from web import panels_api

    news_path = tmp_path / "news_prefs.json"
    tracker_path = tmp_path / "tracker_prefs.json"
    legacy_news_writable = tmp_path / "legacy_news_writable.json"
    legacy_news_config = tmp_path / "legacy_news_config.json"
    legacy_tracker_writable = tmp_path / "legacy_tracker_writable.json"
    legacy_tracker_config = tmp_path / "legacy_tracker_config.json"
    monkeypatch.setattr(panels_api, "_news_prefs_write_path", lambda: news_path)
    monkeypatch.setattr(panels_api, "_tracker_prefs_write_path", lambda: tracker_path)
    monkeypatch.setattr(panels_api, "_NEWS_PREFS_LEGACY_WRITABLE", legacy_news_writable)
    monkeypatch.setattr(panels_api, "_NEWS_PREFS_LEGACY_CONFIG", legacy_news_config)
    monkeypatch.setattr(panels_api, "_TRACKER_PREFS_LEGACY_WRITABLE", legacy_tracker_writable)
    monkeypatch.setattr(panels_api, "_TRACKER_PREFS_LEGACY_CONFIG", legacy_tracker_config)

    app = FastAPI()
    app.include_router(panels_api.router)
    return TestClient(app), news_path, tracker_path


# ---- GET round-trip -------------------------------------------------------
class TestPreferencesGet:
    def test_news_returns_empty_when_no_file(self, client):
        c, _, _ = client
        r = c.get("/api/preferences/news")
        assert r.status_code == 200
        assert r.json() == {"data": {}}

    def test_trackers_returns_empty_when_no_file(self, client):
        c, _, _ = client
        r = c.get("/api/preferences/trackers")
        assert r.status_code == 200
        assert r.json() == {"data": {}}

    def test_news_returns_existing_file_contents(self, client):
        c, news_path, _ = client
        news_path.write_text(json.dumps({"enabled": True, "regions": []}),
                             encoding="utf-8")
        r = c.get("/api/preferences/news")
        assert r.status_code == 200
        assert r.json()["data"] == {"enabled": True, "regions": []}

    def test_news_falls_back_to_legacy_path(self, client, monkeypatch, tmp_path):
        """When only the legacy hand-edited path exists (no profile
        copy), the GET still surfaces it. This is the migration path:
        existing setups keep working without the user having to re-save
        through the UI."""
        from web import panels_api
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps({"enabled": True, "from": "legacy"}),
                          encoding="utf-8")
        monkeypatch.setattr(panels_api, "_NEWS_PREFS_LEGACY_CONFIG", legacy)
        c, _, _ = client
        rsp = c.get("/api/preferences/news")
        assert rsp.json()["data"] == {"enabled": True, "from": "legacy"}

    def test_writable_path_takes_priority_over_legacy(self, client, monkeypatch, tmp_path):
        """When both the profile-resolved write path and the legacy
        path exist, the profile (UI-saved) version wins — otherwise a
        UI save would be silently overridden by stale legacy data."""
        from web import panels_api
        c, news_path, _ = client
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps({"from": "legacy"}), encoding="utf-8")
        news_path.write_text(json.dumps({"from": "writable"}), encoding="utf-8")
        monkeypatch.setattr(panels_api, "_NEWS_PREFS_LEGACY_CONFIG", legacy)
        rsp = c.get("/api/preferences/news")
        assert rsp.json()["data"] == {"from": "writable"}


# ---- POST writes to disk + reloads singleton -----------------------------
class TestPreferencesPost:
    def test_news_post_writes_pretty_json(self, client, monkeypatch):
        c, news_path, _ = client
        # Stub the singleton-reset so the test doesn't depend on the
        # news subsystem actually being importable / configured.
        from bridge.news import service as _news_svc
        called = []
        monkeypatch.setattr(
            _news_svc, "reset_default_service",
            lambda: called.append("reset"),
        )
        body = {"data": {"enabled": True, "regions": [{"id": "us"}]}}
        r = c.post("/api/preferences/news", json=body)
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # File on disk matches what we sent — pretty-printed.
        assert news_path.exists()
        on_disk = json.loads(news_path.read_text(encoding="utf-8"))
        assert on_disk == body["data"]
        # The raw text is indented (not a single-line dump), so a
        # human SSHing to the box can read it without reformat.
        assert "\n" in news_path.read_text(encoding="utf-8")
        # And the singleton reset was invoked so the change takes
        # effect on the very next get_news call.
        assert called == ["reset"]

    def test_trackers_post_writes_and_resets(self, client, monkeypatch):
        c, _, tracker_path = client
        from bridge.trackers import service as _tracker_svc
        called = []
        monkeypatch.setattr(
            _tracker_svc, "reset_default_tracker_service",
            lambda: called.append("reset"),
        )
        body = {"data": {"enabled": True, "home_state": "CA"}}
        r = c.post("/api/preferences/trackers", json=body)
        assert r.status_code == 200
        on_disk = json.loads(tracker_path.read_text(encoding="utf-8"))
        assert on_disk == body["data"]
        assert called == ["reset"]

    def test_post_rejects_non_object_body(self, client):
        c, _, _ = client
        # FastAPI / Pydantic enforces the BaseModel shape — a list
        # in the `data` slot returns 422 (validation error). The
        # endpoint never sees the bad payload.
        r = c.post("/api/preferences/news", json={"data": ["not", "an", "object"]})
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /api/reminders/{id}/cancel + /api/reminders/upcoming
# ---------------------------------------------------------------------------
@pytest.fixture
def reminders_client(tmp_path: Path, monkeypatch):
    """Fresh FastAPI app + ReminderStore redirected into tmp_path so
    cancel / list tests don't touch a shared DB."""
    from web import panels_api
    from bridge.reminders import api as rem_api
    from bridge.reminders.store import ReminderStore

    store = ReminderStore(tmp_path / "reminders.db")
    rem_api.reset_default_store()
    monkeypatch.setattr(rem_api, "_get_store", lambda: store)

    app = FastAPI()
    app.include_router(panels_api.router)
    return TestClient(app), store


class TestRemindersEndpoints:
    def test_cancel_removes_existing_reminder(self, reminders_client):
        from datetime import datetime, timedelta, timezone
        from bridge.reminders import api as rem_api
        c, store = reminders_client
        rem = rem_api.create_reminder(
            trigger_at=datetime.now(timezone.utc) + timedelta(hours=1),
            message="cancellable", _store=store,
        )
        r = c.post(f"/api/reminders/{rem.id}/cancel")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # Store confirms the row is gone.
        assert store.get(rem.id) is None

    def test_cancel_unknown_id_is_404(self, reminders_client):
        c, _ = reminders_client
        r = c.post("/api/reminders/nonexistent-id-xyz/cancel")
        assert r.status_code == 404

    def test_upcoming_lists_pending_by_trigger_asc(self, reminders_client):
        from datetime import datetime, timedelta, timezone
        from bridge.reminders import api as rem_api
        c, store = reminders_client
        now = datetime.now(timezone.utc)
        late = rem_api.create_reminder(
            trigger_at=now + timedelta(hours=2), message="late", _store=store,
        )
        soon = rem_api.create_reminder(
            trigger_at=now + timedelta(minutes=30), message="soon", _store=store,
        )
        r = c.get("/api/reminders/upcoming")
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()["reminders"]]
        assert ids == [soon.id, late.id]

    def test_upcoming_excludes_delivered(self, reminders_client):
        from datetime import datetime, timedelta, timezone
        from bridge.reminders import api as rem_api
        c, store = reminders_client
        rem_api.create_reminder(
            trigger_at=datetime.now(timezone.utc) + timedelta(hours=1),
            message="still pending", _store=store,
        )
        # Past-due reminder, already fired + marked.
        past = rem_api.create_reminder(
            trigger_at=datetime.now(timezone.utc) - timedelta(hours=1),
            message="already fired", _store=store,
        )
        store.mark_delivered(past.id)
        r = c.get("/api/reminders/upcoming")
        ids = [x["id"] for x in r.json()["reminders"]]
        assert past.id not in ids
