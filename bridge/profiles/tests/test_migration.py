"""Tests for the Phase H legacy->profile migration runner.

Covers:
  - Single-file moves (news/tracker prefs, alerts state, learned)
  - Directory moves (reminders, conversations, memory, feedback, calendar)
  - user_location copy from assistant.yaml
  - Idempotence (second run is a no-op)
  - Non-clobbering (destination files already in place are preserved)
  - Fresh-install path (nothing to migrate)

Every test uses a tmp_path 'repo' so the real data/ dir is never touched.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    """Point both bridge.utils.REPO_ROOT and bridge.profiles.REPO_ROOT
    at a fresh tmp dir. The migration module also imports REPO_ROOT at
    module scope, so patch that too."""
    monkeypatch.setattr("bridge.utils.REPO_ROOT", tmp_path)
    monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
    monkeypatch.setattr("bridge.profiles.migration.REPO_ROOT", tmp_path)
    monkeypatch.delenv("KARIN_PROFILE", raising=False)
    yield tmp_path


def _seed_legacy(tmp_path, rel_path: str, content: str) -> None:
    """Write a file at ``tmp_path/rel_path``, creating parents as needed."""
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# --- fresh install ------------------------------------------------------


class TestFreshInstall:
    def test_no_legacy_data_no_op(self, tmp_repo):
        from bridge.profiles.migration import run_legacy_migration
        ran = run_legacy_migration()
        assert ran is True   # marker gets written even on empty repo
        # But nothing was actually moved.
        default_root = tmp_repo / "data" / "profiles" / "default"
        assert default_root.is_dir()
        marker = default_root / ".migrated_from_legacy"
        assert marker.is_file()


# --- idempotence --------------------------------------------------------


class TestIdempotence:
    def test_second_call_returns_false(self, tmp_repo):
        from bridge.profiles.migration import run_legacy_migration
        assert run_legacy_migration() is True
        assert run_legacy_migration() is False

    def test_second_call_does_not_reclobber(self, tmp_repo):
        from bridge.profiles.migration import run_legacy_migration
        # Seed legacy file, migrate once.
        _seed_legacy(tmp_repo, "data/news/preferences.json",
                     '{"enabled": true, "v": 1}')
        run_legacy_migration()
        # User edits the migrated file after migration completes.
        prefs = tmp_repo / "data/profiles/default/news/preferences.json"
        prefs.write_text('{"enabled": true, "v": 2}', encoding="utf-8")
        # Second migration must not touch the edit.
        run_legacy_migration()
        assert json.loads(prefs.read_text(encoding="utf-8"))["v"] == 2


# --- single-file moves --------------------------------------------------


class TestSingleFileMoves:
    def test_news_prefs_moves(self, tmp_repo):
        _seed_legacy(tmp_repo, "data/news/preferences.json",
                     '{"enabled": true}')
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        src = tmp_repo / "data/news/preferences.json"
        dst = tmp_repo / "data/profiles/default/news/preferences.json"
        assert not src.exists(), "source should have been moved"
        assert dst.is_file()
        assert json.loads(dst.read_text(encoding="utf-8")) == {"enabled": True}

    def test_tracker_prefs_moves(self, tmp_repo):
        _seed_legacy(tmp_repo, "data/trackers/tracker_preferences.json",
                     '{"order": ["fx", "energy"]}')
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        assert (tmp_repo / "data/profiles/default/trackers/tracker_preferences.json").is_file()

    def test_all_alerts_state_moves(self, tmp_repo):
        _seed_legacy(tmp_repo, "bridge/alerts/data/alerts.jsonl", '{"a":1}\n')
        _seed_legacy(tmp_repo, "bridge/alerts/data/cooldowns.json", '{}')
        _seed_legacy(tmp_repo, "bridge/alerts/data/advisory_state.json", '{}')
        _seed_legacy(tmp_repo, "bridge/alerts/data/threat_decisions.json", '{}')
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        dst = tmp_repo / "data/profiles/default/alerts"
        for f in ("alerts.jsonl", "cooldowns.json", "advisory_state.json", "threat_decisions.json"):
            assert (dst / f).is_file(), f"{f} should have moved into profile alerts/"

    def test_learned_keywords_moves(self, tmp_repo):
        _seed_legacy(tmp_repo, "bridge/news/data/learned_keywords.json",
                     '{"Disasters": {}}')
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        dst = tmp_repo / "data/profiles/default/news/learned_keywords.json"
        assert dst.is_file()

    def test_existing_destination_is_not_clobbered(self, tmp_repo):
        """If the profile already has a file at the destination path
        (user created it before migration ran — shouldn't happen, but
        we defend against it), migration must not overwrite."""
        from bridge.profiles.migration import run_legacy_migration
        # Seed both legacy and destination.
        _seed_legacy(tmp_repo, "data/news/preferences.json",
                     '{"from": "legacy"}')
        dst = tmp_repo / "data/profiles/default/news/preferences.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text('{"from": "profile"}', encoding="utf-8")
        run_legacy_migration()
        # Profile version survives; legacy source was left alone.
        assert json.loads(dst.read_text(encoding="utf-8")) == {"from": "profile"}
        # Legacy source stays (not clobbered on either side).
        assert (tmp_repo / "data/news/preferences.json").is_file()


# --- directory moves ----------------------------------------------------


class TestDirectoryMoves:
    def test_reminders_dir_contents_move(self, tmp_repo):
        _seed_legacy(tmp_repo, "data/reminders/reminders.db", "sqlite_bytes")
        _seed_legacy(tmp_repo, "data/reminders/extra.log", "x")
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        dst = tmp_repo / "data/profiles/default/reminders"
        assert (dst / "reminders.db").is_file()
        assert (dst / "extra.log").is_file()

    def test_conversations_move(self, tmp_repo):
        _seed_legacy(tmp_repo, "data/conversations/20260101T000000Z.json", "{}")
        _seed_legacy(tmp_repo, "data/conversations/index.json", "{}")
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        dst = tmp_repo / "data/profiles/default/conversations"
        assert (dst / "20260101T000000Z.json").is_file()
        assert (dst / "index.json").is_file()

    def test_memory_files_move(self, tmp_repo):
        _seed_legacy(tmp_repo, "data/memory/user.md", "user facts")
        _seed_legacy(tmp_repo, "data/memory/agent.md", "agent skills")
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        dst = tmp_repo / "data/profiles/default/memory"
        assert (dst / "user.md").read_text(encoding="utf-8") == "user facts"
        assert (dst / "agent.md").read_text(encoding="utf-8") == "agent skills"

    def test_feedback_moves(self, tmp_repo):
        _seed_legacy(tmp_repo, "data/feedback/entries.jsonl", '{"v":1}\n')
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        assert (tmp_repo / "data/profiles/default/feedback/entries.jsonl").is_file()

    def test_calendar_moves(self, tmp_repo):
        _seed_legacy(tmp_repo, "data/calendar/notified.db", "sqlite")
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        assert (tmp_repo / "data/profiles/default/calendar/notified.db").is_file()


# --- user_location migration -------------------------------------------


class TestUserLocationMigration:
    def test_location_copied_from_yaml(self, tmp_repo):
        _seed_legacy(
            tmp_repo, "config/assistant.yaml",
            "user_location:\n"
            "  city: University Park\n"
            "  region: Pennsylvania\n"
            "  country: United States\n"
            "  latitude: 40.7982\n"
            "  longitude: -77.8599\n",
        )
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        prefs = tmp_repo / "data/profiles/default/preferences.json"
        data = json.loads(prefs.read_text(encoding="utf-8"))
        loc = data["user_location"]
        assert loc["city"] == "University Park"
        assert loc["region"] == "Pennsylvania"
        assert loc["latitude"] == 40.7982

    def test_yaml_untouched_after_migration(self, tmp_repo):
        """The yaml block must stay in place — it's the readable
        fallback for users SSH-ing into the box who haven't learned
        about profiles yet."""
        yaml_body = (
            "user_location:\n"
            "  city: PHL\n"
        )
        _seed_legacy(tmp_repo, "config/assistant.yaml", yaml_body)
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        assert (tmp_repo / "config/assistant.yaml").read_text(
            encoding="utf-8"
        ) == yaml_body

    def test_existing_profile_location_not_clobbered(self, tmp_repo):
        """If the profile already has a user_location set (e.g. from a
        prior UI save), the yaml copy must NOT overwrite it."""
        from bridge.profiles import create_profile, save_profile_preferences
        _seed_legacy(
            tmp_repo, "config/assistant.yaml",
            "user_location:\n  city: YAML\n",
        )
        p = create_profile("default")
        save_profile_preferences({"user_location": {"city": "Profile"}}, p)
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        data = json.loads(p.preferences_path.read_text(encoding="utf-8"))
        assert data["user_location"]["city"] == "Profile"

    def test_no_yaml_is_not_an_error(self, tmp_repo):
        """Fresh install with no assistant.yaml at all shouldn't crash
        migration."""
        from bridge.profiles.migration import run_legacy_migration
        ran = run_legacy_migration()
        assert ran is True   # marker still written

    def test_empty_yaml_user_location_skipped(self, tmp_repo):
        _seed_legacy(
            tmp_repo, "config/assistant.yaml",
            "user_location:\n  city: ''\n",
        )
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        # preferences.json seeded as '{}' by create_profile; all-empty
        # yaml fields get filtered out before save, so no user_location
        # key should be written.
        data = json.loads((tmp_repo / "data/profiles/default/preferences.json").read_text(encoding="utf-8"))
        assert "user_location" not in data


# --- global content NOT moved ------------------------------------------


class TestGlobalContentStaysGlobal:
    """Articles, clusters, market snapshots etc. are GLOBAL — same for
    every profile. The migration must leave them alone."""

    def test_articles_not_moved(self, tmp_repo):
        _seed_legacy(tmp_repo, "bridge/news/data/articles.json", '[]')
        _seed_legacy(tmp_repo, "bridge/news/data/clusters.json", '[]')
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        # Still at original location.
        assert (tmp_repo / "bridge/news/data/articles.json").is_file()
        assert (tmp_repo / "bridge/news/data/clusters.json").is_file()
        # Not present under profile dir.
        assert not (tmp_repo / "data/profiles/default/news/articles.json").exists()

    def test_tracker_snapshots_not_moved(self, tmp_repo):
        _seed_legacy(tmp_repo, "bridge/trackers/data/trackers.json", '{}')
        from bridge.profiles.migration import run_legacy_migration
        run_legacy_migration()
        assert (tmp_repo / "bridge/trackers/data/trackers.json").is_file()
