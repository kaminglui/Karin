"""Tests for the Phase H profile isolation scaffolding.

Covers validation (path-traversal rejection, case folding), on-disk
registry behaviour (list, create idempotence, existence), and the
active-profile resolver priority (env > file > default).

Every test monkey-patches ``bridge.utils.REPO_ROOT`` to a tmp dir so
nothing here can touch your real data/profiles/ directory.
"""
from __future__ import annotations

import pytest

from bridge import profiles


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    """Point the profiles module at a fresh tmp 'repo root' for each
    test. Prevents cross-test leakage and keeps real data untouched."""
    monkeypatch.setattr("bridge.profiles.REPO_ROOT", tmp_path)
    monkeypatch.delenv("KARIN_PROFILE", raising=False)
    yield tmp_path


# --- name validation -----------------------------------------------------


class TestValidateName:
    def test_simple_lowercase_passes(self):
        assert profiles.validate_name("work") == "work"

    def test_casing_is_folded(self):
        # "Work" and "work" are the same profile.
        assert profiles.validate_name("Work") == "work"

    def test_underscore_and_dash(self):
        assert profiles.validate_name("my_work-2") == "my_work-2"

    def test_digits_ok(self):
        assert profiles.validate_name("p2") == "p2"

    @pytest.mark.parametrize("bad", [
        "",              # empty
        "-leading",      # dash-leading (CLI unfriendly)
        "../etc",        # path traversal
        "a/b",           # slash
        "a\\b",          # backslash
        "a.b",           # dots (hidden dirs, traversal)
        "has space",     # internal space
        "a" * 33,        # too long
        "emoji_\u2603",  # non-ASCII
    ])
    def test_unsafe_names_rejected(self, bad):
        with pytest.raises(profiles.ProfileNameError):
            profiles.validate_name(bad)

    def test_non_str_rejected(self):
        with pytest.raises(profiles.ProfileNameError):
            profiles.validate_name(123)  # type: ignore[arg-type]


# --- registry ------------------------------------------------------------


class TestRegistry:
    def test_list_empty_when_no_dir(self, tmp_repo):
        assert profiles.list_profiles() == []

    def test_create_then_list(self, tmp_repo):
        profiles.create_profile("work")
        profiles.create_profile("family")
        assert profiles.list_profiles() == ["family", "work"]

    def test_create_is_idempotent(self, tmp_repo):
        a = profiles.create_profile("work")
        b = profiles.create_profile("work")
        assert a == b
        assert a.root.is_dir()

    def test_create_seeds_preferences_file(self, tmp_repo):
        p = profiles.create_profile("work")
        assert p.preferences_path.is_file()
        assert p.preferences_path.read_text(encoding="utf-8").strip() == "{}"

    def test_create_preserves_existing_preferences(self, tmp_repo):
        # First create writes "{}" — subsequent create must not clobber
        # whatever the user has configured since then.
        p = profiles.create_profile("work")
        p.preferences_path.write_text('{"a": 1}', encoding="utf-8")
        profiles.create_profile("work")
        assert p.preferences_path.read_text(encoding="utf-8") == '{"a": 1}'

    def test_create_invalid_raises(self, tmp_repo):
        with pytest.raises(profiles.ProfileNameError):
            profiles.create_profile("../evil")

    def test_list_ignores_junk_dirs(self, tmp_repo):
        profiles.create_profile("work")
        (profiles.profiles_root() / "_trash").mkdir()
        (profiles.profiles_root() / ".hidden").mkdir()
        (profiles.profiles_root() / "a_file").write_text("x", encoding="utf-8")
        assert profiles.list_profiles() == ["work"]

    def test_profile_exists(self, tmp_repo):
        assert not profiles.profile_exists("work")
        profiles.create_profile("work")
        assert profiles.profile_exists("work")
        assert not profiles.profile_exists("never-made")
        # Invalid name -> False, not raise.
        assert not profiles.profile_exists("../etc")


# --- typed subdirs -------------------------------------------------------


class TestProfilePaths:
    def test_typed_subdirs(self, tmp_repo):
        p = profiles.create_profile("work")
        assert p.news_dir == p.root / "news"
        assert p.alerts_dir == p.root / "alerts"
        assert p.trackers_dir == p.root / "trackers"
        assert p.reminders_dir == p.root / "reminders"
        assert p.conversations_dir == p.root / "conversations"
        assert p.memory_dir == p.root / "memory"
        assert p.feedback_dir == p.root / "feedback"
        assert p.preferences_path == p.root / "preferences.json"

    def test_root_is_under_profiles_root(self, tmp_repo):
        p = profiles.create_profile("work")
        assert p.root.parent == profiles.profiles_root()
        assert p.root.name == "work"

    def test_get_profile_does_not_create(self, tmp_repo):
        p = profiles.get_profile("future")
        assert not p.root.exists()
        assert not profiles.profile_exists("future")


# --- active profile resolver --------------------------------------------


class TestActiveProfile:
    def test_default_auto_created_when_nothing_set(self, tmp_repo):
        p = profiles.active_profile()
        assert p.name == "default"
        assert p.root.is_dir()

    def test_env_var_wins(self, tmp_repo, monkeypatch):
        profiles.create_profile("work")
        monkeypatch.setenv("KARIN_PROFILE", "work")
        assert profiles.active_profile().name == "work"

    def test_env_var_invalid_falls_through(self, tmp_repo, monkeypatch):
        profiles.create_profile("work")
        monkeypatch.setenv("KARIN_PROFILE", "../etc")
        # Bad env -> fall through to file (empty) -> default
        assert profiles.active_profile().name == "default"

    def test_env_var_nonexistent_falls_through(self, tmp_repo, monkeypatch):
        # Valid name but not on disk -> fall through to default
        monkeypatch.setenv("KARIN_PROFILE", "ghost")
        assert profiles.active_profile().name == "default"

    def test_file_chosen_when_no_env(self, tmp_repo):
        profiles.create_profile("work")
        profiles.set_active("work")
        assert profiles.active_profile().name == "work"

    def test_env_beats_file(self, tmp_repo, monkeypatch):
        profiles.create_profile("work")
        profiles.create_profile("family")
        profiles.set_active("family")
        monkeypatch.setenv("KARIN_PROFILE", "work")
        assert profiles.active_profile().name == "work"

    def test_corrupt_file_falls_through_to_default(self, tmp_repo):
        fpath = tmp_repo / "data" / "active_profile.txt"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("../evil\n", encoding="utf-8")
        assert profiles.active_profile().name == "default"

    def test_file_names_deleted_profile_falls_through(self, tmp_repo):
        # File says "work" but the dir is gone — should not crash,
        # should fall through to default.
        profiles.create_profile("work")
        profiles.set_active("work")
        # Remove the profile dir behind set_active's back.
        import shutil
        shutil.rmtree(profiles.profiles_root() / "work")
        assert profiles.active_profile().name == "default"


class TestProfilePreferences:
    def test_empty_profile_returns_empty_dict(self, tmp_repo):
        p = profiles.create_profile("work")
        # Seeded preferences.json is '{}' — should be read as empty dict.
        assert profiles.load_profile_preferences(p) == {}

    def test_save_then_load_roundtrip(self, tmp_repo):
        p = profiles.create_profile("work")
        data = {"user_location": {"city": "PHL", "country": "US"}}
        profiles.save_profile_preferences(data, p)
        assert profiles.load_profile_preferences(p) == data

    def test_malformed_json_returns_empty(self, tmp_repo):
        p = profiles.create_profile("work")
        p.preferences_path.write_text("not json {", encoding="utf-8")
        assert profiles.load_profile_preferences(p) == {}

    def test_non_dict_top_level_returns_empty(self, tmp_repo):
        p = profiles.create_profile("work")
        p.preferences_path.write_text('["a", "b"]', encoding="utf-8")
        assert profiles.load_profile_preferences(p) == {}

    def test_load_uses_active_profile_by_default(self, tmp_repo):
        profiles.create_profile("work")
        profiles.set_active("work")
        profiles.save_profile_preferences({"x": 1})
        # No arg -> active profile is "work" -> reads work's prefs.
        assert profiles.load_profile_preferences() == {"x": 1}

    def test_save_rejects_non_dict(self, tmp_repo):
        p = profiles.create_profile("work")
        with pytest.raises(TypeError):
            profiles.save_profile_preferences(["a"], p)  # type: ignore[arg-type]

    def test_two_profiles_are_isolated(self, tmp_repo):
        work = profiles.create_profile("work")
        family = profiles.create_profile("family")
        profiles.save_profile_preferences({"who": "work"}, work)
        profiles.save_profile_preferences({"who": "family"}, family)
        assert profiles.load_profile_preferences(work) == {"who": "work"}
        assert profiles.load_profile_preferences(family) == {"who": "family"}


class TestSetActive:
    def test_persists_across_calls(self, tmp_repo):
        profiles.create_profile("work")
        profiles.set_active("work")
        assert profiles.active_profile().name == "work"

    def test_rejects_missing_profile(self, tmp_repo):
        with pytest.raises(profiles.ProfileNameError):
            profiles.set_active("never-made")

    def test_rejects_invalid_name(self, tmp_repo):
        with pytest.raises(profiles.ProfileNameError):
            profiles.set_active("../etc")

    def test_case_folding_on_set(self, tmp_repo):
        profiles.create_profile("work")
        profiles.set_active("WORK")
        assert profiles.active_profile().name == "work"
