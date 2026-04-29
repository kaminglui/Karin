"""One-shot migration from pre-H paths into data/profiles/default/.

Runs at bridge startup. Idempotent — the second run is a no-op. If the
default profile's subdirectories already exist we assume the migration
has happened and skip.

What moves (per-profile preferences):
    data/news/preferences.json                -> news/preferences.json
    data/trackers/tracker_preferences.json    -> trackers/tracker_preferences.json
    data/reminders/*                          -> reminders/*
    data/conversations/*                      -> conversations/*
    data/memory/*                             -> memory/*
    data/feedback/*                           -> feedback/*
    data/calendar/*                           -> calendar/*
    bridge/alerts/data/alerts.jsonl           -> alerts/alerts.jsonl
    bridge/alerts/data/cooldowns.json         -> alerts/cooldowns.json
    bridge/alerts/data/advisory_state.json    -> alerts/advisory_state.json
    bridge/alerts/data/threat_decisions.json  -> alerts/threat_decisions.json
    bridge/news/data/learned_keywords.json    -> news/learned_keywords.json
    config/assistant.yaml:user_location       -> preferences.json (key: user_location)

What stays global:
    bridge/news/data/{articles,clusters,events,extracted,translations}.json
    bridge/trackers/data/{trackers,events}.jsonl
    bridge/routing/data/*
    data/jobs/, data/holidays/, data/notify/, data/digest/

The yaml user_location block is LEFT in place after migration as a
readable fallback, so a user SSH'd into the box who hasn't learned about
profiles yet can still edit config/assistant.yaml and have it take
effect (profile wins, yaml fills the gaps).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from bridge.profiles import (
    DEFAULT_PROFILE_NAME,
    Profile,
    create_profile,
    load_profile_preferences,
    save_profile_preferences,
)
from bridge.utils import REPO_ROOT

log = logging.getLogger("bridge.profiles.migration")

# Sentinel file inside the default profile that records migration has
# run. Using a file (not a flag in preferences.json) means the check is
# filesystem-cheap and won't interfere with user-editable preferences.
_MIGRATION_MARKER = ".migrated_from_legacy"


def run_legacy_migration() -> bool:
    """Move pre-H user data into data/profiles/default/ exactly once.

    Returns True if a migration actually ran, False if skipped. Safe to
    call on every boot — the second call sees the marker and returns
    False without touching anything.
    """
    default = create_profile(DEFAULT_PROFILE_NAME)
    marker = default.root / _MIGRATION_MARKER
    if marker.exists():
        log.debug("profile migration already ran; skipping")
        return False

    log.info("running one-shot legacy-to-profile migration into %s", default.root)
    moved: list[str] = []

    # Preference / state files (single-file moves).
    for src_rel, dst_path in _single_file_moves(default):
        src = REPO_ROOT / src_rel
        if src.is_file() and not dst_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst_path))
            moved.append(f"{src_rel} -> {dst_path.relative_to(REPO_ROOT)}")

    # Whole-directory moves (each file inside moves individually so we
    # don't clobber if destination already has content).
    for src_rel, dst_dir in _directory_moves(default):
        src = REPO_ROOT / src_rel
        if not src.is_dir():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            dst_entry = dst_dir / entry.name
            if dst_entry.exists():
                continue
            shutil.move(str(entry), str(dst_entry))
            moved.append(f"{src_rel}/{entry.name} -> {dst_entry.relative_to(REPO_ROOT)}")

    # user_location: copy (not move) from assistant.yaml into
    # preferences.json. Leaving the yaml block in place means a user
    # who only knows the old config file can still tweak it.
    try:
        migrated_location = _migrate_user_location(default)
        if migrated_location:
            moved.append("config/assistant.yaml:user_location -> preferences.json")
    except Exception as e:
        log.warning("user_location migration failed (non-fatal): %s", e)

    marker.write_text(
        "Legacy paths migrated on bridge startup. This marker prevents\n"
        "the migration from running again. Delete it to force re-run\n"
        "(destination files already in place will not be overwritten).\n",
        encoding="utf-8",
    )
    if moved:
        log.info(
            "profile migration moved %d item(s):\n  %s",
            len(moved), "\n  ".join(moved),
        )
    else:
        log.info("profile migration: nothing to move (fresh install)")
    return True


def _single_file_moves(profile: Profile) -> list[tuple[str, Path]]:
    """Return (src_relative_to_repo, absolute_dst) pairs for single-file
    moves. Ordered deterministically so logs are reproducible."""
    return [
        ("data/news/preferences.json",
         profile.news_dir / "preferences.json"),
        ("data/trackers/tracker_preferences.json",
         profile.trackers_dir / "tracker_preferences.json"),
        ("bridge/alerts/data/alerts.jsonl",
         profile.alerts_dir / "alerts.jsonl"),
        ("bridge/alerts/data/cooldowns.json",
         profile.alerts_dir / "cooldowns.json"),
        ("bridge/alerts/data/advisory_state.json",
         profile.alerts_dir / "advisory_state.json"),
        ("bridge/alerts/data/threat_decisions.json",
         profile.alerts_dir / "threat_decisions.json"),
        ("bridge/news/data/learned_keywords.json",
         profile.news_dir / "learned_keywords.json"),
    ]


def _directory_moves(profile: Profile) -> list[tuple[str, Path]]:
    """Return (src_relative_to_repo, dst_dir) pairs for whole-directory
    moves. We iterate children individually so a destination that
    already has content doesn't get clobbered."""
    return [
        ("data/reminders", profile.reminders_dir),
        ("data/conversations", profile.conversations_dir),
        ("data/memory", profile.memory_dir),
        ("data/feedback", profile.feedback_dir),
        ("data/calendar", profile.root / "calendar"),
    ]


def _migrate_user_location(profile: Profile) -> bool:
    """Copy user_location from assistant.yaml into the profile's
    preferences.json under the ``user_location`` key. Only runs when
    the profile doesn't already have a user_location set — we never
    clobber a user's explicit per-profile choice.
    """
    current = load_profile_preferences(profile)
    if isinstance(current.get("user_location"), dict) and current["user_location"]:
        return False  # profile already has its own; leave it alone
    try:
        from bridge.utils import load_config
        cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
        block = cfg.get("user_location") or {}
    except Exception as e:
        log.debug("assistant.yaml unreadable during user_location copy: %s", e)
        return False
    if not isinstance(block, dict) or not block:
        return False
    # Filter to the keys UserContext understands, drop empties.
    KEEP = ("city", "region", "country", "timezone", "latitude", "longitude")
    cleaned: dict = {}
    for k in KEEP:
        v = block.get(k)
        if isinstance(v, str) and v.strip():
            cleaned[k] = v.strip()
        elif isinstance(v, (int, float)):
            cleaned[k] = float(v) if k in ("latitude", "longitude") else v
    if not cleaned:
        return False
    current["user_location"] = cleaned
    save_profile_preferences(current, profile)
    return True
