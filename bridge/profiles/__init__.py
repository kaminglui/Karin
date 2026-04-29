"""Phase H: Profile isolation.

One active profile per process. Every subsystem that used to write to
``bridge/<sub>/data/`` or ``data/<sub>/`` gets rerouted to
``data/profiles/<active>/<sub>/`` instead, so two profiles on the same
box have completely independent preferences, reminders, cooldowns,
etc.

What stays global (shared across profiles):
    - RSS article cache + cluster graph + extracted bodies
    - Translations cache
    - Tracker market snapshots (time-series data)
    - Source configs + notify secrets
    - Anything in config/ — that's system configuration, not user data

What moves per-profile:
    - News preferences (watchlists, target_language, user_location)
    - Phase E learned keywords
    - Alert cooldowns, advisory_state, fired alerts, threat decisions
    - Tracker preferences (which trackers the user cares about)
    - Reminders, conversations, feedback, per-user memory
    - Future: per-profile notify channels if the gray-zone case comes up

Active profile resolution order:
    1. ``KARIN_PROFILE`` env var — for docker / CI overrides
    2. ``data/active_profile.txt`` — the persistent choice, survives reboot
    3. ``"default"`` — always exists, can't be deleted

Public API:
    Profile                 — dataclass with typed subdir properties
    active_profile()        — current profile (reads env + file + fallback)
    set_active(name)        — persist a switch; caller restarts the process
    list_profiles()         — names on disk, alphabetical
    create_profile(name)    — mkdir + seed empty preferences.json
    profiles_root()         — data/profiles/ — for migration code
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bridge.utils import REPO_ROOT, atomic_write_text

log = logging.getLogger("bridge.profiles")

DEFAULT_PROFILE_NAME = "default"
_ACTIVE_PROFILE_FILE = "active_profile.txt"
_ENV_VAR = "KARIN_PROFILE"

# Filesystem-safe profile names. Rejecting dots + slashes + leading -
# prevents path traversal (e.g. "../etc") and hidden dirs.
_VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_UNSAFE_CHARS_RE = re.compile(r"[\\/.~]")


class ProfileNameError(ValueError):
    """Raised when a caller hands us a profile name that would not be
    safe as a directory (case, length, traversal characters)."""


@dataclass(frozen=True)
class Profile:
    """An isolated per-user data root.

    ``root`` is the profile's own directory on disk. The typed
    ``*_dir`` properties give each subsystem a stable place to write
    without concatenating strings at the call site. Creating the
    physical directories is the subsystem's responsibility (the store
    constructors already mkdir their data_dir) — this class just
    computes paths.
    """
    name: str
    root: Path

    @property
    def news_dir(self) -> Path:
        return self.root / "news"

    @property
    def alerts_dir(self) -> Path:
        return self.root / "alerts"

    @property
    def trackers_dir(self) -> Path:
        return self.root / "trackers"

    @property
    def reminders_dir(self) -> Path:
        return self.root / "reminders"

    @property
    def conversations_dir(self) -> Path:
        return self.root / "conversations"

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def feedback_dir(self) -> Path:
        return self.root / "feedback"

    @property
    def preferences_path(self) -> Path:
        """Per-profile preferences bundle. Houses the settings that
        don't belong to any one subsystem — most importantly
        user_location, which used to live in config/assistant.yaml and
        could only hold one value per box."""
        return self.root / "preferences.json"


# --- validation ---------------------------------------------------------


def validate_name(name: str) -> str:
    """Return a sanitized profile name or raise ProfileNameError.

    Rules:
      - lowercase
      - 1-32 chars
      - [a-z0-9_-] only
      - cannot start with '-' (confuses CLIs)
      - no '..', '/', or '\\' (path traversal)

    We lower() before validating so casing differences collapse —
    "Work" and "work" are the same profile."""
    if not isinstance(name, str):
        raise ProfileNameError(f"profile name must be str, got {type(name).__name__}")
    n = name.strip().lower()
    if _UNSAFE_CHARS_RE.search(n) or not _VALID_NAME_RE.match(n):
        raise ProfileNameError(
            f"profile name {name!r} is not filesystem-safe; "
            f"use 1-32 chars of [a-z0-9_-], starting with alphanumeric"
        )
    return n


# --- paths --------------------------------------------------------------


def profiles_root() -> Path:
    """data/profiles/ — parent of every profile directory."""
    return REPO_ROOT / "data" / "profiles"


def _active_profile_file() -> Path:
    return REPO_ROOT / "data" / _ACTIVE_PROFILE_FILE


def _profile_dir(name: str) -> Path:
    return profiles_root() / name


# --- registry ------------------------------------------------------------


def list_profiles() -> list[str]:
    """Return profile names that exist on disk, alphabetical.

    A directory under data/profiles/ counts as a profile iff its name
    passes validate_name(). This tolerates garbage (_trash/, .tmp/,
    whatever) sitting under data/profiles/ without confusing the UI.
    """
    root = profiles_root()
    if not root.exists():
        return []
    out: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            validate_name(entry.name)
        except ProfileNameError:
            continue
        out.append(entry.name)
    out.sort()
    return out


def create_profile(name: str) -> Profile:
    """Create a new profile directory. Idempotent — if it exists,
    returns the existing Profile rather than raising. Subsystems mkdir
    their own subdirs lazily on first write, so we only create the
    top-level profile root here plus an empty preferences.json so
    later code can rely on its presence."""
    safe = validate_name(name)
    root = _profile_dir(safe)
    root.mkdir(parents=True, exist_ok=True)
    prefs = root / "preferences.json"
    if not prefs.exists():
        atomic_write_text(prefs, "{}\n")
    return Profile(name=safe, root=root)


def get_profile(name: str) -> Profile:
    """Return a Profile object for ``name`` without creating it on
    disk. Useful for read paths that want to handle "doesn't exist"
    explicitly rather than auto-create."""
    safe = validate_name(name)
    return Profile(name=safe, root=_profile_dir(safe))


def profile_exists(name: str) -> bool:
    try:
        safe = validate_name(name)
    except ProfileNameError:
        return False
    return _profile_dir(safe).is_dir()


# --- active profile ------------------------------------------------------


def active_profile() -> Profile:
    """Resolve the currently active profile.

    Priority:
        1. KARIN_PROFILE env var (docker / systemd override)
        2. data/active_profile.txt (persistent user choice)
        3. "default" (always exists, can't be deleted)

    If the resolved name doesn't exist on disk, or is invalid, we
    silently fall through to the next source. The very last fallback
    (default) is auto-created if missing — the bridge should never
    start in a state where there's no active profile.
    """
    # 1. env var
    env_name = os.environ.get(_ENV_VAR, "").strip()
    if env_name:
        try:
            safe = validate_name(env_name)
        except ProfileNameError:
            safe = None
        if safe and profile_exists(safe):
            return get_profile(safe)

    # 2. persisted file
    fpath = _active_profile_file()
    if fpath.exists():
        try:
            stored = fpath.read_text(encoding="utf-8").strip()
            safe = validate_name(stored)
            if profile_exists(safe):
                return get_profile(safe)
        except (ProfileNameError, OSError):
            pass

    # 3. default, auto-created if absent
    return create_profile(DEFAULT_PROFILE_NAME)


# --- profile preferences bundle -----------------------------------------


def load_profile_preferences(profile: Profile | None = None) -> dict[str, Any]:
    """Read the profile's preferences.json, returning {} on any failure.

    This is the single source of truth for settings that don't belong
    to a specific subsystem — most importantly ``user_location``, which
    used to live in config/assistant.yaml and could only hold one value
    per physical device. Malformed JSON or a missing file collapses to
    an empty dict so callers don't need to branch on existence.
    """
    p = (profile or active_profile()).preferences_path
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("profile preferences unreadable at %s (%s); using empty", p, e)
        return {}
    return raw if isinstance(raw, dict) else {}


def save_profile_preferences(
    data: dict[str, Any], profile: Profile | None = None,
) -> Path:
    """Atomically write the profile's preferences.json. Returns the path
    written. Pretty-printed so a human inspecting the file on disk gets
    a reasonable layout.
    """
    if not isinstance(data, dict):
        raise TypeError(f"preferences must be a dict, got {type(data).__name__}")
    p = (profile or active_profile()).preferences_path
    atomic_write_text(
        p,
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
    )
    return p


# --- per-profile API-key resolution ------------------------------------
#
# Resolves data-source API keys (CDC, Census, FBI, etc.) used by the
# county-overlay populators. Names match the Settings UI field IDs.
#
# **Distinct from ``bridge.api_keys.get_api_key()``** — that helper
# reads system-wide keys from ``config/api_keys.json`` (used by Brave
# search / Spoonacular / OpenWeatherMap, with env vars of the form
# ``KARIN_API_KEY_<NAME_UPPER>``). This helper reads per-profile keys
# from ``data/profiles/<active>/preferences.json`` (with conventional
# env vars like ``KARIN_CENSUS_API_KEY``). The two systems coexist —
# they're scoped differently and have non-overlapping name registries.
#
# Resolution order (this helper):
#   1. Environment variable (e.g. ``KARIN_CENSUS_API_KEY``) — wins so
#      ``deploy/.env`` keeps working unchanged.
#   2. Active profile's ``preferences.json`` ``api_keys.<name>`` —
#      written from the Settings panel.
#   3. ``None`` — caller's responsibility to handle (e.g. NewsAPI
#      returns []; the CDC fetcher proceeds without the rate-limit
#      boost).

# Single registry — both the resolver below and the Settings UI
# (web/panels_api.py imports this) read from PROFILE_API_KEY_FIELDS
# so a new key needs exactly one entry.
PROFILE_API_KEY_FIELDS: dict[str, dict[str, str]] = {
    "cdc_app_token": {
        "label": "CDC Socrata app token (optional)",
        "env": "KARIN_CDC_APP_TOKEN",
        "purpose": (
            "Optional. Raises the per-IP rate limit on data.cdc.gov "
            "fetches (used by the county mortality populator). The "
            "fetcher works without it, just slower at scale."
        ),
        "register_url": (
            "https://data.cdc.gov/profile/edit/developer_settings"
        ),
    },
    "census_api_key": {
        "label": "Census ACS API key",
        "env": "KARIN_CENSUS_API_KEY",
        "purpose": (
            "Required to populate the ACS county data (gini, "
            "population, rent, income brackets) used by the "
            "county-overlay panel. Free, registers in ~1 minute."
        ),
        "register_url": "https://api.census.gov/data/key_signup.html",
    },
    "fbi_api_key": {
        "label": "FBI Crime Data Explorer API key",
        "env": "KARIN_FBI_API_KEY",
        "purpose": (
            "Required to populate FBI UCR violent + property crime "
            "rates by county. Any api.data.gov key works (the FBI "
            "CDE uses api.data.gov for auth). Free."
        ),
        "register_url": "https://api.data.gov/signup/",
    },
}


def get_profile_api_key(
    name: str, profile: Profile | None = None,
) -> str | None:
    """Return a per-profile API key from env or preferences.

    Returns ``None`` (not empty string) when nothing is configured, so
    callers can write ``if get_profile_api_key(...)`` without worrying
    about whitespace-only values.

    Distinct from ``bridge.api_keys.get_api_key()`` — see this module's
    docstring for the comparison.
    """
    field = PROFILE_API_KEY_FIELDS.get(name)
    if field:
        env_val = (os.environ.get(field["env"]) or "").strip()
        if env_val:
            return env_val
    prefs = load_profile_preferences(profile)
    block = prefs.get("api_keys") or {}
    if not isinstance(block, dict):
        return None
    val = block.get(name)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def list_profile_api_key_names() -> list[str]:
    """Names accepted by ``get_profile_api_key()`` / Settings UI /
    ``/api/preferences/api-keys`` endpoints."""
    return sorted(PROFILE_API_KEY_FIELDS)


def set_active(name: str) -> Profile:
    """Persist ``name`` as the active profile for future process starts.

    Doesn't swap the in-process subsystem singletons — callers are
    expected to restart the bridge after this (the UI shows a banner).
    Hot-switching would require resetting every ``_default_X_service``
    cache and is out of scope for V1 (see bridge/profiles/__init__.py
    docstring).
    """
    safe = validate_name(name)
    if not profile_exists(safe):
        raise ProfileNameError(f"profile {name!r} does not exist")
    fpath = _active_profile_file()
    atomic_write_text(fpath, safe + "\n")
    return get_profile(safe)
