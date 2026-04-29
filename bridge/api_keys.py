"""Load **system-wide** third-party API keys from ``config/api_keys.json``.

The JSON file is gitignored; a template lives at
``config/api_keys.example.json``. Tools that need a key should call
:func:`get_api_key` and degrade gracefully when it returns ``None``.

Used by tools that are scoped to the whole deploy (Brave search,
Spoonacular, OpenWeatherMap). Env vars are of the form
``KARIN_API_KEY_<NAME_UPPER>``.

**Distinct from ``bridge.profiles.get_profile_api_key()``** — that
helper reads **per-profile** keys from
``data/profiles/<active>/preferences.json`` (used by the
county-overlay populators: CDC / Census / FBI). See its module
docstring for the full comparison. The two systems have
non-overlapping name registries and coexist.

Design:
- Loads once on first use and caches.
- Missing file, empty string, or an example/template slot all look
  the same to the caller: ``None`` is returned.
- Warnings for a missing file are logged at INFO level only (this is
  an OPTIONAL feature; users who don't want any paid APIs should see
  no noise).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from bridge.utils import REPO_ROOT

log = logging.getLogger("bridge.api_keys")

_KEYS_PATH: Path = REPO_ROOT / "config" / "api_keys.json"
_EXAMPLE_PATH: Path = REPO_ROOT / "config" / "api_keys.example.json"
_cache: dict | None = None
_cache_lock = threading.Lock()


def _load() -> dict:
    """Read and return the api_keys.json contents (cached).

    Returns an empty dict if the file doesn't exist — tools should
    then treat every key lookup as ``None``.
    """
    global _cache
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
    if not _KEYS_PATH.exists():
        log.info(
            "no %s found; third-party APIs disabled. To enable, copy "
            "%s to %s and fill in the keys you want.",
            _KEYS_PATH.name, _EXAMPLE_PATH.name, _KEYS_PATH.name,
        )
        _cache = {}
        return _cache
    try:
        _cache = json.loads(_KEYS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("failed to parse %s: %s — all third-party APIs disabled", _KEYS_PATH, e)
        _cache = {}
    return _cache


def get_api_key(name: str) -> str | None:
    """Return the configured API key for ``name``, or ``None`` if absent.

    ``name`` is the top-level key in api_keys.json (e.g. ``"brave_search"``,
    ``"spoonacular"``). Empty strings count as "not configured" and also
    return ``None``.

    Env-var override: ``KARIN_API_KEY_<NAME_UPPER>`` wins over the JSON
    file, useful for Docker deployments that inject secrets via env
    rather than a mounted file.
    """
    # Env var wins so Docker secret injection works without editing files.
    env_name = f"KARIN_API_KEY_{name.upper()}"
    env_val = os.environ.get(env_name)
    if env_val:
        return env_val

    data = _load()
    slot = data.get(name)
    if not isinstance(slot, dict):
        return None
    key = slot.get("api_key")
    if not isinstance(key, str) or not key.strip():
        return None
    return key.strip()


def has_api_key(name: str) -> bool:
    """True if a usable key exists for ``name``."""
    return get_api_key(name) is not None


def reload() -> None:
    """Flush the cache so the next lookup re-reads the file.

    Useful after an operator edits api_keys.json while the bridge is
    running; no container restart needed.
    """
    global _cache
    _cache = None
