"""Disk-backed secret store for notification channel URLs.

Two sources, in this precedence order:

1. ``config/notify_secrets.json`` — written by the Settings panel.
   Path is gitignored; created on first save by the prefs API.
2. Environment variables (legacy / `.env`-driven setups).

If both are set, the file wins. Channel factories call
:func:`get_secret(name)` to resolve their URL — keeps each channel
unaware of where the value came from.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from bridge.utils import REPO_ROOT

log = logging.getLogger("bridge.notify.secrets")

# Lives under data/ (writable runtime state) rather than config/
# (read-only template directory in the Jetson docker mount). Same
# secret-out-of-source rule — file is mode 0600 + gitignored.
SECRETS_PATH = REPO_ROOT / "data" / "notify" / "secrets.json"

# Maps the JSON key to the legacy env var. New channels register both.
_KEY_TO_ENV: dict[str, str] = {
    "discord_webhook": "KARIN_NOTIFY_DISCORD_WEBHOOK",
    "ntfy_topic":      "KARIN_NOTIFY_NTFY_TOPIC",
}


@lru_cache(maxsize=1)
def _load_file() -> dict:
    """Load the JSON file once per process. Bust with :func:`reload`
    after a save (the prefs API does this automatically)."""
    if not SECRETS_PATH.exists():
        return {}
    try:
        data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.warning("notify_secrets.json invalid: %s — env vars only", e)
        return {}
    if not isinstance(data, dict):
        log.warning("notify_secrets.json must be an object — env vars only")
        return {}
    return data


def reload() -> None:
    """Drop the cached file. Tests + the prefs save endpoint use this
    so the next :func:`get_secret` call sees fresh values."""
    _load_file.cache_clear()


def get_secret(key: str) -> str:
    """Resolve ``key`` from the secrets file, falling back to its
    legacy env var. Returns ``""`` (not None) when neither is set —
    callers can do ``if value:`` without a None-check."""
    file_value = (_load_file().get(key) or "").strip()
    if file_value:
        return file_value
    env_var = _KEY_TO_ENV.get(key)
    if env_var:
        return (os.environ.get(env_var) or "").strip()
    return ""


def write_secrets(data: dict) -> None:
    """Persist the full secrets object (overwrite). The prefs API
    calls this after validating the body. Sets file mode 0600 so
    a stray cat / scp doesn't leak the URLs.

    On non-POSIX filesystems (Windows, CIFS) chmod is a no-op — the
    file still inherits parent-directory ACLs, which may be world-
    readable on a shared machine. We warn once so the operator knows
    the webhook URLs aren't OS-protected there.
    """
    from bridge.utils import atomic_write_text
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        SECRETS_PATH,
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
    )
    try:
        os.chmod(SECRETS_PATH, 0o600)
    except OSError as e:
        log.warning(
            "chmod 0600 on %s failed (%s); file permissions fall back to "
            "parent-directory ACLs. Webhook tokens are NOT OS-protected "
            "on this filesystem — verify the containing directory is "
            "not world-readable.",
            SECRETS_PATH, e,
        )
    reload()


def read_secrets() -> dict:
    """Return the current persisted secrets dict (without env-var
    fallbacks). Used by the prefs API's GET endpoint so the editor
    shows what's actually on disk, not whatever env vars happen to
    be set."""
    return dict(_load_file())
