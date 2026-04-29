"""Feature registry — single source of truth for subsystem/tool toggles.

Reads ``config/features.yaml`` at module import, caches the result, and
exposes a couple of small helpers the rest of the codebase calls
instead of scattering environment-variable checks everywhere. Env vars
still work (they override the YAML value) so deploy rollouts don't
require editing the file.

Design notes:
  - Refresh on startup only. No hot reload; a restart is the expected
    way to toggle. Keeps the API surface tiny.
  - No dependency injection / plugin loading — we're deliberately
    building a registry, not a framework. See CLAUDE.md guidance.
  - Everything is in-process. No MCP, no subprocess isolation.
  - Failure mode is loud-fallback: if `features.yaml` can't be parsed
    the server boots with EVERYTHING enabled and logs a warning.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from bridge.utils import REPO_ROOT, load_config

log = logging.getLogger(__name__)

FEATURES_YAML = REPO_ROOT / "config" / "features.yaml"

# Env-var values that count as DISABLED. Anything else is treated as
# enabled, matching the `_env_flag` logic we used in web/server.py
# before centralizing it here.
_DISABLE_TOKENS = ("0", "false", "off", "no", "")


def _env_disabled(env_var: str | None) -> bool | None:
    """Return True/False if env_var is set, None if unset.

    When the value matches a disable token (case-insensitive), we
    return True (disabled); any other non-empty string reads as
    enabled. Unset env returns None — caller falls back to YAML.
    """
    if not env_var:
        return None
    raw = os.environ.get(env_var)
    if raw is None:
        return None
    return raw.strip().lower() in _DISABLE_TOKENS


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    """Load + cache the YAML. lru_cache makes the first call do the
    work and subsequent calls O(1). Call :func:`reload` to bust."""
    if not FEATURES_YAML.exists():
        log.warning(
            "%s not found — every feature defaults to ENABLED", FEATURES_YAML,
        )
        return {"subsystems": {}, "tools": {"disabled": []}}
    try:
        data = load_config(FEATURES_YAML) or {}
    except Exception as e:
        log.warning(
            "failed to parse %s: %s — every feature defaults to ENABLED",
            FEATURES_YAML, e,
        )
        return {"subsystems": {}, "tools": {"disabled": []}}
    # Normalize shape so callers can depend on the keys existing.
    data.setdefault("subsystems", {})
    data.setdefault("tools", {})
    data["tools"].setdefault("disabled", [])
    return data


def reload() -> None:
    """Force re-read of features.yaml on next call. Tests use this."""
    _load.cache_clear()


def is_enabled(name: str, default: bool = True) -> bool:
    """Is the named subsystem enabled?

    Precedence (highest wins):
      1. Environment variable (if ``env`` mapped in yaml is set)
      2. ``subsystems.<name>.enabled`` in features.yaml
      3. ``default`` (True unless the caller wants stricter)

    Unknown subsystem name returns ``default`` — lets new feature
    code go in without requiring a yaml edit first.
    """
    cfg = _load()
    entry = (cfg.get("subsystems") or {}).get(name) or {}
    env_override = _env_disabled(entry.get("env"))
    if env_override is not None:
        return not env_override
    if "enabled" in entry:
        return bool(entry["enabled"])
    return default


def tool_enabled(tool_name: str) -> bool:
    """Is this tool exposed to the LLM?

    Simple denylist model: anything on ``tools.disabled`` is off;
    anything else is on. Keeps the common case (all tools enabled)
    zero-config while still letting deploys kill-switch specific
    tools without a code change.
    """
    cfg = _load()
    disabled = set((cfg.get("tools") or {}).get("disabled") or [])
    return tool_name not in disabled


def snapshot() -> dict[str, Any]:
    """Return a small dict describing current feature state — used
    by the /api/features endpoint and debug logging."""
    cfg = _load()
    subs = {
        name: is_enabled(name)
        for name in (cfg.get("subsystems") or {}).keys()
    }
    disabled_tools = list((cfg.get("tools") or {}).get("disabled") or [])
    return {
        "subsystems": subs,
        "tools": {"disabled": disabled_tools},
    }
