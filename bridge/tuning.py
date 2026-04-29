"""Centralised tuning knobs for power-user adjustments.

Lives alongside :mod:`bridge.features` (binary flags) and
:mod:`bridge.model_config` (per-model overrides) — this module owns the
*continuous* knobs that aren't worth a feature flag but that someone
might reasonably tune when swapping APIs, profiling, or chasing edge
cases:

* news clustering thresholds (Jaccard cut-offs, time bucket window)
* cross-verify LLM call budget per ingest cycle
* digest item caps + tracker-move threshold
* per-fetcher network timeouts
* service-level TTLs

Design rules:

* Loads ``config/tuning.yaml`` once on first call via ``lru_cache``.
  Edit the YAML + restart to apply (call :func:`reload` from a REPL
  to bust the cache mid-process).
* Missing file → empty registry → every caller falls back to the
  ``default`` it passed in. Behavior is therefore identical to the
  pre-tuning era when the file is absent.
* Malformed YAML logs a warning but does NOT raise; we don't want a
  typo to wedge the whole boot. Values that fail type coercion fall
  back to the default per call.
* Dotted-path lookup: ``tuning.get("news.cluster.title_jaccard", 0.55)``.
* No env-var per knob — that pattern fits binary subsystems
  (features.py) but blows up when you have 30+ continuous values.
  YAML edit + restart is the expected workflow.

Adding a new tuning value is a three-step contract:

1. Pick a stable dotted key (``<subsystem>.<group>.<name>``).
2. Replace the current literal in code with
   ``tuning.get("subsystem.group.name", <current_literal>)``.
3. Document the key + default in ``tuning.example.yaml``.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, TypeVar

from bridge.utils import REPO_ROOT, load_config

log = logging.getLogger(__name__)

TUNING_YAML = REPO_ROOT / "config" / "tuning.yaml"

T = TypeVar("T")


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    """Load + cache tuning.yaml. Missing file → empty dict (defaults
    win in callers). Parse error → logged warning + empty dict."""
    if not TUNING_YAML.exists():
        log.debug("no tuning.yaml at %s — using code defaults", TUNING_YAML)
        return {}
    try:
        data = load_config(TUNING_YAML) or {}
    except Exception as e:
        log.warning(
            "failed to parse %s: %s — falling back to code defaults",
            TUNING_YAML, e,
        )
        return {}
    if not isinstance(data, dict):
        log.warning(
            "%s root must be a YAML mapping, got %s — using code defaults",
            TUNING_YAML, type(data).__name__,
        )
        return {}
    return data


def reload() -> None:
    """Clear the cache. Tests + interactive REPL use this; production
    cycles via container restart."""
    _load.cache_clear()


def get(key_path: str, default: T) -> T:
    """Return ``tuning.yaml[key_path]`` or ``default``.

    ``key_path`` is dotted (``"news.cluster.title_jaccard"``). The
    return value is coerced to the ``type(default)`` so callers can
    rely on ``isinstance(get(..., 0.5), float)``. Coercion failures
    fall back to ``default`` with a warning — a typo'd value never
    crashes the caller.
    """
    cfg = _load()
    cur: Any = cfg
    for key in key_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    # Coerce to the default's type so str-vs-int mistakes in the
    # YAML don't propagate as wrong types into the caller.
    target_type = type(default)
    if target_type is bool:
        # bool is a subclass of int, so isinstance(True, int) is True.
        # Only "true"/"false" strings or actual bools should pass.
        if isinstance(cur, bool):
            return cur  # type: ignore[return-value]
        if isinstance(cur, str):
            v = cur.strip().lower()
            if v in ("true", "1", "yes"):
                return True  # type: ignore[return-value]
            if v in ("false", "0", "no"):
                return False  # type: ignore[return-value]
        log.warning(
            "tuning %r: expected bool, got %r — using default %r",
            key_path, cur, default,
        )
        return default
    try:
        return target_type(cur)  # type: ignore[call-arg]
    except (TypeError, ValueError) as e:
        log.warning(
            "tuning %r: cannot coerce %r to %s (%s) — using default %r",
            key_path, cur, target_type.__name__, e, default,
        )
        return default


def snapshot() -> dict[str, Any]:
    """Return the raw loaded config — used by debug endpoints."""
    return dict(_load())
