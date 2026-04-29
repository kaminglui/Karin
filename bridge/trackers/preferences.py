"""User preferences + ordering for bridge/trackers.

Mirrors the shape of :mod:`bridge.news.preferences` but simpler because
trackers don't have free-text haystacks to match against — each tracker
has a fixed `category` in its config, and ordering is a straightforward
priority list.

Schema (``config/tracker_preferences.json``):

.. code-block:: json

    {
      "enabled": true,
      "category_priority": ["metal", "energy", "fx", "food_index", "crypto"],
      "tracker_order": {
        "energy": ["gas_retail"]
      },
      "home_state": "PA"
    }

* ``category_priority`` — ordered list of categories. Sections render
  in this order in the UI (and LLM voice-line listings follow the same
  order). Unlisted categories fall to the end, in config-file order.

* ``tracker_order`` — per-category override list. Trackers named here
  float to the top of their section in the given order. Unlisted
  trackers within the category fall back to config-file order.

* ``home_state`` — two-letter US state code. Fetcher params containing
  the literal string ``"${home_state}"`` get substituted before the
  HTTP call. Used for state-level electricity / natural-gas trackers
  once those land (see docs/ideas.md). Irrelevant for everything
  currently shipped, so no default — caller should fall back to ``""``
  (no substitution) when this is missing.

Design rules (mirroring news.preferences):

* Missing file → ``TrackerPreferences(enabled=False)``. Behavior
  identical to Phase 5 (config-file order across the board).
* ``enabled=False`` collapses all sort logic to config order — any
  preference noise short-circuits to zero influence.
* Preferences reorder; they never filter. A disabled tracker stays
  disabled; an enabled one always appears somewhere.
* Pure data + a thin loader. No side effects, no LLM, no network.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("bridge.trackers.preferences")


@dataclass(frozen=True)
class TrackerPreferences:
    """Loaded tracker ordering preferences.

    ``enabled=False`` makes every ordering op a no-op so callers can
    unconditionally pass the preferences object through without
    branching on whether the file existed.

    `tracker_overrides` + `disabled_categories` are the user-visible
    enable toggles from the settings UI. They override the per-tracker
    ``enabled`` flag in trackers.json in either direction — letting
    users turn a config-disabled tracker on (e.g. crypto which ships
    off) or a config-enabled one off without editing that file.
    """

    enabled: bool = False
    category_priority: tuple[str, ...] = ()
    tracker_order: dict[str, tuple[str, ...]] = field(default_factory=dict)
    home_state: str = ""
    # id -> forced state. Absent means "honor cfg.enabled".
    tracker_overrides: dict[str, bool] = field(default_factory=dict)
    disabled_categories: tuple[str, ...] = ()


def is_tracker_visible(
    tracker_id: str,
    category: str,
    cfg_enabled: bool,
    preferences: "TrackerPreferences",
) -> bool:
    """Effective visibility after overlaying preferences on config.

    Order of precedence (most specific wins):

    1. ``tracker_overrides[id]`` — explicit per-tracker toggle. Wins
       over everything else.
    2. ``disabled_categories`` — blanket off for the whole category.
    3. ``cfg.enabled`` — the trackers.json default.

    With preferences disabled (``enabled=False``), override logic is
    skipped entirely and only ``cfg.enabled`` matters. That keeps the
    "turn off all prefs → legacy behavior" contract intact.
    """
    if not preferences.enabled:
        return cfg_enabled
    if tracker_id in preferences.tracker_overrides:
        return bool(preferences.tracker_overrides[tracker_id])
    if category in preferences.disabled_categories:
        return False
    return cfg_enabled


def load_tracker_preferences(path: Path) -> TrackerPreferences:
    """Load ordering preferences from ``path``.

    Missing file → disabled preferences (config-file order preserved).
    Malformed JSON raises — a typo shouldn't silently disable the
    user's ordering; the caller should let it surface during startup
    so they notice and fix it.
    """
    if not path.exists():
        log.info(
            "no tracker_preferences.json at %s; running with preferences disabled",
            path,
        )
        return TrackerPreferences(enabled=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"tracker_preferences.json must be a JSON object, got {type(data).__name__}"
        )

    enabled = bool(data.get("enabled", True))
    cat_priority = data.get("category_priority") or []
    if not isinstance(cat_priority, list):
        log.warning("category_priority must be a list, got %r; ignoring", type(cat_priority))
        cat_priority = []
    cat_priority_tup = tuple(str(c).strip() for c in cat_priority if str(c).strip())

    raw_order = data.get("tracker_order") or {}
    if not isinstance(raw_order, dict):
        log.warning("tracker_order must be an object, got %r; ignoring", type(raw_order))
        raw_order = {}
    tracker_order: dict[str, tuple[str, ...]] = {}
    for cat, ids in raw_order.items():
        if not isinstance(ids, list):
            log.warning("tracker_order[%r] must be a list; skipping", cat)
            continue
        tracker_order[str(cat)] = tuple(str(i).strip() for i in ids if str(i).strip())

    home_state = str(data.get("home_state", "")).strip().upper()

    raw_overrides = data.get("tracker_overrides") or {}
    tracker_overrides: dict[str, bool] = {}
    if isinstance(raw_overrides, dict):
        for tid, state in raw_overrides.items():
            if isinstance(state, bool):
                tracker_overrides[str(tid)] = state
    else:
        log.warning(
            "tracker_overrides must be an object, got %r; ignoring",
            type(raw_overrides),
        )

    raw_disabled = data.get("disabled_categories") or []
    if not isinstance(raw_disabled, list):
        log.warning(
            "disabled_categories must be a list, got %r; ignoring",
            type(raw_disabled),
        )
        raw_disabled = []
    disabled_cats = tuple(
        str(c).strip() for c in raw_disabled if str(c).strip()
    )

    log.info(
        "loaded tracker_preferences: enabled=%s categories=%d orders=%d "
        "home_state=%s overrides=%d disabled_cats=%d",
        enabled, len(cat_priority_tup), len(tracker_order),
        home_state or "(unset)", len(tracker_overrides), len(disabled_cats),
    )
    return TrackerPreferences(
        enabled=enabled,
        category_priority=cat_priority_tup,
        tracker_order=tracker_order,
        home_state=home_state,
        tracker_overrides=tracker_overrides,
        disabled_categories=disabled_cats,
    )


# --- ordering helpers ------------------------------------------------------


def category_sort_key(
    category: str, preferences: TrackerPreferences,
) -> tuple[int, int]:
    """Return a sort key for a category string.

    Listed categories sort by their position in ``category_priority``.
    Unlisted ones sort at the tail, with a secondary key of 0 so their
    relative order is stable (call sites should pass them in
    config-file order so the tiebreak lands where the user expects).

    Returns ``(0, 0)`` when preferences are disabled — flattens the
    sort so the caller's original order wins.
    """
    if not preferences.enabled:
        return (0, 0)
    try:
        return (0, preferences.category_priority.index(category))
    except ValueError:
        return (1, 0)


def tracker_sort_key(
    tracker_id: str, category: str, preferences: TrackerPreferences,
) -> tuple[int, int]:
    """Return a sort key for a tracker within its category.

    Trackers named in ``tracker_order[category]`` sort by their
    position there. Unlisted trackers sort at the tail. Preferences
    disabled → flat key so the caller's order wins.
    """
    if not preferences.enabled:
        return (0, 0)
    order = preferences.tracker_order.get(category, ())
    try:
        return (0, order.index(tracker_id))
    except ValueError:
        return (1, 0)


# --- placeholder substitution ---------------------------------------------

def _resolve_placeholders(
    value: str, preferences: "TrackerPreferences",
) -> str:
    """Substitute ``${home_state}`` / ``${home_state_padd}`` in a
    string. Unknown placeholders are left as-is so a stray typo is
    loud in the logs rather than silently swallowed."""
    if "${" not in value:
        return value
    from bridge.trackers.fetch import state_to_padd

    substitutions = {
        "home_state": preferences.home_state or "",
        "home_state_padd": state_to_padd(preferences.home_state),
    }
    out = value
    for key, replacement in substitutions.items():
        out = out.replace("${" + key + "}", replacement)
    return out


def resolve_params(
    params: dict, preferences: "TrackerPreferences",
) -> dict:
    """Return a copy of ``params`` with placeholders resolved.

    Only string values are substituted; nested dicts / lists pass
    through unchanged. No-ops when every value is a plain literal,
    so non-templated tracker configs don't pay any cost.

    Placeholders:

    * ``${home_state}`` — two-letter US code from
      ``preferences.home_state`` (empty when unset).
    * ``${home_state_padd}`` — EIA PADD duoarea for the home state
      (e.g. "R1Y" for PA). Falls back to a default PADD when no
      home_state is configured. See ``bridge.trackers.fetch.STATE_TO_PADD``.
    """
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, str):
            out[k] = _resolve_placeholders(v, preferences)
        else:
            out[k] = v
    return out


def resolve_label(label: str, preferences: "TrackerPreferences") -> str:
    """Substitute placeholders in a tracker label. Same rules as
    :func:`resolve_params`. Non-templated labels pass through
    unchanged."""
    return _resolve_placeholders(label, preferences)
