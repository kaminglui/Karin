"""User context for the threat-assessment layer.

Currently holds location only. The shape is future-extensible: health
conditions, mobility, travel plans, dependents, etc. all slot in
later as the assessor grows — rules simply check whatever fields
they care about via attribute access, with sane defaults for
missing fields so existing assessments keep working when new fields
land.

Loaded once per AlertService from ``config/assistant.yaml`` under the
``user_location`` key (same config surface already used by bridge.
location for the LLM system prompt). No new config file, no new env
vars to manage — when the user updates their yaml, a docker compose
restart picks up the new values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("bridge.alerts.user_context")


@dataclass(frozen=True)
class UserContext:
    """User state the threat assessor reads from.

    All fields default to empty so rules can safely call
    ``ctx.city.lower()`` without branching on presence. A fully-empty
    context produces threat_score=0 for location-dependent rules
    (correct — no location, no proximity signal).
    """
    city: str = ""
    region: str = ""      # state / province
    country: str = ""
    latitude: float | None = None
    longitude: float | None = None


def load_user_context() -> UserContext:
    """Construct a UserContext from the active profile (or legacy yaml).

    Resolution order (first non-empty wins field-by-field):
        1. Active profile's preferences.json → user_location block
        2. Legacy config/assistant.yaml → user_location block
        3. Empty context (score 0 on every location-dependent rule)

    Per-field fallback (not whole-block) means a profile that only sets
    city still inherits region/country from the yaml if present — useful
    during the migration window where a user might only have edited a
    partial profile.

    Any exception produces an empty context with a debug log; the
    threat assessor never raises on bad config.
    """
    profile_block = _load_from_profile()
    yaml_block = _load_from_yaml()

    def _str(v):
        return str(v).strip() if isinstance(v, str) else ""

    def _float(v):
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _pick(key, coerce):
        # Profile wins; fall through to yaml only when profile is empty.
        p = coerce(profile_block.get(key))
        if p not in ("", None):
            return p
        return coerce(yaml_block.get(key))

    return UserContext(
        city=_pick("city", _str),
        region=_pick("region", _str),
        country=_pick("country", _str),
        latitude=_pick("latitude", _float),
        longitude=_pick("longitude", _float),
    )


def _load_from_profile() -> dict:
    try:
        from bridge.profiles import load_profile_preferences
        prefs = load_profile_preferences()
        block = prefs.get("user_location") or {}
    except Exception as e:
        log.debug("profile user_location load failed (%s)", e)
        return {}
    return block if isinstance(block, dict) else {}


def _load_from_yaml() -> dict:
    try:
        from bridge.utils import REPO_ROOT, load_config
        cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
        block = cfg.get("user_location") or {}
    except Exception as e:
        log.debug("yaml user_location load failed (%s)", e)
        return {}
    return block if isinstance(block, dict) else {}
