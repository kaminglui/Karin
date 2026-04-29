"""Tests for bridge.trackers.preferences."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge.trackers.preferences import (
    TrackerPreferences,
    category_sort_key,
    load_tracker_preferences,
    tracker_sort_key,
)


# ---------------------------------------------------------------------------
# load_tracker_preferences
# ---------------------------------------------------------------------------
def test_missing_file_returns_disabled(tmp_path: Path) -> None:
    prefs = load_tracker_preferences(tmp_path / "nope.json")
    assert prefs.enabled is False
    assert prefs.category_priority == ()
    assert prefs.tracker_order == {}
    assert prefs.home_state == ""


def test_loads_full_config(tmp_path: Path) -> None:
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({
        "enabled": True,
        "category_priority": ["metal", "energy"],
        "tracker_order": {"energy": ["gas_retail", "electricity"]},
        "home_state": "pa",
    }))
    prefs = load_tracker_preferences(p)
    assert prefs.enabled is True
    assert prefs.category_priority == ("metal", "energy")
    assert prefs.tracker_order == {"energy": ("gas_retail", "electricity")}
    assert prefs.home_state == "PA"   # upper-cased on load


def test_enabled_false_preserved(tmp_path: Path) -> None:
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({
        "enabled": False,
        "category_priority": ["metal"],
    }))
    prefs = load_tracker_preferences(p)
    assert prefs.enabled is False
    # The non-empty categories are loaded, but their effect is zero
    # because preferences.enabled is False — sort keys flatten.
    assert prefs.category_priority == ("metal",)


def test_malformed_root_raises(tmp_path: Path) -> None:
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ValueError):
        load_tracker_preferences(p)


def test_bad_category_priority_type_ignored(tmp_path: Path) -> None:
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({
        "enabled": True,
        "category_priority": "metal",   # wrong type — should be a list
    }))
    prefs = load_tracker_preferences(p)
    assert prefs.category_priority == ()


def test_bad_tracker_order_type_ignored(tmp_path: Path) -> None:
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({
        "enabled": True,
        "tracker_order": ["energy", "metal"],   # wrong type
    }))
    prefs = load_tracker_preferences(p)
    assert prefs.tracker_order == {}


def test_ignores_unknown_keys(tmp_path: Path) -> None:
    """The about/notes fields at the root shouldn't error out."""
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({
        "_about": "free-form documentation",
        "enabled": True,
        "category_priority": ["metal"],
    }))
    prefs = load_tracker_preferences(p)
    assert prefs.enabled is True
    assert prefs.category_priority == ("metal",)


# ---------------------------------------------------------------------------
# sort-key helpers
# ---------------------------------------------------------------------------
def test_category_sort_key_listed_order() -> None:
    prefs = TrackerPreferences(
        enabled=True,
        category_priority=("metal", "energy", "fx"),
    )
    assert category_sort_key("metal", prefs) < category_sort_key("energy", prefs)
    assert category_sort_key("energy", prefs) < category_sort_key("fx", prefs)


def test_category_sort_key_unlisted_goes_last() -> None:
    prefs = TrackerPreferences(
        enabled=True,
        category_priority=("metal", "energy"),
    )
    assert category_sort_key("energy", prefs) < category_sort_key("crypto", prefs)
    assert category_sort_key("fx", prefs) == category_sort_key("crypto", prefs)


def test_category_sort_key_disabled_is_flat() -> None:
    prefs = TrackerPreferences(enabled=False, category_priority=("metal",))
    # All categories compare equal when prefs are disabled.
    assert category_sort_key("metal", prefs) == category_sort_key("fx", prefs)


def test_tracker_sort_key_within_category() -> None:
    prefs = TrackerPreferences(
        enabled=True,
        tracker_order={"energy": ("gas_retail", "electricity")},
    )
    assert (
        tracker_sort_key("gas_retail", "energy", prefs)
        < tracker_sort_key("electricity", "energy", prefs)
    )
    # Unlisted tracker in a listed category sorts at the tail.
    assert (
        tracker_sort_key("electricity", "energy", prefs)
        < tracker_sort_key("unknown_tracker", "energy", prefs)
    )


def test_tracker_sort_key_unknown_category_is_flat() -> None:
    prefs = TrackerPreferences(
        enabled=True,
        tracker_order={"energy": ("gas_retail",)},
    )
    # A category not in tracker_order has no per-tracker ordering.
    assert (
        tracker_sort_key("gold_usd", "metal", prefs)
        == tracker_sort_key("silver_usd", "metal", prefs)
    )


# ---------------------------------------------------------------------------
# state → PADD resolver + param / label substitution
# ---------------------------------------------------------------------------
def test_state_to_padd_known_states() -> None:
    from bridge.trackers.fetch import state_to_padd
    assert state_to_padd("PA") == "R1Y"
    assert state_to_padd("pa") == "R1Y"        # case-insensitive
    assert state_to_padd("  PA  ") == "R1Y"    # whitespace-tolerant
    assert state_to_padd("CA") == "R50"
    assert state_to_padd("TX") == "R30"
    assert state_to_padd("ME") == "R1X"        # PADD 1A
    assert state_to_padd("DC") == "R1Y"


def test_state_to_padd_falls_back_to_default() -> None:
    from bridge.trackers.fetch import DEFAULT_PADD, state_to_padd
    assert state_to_padd("") == DEFAULT_PADD
    assert state_to_padd(None) == DEFAULT_PADD
    assert state_to_padd("PR") == DEFAULT_PADD  # territory, not mapped
    assert state_to_padd("ZZ") == DEFAULT_PADD


def test_resolve_params_substitutes_home_state_padd() -> None:
    from bridge.trackers.preferences import resolve_params
    prefs = TrackerPreferences(enabled=True, home_state="CA")
    got = resolve_params({
        "duoarea": "${home_state_padd}",
        "product": "EPMR",
        "literal": "R1Y",
    }, prefs)
    assert got == {"duoarea": "R50", "product": "EPMR", "literal": "R1Y"}


def test_resolve_params_substitutes_home_state_raw() -> None:
    from bridge.trackers.preferences import resolve_params
    prefs = TrackerPreferences(enabled=True, home_state="pa")
    # home_state is upper-cased on load; simulate that here directly.
    prefs_up = TrackerPreferences(enabled=True, home_state="PA")
    got = resolve_params({"state": "${home_state}"}, prefs_up)
    assert got == {"state": "PA"}


def test_resolve_params_empty_home_state_falls_back_to_default() -> None:
    from bridge.trackers.fetch import DEFAULT_PADD
    from bridge.trackers.preferences import resolve_params
    prefs = TrackerPreferences(enabled=True, home_state="")
    got = resolve_params({"duoarea": "${home_state_padd}"}, prefs)
    # Missing state still resolves (to the default PADD) so a missing
    # preference never ends up sending a literal "${home_state_padd}"
    # to the upstream API.
    assert got == {"duoarea": DEFAULT_PADD}


def test_resolve_params_passes_through_non_string_values() -> None:
    from bridge.trackers.preferences import resolve_params
    prefs = TrackerPreferences(enabled=True, home_state="PA")
    got = resolve_params({
        "duoarea": "${home_state_padd}",
        "limit": 10,
        "facets": ["a", "b"],
    }, prefs)
    assert got["duoarea"] == "R1Y"
    assert got["limit"] == 10
    assert got["facets"] == ["a", "b"]


def test_resolve_label_substitutes() -> None:
    from bridge.trackers.preferences import resolve_label
    prefs = TrackerPreferences(enabled=True, home_state="CA")
    assert resolve_label("Retail Gas — ${home_state_padd}", prefs) == "Retail Gas — R50"
    assert resolve_label("Gold", prefs) == "Gold"   # passthrough


def test_resolve_preserves_unknown_placeholders() -> None:
    """An unknown placeholder is left intact so a typo surfaces in
    logs / debugging rather than being silently swallowed."""
    from bridge.trackers.preferences import resolve_params
    prefs = TrackerPreferences(enabled=True, home_state="PA")
    got = resolve_params({"p": "${not_a_real_key}"}, prefs)
    assert got == {"p": "${not_a_real_key}"}


# ---------------------------------------------------------------------------
# is_tracker_visible — user enable/disable overrides config.enabled
# ---------------------------------------------------------------------------

from bridge.trackers.preferences import is_tracker_visible


class TestIsTrackerVisible:
    def test_prefs_disabled_returns_config_enabled(self):
        # When the whole prefs object is disabled, config wins.
        prefs = TrackerPreferences(enabled=False)
        assert is_tracker_visible("gold_usd", "metal", True, prefs) is True
        assert is_tracker_visible("btc_usd", "crypto", False, prefs) is False

    def test_tracker_override_wins_over_config(self):
        # User force-enables a config-disabled tracker (crypto case).
        prefs = TrackerPreferences(
            enabled=True, tracker_overrides={"btc_usd": True},
        )
        assert is_tracker_visible("btc_usd", "crypto", False, prefs) is True
        # User force-disables a config-enabled tracker.
        prefs2 = TrackerPreferences(
            enabled=True, tracker_overrides={"gold_usd": False},
        )
        assert is_tracker_visible("gold_usd", "metal", True, prefs2) is False

    def test_disabled_categories_blanket_hides(self):
        prefs = TrackerPreferences(
            enabled=True, disabled_categories=("fx",),
        )
        assert is_tracker_visible("usd_cny", "fx", True, prefs) is False
        # Unrelated category unaffected.
        assert is_tracker_visible("gold_usd", "metal", True, prefs) is True

    def test_tracker_override_beats_category_disable(self):
        # User disabled a whole category but enabled one tracker in it
        # — override wins. Lets you pin a single tracker from an
        # otherwise-hidden group.
        prefs = TrackerPreferences(
            enabled=True,
            disabled_categories=("fx",),
            tracker_overrides={"usd_jpy": True},
        )
        assert is_tracker_visible("usd_jpy", "fx", True, prefs) is True
        assert is_tracker_visible("usd_cny", "fx", True, prefs) is False


# ---------------------------------------------------------------------------
# load_tracker_preferences — new schema fields round-trip cleanly
# ---------------------------------------------------------------------------

def test_loads_tracker_overrides_and_disabled_categories(tmp_path: Path) -> None:
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({
        "enabled": True,
        "tracker_overrides": {"btc_usd": True, "gold_usd": False},
        "disabled_categories": ["fx"],
    }))
    prefs = load_tracker_preferences(p)
    assert prefs.tracker_overrides == {"btc_usd": True, "gold_usd": False}
    assert prefs.disabled_categories == ("fx",)


def test_malformed_overrides_are_ignored(tmp_path: Path) -> None:
    # Non-bool override values skipped (typed-in string etc.). Non-dict
    # overrides value ignored wholesale with a warning. Users shouldn't
    # lose the rest of their prefs because one field is wrong.
    p = tmp_path / "prefs.json"
    p.write_text(json.dumps({
        "enabled": True,
        "tracker_overrides": {"btc_usd": "yes", "gold_usd": False},
        "disabled_categories": "not a list",
    }))
    prefs = load_tracker_preferences(p)
    assert prefs.tracker_overrides == {"gold_usd": False}
    assert prefs.disabled_categories == ()
