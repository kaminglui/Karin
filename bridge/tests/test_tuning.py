"""Tests for bridge.tuning — the YAML-backed tuning-knob loader."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from bridge import tuning


@pytest.fixture(autouse=True)
def _clean_cache():
    """Bust the lru_cache around _load before AND after each test so
    state from one case can't bleed into another."""
    tuning.reload()
    yield
    tuning.reload()


def _point_at(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(tuning, "TUNING_YAML", path)


# ---------------------------------------------------------------------------
# Missing / empty / malformed file → defaults win
# ---------------------------------------------------------------------------
def test_missing_file_returns_default(tmp_path: Path, monkeypatch) -> None:
    _point_at(monkeypatch, tmp_path / "nope.yaml")
    assert tuning.get("anything.at.all", 42) == 42
    assert tuning.get("anything.at.all", "fallback") == "fallback"


def test_malformed_yaml_falls_back(tmp_path: Path, monkeypatch, caplog) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text(": :\n  - bad indent", encoding="utf-8")  # invalid YAML
    _point_at(monkeypatch, p)
    with caplog.at_level("WARNING"):
        out = tuning.get("foo.bar", 7)
    assert out == 7
    assert any("failed to parse" in r.message for r in caplog.records)


def test_non_mapping_root_falls_back(tmp_path: Path, monkeypatch, caplog) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    _point_at(monkeypatch, p)
    with caplog.at_level("WARNING"):
        out = tuning.get("foo", 7)
    # Either the loader's "must be a YAML mapping" branch fires, or
    # bridge.utils.load_config raises during its own validation —
    # both surface as a WARNING + fallback to defaults. Tolerating
    # either keeps the test resilient to changes in load_config.
    assert out == 7
    assert any(
        "must be a YAML mapping" in r.message
        or "failed to parse" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Happy-path lookups
# ---------------------------------------------------------------------------
def test_returns_yaml_value_when_present(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text(
        "news:\n  cluster:\n    title_jaccard: 0.42\n",
        encoding="utf-8",
    )
    _point_at(monkeypatch, p)
    assert tuning.get("news.cluster.title_jaccard", 0.99) == 0.42


def test_missing_key_returns_default(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text("news:\n  cluster:\n    title_jaccard: 0.42\n", encoding="utf-8")
    _point_at(monkeypatch, p)
    # Sibling key not present → default.
    assert tuning.get("news.cluster.something_else", 99) == 99
    # Whole branch missing → default.
    assert tuning.get("alerts.scan_ttl_minutes", 15) == 15


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------
def test_int_default_coerces_yaml_string(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text("digest:\n  max_news_items: '7'\n", encoding="utf-8")  # string in yaml
    _point_at(monkeypatch, p)
    assert tuning.get("digest.max_news_items", 5) == 7


def test_float_default_coerces_int(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text("alerts:\n  nws:\n    fetch_timeout_s: 12\n", encoding="utf-8")
    _point_at(monkeypatch, p)
    out = tuning.get("alerts.nws.fetch_timeout_s", 15.0)
    assert out == 12.0
    assert isinstance(out, float)


def test_uncoercible_value_falls_back_with_warning(tmp_path: Path, monkeypatch, caplog) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text("digest:\n  max_news_items: 'seven'\n", encoding="utf-8")
    _point_at(monkeypatch, p)
    with caplog.at_level("WARNING"):
        out = tuning.get("digest.max_news_items", 5)
    assert out == 5
    assert any("cannot coerce" in r.message for r in caplog.records)


def test_bool_coercion_from_string(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text("flags:\n  on_str: 'true'\n  off_str: 'false'\n", encoding="utf-8")
    _point_at(monkeypatch, p)
    assert tuning.get("flags.on_str", False) is True
    assert tuning.get("flags.off_str", True) is False


def test_bool_with_garbage_falls_back(tmp_path: Path, monkeypatch, caplog) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text("flags:\n  weird: 'maybe'\n", encoding="utf-8")
    _point_at(monkeypatch, p)
    with caplog.at_level("WARNING"):
        out = tuning.get("flags.weird", False)
    assert out is False


# ---------------------------------------------------------------------------
# Cache + reload
# ---------------------------------------------------------------------------
def test_reload_picks_up_new_values(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text("digest:\n  max_news_items: 3\n", encoding="utf-8")
    _point_at(monkeypatch, p)
    assert tuning.get("digest.max_news_items", 5) == 3
    # Edit the file — without reload, lru_cache still returns the old value.
    p.write_text("digest:\n  max_news_items: 9\n", encoding="utf-8")
    assert tuning.get("digest.max_news_items", 5) == 3
    tuning.reload()
    assert tuning.get("digest.max_news_items", 5) == 9


def test_snapshot_returns_full_config(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "tuning.yaml"
    p.write_text(
        "news:\n  cluster:\n    title_jaccard: 0.5\n",
        encoding="utf-8",
    )
    _point_at(monkeypatch, p)
    snap = tuning.snapshot()
    assert snap == {"news": {"cluster": {"title_jaccard": 0.5}}}
