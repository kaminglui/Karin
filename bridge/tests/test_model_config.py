"""Tests for the per-model tuning overlay loader."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from bridge import model_config


@pytest.fixture
def fake_yaml(tmp_path: Path) -> Path:
    """Write a minimal models.yaml fixture that model_config will read."""
    p = tmp_path / "models.yaml"
    p.write_text(
        """
defaults:
  max_per_tool: 1
  max_tool_iters: 5
  temperature: 0.3
  num_ctx: 3072
  think: "off"
  request_timeout: 300

models:
  "karin-tuned:latest":
    temperature: 0.25
    num_ctx: 2048
  "big:7b":
    max_per_tool: 2
    max_tool_iters: 6
    num_ctx: 4096
    request_timeout: 600
""",
        encoding="utf-8",
    )
    return p


def test_overrides_applied_to_known_tag(fake_yaml: Path) -> None:
    """A tag listed in models: gets its knobs merged on top of defaults."""
    with patch.object(model_config, "MODELS_YAML", fake_yaml):
        cfg = model_config.resolve_for("big:7b")
    assert cfg["max_per_tool"] == 2
    assert cfg["max_tool_iters"] == 6
    assert cfg["num_ctx"] == 4096
    assert cfg["request_timeout"] == 600
    # Untouched keys come from defaults
    assert cfg["temperature"] == 0.3
    assert cfg["think"] == "off"


def test_unknown_tag_falls_back_to_defaults(fake_yaml: Path) -> None:
    """A tag not listed still gets a valid config — the defaults block."""
    with patch.object(model_config, "MODELS_YAML", fake_yaml):
        cfg = model_config.resolve_for("unheard_of:42b")
    assert cfg["max_per_tool"] == 1
    assert cfg["max_tool_iters"] == 5
    assert cfg["temperature"] == 0.3
    assert cfg["num_ctx"] == 3072


def test_missing_file_uses_hardcoded_fallback(tmp_path: Path) -> None:
    """If config/models.yaml doesn't exist, the hardcoded fallback keeps
    the server bootable."""
    with patch.object(model_config, "MODELS_YAML", tmp_path / "does_not_exist.yaml"):
        cfg = model_config.resolve_for("whatever:1b")
    # Matches the fallback block in resolve_for()
    assert cfg["max_per_tool"] == 1
    assert cfg["max_tool_iters"] == 5
    assert cfg["num_ctx"] == 3072


def test_empty_file_uses_fallback(tmp_path: Path) -> None:
    """An empty YAML file shouldn't crash — empty dict means no overrides."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with patch.object(model_config, "MODELS_YAML", empty):
        cfg = model_config.resolve_for("anything")
    assert cfg["max_per_tool"] == 1


def test_partial_override_only_touches_named_keys(fake_yaml: Path) -> None:
    """Verifies shallow-merge semantics: a per-model entry with just one
    key leaves the rest of the defaults intact."""
    with patch.object(model_config, "MODELS_YAML", fake_yaml):
        cfg = model_config.resolve_for("karin-tuned:latest")
    assert cfg["temperature"] == 0.25   # overridden
    assert cfg["num_ctx"] == 2048       # overridden
    assert cfg["max_per_tool"] == 1     # from defaults (untouched)
    assert cfg["max_tool_iters"] == 5   # from defaults
