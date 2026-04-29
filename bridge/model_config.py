"""Per-model tuning overlay loader.

Resolves the knobs in ``config/models.yaml`` for the currently-selected
LLM tag. Entries under ``models.<tag>`` override the ``defaults`` block
by shallow merge; unknown tags fall back to ``defaults``.

The returned dict is plain primitives so it can be:
  1. Merged into the ``assistant.yaml`` ``llm`` block at startup, and
  2. Passed as kwargs-ish into :class:`bridge.llm.OllamaLLM` (for the
     tool-loop caps that aren't Ollama API fields).

Example::

    cfg = resolve_for("huihui_ai/qwen3.5-abliterated:4b")
    # cfg == {
    #     "max_per_tool": 1, "max_tool_iters": 4,
    #     "temperature": 0.35, "num_ctx": 4096,
    #     "think": "off", "request_timeout": 420,
    # }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bridge.utils import REPO_ROOT, load_config

log = logging.getLogger(__name__)

MODELS_YAML = REPO_ROOT / "config" / "models.yaml"


def _load_raw() -> dict:
    """Load the YAML file; empty dict if missing so callers can rely
    on the defaults-only fallback without a guard."""
    if not MODELS_YAML.exists():
        log.info("%s not found — using hardcoded defaults", MODELS_YAML)
        return {}
    try:
        return load_config(MODELS_YAML)
    except Exception as e:
        log.warning("failed to parse %s: %s — using defaults", MODELS_YAML, e)
        return {}


def resolve_for(model_tag: str) -> dict[str, Any]:
    """Return merged tuning config for ``model_tag``.

    Merge order: hardcoded fallback → ``defaults`` block → ``models.<tag>``.
    Unknown tags only use the first two layers, which is what we want —
    a missing entry should still boot with sane knobs, not crash.
    """
    fallback: dict[str, Any] = {
        "max_per_tool": 1,
        "max_tool_iters": 5,
        "temperature": 0.3,
        "num_ctx": 3072,
        "think": "off",
        "request_timeout": 300,
    }
    raw = _load_raw()
    merged = dict(fallback)
    merged.update(raw.get("defaults") or {})
    overrides = (raw.get("models") or {}).get(model_tag) or {}
    merged.update(overrides)
    if overrides:
        log.info("model_config: %s → overrides applied: %s",
                 model_tag, sorted(overrides.keys()))
    else:
        log.info("model_config: %s → no per-model entry, using defaults", model_tag)
    return merged
