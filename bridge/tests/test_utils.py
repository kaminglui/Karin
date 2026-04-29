"""Tests for bridge.utils.load_config — env-var + character overrides.

All these paths are what Docker-deployed instances use to avoid
editing `assistant.yaml` per-host. Keeping them under test ensures a
future refactor can't silently break the deploy ergonomics.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bridge.utils import load_config


@pytest.fixture
def yaml_factory(tmp_path):
    """Write a minimal assistant.yaml-shaped file to tmp and return its Path."""
    def make(body: str) -> Path:
        p = tmp_path / "assistant.yaml"
        p.write_text(textwrap.dedent(body), encoding="utf-8")
        return p
    return make


BASE = """
    llm:
      backend: ollama
      base_url: http://localhost:11434
      model: original-model
      num_ctx: 4096
      system_prompt: |
        Inline prompt content.
    stt:
      device: cuda
      compute_type: float16
    tts:
      base_url: http://localhost:9880
"""


def _base_with_character(name: str) -> str:
    """Return a YAML with `character:` key before llm: — appending after
    the block-scalar `system_prompt:` would put the key inside the scalar."""
    return f"character: {name}\n" + textwrap.dedent(BASE)


class TestEnvOverrides:
    def test_ollama_base_url_override(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_OLLAMA_BASE_URL", "http://ollama-host:11434")
        cfg = load_config(p)
        assert cfg["llm"]["base_url"] == "http://ollama-host:11434"

    def test_tts_base_url_override(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_TTS_BASE_URL", "http://sovits-host:9880")
        cfg = load_config(p)
        assert cfg["tts"]["base_url"] == "http://sovits-host:9880"

    def test_stt_device_and_compute_type(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_STT_DEVICE", "cpu")
        monkeypatch.setenv("KARIN_STT_COMPUTE_TYPE", "int8")
        cfg = load_config(p)
        assert cfg["stt"]["device"] == "cpu"
        assert cfg["stt"]["compute_type"] == "int8"

    def test_llm_model_override(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_LLM_MODEL", "qwen3.5:2b")
        cfg = load_config(p)
        assert cfg["llm"]["model"] == "qwen3.5:2b"

    def test_llm_timeout_parses_float(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_LLM_TIMEOUT", "180")
        cfg = load_config(p)
        assert cfg["llm"]["request_timeout"] == 180.0

    def test_llm_timeout_ignored_if_garbage(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_LLM_TIMEOUT", "not-a-number")
        cfg = load_config(p)
        # Silent ignore — no crash, no override applied.
        assert "request_timeout" not in cfg["llm"]

    def test_num_ctx_override(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_NUM_CTX", "3072")
        cfg = load_config(p)
        assert cfg["llm"]["num_ctx"] == 3072

    def test_num_ctx_ignored_if_garbage(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_NUM_CTX", "mumble")
        cfg = load_config(p)
        assert cfg["llm"]["num_ctx"] == 4096   # unchanged default

    def test_no_env_keeps_yaml_values(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        for v in ["KARIN_OLLAMA_BASE_URL", "KARIN_TTS_BASE_URL",
                  "KARIN_STT_DEVICE", "KARIN_STT_COMPUTE_TYPE",
                  "KARIN_STT_LANGUAGE", "KARIN_STT_MODEL",
                  "KARIN_LLM_MODEL", "KARIN_LLM_TIMEOUT", "KARIN_NUM_CTX"]:
            monkeypatch.delenv(v, raising=False)
        cfg = load_config(p)
        assert cfg["llm"]["base_url"] == "http://localhost:11434"
        assert cfg["stt"]["device"] == "cuda"

    def test_stt_language_override(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_STT_LANGUAGE", "auto")
        cfg = load_config(p)
        assert cfg["stt"]["language"] == "auto"

    def test_stt_model_override(self, yaml_factory, monkeypatch):
        p = yaml_factory(BASE)
        monkeypatch.setenv("KARIN_STT_MODEL", "small")
        cfg = load_config(p)
        assert cfg["stt"]["model"] == "small"


class TestCharacterLoading:
    def test_character_from_yaml_key_is_loaded(self, yaml_factory, tmp_path, monkeypatch):
        p = yaml_factory(_base_with_character("my_persona"))
        # Place the character file in the tmp 'config/characters' layout.
        # load_config uses REPO_ROOT — monkeypatch that to tmp_path's parent.
        from bridge import utils as utils_mod
        monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
        (tmp_path / "config" / "characters").mkdir(parents=True)
        (tmp_path / "config" / "characters" / "my_persona.yaml").write_text(
            textwrap.dedent("""
                system_prompt: |
                  You are a pirate. Arrr.
            """),
            encoding="utf-8",
        )
        cfg = load_config(p)
        assert cfg["llm"]["system_prompt"].startswith("You are a pirate")

    def test_env_override_beats_yaml_character(self, yaml_factory, tmp_path, monkeypatch):
        p = yaml_factory(_base_with_character("persona_a"))
        from bridge import utils as utils_mod
        monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
        (tmp_path / "config" / "characters").mkdir(parents=True)
        (tmp_path / "config" / "characters" / "persona_a.yaml").write_text(
            "system_prompt: AAA\n", encoding="utf-8",
        )
        (tmp_path / "config" / "characters" / "persona_b.yaml").write_text(
            "system_prompt: BBB\n", encoding="utf-8",
        )
        monkeypatch.setenv("KARIN_CHARACTER", "persona_b")
        cfg = load_config(p)
        assert cfg["llm"]["system_prompt"] == "BBB"

    def test_missing_character_file_falls_back_to_inline(self, yaml_factory, tmp_path, monkeypatch):
        p = yaml_factory(_base_with_character("nonexistent"))
        from bridge import utils as utils_mod
        monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
        cfg = load_config(p)
        # Character file wasn't found → inline system_prompt from BASE wins.
        assert "Inline prompt content" in cfg["llm"]["system_prompt"]

    def test_no_character_key_falls_back_to_default(self, yaml_factory, tmp_path, monkeypatch):
        # No `character:` key anywhere → falls back to the "default"
        # neutral character. The REAL repo root (not monkeypatched
        # here) has characters/profile.yaml + characters/default/voice.yaml
        # so the template is loaded with "default" persona/language_note
        # — NOT the legacy inline fallback. Asserts the new "always
        # resolve to a character" contract.
        p = yaml_factory(BASE)
        monkeypatch.delenv("KARIN_CHARACTER", raising=False)
        cfg = load_config(p)
        prompt = cfg["llm"]["system_prompt"]
        # The shipped profile.yaml contains this section header; its
        # presence proves we loaded the template rather than the legacy
        # inline prompt.
        assert "RULE ZERO" in prompt
        # The inline BASE prompt should NOT be present.
        assert "Inline prompt content" not in prompt

    def test_no_character_key_with_monkeypatched_root_keeps_inline(
        self, yaml_factory, tmp_path, monkeypatch,
    ):
        # When REPO_ROOT is monkeypatched to a tmp dir WITHOUT any
        # characters/ scaffolding, the default character path can't
        # resolve and we fall back to the inline system_prompt — the
        # silent-safety pathway for odd test / CI environments.
        p = yaml_factory(BASE)
        monkeypatch.delenv("KARIN_CHARACTER", raising=False)
        from bridge import utils as utils_mod
        monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
        cfg = load_config(p)
        assert "Inline prompt content" in cfg["llm"]["system_prompt"]

    def test_malformed_character_file_keeps_inline(self, yaml_factory, tmp_path, monkeypatch):
        p = yaml_factory(_base_with_character("broken"))
        from bridge import utils as utils_mod
        monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
        (tmp_path / "config" / "characters").mkdir(parents=True)
        (tmp_path / "config" / "characters" / "broken.yaml").write_text(
            "::: this is not valid yaml :::",
            encoding="utf-8",
        )
        cfg = load_config(p)
        # Broken yaml → silent fallback to inline prompt (no crash).
        assert "Inline prompt content" in cfg["llm"]["system_prompt"]
