"""Tests for bridge.stt.resolve_stt_settings.

The model-loading half of WhisperSTT can't run in CI (needs CUDA +
faster-whisper), so these tests pin the pure-function normalizer that
decides which model tag + language to actually hand to the engine.
"""
from __future__ import annotations

import logging

import pytest

from bridge.stt import resolve_stt_settings


class TestResolveSttSettings:
    def test_english_config_passes_through(self):
        model, lang = resolve_stt_settings("small.en", "en")
        assert model == "small.en"
        assert lang == "en"

    def test_auto_with_en_model_strips_suffix(self, caplog):
        caplog.set_level(logging.WARNING, logger="bridge.stt")
        model, lang = resolve_stt_settings("small.en", "auto")
        assert model == "small"
        assert lang is None
        # Warning surfaced so users don't silently run the wrong model
        assert any("English-only" in r.message for r in caplog.records)

    def test_auto_with_multilingual_model_unchanged(self):
        model, lang = resolve_stt_settings("small", "auto")
        assert model == "small"
        assert lang is None

    def test_non_english_locks_to_language(self):
        model, lang = resolve_stt_settings("small", "ja")
        assert model == "small"
        assert lang == "ja"

    def test_non_english_with_en_model_strips_suffix(self, caplog):
        caplog.set_level(logging.WARNING, logger="bridge.stt")
        model, lang = resolve_stt_settings("base.en", "es")
        assert model == "base"
        assert lang == "es"
        assert any("English-only" in r.message for r in caplog.records)

    def test_empty_language_treated_as_auto(self):
        model, lang = resolve_stt_settings("small.en", "")
        # Empty → auto → strip .en
        assert model == "small"
        assert lang is None

    def test_none_language_treated_as_auto(self):
        model, lang = resolve_stt_settings("small", None)
        assert lang is None

    @pytest.mark.parametrize("value", ["AUTO", "Auto", " auto ", "aUtO"])
    def test_auto_is_case_insensitive(self, value):
        _model, lang = resolve_stt_settings("small", value)
        assert lang is None

    def test_uppercase_language_code_lowercased(self):
        _model, lang = resolve_stt_settings("small", "EN")
        assert lang == "en"

    def test_local_path_not_altered(self):
        # Local path ending in ".en" is not treated as a tag suffix.
        # faster-whisper accepts local paths; stripping would break them.
        # (Our guard only triggers on the tag-style suffix form, so a
        # realistic path like /models/whisper-small passes through.)
        model, lang = resolve_stt_settings("/models/whisper-small", "auto")
        assert model == "/models/whisper-small"
        assert lang is None
