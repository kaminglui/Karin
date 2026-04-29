"""Tests for the fill_format tool — directive shape + skeleton helper."""
from __future__ import annotations

import pytest

from bridge.tools import _fill_format, _format_skeleton


class TestFormatSkeleton:
    def test_extracts_non_slot_text(self):
        t = "Mood: {one mood}; Vibe: {atmospheric noun}"
        s = _format_skeleton(t)
        assert "Mood:" in s
        assert "Vibe:" in s
        assert "{" not in s and "}" not in s

    def test_collapses_whitespace(self):
        t = "Line 1: {a}\n  Line 2: {b}"
        s = _format_skeleton(t)
        # Newlines and runs of spaces both collapse to single spaces
        # so trivial indentation drift doesn't fail validation later.
        assert "  " not in s
        assert "\n" not in s


class TestFillFormatDirective:
    def test_missing_template_errors(self):
        out = _fill_format("", topic="x")
        assert out.startswith("Error:")

    def test_no_slots_errors(self):
        out = _fill_format("just a plain string", topic="x")
        assert out.startswith("Error:")
        assert "placeholders" in out.lower()

    def test_directive_mentions_template(self):
        out = _fill_format("Mood: {one word}; Vibe: {noun}", topic="rain")
        assert "Mood: {one word}" in out
        assert "Vibe: {noun}" in out

    def test_directive_lists_slots(self):
        out = _fill_format("A: {one}; B: {two}; C: {three}", topic="x")
        # Slot descriptions echoed back so model pays attention
        assert "{one}" in out
        assert "{two}" in out
        assert "{three}" in out
        assert "3" in out

    def test_topic_and_style_echoed(self):
        out = _fill_format(
            "Flavor: {adjective}",
            topic="green tea",
            style="poetic",
        )
        assert "green tea" in out
        assert "poetic" in out

    def test_style_optional(self):
        out = _fill_format("Flavor: {adjective}", topic="green tea")
        assert "green tea" in out
        # No style = no style line
        assert "style" not in out.lower() or "no additional" in out.lower()

    def test_preservation_rule_in_directive(self):
        out = _fill_format("X: {a}", topic="t")
        assert "PRESERVE" in out or "preserve" in out.lower()

    def test_output_only_rule_in_directive(self):
        out = _fill_format("X: {a}", topic="t")
        # Model is explicitly told no preamble / no commentary
        low = out.lower()
        assert "only" in low
        assert "preamble" in low or "commentary" in low or "no prose" in low
