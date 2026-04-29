"""Tests for bridge.news.noun_extract — the spaCy-backed NER entity
extractor used as a second label source in the graph layer.

These tests don't load the real spaCy model (keeps the test suite
dep-free and fast). Instead we patch `_load_nlp` to return a fake
pipeline that yields canned Doc-shaped objects — enough surface to
exercise the filter/clean/aggregate logic.
"""
from __future__ import annotations

from unittest.mock import patch

from bridge.news import noun_extract


# --- fake spaCy plumbing ---------------------------------------------------

class _FakeEnt:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNlp:
    """Minimal spaCy pipeline stand-in. Constructed with a dict
    {text: [(ent_text, ent_label), ...]}; returns _FakeDoc objects
    whose .ents reflect the dict."""

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, text):
        return _FakeDoc([_FakeEnt(t, l) for t, l in self.mapping.get(text, [])])

    def pipe(self, texts, batch_size=64):
        for t in texts:
            yield self(t)


def _patch_nlp(ents_by_text):
    """Context manager to inject a fake nlp singleton into the module."""
    return patch.object(
        noun_extract, "_load_nlp", lambda: _FakeNlp(ents_by_text),
    )


# --- cleanup helpers ------------------------------------------------------

class TestCleanLabel:
    def test_strip_ascii_possessive(self):
        assert noun_extract._clean_label("Trump's") == "Trump"

    def test_strip_unicode_possessive(self):
        assert noun_extract._clean_label("Trump\u2019s") == "Trump"

    def test_strip_whitespace_and_punct(self):
        assert noun_extract._clean_label(" Tokyo . ") == "Tokyo"

    def test_no_change_when_clean(self):
        assert noun_extract._clean_label("Xi Jinping") == "Xi Jinping"


# --- extract_entities (single text) ---------------------------------------

class TestExtractEntities:
    def test_keeps_proper_noun_types(self):
        mapping = {
            "text": [("Trump", "PERSON"), ("Washington", "GPE"), ("ASML", "ORG")],
        }
        with _patch_nlp(mapping):
            out = noun_extract.extract_entities("text")
        assert out == ["Trump", "Washington", "ASML"]

    def test_drops_adjectival_norp(self):
        # NORP = nationalities / political groups. "Chinese" as an
        # adjective should NOT land on the graph.
        mapping = {"text": [("Chinese", "NORP"), ("China", "GPE")]}
        with _patch_nlp(mapping):
            out = noun_extract.extract_entities("text")
        assert "Chinese" not in out
        assert "China" in out

    def test_drops_numeric_types(self):
        mapping = {"text": [
            ("2026", "DATE"), ("$5 billion", "MONEY"),
            ("ASML", "ORG"),
        ]}
        with _patch_nlp(mapping):
            out = noun_extract.extract_entities("text")
        assert out == ["ASML"]

    def test_strips_possessive_from_span(self):
        # spaCy sometimes includes the clitic in a span.
        mapping = {"text": [("Trump's", "PERSON")]}
        with _patch_nlp(mapping):
            out = noun_extract.extract_entities("text")
        assert out == ["Trump"]

    def test_empty_input(self):
        assert noun_extract.extract_entities("") == []
        assert noun_extract.extract_entities(None) == []

    def test_nlp_unavailable_returns_empty(self):
        # _load_nlp returns None when spaCy isn't installed or the
        # model load failed. Caller gets an empty list, no exception.
        with patch.object(noun_extract, "_load_nlp", lambda: None):
            assert noun_extract.extract_entities("anything") == []


# --- rank_corpus_phrases ---------------------------------------------------

class TestRankCorpusPhrases:
    def test_min_count_drops_rare(self):
        mapping = {
            "a": [("ASML", "ORG")],
            "b": [("ASML", "ORG"), ("Zeiss", "ORG")],
        }
        with _patch_nlp(mapping):
            out = noun_extract.rank_corpus_phrases(["a", "b"], min_count=2)
        labels = [p for p, _ in out]
        assert "ASML" in labels
        assert "Zeiss" not in labels

    def test_case_insensitive_dedup(self):
        mapping = {
            "t1": [("New York", "GPE")],
            "t2": [("NEW YORK", "GPE")],
        }
        with _patch_nlp(mapping):
            out = noun_extract.rank_corpus_phrases(["t1", "t2"], min_count=2)
        # One entry, count = 2, first-seen display-form wins.
        assert len(out) == 1
        assert out[0][0] == "New York"
        assert out[0][1] == 2

    def test_sorted_by_count_desc(self):
        mapping = {
            "a": [("Iran", "GPE")],
            "b": [("Iran", "GPE"), ("Israel", "GPE")],
            "c": [("Iran", "GPE"), ("Israel", "GPE"), ("Lebanon", "GPE")],
        }
        with _patch_nlp(mapping):
            out = noun_extract.rank_corpus_phrases(
                ["a", "b", "c"], min_count=1,
            )
        assert [p for p, _ in out[:3]] == ["Iran", "Israel", "Lebanon"]

    def test_top_n_cap(self):
        mapping = {
            chr(97 + i): [(f"Ent{i}", "PERSON"), (f"Ent{i}", "PERSON")]
            for i in range(10)
        }
        with _patch_nlp(mapping):
            out = noun_extract.rank_corpus_phrases(
                list(mapping.keys()), min_count=2, top_n=5,
            )
        assert len(out) == 5

    def test_empty_corpus(self):
        with _patch_nlp({}):
            assert noun_extract.rank_corpus_phrases([]) == []
            assert noun_extract.rank_corpus_phrases([""]) == []
            assert noun_extract.rank_corpus_phrases([None]) == []

    def test_nlp_unavailable_returns_empty(self):
        with patch.object(noun_extract, "_load_nlp", lambda: None):
            assert noun_extract.rank_corpus_phrases(["anything"]) == []
