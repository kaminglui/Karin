"""spaCy-backed named-entity extractor for the news graph.

Replaces the earlier regex-based extractor. spaCy's en_core_web_sm
model ships with an NER head trained on OntoNotes (news corpus) and
gives us typed entities out of the box — PERSON, ORG, GPE
(country / state / city), EVENT, FAC (airport, highway, building),
LOC (non-GPE location like "Middle East" or "Strait of Hormuz"). We
filter out types that don't belong on the graph (DATE, MONEY,
PERCENT, QUANTITY, NORP, LANGUAGE, etc.) so adjectival forms like
"Chinese" (NORP) and numeric tokens never become nodes.

Design rules:

* The spaCy model loads lazily on first call and is cached in a
  module-level singleton. Load time is ~200 ms; inference is fast on
  short headlines (~1-5 ms each with nlp.pipe batching).
* Fail-soft: if spaCy or the model isn't installed (minimal Docker
  image, test env), ``rank_corpus_phrases`` returns an empty list
  with a warning — the graph still renders, just with only Qwen's
  label source.
* Public API is unchanged from the regex version —
  ``rank_corpus_phrases(texts, min_count, top_n)`` — so the graph
  module doesn't need to know which backend produced the labels.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable

from bridge import tuning as _tuning

log = logging.getLogger("bridge.news.noun_extract")

# Entity types we keep. Sourced from tuning.yaml so operators can
# change the filter without code edits — add NORP to get adjectival
# forms back, or drop EVENT if it's too noisy. Default matches the
# previous hard-coded set; tuning overrides win if present.
#
# OntoNotes full label set (for reference when editing tuning):
#   PERSON ORG GPE LOC FAC EVENT NORP PRODUCT WORK_OF_ART LAW
#   LANGUAGE DATE TIME PERCENT MONEY QUANTITY ORDINAL CARDINAL
_DEFAULT_KEEP = ("PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT")


def _load_keep_types() -> frozenset[str]:
    """Read the tuning-defined entity-type whitelist. Defaults to the
    same set noun_extract shipped with before. Invalid / non-string
    entries silently fall through — keeps a typo from crashing the
    graph build."""
    raw = _tuning.get("news.graph.keep_entity_types", _DEFAULT_KEEP)
    if not isinstance(raw, (list, tuple)):
        log.warning(
            "news.graph.keep_entity_types should be a list; got %r — "
            "using defaults", type(raw).__name__,
        )
        return frozenset(_DEFAULT_KEEP)
    keep = {str(t).strip().upper() for t in raw if isinstance(t, str) and t.strip()}
    if not keep:
        return frozenset(_DEFAULT_KEEP)
    return frozenset(keep)


# Cached on first access; the tuning loader itself caches the parsed
# YAML so a config edit + web restart picks up new values.
_KEEP_ENTITY_TYPES: frozenset[str] = _load_keep_types()

# Minimum length / occurrence / result-cap — all tunable.
_MIN_LABEL_LEN: int = int(_tuning.get("news.graph.min_label_length", 2))
_DEFAULT_MIN_OCCURRENCES: int = int(_tuning.get("news.graph.min_occurrences", 2))
_DEFAULT_TOP_N: int = int(_tuning.get("news.graph.top_n", 40))

# Import spacy lazily so test environments without the package still
# pass. `_nlp` caches the loaded model.
_nlp = None
_load_failed = False


def _load_nlp():
    """Return the cached spaCy pipeline. ``None`` on any import /
    model-load failure — caller treats as "no NER available"."""
    global _nlp, _load_failed
    if _nlp is not None:
        return _nlp
    if _load_failed:
        return None
    try:
        import spacy
        _nlp = spacy.load(
            "en_core_web_sm",
            # Disable components we don't need — lemmatizer, parser,
            # textcat — so the pipeline is NER-only. Cuts per-doc
            # latency roughly in half.
            disable=["parser", "lemmatizer", "textcat"],
        )
    except Exception as e:
        log.warning(
            "spaCy / en_core_web_sm unavailable: %s — noun_extract "
            "will return no labels. Qwen remains the sole label "
            "source for the graph.", e,
        )
        _load_failed = True
        _nlp = None
    return _nlp


def _clean_label(text: str) -> str:
    """Strip possessives + surrounding whitespace/punctuation.

    spaCy sometimes includes the possessive clitic in a PERSON /
    ORG span ("Trump's" instead of "Trump"). Normalising here keeps
    the aggregation counter from treating "Trump" and "Trump's" as
    separate entities on the graph.
    """
    out = text.strip()
    # Common possessives: "'s", "'s" (Unicode right single quote),
    # "s'" (already-plural possessive — rare in news, handle if seen).
    for suffix in ("'s", "\u2019s", "'S", "\u2019S"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
            break
    return out.strip("., ;:")


def extract_entities(text: str) -> list[str]:
    """Return PROPER-NOUN entity labels in ``text``.

    Order-preserving, duplicates kept (the caller aggregates across
    the corpus). Empty list if spaCy isn't available or nothing
    matched.
    """
    if not text:
        return []
    nlp = _load_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    out: list[str] = []
    for ent in doc.ents:
        if ent.label_ not in _KEEP_ENTITY_TYPES:
            continue
        label = _clean_label(ent.text)
        if len(label) < _MIN_LABEL_LEN:
            continue
        out.append(label)
    return out


def rank_corpus_phrases(
    texts: Iterable[str],
    min_count: int | None = None,
    top_n: int | None = None,
) -> list[tuple[str, int]]:
    """Aggregate entity frequencies across ``texts``.

    Uses ``nlp.pipe`` so spaCy can batch tokenization + NER across
    the whole corpus — roughly a 3-5x speedup vs per-text calls on
    our ~800-title graph corpus. Case-insensitive dedup: first-seen
    display form wins, counts sum across casings.

    Drops labels appearing fewer than ``min_count`` times. Caps
    return length at ``top_n``.
    """
    # Fall back to tuning defaults when the caller doesn't pin values.
    # Lets integrators call rank_corpus_phrases(texts) and have the
    # yaml knobs take over naturally.
    if min_count is None:
        min_count = _DEFAULT_MIN_OCCURRENCES
    if top_n is None:
        top_n = _DEFAULT_TOP_N
    clean = [t for t in (texts or []) if t]
    if not clean:
        return []
    nlp = _load_nlp()
    if nlp is None:
        return []
    counter: Counter[str] = Counter()
    display: dict[str, str] = {}
    # batch_size=64 fits comfortably in Jetson RAM with the sm model;
    # can bump if profile shows GC pressure on larger corpora.
    for doc in nlp.pipe(clean, batch_size=64):
        for ent in doc.ents:
            if ent.label_ not in _KEEP_ENTITY_TYPES:
                continue
            label = _clean_label(ent.text)
            if len(label) < _MIN_LABEL_LEN:
                continue
            key = label.lower()
            counter[key] += 1
            display.setdefault(key, label)
    pairs = [(display[k], c) for k, c in counter.items() if c >= min_count]
    pairs.sort(key=lambda p: (-p[1], p[0].lower()))
    return pairs[:top_n]
