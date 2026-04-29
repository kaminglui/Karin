"""spaCy-based structural filter for the regex pre-classifier.

Runs *after* the regex classifier picks a tool. Its job is to veto
that pick when spaCy's dependency parse reveals a structural signal
the regex can't see — specifically **negation on the tool's trigger
verb**.

Example failure the pure-regex pipeline gave us:

    "Don't remind me, I already know."

Regex matched ``\\bremind\\s+me\\b`` and suggested ``schedule_reminder``.
spaCy parses the same prompt and finds ``neg -> remind`` (i.e. the
word "not" attached to the root verb "remind"), which means the user
is declining a reminder, not asking for one. The filter returns True
→ the hint is suppressed → the LLM reads no hint and answers in prose.

Design notes:

- **Complementary to regex**, not a replacement. Regex is microseconds
  and catches 80%+ of cases. spaCy runs only when regex already has a
  candidate.
- **Graceful degradation**: if spaCy fails to load (missing model,
  import error), the filter silently becomes a no-op. The routing
  pipeline still works.
- **Lazy load** on first call — pays the ~200 ms spaCy import cost
  only if anyone routes through the filter. Subsequent calls use the
  cached model.
- **Parser + tagger only** — we disable NER/textcat components since
  we don't use them (saves a bit of latency and memory).

See ``bridge/routing/classifier.py::routing_hint`` for the integration
point.
"""
from __future__ import annotations

import logging

log = logging.getLogger("bridge.routing.spacy_filter")

# Lazy-loaded spaCy pipeline. ``None`` = not tried yet; ``False`` =
# tried and failed (don't retry); a Language object = loaded.
_nlp: object | None = None


# Tool-name -> set of trigger lemmas whose direct negation should veto
# the classifier's pick. Matches the regex patterns in _schemas.py: if
# a pattern relies on one of these lemmas (e.g. schedule_reminder's
# ``remind me`` pattern relies on ``remind``), the filter needs that
# lemma here so the spaCy check can cancel the hint when the verb is
# negated ("don't remind me").
_TRIGGER_LEMMAS: dict[str, set[str]] = {
    "schedule_reminder": {"remind", "alarm", "wake", "nudge", "set"},
    "update_memory": {"remember", "call"},
    "get_weather": {"worry"},        # "don't worry about the weather"
    "get_news": {"tell"},            # "don't tell me the news"
    "find_places": {"find", "buy"},  # "don't find..."
}


def _load_nlp():
    """Return a loaded spaCy Language object, or False if unavailable."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        # NER is the only component we don't need for negation dep
        # detection (we want parser + tagger + lemmatizer). Keeping
        # attribute_ruler and tok2vec since the lemmatizer depends on
        # them in en_core_web_sm's pipeline.
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        log.info("spaCy filter: en_core_web_sm loaded (NER disabled)")
    except Exception as e:
        log.warning("spaCy filter disabled — %s", e)
        _nlp = False
    return _nlp


def should_veto(prompt: str, suggested_tool: str) -> bool:
    """Return True if a structural signal contradicts routing to
    ``suggested_tool``.

    Current check: **direct negation on a trigger lemma**. For example,
    with suggested_tool='schedule_reminder', a prompt like
    ``"Don't remind me"`` has ``neg -> remind`` in its dep parse, so
    we return True. A prompt like ``"Don't let me forget X"`` has
    ``neg -> let`` (not a schedule_reminder trigger) so we return
    False — the affirmative reminder request stays intact.

    Returns False (no veto) on any parse error / disabled filter, so
    the caller's behavior degrades gracefully to pure regex.
    """
    if not prompt or not suggested_tool:
        return False
    triggers = _TRIGGER_LEMMAS.get(suggested_tool)
    if not triggers:
        return False
    nlp = _load_nlp()
    if not nlp:
        return False
    try:
        doc = nlp(prompt)
    except Exception as e:
        log.debug("spaCy parse failed on %r: %s", prompt[:40], e)
        return False

    # Walk tokens looking for a negation dependency whose head lemma
    # is one of the suggested tool's trigger lemmas.
    for tok in doc:
        if tok.dep_ == "neg":
            head = tok.head
            head_lemma = (head.lemma_ or head.text or "").lower()
            if head_lemma in triggers:
                log.info(
                    "spaCy veto: prompt=%r  neg->%s(lemma=%s)  "
                    "suggested=%s  suppressing hint",
                    prompt[:70], head.text, head_lemma, suggested_tool,
                )
                return True
    return False
