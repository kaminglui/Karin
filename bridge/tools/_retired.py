"""Retired tools kept for conversation replay compatibility."""
from __future__ import annotations

import logging
import re

import httpx

log = logging.getLogger("bridge.tools")


def _unfold_abbreviation(abbreviation: str) -> str:
    """Look up common meanings of an abbreviation via Wikipedia.

    Parses Wikipedia's disambiguation page for the term and returns a
    flat list of alternative meanings. Lets the LLM choose an interesting
    interpretation for ambiguous prompts (e.g. a creative request about
    "PA" could mean Pennsylvania, Physician Assistant, Public Address,
    etc.) instead of defaulting to one fixed sense.
    """
    abbr = str(abbreviation or "").strip()
    if not abbr:
        return "Error: no abbreviation given."
    if len(abbr) > 15:
        return "Error: too long to be an abbreviation."

    headers = {
        "User-Agent": (
            "karin/0.1 (https://github.com/kaminglui/Karin; "
            "personal voice assistant) httpx"
        ),
        "Accept": "application/json",
    }
    # "(disambiguation)" suffix hits the dedicated disambig page when a
    # term also has a dominant article (e.g. "PA" → Pennsylvania). Plain
    # title is the fallback for pure disambiguation terms.
    candidates = [f"{abbr} (disambiguation)", abbr]
    try:
        with httpx.Client(timeout=8.0, headers=headers) as client:
            for title in candidates:
                resp = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": 1,
                        "exlimit": 1,
                        "titles": title,
                        "format": "json",
                        "redirects": 1,
                    },
                )
                if resp.status_code != 200:
                    continue
                pages = (resp.json().get("query") or {}).get("pages") or {}
                for _pid, page in pages.items():
                    if "missing" in page:
                        continue
                    extract = (page.get("extract") or "").strip()
                    if not extract:
                        continue
                    low = extract.lower()
                    is_disambig = any(m in low for m in _DISAMBIG_MARKERS)
                    if not is_disambig and title == abbr:
                        # Dominant article — keep looking for the
                        # disambig page sibling.
                        continue
                    # Parse the extract by section so we can round-robin
                    # across topic areas. Wikipedia disambiguation pages
                    # group by category (Arts, Government, Places, Other
                    # uses, ...) and taking only the first N entries
                    # misses common senses that live late in the list —
                    # e.g. "Pennsylvania" lands around entry 46 on PA's
                    # disambig page because "Places" comes after the
                    # arts/gov/airline sections.
                    lines = [l.strip() for l in extract.splitlines() if l.strip()]
                    sections: list[tuple[str, list[str]]] = [("Top", [])]
                    for line in lines:
                        if line.startswith("=") and line.endswith("="):
                            name = line.strip("=").strip() or "Other"
                            sections.append((name, []))
                            continue
                        if any(m in line.lower() for m in _DISAMBIG_MARKERS):
                            continue
                        if len(line) < 8 or len(line) > 180:
                            continue
                        sections[-1][1].append(line)

                    meanings: list[str] = []
                    # Round-robin: take one entry from each section in
                    # sequence, loop until we hit the budget. Gives a
                    # diverse mix (one or two from each topic) instead of
                    # 30 entries from just "Arts and media".
                    depth = 0
                    while len(meanings) < 15 and depth < 6:
                        added_any = False
                        for _name, entries in sections:
                            if depth < len(entries):
                                meanings.append(entries[depth])
                                added_any = True
                                if len(meanings) >= 15:
                                    break
                        if not added_any:
                            break
                        depth += 1

                    if meanings:
                        bullets = "\n".join(f"- {m}" for m in meanings)
                        return f"Common meanings of '{abbr}':\n{bullets}"
            return f"No disambiguation entry found for '{abbr}'."
    except httpx.HTTPError as e:
        return f"Error looking up '{abbr}': {e}"
    except Exception as e:
        return f"Error: {e}"


_FORMAT_SLOT_RE = re.compile(r"\{([^{}]+)\}")


def _format_skeleton(template: str) -> str:
    """The non-slot 'scaffolding' of a template: everything outside
    every ``{...}`` block. Used by the post-fill validator to check
    the model preserved the surrounding punctuation + static words.
    Whitespace is collapsed so trivial indentation drift doesn't
    cause a false negative.
    """
    skeleton = _FORMAT_SLOT_RE.sub("\x00SLOT\x00", template)
    return " ".join(skeleton.split())


def _fill_format(
    template: str,
    topic: str | None = None,
    style: str | None = None,
) -> str:
    """Pass-through directive telling the LLM to fill a template.

    Same pattern as :func:`_tell_story` — the tool itself does no
    external work; it returns instructions the LLM reads on its next
    pass. Slot descriptions in the template are natural language
    (e.g. ``{one mood word}``, ``{short phrase about rain}``) — the
    LLM reads them like a human would. The validator in
    :mod:`bridge.llm`'s post-run hook (future) would confirm the
    skeleton survived the fill. For now we just record the skeleton
    in the directive so the model is reminded to preserve it.

    NOT included here: syllable/meter/rhyme validation. That's the
    lyrics-engine extension — would need phonetic tools (pronouncing /
    CMU dict) and an iterative generate-validate-regenerate loop.
    """
    tmpl = (template or "").strip()
    if not tmpl:
        return "Error: no template provided."
    slots = _FORMAT_SLOT_RE.findall(tmpl)
    if not slots:
        return (
            "Error: template has no {slot} placeholders — nothing "
            "to fill. Ask the user for a format with {...} slots."
        )

    skeleton = _format_skeleton(tmpl)
    hints: list[str] = []
    if topic:
        hints.append(f"Topic: {topic.strip()}.")
    if style:
        hints.append(f"Style / tone: {style.strip()}.")
    hint_line = "  ".join(hints) if hints else "(no additional hints)"

    return (
        "Fill the user's template below. Rules:\n"
        "1. PRESERVE the template's structure EXACTLY — same "
        "punctuation, same non-slot words, same line breaks.\n"
        "2. Replace each {slot description} with an appropriate "
        "filler that matches the description. The slot text is a "
        "natural-language hint; read it like a human would.\n"
        "3. Output ONLY the filled template. No preamble, no "
        "commentary, no quotation marks around the result.\n\n"
        f"Template:\n{tmpl}\n\n"
        f"Slots to fill ({len(slots)}): "
        + ", ".join(f'"{{{s}}}"' for s in slots)
        + "\n\n"
        f"Context — {hint_line}"
    )


_ABBR_RE = re.compile(r"\b([A-Z](?:\.?[A-Z]){1,4}\.?)\b")


def _detect_abbreviation(text: str) -> str | None:
    """Return the first 2-5 letter acronym found in ``text``, or None.

    Handles plain (``PA``), dotted (``P.A.``), and mixed-case-with-caps
    forms. Skips very common English words that happen to look acronym-
    ish (``OK``, ``ID``, ``USA`` is fine to keep, ``A`` alone won't match).
    """
    if not text:
        return None
    for m in _ABBR_RE.findall(text):
        clean = m.replace(".", "").strip()
        if 2 <= len(clean) <= 5 and clean.isupper():
            # Common false positives — skip so we don't launch Wikipedia
            # disambig lookups for non-acronym all-caps words.
            if clean in {"OK", "NO", "YES", "HELLO", "HI"}:
                continue
            return clean
    return None


def _tell_story(topic: str | None = None, kind: str | None = None) -> str:
    """Pass-through "permission slip" for creative/opinion requests.

    Smaller models (2B-4B) tend to shoehorn ambiguous narrative requests
    into wiki_search / wiki_random because those are the only tools that
    look vaguely related. Exposing this explicit routing target lets the
    model pick it, gets a directive saying "yep, compose it yourself
    from your own knowledge", and then produces the actual reply on
    its next turn without re-querying an external API.

    If the topic contains a short acronym (e.g. "PA", "NY", "AI"), we
    also run Wikipedia disambiguation server-side and inject the top
    meanings into the directive. That way the model gets abbreviation
    unfolding for free on its single tool call, without needing a
    second tool round-trip it often doesn't know how to chain.

    Args:
        topic: What the reply should be about (the user's phrasing).
        kind: "story" / "joke" / "explanation" / "opinion" / "anecdote".

    Returns a short directive the LLM will then paraphrase through its
    own persona — no external side effect.
    """
    hints: list[str] = []
    if topic:
        hints.append(f"Topic: {topic.strip()}.")
    if kind:
        hints.append(f"Style: {kind.strip()}.")

    # Inline abbreviation unfold: if the topic mentions a short acronym,
    # look it up and hand the meanings to the LLM as part of the directive.
    meanings_block = ""
    if topic:
        abbr = _detect_abbreviation(topic)
        if abbr:
            unfold = _unfold_abbreviation(abbr)
            if unfold and not unfold.startswith("Error:") and not unfold.startswith("No "):
                meanings_block = (
                    f"\n\nThe topic mentions the abbreviation '{abbr}'. "
                    f"Here are common meanings — pick ONE (preferably one "
                    f"that makes a fun story/joke/opinion in context), "
                    f"acknowledge the chosen sense in your reply, and "
                    f"compose around it:\n{unfold}"
                )

    suffix = " " + " ".join(hints) if hints else ""
    return (
        "Compose the reply yourself from what you know — no external "
        "lookup is needed or wanted. Keep it to ONE short sentence in "
        "Karin's casual voice. For a 'real story' request, pick a "
        "well-known true event or anecdote; for a joke or opinion, "
        "just answer directly." + suffix + meanings_block
    )

