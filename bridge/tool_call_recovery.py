"""Leak recovery + tool-output sanitization (L5 in the routing pipeline).

Two concerns:

1. **Leak recovery.** The LoRA sometimes serializes a tool-call into the
   ``content`` field instead of populating the structured ``tool_calls``
   field the Ollama API expects. ``recover_leaked_tool_call`` detects
   the JSON-shaped content and synthesizes the structured tool_call the
   chat loop would have received natively. Without this, observed leaks
   would be rendered as raw JSON to the user instead of executing the
   intended tool.

2. **Tool-output sanitization.** External sources (web_search, wiki,
   news fetch) can return adversarial text designed to hijack the LLM
   ("ignore previous instructions and …"). ``sanitize_tool_result``
   strips known prompt-injection preambles before the result is fed
   back into the chat context.

Both routines were factored out of ``bridge.llm.OllamaLLM`` so the
chat loop reads more like a state machine and so this regex catalog
has a single home — when a new injection shape shows up, the fix
lives here, not in the loop.
"""
from __future__ import annotations

import json as _json
import logging
import re
from typing import Callable, Iterable

log = logging.getLogger("bridge.llm")


# --- prompt-injection sanitization ----------------------------------------

INJECTION_PATTERNS = re.compile(
    r"(?i)"
    r"(ignore\s+(all\s+)?previous\s+instructions"
    r"|you\s+are\s+now\s+a\b"
    r"|system\s*:\s*you\s+are"
    r"|new\s+instructions?\s*:"
    r"|override\s+(system|prompt)"
    r"|disregard\s+(all|previous|above)"
    r"|forget\s+(all|everything|your)\s+(instructions|rules|prompt)"
    r"|pretend\s+you\s+are"
    r"|act\s+as\s+if\s+you\s+are"
    r"|do\s+not\s+follow\s+(your|the)\s+(rules|instructions|prompt))"
)


def sanitize_tool_result(result: str) -> str:
    """Strip prompt-injection payloads from tool output.

    External sources (web_search, wiki) can return adversarial text
    designed to hijack the LLM's behavior. Replace known injection
    preambles with ``[REDACTED]`` and cap the total length so a
    pathologically large fetch doesn't blow the LLM's context budget.
    The system prompt instructs the LoRA to treat tool output as
    *data*, not instructions; this is the runtime safety net for the
    cases where it doesn't.
    """
    if not isinstance(result, str):
        return str(result) if result is not None else ""
    cleaned = INJECTION_PATTERNS.sub("[REDACTED]", result)
    if len(cleaned) > 4000:
        cleaned = cleaned[:4000] + "\n[... truncated]"
    return cleaned


# --- leaked-tool-call recovery --------------------------------------------

_BARE_NAME_FIX_RE = re.compile(
    r'("(?:name|function)"\s*:\s*)([A-Za-z_][A-Za-z0-9_]*)',
)


def recover_leaked_tool_call(
    content: str,
    tools: list | None,
    *,
    data_fetch_tools: Iterable[str],
    looks_like_chitchat: Callable[[str | None], bool],
    user_text: str | None = None,
) -> dict | None:
    """Recover a tool-call that the model serialized into ``content``.

    Llama 3.1 (and some other tunes) occasionally emit the tool-call
    JSON as user-facing ``content`` instead of filling the structured
    ``tool_calls`` field. Two shapes we've seen:

      A) Raw args only::
         ``{"kind": "story", "topic": "yourself"}``
      B) Named wrapper::
         ``{"name": "get_digest", "params": {}}`` or
         ``{"function": "get_news", "arguments": {...}}``

    Shape A: we look up which tool accepts exactly these kwarg names
    and pick the narrowest match.

    Shape B: we pull the tool name directly from ``name``/``function``
    and the args from ``params``/``arguments``/``args`` — useful for
    tools with NO arg schema (like ``get_digest``), where shape A
    can't match anything.

    Args:
        content: The assistant message's ``content`` field.
        tools: Tool schemas offered this turn. Used to resolve the
            tool name in both shapes.
        data_fetch_tools: Names of "data fetch" tools (weather, news,
            etc.). Recovery is refused on these when ``user_text``
            looks like chitchat — Rule Zero in the pipeline.
        looks_like_chitchat: Predicate over ``user_text``. Passed in
            so this module doesn't need to import the OllamaLLM heuristic.
        user_text: The user's raw prompt. Used by the chitchat guard.

    Returns a synthesized tool_call dict (Ollama-format) or None if no
    recovery applies.
    """
    if not content or not tools:
        return None
    stripped = content.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed = _json.loads(stripped)
    except (_json.JSONDecodeError, ValueError):
        # The LoRA sometimes emits malformed JSON with unquoted bare
        # identifiers (``{"name": wiki, ...}``). Normalize and retry
        # before giving up.
        fixed = _BARE_NAME_FIX_RE.sub(r'\1"\2"', stripped)
        try:
            parsed = _json.loads(fixed)
        except (_json.JSONDecodeError, ValueError):
            return None
    if not isinstance(parsed, dict) or not parsed:
        return None

    known_names = {
        (s.get("function") or {}).get("name", "")
        for s in tools
    }

    # Shape B: explicit {"name": "<tool>", ...} or
    # {"function": "<tool>", ...}. Accept either, and treat
    # ``params`` / ``arguments`` / ``args`` as the args payload.
    named = (
        parsed.get("name")
        if isinstance(parsed.get("name"), str)
        else parsed.get("function")
        if isinstance(parsed.get("function"), str)
        else None
    )
    if named and named in known_names:
        # Rule Zero guard: never auto-recover a data-fetch tool
        # call when the user's prompt is clearly chitchat. The
        # persona forbids this case; the LLM leaking a JSON
        # placeholder shouldn't override that.
        data_fetch = frozenset(data_fetch_tools)
        if named in data_fetch and looks_like_chitchat(user_text):
            log.info(
                "refused leak-recovery: %s on chitchat prompt %r",
                named, (user_text or "")[:40],
            )
            return None
        args_payload = (
            parsed.get("parameters")
            if isinstance(parsed.get("parameters"), dict)
            else parsed.get("params")
            if isinstance(parsed.get("params"), dict)
            else parsed.get("arguments")
            if isinstance(parsed.get("arguments"), dict)
            else parsed.get("args")
            if isinstance(parsed.get("args"), dict)
            else {}
        )
        log.info("recovered leaked tool_call: %s(%s)", named, args_payload)
        return {"id": "leak_0", "type": "function", "function": {"name": named, "arguments": args_payload}}

    # Shape A: bare args dict → match via narrowest-schema rule.
    leaked_keys = set(parsed.keys())
    best: tuple[str, int] | None = None
    for schema in tools:
        fn = (schema or {}).get("function") or {}
        params = (fn.get("parameters") or {}).get("properties") or {}
        allowed = set(params.keys())
        if not allowed or not leaked_keys.issubset(allowed):
            continue
        slack = len(allowed) - len(leaked_keys)
        if best is None or slack < best[1]:
            best = (fn.get("name", ""), slack)
    if best is None:
        return None
    name = best[0]
    log.info("recovered leaked tool_call: %s(%s)", name, parsed)
    return {"id": "leak_0", "type": "function", "function": {"name": name, "arguments": parsed}}
