"""Tool dispatcher: registry, arg filtering, and execute()."""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable

log = logging.getLogger("bridge.tools")


# Built lazily on first execute() call so domain modules don't import
# at package-init time. Mirrors the lazy-import pattern in bridge.llm.
_DISPATCH: dict[str, Callable[..., str]] | None = None


def _build_dispatch() -> dict[str, Callable[..., str]]:
    from bridge.tools._time import _get_time
    from bridge.tools._weather import _get_weather
    from bridge.tools._news import _get_news, _get_alerts, _get_digest
    from bridge.tools._tracker import _tracker
    from bridge.tools._reminder import _schedule_reminder
    from bridge.tools._math import _math
    from bridge.tools._convert import _convert
    from bridge.tools._graph import _graph
    from bridge.tools._circuit import _circuit
    from bridge.tools._wiki import _wiki, _wiki_search
    from bridge.tools._web import _web_search, _find_places
    from bridge.tools._memory import _update_memory
    from bridge.tools._say import _say
    from bridge.tools._inflation import _inflation
    from bridge.tools._population import _population
    from bridge.tools._facts import _facts
    from bridge.tools._analyze import _analyze
    from bridge.tools._alice import _alice
    return {
        "get_time": _get_time,
        "get_weather": _get_weather,
        "get_news": _get_news,
        "get_alerts": _get_alerts,
        "get_digest": _get_digest,
        "tracker": _tracker,
        "schedule_reminder": _schedule_reminder,
        "math": _math,
        "convert": _convert,
        "graph": _graph,
        "circuit": _circuit,
        "wiki": _wiki,
        "web_search": _web_search,
        "find_places": _find_places,
        "update_memory": _update_memory,
        "say": _say,
        "inflation": _inflation,
        "population": _population,
        "facts": _facts,
        "analyze": _analyze,
        "alice": _alice,
        # Legacy — replay compat only (not in TOOL_SCHEMAS).
        "wiki_search": _wiki_search,
    }


def _get_dispatch() -> dict[str, Callable[..., str]]:
    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = _build_dispatch()
    return _DISPATCH


# ---- Dispatcher ------------------------------------------------------------

def _filter_kwargs(fn: Callable[..., Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop kwargs the function's signature doesn't accept.

    Small LLMs often hallucinate plausible-sounding args that aren't in
    the tool schema (e.g. passing ``location`` to ``get_alerts``).
    Rather than raising a TypeError that leaks internals to the model
    and encourages it to parrot Python debug messages, we silently
    drop unknown kwargs and log the drop server-side. If the function
    takes ``**kwargs``, everything passes through untouched.

    Returns ``(filtered_args, dropped_names)``. Caller decides what to
    do with the dropped list (we just log it).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return arguments, []
    params = sig.parameters
    # If the function accepts **kwargs, every name is valid.
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return arguments, []
    accepted = {
        name for name, p in params.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    filtered = {k: v for k, v in arguments.items() if k in accepted}
    dropped = [k for k in arguments if k not in accepted]
    return filtered, dropped


def _friendly_error(name: str, exc: BaseException) -> str:
    """Translate a raw exception into a short string safe for the LLM."""
    msg = str(exc).strip()
    low = msg.lower()
    if any(k in low for k in ("timeout", "timed out", "connection", "network", "unreachable")):
        return f"[{name} temporarily unavailable — try again shortly]"
    if any(k in low for k in ("rate limit", "429", "quota")):
        return f"[{name} rate-limited — try again in a minute]"
    if "missing" in low and "argument" in low:
        return f"[{name} needs more info to run — ask the user for the specifics]"
    short = msg.split("\n")[0][:160]
    return f"[{name} couldn't complete: {short}]"


def _unwrap_nested_args(arguments: dict[str, Any], fn: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Unwrap nested ``args`` dicts that some models emit for tool calls.

    Llama 3.1 under Ollama occasionally produces tool-call arguments
    nested one level deep, in variations like:

      a) ``{"function": "get_weather", "args": {"location": "Tokyo"}}``
      b) ``{"op": "math", "args": {"op": "evaluate", "expression": "1+1"}}``
      c) ``{"name": "X", "arguments": {...}}``
    """
    if not isinstance(arguments, dict):
        return arguments
    inner = None
    if isinstance(arguments.get("args"), dict):
        inner = arguments["args"]
    elif isinstance(arguments.get("arguments"), dict):
        inner = arguments["arguments"]
    if inner is None:
        return arguments
    if fn is None:
        keys = set(arguments.keys())
        if {"function", "args"}.issubset(keys) and \
                not (keys - {"function", "args", "name"}):
            return arguments["args"]
        return arguments
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return arguments
    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in sig.parameters.values()):
        return arguments
    accepted = {
        n for n, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }
    outer_hits = sum(1 for k in arguments if k in accepted)
    inner_hits = sum(1 for k in inner if k in accepted)
    if inner_hits > outer_hits:
        return inner
    return arguments


def active_tool_schemas() -> list[dict]:
    """Return TOOL_SCHEMAS with features-registry gating applied."""
    from bridge.tools._schemas import TOOL_SCHEMAS
    try:
        from bridge import features
        llm_reminder_tool = features.is_enabled(
            "reminders_llm_tool", default=False,
        )
        out: list[dict] = []
        for s in TOOL_SCHEMAS:
            name = (s.get("function") or {}).get("name", "")
            if not features.tool_enabled(name):
                continue
            if name == "schedule_reminder" and not llm_reminder_tool:
                continue
            out.append(s)
        return out
    except Exception as e:
        log.warning("active_tool_schemas: features filter failed: %s", e)
        return list(TOOL_SCHEMAS)


def execute(name: str, arguments: dict[str, Any] | str) -> str:
    """Execute a tool by name with the given arguments, return a string result."""
    try:
        from bridge import features
        if not features.tool_enabled(name):
            log.info("tool %s is disabled via features.yaml", name)
            return f"[{name} is disabled on this server — answer without it]"
    except Exception as e:
        log.debug("features denylist check failed (non-fatal): %s", e)

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return f"[{name} got malformed arguments — retrying won't help, answer without the tool]"
    if not isinstance(arguments, dict):
        arguments = {}

    dispatch = _get_dispatch()
    fn = dispatch.get(name)
    if fn is None:
        return f"[unknown tool '{name}' — answer without it]"

    arguments = _unwrap_nested_args(arguments, fn)
    arguments, dropped = _filter_kwargs(fn, arguments)
    if dropped:
        log.info("tool call: %s dropped unknown args: %s", name, dropped)

    log.info("tool call: %s(%s)", name, arguments)
    try:
        result = fn(**arguments)
    except Exception as e:
        log.error("tool %s raised: %r", name, e)
        result = _friendly_error(name, e)

    log.info("tool %s result: %s", name, (result or "")[:200])
    return result
