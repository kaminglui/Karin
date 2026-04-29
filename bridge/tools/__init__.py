"""Tools Karin can call during a chat turn.

Subpackage layout — see each module's docstring for details.
All public names are lazily importable from ``bridge.tools``.
"""
from __future__ import annotations

import importlib

# --- Eagerly available (no heavy deps) ---
from bridge.tools._schemas import TOOL_SCHEMAS
from bridge.tools._dispatch import (
    execute,
    active_tool_schemas,
    _filter_kwargs,
    _friendly_error,
    _unwrap_nested_args,
)

# Eagerly bind the entrypoint functions for tools whose function name
# collides with the submodule name. Without this, `import bridge.tools`
# would later resolve `bridge.tools._wiki` to the SUBMODULE (because
# Python auto-binds submodules onto their parent package on first
# import) and `tools._wiki(...)` would TypeError. By eagerly importing
# and binding the function, we win the attribute lookup.
from bridge.tools._wiki import _wiki  # noqa: F401
from bridge.tools._tracker import _tracker  # noqa: F401


# --- Lazy imports for names that pull in httpx/sympy/pint/numpy ---
# Consumers that do `from bridge.tools import fetch_weather` get the
# real function on first access without forcing httpx to load at
# package-init time. This keeps `from bridge.tools import TOOL_SCHEMAS`
# fast and side-effect-free.
#
# `_LAZY_NAME_TO_MODULE` maps each lazily-resolved name to the submodule
# that owns it. ``__getattr__`` resolves via ``importlib.import_module``
# so the call doesn't recurse through ``__getattr__`` on Python 3.12+
# (where ``from bridge.tools import _foo`` triggers attribute lookup
# on the package itself).

_LAZY_NAME_TO_MODULE: dict[str, str] = {
    # Weather
    "fetch_weather": "bridge.tools._weather",
    "weather_emoji": "bridge.tools._weather",
    "_get_weather": "bridge.tools._weather",
    "_WEATHER_CACHE": "bridge.tools._weather",
    "_WEATHER_CODES": "bridge.tools._weather",
    "_WEATHER_EMOJI": "bridge.tools._weather",
    "_is_vague_location": "bridge.tools._weather",
    "_looks_like_coords": "bridge.tools._weather",
    # Math / convert / graph / circuit / web (data-shape exports)
    "math_data": "bridge.tools._math",
    "convert_data": "bridge.tools._convert",
    "graph_data": "bridge.tools._graph",
    "circuit_data": "bridge.tools._circuit",
    "find_places_data": "bridge.tools._web",
    # Wiki
    "_wiki": "bridge.tools._wiki",
    "_wiki_search": "bridge.tools._wiki",
    "_wiki_random": "bridge.tools._wiki",
    # Tracker
    "_tracker": "bridge.tools._tracker",
    "_get_tracker": "bridge.tools._tracker",
    "_get_trackers": "bridge.tools._tracker",
    "_resolve_tracker_alias": "bridge.tools._tracker",
    # Reminder
    "_schedule_reminder": "bridge.tools._reminder",
    # Retired (legacy names — kept for replay compatibility + tests)
    "_fill_format": "bridge.tools._retired",
    "_format_skeleton": "bridge.tools._retired",
}


def __getattr__(name: str):
    mod_path = _LAZY_NAME_TO_MODULE.get(name)
    if mod_path is not None:
        mod = importlib.import_module(mod_path)
        obj = getattr(mod, name)
        # Pin the resolved object on this package's namespace so the
        # next attribute lookup skips __getattr__ and finds the cached
        # value directly.
        import sys
        sys.modules[__name__].__dict__[name] = obj
        return obj
    # Dispatch internals — tests `monkeypatch.setitem(tools._DISPATCH, ...)`
    # to register fake tools, which requires the returned dict to be the
    # SAME object the dispatcher uses. _get_dispatch() memoizes a single
    # dict, so every call hands back the live one.
    if name == "_DISPATCH":
        from bridge.tools._dispatch import _get_dispatch
        return _get_dispatch()
    # httpx — re-exported for tests that `patch.object(tools.httpx, ...)`.
    # Stable across calls because Python caches the import in sys.modules.
    if name == "httpx":
        return importlib.import_module("httpx")
    raise AttributeError(f"module 'bridge.tools' has no attribute {name!r}")


__all__ = [
    "TOOL_SCHEMAS",
    "execute",
    "active_tool_schemas",
    "fetch_weather",
    "weather_emoji",
    "math_data",
    "convert_data",
    "graph_data",
    "circuit_data",
    "find_places_data",
]
