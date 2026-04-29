"""Shared helpers for the data-fetch scripts (BLS / World Bank / NBER).

Currently provides:

- ``ShapeError``: distinct exception for upstream-API schema drift, so
  callers can present a clear "upstream changed" error instead of a
  generic KeyError/TypeError dump.
- ``check_shape(data, expected, path="")``: walk a payload and a sentinel
  describing the expected shape; raise ``ShapeError`` when the live
  payload deviates. Designed to be cheap, called once per response.
- ``write_status(tool, region, ok, error="", **fields)``: append to a
  per-tool ``_last_fetch_status.json`` so the web diagnostics panel
  (future) can read it.
- ``resolve_api_key(name, *, cli_value, env_var)``: CLI flag → env var →
  active profile preferences. Used by the county-overlay populators so
  Settings-UI-saved keys reach scripts without an env-var export step.
- ``http_request(url, *, data=None, headers=None, timeout=60, retries=3)``:
  resilient HTTP GET/POST with retry-on-transient + identifying
  User-Agent. Replaces the bare ``urllib.request.urlopen`` pattern that
  every fetcher used to repeat (no UA header, no retry, no consistent
  timeout). Returns parsed JSON.

Per docs/data-fetch-resilience.md — Tier A protection.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# User-Agent string sent on every fetch. Identifying the client is
# basic web politeness — APIs that 403 generic urllib UAs (some BLS
# proxies, Wikipedia, Census) accept this. The github-noreply form
# avoids leaking a personal email.
_USER_AGENT = (
    "Karin/0.1 (https://github.com/kaminglui/Karin; "
    "kaminglui+karin@users.noreply.github.com) urllib"
)

# HTTP status codes worth retrying. 429 = rate-limited; 5xx =
# transient server-side. 4xx other than 429 (auth, malformed query)
# are NOT retryable — caller fixes the request, doesn't pound harder.
_RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})


class ShapeError(RuntimeError):
    """Raised when an upstream JSON payload deviates from the expected
    shape. Callers should surface this without retrying — re-fetching
    won't fix a schema change."""


def check_shape(data: Any, expected: Any, path: str = "$") -> None:
    """Validate ``data`` against an ``expected`` sentinel structure.

    Sentinels:
      * a Python type (``list``, ``dict``, ``str``, ``int``, ``float``,
        ``bool``) — the live value must be an instance of that type.
      * a ``list`` of length 1 — the live value must be a list, and the
        FIRST element of the live list must match the sentinel's
        single element. (Subsequent elements are not checked — they
        are assumed homogeneous; this saves time on large arrays.)
      * a ``dict`` — the live value must be a dict, and EVERY key in
        the sentinel must be present in the live dict with a matching
        recursive shape. Live dicts may have EXTRA keys (forward-
        compatible).
      * the literal value ``None`` — match anything (escape hatch for
        fields whose internal shape isn't part of the contract).

    Raises ``ShapeError`` with a clear path on first mismatch.
    """
    if expected is None:
        return
    if isinstance(expected, type):
        if not isinstance(data, expected):
            raise ShapeError(
                f"{path}: expected {expected.__name__}, "
                f"got {type(data).__name__}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(data, list):
            raise ShapeError(
                f"{path}: expected list, got {type(data).__name__}"
            )
        if expected and data:
            check_shape(data[0], expected[0], f"{path}[0]")
        return
    if isinstance(expected, dict):
        if not isinstance(data, dict):
            raise ShapeError(
                f"{path}: expected dict, got {type(data).__name__}"
            )
        for k, v in expected.items():
            if k not in data:
                raise ShapeError(f"{path}: missing required key {k!r}")
            check_shape(data[k], v, f"{path}.{k}")
        return
    # Literal sentinel — must equal exactly.
    if data != expected:
        raise ShapeError(f"{path}: expected literal {expected!r}, got {data!r}")


def write_status(
    data_dir: Path,
    tool: str,
    key: str,
    *,
    ok: bool,
    error: str | None = None,
    **fields: Any,
) -> None:
    """Append/update a per-(tool, key) entry in
    ``<data_dir>/_last_fetch_status.json``.

    Format:

        {
          "<tool>:<key>": {
            "ok": true|false,
            "ran_at": "2026-04-27T10:30:00+00:00",
            "error": "...",
            ...extra fields...
          },
          ...
        }

    Future web UI reads this to surface stale/broken sources. Quiet on
    write failure — diagnostics shouldn't crash the populator itself.
    """
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "_last_fetch_status.json"
        existing: dict = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except json.JSONDecodeError:
                existing = {}
        entry = {
            "ok": bool(ok),
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if error:
            entry["error"] = str(error)[:500]
        entry.update(fields)
        existing[f"{tool}:{key}"] = entry
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    except Exception as e:  # pragma: no cover — diagnostics never crash
        # Last-resort: print to stderr so it shows in the populator's
        # output but doesn't propagate.
        import sys
        print(f"  [write_status warn] {e}", file=sys.stderr)


def http_request(
    url: str,
    *,
    data: bytes | dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
    method: str | None = None,
) -> Any:
    """GET or POST with User-Agent + retry on transient failures.

    ``data`` semantics:
      * ``None`` → GET request.
      * ``dict`` → POST request, body is ``json.dumps(data).encode()``,
        ``Content-Type: application/json`` is set automatically (caller
        can override via ``headers``).
      * ``bytes`` → POST with the body as-is; caller sets Content-Type.

    Retries up to ``retries`` times on HTTP 408/425/429/5xx and
    network-level URLError, with exponential backoff
    (``backoff * 2**attempt`` seconds — 2, 4, 8 by default). Other
    HTTP errors (4xx auth/malformed) raise immediately on the first
    attempt — pounding the API harder won't fix those.

    Returns the parsed JSON body on success. Raises ``RuntimeError``
    with a clear message on persistent failure.
    """
    base_headers = {"User-Agent": _USER_AGENT}
    if headers:
        base_headers.update(headers)
    body: bytes | None = None
    if isinstance(data, dict):
        body = json.dumps(data).encode("utf-8")
        base_headers.setdefault("Content-Type", "application/json")
    elif isinstance(data, (bytes, bytearray)):
        body = bytes(data)
    elif data is not None:
        raise TypeError(f"http_request data must be dict|bytes|None, got {type(data).__name__}")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body, headers=base_headers, method=method,
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:300]
            if e.code not in _RETRYABLE_HTTP:
                # Non-retryable: caller's request is bad. Fail fast with
                # the API's own error message attached.
                raise RuntimeError(
                    f"HTTP {e.code} at {url[:120]}: {err_body}"
                ) from e
            last_error = RuntimeError(
                f"HTTP {e.code} at {url[:120]}: {err_body}"
            )
        except urllib.error.URLError as e:
            last_error = RuntimeError(f"network error at {url[:120]}: {e}")
        except (json.JSONDecodeError, OSError) as e:
            # Malformed JSON or socket-level issue — treat as transient
            # but not retryable past one re-try.
            last_error = RuntimeError(f"transport error at {url[:120]}: {e}")
        # Last attempt failed — bail.
        if attempt == retries:
            break
        wait = backoff * (2 ** attempt)
        print(
            f"  retry {attempt + 1}/{retries} in {wait:.0f}s: {last_error}",
            file=sys.stderr,
        )
        time.sleep(wait)
    assert last_error is not None
    raise last_error


def resolve_api_key(
    name: str,
    *,
    cli_value: str | None,
    env_var: str,
    from_prefs: bool = False,
    log_to_stderr: bool = True,
) -> str:
    """Resolve an API key from CLI > env > profile preferences.

    Returns the resolved key (stripped) or empty string. Identical
    fallback chain across all county-overlay populators.

    The bridge import is deferred so the script still runs in a thin
    Python environment without the bridge venv (one-off populator
    boxes). On import failure, falls through to "" with a stderr note.
    """
    if cli_value and cli_value.strip():
        return cli_value.strip()
    env_val = (os.environ.get(env_var) or "").strip()
    if env_val:
        return env_val
    if not from_prefs:
        return ""
    try:
        # scripts/ is a sibling of bridge/, so add the repo root to
        # sys.path before importing.
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from bridge.profiles import get_profile_api_key
    except Exception as e:
        if log_to_stderr:
            print(
                f"  --from-prefs lookup failed ({e}); proceeding without",
                file=sys.stderr,
            )
        return ""
    val = (get_profile_api_key(name) or "").strip()
    if val and log_to_stderr:
        print(
            f"  using {name} from active profile preferences",
            file=sys.stderr,
        )
    return val


# Expected shapes for each known upstream API. Keep these minimal —
# only the keys we actually depend on. Adding too much makes false
# positives more likely as APIs evolve cosmetic fields.

BLS_API_V2_SHAPE = {
    "status": str,
    "Results": {
        "series": [{
            "seriesID": str,
            "data": [{
                "year": str,
                "period": str,
                "value": str,
            }],
        }],
    },
}

WORLDBANK_INDICATOR_SHAPE = [
    # Outer list: [meta, records]
    None,  # meta varies; don't pin
]
# Tighter shape for the records side (body[1]) when present.
WORLDBANK_RECORD_SHAPE = {
    "date": str,
    "value": None,  # may be float, int, or null
}
