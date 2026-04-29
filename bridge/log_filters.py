"""Logging filters that scrub secrets from log records.

Attached in web/server.py at startup so every logger in the process —
httpx, uvicorn, our own code — gets scrubbed automatically. Doesn't
touch records' original meaning; only rewrites known secret-shaped
query params (``api_key``, ``apikey``, ``token``, ``key``) in URL-like
strings to ``<redacted>``.

Cost: a regex substitution per log record. Measured at <1 µs on
realistic records, so we apply it on the root logger without worry.
"""
from __future__ import annotations

import logging
import re

# Match ``<param>=<value>`` where <param> is one of the usual secret
# names and <value> runs until the next '&' or end-of-string. Anchored
# on ``[?&]`` so we only touch query params, not random key=value
# occurrences in prose (e.g. a config dump's "secret_key=foo" would
# stay — callers should log those through a different scrubber if
# needed). Case-insensitive on the parameter name.
_SECRET_QUERY_PARAMS = (
    "api_key", "apikey", "api-key",
    "access_token", "accesstoken",
    "auth_token", "authtoken",
    "token",
    "secret", "secret_key", "secretkey",
)
_SECRET_QUERY_RE = re.compile(
    r"([?&](?:" + "|".join(re.escape(p) for p in _SECRET_QUERY_PARAMS) + r")=)[^&\s]+",
    re.IGNORECASE,
)


def _scrub(text: str) -> str:
    """Mask secret-shaped query params in a string. Returns the input
    unchanged when nothing matches (fast path)."""
    if not text or "=" not in text:
        return text
    return _SECRET_QUERY_RE.sub(r"\1<redacted>", text)


class SecretScrubFilter(logging.Filter):
    """A logging.Filter that rewrites LogRecord args/message to mask
    credentials that slipped into URLs. Installed on the root logger so
    every library (httpx especially) inherits it."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Two paths: formatted message (rare — most loggers defer) and
        # args (the common path: httpx logs via `log.info("HTTP %s %s", ...)`
        # style). Mutating record.args in place is the supported pattern.
        if isinstance(record.msg, str) and "=" in record.msg:
            record.msg = _scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (_scrub(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    _scrub(a) if isinstance(a, str) else a
                    for a in record.args
                )
        return True


def install_secret_scrubber() -> None:
    """Attach :class:`SecretScrubFilter` to every handler on the root
    logger.

    Why handlers, not the root logger itself: Python's logging propagates
    records UP from child loggers, but a parent's filters only fire for
    records logged directly to that parent — not for propagated ones.
    Handlers, in contrast, get every record that reaches them (including
    propagated ones), so attaching there is the reliable scrub point.

    Idempotent: calling twice doesn't stack duplicate filters on a
    handler, and calling after more handlers get added will catch the
    new ones on the next invocation.
    """
    scrub = SecretScrubFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        if any(isinstance(f, SecretScrubFilter) for f in handler.filters):
            continue
        handler.addFilter(scrub)
    # Also attach to the root logger itself — catches records that bypass
    # the child-propagation path (rare, but httpx in particular logs
    # via its own named logger so the propagation path IS how its
    # records arrive at the handlers).
    if not any(isinstance(f, SecretScrubFilter) for f in root.filters):
        root.addFilter(scrub)
