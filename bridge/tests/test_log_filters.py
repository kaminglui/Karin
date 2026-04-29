"""Tests for the secret-scrubbing logging filter."""
from __future__ import annotations

import logging

import pytest

from bridge.log_filters import (
    SecretScrubFilter,
    _scrub,
    install_secret_scrubber,
)


# ---- _scrub (the pure string rewriter) -----------------------------------


class TestScrubFunction:
    @pytest.mark.parametrize("raw,expected", [
        ("https://x.com/a?api_key=abc123&b=1",
         "https://x.com/a?api_key=<redacted>&b=1"),
        ("https://x.com/a?b=1&api_key=abc123",
         "https://x.com/a?b=1&api_key=<redacted>"),
        ("GET ?token=zzz HTTP/1.1", "GET ?token=<redacted> HTTP/1.1"),
        # Case insensitive on the param name
        ("?API_KEY=XYZ", "?API_KEY=<redacted>"),
        ("?ApiKey=abc", "?ApiKey=<redacted>"),
        # Multiple secrets in one URL
        ("?api_key=a&token=b", "?api_key=<redacted>&token=<redacted>"),
        # Variants we declared
        ("?access_token=a", "?access_token=<redacted>"),
        ("?auth_token=a", "?auth_token=<redacted>"),
        ("?secret=a", "?secret=<redacted>"),
    ])
    def test_masks_secret_query_params(self, raw, expected):
        assert _scrub(raw) == expected

    def test_leaves_non_secret_params_alone(self):
        assert _scrub("?foo=bar&q=hello") == "?foo=bar&q=hello"

    def test_leaves_non_query_assignments_alone(self):
        """Without '?' / '&' in front, it's not a URL query — don't touch."""
        assert _scrub("config: api_key=xyz") == "config: api_key=xyz"

    def test_empty_or_no_equals_fast_path(self):
        assert _scrub("") == ""
        assert _scrub(None) is None  # type: ignore[arg-type]
        assert _scrub("no secrets here") == "no secrets here"

    def test_stops_at_ampersand(self):
        """The value must end at '&' — it must not swallow the next param."""
        result = _scrub("?api_key=ABCDEF&user=alice")
        assert "user=alice" in result
        assert "ABCDEF" not in result


# ---- SecretScrubFilter (logging plumbing) --------------------------------


class TestSecretScrubFilter:
    def _make_record(self, msg: str, args=None) -> logging.LogRecord:
        return logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=args, exc_info=None,
        )

    def test_scrubs_msg_directly(self):
        rec = self._make_record("fetched ?api_key=XYZ OK")
        SecretScrubFilter().filter(rec)
        assert "XYZ" not in rec.msg
        assert "<redacted>" in rec.msg

    def test_scrubs_positional_args(self):
        """httpx-style: log.info('HTTP Request: GET %s %s', url, status)."""
        rec = self._make_record(
            "HTTP Request: GET %s %s",
            ("https://api.example.com/?api_key=LEAKME", "200 OK"),
        )
        SecretScrubFilter().filter(rec)
        assert "LEAKME" not in str(rec.args)
        assert "<redacted>" in str(rec.args[0])
        # Non-secret args stay untouched
        assert rec.args[1] == "200 OK"

    def test_scrubs_dict_args(self):
        rec = self._make_record(
            "request %(method)s %(url)s",
            {"method": "GET", "url": "?api_key=LEAK"},
        )
        SecretScrubFilter().filter(rec)
        assert "LEAK" not in rec.args["url"]
        assert rec.args["method"] == "GET"

    def test_does_not_mutate_non_secret_records(self):
        rec = self._make_record("user logged in %s", ("alice",))
        SecretScrubFilter().filter(rec)
        assert rec.msg == "user logged in %s"
        assert rec.args == ("alice",)

    def test_filter_returns_true_so_record_is_kept(self):
        """Filters that return False suppress the record entirely. We
        want the log line to STILL print, just scrubbed."""
        rec = self._make_record("?api_key=X")
        assert SecretScrubFilter().filter(rec) is True


# ---- install_secret_scrubber (root-logger wiring) ------------------------


class TestInstallScrubber:
    def setup_method(self):
        # Save + restore root filters AND handler filters around each
        # test so we don't pollute other tests' logger state.
        root = logging.getLogger()
        self._saved_filters = list(root.filters)
        self._saved_handler_filters = [list(h.filters) for h in root.handlers]

    def teardown_method(self):
        root = logging.getLogger()
        root.filters = self._saved_filters
        for h, saved in zip(root.handlers, self._saved_handler_filters):
            h.filters = saved

    def test_adds_filter_to_root_logger_and_handlers(self):
        install_secret_scrubber()
        root = logging.getLogger()
        assert any(isinstance(f, SecretScrubFilter) for f in root.filters)
        # If there are handlers, each should have the scrubber too.
        for h in root.handlers:
            assert any(isinstance(f, SecretScrubFilter) for f in h.filters)

    def test_is_idempotent(self):
        install_secret_scrubber()
        install_secret_scrubber()
        install_secret_scrubber()
        root = logging.getLogger()
        count = sum(1 for f in root.filters if isinstance(f, SecretScrubFilter))
        assert count == 1
        for h in root.handlers:
            handler_count = sum(
                1 for f in h.filters if isinstance(f, SecretScrubFilter)
            )
            assert handler_count == 1

    def test_end_to_end_child_logger_inherits_scrubbing(self):
        """A record logged by a child logger should arrive at the
        (installed) handlers scrubbed. Verified by attaching our own
        capture handler AFTER install_secret_scrubber so the scrubber
        gets added to it, then checking the message at emit time."""
        seen: list[str] = []
        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                seen.append(record.getMessage())
        handler = _CaptureHandler(level=logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            install_secret_scrubber()  # now adds the filter to the new handler
            logger = logging.getLogger("some.child")
            logger.setLevel(logging.INFO)
            logger.info("calling %s", "https://x/y?api_key=SECRETVALUE")
        finally:
            root.removeHandler(handler)
        joined = "\n".join(seen)
        assert "SECRETVALUE" not in joined
        assert "<redacted>" in joined
