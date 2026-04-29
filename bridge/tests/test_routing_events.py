"""Tests for bridge.routing.events log writer.

Each test writes to a ``tmp_path``-scoped file so the real
``bridge/routing/data/events.jsonl`` is never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

from bridge.routing.events import log_decision


def _read_all(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def test_log_decision_appends_one_record(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    log_decision("what time is it", "get_time", ["get_time"], path=target)
    records = _read_all(target)
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "routing_decision"
    assert rec["prompt"] == "what time is it"
    assert rec["hint"] == "get_time"
    assert rec["picked"] == ["get_time"]
    assert rec["hint_followed"] is True
    assert rec["abstained"] is False
    assert "ts" in rec


def test_log_decision_hint_not_followed(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    log_decision("gold price", "tracker", ["wiki"], path=target)
    rec = _read_all(target)[0]
    assert rec["hint"] == "tracker"
    assert rec["picked"] == ["wiki"]
    assert rec["hint_followed"] is False


def test_log_decision_abstained(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    log_decision("hi karin", None, [], path=target)
    rec = _read_all(target)[0]
    assert rec["hint"] is None
    assert rec["picked"] == []
    assert rec["abstained"] is True
    assert rec["hint_followed"] is False


def test_log_decision_truncates_long_prompts(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    long_prompt = "x" * 500
    log_decision(long_prompt, None, [], path=target)
    rec = _read_all(target)[0]
    assert len(rec["prompt"]) <= 120


def test_log_decision_appends_across_calls(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    log_decision("a", "wiki", ["wiki"], path=target)
    log_decision("b", None, [], path=target)
    log_decision("c", "tracker", ["tracker", "get_time"], path=target)
    records = _read_all(target)
    assert [r["prompt"] for r in records] == ["a", "b", "c"]


def test_log_decision_swallows_oserror(tmp_path: Path, monkeypatch) -> None:
    """If the file can't be written, log_decision must not raise —
    logging is best-effort and should never break a chat turn."""
    target = tmp_path / "subdir" / "events.jsonl"

    def boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", boom)
    log_decision("x", "wiki", ["wiki"], path=target)


def test_log_decision_handles_none_prompt(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    log_decision(None, None, [], path=target)
    rec = _read_all(target)[0]
    assert rec["prompt"] == ""
