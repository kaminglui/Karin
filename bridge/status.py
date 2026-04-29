"""Read-only health snapshot for the bridge subsystems.

Run:
    python -m bridge.status

Inspects config presence, data-dir state, persistent fetch errors,
advisory baseline status, and alert counts. Performs NO network I/O,
NO rescans, and NO writes -- safe to run any time.

Output uses three tags:
    [ok]    everything looks healthy for this item
    [info]  expected first-run state, or empty-but-fine
    [warn]  something is misconfigured or has a persistent error worth investigating

Exit code is always 0; this is a status display, not a CI check.

ASCII-only output -- renders cleanly on cp1252 Windows terminals
without UTF-8 codepage tweaks.
"""
from __future__ import annotations

import json
from pathlib import Path

from bridge.utils import REPO_ROOT


def _ok(msg: str) -> str:    return f"  [ok]    {msg}"
def _info(msg: str) -> str:  return f"  [info]  {msg}"
def _warn(msg: str) -> str:  return f"  [warn]  {msg}"


def _read_json(path: Path):
    """Best-effort JSON read. Returns (data, error_message_or_None)."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, str(e)


# --- news -----------------------------------------------------------------

def check_news() -> list[str]:
    out = ["News subsystem:"]
    base = REPO_ROOT / "bridge" / "news"
    prefs_real = base / "config" / "preferences.json"
    prefs_example = base / "config" / "preferences.example.json"

    if prefs_real.exists():
        out.append(_ok("preferences.json present (curated)"))
    elif prefs_example.exists():
        out.append(_info(
            "no preferences.json: running with preferences disabled "
            "(state + recency ranking only). Copy preferences.example.json "
            "to preferences.json to enable watchlists."
        ))
    else:
        out.append(_warn("no preferences template found at all"))

    data_dir = base / "data"
    articles_path = data_dir / "articles.json"
    clusters_path = data_dir / "clusters.json"

    if articles_path.exists():
        data, err = _read_json(articles_path)
        if err:
            out.append(_warn(f"articles.json unreadable: {err}"))
        else:
            out.append(_ok(f"articles.json: {len(data)} articles stored"))
    else:
        out.append(_info("articles.json missing: first get_news call will populate"))

    if clusters_path.exists():
        data, err = _read_json(clusters_path)
        if err:
            out.append(_warn(f"clusters.json unreadable: {err}"))
        else:
            multi = sum(1 for c in data.values() if c.get("article_count", 0) > 1)
            out.append(_ok(f"clusters.json: {len(data)} clusters ({multi} multi-source)"))
    else:
        out.append(_info("clusters.json missing: first get_news call will populate"))

    return out


# --- trackers --------------------------------------------------------------

def check_trackers() -> list[str]:
    out = ["Tracker subsystem:"]
    base = REPO_ROOT / "bridge" / "trackers"
    cfg_real = base / "config" / "trackers.json"
    cfg_example = base / "config" / "trackers.example.json"

    if cfg_real.exists():
        out.append(_ok("trackers.json present (curated)"))
    elif cfg_example.exists():
        out.append(_info(
            "no trackers.json: using trackers.example.json directly. "
            "Copy it to trackers.json to customize without affecting the template."
        ))
    else:
        out.append(_warn("no tracker config found: trackers will not work"))

    data_path = base / "data" / "trackers.json"
    if not data_path.exists():
        out.append(_info(
            "no tracker history yet: first get_tracker/get_trackers call "
            "will populate. Derived labels (direction/movement/shock) need "
            "~11-26 days of history before they activate."
        ))
        return out

    data, err = _read_json(data_path)
    if err:
        out.append(_warn(f"data/trackers.json unreadable: {err}"))
        return out

    out.append(_ok(f"data/trackers.json: {len(data)} trackers tracked"))
    for tid, rec in sorted(data.items()):
        hist_n = len(rec.get("history", []))
        err_msg = rec.get("last_fetch_error")
        if err_msg:
            out.append(_warn(f"  {tid}: {hist_n} readings; last fetch error: {err_msg}"))
        elif hist_n == 0:
            out.append(_info(f"  {tid}: no readings yet"))
        else:
            out.append(_ok(f"  {tid}: {hist_n} readings"))
    return out


# --- alerts ---------------------------------------------------------------

def check_alerts() -> list[str]:
    out = ["Alerts subsystem:"]
    data_dir = REPO_ROOT / "bridge" / "alerts" / "data"

    advisory_state = data_dir / "advisory_state.json"
    if advisory_state.exists():
        data, err = _read_json(advisory_state)
        if err:
            out.append(_warn(f"advisory_state.json unreadable: {err}"))
        else:
            high = sum(1 for lvl in data.values() if isinstance(lvl, int) and lvl >= 3)
            out.append(_ok(
                f"advisory_state.json: {len(data)} countries baselined "
                f"({high} at Level 3+)"
            ))
    else:
        out.append(_info(
            "advisory_state.json missing: first get_alerts call will baseline silently"
        ))

    cooldowns_path = data_dir / "cooldowns.json"
    if cooldowns_path.exists():
        data, err = _read_json(cooldowns_path)
        if err:
            out.append(_warn(f"cooldowns.json unreadable: {err}"))
        else:
            out.append(_ok(f"cooldowns.json: {len(data)} (rule, scope) entries"))
    else:
        out.append(_info("cooldowns.json missing: no rule has fired yet"))

    alerts_log = data_dir / "alerts.jsonl"
    if not alerts_log.exists():
        out.append(_info("alerts.jsonl missing: no scans have run yet"))
        return out

    n_fired = n_suppressed = n_collect_errors = 0
    last_collect_err = None
    try:
        with alerts_log.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = obj.get("kind")
                if kind == "alert_fired":
                    n_fired += 1
                elif kind == "alert_suppressed":
                    n_suppressed += 1
                elif kind == "collect_error":
                    n_collect_errors += 1
                    last_collect_err = obj
    except Exception as e:
        out.append(_warn(f"alerts.jsonl unreadable: {e}"))
        return out

    out.append(_ok(
        f"alerts.jsonl: {n_fired} alerts fired total, {n_suppressed} suppressed"
    ))
    if n_collect_errors:
        msg = f"{n_collect_errors} collect_error events in log"
        if last_collect_err:
            src = last_collect_err.get("source", "?")
            err = last_collect_err.get("error", "?")
            msg += f" (latest: source={src}, error={err})"
        out.append(_warn(msg))
    return out


# --- main -----------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("Karin subsystem status")
    print("=" * 64)
    for section in (check_news(), check_trackers(), check_alerts()):
        for line in section:
            print(line)
        print()


if __name__ == "__main__":
    main()
