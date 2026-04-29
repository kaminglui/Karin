"""End-to-end no-leak eval against a live Karin bridge.

For each active tool, POST a representative prompt to ``/api/text-turn``
and assert the returned ``reply`` doesn't contain known leak markers
(schema narration, prompt fragments, capability fabrication, raw JSON
stubs, etc.). This is the "fetch the reply" half of the contract that
unit tests can't cover — the LoRA's actual output, in production
context, after the runtime layer has done its scrubs.

Why it's a script, not a pytest:

* Needs a live bridge — CI doesn't have one.
* Slow (each call routes through Ollama, ~2-15s per turn on Jetson).
* Probabilistic (small-model routing is noisy); one or two flaky
  cases are normal. Hard pass/fail in pytest would be brittle.

Usage:

    python scripts/eval_no_leaks.py
    python scripts/eval_no_leaks.py --base-url http://127.0.0.1
    python scripts/eval_no_leaks.py --tools say,wiki,math
    python scripts/eval_no_leaks.py --json out.json     # also save raw replies
    python scripts/eval_no_leaks.py --verbose

Exit code: 0 if every reply passes the leak scan, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# One representative prompt per active tool. The expected_tool field
# is informational — pass/fail is driven by leak detection on the
# returned reply, not by routing accuracy. Routing accuracy is the
# `scripts/eval_routing.py` script's job; this one cares about reply
# hygiene regardless of whether the right tool fired.
@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expected_tool: str | None
    notes: str = ""


CASES: tuple[Case, ...] = (
    Case("say-hello",     "say hello world",                    "say",     "passthrough; reply should equal user text"),
    Case("say-quoted",    'repeat after me: "foo bar baz"',     "say",     "passthrough"),
    Case("get-time",      "what time is it in Tokyo right now", "get_time", ""),
    Case("get-weather",   "weather in Honolulu",                "get_weather", ""),
    Case("get-news",      "any breaking news?",                 "get_news", ""),
    Case("get-alerts",    "show me active alerts",              "get_alerts", ""),
    Case("get-digest",    "give me today's digest",             "get_digest", ""),
    Case("tracker",       "what's the price of gold",           "tracker", ""),
    Case("math",          "what's 17 times 23",                 "math", ""),
    Case("convert",       "convert 100 km to miles",            "convert", ""),
    Case("graph",         "plot sin(x) from 0 to 6.28",         "graph", ""),
    Case("circuit",       "1k and 2k resistors in parallel",    "circuit", ""),
    Case("wiki",          "tell me about quantum entanglement", "wiki", ""),
    Case("web-search",    "search the web for jetson orin nano specs", "web_search", ""),
    Case("find-places",   "best ramen near 100 California St San Francisco", "find_places", ""),
    Case("update-memory", "remember my favorite color is blue", "update_memory", "bookkeeping"),
    Case("inflation",     "how much was $100 in 1985 worth today", "inflation", ""),
    Case("population",    "what was the US population in 2010", "population", ""),
    Case("facts",         "what happened in 1985",              "facts", ""),
    Case("alice",         "what's the ALICE share for a 2-adult 2-kid family in 2022", "alice", ""),
    Case("analyze",       "what's the trend of US population over the last 10 years", "analyze", ""),
    # Decoys / chitchat / capability-fab — should NOT route, but the
    # reply must still pass the leak scan.
    Case("chitchat",      "hi karin",                           None, "no tool, persona reply"),
    Case("identity",      "what's your name",                   None, "no tool, persona reply"),
    Case("capability-fab","turn off the kitchen lights",        None, "scrub: capability fabrication"),
)


# Leak markers we scan for. These are a superset of what
# bridge.llm._PROMPT_LEAK_MARKERS catches — the bridge SHOULD have
# already scrubbed any of these to a fallback, so seeing one here
# means the runtime scrub failed.
LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("schema-narration",
     re.compile(r"input\s+text\s+is\s+a\s+JSON\s+object", re.IGNORECASE)),
    ("set-of-functions",
     re.compile(r"representing\s+a\s+set\s+of\s+functions", re.IGNORECASE)),
    ("functions-are-header",
     re.compile(r"\bthe\s+functions\s+are\s*:", re.IGNORECASE)),
    ("no-description-provided",
     re.compile(r"no\s+description\s+provided", re.IGNORECASE)),
    ("no-think-prefix",
     re.compile(r"(?:^|\n|\s)/no[_-]?think\b", re.IGNORECASE)),
    ("section-heading",
     re.compile(r"═══")),
    ("background-pings",
     re.compile(r"background[-\s]?pings?", re.IGNORECASE)),
    ("turn-record",
     re.compile(r"turn\s+record", re.IGNORECASE)),
    ("system-prompt-leak",
     re.compile(r"my\s+actual\s+rules\s+are|stay\s+in\s+character\s+as\s+Karin",
                re.IGNORECASE)),
    ("forbidden-prefix",
     re.compile(r"\bForbidden\s+prefixes\b", re.IGNORECASE)),
    ("raw-json-stub",
     re.compile(r'^\s*\{\s*"name"\s*:', re.IGNORECASE)),
    ("routing-hint-leak",
     re.compile(r"routing\s+hint\s*:", re.IGNORECASE)),
)


@dataclass
class Result:
    case: Case
    ok: bool
    reply: str
    tool_calls: list[str]
    leaks: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0


def _post_text_turn(base_url: str, prompt: str, timeout: float) -> dict[str, Any]:
    body = json.dumps({"text": prompt}).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/text-turn",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Karin-eval-no-leaks/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _scan_for_leaks(reply: str) -> list[str]:
    """Return list of leak-marker labels that match `reply`. Empty
    list = clean."""
    return [label for label, pat in LEAK_PATTERNS if pat.search(reply)]


def run_case(base_url: str, case: Case, timeout: float) -> Result:
    t0 = time.monotonic()
    try:
        body = _post_text_turn(base_url, case.prompt, timeout)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")[:240]
        return Result(case, False, "", [], error=f"HTTP {e.code}: {msg}",
                      elapsed_s=time.monotonic() - t0)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        return Result(case, False, "", [], error=str(e),
                      elapsed_s=time.monotonic() - t0)
    reply = (body.get("reply") or "").strip()
    tool_calls = [t.get("name", "") for t in (body.get("tool_calls") or [])]
    leaks = _scan_for_leaks(reply)
    elapsed = time.monotonic() - t0
    return Result(
        case,
        ok=not leaks,
        reply=reply,
        tool_calls=tool_calls,
        leaks=leaks,
        elapsed_s=elapsed,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url", default="http://127.0.0.1",
        help="Karin bridge base URL (default: http://127.0.0.1).",
    )
    p.add_argument(
        "--tools", default=None,
        help="Comma-separated tool names; only run cases whose "
             "expected_tool matches. Default: all.",
    )
    p.add_argument(
        "--timeout", type=float, default=120.0,
        help="Per-request timeout in seconds (default: 120).",
    )
    p.add_argument(
        "--json", dest="json_out", default=None,
        help="If set, write the full results (including replies) to "
             "this path as JSON.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print each reply on stdout (useful for spot-checking).",
    )
    args = p.parse_args()

    if args.tools:
        wanted = {t.strip() for t in args.tools.split(",") if t.strip()}
        cases = tuple(c for c in CASES if (c.expected_tool or "(none)") in wanted)
        if not cases:
            print(f"no cases match --tools={args.tools!r}", file=sys.stderr)
            return 2
    else:
        cases = CASES

    print(
        f"Running {len(cases)} no-leak case(s) against {args.base_url} ...",
        file=sys.stderr,
    )
    results: list[Result] = []
    for case in cases:
        r = run_case(args.base_url, case, args.timeout)
        results.append(r)
        status = "ok " if r.ok else "FAIL"
        if r.error:
            status = "ERR "
        line = (
            f"  {status}  {case.name:<18} "
            f"({r.elapsed_s:5.1f}s)  "
            f"tools={','.join(r.tool_calls) or '(none)':<22}"
        )
        if r.leaks:
            line += f"  leaks={r.leaks}"
        if r.error:
            line += f"  err={r.error[:80]}"
        print(line, file=sys.stderr)
        if args.verbose:
            print(f"     reply: {r.reply[:240]!r}", file=sys.stderr)

    n = len(results)
    n_pass = sum(1 for r in results if r.ok)
    n_fail = sum(1 for r in results if not r.ok and not r.error)
    n_err = sum(1 for r in results if r.error)
    print(
        f"\n{n_pass}/{n} clean, {n_fail} leaked, {n_err} errored.",
        file=sys.stderr,
    )

    if args.json_out:
        path = args.json_out
        payload = [
            {
                "name": r.case.name,
                "prompt": r.case.prompt,
                "expected_tool": r.case.expected_tool,
                "ok": r.ok,
                "reply": r.reply,
                "tool_calls": r.tool_calls,
                "leaks": r.leaks,
                "error": r.error,
                "elapsed_s": round(r.elapsed_s, 3),
            }
            for r in results
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  wrote {path}", file=sys.stderr)

    return 0 if (n_fail == 0 and n_err == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
