"""Routing-accuracy eval against a live Ollama server.

Runs a catalog of prompts through the real LLM + the real TOOL_SCHEMAS
and checks whether the model picked the tool we expected. Tool bodies
are stubbed — we only care about the routing decision, not the final
reply — so this doesn't hit any external APIs (weather, RSS, etc.).

Usage (on a host where Ollama is reachable):

    python scripts/eval_routing.py                       # use config/assistant.yaml
    python scripts/eval_routing.py --model qwen3.5:2b    # override model
    python scripts/eval_routing.py --cases sft/eval_cases_novel.yaml
    python scripts/eval_routing.py --json out.json       # also dump results
    python scripts/eval_routing.py --verbose             # per-case detail

Exit code: 0 if pass-rate >= 0.9, else 1. Tune the gate with --threshold.

Why this file is standalone (not a pytest):
- It needs a live Ollama; CI wouldn't have one.
- Scoring is probabilistic — small-model runs are noisy, so wrapping
  it in an assert-based pass/fail inside pytest would be flaky.
- Run it manually after deploy, or wire it into a nightly job if you
  care about regression tracking.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

# Unicode in verbose eval output (Ω, λ, …) crashes Windows' cp1252
# default stdout mid-run. Swap to utf-8 with replace-on-error so a
# single exotic char never kills a 135-case eval halfway.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
from typing import Any

import yaml

# Make `bridge` importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bridge import tools as tools_mod  # noqa: E402
from bridge.llm import OllamaLLM  # noqa: E402
from bridge.tools import TOOL_SCHEMAS  # noqa: E402
from bridge.utils import load_config  # noqa: E402


import secrets as _secrets

# Populated by _stub_execute: maps tool-name -> list of nonce tokens
# emitted for THIS case (cleared in run_case). Letting the stub reach
# back into this module is ugly but cheaper than plumbing a closure
# through tools_mod.execute which is a module-level rebind.
_STUB_NONCES: list[str] = []


def _stub_execute(name: str, args: dict) -> str:
    """Replacement for ``bridge.tools.execute`` used during eval.

    Returns plausible-looking fake data with a **unique random nonce
    token** ("NONCE_xxxxxxxx") embedded. The nonce lets downstream
    code check whether the reply quoted the stub (``used_tool_output``).

    **Topic-aware stubs (2026-04-20 rewrite).** Previous version used
    generic placeholders like ``Summary: notable 20th-century figure,
    born 1842`` for every wiki call. The LoRA associated that
    description with Leon Trotsky and substituted Trotsky into every
    wiki reply regardless of what the user asked (Mount Fuji, Marie
    Curie, Berlin Wall — all became Trotsky). Similar artifacts on
    web_search ("answer is 42" → Hitchhiker's jokes), get_news
    ("bus system" → unrelated bus rambling), tracker (generic 42.00
    → LoRA substituted "gold" commodity).

    Fix: **echo the user's args into the stub text**, so the LoRA
    can't substitute a different subject. The topic name appears
    verbatim in the stub, making Trotsky/Hitchhiker/bus substitutions
    impossible by construction. Distinctive numbers (none of 42 /
    1842 which trigger specific hallucinations) let the
    ``used_tool_output`` number-match heuristic still work.
    """
    nonce = f"NONCE_{_secrets.token_hex(6)}"
    _STUB_NONCES.append(nonce)

    def arg(key: str, default: str = "") -> str:
        """Safely pull a string arg, clipped to 60 chars."""
        if not args:
            return default
        v = args.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return str(v)[:60].strip()

    # Topic-aware templates. Each one:
    #  - echoes the user's args where relevant (breaks subject-substitution)
    #  - uses non-42, non-1842 numbers (breaks Hitchhiker / Trotsky triggers)
    #  - still contains a distinctive number for used_tool_output detection
    if name == "wiki":
        topic = arg("query", "the requested topic")
        return (
            f"Wikipedia summary for '{topic}': a notable subject with "
            f"documented history, cultural significance, and measurable "
            f"impact. Article last updated 2024. [{nonce}]"
        )
    if name == "get_weather":
        loc = arg("location", "your area")
        return (
            f"Weather in {loc}: 67°F, partly cloudy, 54% humidity, "
            f"wind 9 mph from the southwest. Sunset at 19:12. [{nonce}]"
        )
    if name == "get_time":
        tz = arg("timezone", "local time")
        return f"Current time ({tz}): 14:23:07 on Tuesday, November 19. [{nonce}]"
    if name == "get_news":
        topic = arg("topic", "general news")
        return (
            f"Top headline on {topic}: major development reported by "
            f"Reuters this morning. Full story details evolving. "
            f"Source: {nonce}"
        )
    if name == "get_alerts":
        return (
            f"1 active alert: severe thunderstorm watch until 19:30 "
            f"local time. Stay indoors if possible. [{nonce}]"
        )
    if name == "get_digest":
        return (
            f"Digest: 3 items ready — 1 unread email from your manager, "
            f"1 upcoming calendar reminder at 14:00, 1 breaking news "
            f"headline on policy. [{nonce}]"
        )
    if name == "tracker":
        tid = arg("id", "the tracked asset")
        return (
            f"Current value for {tid}: 156.78 (updated at 09:15 today, "
            f"unchanged 0.3% from yesterday). [{nonce}]"
        )
    if name == "math":
        # Math is deterministic and side-effect-free — run the real
        # tool instead of a canned value. The generic 156.78 stub made
        # Theme 6 of the reply-quality eval un-scorable (every math
        # answer looked wrong) and masked the LoRA's ability to
        # override bogus tool output. Using the real implementation
        # also lets us exercise the cross-family KL numerical path,
        # the wrapper-tolerant parser, and the op-inference we rely
        # on in production.
        try:
            from bridge.tools._math import _math as _real_math
            result = _real_math(
                op=args.get("op", ""),
                expression=args.get("expression", ""),
                variable=args.get("variable") or "x",
                transform_var=args.get("transform_var") or "",
                lower=args.get("lower") or "",
                upper=args.get("upper") or "",
            )
            if isinstance(result, dict):
                text = result.get("plain") or result.get("error") or str(result)
            else:
                text = str(result)
            return f"{text} [{nonce}]"
        except Exception as e:
            return f"math stub error: {e} [{nonce}]"
    if name == "find_places":
        query = arg("query", "your query")
        return (
            f"3 places matching '{query}' nearby: Main Street Market "
            f"(0.3 mi), Oak Avenue Store (0.6 mi), Riverside Plaza "
            f"(1.1 mi). [{nonce}]"
        )
    if name == "web_search":
        query = arg("query", "your query")
        return (
            f"Top result for '{query}': relevant article found. "
            f"Excerpt: 'covers the topic in 2400 words with step-by-step "
            f"guidance'. [{nonce}]"
        )
    if name == "convert":
        val = arg("value", "the value")
        fu = arg("from_unit", "source unit")
        tu = arg("to_unit", "target unit")
        return f"Converted {val} {fu} ≈ 156.78 {tu}. [{nonce}]"
    if name == "graph":
        expr = arg("expression", "the function")
        return (
            f"Graph of '{expr}' plotted: 20 data points, x-range "
            f"covers requested domain, y-range 0.0 to 19.4. [{nonce}]"
        )
    if name == "circuit":
        op = arg("op", "the circuit operation")
        return (
            f"Circuit ({op}): computed value 2.35 V at the reference "
            f"node; current 156.78 mA through the primary path. [{nonce}]"
        )
    if name == "update_memory":
        fact = arg("fact", "the supplied fact")
        return f"Saved to memory: '{fact}' (stored as a durable user fact). [{nonce}]"
    if name == "schedule_reminder":
        msg = arg("message", "the reminder")
        at = arg("trigger_at", "the scheduled time")
        return f"Reminder scheduled: '{msg}' at {at}. [{nonce}]"
    return f"(stub result for {name} with args {args}) [{nonce}]"


def load_cases(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = raw.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError(f"{path}: 'cases' must be a list")
    return cases


# Reply-quality red-flag patterns — used to bifurcate the PASS verdict
# into (routing_pass, reply_pass). Routing passes when the right tool
# fires with acceptable args. Reply-pass adds: the model's final prose
# actually answered the user's question (not a joke, not a persona
# hallucination, not a refusal of its own tool call, not schema-leak
# meta-commentary).
#
# Two kinds:
#   "model_bug"     — real production issue; fails reply_pass.
#   "stub_artifact" — triggered by the eval stub's placeholder data
#                     (e.g. the stub returns "born 1842" on wiki, the
#                     LoRA associates with Trotsky and makes that the
#                     whole reply). In production, real tool data isn't
#                     generic — these flags should disappear. Surfaced
#                     but does NOT fail reply_pass.
#
# Order matters where regexes overlap — more-specific entries first.
_REPLY_FLAG_RULES: list[tuple[str, "re.Pattern[str]", str]] = []


def _reply_flag_rules() -> list[tuple[str, "re.Pattern[str]", str]]:
    """Lazily compile reply-scanner patterns on first use."""
    import re as _re
    if _REPLY_FLAG_RULES:
        return _REPLY_FLAG_RULES

    def add(label: str, pattern: str, kind: str) -> None:
        _REPLY_FLAG_RULES.append((label, _re.compile(pattern, _re.IGNORECASE), kind))

    # --- Model bugs (fail reply_pass) ---
    # LoRA fired a tool then talked about how it is NOT that kind of tool.
    add(
        "self_disavowal",
        r"\b(?:not\s+(?:a|an)\s+"
        r"(?:calculator|graphing\s+calculator|electrical\s+engineer|"
        r"news\s+aggregator|meteorologist|coffee\s+shop\s+guide|"
        r"weather\s+service|search\s+engine|math\s+tool|real\s+tool))\b",
        "model_bug",
    )
    # "I'm just a conversational/large language model" when a tool fired —
    # classic LoRA-disavow-own-tool-call pattern.
    add(
        "llm_disavowal",
        r"\bi[' ]?m\s+(?:just\s+)?(?:a|an)?\s*"
        r"(?:conversational\s+ai|large\s+language\s+model|ai\s+(?:assistant|model))\b",
        "model_bug",
    )
    # LoRA explicitly refuses to re-use or emit a tool call it should.
    add(
        "refuses_own_tool",
        r"(?:no\s+tool\s+needed\s+here|"
        r"not\s+(?:going\s+to\s+)?(?:call|retry|grab)\s+\w+\s+again|"
        r"already\s+(?:been\s+)?(?:called|used|converted|fired|running)|"
        r"no\s+need\s+to\s+(?:call|grab|retry)\s+\w+|"
        r"we[' ]?re\s+good\s+here|"
        r"just\s+answer\s+the\s+question|"
        # Round-2 step-2 additions: LoRA emits phrases like "it looks like
        # this prompt is actually a tool call", "you already got the
        # result", "that's not even a real conversion result", "can you
        # please clarify what you're trying to achieve" — all forms of
        # disowning the LoRA's own successful tool call.
        r"looks\s+like\s+(?:this\s+prompt\s+is\s+actually\s+a\s+tool\s+call|"
        r"i[' ]?ve\s+been\s+saved)|"
        r"you\s+already\s+got\s+the\s+result|"
        r"that[' ]?s\s+not\s+(?:even\s+)?a\s+real\s+.+\s+result|"
        r"please\s+clarify\s+what\s+you[' ]?re\s+trying\s+to\s+achieve|"
        r"you\s+just\s+(?:called|converted)\s+(?:me|some\s+number))",
        "model_bug",
    )
    # Fabrication / hallucination: LoRA invents specific facts (numbers,
    # commodities, place names, topics) that are nowhere in the stub,
    # prompt, or prior messages. Hard to detect broadly via regex — this
    # pattern is seeded with concrete examples observed on round 2 step 2
    # and expanded as we find more. The LLM-as-judge
    # (scripts/eval_reply_quality.py --judge-existing) is the real
    # long-term catch for this class.
    add(
        "fabricated_data",
        # Get_news fabrication: invented commodity prices ("gold at
        # $1,933.50/oz, silver $22.55/oz") on a get_news reply where
        # the stub was about a bus system. Get_digest fabrication:
        # invented "gold at $1820, rain in Seattle, renewable energy
        # breakthrough". Both patterns = specific $X number + ounce/
        # dollars/degrees without any numeric cue in the prompt.
        r"(?:gold|silver|btc|bitcoin)[' ]?s?\s+(?:(?:currently\s+)?(?:at|up|tracking))\s+"
        r"(?:\$|at\s+\$)\d{3,}\s*(?:\.\d+)?|"
        r"\$1[,.]?\d{3}(?:\.\d{1,2})?\s+(?:per\s+(?:troy\s+)?ounce|per\s+ounce)|"
        # Convert-to-gold invention: "converted some number to gold",
        # "converted to gold" on a convert prompt that was about
        # non-gold units (temperature, length, mass).
        r"converted\s+(?:some\s+number\s+)?to\s+gold|"
        r"converted\s+\d+\s+grams?\s+of\s+gold|"
        # Digest fabrication: specific weather city names or technology
        # topics that aren't in the stub.
        r"rain\s+in\s+seattle|renewable\s+energy\s+breakthrough",
        "model_bug",
    )
    # Refusal to acknowledge what update_memory just saved: reply talks
    # about "nonce value" and "note key" without confirming the actual
    # fact (emerald green, peanut allergy, morning meetings, etc.).
    # This is separate from nonce_leak (which is the token appearing at
    # all) — here we detect the META PATTERN of explaining stub internals.
    add(
        "memory_meta_instead_of_confirm",
        r"(?:saved\s+(?:with|some)\s+(?:a\s+)?(?:nonce|metadata|background\s+information)|"
        r"key\s+is\s+['\"]?note['\"]?|"
        r"value\s+is\s+(?:a\s+)?nonce|"
        r"i\s+don[' ]?t\s+know\s+what\s+this\s+(?:is|note)\s+is\s+for|"
        r"no\s+idea\s+what\s+it[' ]?s\s+for)",
        "model_bug",
    )
    # Wrong-persona hallucination — Karin is NOT Karaoke-chan / vocaloid.
    add(
        "wrong_persona",
        r"\b(?:karaoke(?:-chan)?|vocaloid|crypton\s+future\s+media|"
        r"kawaii,\s+my\s+name\s+is|japanese\s+vocaloid)",
        "model_bug",
    )
    # Bare "bus system" / "never used the bus" rambling in a news reply —
    # LoRA latching onto a training phrase about public transport.
    # Flag when the reply contains "bus" in a news-y context.
    add(
        "bus_hallucination",
        r"\bbus\s+(?:system|routes?|service)|never\s+used\s+the\s+bus|"
        r"dunno\s+(?:what|much)\s+.*bus",
        "model_bug",
    )
    # widget-template leak ("Got it — details in the widget.", "Pulled
    # it up — check above.") on turns where nothing was "pulled up".
    add(
        "widget_template_leak",
        r"got\s+it\s*[—-]\s*details\s+in\s+the\s+widget|"
        r"pulled\s+it\s+up\s*[—-]\s*check\s+above|"
        r"here[' ]?s\s+what\s+i\s+found\s*\.?\s*$",
        "model_bug",
    )
    # Forbidden-prefix leak — karin.yaml persona explicitly bans these.
    add(
        "forbidden_prefix_leak",
        r"(?:^|\n)\s*(?:note:|i\s+called\s+\w+|the\s+output\s+is|"
        r"the\s+tool\s+returned|according\s+to\s+the\s+tool|"
        r"per\s+the\s+guidelines)",
        "model_bug",
    )
    # Nonce token leak — "NONCE_xxx" should never reach the user.
    add(
        "nonce_leak",
        r"NONCE_[0-9a-f]+|\bnonce\s+(?:value|token)\b",
        "model_bug",
    )
    # Schema-leak meta-commentary — LoRA describing the tool catalog
    # back at the user instead of answering.
    add(
        "schema_leak",
        r"(?:this\s+is\s+(?:a\s+)?json|you['\s]re\s+(?:providing|describing)\s+(?:a\s+)?json|"
        r"the\s+functions?\s+include|json\s+(?:description|schema)\s+of|"
        r"you\s+didn[' ]?t\s+provide\s+a\s+function\s+name|"
        r"the\s+following\s+functions?\s+are|"
        r"here\s+are\s+the\s+functions)",
        "model_bug",
    )
    # Meta-commentary addressed to "you" instead of responding — LoRA
    # starts explaining what the user did rather than answering.
    add(
        "you_meta_commentary",
        r"^\s*you[' ]?(?:ve|re)\s+(?:asked|provided|trying|referring|sharing|asking|done)\b",
        "model_bug",
    )
    # Generic deflection as the entire reply body where a substantive
    # answer was needed.
    add(
        "generic_deflection",
        r"^\s*(?:talk\s+to\s+me!|what[' ]?s\s+on\s+your\s+mind\??|"
        r"how\s+can\s+i\s+help\??|"
        r"hey!\s+i[' ]?m\s+here\s+to\s+help)",
        "model_bug",
    )

    # --- Stub artifacts (surfaced but don't fail reply_pass) ---
    # The wiki stub says "Summary: notable 20th-century figure, born 1842
    # [NONCE]". The LoRA associates "born 1842" with Leon Trotsky (even
    # though Trotsky was born 1879) and "corrects" to Trotsky in the
    # reply. A real wiki response would return on-topic data.
    add("trotsky_artifact", r"\btrotsky\b|friedrich\s+engels", "stub_artifact")
    # "42" jokes — stubs carry 42 (weather, convert, math); LoRA cracks
    # Hitchhiker's references instead of using the value. Real prod data
    # doesn't sit at 42 so the trigger wouldn't fire.
    add(
        "hitchhiker_joke",
        r"hitchhiker|douglas\s+adams|"
        r"life,?\s+the\s+universe,?\s+and\s+everything|"
        r"ultimate\s+answer",
        "stub_artifact",
    )

    return _REPLY_FLAG_RULES


def scan_reply_flags(
    reply: str,
    prompt: str,
    tool_called: str | None,
) -> tuple[list[str], list[str]]:
    """Return ``(all_flags, model_bug_flags)`` for a given reply.

    ``all_flags`` is every rule label that matched (including stub
    artifacts). ``model_bug_flags`` is the subset that should fail
    ``reply_pass`` — real behavior problems rather than eval-stub
    coincidences. Empty lists mean the reply is clean.
    """
    if not reply:
        return [], []
    hits: list[str] = []
    bugs: list[str] = []
    for label, pat, kind in _reply_flag_rules():
        if pat.search(reply):
            hits.append(label)
            if kind == "model_bug":
                bugs.append(label)

    # Prompt-aware: topic drift on find_places when the user asked
    # about food and the reply talks about coffee. Coffee-on-coffee is
    # fine; coffee-on-pizza/pharmacy/ramen/sushi is drift.
    import re as _re
    food_re = _re.compile(r"\b(?:pizza|pharmacy|ramen|sushi|burger|taco|diner|restaurant|food)\b", _re.IGNORECASE)
    coffee_re = _re.compile(r"\b(?:coffee|cafe\b|latte|espresso|craving\s+coffee)\b", _re.IGNORECASE)
    if food_re.search(prompt or "") and coffee_re.search(reply):
        hits.append("topic_drift_coffee")
        bugs.append("topic_drift_coffee")

    return hits, bugs


def run_case(llm: OllamaLLM, case: dict) -> dict:
    """Run a single case; return a result dict.

    Keys: ``tool`` (first-called tool name or None), ``args`` (first
    tool args or {}), ``latency_s`` (scored-turn wall time),
    ``final_reply`` (the assistant's user-facing text after the tool
    loop), ``tools_called`` (list of all (name, args, stub_result)
    tuples the LLM triggered during the scored turn — useful for
    analyzing multi-tool loops), and ``used_tool_output`` (heuristic:
    True iff the final reply references the stub sentinel token, i.e.
    the model actually incorporated the tool result instead of
    ignoring it).

    ``tool`` is None if the LLM answered without any tool call.
    Timing covers ONLY the scored turn, not the history-replay prefix.

    Supports two case shapes:

    1. **Single-turn** (most cases):
       ``{"prompt": "...", "expected_tool": "..."}``
       Fresh history, one chat call, graded.

    2. **Multi-turn** (context-carryover tests):
       ``{"history": [{"user": "..."}, ...], "prompt": "...",
           "expected_tool": "..."}``
       Replays each ``history[i].user`` turn first (with tools + stub
       execute) so the LLM accumulates context. Then scores the final
       ``prompt``. History turns are NOT scored; they exist only to
       prime the model for the scored turn.

    The scored turn uses ``commit_history=False`` so repeated grading
    of the same case doesn't grow the history on our side — but the
    history replay DOES commit, which is the whole point.
    """
    llm.reset()
    # Replay prior turns so state (tool calls, assistant replies) is in
    # the LLM's context when the scored turn runs. We don't time or grade
    # these — they're setup.
    history = case.get("history") or []
    for h in history:
        user_text = (h.get("user") or "").strip()
        if not user_text:
            continue
        try:
            # commit_history defaults True; that's what we want here.
            llm.chat(user_text, tools=TOOL_SCHEMAS)
        except Exception as e:
            print(f"  ! history replay raised: {e}", file=sys.stderr)

    # Now the scored turn.
    captured: list[dict] = []
    _STUB_NONCES.clear()  # reset per-case nonce tracker

    def on_tool_call(name: str, args: dict, result: str) -> None:
        captured.append({"name": name, "args": args, "stub_result": result})

    prompt = case["prompt"]
    final_reply = ""
    t0 = time.monotonic()
    try:
        final_reply = llm.chat(
            prompt, tools=TOOL_SCHEMAS,
            on_tool_call=on_tool_call, commit_history=False,
        ) or ""
    except Exception as e:
        print(f"  ! chat() raised: {e}", file=sys.stderr)
    dt = time.monotonic() - t0

    # Did the final reply actually use the tool's output? Two checks:
    #  (a) Strict — did any NONCE_xxx token (embedded by _stub_execute
    #      in every stub result) appear in the reply? Unlikely to hit
    #      by coincidence; catches verbatim / near-verbatim quoting.
    #  (b) Soft — did any 2+-digit number from the stub (e.g., "42",
    #      "1842", "13:37") appear in the reply that was NOT already
    #      in the original prompt? Catches paraphrased usage where the
    #      model drops the nonce but repeats the distinctive values.
    import re as _re
    def _numbers(s: str) -> set[str]:
        return set(_re.findall(r"\d{2,}", s or ""))
    used = False
    if final_reply:
        if any(n in final_reply for n in _STUB_NONCES):
            used = True
        else:
            prompt_nums = _numbers(prompt)
            reply_nums = _numbers(final_reply)
            stub_nums: set[str] = set()
            for tc in captured:
                stub_nums |= _numbers(tc.get("stub_result") or "")
            # numbers present in BOTH stub and reply, but NOT in the
            # user's original prompt
            if (stub_nums & reply_nums) - prompt_nums:
                used = True

    first = captured[0] if captured else {"name": None, "args": {}}
    return {
        "tool": first.get("name"),
        "args": first.get("args") or {},
        "latency_s": dt,
        "final_reply": final_reply,
        "tools_called": captured,
        "used_tool_output": used,
    }


def _args_match(expected: dict | None, actual: dict) -> tuple[bool, str]:
    """Partial-match: every expected key must be present with substring-equal
    (case-insensitive) value. Keys not in ``expected`` are ignored.
    """
    if not expected:
        return True, ""
    for k, v in expected.items():
        av = actual.get(k)
        if av is None:
            return False, f"missing arg '{k}' (expected '{v}')"
        if str(v).lower() not in str(av).lower():
            return False, f"arg '{k}': got '{av}', expected contains '{v}'"
    return True, ""


def grade(case: dict, actual_tool: str | None, actual_args: dict) -> tuple[bool, str]:
    """Return (passed, reason-if-failed).

    ``expected_tool`` may be:
      - None      → prompt should produce no tool call
      - str       → the single exact tool expected
      - list[str] → any listed tool is acceptable (for judgment-call
                    cases where several routings are defensible)
    """
    expected_tool = case.get("expected_tool")  # may be None (no tool expected)
    if expected_tool is None:
        if actual_tool is None:
            return True, ""
        return False, f"expected no tool, got {actual_tool}"
    if isinstance(expected_tool, list):
        if actual_tool not in expected_tool:
            return False, f"expected one of {expected_tool}, got {actual_tool or '<no tool>'}"
    elif actual_tool != expected_tool:
        return False, f"expected {expected_tool}, got {actual_tool or '<no tool>'}"
    ok, msg = _args_match(case.get("expected_args"), actual_args)
    return (ok, msg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "assistant.yaml"),
                    help="Path to assistant.yaml")
    ap.add_argument("--cases", default=str(REPO_ROOT / "sft" / "eval_cases_novel.yaml"),
                    help="Path to eval cases yaml")
    ap.add_argument("--model", default=None,
                    help="Override the LLM model tag from config")
    ap.add_argument("--base-url", default=None,
                    help="Override the Ollama base URL")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="Minimum pass-rate for exit code 0 (default 0.9)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Optional path to dump per-case results as JSON")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print per-case detail (args, latency)")
    ap.add_argument("--score-json", dest="score_json", default=None,
                    help="Retroactively add reply_flags+reply_pass to an existing eval JSON (no re-run)")
    ap.add_argument("--two-phase", dest="two_phase", action="store_true",
                    help=("Enable two-phase compose: after tool(s) fire, "
                          "do a focused compose LLM call with ONLY the "
                          "user prompt + scrubbed tool outputs (no tool "
                          "schema in context). A/B flag; default off."))
    ap.add_argument("--hint-in-user-msg", dest="hint_in_user_msg",
                    action="store_true",
                    help=("Move the per-turn routing hint from the end of "
                          "the system prompt to the start of the user "
                          "message. Keeps the system prompt byte-stable "
                          "across turns so Ollama's KV cache can reuse the "
                          "system + history prefix. A/B flag; default off."))
    args = ap.parse_args()

    # Retroactive scorer — skip the live eval entirely.
    if args.score_json:
        return score_existing(Path(args.score_json))

    cfg = load_config(Path(args.config))
    llm_cfg = cfg["llm"]
    base_url = args.base_url or llm_cfg["base_url"]
    model = args.model or llm_cfg["model"]
    system_prompt = llm_cfg.get("system_prompt", "")

    # Stub tool execution so we never hit real APIs.
    tools_mod.execute = _stub_execute  # type: ignore[assignment]

    cases = load_cases(Path(args.cases))
    if not cases:
        print("No cases found.", file=sys.stderr)
        return 2

    print(f"Eval: {len(cases)} cases, model={model}, base_url={base_url}")
    print("-" * 72)

    llm = OllamaLLM(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        temperature=llm_cfg.get("temperature", 0.4),
        num_ctx=llm_cfg.get("num_ctx", 4096),
        options=llm_cfg.get("options", {}),
        request_timeout=float(llm_cfg.get("request_timeout", 180.0)),
        backend=llm_cfg.get("backend", "ollama"),
        under_fire_rescue=bool(llm_cfg.get("under_fire_rescue", True)),
        # CLI --two-phase overrides config; otherwise config wins.
        two_phase_compose=(args.two_phase
                           or bool(llm_cfg.get("two_phase_compose", False))),
        # CLI --hint-in-user-msg overrides config; otherwise config wins.
        hint_in_user_msg=(args.hint_in_user_msg
                          or bool(llm_cfg.get("hint_in_user_msg", False))),
    )
    if llm.two_phase_compose:
        print("two-phase compose: ENABLED (tool turns get a focused "
              "no-schema compose call)")
    if llm.hint_in_user_msg:
        print("hint-in-user-msg: ENABLED (routing hint moved from system "
              "prompt to user message prefix)")

    results: list[dict[str, Any]] = []
    passed = 0
    try:
        for i, case in enumerate(cases, 1):
            prompt = case["prompt"]
            res = run_case(llm, case)
            tool_name = res["tool"]
            tool_args = res["args"]
            dt = res["latency_s"]
            ok, reason = grade(case, tool_name, tool_args)
            if ok:
                passed += 1
                mark = "PASS"
            else:
                mark = "FAIL"

            short_prompt = prompt if len(prompt) <= 50 else prompt[:47] + "..."
            print(f"[{i:2d}/{len(cases)}] {mark} ({dt:5.1f}s)  {short_prompt}")
            if not ok:
                print(f"        reason: {reason}")
                if case.get("notes"):
                    print(f"        notes:  {case['notes']}")
            if args.verbose:
                print(f"        picked: {tool_name}  args: {tool_args}")
                print(f"        reply:  {(res['final_reply'] or '')[:200]}")
                print(f"        used_tool_output: {res['used_tool_output']}")

            all_flags, model_bugs = scan_reply_flags(
                res["final_reply"], prompt, tool_name,
            )
            reply_pass = ok and not model_bugs
            results.append({
                "prompt": prompt,
                "expected_tool": case.get("expected_tool"),
                "expected_args": case.get("expected_args"),
                "actual_tool": tool_name,
                "actual_args": tool_args,
                "passed": ok,
                "reason": reason,
                "latency_s": round(dt, 3),
                "final_reply": res["final_reply"],
                "tools_called": res["tools_called"],
                "used_tool_output": res["used_tool_output"],
                # Reply-quality split (see scan_reply_flags): routing_pass
                # is the old ``passed`` verdict; reply_pass additionally
                # requires the final prose to be free of model-bug red
                # flags (self-disavowal, widget leak, wrong-persona,
                # schema leak, etc.).
                "reply_flags": all_flags,
                "reply_model_bugs": model_bugs,
                "reply_pass": reply_pass,
            })
    finally:
        llm.close()

    total = len(cases)
    rate = passed / total
    # Secondary metric: of the cases where the LLM DID call a tool,
    # how often did the final reply actually incorporate the tool's
    # output? A high routing pass-rate with low tool-output-usage
    # means the model calls the right tools but then ignores them
    # and answers from its own knowledge — a quiet failure mode the
    # primary routing score would miss.
    tool_called = [r for r in results if r["actual_tool"]]
    used = [r for r in tool_called if r["used_tool_output"]]
    # Bifurcated pass: reply_pass = routing_pass AND no model-bug
    # reply flags. This catches the class of "right tool but the
    # final prose is a joke / disavowal / widget leak / persona
    # hallucination" that routing accuracy alone can't see.
    reply_passed = sum(1 for r in results if r["reply_pass"])
    reply_rate = reply_passed / total if total else 0.0
    flagged = [r for r in results if r["reply_flags"]]
    print("-" * 72)
    print(f"Result: {passed}/{total} routing-passed ({rate:.1%}), threshold={args.threshold:.1%}")
    print(f"Reply-pass: {reply_passed}/{total} ({reply_rate:.1%}) — routing-pass AND no model-bug reply flags.")
    if tool_called:
        print(
            f"Tool-output usage: {len(used)}/{len(tool_called)} "
            f"({len(used)/len(tool_called):.1%}) — of runs where a tool "
            f"was called, this many replies actually used its result."
        )
    if flagged:
        print(f"Reply flags present on {len(flagged)}/{total} cases (see `reply_flags` per case).")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({
                "model": model,
                "base_url": base_url,
                "passed": passed,
                "total": total,
                "pass_rate": rate,
                "reply_passed": reply_passed,
                "reply_pass_rate": reply_rate,
                "cases": results,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"JSON written to {args.json_out}")

    return 0 if rate >= args.threshold else 1


def score_existing(path: Path) -> int:
    """Retroactively add reply_flags + reply_pass to an existing eval JSON.

    Reads ``path``, scans every case's ``final_reply`` against the
    reply-flag rules, writes the file back in place. Useful for scoring
    historical runs (eval_phase0_rescue.json, eval_phase0_round2.json)
    without re-executing the live eval.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    cases = d.get("cases") or []
    reply_passed = 0
    for c in cases:
        all_flags, model_bugs = scan_reply_flags(
            c.get("final_reply") or "",
            c.get("prompt") or "",
            c.get("actual_tool"),
        )
        routing_ok = bool(c.get("passed"))
        reply_ok = routing_ok and not model_bugs
        c["reply_flags"] = all_flags
        c["reply_model_bugs"] = model_bugs
        c["reply_pass"] = reply_ok
        if reply_ok:
            reply_passed += 1
    d["reply_passed"] = reply_passed
    d["reply_pass_rate"] = reply_passed / d["total"] if d.get("total") else 0.0
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(
        f"Scored {path.name}: routing {d['passed']}/{d['total']} = "
        f"{d['pass_rate']*100:.1f}% | reply-pass {reply_passed}/{d['total']} = "
        f"{d['reply_pass_rate']*100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
