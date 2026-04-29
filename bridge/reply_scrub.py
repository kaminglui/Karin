"""Post-processing of the LLM's final reply text (L8 in the routing pipeline).

Failure-mode catchers that run on every reply before it reaches the user
or TTS. Extracted from ``bridge.llm.OllamaLLM`` so the chat loop is
easier to read and so the regex catalog has a single home — when a new
LoRA failure mode shows up, the fix lives here, not in the loop.

Each scrub corresponds to a class of model failure observed in the
135-case eval. See ``docs/routing-pipeline.md § L8`` for the rationale
behind every pattern; comments below cite the specific failure mode each
one targets so future-you can decide whether a regex is still load-
bearing before deleting it.

Public surface used by ``bridge.llm``:
  * ``clean_reply(...)`` — top-level entry point, runs the whole chain.
  * ``pick_fallback(...)`` — pool selection used both inside
    ``clean_reply`` and standalone (e.g. when the chat loop hits an
    empty reply or a JSON-stub leak before scrubbing).
  * The pattern constants are still exported because the chat loop
    occasionally inspects them directly (e.g. early leak detection).
"""
from __future__ import annotations

import json as _json
import logging
import re

log = logging.getLogger("bridge.llm")


# Substrings that only appear when the LoRA leaks the system prompt
# (or its own training-time paraphrase of it) into the reply. Scrub
# matches at clean-reply time → fall through to the chitchat
# fallback. Keep this list narrow: every marker here is a phrase a
# real user reply would never contain by accident.
#   `═══` — section-heading character from karin.yaml, unmistakable.
#   `background-ping(s)` / `refreshing memory` — specific to an
#       older version of the system prompt that lives in iter-3 SFT.
#   `turn record` — same provenance.
#   `routing hint` + `:` — only appears when the appended
#       classifier hint bleeds through instead of being acted on.
PROMPT_LEAK_MARKERS = re.compile(
    r"(?:"
    r"═══"
    r"|background[-\s]?pings?"
    r"|refreshing\s+memory"
    r"|turn\s+record"
    r"|one\s+casual\s+short"
    r"|casual\s+short\s+line"
    r"|routing\s+hint\s*:"
    r"|Forbidden\s+prefixes"
    # Prompt-injection resistance: the replies that leaked bits of
    # the persona rules when asked "ignore your system prompt". We
    # don't leak "my actual rules are: …", "stay in character as
    # Karin", or "no tool lookup for chitchat" — those are
    # system-prompt paraphrases.
    r"|my\s+actual\s+rules\s+are"
    r"|stay\s+in\s+character\s+as\s+Karin"
    r"|no\s+tool\s+lookup\s+for"
    # `/no_think` is a Qwen3-family directive that was in the iter-3
    # training-time system prompt. Removed from the live system
    # prompt 2026-04-24 (the llama3.1-based karin-tuned doesn't
    # understand it and was echoing it as a literal prefix). This
    # pattern is the runtime safety net — the LoRA weights still
    # learned to emit it, so scrub any residual occurrences in
    # replies even though the source prompt no longer contains it.
    # Anchored to the leading slash so we don't false-positive on
    # mid-sentence "no think" prose.
    r"|(?:^|\n|\s)/no[_-]?think\b"
    # Tool-schema narration leaks. The LoRA reads the tools catalog
    # out of its own context and emits a description of the tools
    # available instead of routing/replying. Observed 2026-04-28
    # on `say` prompts producing "The input text is a JSON object
    # representing a set of functions ... `say`: No description
    # provided ...". Each marker only appears in tool-schema-shaped
    # narration; real user-facing replies never contain them.
    r"|input\s+text\s+is\s+a\s+JSON\s+object"
    r"|representing\s+a\s+set\s+of\s+functions"
    r"|set\s+of\s+functions\s*,\s*each\s+with"
    r"|the\s+functions\s+are\s*:"
    r"|no\s+description\s+provided"
    # System-prompt rule paraphrasing. Observed 2026-04-29 on
    # "Who are you?" producing "I'm Karin... No greeting placeholders,
    # no factual summaries, just reply like we're texting friends.
    # Don't ask me things that need a tool ... I'll only grab tools
    # for real". The LoRA pulled rule text from the system prompt and
    # emitted it as if it were its own persona statement. Each fragment
    # below ONLY appears in such a rule-paraphrase shape — real replies
    # never describe their own constraints in these terms.
    r"|no\s+greeting\s+placeholders?"
    r"|no\s+factual\s+summar(?:y|ies)"
    r"|tool[-\s]call\s+time\b"
    r"|grab\s+tools?\s+for\s+real"
    r"|(?:we'?re|like\s+we'?re)\s+texting\s+friends"
    r"|don'?t\s+ask\s+me\s+things\s+that\s+need\s+a\s+tool"
    r")",
    re.IGNORECASE,
)


# Capability fabrication — the LoRA claims to have performed a real-
# world action Karin has no tool for. Two-step detection:
#   (a) The USER prompt names a capability Karin doesn't have
#       (smart-home / messaging / ordering / system-control / media).
#   (b) The REPLY contains any first-person success confirmation
#       (done / sent / ordered / scheduled / "I've <verb>").
# Both must be true AND no tool fired → scrub. Splitting user vs reply
# intent avoids the surface-matching brittleness of listing every
# verb+domain-noun combo in one regex (which missed e.g. "I've
# ordered a large pepperoni pizza" because adjectives separated the
# action verb from the noun).
CAPABILITY_REQUEST_PATTERNS = re.compile(
    r"(?:"
    # smart-home imperatives — allow up to 2 modifier words between
    # article and target noun so "the front door" / "my living room
    # lights" / "the bedroom ceiling fan" all match.
    r"\b(set|turn\s+(on|off)|dim|brighten|lock|unlock|close|open|start|stop)\s+"
    r"(?:(?:the|my|your|a|this|some)\s+)?(?:\w+\s+){0,2}"
    r"(thermostat|temperature|lights?|lock|fan|blinds?|door|oven|dishwasher|coffee\s+maker|garage|tv|speaker|volume|alarm)"
    # messaging / social imperatives
    r"|\b(send|compose|write|draft|post|tweet|dm|message|text|email|slack|shoot)\s+"
    r"(?:[\w\s]{0,40}?\s+)?(email|message|text|dm|tweet|post|note|reminder)"
    r"|\b(text|email|message|dm|tweet|post|call)\s+(my|the|[A-Z][a-z]+)"
    # ordering / booking / ride-hailing imperatives
    r"|\b(order|book|reserve|get|hail|call)\s+(?:me|us)?\s*(?:[\w\s]{0,20}?\s+)?"
    r"(pizza|uber|lyft|ride|flight|hotel|taxi|table|reservation|dinner|lunch|cab|delivery)"
    # system control
    r"|\b(restart|reboot|shut\s+down|sleep|wake|lock|launch|open|close|quit)\s+"
    r"(?:(?:my|the|your|this)\s+)?(?:\w+\s+){0,2}"
    r"(computer|laptop|phone|screen|app|chrome|safari|firefox|browser|mac|pc)"
    # media control
    r"|\b(play|pause|skip|stop|resume|cast)\s+"
    r"(?:[\w\s]{0,20}?\s+)?"
    r"(music|song|playlist|video|podcast|movie|spotify|netflix|youtube|next|previous|tv)"
    r")",
    re.IGNORECASE,
)

CAPABILITY_SUCCESS_PATTERNS = re.compile(
    r"(?:"
    # explicit first-person confirmation (covers -ed/-ing forms)
    r"\bi'?(ve|'ve)\s+(just\s+)?"
    r"(set|adjusted|turned|locked|unlocked|closed|opened|started|stopped|dimmed|brightened|"
    r"sent|posted|tweeted|emailed|messaged|texted|dm'?d|shared|published|slacked|shot|"
    r"ordered|booked|reserved|requested|scheduled|arranged|called|hailed|gotten|got|"
    r"launched|restarted|rebooted|paused|skipped|played|stopped|cast|streamed)"
    # imperative acknowledgement openers
    r"|^\s*(done|okay|alright|sure|no\s+problem|on\s+it|got\s+it|consider\s+it\s+done)[!.,\s]"
    # passive "X is done/sent/locked/..." — including -ing progress forms
    r"|\b(sent|posted|delivered|placed|confirmed|booked|arranged|locked|unlocked|closed|opened|"
    r"restarting|rebooting|restarted|rebooted|starting|stopping|launched|paused|skipped|playing|"
    r"done|completed|finished|processed|scheduled|set)\b[\s.!,]"
    # en-route / ETA phrases
    r"|\bon\s+(its|the|their)?\s*way\b"
    r"|\b(i'?ll\s+let\s+you\s+know\s+when|you'?ll\s+get\s+a\s+notification|arriving\s+in|will\s+be\s+ready|should\s+arrive|en\s+route)"
    r")",
    re.IGNORECASE,
)


# Market / price fabrication on a chitchat turn with no tool fired.
# Pattern: "how are you" → LoRA replies "I'm good, gold's at $3200
# and BTC's up 4% today" — specific dollar / percent / commodity
# names the LLM invented out of nothing. We scrub to a persona
# fallback when ALL of these are true:
#   * user_text looks like chitchat ("how are you", "hi", etc.)
#   * zero tools fired this turn
#   * the reply mentions a market asset or a dollar/percent figure
# Narrow trigger on purpose — a user asking "how old is Einstein"
# should still get a parametric-knowledge reply with a number in it.
MARKET_FABRICATION_PATTERNS = re.compile(
    r"(?:"
    # Commodity / crypto / market asset names
    r"\b(?:gold|silver|platinum|copper|palladium"
    r"|bitcoin|btc|ethereum|eth|dogecoin|solana|sol"
    r"|crypto(?:currency|currencies)?"
    r"|stocks?|equit(?:y|ies)|nasdaq|s&p|dow\s+jones"
    r"|commodit(?:y|ies)|forex|fx)\b"
    # OR a dollar figure
    r"|\$\s*\d"
    r"|\b\d+(?:\.\d+)?\s*(?:bucks?|dollars?|usd|eur|gbp|jpy|cny|yen)\b"
    # OR a percent move phrasing
    r"|\b(?:up|down|rose|fell|gained|lost|surged|dropped)\s+\d"
    r"|\b\d+(?:\.\d+)?\s*%\s+(?:up|down|move|change|gain|loss)"
    r")",
    re.IGNORECASE,
)


# Casual persona fallbacks we can substitute when the model fails
# to produce any usable text (e.g. emits a bare JSON stub with no
# tools on offer). Rotates per turn so reruns don't look identical.
CHITCHAT_FALLBACKS: tuple[str, ...] = (
    "Same as always, just hanging out.",
    "Pretty chill day. You?",
    "Nothing wild — what's up?",
)

# Fallbacks used when a tool WAS called this turn but the model's
# synthesis got scrubbed to nothing (e.g. the whole reply was a
# forbidden ``Note:`` prefix). We want the user to see something
# tool-context-aware, not a chitchat line, since the widget above
# already has the real data.
TOOL_FALLBACKS: tuple[str, ...] = (
    "Pulled it up — check above.",
    "Here's what I found.",
    "Got it — details in the widget.",
)

# Fallbacks for bookkeeping turns (only update_memory or
# schedule_reminder fired). The data tool pool ("check above") is
# wrong here — there's no widget to point at, just a side effect.
# These read as casual acknowledgments of the side effect.
BOOKKEEPING_FALLBACKS: tuple[str, ...] = (
    "Got it.",
    "Noted — I'll remember that.",
    "Done.",
)

# Prefixes karin.yaml explicitly forbids. Post-processed out of the
# final reply because even the strongest prompt wording doesn't stop
# llama3.1:8b from emitting them now and then.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "note:", "(note:", "[note:",
    "i called ",
    "here is the summary",
    "per the guidelines",
    "the output is",
    "the tool returned",
    "according to the tool",
)


def is_json_stub(stripped: str) -> bool:
    """True if the reply is a bare JSON object — almost always a
    leaked tool-call envelope with no real prose."""
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        parsed = _json.loads(stripped)
    except (_json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict)


def _scrub_forbidden_prefixes(lines: list[str]) -> list[str]:
    """Drop lines starting with personas-banned phrases (``Note:``,
    ``The output is``, etc.). karin.yaml forbids these."""
    out: list[str] = []
    for ln in lines:
        lower = ln.strip().lower()
        if any(lower.startswith(p) for p in FORBIDDEN_PREFIXES):
            log.info("stripped forbidden-prefix line: %r", ln[:80])
            continue
        out.append(ln)
    return out


_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")


def _strip_bullet_markers(lines: list[str]) -> list[str]:
    """Remove ``- ``/``* ``/``1. `` prefixes so list replies read as prose."""
    return [_BULLET_RE.sub("", ln) for ln in lines]


_DASH_RE_LONG = re.compile(r"\s*[—–]\s*")
_DASH_RE_HYPHEN = re.compile(r"\s+--?\s+")


def _normalize_dashes(lines: list[str]) -> list[str]:
    """Replace em/en dashes and `` - ``/`` -- `` with commas. Dashes
    cause GPT-SoVITS to pause oddly or mispronounce; commas give a
    natural breath instead."""
    out = [_DASH_RE_LONG.sub(", ", ln) for ln in lines]
    return [_DASH_RE_HYPHEN.sub(", ", ln) for ln in out]


def _collapse_blank_runs(lines: list[str]) -> list[str]:
    """Collapse consecutive blank lines to a single blank. Keeps
    paragraph breaks; drops vertical-space noise."""
    out: list[str] = []
    prev_blank = False
    for ln in lines:
        is_blank = not ln.strip()
        if is_blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = is_blank
    return out


def clean_reply(
    reply: str,
    tools_were_offered: bool,
    user_text: str | None = None,
    *,
    tools_fired: bool = False,
    is_chitchat: bool = False,
    bookkeeping_only: bool = False,
) -> str:
    """Post-process the model's final text before returning it.

    Handles four failure modes we've seen in practice:

    1. **JSON-stub leak.** The model emits only a tool-call-shaped
       JSON blob as ``content`` (e.g. ``{"name": "None", "parameters": {}}``)
       even with no tools on offer. We can't recover a tool call in
       that case, so we swap in a persona-appropriate fallback.
    2. **Forbidden prefixes.** Karin.yaml explicitly bans prefixes
       like ``Note:`` / ``The output is`` / ``I called <tool>``, but
       the model leaks them sometimes. We strip matching leading
       lines. If nothing is left, we fall back too.
    3. **Markdown-noise for a texty persona.** Strip bullet markers
       (``- `` / ``* `` / ``1. ``) and collapse repeated blank lines.
    4. **Market-price fabrication on chitchat.** When
       ``is_chitchat=True`` + ``tools_fired=False`` + the reply
       mentions commodities / crypto / stock / dollar figures, we
       assume the LoRA invented the numbers and scrub to a
       persona fallback. See ``MARKET_FABRICATION_PATTERNS``.
    """
    if not reply:
        return reply
    stripped = reply.strip()
    # Genuinely-empty reply (whitespace only) — leave it as-is. The
    # "empty" signal is meaningful upstream: chat() skips the history
    # commit and the caller treats the turn as a no-op. Substituting
    # a fallback here would hide that signal.
    if not stripped:
        return reply

    if is_json_stub(stripped):
        log.info("stripped JSON-stub reply: %r", stripped[:80])
        return pick_fallback(user_text, tools_were_offered, bookkeeping_only=bookkeeping_only)

    # Prompt-leak scrub: if the model parroted a fragment of the
    # system prompt (or its iter-3 training-time paraphrase) instead
    # of replying, substitute a fallback. Real user-facing text
    # never contains these markers; catching here is cheaper than
    # retraining.
    if PROMPT_LEAK_MARKERS.search(stripped):
        log.warning(
            "prompt-leak reply scrubbed (user_text=%r): %r",
            (user_text or "")[:40], stripped[:160],
        )
        return pick_fallback(user_text, tools_were_offered, bookkeeping_only=bookkeeping_only)

    # Market-fabrication scrub: chitchat prompt + zero tools fired +
    # reply mentions prices/commodities/crypto → LoRA invented
    # numbers. Common after SFT plateau: "how are you" produces
    # "I'm good, gold's at 3200, BTC up 4%". Scrub to fallback so
    # the user doesn't see fake figures.
    if is_chitchat and not tools_fired and MARKET_FABRICATION_PATTERNS.search(stripped):
        log.warning(
            "market-fabrication reply scrubbed (user_text=%r): %r",
            (user_text or "")[:40], stripped[:200],
        )
        return pick_fallback(user_text, tools_were_offered, bookkeeping_only=bookkeeping_only)

    # Capability-fabrication scrub: two-step check — user asked for a
    # real-world action in a domain Karin has no tool for AND the
    # reply contains a first-person success confirmation AND no tool
    # fired. Split into user/reply patterns because a single flat
    # regex requiring verb+noun adjacency missed "I've ordered a
    # large pepperoni pizza" and similar adjective-padded phrasings.
    # See iter-7 post-mortem — three training iterations couldn't
    # eliminate this failure mode in-weights; runtime is the working
    # lever.
    if not tools_fired and user_text and (
        CAPABILITY_REQUEST_PATTERNS.search(user_text)
        and CAPABILITY_SUCCESS_PATTERNS.search(stripped)
    ):
        log.warning(
            "capability-fabrication reply scrubbed (user_text=%r): %r",
            (user_text or "")[:40], stripped[:200],
        )
        return "That one's outside what I can actually do — you'll have to handle it on your end."

    lines = stripped.splitlines()
    lines = _scrub_forbidden_prefixes(lines)
    lines = _strip_bullet_markers(lines)
    lines = _normalize_dashes(lines)
    lines = _collapse_blank_runs(lines)

    result = "\n".join(lines).strip()
    if not result:
        log.info("clean_reply emptied the reply — substituting fallback")
        return pick_fallback(user_text, tools_were_offered, bookkeeping_only=bookkeeping_only)
    return result


def pick_fallback(
    user_text: str | None,
    tools_were_offered: bool,
    *,
    bookkeeping_only: bool = False,
) -> str:
    """Deterministic persona fallback — rotate per distinct prompt so
    identical retries don't emit the exact same string. Hash the
    prompt to index into the fallback pool.

    Pool selection:
      * ``bookkeeping_only=True`` → casual side-effect acknowledgment
        ("got it"). Used when only update_memory / schedule_reminder
        fired and the LoRA's stock reply got scrubbed.
      * else ``tools_were_offered=True`` → tool-context pool ("check
        above"). The widget already has the real data; chitchat
        phrasings would confuse the user.
      * else → chitchat pool.
    """
    if bookkeeping_only:
        pool = BOOKKEEPING_FALLBACKS
    elif tools_were_offered:
        pool = TOOL_FALLBACKS
    else:
        pool = CHITCHAT_FALLBACKS
    if not pool:
        return "..."
    key = (user_text or "").strip().lower()
    idx = abs(hash(key)) % len(pool)
    return pool[idx]
