"""Two-phase compose path (L7 in the routing pipeline).

The default chat loop calls the LLM one final time at the end of a
tool turn with the FULL conversation in context (system prompt + tool
schema + coaching + multi-turn messages + tool results). On a 4-8B
quantized LoRA, that context routinely produces:

  * schema-catalog narration ("the functions available are …")
  * disavowal of its own tool call ("I'm just an AI, I can't actually
    look that up")
  * meta-commentary on the JSON it sees ("you're providing me a JSON
    object that …")
  * persona slip ("As KaraoKe-chan, I …")

Two-phase compose replaces that final call with a focused one: a fresh
system prompt, no tool schema, no coaching, just ``user asked X →
tools looked up Y → write a reply``. Eliminates each failure class by
construction. See ``docs/routing-pipeline.md § L7`` for the rationale.

Public surface used by ``bridge.llm.OllamaLLM``:
  * ``compose_from_tools(client, user_text, tool_record)``
  * ``compose_no_tool(client, user_text, history)``
  * ``output_ok(composed)`` — sanity check for either path's output
  * ``NONCE_SCRUB`` — exposed because the chat loop occasionally
    pre-scrubs tool outputs before they reach the compose call.

The ``client`` argument is duck-typed — anything exposing
``model: str``, ``_options: dict``, ``_think: bool | None``,
``_post_chat(body) -> httpx.Response`` works. In practice that's
always an ``OllamaLLM`` instance; passing it as an argument keeps this
module from importing the chat module (avoids the dependency cycle and
makes the seam explicit).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Protocol

log = logging.getLogger("bridge.llm")


class _ComposeClient(Protocol):
    """Minimal duck-type the compose path needs from its caller."""

    model: str

    @property
    def _options(self) -> dict[str, Any]: ...

    @property
    def _think(self) -> bool | None: ...

    def _post_chat(self, body: dict) -> Any: ...


# Regex used by ``compose_from_tools`` to strip the stub's nonce
# sentinel (and any stray ``NONCE_`` token bleeding through from a
# real upstream response) before the compose LLM sees the tool
# output. Without this, the LoRA faithfully echoes the nonce into the
# user-facing reply.
NONCE_SCRUB = re.compile(r"\s*\[?NONCE_[0-9a-f]+\]?", re.IGNORECASE)


# Sanity-check patterns — if compose output matches any of these, we
# discard it and fall back to the stock reply. Three failure classes
# each branch corresponds to:
#   (a) schema / meta leaks  — describing the JSON instead of replying
#   (b) placeholder leaks    — "X.XX" etc. slipping through
#   (c) compound fabrication — specific patterns observed on Step 2
#       where compose invented the missing half of an A+B query
COMPOSE_REJECT_PATTERNS: list[re.Pattern[str]] = [
    # --- (a) schema / meta leaks ---
    re.compile(r"this\s+is\s+(?:a\s+)?json\s+(?:object|description|schema)", re.IGNORECASE),
    re.compile(r"you['\s]re\s+(?:providing|describing)\s+(?:a\s+)?json", re.IGNORECASE),
    re.compile(r"the\s+functions?\s+include", re.IGNORECASE),
    re.compile(r"here['\s]s\s+a\s+breakdown\s+of\s+(?:the|each)\s+function", re.IGNORECASE),
    re.compile(r"^\s*\{[\s\S]*?\"name\"\s*:", re.IGNORECASE),  # raw JSON
    re.compile(r"^\s*name:\s*\w+\s*\n", re.IGNORECASE),         # raw YAML
    # --- (b) placeholder leaks ---
    # Compose slipped through "X.XX" or "<number>" instead of a real
    # value — seen on gold-vs-USD/JPY compound cases.
    re.compile(r"\bis\s+at\s+X\.XX\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]+/[A-Z]+\s+(?:is\s+)?(?:at\s+)?X\.XX\b"),
    re.compile(r"<(?:number|value|amount|placeholder)>", re.IGNORECASE),
    # --- (c) compound-fabrication (observed on Step 2 regressions) ---
    # Fabricated commodity prices on get_news ("gold up $1,933.50/oz").
    # The topic-aware stubs never return this shape; if the compose
    # output has it, the LoRA invented it. Kept narrow — requires a
    # commodity keyword + action verb + dollar-with-cents to avoid
    # flagging legitimate compose outputs that happen to mention $.
    re.compile(
        r"\b(?:gold|silver|bitcoin|btc|oil|crude|platinum)[\w'\s]*?"
        r"(?:at|up|down|trading|currently|going\s+for)\s+\$[\d,]+\.\d{2}",
        re.IGNORECASE,
    ),
    # Invented "per ounce" / "per share" pricing when the stub didn't
    # return a per-unit price. The topic-stubs quote values as plain
    # numbers; a reply that adds "per troy ounce" etc is inventing.
    re.compile(
        r"\$[\d,]+\.\d{2}\s+per\s+(?:troy\s+)?(?:ounce|share|gram)",
        re.IGNORECASE,
    ),
]


def output_ok(composed: str) -> bool:
    """True iff the compose LLM produced something usable, not a
    new schema/meta leak. Used after both compose paths as a
    second-layer sanity check before we commit to the compose output.
    """
    if not composed or not composed.strip():
        return False
    for pat in COMPOSE_REJECT_PATTERNS:
        if pat.search(composed):
            log.info(
                "compose output rejected by sanity check (pattern %r): %r",
                pat.pattern[:60], composed[:120],
            )
            return False
    return True


# Per-tool compose guidance. Only the suffixes for tools that
# actually fired this turn are appended to the compose system
# prompt — keeps the prompt short while nudging the LoRA toward
# the tool-appropriate reply shape. Targeted at the failure modes
# observed in the Step 2 manual judge (schema_leak is addressed by
# the base compose prompt; these cover tool-specific bugs: wrong-
# commodity on tracker, fabricated headlines on get_news, invented
# digest items, refuse-after-fire on convert/math/graph/circuit,
# nonce leak on schedule_reminder/update_memory, meta-commentary on
# update_memory instead of confirming the saved fact).
TOOL_COMPOSE_SUFFIX: dict[str, str] = {
    "get_weather": (
        "Lead with the exact temperature and condition from the "
        "tool output. End with a short practical cue (jacket / "
        "sunscreen / stay hydrated). Don't re-interpret the "
        "temperature scale; if the tool said '42°F' it's cold, "
        "not warm."
    ),
    "get_news": (
        "Quote the ACTUAL headline from the tool. Do not invent "
        "other headlines, mention commodity prices, or reference "
        "unrelated topics that weren't in the tool output."
    ),
    "get_digest": (
        "List EXACTLY the items the tool returned. Do not invent "
        "specifics (email senders, city names, meeting topics) "
        "that aren't in the tool output."
    ),
    "tracker": (
        "Quote the exact value AND asset identifier from the tool. "
        "Do not substitute a different commodity (asked about "
        "USD/JPY → don't say 'gold'). If only one asset was "
        "queried, say so honestly."
    ),
    "convert": (
        "Give the converted value directly in format 'X <unit> ≈ "
        "Y <unit>'. Never say 'we already called convert' or "
        "refuse to quote the result. Never substitute 'gold' "
        "for a unit the user asked about."
    ),
    "math": (
        "Give the numeric result the tool returned. Do not disavow "
        "('I'm not a calculator', 'I'm just an AI') — the tool "
        "already computed it; your job is to quote the answer."
    ),
    "graph": (
        "Describe what was plotted based on the tool's output "
        "(data points, y-range, shape). Do not disavow being a "
        "graphing calculator — the tool already produced the plot."
    ),
    "circuit": (
        "Quote the tool's computed value with units. Do not "
        "disavow being an electrical engineer — the tool did the "
        "calculation."
    ),
    "schedule_reminder": (
        "Confirm the reminder was set, stating the message + time "
        "in natural language (not ISO timestamps). NEVER echo the "
        "NONCE_xxx token from the tool output — that's internal."
    ),
    "update_memory": (
        "Confirm the ACTUAL fact that was saved — repeat it back "
        "naturally. Do NOT describe the save as 'nonce value', "
        "'note key', 'metadata', or similar internal terminology. "
        "Example: user saves 'allergic to peanuts' → reply 'Got "
        "it, no peanuts' (NOT 'I've been saved a nonce value')."
    ),
}


_COMPOSE_FROM_TOOLS_SYSTEM = (
    "You are Karin, a casual voice assistant. The user just asked "
    "something and tools have already looked up the answer for you. "
    "Write a single short reply to the user using ONLY the tool "
    "output below.\n\n"
    "Rules:\n"
    "- 1-2 sentences. Sound like a friend, not a bot.\n"
    "- Use the tool's actual numbers/facts. Do not invent data the "
    "tool didn't return.\n"
    "- If the user's question had TWO parts and the tool only "
    "covered one, say so honestly ('didn't get the second bit, "
    "but here's the first'). Never fabricate the missing part — "
    "no made-up headlines, exchange rates, times, city names, or "
    "product details.\n"
    "- If the tool output doesn't match the user's actual question "
    "(e.g. they asked about X, the tool returned Y), acknowledge "
    "the mismatch instead of pretending Y answers X.\n"
    "- No meta-commentary: no 'you asked', 'the tool returned', "
    "'this is a JSON', 'according to the tool'.\n"
    "- No schema descriptions or function listings.\n"
    "- Never echo internal tokens like NONCE_xxx.\n"
    "- No disclaimers like 'I'm just an AI' / 'not a calculator'.\n"
    "- Just answer the user directly, in character."
)

_COMPOSE_NO_TOOL_SYSTEM = (
    "You are Karin, a casual voice assistant. The user just said "
    "something that doesn't need a tool lookup — chitchat, an "
    "idiom or figurative phrase, a greeting, a farewell, or a "
    "question you can answer from general knowledge.\n\n"
    "Rules:\n"
    "- Reply in 1-2 natural sentences, like a friend. Not a bot.\n"
    "- Stay in character as Karin. Your name is Karin. You are "
    "not a 'large language model', 'Japanese vocaloid', "
    "'KaraoKe-chan', or any other identity. Never introduce "
    "yourself with a different name.\n"
    "- Never describe the tools/schema/functions available to "
    "you. Never list function names (convert/schedule_reminder/"
    "wiki/etc.).\n"
    "- Never narrate the prompt back to the user ('You've "
    "asked...', 'You're providing...', 'You didn't specify...').\n"
    "- Never say 'details in the widget' or 'pulled it up — "
    "check above' on a no-tool turn. Nothing was pulled up.\n"
    "- If the user asks for a real action that needs a tool but "
    "none fired, briefly say what you'd need (e.g. 'give me a "
    "city and I'll check') instead of dumping function syntax.\n"
    "- No disclaimers about being an AI unless directly relevant."
)


def _build_body(
    client: "_ComposeClient",
    messages: list[dict],
) -> dict:
    """Shared body-shaping for both compose paths.

    Forces ``temperature: 0.3`` regardless of the client's default —
    earlier experiments at 0.4 / 0.55 each regressed >5 cases on the
    eval, so 0.3 is the conservative pin. ``stream: False`` because
    compose emits one short reply, not a token stream. No ``tools``
    key — that's the whole point of the path.
    """
    body: dict = {
        "model": client.model,
        "messages": messages,
        "stream": False,
        "options": {**client._options, "temperature": 0.3},
        "keep_alive": -1,
    }
    if client._think is not None:
        body["think"] = client._think
    return body


def compose_from_tools(
    client: "_ComposeClient",
    user_text: str | None,
    tool_record: list[dict],
) -> str:
    """Two-phase compose: ask the LLM for a user-facing reply given
    ONLY the user's prompt and the tool outputs.

    Returns empty string on failure — caller should fall back to the
    stock reply path.
    """
    if not tool_record:
        return ""

    lines: list[str] = []
    for rec in tool_record:
        name = rec.get("name") or "?"
        args = rec.get("args") or {}
        result = rec.get("result") or ""
        scrubbed = NONCE_SCRUB.sub("", str(result)).strip()
        # Keep the arg dict compact — the compose model doesn't
        # need the full JSON schema, just the substantive values.
        if isinstance(args, dict) and args:
            args_str = ", ".join(
                f"{k}={v!r}" for k, v in args.items() if v not in (None, "")
            )
        else:
            args_str = ""
        header = f"{name}({args_str})" if args_str else name
        lines.append(f"- {header} → {scrubbed}")
    tool_summary = "\n".join(lines) or "(no tools called)"

    compose_system = _COMPOSE_FROM_TOOLS_SYSTEM
    # Append per-tool guidance for the tool(s) that actually fired
    # this turn. Only suffixes for fired tools get added to keep the
    # prompt short.
    fired_tools = {rec.get("name") for rec in tool_record if rec.get("name")}
    tool_suffixes = [
        (name, TOOL_COMPOSE_SUFFIX[name])
        for name in fired_tools
        if name in TOOL_COMPOSE_SUFFIX
    ]
    if tool_suffixes:
        compose_system += "\n\nTool-specific guidance for this turn:"
        for name, suffix in tool_suffixes:
            compose_system += f"\n- **{name}**: {suffix}"
    compose_user = (
        f"User's question: {user_text or ''}\n\n"
        f"Tool result(s):\n{tool_summary}\n\n"
        "Your reply (to the user, in character, using the data above):"
    )

    body = _build_body(
        client,
        [
            {"role": "system", "content": compose_system},
            {"role": "user", "content": compose_user},
        ],
    )
    try:
        resp = client._post_chat(body)
        data = resp.json()
        composed = (data.get("message") or {}).get("content", "") or ""
    except Exception as e:
        log.warning("two-phase compose failed: %s — falling back", e)
        return ""
    log.info(
        "two-phase compose: %d tools, reply=%r",
        len(tool_record), composed[:120],
    )
    return composed.strip()


def compose_no_tool(
    client: "_ComposeClient",
    user_text: str | None,
    history: list[dict],
) -> str:
    """Casual-reply compose for turns where NO tool fired.

    Mirror of :func:`compose_from_tools` for the no-tool case. The
    stock no-tool reply comes from an LLM call with the full tool
    schema + coaching still in context, and the LoRA often responds
    by describing the schema back to the user, slipping into the
    wrong persona, or leaking widget-template phrases. This path
    asks the LLM the same question with a fresh tool-free context.

    Returns empty string to signal "fall back" when the compose LLM
    call fails.
    """
    # Pass the last couple of turns so the compose LLM has enough
    # context to handle continuations ("how about Osaka?" after
    # "weather in Tokyo") without the full tool catalog noise.
    recent: list[dict] = []
    pair_budget = 4  # up to 4 messages = ~2 user/assistant pairs
    for m in reversed(list(history)):
        role = m.get("role")
        if role in ("tool",):
            continue
        if role == "assistant":
            content = m.get("content") or ""
            if not content.strip():
                continue
        if role in ("user", "assistant"):
            recent.append({"role": role, "content": m.get("content") or ""})
            if len(recent) >= pair_budget:
                break
    recent.reverse()

    messages: list[dict] = [{"role": "system", "content": _COMPOSE_NO_TOOL_SYSTEM}]
    messages.extend(recent)
    # recent may already end in the current user turn if the caller
    # committed it; dedupe by appending user_text only if it's not
    # already the last message.
    if not recent or recent[-1].get("role") != "user" or (
        recent[-1].get("content") or "") != (user_text or ""):
        messages.append({"role": "user", "content": user_text or ""})

    body = _build_body(client, messages)
    try:
        resp = client._post_chat(body)
        composed = (resp.json().get("message") or {}).get("content", "") or ""
    except Exception as e:
        log.warning("no-tool compose failed: %s — falling back", e)
        return ""
    log.info("no-tool compose: reply=%r", composed[:120])
    return composed.strip()
