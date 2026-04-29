"""LLM transport backends — Ollama / llama.cpp / MLC-LLM.

Each backend speaks a slightly different wire format. Karin's chat
loop is written against Ollama's ``/api/chat`` shape (top-level
``message`` field with ``content`` + structured ``tool_calls``); the
other two backends are translated in/out at this seam so the loop is
unchanged.

Public surface used by ``bridge.llm.OllamaLLM``:
  * ``post_ollama(client, body, retry)`` — POST /api/chat with one
    transient-5xx retry. Returns the raw ``httpx.Response``.
  * ``post_llamacpp(client, body, retry)`` — translates to OpenAI
    chat-completions, POSTs, translates back to Ollama shape.
  * ``post_mlc(client, body, retry)`` — same idea but with MLC-LLM's
    additional quirks (no assistant ``tool_calls``, requires
    coaching injection on the last user message).

The ``Retry`` dataclass packages the three knobs the caller controls
(status codes that retry, delay between attempts, max retries) so the
backend signatures stay short.

Why a separate module: the three implementations share a duck-typed
``_Translated`` response wrapper, the retry loop, and the OpenAI-style
body translation, but ``bridge.llm.OllamaLLM`` was carrying ~280 lines
of transport logic that has nothing to do with the chat loop's
control flow. Pulling it out shrinks the chat module and makes adding
a fourth backend (e.g. vLLM) a single-file change.
"""
from __future__ import annotations

import json as _json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger("bridge.llm")


@dataclass(frozen=True)
class Retry:
    """Retry policy shared by every backend.

    On Jetson the LLM runner can crash mid-load when sovits is
    restarting / actively synthesizing — the next moment the runner
    is fine. A single short-delayed retry recovers from this without
    surfacing the error to the user. ``max_retries=1`` means up to
    two total attempts per logical request.
    """

    status_codes: tuple[int, ...] = (500, 502, 503)
    delay_s: float = 3.0
    max_retries: int = 1


DEFAULT_RETRY = Retry()


class _Translated:
    """Duck-types ``httpx.Response`` for callers that only use
    ``.json()`` / ``.raise_for_status()``. Used by the OpenAI-compat
    backends after they translate the response shape back into
    Ollama's ``{"message": ...}`` envelope.
    """

    def __init__(self, d: dict) -> None:
        self._d = d

    def json(self) -> dict:
        return self._d

    def raise_for_status(self) -> None:
        return None


def _post_with_retry(
    client: httpx.Client,
    path: str,
    json_body: dict,
    retry: Retry,
    *,
    backend_label: str,
) -> httpx.Response:
    """Run ``client.post(path, json=json_body)`` with the configured
    retry policy. Returns the final response — caller decides whether
    to ``.raise_for_status()`` (some backends salvage 5xx bodies)."""
    attempt = 0
    while True:
        resp = client.post(path, json=json_body)
        if resp.status_code not in retry.status_codes or attempt >= retry.max_retries:
            return resp
        log.warning(
            "%s %d on %s — retrying in %.1fs (attempt %d/%d)",
            backend_label, resp.status_code, path,
            retry.delay_s, attempt + 1, retry.max_retries,
        )
        attempt += 1
        time.sleep(retry.delay_s)


# --- Ollama --------------------------------------------------------------


def post_ollama(
    client: httpx.Client,
    body: dict,
    retry: Retry = DEFAULT_RETRY,
) -> httpx.Response:
    """POST a body to Ollama's /api/chat with one transient-5xx retry."""
    resp = _post_with_retry(client, "/api/chat", body, retry, backend_label="ollama")
    resp.raise_for_status()
    return resp


# --- llama.cpp / llama-server -------------------------------------------


_LLAMA_NO_TOOL_SHIM: dict = {
    "type": "function",
    "function": {
        "name": "no_tool",
        "description": "Emit when no other tool is applicable.",
        "parameters": {"type": "object", "properties": {}},
    },
}

_BARE_NAME_FIX_RE = re.compile(
    r'("(?:name|function)"\s*:\s*)([A-Za-z_][A-Za-z0-9_]*)',
)
_LLAMACPP_PARSE_ERROR_RE = re.compile(
    r"Failed to parse input at pos \d+:\s*(.+)$", re.DOTALL,
)


def _ollama_to_openai_body(body: dict, *, inject_no_tool_shim: bool) -> dict:
    """Body translation: Ollama → OpenAI chat-completions.

    Drops ``keep_alive`` / ``think`` / ``options`` (llama-server doesn't
    accept them; sampling params that matter are set at server start).
    Lifts ``temperature`` / ``top_p`` / ``top_k`` / ``seed`` from
    options to top-level. Maps ``num_predict`` → ``max_tokens``.
    """
    openai_body: dict = {
        "model": body["model"],
        "messages": body["messages"],
        "stream": False,
    }
    if body.get("tools"):
        if inject_no_tool_shim:
            # Shim a synthetic `no_tool` function so llama-server's
            # strict tool-call parser accepts the LoRA's trained
            # no-tool signal (`{"name":"no_tool","parameters":{}}`).
            # Without this, prompts like "hi" hit HTTP 500 "Failed to
            # parse input at pos 0". Filtered back out in the
            # response translation below.
            openai_body["tools"] = list(body["tools"]) + [_LLAMA_NO_TOOL_SHIM]
        else:
            openai_body["tools"] = list(body["tools"])
    opts = body.get("options") or {}
    for k in ("temperature", "top_p", "top_k", "seed"):
        if k in opts:
            openai_body[k] = opts[k]
    if "num_predict" in opts:
        openai_body["max_tokens"] = opts["num_predict"]
    return openai_body


def post_llamacpp(
    client: httpx.Client,
    body: dict,
    retry: Retry = DEFAULT_RETRY,
) -> _Translated:
    """llama-server's /v1/chat/completions path with response
    translation back to Ollama's ``{"message": ...}`` shape.

    Salvages 500 "Failed to parse input" errors: llama-server's strict
    tool-call parser rejects any model output that isn't clean JSON
    with a known tool name. The LoRA sometimes emits malformed
    variants like ``{"name": no_tool, "parameters": {}}`` (unquoted
    value). Rather than let the whole request fail, the leaked blob
    is extracted from the error message and handed to the caller as
    ``message.content`` so the chat loop's leak-recovery path can
    salvage it.
    """
    openai_body = _ollama_to_openai_body(body, inject_no_tool_shim=True)
    resp = _post_with_retry(
        client, "/v1/chat/completions", openai_body, retry,
        backend_label="llama-server",
    )

    if resp.status_code == 500:
        try:
            err = resp.json().get("error", {}).get("message", "") or ""
        except Exception:
            err = resp.text or ""
        m = _LLAMACPP_PARSE_ERROR_RE.search(err)
        if m:
            leaked = m.group(1).strip()
            # Best-effort: fix unquoted bare identifiers after `:` like
            # ``"name": no_tool`` → ``"name": "no_tool"``.
            leaked_fixed = _BARE_NAME_FIX_RE.sub(r'\1"\2"', leaked)
            log.info(
                "llama-server 500 salvage: leaked=%r (fixed=%r)",
                leaked[:160], leaked_fixed[:160],
            )
            return _Translated({"message": {
                "role": "assistant",
                "content": leaked_fixed,
            }})

    resp.raise_for_status()

    data = resp.json()
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"llama-server returned unexpected shape: {data!r}"
        ) from e
    # llama-server sometimes omits "type" and "id" from tool_calls it
    # emits. If we append the message verbatim to history and send it
    # back, the next request fails with 'Failed to parse messages:
    # Missing tool call type: ...'. Fill in the OpenAI required
    # fields so round-tripping works. Also parse `arguments` from JSON
    # string → dict so Karin's tool executor (which expects a dict)
    # can dispatch correctly.
    tcs = msg.get("tool_calls") or []
    # Filter out the no_tool shim. A no_tool call is the LoRA's way of
    # saying "no tool needed" — downstream code expects that as
    # tool_calls=None, content=''.
    tcs = [
        tc for tc in tcs
        if (tc.get("function") or {}).get("name") != "no_tool"
    ]
    if tcs:
        msg["tool_calls"] = tcs
    else:
        msg.pop("tool_calls", None)
        if not msg.get("content"):
            msg["content"] = ""
    for i, tc in enumerate(tcs):
        tc.setdefault("type", "function")
        tc.setdefault("id", f"call_{i}")
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = _json.loads(args)
            except Exception:
                pass

    return _Translated({"message": msg})


# --- MLC-LLM ------------------------------------------------------------


_MLC_COACHING_TEMPLATE = (
    "If a tool call is genuinely needed to answer, "
    "respond with a single-line JSON stub in the format "
    "{{\"name\": function_name, \"parameters\": dict_of_args_and_values}}. "
    "If no tool fits (chitchat, acknowledgments, figurative phrases, "
    "questions already answerable from context), reply in plain "
    "text like normal. Do not emit a JSON stub unless a tool "
    "really applies.\n\n"
    "Tools: {tool_list}\n\n"
)


def _mlc_translate_messages(messages_in: list[dict]) -> list[dict]:
    """Flatten assistant turns with structured ``tool_calls`` to a
    JSON-stub content shape, and drop ``tool_call_id`` from tool
    messages. MLC errors 400 on either of those when present, so we
    pre-translate the chat history into shapes it accepts.
    """
    out: list[dict] = []
    for m in messages_in:
        m = dict(m)
        tcs = m.get("tool_calls")
        if m.get("role") == "assistant" and tcs:
            rendered = []
            for tc in tcs:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = _json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {}
                rendered.append(
                    _json.dumps({"name": name, "parameters": args or {}})
                )
            content = m.get("content") or ""
            if rendered:
                joined = "\n".join(rendered)
                content = joined if not content else content + "\n" + joined
            m = {"role": "assistant", "content": content}
        elif m.get("role") == "tool":
            m.pop("tool_call_id", None)
        out.append(m)
    return out


def _mlc_inject_coaching(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Prefix the last user message with Ollama-style coaching so the
    LoRA sees the same nudge it was trained under. MLC's
    ``llama-3_1`` conv_template defaults to ``use_function_calling
    = false`` and doesn't inject this on its own.
    """
    if not tools or not messages:
        return messages
    tool_names = [
        (t.get("function") or {}).get("name")
        for t in tools
        if isinstance(t, dict)
    ]
    tool_names = [n for n in tool_names if n]
    if not tool_names:
        return messages
    coaching = _MLC_COACHING_TEMPLATE.format(tool_list=", ".join(tool_names))
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {**out[i], "content": coaching + (out[i].get("content") or "")}
            break
    return out


def post_mlc(
    client: httpx.Client,
    body: dict,
    retry: Retry = DEFAULT_RETRY,
) -> _Translated:
    """MLC-LLM's /v1/chat/completions path.

    MLC's ``llama-3_1`` conv_template defaults to
    ``use_function_calling = false`` and does NOT inject a
    JSON-format coaching block or tool schemas. Left to its own
    devices, the LoRA returns natural-language replies on tool-worthy
    prompts. Fix: mirror Ollama's Modelfile TEMPLATE by prefixing the
    LAST user message with the same coaching sentence + terse
    comma-separated tool list. That restores the trained JSON
    emission shape — the model emits ``{"name": ..., "parameters":
    ...}`` in content, and the chat loop's leak-recovery path
    converts it back into a structured tool_call downstream.
    """
    messages_in = body.get("messages") or []
    tools = body.get("tools") or []

    messages = _mlc_translate_messages(messages_in)
    messages = _mlc_inject_coaching(messages, tools)

    openai_body: dict = {
        "model": body["model"],
        "messages": messages,
        "stream": False,
    }
    opts = body.get("options") or {}
    for k in ("temperature", "top_p", "seed"):
        if k in opts:
            openai_body[k] = opts[k]
    if "num_predict" in opts:
        openai_body["max_tokens"] = opts["num_predict"]

    resp = _post_with_retry(
        client, "/v1/chat/completions", openai_body, retry,
        backend_label="mlc",
    )
    resp.raise_for_status()

    data = resp.json()
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"mlc_llm serve returned unexpected shape: {data!r}"
        ) from e
    # MLC returns natural-language content or a JSON stub (our trained
    # format). Keep as content — the chat loop's leak-recovery path
    # parses the JSON stub into a structured tool_call.
    return _Translated({"message": {
        "role": msg.get("role", "assistant"),
        "content": msg.get("content") or "",
    }})


def post_chat(
    backend: str,
    client: httpx.Client,
    body: dict,
    retry: Retry = DEFAULT_RETRY,
) -> Any:
    """Single dispatch entry point. ``backend`` is one of ``ollama``,
    ``llamacpp``, ``mlc``. Returns either an ``httpx.Response`` (for
    Ollama) or a ``_Translated`` duck-type with the same .json() /
    .raise_for_status() interface (for OpenAI-compat backends).
    """
    if backend == "llamacpp":
        return post_llamacpp(client, body, retry)
    if backend == "mlc":
        return post_mlc(client, body, retry)
    return post_ollama(client, body, retry)
