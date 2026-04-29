"""Ollama-backed translator for the news subsystem.

Small, self-contained module. Does one thing: translate a short string
from one language to another using whatever model Ollama is serving.
Stateless (no rolling history) and strictly prompted so the model
returns just the translation — no "Sure, here you go:" preamble, no
quote marks, no bullet list.

Design rules:
  - Fail-soft. Any Ollama error -> TranslationResult(text=original,
    translated=False). Never raises. The news pipeline keeps flowing
    with the original language even if the LLM is unreachable.
  - Deterministic. temperature=0 + a rigid prompt shape so identical
    input produces identical output across calls (critical for cache
    correctness and repeatable tests).
  - Write-through disk cache. Each translation is hashed by
    sha1(source_text + '|' + target_lang + '|' + source_lang) so the
    same headline rendered twice doesn't hit the LLM twice.
  - Model-agnostic. Works with any Ollama model, but Qwen-family
    models produce the best EN<->ZH quality (training data bias).
    If you swap to Llama/Hermes, EN<->ZH suffers. Flagged here so
    future-you has warning.

Public API:
    Translator(base_url, model, cache_path, request_timeout)
    translator.translate(text, target_lang, source_lang="auto")
        -> TranslationResult(text, translated, from_cache)
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from bridge.utils import atomic_write_text

log = logging.getLogger("bridge.news.translate")


# --- result type ----------------------------------------------------------

@dataclass(frozen=True)
class TranslationResult:
    """Outcome of one translate() call.

    `translated=False` covers three paths: target == source (short-
    circuit, no LLM call), LLM error, and empty input. Callers that
    need to render "may be outdated" style caveats check this flag.

    `from_cache` is True when we served from the disk cache instead of
    calling the LLM. Useful for metrics / debug logging; never affects
    correctness.
    """

    text: str
    translated: bool
    from_cache: bool = False


# --- prompt ---------------------------------------------------------------

# Human-readable language names for the prompt body. Qwen behaves better
# when we say "Chinese" than "zh"; the model's trained to think in
# language names, not BCP-47 codes.
_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

# System prompt is blunt on purpose. Every extra sentence the model
# generates is latency we pay, and any wiggle room invites "Sure, here's
# the translation:" preambles or trailing commentary.
_SYSTEM_PROMPT = (
    "You are a professional translator. You translate the user's text "
    "accurately and fluently. You output ONLY the translation — no "
    "explanation, no quotes, no labels, no commentary, no Markdown. "
    "Preserve proper nouns. Preserve numbers and dates verbatim."
)

# Guards against the model answering the user's text instead of
# translating it (happens when the input reads like a question).
# "Do not answer" is the key phrase that flips Qwen from instruction-
# following to translation-mode on ambiguous inputs.
_USER_TEMPLATE = (
    "Translate the following text from {src} to {tgt}. "
    "Do not answer the text; translate it.\n\n"
    "TEXT:\n{text}"
)


def _build_messages(text: str, source_lang: str, target_lang: str) -> list[dict]:
    src_name = _LANG_NAMES.get(source_lang, source_lang)
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(
            src=src_name, tgt=tgt_name, text=text,
        )},
    ]


# --- cache ----------------------------------------------------------------

def _cache_key(text: str, target_lang: str, source_lang: str) -> str:
    """sha1-derived cache key. Includes both langs so a text reused
    with a different source-language hint doesn't hit a stale entry."""
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    h.update(b"|")
    h.update(target_lang.encode("utf-8"))
    h.update(b"|")
    h.update(source_lang.encode("utf-8"))
    return h.hexdigest()


# --- output cleanup -------------------------------------------------------

# Prefixes Qwen occasionally adds despite the "ONLY the translation"
# rule. Checked case-insensitively; stripped with everything up to and
# including the first colon+space so we keep the translation intact.
_STRIP_PREFIXES: tuple[str, ...] = (
    "translation:",
    "here is the translation:",
    "here's the translation:",
    "translated:",
    "sure, here is the translation:",
    "sure, here's the translation:",
)


# Quote pairs the model sometimes wraps its output in. Map each opening
# quote to its matching closing quote so we strip both straight (") and
# typographic ("...") pairs. Single quotes included for completeness —
# models occasionally use them around short phrases.
_QUOTE_PAIRS: dict[str, str] = {
    "\"": "\"",
    "'": "'",
    "\u201c": "\u201d",   # left / right double quotation mark
    "\u2018": "\u2019",   # left / right single quotation mark
    "\u300c": "\u300d",   # CJK corner brackets, used as quotes in JP
    "\u300e": "\u300f",   # CJK white corner brackets
}


def _clean_output(raw: str) -> str:
    """Trim whitespace + strip common model prefix/suffix noise.

    Defensive; the prompt is usually enough on Qwen but weaker models
    drift. Keep the list short — aggressive stripping risks eating
    legitimate content that starts with "Translation".
    """
    text = raw.strip()
    # Drop a surrounding pair of quotes if the whole output is wrapped.
    # Handles both straight ("...") and typographic ("...") pairs as
    # well as CJK corner brackets which JP/ZH models occasionally emit.
    if len(text) >= 2:
        close_match = _QUOTE_PAIRS.get(text[0])
        if close_match is not None and text[-1] == close_match:
            text = text[1:-1].strip()
    lower = text.lower()
    for prefix in _STRIP_PREFIXES:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text


# --- translator class -----------------------------------------------------

class Translator:
    """Ollama-backed translator with a disk-backed result cache.

    Thread-safety: a single instance is safe for concurrent reads but
    writes to the cache are serialized by a lock. The news service
    doesn't (currently) call translate() concurrently, but the web path
    might, so we guard it.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        cache_path: Path,
        request_timeout: float = 60.0,
        num_predict: int = 256,
    ) -> None:
        """Configure the translator.

        Args:
            base_url: Ollama server base URL, e.g. ``http://localhost:11434``.
            model: Ollama model tag. Qwen-family recommended for EN<->ZH.
            cache_path: Path to the JSON cache file. Parent dir is
                created on first write. File is loaded lazily.
            request_timeout: HTTP timeout per Ollama call, in seconds.
                Translations are short so 60s is generous; picks up
                cold-model-load delays without hanging forever.
            num_predict: Upper bound on generated tokens per call.
                Translations are typically <100 tokens; cap guards
                against runaway generation if the model ignores the
                prompt and starts producing an essay.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache_path = Path(cache_path)
        self.request_timeout = float(request_timeout)
        self.num_predict = int(num_predict)
        self._cache: dict[str, str] | None = None
        self._cache_lock = threading.Lock()

    # --- cache I/O ----------------------------------------------------------

    def _load_cache(self) -> dict[str, str]:
        """Read the cache file once, then hold the dict in memory. Missing
        file -> empty dict; malformed file -> empty dict + warning (better
        than crashing the news path on a corrupted cache)."""
        if self._cache is not None:
            return self._cache
        with self._cache_lock:
            if self._cache is not None:
                return self._cache
            if not self.cache_path.exists():
                self._cache = {}
                return self._cache
            try:
                loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("not a JSON object")
                self._cache = {str(k): str(v) for k, v in loaded.items()}
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(
                    "translation cache at %s unreadable (%s); starting fresh",
                    self.cache_path, e,
                )
                self._cache = {}
            return self._cache

    def _save_cache(self) -> None:
        """Write the whole cache to disk. V1 writes the full dict on
        each save — fine at our volume (a few thousand entries).
        Revisit when/if the cache grows past ~MB."""
        if self._cache is None:
            return
        atomic_write_text(
            self.cache_path,
            json.dumps(self._cache, ensure_ascii=False, indent=2) + "\n",
        )

    # --- HTTP call ----------------------------------------------------------

    def _call_ollama(self, messages: list[dict]) -> str:
        """Post to /api/chat and return the assistant message content.

        Raises httpx.HTTPError on transport failures; translate() catches
        and degrades. Keeps this method focused — it's the boundary with
        the external service.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # think is a TOP-LEVEL field in Ollama's /api/chat, not an
            # option (memory: Ollama think field is top-level not in
            # options). Forcing False here is critical for Qwen3/Qwen3.5
            # — otherwise the model emits 700+ thinking tokens and turns
            # a 2s translation into 30s.
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 1024,
                "num_predict": self.num_predict,
            },
        }
        with httpx.Client(base_url=self.base_url, timeout=self.request_timeout) as client:
            resp = client.post("/api/chat", json=body)
            resp.raise_for_status()
            payload = resp.json()
        msg = payload.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Ollama returned empty content: {payload!r}")
        return content

    # --- public API ---------------------------------------------------------

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> TranslationResult:
        """Translate ``text`` into ``target_lang``.

        Short-circuits (no LLM call) when:
          - text is empty / whitespace-only
          - source_lang == target_lang (already in target language)

        Args:
            text: Source text. Any language; the LLM figures it out if
                source_lang is "auto".
            target_lang: BCP-47-ish code — one of {en, zh, ja, ko, ...}.
                Unknown codes pass through to the prompt; the model
                does its best but results are untested.
            source_lang: Either "auto" or a known language code. "auto"
                tells the LLM to detect before translating; passing the
                detector's output is faster and more accurate.

        Returns:
            TranslationResult with `translated=True` on success,
            `translated=False` on short-circuit or LLM failure.
        """
        if not text or not text.strip():
            return TranslationResult(text=text, translated=False)
        if source_lang == target_lang:
            # Trivially in-target; no LLM call needed. translated=False
            # because we didn't actually change anything — the caller
            # may want to distinguish "LLM rendered to target" from
            # "already in target".
            return TranslationResult(text=text, translated=False)

        cache = self._load_cache()
        key = _cache_key(text, target_lang, source_lang)
        cached = cache.get(key)
        if cached is not None:
            return TranslationResult(text=cached, translated=True, from_cache=True)

        try:
            messages = _build_messages(text, source_lang, target_lang)
            raw = self._call_ollama(messages)
        except (httpx.HTTPError, ValueError) as e:
            log.warning(
                "translation failed (%s %s->%s): %s",
                self.model, source_lang, target_lang, e,
            )
            return TranslationResult(text=text, translated=False)

        translated = _clean_output(raw)
        if not translated:
            log.warning("translation empty after cleanup; using original")
            return TranslationResult(text=text, translated=False)

        with self._cache_lock:
            cache[key] = translated
            self._save_cache()
        return TranslationResult(text=translated, translated=True, from_cache=False)
