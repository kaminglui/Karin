"""Shared helpers used by the bridge and the scripts/ directory.

Centralises path resolution and config loading so ``bridge/main.py`` and
``scripts/test_tts.py`` agree on where the repo root is and how to read
``assistant.yaml``.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("bridge.utils")

# bridge/utils.py -> bridge/ -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CHARACTERS_DIR: Path = REPO_ROOT / "characters"


def resolve_path(p: str | Path) -> str:
    """Resolve a repo-relative path to an absolute path.

    Args:
        p: A path. If it's already absolute it's returned unchanged;
            otherwise it's resolved against ``REPO_ROOT``.

    Returns:
        The absolute path as a string, suitable for passing to subprocess
        and HTTP APIs that run on the same host.
    """
    path = Path(p)
    return str(path if path.is_absolute() else (REPO_ROOT / path).resolve())


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its parsed contents.

    Env-var overrides so the same assistant.yaml works across the
    x86-CUDA, Jetson, and CPU-only Docker overlays without edits:

      KARIN_OLLAMA_BASE_URL    -> llm.base_url
      KARIN_TTS_BASE_URL       -> tts.base_url
      KARIN_STT_DEVICE         -> stt.device         (cuda / cpu)
      KARIN_STT_COMPUTE_TYPE   -> stt.compute_type   (float16, int8, int8_float16, ...)
      KARIN_STT_LANGUAGE       -> stt.language       (en / ja / auto / ...)
      KARIN_STT_MODEL          -> stt.model          (small.en, small, base, ...)

    Args:
        path: Path to a YAML file.

    Returns:
        The file's parsed structure as a dict, with env overrides applied.
    """
    import os
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Character / system-prompt override. The shared prompt template
    # lives at characters/profile.yaml (with {persona} and
    # {language_note} placeholders). Falls back to the legacy
    # config/characters/<name>.yaml, then to the inline system_prompt.
    # Env var `KARIN_CHARACTER` wins (useful for Docker profiles).
    # Fall back to "default" (shipped neutral character) so a fresh
    # clone with no `character:` field still resolves cleanly.
    char_name = (
        os.environ.get("KARIN_CHARACTER")
        or cfg.get("character")
        or "default"
    )
    if char_name and isinstance(cfg.get("llm"), dict):
        # New path: shared template at characters/profile.yaml
        # Use REPO_ROOT (not CHARACTERS_DIR) so tests that monkeypatch
        # REPO_ROOT get consistent behavior.
        new_profile = REPO_ROOT / "characters" / "profile.yaml"
        # Legacy path: per-character file at config/characters/<name>.yaml
        legacy_path = REPO_ROOT / "config" / "characters" / f"{char_name}.yaml"
        char_path = new_profile if new_profile.exists() else legacy_path
        if char_path.exists():
            try:
                with char_path.open("r", encoding="utf-8") as cf:
                    char_cfg = yaml.safe_load(cf) or {}
                if char_cfg.get("system_prompt"):
                    cfg["llm"]["system_prompt"] = char_cfg["system_prompt"]
            except Exception as e:
                # Malformed YAML at boot — surface loudly so a broken
                # character file doesn't silently fall back to defaults.
                log.warning("character config %s unreadable: %s", char_path, e)

    # Voice-aware persona substitution. The character template may
    # contain {persona} and {language_note} placeholders filled from
    # the active voice's entry in voices.yaml. When no voice is
    # configured or the voice lacks these fields, the placeholders
    # are replaced with sensible defaults so the prompt never contains
    # raw {persona} text.
    _fill_voice_persona(cfg)

    ollama = os.environ.get("KARIN_OLLAMA_BASE_URL")
    if ollama and isinstance(cfg.get("llm"), dict):
        cfg["llm"]["base_url"] = ollama

    tts_url = os.environ.get("KARIN_TTS_BASE_URL")
    if tts_url and isinstance(cfg.get("tts"), dict):
        cfg["tts"]["base_url"] = tts_url

    stt_device = os.environ.get("KARIN_STT_DEVICE")
    if stt_device and isinstance(cfg.get("stt"), dict):
        cfg["stt"]["device"] = stt_device

    stt_compute = os.environ.get("KARIN_STT_COMPUTE_TYPE")
    if stt_compute and isinstance(cfg.get("stt"), dict):
        cfg["stt"]["compute_type"] = stt_compute

    # Multilingual input support: KARIN_STT_LANGUAGE=auto flips Whisper
    # to per-utterance language detection. Pair with a non-.en model tag
    # (bridge.stt will auto-strip the suffix and warn if you forget).
    stt_language = os.environ.get("KARIN_STT_LANGUAGE")
    if stt_language and isinstance(cfg.get("stt"), dict):
        cfg["stt"]["language"] = stt_language

    stt_model = os.environ.get("KARIN_STT_MODEL")
    if stt_model and isinstance(cfg.get("stt"), dict):
        cfg["stt"]["model"] = stt_model

    llm_model = os.environ.get("KARIN_LLM_MODEL")
    if llm_model and isinstance(cfg.get("llm"), dict):
        cfg["llm"]["model"] = llm_model

    llm_timeout = os.environ.get("KARIN_LLM_TIMEOUT")
    if llm_timeout and isinstance(cfg.get("llm"), dict):
        try:
            cfg["llm"]["request_timeout"] = float(llm_timeout)
        except ValueError:
            pass

    # KV-cache memory lever. Smaller num_ctx ⇒ less VRAM per loaded
    # model. On Jetson, dropping from 4096 to 3072 saves ~25% of KV
    # cache memory at the cost of forgetting older turns sooner.
    num_ctx = os.environ.get("KARIN_NUM_CTX")
    if num_ctx and isinstance(cfg.get("llm"), dict):
        try:
            cfg["llm"]["num_ctx"] = int(num_ctx)
        except ValueError:
            pass

    return cfg


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` atomically.

    Writes to a sibling temp file and ``os.replace()``s into place. On
    both POSIX and Windows the replace is atomic for same-filesystem
    paths, so a concurrent reader either sees the old file or the full
    new file — never a truncated mid-write state.

    Use this instead of ``path.write_text()`` for any file that another
    process/thread might read while we're writing: alerts/trackers/news
    ledgers, preferences JSON, etc.
    """
    import os
    import tempfile
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def parse_iso_utc(s: str) -> datetime:
    """Parse an ISO-8601 string back to a tz-aware UTC ``datetime``.

    A naive datetime (no tz suffix) is treated as UTC — matches the
    convention every Karin store uses on disk. Anything tz-aware is
    converted to UTC.

    Centralized here because the JSON-on-disk stores in ``news/``,
    ``trackers/``, ``alerts/`` and ``learned_store`` all need the same
    "fromisoformat → coerce-naive-to-UTC → astimezone(UTC)" pattern;
    five copies were drifting on the kind of bug that surfaces only
    when one store starts emitting a different format.
    """
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def json_default(obj: Any) -> Any:
    """Canonical JSON serializer for bridge dataclass/datetime/enum types.

    Usable as the ``default=`` argument to ``json.dumps``. Handles:
      - datetime -> ISO-8601 UTC string
      - Enum (incl. str-Enum and IntEnum) -> its .value
      - any @dataclass instance -> recursively via dataclasses.asdict

    Centralised here because bridge/alerts/store.py, bridge/trackers/store.py,
    and the web panels API (web/panels_api.py) all serialize the same
    mix of types to JSON; keeping one implementation avoids drift.
    """
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")


# --- voice persona substitution ----------------------------------------

_DEFAULT_PERSONA = (
    "You are a helpful, friendly voice assistant. Concise and clear."
)
_DEFAULT_LANGUAGE_NOTE = (
    "Replies are ALWAYS in English. If the user writes in another "
    "language, understand it but answer in English. If a tool returns "
    "content in another language, translate or summarize it in English."
)


def _fill_voice_persona(cfg: dict) -> None:
    """Replace ``{persona}`` and ``{language_note}`` in the system prompt.

    Resolution order for persona metadata:
      1. ``characters/{char_name}/voice.yaml`` (new per-character layout)
      2. ``{voice_dir}/voices.yaml`` entry keyed by voice name (legacy)
      3. Built-in defaults

    If anything is missing, defaults are substituted so the prompt is
    always valid — never contains raw ``{persona}`` text.
    """
    import os

    llm = cfg.get("llm")
    if not isinstance(llm, dict):
        return
    prompt = llm.get("system_prompt", "")
    if not isinstance(prompt, str):
        return
    if "{persona}" not in prompt and "{language_note}" not in prompt:
        return

    # Fall back to the shipped `default` neutral character when no
    # character is configured. `characters/default/voice.yaml` ships
    # with the repo (see .gitignore exception) and carries a bland,
    # friendly persona so the prompt is always valid without the
    # operator having to pick a character first.
    char_name = (
        os.environ.get("KARIN_CHARACTER")
        or cfg.get("character")
        or "default"
    )
    tts = cfg.get("tts") or {}
    voice_name = str(tts.get("voice", "")).strip().lower()
    voice_dir = tts.get("voice_dir", "")

    persona = _DEFAULT_PERSONA
    language_note = _DEFAULT_LANGUAGE_NOTE

    # Try 1: new per-character voice.yaml
    if char_name:
        char_voice = REPO_ROOT / "characters" / str(char_name) / "voice.yaml"
        if char_voice.exists():
            try:
                with char_voice.open("r", encoding="utf-8") as f:
                    entry = yaml.safe_load(f) or {}
                if isinstance(entry, dict):
                    if entry.get("persona"):
                        persona = str(entry["persona"]).strip()
                    if entry.get("language_note"):
                        language_note = str(entry["language_note"]).strip()
            except Exception as e:
                # Malformed voice.yaml — log so a typo doesn't silently
                # fall back to the empty persona on every server start.
                log.warning("voice config %s unreadable: %s", char_voice, e)

    # Try 2: legacy voice_dir/voices.yaml (keyed by voice name)
    elif voice_name and voice_dir:
        voice_dir_path = Path(voice_dir)
        if not voice_dir_path.is_absolute():
            voice_dir_path = REPO_ROOT / voice_dir_path
        meta_path = voice_dir_path / "voices.yaml"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                entry = meta.get(voice_name) or {}
                if isinstance(entry, dict):
                    if entry.get("persona"):
                        persona = str(entry["persona"]).strip()
                    if entry.get("language_note"):
                        language_note = str(entry["language_note"]).strip()
            except Exception as e:
                log.warning("legacy voice config %s unreadable: %s", meta_path, e)

    llm["system_prompt"] = prompt.replace(
        "{persona}", persona,
    ).replace(
        "{language_note}", language_note,
    )
