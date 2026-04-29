"""Auto-discover TTS voice models from the characters/ directory.

Each character's voice lives under ``characters/<name>/voices/``:

    characters/
      karin/
        voice.yaml              metadata (prompt_lang, persona, ...)
        voices/
          ref.wav               reference clip (≤10 s)
          gpt_model*.ckpt       GPT weights (first match wins)
          sovits_model*.pth     SoVITS weights (first match wins)
      default/
        voice.yaml
        voices/
          ref.wav
          gpt_model*.ckpt
          sovits_model*.pth

Voice name = character folder name. The files themselves don't need a
prefix — the folder namespaces them. Only one voice is loaded into
GPT-SoVITS at a time; discovery just enumerates what's available so
the Settings panel can offer a switcher (``POST /api/tts/voice``).

Legacy fallback: if ``voice_dir`` is passed explicitly and is a flat
directory (the old ``voice_training/`` layout), we match the prior
prefix-based naming convention
(``{voice}_ref.wav`` / ``{voice}_gpt_model*.ckpt`` /
``{voice}_sovits_model*.pth``). Used by deployments that haven't yet
migrated their binaries into ``characters/<name>/voices/``.

Public API:
    discover_voices(voice_dir=None) -> dict[str, VoiceModel]
    VoiceModel — dataclass with paths + per-voice metadata
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger("bridge.tts_voices")

_GPT_RE = re.compile(r"^(.+?)_gpt_model.*\.ckpt$", re.IGNORECASE)
_SOVITS_RE = re.compile(r"^(.+?)_sovits_model.*\.pth$", re.IGNORECASE)
_REF_RE = re.compile(r"^(.+?)_ref\.wav$", re.IGNORECASE)


@dataclass(frozen=True)
class VoiceModel:
    """A complete voice model triplet + optional metadata."""
    name: str
    ref_wav: Path
    gpt_ckpt: Path
    sovits_pth: Path
    prompt_lang: str = ""
    text_lang: str = ""
    description: str = ""


def _load_yaml(path: Path) -> dict:
    """Best-effort YAML load. Returns {} on any failure."""
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("%s unreadable (%s); ignoring", path, e)
        return {}
    return raw if isinstance(raw, dict) else {}


def _pick_one(root: Path, patterns: tuple[str, ...], label: str) -> Path | None:
    """Return the newest file in ``root`` matching any of the glob patterns.

    Convention: each ``characters/<name>/voices/`` dir should contain
    **one** file of each type. When multiple match (e.g. the user
    copied in several training epochs), we pick the most-recently
    modified and log a warning so the resolution is visible.
    """
    matches: list[Path] = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                matches.append(p)
    if not matches:
        return None
    # Dedup (a file could match multiple patterns) then sort by mtime desc.
    uniq = sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(uniq) > 1:
        log.warning(
            "%s: multiple %s files in %s — using newest (%s). "
            "Keep only one of each type to make this deterministic.",
            root.parent.name, label, root.name, uniq[0].name,
        )
    return uniq[0]


def _discover_character_voices(characters_dir: Path) -> dict[str, VoiceModel]:
    """Scan ``characters/*/voices/`` for per-character voice triplets.

    Voice name = character folder name. Files don't need a prefix —
    the folder namespaces them. Missing a file? Voice is skipped.
    """
    out: dict[str, VoiceModel] = {}
    if not characters_dir.is_dir():
        return out

    for char_dir in sorted(characters_dir.iterdir()):
        if not char_dir.is_dir() or char_dir.name.startswith((".", "_")):
            continue
        voices_dir = char_dir / "voices"
        if not voices_dir.is_dir():
            continue

        ref = _pick_one(voices_dir, ("ref.wav", "*_ref.wav"), "ref.wav")
        gpt = _pick_one(voices_dir, ("gpt_model*.ckpt", "*_gpt_model*.ckpt"), "gpt_model*.ckpt")
        sovits = _pick_one(voices_dir, ("sovits_model*.pth", "*_sovits_model*.pth"), "sovits_model*.pth")

        if not (ref and gpt and sovits):
            missing = [k for k, v in (("ref.wav", ref), ("gpt_model*.ckpt", gpt),
                                      ("sovits_model*.pth", sovits)) if v is None]
            log.debug("character %r voices/ incomplete (missing %s); skipping",
                      char_dir.name, ", ".join(missing))
            continue

        meta = _load_yaml(char_dir / "voice.yaml")
        name = char_dir.name.lower()
        out[name] = VoiceModel(
            name=name,
            ref_wav=ref,
            gpt_ckpt=gpt,
            sovits_pth=sovits,
            prompt_lang=str(meta.get("prompt_lang", "")).strip(),
            text_lang=str(meta.get("text_lang", "")).strip(),
            description=str(meta.get("description", "")).strip(),
        )
    return out


def _discover_legacy_flat(voice_dir: Path) -> dict[str, VoiceModel]:
    """Legacy: scan a flat dir with prefix-based naming.

    Kept for ``voice_training/`` style layouts. Voice name = file prefix
    (``karin_ref.wav`` → voice ``karin``). Metadata comes from a single
    ``voices.yaml`` alongside the binaries.
    """
    refs: dict[str, Path] = {}
    gpts: dict[str, Path] = {}
    sovits: dict[str, Path] = {}
    for f in voice_dir.iterdir():
        if not f.is_file():
            continue
        for store, rx in ((refs, _REF_RE), (gpts, _GPT_RE), (sovits, _SOVITS_RE)):
            m = rx.match(f.name)
            if m:
                store[m.group(1).lower()] = f
                break

    meta = _load_yaml(voice_dir / "voices.yaml")

    out: dict[str, VoiceModel] = {}
    for voice in sorted(set(refs) | set(gpts) | set(sovits)):
        if not (voice in refs and voice in gpts and voice in sovits):
            log.debug("legacy voice %r incomplete — skipping", voice)
            continue
        vm = meta.get(voice) or {}
        out[voice] = VoiceModel(
            name=voice,
            ref_wav=refs[voice],
            gpt_ckpt=gpts[voice],
            sovits_pth=sovits[voice],
            prompt_lang=str(vm.get("prompt_lang", "")).strip(),
            text_lang=str(vm.get("text_lang", "")).strip(),
            description=str(vm.get("description", "")).strip(),
        )
    return out


def discover_voices(voice_dir: str | Path | None = None) -> dict[str, VoiceModel]:
    """Return ``{voice_name: VoiceModel}`` for every complete voice found.

    Default (``voice_dir=None``): scans ``characters/*/voices/`` relative
    to the repo root and keys voices by character folder name.

    Legacy: if ``voice_dir`` is passed and resolves to an existing
    directory, scans it as a flat prefix-named layout (the old
    ``voice_training/`` convention).
    """
    from bridge.utils import REPO_ROOT

    if voice_dir is None:
        return _discover_character_voices(REPO_ROOT / "characters")

    vdir = Path(voice_dir)
    if not vdir.is_absolute():
        vdir = REPO_ROOT / vdir
    if not vdir.is_dir():
        log.info("voice_dir %s does not exist; falling back to characters/", vdir)
        return _discover_character_voices(REPO_ROOT / "characters")

    # Legacy flat dir
    out = _discover_legacy_flat(vdir)
    if out:
        log.info("discovered %d voice(s) in legacy %s: %s",
                 len(out), vdir, ", ".join(out))
    else:
        log.info("no voices in %s; falling back to characters/", vdir)
        out = _discover_character_voices(REPO_ROOT / "characters")

    return out
