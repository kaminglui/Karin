"""TTS output cache for replay across page loads.

Each completed turn's synthesized audio is written to
``data/audio_cache/<conversation_id>/<turn_id>.wav`` so the frontend
can replay it after a browser refresh, a tab close/reopen, or a
switch to another tailnet client.

The cache dir is designed to be mounted as **tmpfs** in production
(`/app/data/audio_cache` with `size=256m` in docker-compose). That
means the audio lives in RAM, bounded, and gets wiped on container
restart. The conversation JSON still keeps ``audio_id`` and
``duration_ms`` pointers, so a missing WAV after restart is a
recoverable 404 the frontend can handle — not a data-integrity bug.

Lifecycle:

- **write** at the end of each streaming TTS turn: ``save_turn_audio``
- **read** via the ``/api/audio/<cid>/<tid>`` endpoint
- **cascade-delete** whenever the owning conversation is removed:
  ``ConversationStore.delete_conversation`` calls
  ``delete_conversation_audio``

We persist 32-bit little-endian WAV with the synthesizer's native
sample rate (typically 32 kHz mono int16). No transcoding, no header
rewriting — the PCM chunks go straight from GPT-SoVITS's output to
disk.
"""
from __future__ import annotations

import logging
import shutil
import struct
import uuid
from pathlib import Path

import numpy as np

from bridge.utils import REPO_ROOT

log = logging.getLogger("bridge.audio_cache")

_CACHE_ROOT = REPO_ROOT / "data" / "audio_cache"


def cache_root() -> Path:
    """Root directory for all cached turn audio. Mounted as tmpfs in
    production via ``deploy/docker-compose.yml``."""
    return _CACHE_ROOT


def new_turn_id() -> str:
    """Generate a short, URL-safe turn identifier (12 hex chars)."""
    return uuid.uuid4().hex[:12]


def turn_audio_path(conversation_id: str, turn_id: str) -> Path:
    """Path the WAV for a given turn lives at (whether or not it exists)."""
    return _CACHE_ROOT / conversation_id / f"{turn_id}.wav"


def _int16_chunks_to_wav_bytes(chunks: list[np.ndarray], sample_rate: int) -> bytes:
    """Stitch int16 PCM chunks into a standard WAV container.

    Accepts any iterable of 1-D int16 numpy arrays (what
    ``tts.synthesize_stream`` yields). First chunk may include a 44-byte
    WAV header from GPT-SoVITS; we strip any partial headers and always
    write our own canonical header so the file is always playable.
    """
    if not chunks:
        return b""
    pcm = np.concatenate(chunks).astype("<i2", copy=False).tobytes()
    # Canonical PCM WAV header (16-bit mono).
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + pcm


def save_turn_audio(
    conversation_id: str,
    turn_id: str,
    chunks: list[np.ndarray],
    sample_rate: int,
) -> int:
    """Write a turn's audio to ``<cache>/<cid>/<tid>.wav`` atomically.

    Returns the byte size written (0 on failure). Never raises —
    audio caching is strictly a convenience layer; a failure here
    must not break the LLM turn.
    """
    if not chunks:
        return 0
    try:
        data = _int16_chunks_to_wav_bytes(chunks, sample_rate)
        if not data:
            return 0
        dest = turn_audio_path(conversation_id, turn_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".wav.tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        log.debug("cached %s/%s.wav (%d bytes)", conversation_id, turn_id, len(data))
        return len(data)
    except Exception as e:
        log.warning("audio cache write failed for %s/%s: %s",
                    conversation_id, turn_id, e)
        return 0


def delete_conversation_audio(conversation_id: str) -> None:
    """Remove every cached audio file for a conversation. Idempotent."""
    d = _CACHE_ROOT / conversation_id
    if not d.exists():
        return
    try:
        shutil.rmtree(d, ignore_errors=True)
        log.info("deleted audio cache for conversation %s", conversation_id)
    except Exception as e:
        log.warning("failed to remove %s: %s", d, e)


def resolve_turn_audio(conversation_id: str, turn_id: str) -> Path | None:
    """Resolve a WAV path and verify it sits inside the cache root.
    Returns None if the file is missing or would escape the cache dir.
    """
    candidate = turn_audio_path(conversation_id, turn_id)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(_CACHE_ROOT.resolve())
    except (FileNotFoundError, ValueError):
        return None
    return resolved
