"""Expression-image resolver + normalizer for the avatar face.

Frontend swaps through ``a/e/i/o/u/default.png`` to lip-sync the face
button. Users drop PNGs of any size and aspect into
``characters/<name>/expressions/``; this module:

  1. **Resolves** the requested file with a graceful fallback chain
     (requested → character default.png → legacy static/faces → 404)
     so incomplete bundles still render *something*.
  2. **Normalizes** the image to a consistent square (default 512×512)
     via resize-cover + center-crop, so source photos of any shape
     swap cleanly without the button resizing mid-animation.
  3. **Caches** processed bytes under ``data/expressions_cache/`` keyed
     by source mtime, so cold-start cost is one-shot per file.

Public API:
    resolve_expression(char_dir, filename, char_name, legacy_root)
        → (bytes, media_type) | None
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from bridge.utils import REPO_ROOT

log = logging.getLogger("bridge.expressions")

# Target dimensions. Big enough to stay sharp if the UI ever renders
# the face larger; small enough that processing stays fast and the
# cache stays compact.
_TARGET_SIZE = (512, 512)
_CACHE_ROOT = REPO_ROOT / "data" / "expressions_cache"


def _safe_under(root: Path, candidate: Path) -> Path | None:
    """Return ``candidate.resolve()`` only if it's inside ``root`` and
    exists. Defeats symlink-escape attempts."""
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (FileNotFoundError, ValueError):
        return None
    return resolved


def _resolve_source(
    char_dir: Path,
    filename: str,
    legacy_root: Path,
) -> Path | None:
    """Fallback chain: character's requested file → character's
    default.png → legacy static/faces/<filename> → legacy default.png.
    Returns None if nothing is available.
    """
    expr_dir = char_dir / "expressions"

    # 1. Exact requested file in the character dir
    hit = _safe_under(char_dir, expr_dir / filename)
    if hit is not None:
        return hit

    # 2. Character's default.png (visible fallback for missing phonemes)
    hit = _safe_under(char_dir, expr_dir / "default.png")
    if hit is not None:
        return hit

    # 3. Legacy shared fallback at web/static/faces/
    hit = _safe_under(legacy_root, legacy_root / filename)
    if hit is not None:
        return hit

    # 4. Legacy default.png
    hit = _safe_under(legacy_root, legacy_root / "default.png")
    return hit


def _normalize_bytes(png_path: Path) -> bytes:
    """Resize-cover to ``_TARGET_SIZE`` and center-crop. Preserves
    alpha. Returns PNG-encoded bytes."""
    from PIL import Image

    target_w, target_h = _TARGET_SIZE
    with Image.open(png_path) as im:
        im = im.convert("RGBA")
        src_w, src_h = im.size

        # Cover: scale so the SMALLER ratio is ≥ target (fills both axes)
        scale = max(target_w / src_w, target_h / src_h)
        new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
        im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center-crop to exact target
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        im = im.crop((left, top, left + target_w, top + target_h))

        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def _cache_path(source: Path) -> Path:
    """Deterministic cache location keyed by source path + mtime.

    Including mtime in the key means we auto-invalidate when the user
    replaces a PNG. No explicit cache-busting needed.
    """
    mtime_ns = source.stat().st_mtime_ns
    key = f"{source.resolve()}|{mtime_ns}|{_TARGET_SIZE[0]}x{_TARGET_SIZE[1]}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    # Keep the original basename for readability in the cache dir
    return _CACHE_ROOT / f"{digest}_{source.name}"


def resolve_expression(
    char_dir: Path,
    filename: str,
    legacy_root: Path,
) -> tuple[bytes, str] | None:
    """Resolve + normalize an expression PNG. Returns ``(bytes, media_type)``
    or None if nothing is available even after fallbacks.

    ``char_dir`` is the per-character root (``characters/<name>``), not
    the expressions subdir. ``legacy_root`` is the legacy global faces
    directory (``web/static/faces``).
    """
    source = _resolve_source(char_dir, filename, legacy_root)
    if source is None:
        return None

    cached = _cache_path(source)
    if cached.exists():
        try:
            return cached.read_bytes(), "image/png"
        except OSError as e:
            log.warning("cache read failed for %s: %s; reprocessing", cached, e)

    try:
        data = _normalize_bytes(source)
    except Exception as e:
        # Corrupt PNG, unsupported format, etc. Serve the raw bytes so
        # the browser at least gets SOMETHING rather than a 500.
        log.warning("normalize failed for %s (%s); serving raw", source, e)
        return source.read_bytes(), "image/png"

    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    except OSError as e:
        log.warning("cache write failed for %s: %s", cached, e)

    return data, "image/png"
