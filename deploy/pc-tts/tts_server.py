"""Karin voice sidecar — GPT-SoVITS TTS + faster-whisper STT over Tailscale.

Single process, single port. The Jetson bridge POSTs synthesis requests
to /tts and transcription requests to /transcribe. Keeps the heavy voice
models on a PC with spare VRAM so the Jetson's 7.4 GiB stays available
for the LLM.

Modes:
    python  tts_server.py              Console mode (visible, for debugging)
    python  tts_server.py --tray       Console + system tray icon
    pythonw tts_server.py              Hidden mode (auto-enables tray)

The server auto-detects the Tailscale IP and binds to it only.

STT config comes from env vars (set in start.bat or a wrapper):
    KARIN_STT_MODEL         faster-whisper model name (default: small.en)
    KARIN_STT_DEVICE        cuda | cpu (default: cuda if available)
    KARIN_STT_COMPUTE_TYPE  int8_float16 | float16 | int8 | float32
                            (default: int8_float16 on cuda, int8 on cpu)
    KARIN_STT_LANGUAGE      en | auto | <ISO> (default: en)

Whisper loads LAZILY on the first /transcribe call, so TTS-only users
don't pay the ~1 GB VRAM cost at startup.
"""
from __future__ import annotations

# --- stdout/stderr safety (MUST run before any import) -------------------
import os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "voice_server.log"
ACTIVE_LOG_PATH = LOG_PATH

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _env_int(name: str, default: int, *, min_value: int = 1, max_value: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


_CPU_COUNT = os.cpu_count() or 1
_VOICE_CPU_THREADS = _env_int(
    "KARIN_VOICE_CPU_THREADS",
    min(4, _CPU_COUNT),
    min_value=1,
    max_value=max(1, _CPU_COUNT),
)
_STT_NUM_WORKERS = _env_int("KARIN_STT_NUM_WORKERS", 1, min_value=1, max_value=4)
_TTS_MAX_CHARS = _env_int("KARIN_TTS_MAX_CHARS", 1000, min_value=1, max_value=10000)
_STT_MAX_SECONDS = _env_int("KARIN_STT_MAX_SECONDS", 30, min_value=1, max_value=600)

# Keep CPU-backed preprocessing from grabbing every core on the PC.
os.environ.setdefault("OMP_NUM_THREADS", str(_VOICE_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_VOICE_CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_VOICE_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_VOICE_CPU_THREADS))

_is_pythonw = sys.stdout is None
if _is_pythonw:
    try:
        _log_fh = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    except PermissionError:
        ACTIVE_LOG_PATH = LOG_PATH.with_suffix(".2.log")
        _log_fh = open(ACTIVE_LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = _log_fh
    sys.stderr = _log_fh
else:
    # GPT-SoVITS upstream prints Chinese literals (e.g. "############ 推理 ############")
    # which crashes on the default Windows console codec (cp1252) with
    # UnicodeEncodeError → surfaces as [Errno 22] Invalid argument when
    # wrapped by the API layer. Reconfigure to UTF-8 + ignore so these
    # debug prints never fail synthesis.
    for _s in (sys.stdout, sys.stderr):
        if _s is not None and hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# --- imports -------------------------------------------------------------
import argparse
import atexit
import gc
import io
import logging
import signal
import subprocess
import threading
import time
import traceback

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

REPO_ROOT = SCRIPT_DIR.parent.parent
SOVITS_ROOT = REPO_ROOT / "third_party" / "GPT-SoVITS"
sys.path.insert(0, str(SOVITS_ROOT))
sys.path.insert(0, str(SOVITS_ROOT / "GPT_SoVITS"))
os.chdir(SOVITS_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("voice_server")


def _configure_torch_threads() -> None:
    try:
        import torch
        torch.set_num_threads(_VOICE_CPU_THREADS)
        # Interop threads can only be set once per process. Ignore if an
        # upstream import already touched it.
        try:
            torch.set_num_interop_threads(min(2, _VOICE_CPU_THREADS))
        except RuntimeError:
            pass
        log.info(
            "resource limits: cpu_threads=%d stt_workers=%d tts_max_chars=%d stt_max_seconds=%d",
            _VOICE_CPU_THREADS,
            _STT_NUM_WORKERS,
            _TTS_MAX_CHARS,
            _STT_MAX_SECONDS,
        )
    except Exception as e:
        log.warning("could not apply torch thread limits: %s", e)


_configure_torch_threads()

# --- GPT-SoVITS inference pipeline ----------------------------------------

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

CONFIG_PATH = SOVITS_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
log.info("loading TTS pipeline from %s", CONFIG_PATH)
tts_config = TTS_Config(str(CONFIG_PATH))
tts_pipeline = TTS(tts_config)
log.info("TTS pipeline ready (device=%s, version=%s)", tts_config.device, tts_config.version)

_current_voice: dict = {"gpt": "", "sovits": "", "ref": ""}
_tts_lock = threading.RLock()
_inference_lock = threading.RLock()
_server_ref: dict[str, uvicorn.Server | None] = {"server": None}
_shutdown_lock = threading.Lock()
_shutdown_started = False
_force_exit_timer_started = False

# --- FastAPI app ----------------------------------------------------------

app = FastAPI(title="Karin voice sidecar (TTS + STT)", docs_url=None)


class SynthRequest(BaseModel):
    text: str
    text_lang: str = "en"
    ref_audio_path: str = ""
    aux_ref_audio_paths: list[str] = Field(default_factory=list)
    prompt_text: str = ""
    prompt_lang: str = "en"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 0.8
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool | int = False
    repetition_penalty: float = 1.5
    parallel_infer: bool = True
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


def _request_dict(req: BaseModel) -> dict:
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()


def _to_int16_mono(audio_chunk: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_chunk)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0) * 32767.0
    else:
        audio = np.clip(audio, -32768, 32767)
    return audio.astype(np.int16, copy=False)


@app.get("/health")
def health():
    loaded = _whisper_engine is not None
    settings = _whisper_settings if loaded else _resolve_stt_settings()
    return {
        "ok": True,
        "service": "karin-voice-sidecar",
        "tts": {
            "loaded": True,
            "device": str(tts_config.device),
            "version": str(tts_config.version),
            "current_voice": dict(_current_voice),
        },
        "stt": {
            "loaded": loaded,
            "model": settings.get("model"),
            "device": settings.get("device"),
            "compute_type": settings.get("compute_type"),
            "language": settings.get("language") or "auto",
        },
        "limits": {
            "cpu_threads": _VOICE_CPU_THREADS,
            "stt_workers": _STT_NUM_WORKERS,
            "tts_max_chars": _TTS_MAX_CHARS,
            "stt_max_seconds": _STT_MAX_SECONDS,
            "http_concurrency": 16,
        },
    }


@app.post("/tts")
def synthesize(req: SynthRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    if len(text) > _TTS_MAX_CHARS:
        raise HTTPException(
            413,
            f"text too long for sidecar TTS ({len(text)} > {_TTS_MAX_CHARS} chars)",
        )
    if req.media_type.lower() != "wav":
        raise HTTPException(400, "only media_type=wav is supported by this sidecar")
    params = _request_dict(req)
    params["text"] = text
    params["media_type"] = "wav"
    log.info("synthesizing: %r (%s)", text[:80], req.text_lang)
    try:
        chunks: list[np.ndarray] = []
        sample_rate = 32000
        with _inference_lock, _tts_lock:
            _current_voice["ref"] = req.ref_audio_path
            for sr, audio_chunk in tts_pipeline.run(params):
                if audio_chunk is not None and len(audio_chunk) > 0:
                    sample_rate = sr
                    chunks.append(_to_int16_mono(audio_chunk))
    except Exception as e:
        log.error("synthesis failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"tts failed: {e}")
    if not chunks:
        raise HTTPException(500, "synthesis produced no audio")
    pcm = np.concatenate(chunks)
    log.info("synthesized %d samples @ %d Hz (%.2fs)",
             len(pcm), sample_rate, len(pcm) / sample_rate)
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


@app.get("/set_gpt_weights")
def set_gpt_weights(weights_path: str):
    log.info("loading GPT weights: %s", weights_path)
    try:
        with _inference_lock, _tts_lock:
            if _current_voice["gpt"] != weights_path:
                tts_pipeline.init_t2s_weights(weights_path)
                _current_voice["gpt"] = weights_path
    except Exception as e:
        log.error("GPT weight load failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(400, f"change gpt weight failed: {e}")
    return {"message": "success"}


@app.get("/set_sovits_weights")
def set_sovits_weights(weights_path: str):
    log.info("loading SoVITS weights: %s", weights_path)
    try:
        with _inference_lock, _tts_lock:
            if _current_voice["sovits"] != weights_path:
                tts_pipeline.init_vits_weights(weights_path)
                _current_voice["sovits"] = weights_path
    except Exception as e:
        log.error("SoVITS weight load failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(400, f"change sovits weight failed: {e}")
    return {"message": "success"}


# --- STT (faster-whisper) -------------------------------------------------
#
# Lazy-load Whisper on the first /transcribe call so TTS-only deployments
# don't pay the load cost at startup. Config comes from env vars (see
# module docstring).

_whisper_engine = None        # faster_whisper.WhisperModel instance
_whisper_settings: dict = {}  # resolved model/device/compute_type/language
_whisper_lock = threading.Lock()


def _resolve_stt_settings() -> dict:
    """Pick Whisper settings from env with Orin-Nano-safe defaults."""
    model = os.environ.get("KARIN_STT_MODEL", "small.en").strip()
    device = os.environ.get("KARIN_STT_DEVICE", "").strip().lower()
    compute_type = os.environ.get("KARIN_STT_COMPUTE_TYPE", "").strip().lower()
    language = os.environ.get("KARIN_STT_LANGUAGE", "en").strip().lower()

    if not device:
        # Default to cuda when available on the PC, else cpu.
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if not compute_type:
        compute_type = "int8_float16" if device == "cuda" else "int8"

    # Auto / empty / "auto" → None (Whisper detects per utterance).
    # In that case strip a `.en` suffix, since English-only models
    # reject non-English input.
    lang_norm: str | None = None if language in ("", "auto") else language
    effective_model = model
    if lang_norm is None or lang_norm != "en":
        if effective_model.endswith(".en"):
            effective_model = effective_model[:-3]
            log.warning(
                "STT model %r is English-only but language=%r requires "
                "multilingual; loading %r instead.",
                model, language, effective_model,
            )

    return {
        "model": effective_model,
        "device": device,
        "compute_type": compute_type,
        "language": lang_norm,
        # Expose the raw value too so /stt/status reports what the env
        # actually had, not just the resolved version.
        "requested_model": model,
        "requested_language": language,
    }


def _get_whisper():
    """Lazy singleton. Safe under concurrent /transcribe calls via lock."""
    global _whisper_engine, _whisper_settings
    if _whisper_engine is not None:
        return _whisper_engine
    with _whisper_lock:
        if _whisper_engine is not None:
            return _whisper_engine
        settings = _resolve_stt_settings()
        log.info(
            "loading faster-whisper: model=%s device=%s compute_type=%s language=%s",
            settings["model"], settings["device"],
            settings["compute_type"], settings["language"] or "auto",
        )
        from faster_whisper import WhisperModel
        try:
            _whisper_engine = WhisperModel(
                model_size_or_path=settings["model"],
                device=settings["device"],
                compute_type=settings["compute_type"],
                cpu_threads=_VOICE_CPU_THREADS,
                num_workers=_STT_NUM_WORKERS,
            )
        except Exception as e:
            log.error("whisper load failed: %s\n%s", e, traceback.format_exc())
            raise
        _whisper_settings = settings
        log.info("faster-whisper ready")
    return _whisper_engine


class TranscribeRequest(BaseModel):
    # Raw int16 PCM @ 16 kHz, base64-encoded. Compact on the wire and
    # avoids multipart upload complexity — the Jetson already has the
    # PCM array in hand from VAD so nothing extra to do client-side
    # besides b64 + POST.
    pcm_base64: str
    sample_rate: int = 16000
    beam_size: int = 1
    vad_filter: bool = False
    # Optional per-request language override. Falls back to the
    # env-resolved default if omitted.
    language: str | None = None


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    import base64
    if not req.pcm_base64:
        raise HTTPException(400, "empty pcm_base64")
    try:
        raw = base64.b64decode(req.pcm_base64, validate=True)
    except Exception as e:
        raise HTTPException(400, f"invalid base64: {e}")
    if not raw:
        raise HTTPException(400, "decoded PCM is empty")
    if len(raw) % 2:
        raise HTTPException(400, "decoded PCM byte length must be even for int16 audio")
    pcm = np.frombuffer(raw, dtype=np.int16)
    if pcm.size == 0:
        raise HTTPException(400, "decoded PCM is empty after frombuffer")
    max_samples = max(1, req.sample_rate) * _STT_MAX_SECONDS
    if pcm.size > max_samples:
        raise HTTPException(
            413,
            f"audio too long for sidecar STT ({pcm.size / max(1, req.sample_rate):.1f}s > {_STT_MAX_SECONDS}s)",
        )
    audio = pcm.astype(np.float32) / 32768.0
    log.info("transcribing %d samples (%.2fs) lang=%s beam=%d",
             pcm.size, pcm.size / max(req.sample_rate, 1),
             req.language or "default", req.beam_size)
    try:
        engine = _get_whisper()
    except Exception as e:
        raise HTTPException(503, f"whisper load failed: {e}")
    # Per-request language override; otherwise env-resolved default.
    lang_for_call = _whisper_settings.get("language")
    if req.language:
        low = req.language.strip().lower()
        lang_for_call = None if low in ("", "auto") else low
    try:
        with _inference_lock:
            segments, _info = engine.transcribe(
                audio,
                language=lang_for_call,
                beam_size=max(1, req.beam_size),
                vad_filter=bool(req.vad_filter),
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        log.error("transcribe failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"transcribe failed: {e}")
    log.info("transcribed: %r", text[:120])
    return {"text": text}


@app.get("/stt/status")
def stt_status():
    """Health + settings snapshot. Reports whether the model is loaded
    (lazy init) and which config it would use. The Jetson calls this on
    page-load to populate the STT indicator hover tooltip."""
    loaded = _whisper_engine is not None
    settings = _whisper_settings if loaded else _resolve_stt_settings()
    return {
        "loaded": loaded,
        "model": settings.get("model"),
        "device": settings.get("device"),
        "compute_type": settings.get("compute_type"),
        "language": settings.get("language") or "auto",
    }


# --- helpers --------------------------------------------------------------

def _detect_tailscale_ip(wait_seconds: int = 60) -> str | None:
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        try:
            r = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    ip = line.strip()
                    if ip:
                        return ip
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(2)


def _mark_inference_stop() -> None:
    """Ask long-running model loops to stop if the upstream supports it."""
    try:
        pipeline = globals().get("tts_pipeline")
        if pipeline is not None:
            setattr(pipeline, "stop_flag", True)
    except Exception:
        pass


def _free_model_resources(reason: str) -> None:
    """Best-effort release of Python, CUDA, and file resources on exit."""
    global _whisper_engine, tts_pipeline, _shutdown_started
    with _shutdown_lock:
        if _shutdown_started:
            return
        _shutdown_started = True

    log.info("releasing voice sidecar resources (%s)", reason)
    _mark_inference_stop()

    try:
        _whisper_engine = None
    except Exception:
        pass

    try:
        tts_pipeline = None
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as e:
        log.debug("CUDA cleanup skipped: %s", e)


def _request_server_shutdown(reason: str) -> None:
    log.info("shutdown requested (%s)", reason)
    _mark_inference_stop()
    server = _server_ref.get("server")
    if server is not None:
        server.should_exit = True


def _install_signal_handlers() -> None:
    def _force_exit_later() -> None:
        time.sleep(20)
        log.error("graceful signal shutdown timed out; forcing process exit")
        os._exit(0)

    def _handler(signum, _frame):
        global _force_exit_timer_started
        _request_server_shutdown(f"signal {signum}")
        if not _force_exit_timer_started:
            _force_exit_timer_started = True
            threading.Thread(target=_force_exit_later, daemon=True).start()

    for sig_name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


atexit.register(lambda: _free_model_resources("process exit"))
_install_signal_handlers()


# --- system tray (optional) -----------------------------------------------

def _run_with_tray(host: str, port: int):
    """Run uvicorn in a thread + pystray icon in the main thread."""
    import pystray
    from PIL import Image, ImageDraw

    def _icon_img(color="#8b5fbf"):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([14, 20, 28, 44], fill=color)
        d.polygon([(28, 16), (42, 8), (42, 56), (28, 48)], fill=color)
        d.arc([44, 16, 58, 48], -60, 60, fill=color, width=3)
        return img

    def _start_uvicorn():
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            workers=1,
            log_level="info",
            limit_concurrency=16,
        )
        server = uvicorn.Server(config)
        _server_ref["server"] = server
        server.run()

    def _quit(icon, *_):
        _request_server_shutdown("tray quit")
        icon.stop()

    server_thread = threading.Thread(target=_start_uvicorn, daemon=False)
    server_thread.start()

    icon = pystray.Icon(
        "KarinTTS", _icon_img(), f"KarinTTS — {host}:{port}",
        menu=pystray.Menu(
            pystray.MenuItem(f"Running on {host}:{port}", lambda *_: None, default=True),
            pystray.MenuItem("Open log", lambda *_: os.startfile(str(ACTIVE_LOG_PATH))),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", _quit),
        ),
    )
    try:
        icon.run()
    finally:
        _request_server_shutdown("tray loop ended")
        server_thread.join(timeout=15.0)
        _free_model_resources("tray exit")
        if server_thread.is_alive():
            log.error("uvicorn thread did not stop after tray quit; forcing process exit")
            os._exit(0)


# --- entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--tray", action="store_true",
                        help="Show system tray icon")
    args = parser.parse_args()

    host = args.host
    if host is None:
        host = _detect_tailscale_ip()
        if host is None:
            host = "127.0.0.1"
            log.error(
                "Tailscale IP was not available; binding to 127.0.0.1 only. "
                "Start Tailscale and relaunch, or pass --host explicitly."
            )
    use_tray = args.tray or _is_pythonw

    log.info("binding to %s:%d (tray=%s)", host, args.port, use_tray)

    if use_tray:
        _run_with_tray(host, args.port)
    else:
        config = uvicorn.Config(
            app,
            host=host,
            port=args.port,
            workers=1,
            log_level="info",
            limit_concurrency=16,
        )
        server = uvicorn.Server(config)
        _server_ref["server"] = server
        try:
            server.run()
        finally:
            _free_model_resources("server exit")
