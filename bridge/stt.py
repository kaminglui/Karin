"""faster-whisper wrapper.

Takes int16 mono PCM at 16 kHz (the rate audio_io captures at), converts
to float32 in ``[-1, 1]``, and runs transcription. ``vad_filter`` is false
because upstream VAD has already done the work — double-filtering drops
words.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("bridge.stt")


def resolve_stt_settings(model: str, language: str | None) -> tuple[str, str | None]:
    """Normalize (model, language) for faster-whisper.

    - ``language`` of ``None``, ``""``, or ``"auto"`` (case-insensitive)
      maps to ``None`` → faster-whisper detects per utterance.
    - When the requested language needs multilingual capability but the
      model tag ends in ``.en`` (English-only), strip the suffix and warn.
      Otherwise faster-whisper would silently reject non-English audio
      or produce garbage transcriptions.

    Returns ``(effective_model, effective_language)``. Pure function so
    it's testable without loading a model.
    """
    if not language or str(language).strip().lower() == "auto":
        lang_norm: str | None = None
    else:
        lang_norm = str(language).strip().lower()

    effective_model = model
    needs_multilingual = lang_norm is None or lang_norm != "en"
    if needs_multilingual and str(model).endswith(".en"):
        effective_model = str(model)[:-3]
        log.warning(
            "STT model %r is English-only but language=%r requires multilingual; "
            "loading %r instead.",
            model, language, effective_model,
        )
    return effective_model, lang_norm


class WhisperSTT:
    """faster-whisper wrapper with a simple ``transcribe()`` interface."""

    def __init__(
        self,
        model: str,
        device: str,
        compute_type: str,
        language: str,
        beam_size: int,
        vad_filter: bool,
    ) -> None:
        """Load a faster-whisper model.

        Args:
            model: Whisper model name (e.g. ``small.en``, ``small``) or
                local path. English-only variants end in ``.en``.
            device: ``cuda`` or ``cpu``.
            compute_type: ctranslate2 compute type — ``int8_float16``,
                ``float16``, ``int8``, or ``float32``.
            language: Whisper language code (``en``, ``ja``, ...) OR
                ``auto`` to let Whisper detect per utterance. When
                ``auto`` is requested with an English-only ``.en``
                model, we log a warning and strip the suffix so the
                multilingual variant loads instead — saves the user a
                silent "why is it still transcribing as English?".
            beam_size: Beam width. 1 = fastest, 5 = more accurate.
            vad_filter: Whether faster-whisper should run its own VAD
                filter. Keep false — upstream VAD already segmented.
        """
        effective_model, lang_norm = resolve_stt_settings(model, language)

        # Lazy import so importing this module doesn't trigger a CUDA init
        # during e.g. `python -m bridge.main --help`.
        from faster_whisper import WhisperModel

        self._engine = WhisperModel(
            model_size_or_path=effective_model,
            device=device,
            compute_type=compute_type,
        )
        self.language = lang_norm   # None = auto-detect
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def transcribe(self, pcm: np.ndarray) -> str:
        """Transcribe a single utterance.

        Args:
            pcm: int16 mono PCM at 16 kHz. If the dtype is not int16 it is
                cast.

        Returns:
            The joined text of all segments, stripped of leading/trailing
            whitespace.
        """
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        # faster-whisper accepts a 1D float32 np array; SR=16000 is assumed.
        audio = pcm.astype(np.float32) / 32768.0
        segments, _info = self._engine.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
