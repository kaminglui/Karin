"""Voice loop entry point: mic -> VAD -> STT -> LLM -> TTS -> speaker.

Half-duplex turn-taking: one utterance in, one reply out. No barge-in, no
streaming. Each stage is wrapped in try/except so a single-turn failure
(bad transcription, Ollama blip, TTS 400) logs and continues instead of
killing the loop.

Sample rates in play:
    mic & VAD & STT:     16 kHz  (audio.sample_rate in config)
    TTS output:          32 kHz  (GPT-SoVITS v2Pro native, not hardcoded —
                                  the Speaker accepts a per-call rate)

Launch from the repo root:

    python -m bridge.main --config config/assistant.yaml

Prerequisites (wired as systemd units by deploy/setup.sh):
    - `ollama serve` on http://localhost:11434 with the configured model pulled
    - GPT-SoVITS `api_v2.py` on http://localhost:9880
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bridge.audio_io import MicStream, Speaker
from bridge.llm import OllamaLLM
from bridge.stt import WhisperSTT
from bridge.tts import SoVITSTTS, TTSError
from bridge.utils import REPO_ROOT, load_config, resolve_path
from bridge.vad import SileroVAD

log = logging.getLogger("bridge")


def main() -> int:
    """Load config, build the pipeline, and run the half-duplex voice loop."""
    parser = argparse.ArgumentParser(description="Karin voice bridge")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "assistant.yaml",
        help="path to assistant.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime_cfg = cfg.get("runtime", {})
    logging.basicConfig(
        level=runtime_cfg.get("log_level", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    audio_cfg = cfg["audio"]
    vad_cfg = cfg["vad"]
    stt_cfg = cfg["stt"]
    llm_cfg = cfg["llm"]
    tts_cfg = cfg["tts"]

    if runtime_cfg.get("interrupt_on_speech", False):
        log.warning(
            "interrupt_on_speech is set but not yet implemented; "
            "proceeding in half-duplex mode."
        )

    log.info("loading models ...")
    vad = SileroVAD(
        sample_rate=audio_cfg["sample_rate"],
        threshold=vad_cfg["threshold"],
        min_silence_ms=vad_cfg["min_silence_ms"],
        speech_pad_ms=vad_cfg["speech_pad_ms"],
        min_speech_ms=vad_cfg["min_speech_ms"],
    )
    stt = WhisperSTT(
        model=stt_cfg["model"],
        device=stt_cfg["device"],
        compute_type=stt_cfg["compute_type"],
        language=stt_cfg["language"],
        beam_size=stt_cfg["beam_size"],
        vad_filter=stt_cfg["vad_filter"],
    )
    llm = OllamaLLM(
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        system_prompt=llm_cfg["system_prompt"],
        temperature=llm_cfg["temperature"],
        num_ctx=llm_cfg["num_ctx"],
        options=llm_cfg.get("options", {}),
        backend=llm_cfg.get("backend", "ollama"),
    )
    tts = SoVITSTTS(
        base_url=tts_cfg["base_url"],
        endpoint=tts_cfg["endpoint"],
        ref_audio_path=resolve_path(tts_cfg["ref_audio_path"]),
        prompt_text=tts_cfg["prompt_text"],
        prompt_lang=tts_cfg["prompt_lang"],
        text_lang=tts_cfg["text_lang"],
        top_k=tts_cfg["top_k"],
        top_p=tts_cfg["top_p"],
        temperature=tts_cfg["temperature"],
        speed_factor=tts_cfg["speed_factor"],
        streaming_mode=tts_cfg["streaming_mode"],
        gpt_weights_path=resolve_path(tts_cfg["gpt_weights_path"]),
        sovits_weights_path=resolve_path(tts_cfg["sovits_weights_path"]),
    )

    log.info(
        "opening mic (rate=%d Hz, frame=%d ms, device=%s)",
        audio_cfg["sample_rate"],
        audio_cfg["frame_ms"],
        audio_cfg["input_device"] or "default",
    )

    try:
        with MicStream(
            sample_rate=audio_cfg["sample_rate"],
            frame_ms=audio_cfg["frame_ms"],
            device=audio_cfg["input_device"],
        ) as mic, Speaker(device=audio_cfg["output_device"]) as speaker, tts:
            log.info("ready — listening")
            for utterance in vad.segment(mic):
                duration_s = len(utterance) / audio_cfg["sample_rate"]
                log.info("utterance captured (%.2fs)", duration_s)

                # --- STT ---
                try:
                    text = stt.transcribe(utterance)
                except Exception as e:
                    log.error("STT failed: %s", e)
                    continue
                if not text.strip():
                    log.info("(empty transcription, skipping)")
                    continue
                log.info("user: %s", text)

                # --- LLM ---
                try:
                    reply = llm.chat(text)
                except Exception as e:
                    log.error("LLM failed: %s", e)
                    continue
                reply = reply.strip()
                if not reply:
                    log.info("(empty reply, skipping)")
                    continue
                log.info("assistant: %s", reply)

                # --- TTS ---
                try:
                    pcm, sr = tts.synthesize(reply)
                except TTSError as e:
                    log.error("TTS rejected: %s", e)
                    continue
                except Exception as e:
                    log.error("TTS unexpected error: %s", e)
                    continue

                # --- playback ---
                try:
                    speaker.play(pcm, sr)
                except Exception as e:
                    log.error("playback failed: %s", e)
                    continue
    except KeyboardInterrupt:
        log.info("ctrl-c, stopping")
    finally:
        llm.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
