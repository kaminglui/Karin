#!/usr/bin/env python
"""Standalone TTS smoke test.

Run this to verify your GPT-SoVITS server is wired up correctly before
launching the full voice loop. Produces a WAV file you can play to confirm
the voice sounds like your trained model.

-----------------------------------------------------------------------------
PREREQUISITE — launch the GPT-SoVITS api_v2.py server in a separate shell
first:

    # From wherever you've cloned RVC-Boss/GPT-SoVITS. Schema pinned to
    # commit 2d9193b0d3c0eae0c3a14d8c68a839f1bae157dc (2026-02-09).
    cd /path/to/GPT-SoVITS
    python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml

The tts_infer.yaml can be any valid upstream config. This client swaps in
the trained weights at startup via /set_gpt_weights and /set_sovits_weights,
so you don't need to edit the upstream yaml.

-----------------------------------------------------------------------------
USAGE:

    # From the Karin repo root:
    python scripts/test_tts.py
    python scripts/test_tts.py "Your custom test phrase."
    python scripts/test_tts.py --out tmp/my_test.wav "Hello world."

Output:
    tmp/tts_test.wav   — synthesized audio, playable with any media player
    stdout             — sample rate and duration

Exit codes:
    0  success
    1  TTS server returned an error (check its logs)
    2  unexpected client-side exception
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

# Make `bridge` importable when running this script directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.tts import SoVITSTTS, TTSError  # noqa: E402
from bridge.utils import REPO_ROOT, load_config, resolve_path  # noqa: E402


DEFAULT_TEXT = "Hello, this is a test of the edge assistant voice synthesis pipeline."


def main() -> int:
    """Parse args, construct the TTS client, synthesize, and write the WAV."""
    parser = argparse.ArgumentParser(description="GPT-SoVITS TTS smoke test")
    parser.add_argument(
        "text",
        nargs="?",
        default=DEFAULT_TEXT,
        help=f"text to synthesize (default: {DEFAULT_TEXT!r})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "assistant.yaml",
        help="path to assistant.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tmp" / "tts_test.wav",
        help="output WAV path",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tts_cfg = cfg["tts"]

    print(f"[test_tts] server:    {tts_cfg['base_url']}{tts_cfg['endpoint']}")
    print(f"[test_tts] ref_audio: {resolve_path(tts_cfg['ref_audio_path'])}")
    print(f"[test_tts] gpt ckpt:  {resolve_path(tts_cfg['gpt_weights_path'])}")
    print(f"[test_tts] sovits:    {resolve_path(tts_cfg['sovits_weights_path'])}")
    print(f"[test_tts] text_lang: {tts_cfg['text_lang']}")
    print(f"[test_tts] text:      {args.text!r}")

    try:
        with SoVITSTTS(
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
        ) as tts:
            print("[test_tts] weights loaded, sending /tts request ...")
            pcm, sr = tts.synthesize(args.text)
    except TTSError as e:
        print(f"[test_tts] TTS error: {e}", file=sys.stderr)
        print(
            "[test_tts] hint: check the api_v2.py server logs. Common causes:\n"
            "  - text_lang not in tts_config.languages (try 'en' or 'ja')\n"
            "  - ref_audio_path / weights paths not readable by the server process\n"
            "  - server not actually running on the configured host:port",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"[test_tts] unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, pcm, sr, subtype="PCM_16")

    duration_s = len(pcm) / sr
    print(f"[test_tts] OK — wrote {args.out} ({duration_s:.2f}s @ {sr} Hz, {len(pcm)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
