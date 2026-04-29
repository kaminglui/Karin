"""Benchmark GPT-SoVITS throughput from the Jetson.

Measures time-to-first-byte, total synthesis time, audio duration, and
real-time factor for short/medium/long phrases in both streaming and
batched modes.

Run:
    docker exec karin-web python3 /app/scripts/bench_tts.py
"""
from __future__ import annotations

import io
import json
import os
import statistics
import sys
import time
import urllib.request
import wave

TTS = os.environ.get("KARIN_TTS_BASE_URL", "http://windows-11-pc:9880").rstrip("/") + "/tts"
REF = os.environ.get("KARIN_TTS_REF", "../../characters/default/voices/ref.wav")
RUNS = 3
CASES = [
    ("short",  "Hello, how are you today?"),
    ("medium", "The weather in Philadelphia is partly cloudy with a high of seventy-two degrees and light winds from the west."),
    ("long",   "Neural text-to-speech began with deep-learning approaches in the mid twenty-teens, replacing concatenative and parametric methods with end-to-end models that produce remarkably natural-sounding speech, and has since improved dramatically in both quality and latency."),
]


def one_run(text: str, streaming_mode: int) -> dict:
    body = json.dumps({
        "text": text, "text_lang": "en",
        "ref_audio_path": REF, "prompt_lang": "ko", "prompt_text": "",
        "streaming_mode": streaming_mode, "media_type": "wav",
        "top_k": 5, "top_p": 1.0, "temperature": 0.8, "speed_factor": 1.0,
        "repetition_penalty": 1.5, "parallel_infer": True,
        "text_split_method": "cut5",
    }).encode()
    req = urllib.request.Request(
        TTS, data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first: float | None = None
    chunks: list[bytes] = []
    with urllib.request.urlopen(req, timeout=180) as r:
        while True:
            c = r.read(4096)
            if not c:
                break
            if first is None:
                first = time.monotonic()
            chunks.append(c)
    t1 = time.monotonic()
    wav_bytes = b"".join(chunks)
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            audio_s = w.getnframes() / w.getframerate()
            sr = w.getframerate()
    except Exception:
        audio_s = len(wav_bytes) / 64000
        sr = 32000
    return {
        "ttfb": (first - t0) if first else 0.0,
        "total": t1 - t0,
        "audio": audio_s,
        "bytes": len(wav_bytes),
        "sr": sr,
    }


def stats(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) < 2:
        return f"mean={xs[0]:.2f}"
    return (
        f"mean={statistics.mean(xs):.2f}  "
        f"median={statistics.median(xs):.2f}  "
        f"min={min(xs):.2f}  "
        f"max={max(xs):.2f}  "
        f"stdev={statistics.stdev(xs):.2f}"
    )


def main() -> int:
    print("GPT-SoVITS benchmark: Jetson -> PC over Tailscale")
    print(f"runs per case: {RUNS}")
    print("=" * 72)
    for label, text in CASES:
        for mode, mode_name in ((1, "stream"), (0, "batch ")):
            runs: list[dict] = []
            for i in range(RUNS):
                try:
                    runs.append(one_run(text, mode))
                except Exception as e:
                    print(f"  [{label} {mode_name}] run {i + 1} ERR: {e}",
                          file=sys.stderr)
            if not runs:
                continue
            ttfbs = [r["ttfb"] for r in runs]
            totals = [r["total"] for r in runs]
            audios = [r["audio"] for r in runs]
            rtfs = [r["total"] / r["audio"] for r in runs if r["audio"] > 0]
            mean_audio = statistics.mean(audios)
            print()
            print(f"[{label:6s} {mode_name}] N={len(runs)} audio~{mean_audio:.2f}s @ {runs[0]['sr']}Hz")
            print(f"  TTFB (s)            {stats(ttfbs)}")
            print(f"  total synth (s)     {stats(totals)}")
            print(f"  RTF (synth/audio)   {stats(rtfs)}")
    print()
    print("TTFB = time until first audio byte arrives")
    print("RTF  = total_synth / audio_duration; <1 is faster-than-realtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
