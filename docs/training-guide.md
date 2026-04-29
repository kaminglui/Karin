# Training a custom voice

This project uses [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for
voice cloning. You don't have to train your own voice — any v2Pro-compatible
checkpoint pair plus a reference clip will work in a character bundle
under `characters/<name>/voices/`. But if you want to fine-tune on your
own target voice, here's the general pipeline.

## What you need

- **A source of clean, per-sentence audio for your target voice.** How you
  produce it is up to you:
    - Record yourself (simplest and free)
    - Use a commercial voice synthesis tool that supports consistent batch
      rendering
    - Use an existing single-speaker dataset you have the right to use

  Requirements: single speaker, clean (no background noise, no clipping),
  reasonably consistent recording conditions, and enough material. 10-30
  minutes total split across 100-500 sentences is typical for fine-tuning.
- **A Google Colab account** (T4 or better) or a local CUDA GPU capable of
  running GPT-SoVITS training.
- **Patience.** Training takes 30-90 minutes per run on Colab.

## Pipeline overview

```
prepared per-sentence WAVs + transcripts
           │
           ▼
 GPT-SoVITS Colab notebook
 (feature extraction + fine-tune)
           │
           ▼
    characters/<your-character>/
    ├── voice.yaml
    └── voices/
        ├── gpt_model_32.ckpt
        ├── sovits_model_16.pth
        └── ref.wav
```

## Step 1 — Write a sentence corpus

A sentence corpus is just a list of text for your target voice to speak.
A simple JSON format is a list of objects, one per utterance:

```json
[
    {"id": "0001", "text": "Hello, this is a test.", "lang": "en"},
    {"id": "0002", "text": "こんにちは", "lang": "ja"}
]
```

You can use this format or any other, as long as you end up with a
transcript that maps to your rendered audio files.

Aim for 100-500 sentences. More doesn't hurt but you hit diminishing
returns past ~200 for voice cloning (as opposed to full TTS training from
scratch).

## Step 2 — Produce per-sentence WAVs

Using whatever method you chose, render or record each sentence as a
separate WAV file:

- Mono, 16 kHz or higher (44.1 / 48 kHz are fine — GPT-SoVITS will
  downsample internally)
- Clean: no background noise, no clipping, no compression artifacts
- Consistent: same voice, same mic/DAC, same room, same gain

Name the files so they align with your transcript — if you used the JSON
format above, name each WAV after its `id` field (e.g. `0001.wav`).

## Step 3 — Fine-tune in Colab

1. Upload the WAVs and the transcript to Google Drive or directly to the
   Colab runtime.
2. Open the
   [GPT-SoVITS Colab notebook](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/colab_webui.ipynb).
3. Follow the notebook's data preparation steps: point it at your WAV
   directory and transcript. The notebook handles feature extraction,
   semantic token generation, and the fine-tune itself.
4. Run both the SoVITS fine-tune (~15-30 min) and the GPT fine-tune
   (~30-60 min). You'll get `sovits_model_e*.pth` and `gpt_model_e*.ckpt`
   checkpoint files, one per epoch.
5. Pick the best checkpoint from each based on validation loss — the
   notebook's webui lets you sample audio per epoch. Don't automatically
   take the final epoch; it's often overfit.
6. Download three files:
    - One `sovits_model_e*.pth`
    - One `gpt_model_e*.ckpt`
    - One short reference WAV (`ref.wav`) — 3-10 seconds of your target
      voice saying a representative phrase. Used at inference time to
      condition the model.

## Step 4 — Drop into the repo

```bash
mkdir -p characters/my_char/voices
cp characters/template/voice.example.yaml characters/my_char/voice.yaml

cp sovits_model_e*.pth    characters/my_char/voices/sovits_model_16.pth
cp gpt_model_e*.ckpt      characters/my_char/voices/gpt_model_32.ckpt
cp ref.wav                characters/my_char/voices/ref.wav
```

Then either:

- Set `tts.voice: my_char` in [`config/assistant.yaml`](../config/assistant.yaml)
  so the bridge picks this bundle by name.
- Or replace the files under `characters/general/voices/` if you want
  the checked-in config to use your new voice without extra edits.

The persona / language metadata for the bundle lives in
`characters/my_char/voice.yaml`. Use
[`characters/template/voice.example.yaml`](../characters/template/voice.example.yaml)
as the starting point.

The weight files are gitignored — they're large and personal. Treat
them as build artifacts, not source.

## Notes on quality

- **Reference clip matters.** Pick a `ref.wav` that captures your target
  voice's neutral timbre. A shouty or whispered clip will bias every
  synthesized phrase in the same direction.
- **Epoch selection matters more than total epochs.** Listen to samples
  at e5, e10, e15 rather than taking the final checkpoint. Overfit
  checkpoints sound unnaturally "on-model."
- **Corpus quality > corpus quantity.** 100 clean sentences beats 500
  noisy ones. Re-render or re-record if anything sounds off.
- **Prefer the language your assistant will actually speak.** If the
  assistant responds in English, train primarily on English material.
  Mixed-language training can help with loanwords but usually isn't
  necessary.
