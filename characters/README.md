# Characters

Each subdirectory is a self-contained character bundle: persona,
voice models, and mouth-shape expressions all in one place.

```
characters/
  profile.yaml              ← shared system prompt template
  template/
    voice.example.yaml      ← starter for voice.yaml (copy this)
  <your-character>/
    voice.yaml              ← persona + language settings
    voices/                 ← GPT-SoVITS model files + reference audio
    expressions/            ← mouth-shape PNGs for avatar animation
```

The `template/` folder isn't a character — it has no `voices/` subdir
so the voice discoverer skips it.

## How it works

`profile.yaml` is the shared system prompt template used by ALL
characters. It contains `{persona}` and `{language_note}` placeholders
that are filled at runtime from the active character's `voice.yaml`.

This means: **to add a new character, you only need to create a new
folder** with a `voice.yaml` (describing the personality) and voice
model files. You do NOT need to duplicate or edit the prompt template.

## Adding a new character

Start from [`template/voice.example.yaml`](template/voice.example.yaml)
— copy it into a new character directory and fill in the fields.

```bash
mkdir -p characters/my_char/voices

# 1. Copy the template + edit the persona/language fields
cp characters/template/voice.example.yaml characters/my_char/voice.yaml
$EDITOR characters/my_char/voice.yaml
# At minimum set: prompt_lang (language of your ref.wav), text_lang,
# description. The persona/language_note blocks are optional — delete
# them to inherit the built-in neutral defaults.

# 2. Place voice model files in voices/ — ONE file per type:
#      ref.wav                  reference audio (≤10 s)
#      gpt_model_*.ckpt         trained GPT weights (e.g. gpt_model_32.ckpt)
#      sovits_model_*.pth       trained SoVITS weights (e.g. sovits_model_16.pth)
#    The folder namespaces the voice — no filename prefix needed.
#    If multiple .ckpt or .pth files are present (e.g. several training
#    epochs), the newest by mtime is used and a warning is logged.
#    Keep only the one you want active.

# 3. (OPTIONAL) Place expression PNGs in expressions/
#    Recommended set: default.png, a.png, e.png, i.png, o.png, u.png
#    Any source size/aspect is fine — the server auto-scales to a
#    512x512 square (resize-cover + center-crop) so frames swap
#    cleanly during lip-sync.
#    Missing files fall back to default.png within the character,
#    then to the legacy web/static/faces/, then to 404. So you can
#    ship just default.png and everything still works — the mouth
#    just won't change shape.

# 4. Activate in config/assistant.yaml:
#    character: my_char
```

## Expression PNGs — what the program does for you

The avatar's top face button lip-syncs through six images:
`default` at rest, one per vowel (`a` / `e` / `i` / `o` / `u`)
while text streams or TTS audio plays.

- **None of them are required.** If `expressions/` is missing entirely,
  the animator falls back to the legacy shared faces in
  `web/static/faces/`. If those are gone too, the browser gets a
  404 and the face stays whatever was last loaded.
- **`default.png` is the minimum useful file.** Ship just that, and
  every vowel frame will transparently fall back to it — the avatar
  won't animate its mouth, but it'll still show a face.
- **Ship all six for lip-sync.** Drop them in `expressions/`; the
  frontend preloads them at page load.
- **No preprocessing needed.** Source images of any dimension or
  aspect are auto-normalized to **512×512** on the server: scaled
  to cover (fill both axes), then center-cropped. Cached under
  `data/expressions_cache/` keyed by source mtime — swap a PNG in
  place and the cache invalidates automatically.

## Switching characters

Three ways, in order of preference:

1. **Sidebar dropdown (runtime).** Pick a character from the sidebar's
   `🎭 Character` select. The browser POSTs `/api/tts/voice` with the
   chosen name; the server atomically swaps voice weights (if the
   character has a `voices/` bundle — see "has_voice" below), re-renders
   the persona from the target's `voice.yaml`, updates the face renderer
   + theme, and sets `os.environ["KARIN_CHARACTER"]` so the next page
   render picks up the new selection. No restart needed. Mid-turn swaps
   are blocked until the turn finishes.

2. **Boot-time env override.** `KARIN_CHARACTER=<name>` at startup. Wins
   over `config/assistant.yaml`. Useful for Docker profiles.

3. **Config file.** Set `character: <name>` in `config/assistant.yaml`
   and restart. Lowest priority — the env override and the runtime
   dropdown both beat it.

### Characters with and without voice bundles

Each dropdown entry is annotated based on what's on disk:

- **Full voice bundle** (`voices/ref.wav` + `*.ckpt` + `*.pth` all
  present) — the entry shows clean (no marker) and selecting it swaps
  both voice weights AND persona.
- **`○` leading indicator** — the character has a `voice.yaml` (and
  optionally a `face.json` / `expressions/`) but no voice bundle on
  this host. The dropdown still shows it; picking it swaps persona +
  face + theme but leaves TTS using the previously loaded voice (or
  silent if none was loaded). The shipped `characters/default/` falls
  in this bucket — it's the neutral fallback used when no `character:`
  is set anywhere.

The previous `(text only)` text suffix was replaced with the leading
`○` indicator in 2026-04-25 — same semantics, less visual noise when
many characters lack local bundles (common on a fresh deploy where
`voices/*.{ckpt,pth,wav}` is gitignored).

### Hidden from the dropdown

A character directory is **skipped** by the scanner (won't appear in
the dropdown at all) when it has neither `voice.yaml` nor a complete
voice bundle. This handles the fresh-tar / fresh-clone case where
`voice.yaml` is gitignored — the directory ships empty, the persona
metadata is missing, and selecting it would 400 with "character not
found". The scanner filter at [`web/server.py::_scan_available_characters`](../web/server.py)
hides those entries instead of offering a broken choice.

### What the dropdown does NOT do

- It doesn't write to `config/assistant.yaml`. The runtime selection
  persists for the life of the server process (via the env var) but
  resets on restart. Set `character:` in the config file if you want
  the selection to survive a restart.
- It doesn't train a voice. Voice bundles are produced by the workflow
  in [../docs/training-guide.md](../docs/training-guide.md) and dropped
  into `characters/<name>/voices/`.

## Deploying a character to a remote host

Custom characters' `voice.yaml` and `voices/*.{ckpt,pth,wav}` are
gitignored on purpose — they're personal per-deploy. The shipped
`characters/default/voice.yaml` is the only exception (kept via the
`!characters/default/voice.yaml` rule in `.gitignore`).

That means: a fresh `git clone` or `tar+scp` of the repo to a new host
arrives with **only `default` activatable**. Other character
directories may transfer (face.json, expressions/ are tracked) but
their persona + voice bits don't, and the scanner correctly hides
them.

To bring a character you've authored on your dev machine to a
production host, transfer the persona + voice files manually:

```bash
# voice.yaml is small (~3 KB); always needed.
scp characters/<name>/voice.yaml \
    <remote-user>@<remote-host>:~/Karin/characters/<name>/

# voices/ holds the trained models (~280 MB per character: gpt_model.ckpt
# ~149 MB, sovits_model.pth ~129 MB, ref.wav small). Needed if you want
# *this host* to push the right weights to its TTS sidecar.
scp -r characters/<name>/voices \
    <remote-user>@<remote-host>:~/Karin/characters/<name>/
```

No bridge restart is required after the transfer — the scanner walks
the filesystem on every `/api/characters/available` call. Hard-reload
the browser and the new character appears in the dropdown.

### With PC-TTS offload, why does the Jetson still need `voices/`?

Even when the actual TTS engine runs on a separate PC sidecar, the
**Jetson's** runtime needs the voice files locally for two reasons:

1. **Discovery** — `bridge.tts_voices.discover_voices()` scans the
   local filesystem to populate the `voices` registry. If a character
   isn't there, the bridge can't push its weight paths to the sidecar.
2. **Path construction** — `web/panels_api.py::tts_voice_switch`
   converts each local path to a `../../characters/<name>/voices/...`
   relative form the PC sidecar can resolve from its own CWD. The local
   files don't need to be readable by the Jetson at runtime, but they
   do need to exist with the same names so the relative paths line up.

In other words: the Jetson's `voices/` is a directory of *placeholder
references* the PC sidecar uses to read its own copy of the same
weights. Same files, two locations. (A future refactor could read paths
from `voice.yaml` directly and skip the placeholder requirement, but
that's not yet done.)

### Maintainer flow on private hosts

If you're the same operator who owns both the dev machine and the
Jetson over Tailscale, the [RUNBOOK.dev.md](../RUNBOOK.dev.md) has a
ready-made command block for transferring characters end-to-end (it's
gitignored — local-only).
