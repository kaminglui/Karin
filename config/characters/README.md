# Legacy character prompts

Current deploys use the character-bundle layout under `characters/`:

- `characters/profile.yaml` is the shared system prompt template.
- `characters/<name>/voice.yaml` holds persona and language metadata.
- `characters/<name>/voices/` holds the TTS bundle for that character.

That newer flow is documented in [characters/README.md](../../characters/README.md).

This `config/characters/` directory is kept only as a legacy fallback
for older setups. As long as `characters/profile.yaml` exists, the
loader prefers that newer path and these per-character YAML files are
ignored.

If you intentionally use the legacy path:

- Each file here is a single YAML document with `system_prompt: | ...`.
- Set `character: "<name>"` in `config/assistant.yaml`.
- If the named file is missing, the bridge falls back to the inline
  `llm.system_prompt` in `assistant.yaml`.

The tracked files here are historical examples (`default.yaml`,
`karin.yaml`). Private fallback files can still use a leading `_`
and stay gitignored.
