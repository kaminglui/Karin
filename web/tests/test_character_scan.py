"""Tests for web.server character discovery + the `has_voice` flag.

Covers the two functions that drive the sidebar character dropdown:

  * `_character_has_voice(path)` — returns True iff `path/voices/`
    contains a complete GPT-SoVITS triplet (ref.wav + *.ckpt + *.pth).
  * `_scan_available_characters()` — walks `characters/*/` and returns
    one dict per renderable character with `{name, type, face_config?,
    label, has_voice}`. Pivots `STATIC_DIR` at test time so the scanner
    walks a fixture tree rather than the repo's real characters/ dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# Import lazily so import-time side effects (model loading, etc.) don't
# slow the test suite down unnecessarily. pytest discovers the tests by
# looking at the module; the actual `import web.server` only happens
# inside the tests themselves when a fixture needs it.


@pytest.fixture
def server_module():
    import web.server as server
    return server


# ---------------------------------------------------------------------------
# _character_has_voice — filesystem triplet check
# ---------------------------------------------------------------------------
class TestCharacterHasVoice:
    """A usable voice bundle needs all three: ref.wav (or *_ref.wav)
    + *.ckpt + *.pth under the character's `voices/` dir. Any missing
    piece → `has_voice: False`."""

    def test_complete_bundle_returns_true(self, server_module, tmp_path):
        char = tmp_path / "alice"
        voices = char / "voices"
        voices.mkdir(parents=True)
        (voices / "ref.wav").write_bytes(b"")
        (voices / "gpt_model_32.ckpt").write_bytes(b"")
        (voices / "sovits_model_16.pth").write_bytes(b"")
        assert server_module._character_has_voice(char) is True

    def test_missing_voices_dir_returns_false(self, server_module, tmp_path):
        char = tmp_path / "alice"
        char.mkdir()
        assert server_module._character_has_voice(char) is False

    def test_voices_dir_is_file_not_dir_returns_false(self, server_module, tmp_path):
        char = tmp_path / "alice"
        char.mkdir()
        # Pathologically: `voices` is a file, not a dir.
        (char / "voices").write_bytes(b"oops")
        assert server_module._character_has_voice(char) is False

    def test_missing_ref_wav_returns_false(self, server_module, tmp_path):
        char = tmp_path / "alice"
        voices = char / "voices"
        voices.mkdir(parents=True)
        (voices / "gpt_model_32.ckpt").write_bytes(b"")
        (voices / "sovits_model_16.pth").write_bytes(b"")
        assert server_module._character_has_voice(char) is False

    def test_missing_ckpt_returns_false(self, server_module, tmp_path):
        char = tmp_path / "alice"
        voices = char / "voices"
        voices.mkdir(parents=True)
        (voices / "ref.wav").write_bytes(b"")
        (voices / "sovits_model_16.pth").write_bytes(b"")
        assert server_module._character_has_voice(char) is False

    def test_missing_pth_returns_false(self, server_module, tmp_path):
        char = tmp_path / "alice"
        voices = char / "voices"
        voices.mkdir(parents=True)
        (voices / "ref.wav").write_bytes(b"")
        (voices / "gpt_model_32.ckpt").write_bytes(b"")
        assert server_module._character_has_voice(char) is False

    def test_prefixed_ref_wav_also_counts(self, server_module, tmp_path):
        """`<name>_ref.wav` satisfies the ref-audio glob alongside `ref.wav`.
        Some training pipelines name the file `alice_ref.wav` to keep a
        single flat voices/ dir across multiple speakers during dev."""
        char = tmp_path / "alice"
        voices = char / "voices"
        voices.mkdir(parents=True)
        (voices / "alice_ref.wav").write_bytes(b"")
        (voices / "gpt_model_32.ckpt").write_bytes(b"")
        (voices / "sovits_model_16.pth").write_bytes(b"")
        assert server_module._character_has_voice(char) is True


# ---------------------------------------------------------------------------
# _scan_available_characters — walks characters/*/ and returns entries
# ---------------------------------------------------------------------------
class TestScanAvailableCharacters:
    """Exercises the scanner's inclusion rules:

      * include: has `face.json` (procedural-sun renderer)
      * include: has `expressions/default.png` (bitmap renderer)
      * exclude: neither above (no renderer → would 404)
      * exclude: hidden dirs (`.foo`)
      * exclude: the `template/` scaffold dir
    """

    @pytest.fixture
    def fake_repo(self, server_module, tmp_path, monkeypatch):
        """Build a fake repo with the expected static/characters layout.

        Layout:
          <tmp>/web/static         ← STATIC_DIR (patched)
          <tmp>/characters/
            alice/                 procedural-sun + complete voice bundle
              face.json
              voices/ref.wav, *.ckpt, *.pth
            bob/                   bitmap renderer, no voice
              expressions/default.png
            template/              scaffold — skipped
            .hidden/               hidden — skipped
            ghost/                 no renderer — skipped
        """
        static_dir = tmp_path / "web" / "static"
        static_dir.mkdir(parents=True)
        characters = tmp_path / "characters"
        characters.mkdir()

        # alice — procedural-sun + voice
        alice = characters / "alice"
        alice.mkdir()
        (alice / "face.json").write_text(
            json.dumps({"type": "procedural-sun", "body": "#ff0"}),
            encoding="utf-8",
        )
        voices = alice / "voices"
        voices.mkdir()
        (voices / "ref.wav").write_bytes(b"")
        (voices / "gpt_model_32.ckpt").write_bytes(b"")
        (voices / "sovits_model_16.pth").write_bytes(b"")

        # bob — bitmap renderer, no voice bundle. Still has voice.yaml
        # so the activatability gate accepts it (persona-only swap is
        # supported).
        bob = characters / "bob"
        (bob / "expressions").mkdir(parents=True)
        (bob / "expressions" / "default.png").write_bytes(b"")
        (bob / "voice.yaml").write_text("persona: bob\n", encoding="utf-8")

        # template, .hidden, ghost — all excluded from the dropdown
        (characters / "template").mkdir()
        (characters / ".hidden").mkdir()
        (characters / "ghost").mkdir()  # neither face nor expressions

        # The scanner locates characters/ via STATIC_DIR.parent.parent, so
        # patching STATIC_DIR pivots the search root without touching the
        # real repo.
        monkeypatch.setattr(server_module, "STATIC_DIR", static_dir)
        yield characters

    def test_alice_listed_procedural_with_voice(self, server_module, fake_repo):
        out = server_module._scan_available_characters()
        alice = next((e for e in out if e["name"] == "alice"), None)
        assert alice is not None
        assert alice["type"] == "procedural-sun"
        assert alice["has_voice"] is True
        assert alice["face_config"] is not None
        assert alice["face_config"].get("body") == "#ff0"

    def test_bob_listed_bitmap_without_voice(self, server_module, fake_repo):
        out = server_module._scan_available_characters()
        bob = next((e for e in out if e["name"] == "bob"), None)
        assert bob is not None
        assert bob["type"] == "bitmap"
        assert bob["has_voice"] is False
        assert bob["face_config"] is None

    def test_template_excluded(self, server_module, fake_repo):
        names = {e["name"] for e in server_module._scan_available_characters()}
        assert "template" not in names

    def test_hidden_dir_excluded(self, server_module, fake_repo):
        names = {e["name"] for e in server_module._scan_available_characters()}
        assert ".hidden" not in names

    def test_renderer_less_dir_excluded(self, server_module, fake_repo):
        """A character dir with neither face.json nor
        expressions/default.png has nothing to render — skip it so the
        dropdown doesn't list a broken option."""
        names = {e["name"] for e in server_module._scan_available_characters()}
        assert "ghost" not in names

    def test_every_entry_has_has_voice_bool(self, server_module, fake_repo):
        """The `has_voice` key is load-bearing — the frontend's dropdown
        label + voice-swap short-circuit both read it. Every entry must
        have it and it must be a bool."""
        for entry in server_module._scan_available_characters():
            assert "has_voice" in entry
            assert isinstance(entry["has_voice"], bool)

    def test_entries_sorted_by_name(self, server_module, fake_repo):
        """The scanner iterates `sorted(chars_root.iterdir())` so entries
        come out alphabetical — stable dropdown order across reloads."""
        names = [e["name"] for e in server_module._scan_available_characters()]
        assert names == sorted(names)

    def test_missing_characters_dir_returns_empty(self, server_module, tmp_path, monkeypatch):
        """If the repo has no characters/ dir at all, the scanner returns
        an empty list rather than raising. Supports fresh-clone text-only
        deploys where voice was never set up."""
        # STATIC_DIR exists, but characters/ sibling doesn't
        static_only = tmp_path / "web" / "static"
        static_only.mkdir(parents=True)
        monkeypatch.setattr(server_module, "STATIC_DIR", static_only)
        assert server_module._scan_available_characters() == []

    def test_malformed_face_json_skipped(self, server_module, tmp_path, monkeypatch):
        """A character with invalid JSON in face.json is skipped (and
        logged) rather than crashing the scanner. Otherwise a single bad
        file could take down the whole dropdown."""
        static_dir = tmp_path / "web" / "static"
        static_dir.mkdir(parents=True)
        characters = tmp_path / "characters"
        characters.mkdir()
        bad = characters / "broken"
        bad.mkdir()
        (bad / "face.json").write_text("{not json at all", encoding="utf-8")
        # Also add a well-formed character so we can verify the rest
        # of the scan continues. Includes voice.yaml so the activatability
        # gate doesn't filter it out.
        good = characters / "good"
        good.mkdir()
        (good / "face.json").write_text(
            json.dumps({"type": "procedural-sun"}), encoding="utf-8",
        )
        (good / "voice.yaml").write_text("persona: good\n", encoding="utf-8")
        monkeypatch.setattr(server_module, "STATIC_DIR", static_dir)
        out = server_module._scan_available_characters()
        names = {e["name"] for e in out}
        assert "broken" not in names
        assert "good" in names
