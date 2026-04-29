"""Tests for the unicode-block language detector.

We care most about:
  - Correct bucketing for the day-one languages (EN + ZH) plus JA/KO
    since the detector already covers them.
  - Conservative behavior on ambiguous / thin input (-> "und") — the
    translation layer depends on this to decide when to ask the LLM.
  - Kana-tiebreaker: any kana forces JP even in mostly-kanji text.
"""
from __future__ import annotations

from bridge.news.detect import SUPPORTED_LANGUAGES, detect_language


class TestDetectLanguage:

    def test_plain_english_headline(self):
        assert detect_language("Trump reacts to latest poll numbers") == "en"

    def test_simplified_chinese_headline(self):
        # Pure Han. No kana -> Chinese.
        assert detect_language("中国央行宣布降息决定") == "zh"

    def test_traditional_chinese_headline(self):
        # Traditional Han also lands in the han block -> "zh".
        assert detect_language("香港政府公布新冠疫情最新措施") == "zh"

    def test_japanese_with_kana_and_kanji(self):
        # Mixed kanji + hiragana is the typical JP headline shape.
        assert detect_language("日本政府が新しい政策を発表した") == "ja"

    def test_pure_hiragana(self):
        assert detect_language("こんにちは、げんきですか") == "ja"

    def test_pure_katakana(self):
        assert detect_language("カタカナノミノテストデスヨ") == "ja"

    def test_japanese_kanji_only_with_kana_marker(self):
        # Even a single kana character flips han -> ja. This is the
        # tiebreaker rule; don't regress it.
        assert detect_language("中国政府の発表") == "ja"

    def test_korean_hangul_syllables(self):
        assert detect_language("한국 정부가 새로운 정책을 발표했다") == "ko"

    def test_short_input_returns_und(self):
        # Below the _MIN_SCRIPT_CHARS floor (3) we refuse to guess.
        assert detect_language("AI") == "und"
        assert detect_language("") == "und"

    def test_all_digits_or_punct_returns_und(self):
        # No script chars at all -> und, even if the string is long.
        assert detect_language("2026-04-16: 42% (!)") == "und"

    def test_mixed_below_dominance_returns_und(self):
        # Exactly 50/50 latin vs han with no kana -> neither bucket
        # clears 55%. Caller should translate or ask the LLM.
        # "AI" + "is" = 4 latin chars; "人工智能" = 4 han chars.
        assert detect_language("AI is 人工智能") == "und"

    def test_latin_headline_with_digits_and_punct_still_english(self):
        # Non-script chars don't count toward the total, so punctuation-
        # heavy English still detects cleanly.
        assert detect_language("U.S. GDP +2.1% in Q1 2026, analysts say") == "en"

    def test_supported_languages_exposes_day_one_pair(self):
        # Contract check — if this ever drops 'en' or 'zh' the
        # translation feature breaks silently.
        assert {"en", "zh"} <= SUPPORTED_LANGUAGES
