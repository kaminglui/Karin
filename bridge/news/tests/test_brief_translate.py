"""Phase C tests: translation + detail wiring in build_brief.

Covers:
  - Default args preserve Phase 4.5 behavior (empty detail, no translator
    call). Backward-compat guard — existing callers must not break.
  - Detail source preference: extracted-body > display_summary > "".
  - "Fits" rule: when cluster language == target, translator is never
    called even if present.
  - Translation happy path: different languages + translator produces
    translated headline + detail + topic.
  - Fail-soft: translator that reports translated=False leaves the
    original text + original language in the brief.
  - Cluster-level language mode: majority of member languages wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bridge.news.confidence import (
    _cluster_language,
    _maybe_translate,
    _pick_detail_source,
    _trim_detail,
    build_brief,
)
from bridge.news.extract import ExtractedArticle
from bridge.news.models import (
    ConfidenceState,
    NormalizedArticle,
    Source,
    StoryCluster,
    Tier,
)
from bridge.news.translate import TranslationResult


_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --- fixtures ------------------------------------------------------------

def _article(
    *,
    aid: str = "a1",
    source_id: str = "bbc",
    display_title: str = "Tokyo Mayor Announces Plan",
    display_summary: str = "",
    language: str = "en",
) -> NormalizedArticle:
    return NormalizedArticle(
        article_id=aid,
        source_id=source_id,
        url=f"https://x.test/{aid}",
        display_title=display_title,
        normalized_title=display_title.lower(),
        summary=display_summary.lower(),
        fingerprint=f"fp_{aid}",
        wire_attribution=None,
        published_at=_EPOCH,
        fetched_at=_EPOCH,
        display_summary=display_summary,
        language=language,
    )


def _sources() -> dict[str, Source]:
    return {
        "bbc": Source(id="bbc", name="BBC", domain="bbc.com",
                      tier=Tier.REPUTABLE, ownership_group="bbc",
                      is_wire_service=False),
        "scmp": Source(id="scmp", name="SCMP", domain="scmp.com",
                       tier=Tier.REPUTABLE, ownership_group="scmp",
                       is_wire_service=False),
    }


def _cluster(
    *,
    cluster_id: str = "c1",
    article_ids: list[str],
    centroid_display_title: str = "Tokyo Mayor Announces Plan",
    state: ConfidenceState = ConfidenceState.DEVELOPING,
) -> StoryCluster:
    return StoryCluster(
        cluster_id=cluster_id,
        article_ids=list(article_ids),
        centroid_display_title=centroid_display_title,
        centroid_normalized_title=centroid_display_title.lower(),
        first_seen_at=_EPOCH,
        latest_update_at=_EPOCH,
        last_checked_at=_EPOCH,
        last_state_change_at=_EPOCH,
        state=state,
        is_stale=False,
        independent_confirmation_count=1,
        article_count=1,
        syndicated_article_count=0,
    )


# --- fake translator -----------------------------------------------------

@dataclass
class FakeTranslator:
    """Stand-in for bridge.news.translate.Translator. Records calls
    and returns canned translations. Lets us assert that translate()
    is called the right number of times and with the right args."""

    calls: list[tuple[str, str, str]]
    rules: dict[str, str]            # map source text -> translation
    fail_silently: bool = False      # if True, return translated=False

    def translate(self, text: str, *, target_lang: str, source_lang: str) -> TranslationResult:
        self.calls.append((text, source_lang, target_lang))
        if self.fail_silently:
            return TranslationResult(text=text, translated=False)
        return TranslationResult(
            text=self.rules.get(text, f"[{target_lang}]{text}"),
            translated=True,
            from_cache=False,
        )


# --- backward compat -----------------------------------------------------

class TestBackwardCompat:
    def test_default_args_match_phase_4_5_output(self):
        # No translator, no extracted, target_language stays "en":
        # behavior should be identical to pre-Phase-C. Brief has empty
        # detail, headline untouched, language reflects article lang.
        arts = {"a1": _article(aid="a1", language="en")}
        c = _cluster(article_ids=["a1"])
        b = build_brief(c, arts, _sources())
        assert b.detail == ""
        assert b.headline == "Tokyo Mayor Announces Plan"
        assert b.language == "en"

    def test_no_translator_keeps_original_language(self):
        # Article in Chinese, target=en, but no translator provided:
        # fall through to original. Caller opted out by passing None.
        arts = {"a1": _article(
            aid="a1", display_title="\u4e1c\u4eac\u5e02\u957f\u5ba3\u5e03\u8ba1\u5212",
            language="zh",
        )}
        c = _cluster(
            article_ids=["a1"],
            centroid_display_title="\u4e1c\u4eac\u5e02\u957f\u5ba3\u5e03\u8ba1\u5212",
        )
        b = build_brief(c, arts, _sources(), target_language="en", translator=None)
        # Original headline + brief.language reflects untranslated state.
        assert b.headline == "\u4e1c\u4eac\u5e02\u957f\u5ba3\u5e03\u8ba1\u5212"
        assert b.language == "zh"


# --- detail source picking ----------------------------------------------

class TestPickDetailSource:
    def test_prefers_extracted_over_summary(self):
        art = _article(aid="a1", display_summary="RSS blurb one sentence.")
        extracted = {
            "a1": ExtractedArticle(
                article_id="a1",
                text="Full article body begins here. It has multiple "
                     "sentences and much more context than the RSS blurb.",
                title="", author="", date="", error="",
                attempted_at=_EPOCH,
            )
        }
        c = _cluster(article_ids=["a1"])
        picked = _pick_detail_source(c, [art], extracted)
        assert picked.startswith("Full article body")

    def test_falls_back_to_display_summary_when_no_extracted(self):
        art = _article(aid="a1", display_summary="RSS blurb one sentence.")
        c = _cluster(article_ids=["a1"])
        picked = _pick_detail_source(c, [art], extracted=None)
        assert picked == "RSS blurb one sentence."

    def test_falls_back_to_display_summary_when_extracted_empty(self):
        # Extraction attempted but returned empty (paywall, no body).
        # Shouldn't force the empty string — keep the RSS fallback.
        art = _article(aid="a1", display_summary="Fallback text.")
        extracted = {
            "a1": ExtractedArticle(
                article_id="a1", text="", title="", author="", date="",
                error="paywall", attempted_at=_EPOCH,
            )
        }
        c = _cluster(article_ids=["a1"])
        picked = _pick_detail_source(c, [art], extracted)
        assert picked == "Fallback text."

    def test_prefers_centroid_member_for_text(self):
        # Centroid-matching member wins the detail slot when multiple
        # members have extracted text.
        a1 = _article(aid="a1", display_title="Other headline")
        a2 = _article(aid="a2", display_title="Tokyo Mayor Announces Plan")
        extracted = {
            "a1": ExtractedArticle(
                article_id="a1", text="Wrong article.",
                title="", author="", date="", error="", attempted_at=_EPOCH,
            ),
            "a2": ExtractedArticle(
                article_id="a2", text="Correct centroid article body.",
                title="", author="", date="", error="", attempted_at=_EPOCH,
            ),
        }
        c = _cluster(
            article_ids=["a1", "a2"],
            centroid_display_title="Tokyo Mayor Announces Plan",
        )
        picked = _pick_detail_source(c, [a1, a2], extracted)
        assert picked == "Correct centroid article body."

    def test_empty_when_nothing_available(self):
        art = _article(aid="a1", display_summary="")
        c = _cluster(article_ids=["a1"])
        assert _pick_detail_source(c, [art], extracted=None) == ""


class TestTrimDetail:
    def test_caps_at_max_sentences(self):
        long_text = (
            "First sentence. Second sentence. Third sentence. "
            "Fourth sentence that should be dropped."
        )
        out = _trim_detail(long_text)
        assert "Fourth" not in out
        assert "Third" in out

    def test_hard_char_cap_with_ellipsis(self):
        # No sentence boundaries within range -> hard cut with ellipsis.
        single = "word " * 200
        out = _trim_detail(single)
        # Hard cap forces an ellipsis since there's no sentence stop.
        assert len(out) <= 321   # max + 1 for ellipsis
        assert out.endswith("\u2026") or out.endswith(".")

    def test_cjk_sentence_punctuation_splits(self):
        zh = "\u4eca\u5929\u4e0b\u96e8\u3002\u660e\u5929\u6674\u3002\u540e\u5929\u96ea\u3002\u5927\u540e\u5929\u4e0d\u8be5\u51fa\u73b0\u3002"
        out = _trim_detail(zh)
        # Fourth JP/ZH sentence should have been dropped.
        assert "\u5927\u540e\u5929" not in out


# --- cluster language ----------------------------------------------------

class TestClusterLanguage:
    def test_majority_wins(self):
        arts = [
            _article(aid="a", language="en"),
            _article(aid="b", language="en"),
            _article(aid="c", language="zh"),
        ]
        assert _cluster_language(arts) == "en"

    def test_und_members_ignored(self):
        arts = [
            _article(aid="a", language="und"),
            _article(aid="b", language="zh"),
        ]
        assert _cluster_language(arts) == "zh"

    def test_all_und_falls_back_to_en(self):
        arts = [
            _article(aid="a", language="und"),
            _article(aid="b", language="und"),
        ]
        assert _cluster_language(arts) == "en"

    def test_empty_cluster_falls_back_to_en(self):
        assert _cluster_language([]) == "en"


# --- fits rule + translation ---------------------------------------------

class TestFitsRule:
    def test_same_language_skips_translator(self):
        translator = FakeTranslator(calls=[], rules={})
        arts = {"a1": _article(aid="a1", language="en")}
        c = _cluster(article_ids=["a1"])
        b = build_brief(
            c, arts, _sources(),
            target_language="en", translator=translator,
        )
        # No calls: source == target short-circuits inside _maybe_translate
        # BEFORE ever reaching the translator. This is the "fits" rule.
        assert translator.calls == []
        assert b.headline == "Tokyo Mayor Announces Plan"
        assert b.language == "en"

    def test_und_language_skips_translator_at_helper_level(self):
        # _cluster_language collapses all-"und" clusters to "en" before
        # build_brief ever sees "und", so we can't exercise the "und"
        # branch via build_brief. Test _maybe_translate directly instead
        # — this pins the helper's defensive skip for any future caller
        # that might pass "und" through.
        translator = FakeTranslator(calls=[], rules={})
        text, lang = _maybe_translate(
            "Tokyo Mayor", source_lang="und", target_lang="zh", translator=translator,
        )
        assert translator.calls == []
        assert text == "Tokyo Mayor"
        assert lang == "zh"


class TestTranslationHappyPath:
    def test_headline_detail_and_topic_all_translated(self):
        translator = FakeTranslator(
            calls=[],
            rules={
                "Tokyo Mayor Announces Plan": "\u4e1c\u4eac\u5e02\u957f\u5ba3\u5e03\u8ba1\u5212",
                "Tokyo Mayor Announces Plan here.": "\u8fd9\u91cc\u4e1c\u4eac\u5e02\u957f\u5ba3\u5e03\u8ba1\u5212\u3002",
            },
        )
        arts = {"a1": _article(
            aid="a1",
            display_title="Tokyo Mayor Announces Plan",
            display_summary="Tokyo Mayor Announces Plan here.",
            language="en",
        )}
        c = _cluster(article_ids=["a1"])
        b = build_brief(
            c, arts, _sources(),
            target_language="zh", translator=translator,
        )
        # Three translation calls: headline, detail, topic.
        assert len(translator.calls) == 3
        assert b.headline == "\u4e1c\u4eac\u5e02\u957f\u5ba3\u5e03\u8ba1\u5212"
        assert b.detail == "\u8fd9\u91cc\u4e1c\u4eac\u5e02\u957f\u5ba3\u5e03\u8ba1\u5212\u3002"
        # Topic short feeds voice_line; it should also be in Chinese.
        assert b.language == "zh"
        assert "Tokyo" not in b.voice_line   # No untranslated English left.

    def test_voice_line_uses_translated_topic(self):
        translator = FakeTranslator(
            calls=[],
            rules={
                "Tokyo Mayor": "\u4e1c\u4eac\u5e02\u957f",     # topic_short
                "Tokyo Mayor Announces Plan":
                    "\u4e1c\u4eac\u5e02\u957f\u5ba3\u5e03\u8ba1\u5212",
            },
        )
        arts = {"a1": _article(
            aid="a1",
            display_title="Tokyo Mayor",  # short so topic_short == full title
            language="en",
        )}
        c = _cluster(
            article_ids=["a1"],
            centroid_display_title="Tokyo Mayor",
        )
        b = build_brief(
            c, arts, _sources(),
            target_language="zh", translator=translator,
        )
        # Voice line template embeds topic_short. Translated topic_short
        # should appear in the voice_line.
        assert "\u4e1c\u4eac\u5e02\u957f" in b.voice_line


class TestTranslationFailSoft:
    def test_translated_false_keeps_original_text_and_language(self):
        # Translator says "I didn't translate, here's the original" —
        # brief should hold original text and mark the source language.
        translator = FakeTranslator(calls=[], rules={}, fail_silently=True)
        arts = {"a1": _article(
            aid="a1",
            display_title="Tokyo Mayor Announces Plan",
            language="en",
        )}
        c = _cluster(article_ids=["a1"])
        b = build_brief(
            c, arts, _sources(),
            target_language="zh", translator=translator,
        )
        # Calls were made (translator tried) but each returned translated=False.
        assert len(translator.calls) == 2   # headline + topic (no detail: summary is empty)
        assert b.headline == "Tokyo Mayor Announces Plan"
        # Language reflects fallback state: we tried but didn't succeed,
        # so the brief is still in source language.
        assert b.language == "en"
