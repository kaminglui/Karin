"""Tests for bridge.news.normalize.

Covers:
  - display_title cleanup (bracketed tags, boilerplate, whitespace, case)
  - normalized_title (lowercased form of display_title)
  - summary normalization (HTML stripping, lowercase, whitespace)
  - wire attribution detection (AP, Reuters, AFP + no-match)
  - article_id stability
  - fingerprint stability + short-text fallback
  - end-to-end normalize() producing a coherent NormalizedArticle
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bridge.news.models import RawArticle
from bridge.news.normalize import (
    article_id,
    clean_display_summary,
    clean_display_title,
    compute_fingerprint,
    infer_wire_attribution,
    normalize,
    normalize_summary,
    normalize_title,
)


def _raw(
    title: str = "",
    summary: str = "",
    url: str = "https://x.test/a",
    source: str = "s",
) -> RawArticle:
    now = datetime.now(timezone.utc)
    return RawArticle(
        source_id=source, url=url, title=title, summary=summary,
        published_at=now, fetched_at=now,
    )


# --- display_title / normalized_title --------------------------------------

class TestDisplayTitle:
    def test_strips_bracketed_tag(self):
        assert clean_display_title("Tokyo news [LIVE UPDATES]") == "Tokyo news"

    def test_strips_parenthetical_tag(self):
        assert clean_display_title("(Video) quake in Osaka") == "quake in Osaka"

    def test_strips_boilerplate_prefix(self):
        assert clean_display_title("LIVE UPDATES: market crash") == "market crash"
        assert clean_display_title("Opinion: why we read") == "why we read"

    def test_collapses_whitespace(self):
        assert clean_display_title("hello   world\n\tfoo") == "hello world foo"

    def test_preserves_case_and_apostrophe(self):
        assert clean_display_title("Tokyo's Mayor Speaks") == "Tokyo's Mayor Speaks"

    def test_strips_leading_trailing_punct(self):
        # Trailing separators left over after boilerplate removal should go.
        assert clean_display_title("BREAKING: headline").endswith("headline")


class TestNormalizeTitle:
    def test_lowercases(self):
        assert normalize_title("Tokyo Mayor") == "tokyo mayor"

    def test_composite_cleanup(self):
        assert normalize_title("BREAKING: Big News [LIVE]") == "big news"

    def test_empty_in_empty_out(self):
        assert normalize_title("") == ""


# --- summary ---------------------------------------------------------------

class TestNormalizeSummary:
    def test_strips_html(self):
        out = normalize_summary("<div>Hello <b>world</b></div>")
        assert "<" not in out and "div" not in out
        assert out == "hello world"

    def test_lowercases_and_collapses(self):
        assert normalize_summary("FOO    BAR") == "foo bar"

    def test_empty(self):
        assert normalize_summary("") == ""


# --- wire attribution ------------------------------------------------------

class TestWireAttribution:
    def test_detects_ap_parenthesized(self):
        assert infer_wire_attribution(_raw(title="Story headline (AP)")) == "AP"

    def test_detects_ap_dash_form(self):
        assert infer_wire_attribution(
            _raw(summary="TOKYO, Jan 1 (AP) — news body")
        ) == "AP"

    def test_detects_associated_press_fullname(self):
        assert infer_wire_attribution(
            _raw(summary="According to Associated Press reports...")
        ) == "AP"

    def test_detects_reuters(self):
        assert infer_wire_attribution(_raw(title="Big news (Reuters)")) == "Reuters"

    def test_detects_afp(self):
        assert infer_wire_attribution(_raw(summary="(AFP) — story")) == "AFP"

    def test_no_match_returns_none(self):
        assert infer_wire_attribution(_raw(title="Local blog post")) is None

    def test_unrelated_mention_does_not_false_positive(self):
        # The word "map" should not match AP.
        assert infer_wire_attribution(_raw(title="map of the region")) is None


# --- article_id ------------------------------------------------------------

class TestArticleId:
    def test_stable(self):
        assert article_id("https://x.test/a") == article_id("https://x.test/a")

    def test_different_urls_differ(self):
        assert article_id("https://x.test/a") != article_id("https://x.test/b")

    def test_length_is_16(self):
        assert len(article_id("https://x.test/a")) == 16


# --- fingerprint -----------------------------------------------------------

class TestFingerprint:
    def test_same_text_same_fingerprint(self):
        fp1 = compute_fingerprint("tokyo mayor elected today", "details follow story here")
        fp2 = compute_fingerprint("tokyo mayor elected today", "details follow story here")
        assert fp1 == fp2

    def test_different_text_different_fingerprint(self):
        fp1 = compute_fingerprint("tokyo mayor elected today", "details follow story here")
        fp2 = compute_fingerprint("osaka disaster hits region", "widespread damage reported tonight")
        assert fp1 != fp2

    def test_short_text_falls_back(self):
        fp = compute_fingerprint("hi", "")
        assert isinstance(fp, str)
        assert len(fp) == 16

    def test_length_is_16(self):
        fp = compute_fingerprint("tokyo mayor elected today", "details follow story here")
        assert len(fp) == 16

    def test_thin_content_different_urls_give_different_fingerprints(self):
        # Regression: two articles with identical stock headlines and empty
        # summaries must not hash to the same fingerprint, or the cluster
        # fast path merges them spuriously ("Here's the latest" across
        # multiple NYT rollups was the motivating case).
        fp1 = compute_fingerprint("here's the latest", "", "https://x.test/1")
        fp2 = compute_fingerprint("here's the latest", "", "https://x.test/2")
        assert fp1 != fp2

    def test_thin_content_same_url_gives_same_fingerprint(self):
        # Determinism for the fallback path.
        fp1 = compute_fingerprint("here's the latest", "", "https://x.test/a")
        fp2 = compute_fingerprint("here's the latest", "", "https://x.test/a")
        assert fp1 == fp2


# --- end-to-end normalize() ------------------------------------------------

class TestNormalizeEndToEnd:
    def test_full_conversion(self):
        raw = _raw(
            title="[BREAKING] Tokyo Mayor Announces Plan (AP)",
            summary="<p>The mayor said today...</p>",
            url="https://apnews.com/a/1",
            source="ap",
        )
        n = normalize(raw)

        assert n.article_id == article_id(raw.url)
        assert n.source_id == "ap"
        assert n.url == raw.url

        # Display title: case preserved, [BREAKING] tag gone, (AP) tag gone.
        assert "Tokyo" in n.display_title
        assert "[" not in n.display_title and "(" not in n.display_title

        # Normalized title: lowercased form of display_title.
        assert n.normalized_title == n.display_title.lower()
        assert "tokyo mayor" in n.normalized_title

        # Wire attribution detected even though (AP) was stripped from
        # display_title — infer_wire_attribution reads raw.title/summary.
        assert n.wire_attribution == "AP"

        # Summary stripped of HTML.
        assert "<p>" not in n.summary
        assert n.summary == n.summary.lower()

        # Fingerprint is a hex string of the expected length.
        assert len(n.fingerprint) == 16

    def test_non_wire_article(self):
        raw = _raw(
            title="Local cafe opens new location",
            summary="A story with no wire byline.",
            url="https://blog.test/a",
            source="blog",
        )
        n = normalize(raw)
        assert n.wire_attribution is None
        assert n.display_title == "Local cafe opens new location"

    def test_fingerprint_resilient_to_minor_reordering(self):
        # Two articles with the same bag-of-shingles but a different sentence
        # order should produce the same fingerprint (shingles are set-sorted).
        a = normalize(_raw(
            title="tokyo mayor elected today in historic vote",
            summary="voters turned out in record numbers",
            url="https://x.test/1",
        ))
        b = normalize(_raw(
            title="tokyo mayor elected today in historic vote",
            summary="voters turned out in record numbers",
            url="https://x.test/2",
        ))
        # Different URLs so different article_ids, but identical text ->
        # identical fingerprint. This is the dedup signal the clusterer uses.
        assert a.article_id != b.article_id
        assert a.fingerprint == b.fingerprint


# --- polish: Unicode cleanup --------------------------------------------

class TestUnicodeCleanup:
    """Phase-7 polish: U+FFFD replacement chars (mojibake artifacts) get
    stripped, NFKC normalization runs."""

    def test_replacement_char_stripped_from_display_title(self):
        # "Iran's" arrives mangled as "Iran<U+FFFD>s" -- strip the
        # replacement char to get "Irans" instead of a literal box glyph.
        out = clean_display_title("Iran\ufffds economy slows")
        assert "\ufffd" not in out
        assert "Irans economy slows" == out

    def test_replacement_char_stripped_from_summary(self):
        out = normalize_summary("Orb\ufffdn speaks at the summit")
        assert "\ufffd" not in out
        assert "orbn speaks at the summit" == out

    def test_nfkc_preserves_real_unicode(self):
        # NFKC must not damage legitimate accented characters.
        out = clean_display_title("Pokemon goes global")
        assert out == "Pokemon goes global"

    def test_nfkc_normalizes_full_width_to_ascii(self):
        # Full-width "Ｈｅｌｌｏ" normalizes to "Hello" under NFKC.
        out = clean_display_title("\uff28\uff45\uff4c\uff4c\uff4f world today")
        assert "Hello world today" == out


# --- display_summary + language (translation feature, Phase A) ------------

class TestDisplaySummary:
    def test_preserves_case_and_strips_html(self):
        # Mirrors normalize_summary's HTML/whitespace rules but keeps
        # case. This is what the translated detail path pulls from.
        out = clean_display_summary("<p>The Mayor <b>announced</b> a plan.</p>")
        assert out == "The Mayor announced a plan."

    def test_collapses_whitespace(self):
        assert clean_display_summary("hello   world\n\tfoo") == "hello world foo"

    def test_empty_stays_empty(self):
        assert clean_display_summary("") == ""

    def test_lowercase_sibling_matches(self):
        # normalize_summary should be an exact lowercase of the display
        # sibling — callers that compare the two must stay consistent.
        raw_html = "<div>U.S. GDP +2.1%</div>"
        assert normalize_summary(raw_html) == clean_display_summary(raw_html).lower()


class TestLanguageDetection:
    def test_english_article_tagged_en(self):
        n = normalize(_raw(
            title="Tokyo Mayor Announces Plan",
            summary="The mayor said today...",
            url="https://x.test/en",
        ))
        assert n.language == "en"

    def test_chinese_article_tagged_zh(self):
        n = normalize(_raw(
            title="中国央行宣布降息",
            summary="据新华社报道，央行决定下调利率。",
            url="https://x.test/zh",
        ))
        assert n.language == "zh"

    def test_japanese_article_tagged_ja(self):
        n = normalize(_raw(
            title="日本政府が新しい政策を発表した",
            summary="首相は記者会見で発表した。",
            url="https://x.test/ja",
        ))
        assert n.language == "ja"

    def test_normalize_end_to_end_populates_new_fields(self):
        # Regression guard: display_summary preserved-case, summary
        # lowercased, language detected.
        raw = _raw(
            title="Tokyo Mayor Announces Plan",
            summary="<p>The Mayor said TODAY.</p>",
            url="https://x.test/e2e",
        )
        n = normalize(raw)
        assert n.display_summary == "The Mayor said TODAY."
        assert n.summary == "the mayor said today."
        assert n.language == "en"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
