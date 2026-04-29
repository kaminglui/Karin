"""Ledger load-path migration tests.

When we added `display_summary` + `language` to NormalizedArticle
(translation feature, Phase A), existing articles.json files on disk
don't carry those keys. The loader must fill them with safe defaults
(language="en" since every pre-translation feed was English;
display_summary="" so the translator knows to fall back to other
sources) rather than raise KeyError.

Write-path gets exercised by asdict() + round-trip; we mostly care the
read side is tolerant.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from bridge.news.ledger import Ledger


def _legacy_article_dict(aid: str = "a1") -> dict:
    """Shape an articles.json entry the way Phase 4 wrote it — no
    display_summary, no language."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "article_id": aid,
        "source_id": "bbc",
        "url": f"https://x.test/{aid}",
        "display_title": "Tokyo Mayor Announces Plan",
        "normalized_title": "tokyo mayor announces plan",
        "summary": "the mayor said today.",
        "fingerprint": "a" * 16,
        "wire_attribution": None,
        "published_at": now,
        "fetched_at": now,
    }


class TestLegacyArticleMigration:
    def test_loads_legacy_row_without_new_fields(self, tmp_path):
        ledger = Ledger(tmp_path)
        ledger.articles_path.write_text(
            json.dumps({"a1": _legacy_article_dict("a1")}),
            encoding="utf-8",
        )
        articles = ledger.load_articles()
        assert "a1" in articles
        row = articles["a1"]
        # Defaults applied silently.
        assert row.display_summary == ""
        assert row.language == "en"
        # Existing fields survived untouched.
        assert row.display_title == "Tokyo Mayor Announces Plan"
        assert row.summary == "the mayor said today."

    def test_round_trip_preserves_new_fields(self, tmp_path):
        # After a save, new fields MUST be present on disk so subsequent
        # reads don't hit the legacy path. Guards against asdict() or
        # JSON shape regressing silently.
        ledger = Ledger(tmp_path)
        legacy = _legacy_article_dict("a2")
        legacy["display_summary"] = "Preserved Case Here"
        legacy["language"] = "zh"
        ledger.articles_path.write_text(
            json.dumps({"a2": legacy}),
            encoding="utf-8",
        )
        articles = ledger.load_articles()
        assert articles["a2"].display_summary == "Preserved Case Here"
        assert articles["a2"].language == "zh"

        ledger.save_articles(articles)
        on_disk = json.loads(ledger.articles_path.read_text(encoding="utf-8"))
        assert on_disk["a2"]["display_summary"] == "Preserved Case Here"
        assert on_disk["a2"]["language"] == "zh"
