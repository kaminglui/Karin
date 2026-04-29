"""Tests for the digest subsystem.

Verifies: section collection filters, headline composition, persistence
roundtrip, graceful handling when a subsystem is missing or raises, and
the on-demand build-if-absent behavior.

No live network or LLM. Subsystems are stub objects that return canned
data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bridge.digest.models import DigestSnapshot
from bridge.digest.service import (
    DigestService,
    _compose_headline,
    _snap_from_dict,
    _snap_to_dict,
)


# ---- stubs ---------------------------------------------------------------


@dataclass
class _FakeBrief:
    """Mirror of the real ``bridge.news.models.StoryBrief`` field
    names. The digest service reads `headline`, `top_sources`, and
    `independent_confirmation_count` first; older fixtures used
    `title` / `source_display_names` / `independent_confirmations`,
    which silently returned the empty/zero fallback and looked like
    they were testing real behavior. Keeping this fixture in sync
    with the production schema is the only way to detect a future
    drift.
    """
    cluster_id: str
    state: str
    headline: str
    voice_line: str
    top_sources: list[str] = field(default_factory=list)
    independent_confirmation_count: int = 1
    latest_update_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _LegacyFakeBrief:
    """Pre-rename brief shape — used to verify that the digest
    service's `getattr(... or fallback)` chain still resolves the
    correct values when an older brief object is fed in. If this
    test starts failing, the fallback chain in digest/service.py
    has regressed."""
    cluster_id: str
    state: str
    title: str
    voice_line: str
    source_display_names: list[str] = field(default_factory=list)
    independent_confirmations: int = 1
    latest_update_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class _FakeNewsService:
    def __init__(self, briefs):
        self.briefs = briefs

    def get_news(self, *, topic=None, max_results=3):
        return self.briefs


@dataclass
class _FakeAlert:
    alert_id: str
    level: object   # AlertLevel-like; must have .name
    category: object
    title: str
    reasoning_bullets: list[str]
    created_at: datetime
    cooldown_until: datetime


class _FakeLevel:
    def __init__(self, name):
        self.name = name


class _FakeCategory:
    def __init__(self, value):
        self.value = value


class _FakeAlertService:
    def __init__(self, alerts):
        self.alerts = alerts

    def get_active_alerts(self, max_results=10):
        return self.alerts


@dataclass
class _FakeSnapshot:
    id: str
    label: str
    latest_value: float | None
    change_1d_pct: float | None = None
    change_1w_pct: float | None = None
    direction_label: str | None = None
    shock_label: str | None = None


class _FakeTrackerService:
    def __init__(self, snaps):
        self.snaps = snaps

    def get_trackers(self):
        return self.snaps


# ---- collection filters --------------------------------------------------


class TestNewsCollection:
    def test_keeps_confirmed_and_provisional_only(self, tmp_path):
        now = datetime.now(timezone.utc)
        briefs = [
            _FakeBrief("c1", "confirmed", "Big story A", "AP reports A."),
            _FakeBrief("c2", "provisionally_confirmed", "Maybe story B", "BBC reports B."),
            _FakeBrief("c3", "developing", "Rumor C", "One source says C."),
        ]
        svc = DigestService(
            news_service=_FakeNewsService(briefs),
            digest_dir=tmp_path,
        )
        snap = svc.build(now=now)
        # "developing" filtered out
        assert [n.cluster_id for n in snap.news] == ["c1", "c2"]

    def test_empty_when_service_missing(self, tmp_path):
        """No news_service wired → empty news section, no crash."""
        svc = DigestService(digest_dir=tmp_path)
        assert svc.build().news == []

    def test_exception_in_news_collection_is_swallowed(self, tmp_path):
        class _Boom:
            def get_news(self, **_):
                raise RuntimeError("nope")
        svc = DigestService(news_service=_Boom(), digest_dir=tmp_path)
        assert svc.build().news == []

    def test_canonical_storybrief_fields_propagate(self, tmp_path):
        """Catch the regression where digest displayed '0 independent
        sources' for every story. Root cause was the service reading
        `title` / `independent_confirmations` from a brief whose real
        attribute names are `headline` / `independent_confirmation_count`.
        The fix adds a multi-name fallback. This test pins it: a brief
        carrying ONLY the canonical names must produce a digest item
        with the headline + outlet list + count visible — not the
        empty-string / 0 / [] fallbacks."""
        briefs = [_FakeBrief(
            cluster_id="c1",
            state="confirmed",
            headline="Pope speaks at Vatican summit",
            voice_line="AP and BBC report Pope speaks at Vatican summit.",
            top_sources=["AP", "BBC", "Guardian"],
            independent_confirmation_count=3,
        )]
        svc = DigestService(
            news_service=_FakeNewsService(briefs),
            digest_dir=tmp_path,
        )
        item = svc.build().news[0]
        assert item.title == "Pope speaks at Vatican summit"
        assert item.sources == ["AP", "BBC", "Guardian"]
        assert item.independent_confirmations == 3

    def test_legacy_storybrief_fields_still_resolve(self, tmp_path):
        """Coverage for the digest service's fallback chain that
        accepts the OLD attribute names (`title`, `source_display_names`,
        `independent_confirmations`). Lets a stale brief fixture or
        downstream rename land without losing the digest display.
        Pair with the canonical-fields test above; together they
        prove both branches of the fallback are live."""
        briefs = [_LegacyFakeBrief(
            cluster_id="c1",
            state="confirmed",
            title="Pope speaks at Vatican summit",
            voice_line="AP and BBC report.",
            source_display_names=["AP", "BBC"],
            independent_confirmations=2,
        )]
        svc = DigestService(
            news_service=_FakeNewsService(briefs),
            digest_dir=tmp_path,
        )
        item = svc.build().news[0]
        assert item.title == "Pope speaks at Vatican summit"
        assert item.sources == ["AP", "BBC"]
        assert item.independent_confirmations == 2


class TestAlertCollection:
    def test_collects_alerts(self, tmp_path):
        now = datetime.now(timezone.utc)
        alerts = [
            _FakeAlert(
                alert_id="a1", level=_FakeLevel("CRITICAL"),
                category=_FakeCategory("weather"),
                title="Tornado Warning", reasoning_bullets=["Extreme"],
                created_at=now, cooldown_until=now + timedelta(hours=1),
            ),
        ]
        svc = DigestService(
            alert_service=_FakeAlertService(alerts), digest_dir=tmp_path,
        )
        snap = svc.build(now=now)
        assert len(snap.alerts) == 1
        assert snap.alerts[0].level == "CRITICAL"
        assert snap.alerts[0].category == "weather"

    def test_empty_alerts_is_fine(self, tmp_path):
        svc = DigestService(
            alert_service=_FakeAlertService([]), digest_dir=tmp_path,
        )
        assert svc.build().alerts == []


class TestTrackerCollection:
    def test_filters_small_moves(self, tmp_path):
        """Anything under the 1% daily threshold is noise — drop it."""
        snaps = [
            _FakeSnapshot("gold_usd", "Gold", 4800.0, change_1d_pct=2.5),
            _FakeSnapshot("usd_cny", "USD/CNY", 7.2, change_1d_pct=0.1),
            _FakeSnapshot("gas_retail", "Gas", 4.1, change_1d_pct=-1.3),
        ]
        svc = DigestService(
            tracker_service=_FakeTrackerService(snaps), digest_dir=tmp_path,
        )
        ids = [t.tracker_id for t in svc.build().trackers]
        assert "gold_usd" in ids
        assert "gas_retail" in ids
        assert "usd_cny" not in ids   # 0.1% move filtered out

    def test_sorts_by_abs_change(self, tmp_path):
        snaps = [
            _FakeSnapshot("a", "A", 1.0, change_1d_pct=1.5),
            _FakeSnapshot("b", "B", 1.0, change_1d_pct=-3.0),
            _FakeSnapshot("c", "C", 1.0, change_1d_pct=2.0),
        ]
        svc = DigestService(
            tracker_service=_FakeTrackerService(snaps), digest_dir=tmp_path,
        )
        ids = [t.tracker_id for t in svc.build().trackers]
        assert ids == ["b", "c", "a"]   # biggest abs-change first

    def test_shock_label_passes_even_if_small_move(self, tmp_path):
        """Phase 5.2 shock label on a series means something unusual
        happened even if the percent-change itself is small. Keep it
        so the digest shows anomalies."""
        snaps = [
            _FakeSnapshot(
                "quiet", "Quiet", 1.0,
                change_1d_pct=0.1, shock_label="1d-shock",
            ),
        ]
        svc = DigestService(
            tracker_service=_FakeTrackerService(snaps), digest_dir=tmp_path,
        )
        assert [t.tracker_id for t in svc.build().trackers] == ["quiet"]


# ---- headline composition ------------------------------------------------


class TestHeadline:
    def test_empty_day(self):
        assert "Quiet day" in _compose_headline([], [], [])

    def test_counts_confirmed_vs_developing(self):
        confirmed = [
            _FakeBriefLike("confirmed"), _FakeBriefLike("confirmed"),
        ]
        headline = _compose_headline(
            _to_news_items(confirmed), [], [],
        )
        assert "2 confirmed" in headline

    def test_critical_alerts_beat_count(self):
        from bridge.digest.models import DigestAlertItem
        now = datetime.now(timezone.utc)
        alerts = [
            DigestAlertItem(
                alert_id=f"a{i}", level=level, category="weather",
                title="t", reasoning_bullets=[],
                created_at=now, cooldown_until=now,
            )
            for i, level in enumerate(["CRITICAL", "ADVISORY"])
        ]
        headline = _compose_headline([], alerts, [])
        assert "1 critical alert" in headline

    def test_tracker_mention_uses_biggest_mover(self):
        from bridge.digest.models import DigestTrackerItem
        trackers = [
            DigestTrackerItem(
                tracker_id="gold", label="Gold", latest_value=4800.0,
                change_1d_pct=3.5, change_1w_pct=None,
                direction_label="up", shock_label=None,
            ),
        ]
        headline = _compose_headline([], [], trackers)
        assert "Gold" in headline
        assert "up" in headline
        assert "3.5" in headline


# ---- persistence roundtrip -----------------------------------------------


class TestPersistence:
    def test_build_and_persist_then_latest_roundtrip(self, tmp_path):
        svc = DigestService(
            news_service=_FakeNewsService([]),
            alert_service=_FakeAlertService([]),
            tracker_service=_FakeTrackerService([]),
            digest_dir=tmp_path,
        )
        snap = svc.build_and_persist()
        # File exists
        expected_path = tmp_path / f"{snap.date_key}.json"
        assert expected_path.exists()
        # latest() returns the same shape
        loaded = svc.latest()
        assert loaded is not None
        assert loaded.date_key == snap.date_key
        assert loaded.headline == snap.headline

    def test_for_date_missing_returns_none(self, tmp_path):
        svc = DigestService(digest_dir=tmp_path)
        assert svc.for_date("1999-01-01") is None

    def test_latest_missing_returns_none(self, tmp_path):
        svc = DigestService(digest_dir=tmp_path)
        assert svc.latest() is None

    def test_tmp_file_cleanup_on_successful_write(self, tmp_path):
        """The atomic-ish write uses .tmp → rename; no stray .tmp files
        should remain after a successful build_and_persist."""
        svc = DigestService(
            news_service=_FakeNewsService([]), digest_dir=tmp_path,
        )
        svc.build_and_persist()
        tmps = list(tmp_path.glob("*.tmp"))
        assert tmps == []


# ---- snapshot dict roundtrip ---------------------------------------------


class TestSnapshotDictRoundtrip:
    def test_empty_snapshot(self):
        snap = DigestSnapshot(
            generated_at=datetime(2026, 4, 15, 12, tzinfo=timezone.utc),
            date_key="2026-04-15", headline="Quiet day.",
        )
        assert _snap_from_dict(_snap_to_dict(snap)).headline == "Quiet day."

    def test_full_snapshot(self):
        from bridge.digest.models import (
            DigestAlertItem, DigestNewsItem, DigestTrackerItem,
        )
        now = datetime(2026, 4, 15, 12, tzinfo=timezone.utc)
        snap = DigestSnapshot(
            generated_at=now, date_key="2026-04-15",
            headline="Today: 1 confirmed story, Gold up 2.5%.",
            news=[
                DigestNewsItem(
                    cluster_id="c1", state="confirmed",
                    title="Big story", voice_line="AP and BBC report...",
                    sources=["AP", "BBC"],
                    independent_confirmations=3,
                    latest_update_at=now,
                ),
            ],
            alerts=[
                DigestAlertItem(
                    alert_id="a1", level="ADVISORY", category="travel",
                    title="Travel advisory", reasoning_bullets=["Level 3"],
                    created_at=now, cooldown_until=now + timedelta(hours=6),
                ),
            ],
            trackers=[
                DigestTrackerItem(
                    tracker_id="gold_usd", label="Gold",
                    latest_value=4800.0, change_1d_pct=2.5,
                    change_1w_pct=5.1, direction_label="up",
                    shock_label=None,
                ),
            ],
        )
        loaded = _snap_from_dict(_snap_to_dict(snap))
        assert loaded.date_key == "2026-04-15"
        assert len(loaded.news) == 1
        assert loaded.news[0].sources == ["AP", "BBC"]
        assert len(loaded.alerts) == 1
        assert loaded.alerts[0].level == "ADVISORY"
        assert len(loaded.trackers) == 1
        assert loaded.trackers[0].change_1d_pct == 2.5


# ---- helpers (for headline tests) ----------------------------------------

@dataclass
class _FakeBriefLike:
    state: str


def _to_news_items(fakes):
    from bridge.digest.models import DigestNewsItem
    now = datetime.now(timezone.utc)
    return [
        DigestNewsItem(
            cluster_id=f"c{i}", state=f.state, title="t",
            voice_line="v", sources=[],
            independent_confirmations=2, latest_update_at=now,
        )
        for i, f in enumerate(fakes)
    ]
