"""NewsService: the entry point that ties fetch -> normalize -> cluster ->
confidence -> ledger into a few public methods the tool layer can call.

Design notes:

- Lazy refresh with a TTL gate (default 15 min). `get_news()` calls
  `ingest_latest(force=False)` first; the gate skips the fetch if the
  last successful ingest is recent. The gate reads the `ingest_ok` event
  from `events.jsonl` — no separate index needed.

- On a real (non-cached) ingest, only clusters that gain new articles
  are re-clustered and rescored. Clusters with no new members keep
  their previous state + last_state_change_at.

- Independently, `get_news()` rescores ALL clusters with the current
  clock before ranking. This keeps `is_stale` accurate at read time
  even when ingest was cached. Cheap: ~200 clusters is microseconds.

- Ranking: (state_priority DESC, latest_update_at DESC). Confirmed
  stories beat provisional beat developing; within a tier, newer wins.

- Topic filtering: case-insensitive substring match on
  `centroid_normalized_title`. If no cluster matches, fall back to
  ranking all clusters — the caller gets the top recent story instead
  of an empty list. That's almost always what a voice UX wants.

- The service does NOT run a background scheduler in V1. If you want
  periodic polling, cron a process that calls `ingest_latest(force=True)`
  — this service is single-writer-safe within one process but not
  across concurrent processes.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bridge.news.cluster import cluster_articles
from bridge.news.confidence import (
    DEFAULT_STALE_THRESHOLD_HOURS,
    build_brief,
    rescore_cluster,
)
from bridge.news.fetch import fetch_all
from bridge.news.newsapi import fetch_newsapi
from bridge.news.extract import EXTRACTION_AVAILABLE, ExtractStore, extract_missing
from bridge.news.learned_store import (
    BUCKET_KEYS as LEARNED_BUCKET_KEYS,
    DEFAULT_TTL_DAYS as LEARNED_DEFAULT_TTL_DAYS,
    LearnedStore,
    record as learned_record,
    sweep_expired as learned_sweep_expired,
)
from bridge.news.ledger import Ledger
from bridge.news.models import (
    ConfidenceState,
    Feed,
    IngestResult,
    NormalizedArticle,
    Source,
    StoryBrief,
    StoryCluster,
    Tier,
)
from bridge.news.normalize import normalize_many
from bridge.news.preferences import (
    Preferences,
    compute_score,
    load_preferences,
    match_watchlist_items,
)


@dataclass(frozen=True)
class KeywordLearnConfig:
    """Runtime config for the Phase E keyword-learning pass.

    Kept off the NewsService __init__ signature itself so the existing
    positional-keyword contract stays backwards compatible; the pass
    is gated on `news_keyword_learn` in config/features.yaml and built
    lazily in get_default_service().
    """
    base_url: str
    model: str
    request_timeout: float = 60.0
    max_calls_per_cycle: int = 3
    min_clusters_per_bucket: int = 3
    max_fragments_per_call: int = 30
    ttl_days: int = LEARNED_DEFAULT_TTL_DAYS


# Map the preferences side's singular watchlist_type ("region") to the
# plural bucket key used by the learned store ("regions"). Keeps the
# two schemas from drifting — a single table both sides read from.
_TYPE_SINGULAR_TO_BUCKET = {
    "region": "regions",
    "topic":  "topics",
    "event":  "events",
}

log = logging.getLogger("bridge.news.service")

# Tunable via config/tuning.yaml under `news.service.*`.
from bridge import tuning as _tuning
DEFAULT_FETCH_TTL_MINUTES = _tuning.get("news.service.fetch_ttl_minutes", 15)
DEFAULT_RETENTION_DAYS = int(_tuning.get("news.service.retention_days", 30))
DEFAULT_MAX_RESULTS = _tuning.get("news.service.default_max_results", 3)

# Ranking weights for cluster state. Higher = better; confirmed beats
# provisional beats developing. Used as the primary sort key in
# get_news, with latest_update_at as the tiebreaker.
_STATE_PRIORITY: dict[ConfidenceState, int] = {
    ConfidenceState.CONFIRMED: 2,
    ConfidenceState.PROVISIONALLY_CONFIRMED: 1,
    ConfidenceState.DEVELOPING: 0,
}


class NewsService:
    """High-level interface for the news subsystem.

    Holds the ledger, source registry, and feed list. Stateless between
    calls — everything persists in the ledger, so restarting the process
    is safe.
    """

    def __init__(
        self,
        *,
        ledger: Ledger,
        sources: dict[str, Source],
        feeds: list[Feed],
        preferences: Preferences | None = None,
        fetch_ttl_minutes: int = DEFAULT_FETCH_TTL_MINUTES,
        stale_threshold_hours: int = DEFAULT_STALE_THRESHOLD_HOURS,
        cross_verifier=None,
        translator=None,
        keyword_learn_cfg: "KeywordLearnConfig | None" = None,
        learned_store_path: "Path | None" = None,
    ) -> None:
        self._ledger = ledger
        self._sources = sources
        self._feeds = feeds
        # None and a disabled Preferences are treated identically: the
        # score function returns 0 across the board, collapsing the
        # ranking tuple to Phase-4 order (state, recency).
        self._preferences = preferences if preferences is not None else Preferences(enabled=False)
        self._fetch_ttl_minutes = fetch_ttl_minutes
        self._stale_threshold_hours = stale_threshold_hours
        # Optional LLM verifier for the Phase B cross-check layer.
        # When None OR the `news_cross_verify` feature flag is off,
        # the cross-verify path is skipped entirely — the existing
        # lexical clustering remains authoritative.
        self._cross_verifier = cross_verifier
        # Optional translator for the Phase C translation path.
        # When None, build_brief runs with translator=None and returns
        # briefs in the article's original language, matching Phase 4.5
        # behavior exactly. Feature-flagged at build time in
        # get_default_service.
        self._translator = translator
        # Sidecar store for extracted article text. Lives next to the
        # news ledger so backups / wipes cover both.
        self._extract_store = ExtractStore(
            ledger.data_dir / "extracted.json"
        )
        # Phase E: LLM-learned watchlist entities. Always open the
        # store (empty file is harmless); only the run path is gated
        # on `keyword_learn_cfg`. This keeps the read endpoint viable
        # even when the feature flag is off — the UI can show an
        # empty-state instead of 500ing.
        # Phase H: learned keywords are PER-PROFILE (they're learned
        # from this profile's watchlist buckets). Default falls back
        # to the legacy shared location next to the ledger so existing
        # tests that don't know about profiles still work.
        self._learned_store = LearnedStore(
            learned_store_path
            if learned_store_path is not None
            else ledger.data_dir / "learned_keywords.json"
        )
        self._keyword_learn_cfg = keyword_learn_cfg
        # In-memory caches for the three big JSON stores. Re-parsing
        # articles (~1 MB), clusters (~1 MB), and extracted (~3+ MB)
        # on every get_news call dominates latency on the Jetson
        # (SD-card reads + Python JSON parse). Single-writer design
        # means we can safely memoize between calls and invalidate on
        # known mutation points: ingest_latest (new articles),
        # save_articles / save_clusters (ledger writes),
        # extract_missing (extract_store writes), and refresh_and_rebuild.
        self._cached_articles: dict[str, "NormalizedArticle"] | None = None
        self._cached_clusters: dict[str, "StoryCluster"] | None = None
        self._cached_extracted = None   # dict[str, ExtractedArticle] | None
        # Graph payload cache. build_news_graph is O(n_clusters · n_labels)
        # on the news data it's computed over (~800 clusters × ~90 labels
        # × spaCy + combined-regex pass). Caching the finished JSON
        # between ingests turns every repeat UI read from ~5 s into a
        # dict-copy. Invalidated whenever the underlying clusters /
        # articles / learned-keywords change.
        self._cached_graph: dict | None = None
        # Lock that guards every cache read AND every invalidation
        # write. Without it, the news poller (background thread) can
        # null one of the cache fields while a UI request is mid-
        # iteration over the dict it just released. Not a contention
        # bottleneck — the protected critical sections are O(1)
        # (assign-or-load) — but it does prevent the poller from
        # observing a half-built cache and a reader from observing a
        # mid-invalidation None. Re-entrant so internal helpers can
        # call each other without self-deadlock.
        self._cache_lock = threading.RLock()

    # --- cache helpers ---------------------------------------------------

    def _invalidate_cache(self) -> None:
        """Drop memoized reads. Called after any ledger / extract write
        or ingest. Cheap — next read repopulates lazily."""
        with self._cache_lock:
            self._cached_articles = None
            self._cached_clusters = None
            self._cached_extracted = None
            self._cached_graph = None

    def _get_articles(self) -> "dict[str, NormalizedArticle]":
        with self._cache_lock:
            if self._cached_articles is None:
                self._cached_articles = self._ledger.load_articles()
            return self._cached_articles

    def _get_clusters(self) -> "dict[str, StoryCluster]":
        with self._cache_lock:
            if self._cached_clusters is None:
                self._cached_clusters = self._ledger.load_clusters()
            return self._cached_clusters

    def _get_extracted(self):
        with self._cache_lock:
            if self._cached_extracted is None:
                self._cached_extracted = self._extract_store.load()
            return self._cached_extracted

    # --- ingestion -------------------------------------------------------

    def ingest_latest(self, *, force: bool = False) -> IngestResult:
        """Fetch feeds (if TTL allows), persist new articles, update clusters.

        When the TTL gate skips the fetch, returns an IngestResult with
        `skipped_due_to_cache=True` and all counts zero. On a real fetch,
        merges new articles into the existing ledger by article_id and
        runs cluster_articles + rescore only for affected clusters.
        """
        if not force and self._is_within_ttl():
            return IngestResult(
                fetched_feeds=0,
                failed_feeds=0,
                new_articles=0,
                new_article_ids=[],
                skipped_due_to_cache=True,
            )

        raw_articles, ok, fail = fetch_all(self._feeds)
        # Wire-tier feed from NewsAPI. Silently no-ops when the env
        # key is unset, so this is safe to call unconditionally — the
        # RSS path continues to work as before. When the key IS set,
        # we pick up AP/Reuters/Bloomberg (etc.) articles and they
        # flow through normalize → cluster just like RSS articles,
        # tagged with the correct tier via sources.json.
        wire_articles = fetch_newsapi(self._sources)
        raw_articles = list(raw_articles) + wire_articles
        normalized = normalize_many(raw_articles)

        existing = self._get_articles()
        new_ids: list[str] = []
        for a in normalized:
            if a.article_id not in existing:
                existing[a.article_id] = a
                new_ids.append(a.article_id)

        if new_ids:
            self._ledger.save_articles(existing)
            # `existing` is still the live cache; no need to invalidate
            # the articles cache since it's the same dict we just
            # mutated. Clusters + extracted WILL be touched below, so
            # drop those. Graph cache also drops since cluster members
            # shifted. Hold the cache lock so a concurrent reader
            # never observes a half-cleared cache state.
            with self._cache_lock:
                self._cached_clusters = None
                self._cached_extracted = None
                self._cached_graph = None
        self._ledger.append_event(
            "ingest_ok",
            {"fetched_feeds": ok, "failed_feeds": fail, "new_articles": len(new_ids)},
        )

        if new_ids:
            self._cluster_and_rescore(existing, now=datetime.now(timezone.utc))

        # Full-article extraction (opt-in, bounded). Runs even on zero-
        # new ingests so a previous cycle's retry queue gets drained.
        # Fail-soft: a missing trafilatura or a flaky publisher site
        # never breaks the ingest result above.
        try:
            summary = extract_missing(existing, self._extract_store)
            if summary.get("attempted"):
                self._ledger.append_event("extract_ok", summary)
                with self._cache_lock:
                    self._cached_extracted = None   # disk changed
                    self._cached_graph = None       # affects detail + graph payload
                log.info(
                    "extract: pending=%d attempted=%d succeeded=%d failed=%d",
                    summary["pending"], summary["attempted"],
                    summary["succeeded"], summary["failed"],
                )
        except Exception as e:
            log.warning("extract pass failed (non-fatal): %s", e)

        # Phase E keyword-learning pass. Feature-flag gated via the
        # presence of keyword_learn_cfg — when absent, this whole
        # block is a no-op and ingest stays identical to pre-E behavior.
        # Wrapped in its own try/except so a flaky Ollama can't break
        # the ingest result or the extraction bookkeeping above.
        if self._keyword_learn_cfg is not None:
            try:
                self._run_keyword_learning_pass(
                    now=datetime.now(timezone.utc),
                )
            except Exception as e:
                log.warning("keyword-learn pass failed (non-fatal): %s", e)

        # Retention sweep — drop old clusters + orphan articles.
        # Tuning-driven; retention_days=0 short-circuits. Runs after
        # everything else so a very old learn pass doesn't reanalyze
        # clusters we're about to drop.
        try:
            self.prune_old()
        except Exception as e:
            log.warning("prune_old failed (non-fatal): %s", e)

        log.info(
            "ingest: feeds_ok=%d feeds_fail=%d new_articles=%d",
            ok, fail, len(new_ids),
        )
        return IngestResult(
            fetched_feeds=ok,
            failed_feeds=fail,
            new_articles=len(new_ids),
            new_article_ids=new_ids,
            skipped_due_to_cache=False,
        )

    def prune_old(
        self,
        now: datetime | None = None,
        retention_days: int | None = None,
    ) -> dict:
        """Drop clusters + orphan articles older than ``retention_days``.

        A cluster is "old" when its ``latest_update_at`` falls outside
        the retention window. That preserves long-running threads that
        keep getting fresh articles (the cluster stays young via its
        most recent member) while aging out stories that genuinely
        stopped updating.

        Article dropping is cluster-driven: after we decide which
        clusters to keep, any article_id that belongs to ONLY dropped
        clusters (not referenced by a kept cluster) is removed from
        the article ledger. The learned-keywords ``_analyzed_clusters``
        sidecar is lazily stale-tolerant — references to dropped
        cluster_ids become no-ops in the next learning pass.

        Returns a summary dict for logging + event-log auditing.
        ``retention_days == 0`` disables pruning (returns zeros).
        """
        if retention_days is None:
            retention_days = DEFAULT_RETENTION_DAYS
        if retention_days <= 0:
            return {"clusters_dropped": 0, "articles_dropped": 0, "skipped": True}

        if now is None:
            now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=retention_days)

        clusters = self._get_clusters()
        articles = self._get_articles()
        if not clusters:
            return {"clusters_dropped": 0, "articles_dropped": 0, "skipped": False}

        keep: dict[str, StoryCluster] = {}
        drop_ids: set[str] = set()
        for cid, c in clusters.items():
            if c.latest_update_at < cutoff:
                drop_ids.add(cid)
            else:
                keep[cid] = c
        if not drop_ids:
            return {"clusters_dropped": 0, "articles_dropped": 0, "skipped": False}

        # Any article referenced only by dropped clusters is orphaned.
        kept_aids: set[str] = set()
        for c in keep.values():
            kept_aids.update(c.article_ids)
        dropped_aids = set()
        for cid in drop_ids:
            for aid in clusters[cid].article_ids:
                if aid not in kept_aids:
                    dropped_aids.add(aid)

        pruned_articles = {aid: a for aid, a in articles.items() if aid not in dropped_aids}

        self._ledger.save_clusters(keep)
        self._ledger.save_articles(pruned_articles)
        # Keep the caches hot with the pruned maps so the next read
        # doesn't re-parse the ledger.
        with self._cache_lock:
            self._cached_clusters = keep
            self._cached_articles = pruned_articles
            self._cached_extracted = None   # extracted.json may contain stale entries
            self._cached_graph = None       # retention changed the cluster set
        self._ledger.append_event("prune_old", {
            "retention_days": retention_days,
            "clusters_dropped": len(drop_ids),
            "articles_dropped": len(dropped_aids),
        })
        log.info(
            "prune_old: dropped %d clusters + %d orphan articles "
            "(retention=%d days)",
            len(drop_ids), len(dropped_aids), retention_days,
        )
        return {
            "clusters_dropped": len(drop_ids),
            "articles_dropped": len(dropped_aids),
            "skipped": False,
        }

    def refresh_and_rebuild(self) -> None:
        """Rebuild all clusters from persisted articles. Does NOT refetch.

        Useful after config changes (e.g. a source's ownership_group
        changed) or manual ledger edits. Clears the cluster layer and
        re-runs cluster_articles + rescore on every article.
        """
        articles = self._get_articles()
        now = datetime.now(timezone.utc)
        clusters, _ = cluster_articles(articles, {}, self._sources, now=now)
        for c in clusters.values():
            rescore_cluster(
                c, articles, self._sources, now=now,
                stale_threshold_hours=self._stale_threshold_hours,
            )
        self._ledger.save_clusters(clusters)
        with self._cache_lock:
            self._cached_clusters = clusters   # keep the rebuilt map hot
        self._ledger.append_event("refresh_and_rebuild", {"clusters": len(clusters)})

    # --- read paths ------------------------------------------------------

    def get_news(
        self,
        *,
        topic: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        fetch: bool = True,
    ) -> list[StoryBrief]:
        """Return up to max_results briefs, ranked by state then recency.

        Flow:
          1. ingest_latest(force=False)   # TTL-gated; often skipped
             (skipped entirely when ``fetch=False`` — see below)
          2. rescore every cluster with current time (refreshes is_stale)
          3. filter by topic keyword; fall back to all if nothing matches
          4. rank by (state priority, latest_update_at) descending
          5. build StoryBrief for the top N

        Topic match is a case-insensitive substring check on
        centroid_normalized_title. Multi-word topics are treated as a
        literal phrase; callers wanting "OR" semantics should call
        get_news per term and dedupe client-side.

        ``fetch`` controls whether this call is allowed to hit upstream
        feeds. Set to False from user-facing code paths (chat tools,
        panel APIs) when a background poller is the authoritative
        source of updates — that way a busy user can't push the total
        call volume past the free-tier NewsAPI ceiling. The background
        poller calls ``ingest_latest(force=True)`` directly on its own
        cadence.
        """
        if fetch:
            self.ingest_latest(force=False)

        articles = self._get_articles()
        clusters = self._get_clusters()
        if not clusters:
            return []

        now = datetime.now(timezone.utc)
        # Rescore all with fresh clock to keep is_stale current at read.
        # Track whether anything actually changed — on the common hot
        # path (user refreshes within the same stale-threshold window),
        # nothing mutates and we skip the ~1 MB clusters.json rewrite.
        mutated = False
        for c in clusters.values():
            before_state = c.state
            before_stale = c.is_stale
            rescore_cluster(
                c, articles, self._sources, now=now,
                stale_threshold_hours=self._stale_threshold_hours,
            )
            if c.state != before_state or c.is_stale != before_stale:
                mutated = True
        if mutated:
            self._ledger.save_clusters(clusters)

        candidates = self._filter_by_topic(list(clusters.values()), topic)
        # Sort key: state categorical primary, preference score secondary,
        # recency tertiary. compute_score returns 0 when preferences are
        # disabled, so the secondary key is a no-op and ordering falls
        # through to recency — order-equivalent to Phase 4.
        ranked = sorted(
            candidates,
            key=lambda c: (
                _STATE_PRIORITY[c.state],
                compute_score(c, articles, self._preferences),
                c.latest_update_at,
            ),
            reverse=True,
        )
        # Extracted-body sidecar is cached in memory — it's 3+ MB of
        # JSON so re-parsing per request was the single biggest
        # latency cost pre-cache. Invalidated by ingest when
        # extract_missing actually wrote new records.
        extracted = self._get_extracted()
        return [
            build_brief(
                c, articles, self._sources,
                target_language=self._preferences.target_language,
                translator=self._translator,
                extracted=extracted,
            )
            for c in ranked[:max_results]
        ]

    def get_cluster(self, cluster_id: str) -> StoryBrief | None:
        """Fetch one cluster by id. Returns None if not found.

        Does NOT trigger ingest or rescore — callers who want fresh
        state should call get_news or ingest_latest first. This exists
        for UI drill-downs (given a cluster_id from a prior get_news,
        fetch the detailed brief).
        """
        clusters = self._get_clusters()
        c = clusters.get(cluster_id)
        if c is None:
            return None
        articles = self._get_articles()
        extracted = self._get_extracted()
        return build_brief(
            c, articles, self._sources,
            target_language=self._preferences.target_language,
            translator=self._translator,
            extracted=extracted,
        )

    # --- read accessors for other subsystems (Phase 6) -------------------
    #
    # These expose the ledger + preferences so AlertService can gather
    # cluster-level watchlist signals without poking at NewsService
    # internals. Read-only; non-invasive.

    def load_all_articles(self):
        # Return a fresh copy so downstream mutation can't poison the
        # shared cache. Callers already expected a freshly-loaded dict.
        return dict(self._get_articles())

    def load_all_clusters(self):
        return dict(self._get_clusters())

    def get_preferences(self):
        return self._preferences

    # --- internals -------------------------------------------------------

    def _is_within_ttl(self) -> bool:
        last = self._ledger.last_event("ingest_ok")
        if last is None:
            return False
        try:
            last_ts = datetime.fromisoformat(last["ts"])
        except (KeyError, ValueError):
            return False
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_ts
        return age < timedelta(minutes=self._fetch_ttl_minutes)

    def _cluster_and_rescore(
        self,
        articles: dict,
        now: datetime,
    ) -> None:
        """Run cluster_articles + rescore only for affected clusters.

        After lexical clustering settles, invoke the LLM-assisted
        cross-verification layer (gated on the ``news_cross_verify``
        feature flag) to merge any borderline pairs the lexical
        matcher missed. Merges from that pass get rescored so their
        confidence state reflects the combined membership.
        """
        existing_clusters = self._get_clusters()
        clusters, affected = cluster_articles(
            articles, existing_clusters, self._sources, now=now,
        )

        # --- Phase B: LLM-assisted semantic dedup (supplementary) -----
        # Only fires when the feature flag is on AND we have a verifier
        # (avoid tight coupling to OllamaLLM — tests swap in a fake).
        try:
            from bridge import features
            flag = features.is_enabled("news_cross_verify", default=False)
        except Exception:
            flag = False
        if flag and self._cross_verifier is not None:
            from bridge.news.cross_verify import cross_verify_clusters
            def _sink(ev):
                # Persist one audit event per LLM decision.
                self._ledger.append_event("cross_verify_decision", {
                    "a_id": ev.cluster_a_id, "b_id": ev.cluster_b_id,
                    "title_a": ev.title_a, "title_b": ev.title_b,
                    "jaccard": round(ev.jaccard, 3),
                    "llm_said_same": ev.llm_said_same,
                    "merged": ev.merged,
                })
            merged_clusters = cross_verify_clusters(
                clusters, articles, self._cross_verifier,
                now=now, event_sink=_sink,
            )
            # Add any newly-mutated target clusters to the rescore set.
            # (A merge changed the target in-place in the dict.)
            affected = set(affected) | {
                cid for cid in merged_clusters
                if cid not in clusters
                or merged_clusters[cid] is not clusters.get(cid)
            }
            clusters = merged_clusters

        for cid in affected:
            if cid not in clusters:
                continue  # cluster was absorbed during cross-verify
            rescore_cluster(
                clusters[cid], articles, self._sources, now=now,
                stale_threshold_hours=self._stale_threshold_hours,
            )
        self._ledger.save_clusters(clusters)
        # Ingest just mutated clusters on disk — refresh the cache so
        # the next get_news call reads the new state, not the stale
        # pre-cluster version.
        with self._cache_lock:
            self._cached_clusters = clusters

    # --- Phase E: keyword learning --------------------------------------

    def learned_keywords(self) -> dict:
        """Return the learned-keyword store as a UI-friendly payload.

        Always available, even when the feature flag is off — lets the
        read endpoint render an empty state rather than 500ing.
        """
        from bridge.news.learned_store import to_ui_payload
        return to_ui_payload(self._learned_store.load())

    def build_graph_payload(self) -> dict:
        """Cached news-graph payload for the /api/news/graph endpoint.

        build_news_graph walks all clusters + runs spaCy NER +
        combined-regex scan — ~4-5 s on the Jetson. The UI hits this
        endpoint fresh on every open-graph-panel, which shouldn't pay
        the full cost more than once per ingest window. Cache the
        payload; invalidate alongside the other cluster/article
        caches via _invalidate_cache() whenever the data under it
        shifts.
        """
        if self._cached_graph is not None:
            return self._cached_graph
        from bridge.news.graph import build_news_graph
        payload = build_news_graph(
            self._get_clusters(),
            self._get_articles(),
            self._preferences,
            learned=self.learned_keywords(),
        )
        with self._cache_lock:
            self._cached_graph = payload
        return payload

    def _run_keyword_learning_pass(self, now: datetime) -> int:
        """One learning cycle: gather per-bucket corpora, LLM-extract,
        record, sweep, save.

        Returns the number of LLM calls made. Caller logs / ignores;
        there's no useful failure-mode signal beyond the pass running
        at all.

        Budget enforcement:
          - At most ``max_calls_per_cycle`` LLM calls per cycle.
          - Only buckets with ≥ ``min_clusters_per_bucket`` matching
            clusters are considered (avoid learning from thin data).
          - Corpora are clipped to ``max_fragments_per_call`` headlines
            per bucket so we stay inside Qwen 3.5 4B's context window.
        """
        cfg = self._keyword_learn_cfg
        if cfg is None:
            return 0
        if not self._preferences.enabled:
            # No user watchlists = nothing to learn against. Don't
            # waste a call guessing.
            return 0

        # Build corpora per (bucket, watchlist_label). Use the cached
        # articles + clusters so no extra disk I/O.
        articles = self._get_articles()
        clusters = self._get_clusters()
        if not clusters:
            return 0

        # cluster_entries[bucket_plural][wl_label] ->
        #   list[(cluster_id, fragment_str)]
        # We track cluster_id alongside the fragment so we can
        # diff against the "already analyzed" sidecar and only pay
        # the LLM for clusters new since the last learning pass.
        cluster_entries: dict[str, dict[str, list[tuple[str, str]]]] = {
            b: {} for b in LEARNED_BUCKET_KEYS
        }
        for cluster in clusters.values():
            matches = match_watchlist_items(cluster, articles, self._preferences)
            if not matches:
                continue
            fragment = _build_fragment(cluster, articles)
            if not fragment:
                continue
            for m in matches:
                bucket = _TYPE_SINGULAR_TO_BUCKET.get(m.watchlist_type)
                if bucket is None:
                    continue
                cluster_entries[bucket].setdefault(m.item_label, []).append(
                    (cluster.cluster_id, fragment)
                )

        # Read the sidecar of "already learned from" cluster ids. A
        # bucket whose new-cluster count falls below the threshold
        # is SKIPPED entirely — the expensive work only fires when
        # genuinely new information lands.
        analyzed = self._learned_store.load_analyzed()

        candidates: list[tuple[str, str, list[tuple[str, str]]]] = []
        for bucket, by_label in cluster_entries.items():
            seen_for_bucket = analyzed.get(bucket, {})
            for wl_label, entries in by_label.items():
                already = seen_for_bucket.get(wl_label, set())
                fresh = [(cid, frag) for cid, frag in entries if cid not in already]
                if len(fresh) >= cfg.min_clusters_per_bucket:
                    candidates.append((bucket, wl_label, fresh))
        # Sort by volume of NEW clusters, not total — buckets with the
        # most unlearned signal get the LLM first.
        candidates.sort(key=lambda t: -len(t[2]))
        candidates = candidates[: cfg.max_calls_per_cycle]
        if not candidates:
            log.info(
                "keyword-learn: no bucket has >= %d NEW clusters; skipping cycle",
                cfg.min_clusters_per_bucket,
            )
            return 0

        from bridge.news.keyword_learn import extract_entities

        data = self._learned_store.load()
        calls = 0
        for bucket, wl_label, fresh in candidates:
            frags = [frag for _cid, frag in fresh][: cfg.max_fragments_per_call]
            entities = extract_entities(
                frags, wl_label,
                base_url=cfg.base_url,
                model=cfg.model,
                request_timeout=cfg.request_timeout,
            )
            calls += 1
            # Mark the cluster_ids we fed to the LLM as "analyzed"
            # regardless of whether entities came back — a null
            # result for a fresh cluster IS useful information (means
            # LLM saw nothing name-worthy in it). Re-running on the
            # same cluster wouldn't change that.
            analyzed.setdefault(bucket, {}).setdefault(
                wl_label, set(),
            ).update(cid for cid, _frag in fresh)
            if not entities:
                continue
            learned_record(data, bucket, wl_label, entities, now)
        dropped = learned_sweep_expired(data, now, ttl_days=cfg.ttl_days)
        self._learned_store.save(data, analyzed=analyzed)
        # Learned entities feed into the graph's leaf nodes, so a
        # cycle that added entities invalidates the cached graph too.
        with self._cache_lock:
            self._cached_graph = None
        log.info(
            "keyword-learn: cycle done (%d calls, %d candidates, %d ttl-dropped)",
            calls, len(candidates), dropped,
        )
        self._ledger.append_event(
            "keyword_learn",
            {"calls": calls, "candidates": len(candidates), "ttl_dropped": dropped},
        )
        return calls

    def _filter_by_topic(
        self,
        clusters: list[StoryCluster],
        topic: str | None,
    ) -> list[StoryCluster]:
        """Keyword match on centroid_normalized_title, with recent fallback.

        Spec-delta #7: first try keyword matching; if no cluster matches,
        fall back to the full list (which the caller then ranks and
        slices). Never returns an empty list just because the topic was
        unfamiliar — the voice UX prefers "here's what's new" to silence.
        """
        if not topic:
            return clusters
        needle = topic.lower().strip()
        if not needle:
            return clusters
        matches = [c for c in clusters if needle in c.centroid_normalized_title]
        return matches if matches else clusters


# --- default service singleton --------------------------------------------

# --- module helpers -------------------------------------------------------

def _build_fragment(
    cluster: StoryCluster,
    articles: dict[str, NormalizedArticle],
) -> str:
    """One-line corpus fragment for a cluster: headline — summary.

    Used by the keyword-learning pass to feed the LLM context that
    matches what a human skimming the news panel would see. Falls back
    to the centroid title alone when no member has a display_summary
    (common for legacy ingest rows). Empty string signals "skip this
    cluster" to the caller.
    """
    title = (cluster.centroid_display_title or "").strip()
    if not title:
        return ""
    # Prefer the member whose title matches the centroid (the "canon"
    # pick for this cluster). Fall back to the first member with a
    # non-empty display_summary.
    preferred = None
    for aid in cluster.article_ids:
        art = articles.get(aid)
        if art is None:
            continue
        if art.display_title == cluster.centroid_display_title and art.display_summary:
            preferred = art
            break
    if preferred is None:
        for aid in cluster.article_ids:
            art = articles.get(aid)
            if art is None:
                continue
            if art.display_summary:
                preferred = art
                break
    summary = preferred.display_summary if preferred else ""
    if summary:
        return f"{title} — {summary}"
    return title


_default_service: NewsService | None = None


def get_default_service() -> NewsService:
    """Lazily construct a NewsService from the repo's config files.

    Used by bridge.tools.get_news so the tool dispatcher doesn't pay
    config-load cost at import time. Caches on first call; subsequent
    calls return the same instance.

    Call reset_default_service() in tests if you need to force a rebuild.
    """
    global _default_service
    if _default_service is None:
        _default_service = _build_default_service()
    return _default_service


def reset_default_service() -> None:
    """Clear the singleton. Tests only."""
    global _default_service
    _default_service = None


def _build_default_service() -> NewsService:
    from bridge.utils import REPO_ROOT
    base = REPO_ROOT / "bridge" / "news"
    sources_data = json.loads((base / "config" / "sources.json").read_text(encoding="utf-8"))
    feeds_data = json.loads((base / "config" / "feeds.json").read_text(encoding="utf-8"))

    sources = {
        s["id"]: Source(
            id=s["id"], name=s["name"], domain=s["domain"],
            tier=Tier(s["tier"]), ownership_group=s["ownership_group"],
            is_wire_service=s["is_wire_service"], notes=s.get("notes", ""),
        )
        for s in sources_data
    }
    feeds = [Feed(source_id=f["source_id"], url=f["url"]) for f in feeds_data]
    # preferences.json is user-curated and gitignored. Absent file ->
    # disabled Preferences -> Phase 4 behavior.
    # Phase H: preferences are PER-PROFILE now. Read resolution:
    #   1. active profile's news/preferences.json
    #   2. legacy data/news/preferences.json (pre-H writable mount)
    #   3. bridge/news/config/preferences.json (hand-edited fallback)
    from bridge.profiles import active_profile
    from bridge.utils import REPO_ROOT as _REPO_ROOT
    _profile = active_profile()
    _profile_prefs = _profile.news_dir / "preferences.json"
    _legacy_writable = _REPO_ROOT / "data" / "news" / "preferences.json"
    _legacy_config = base / "config" / "preferences.json"
    if _profile_prefs.exists():
        prefs_path = _profile_prefs
    elif _legacy_writable.exists():
        prefs_path = _legacy_writable
    else:
        prefs_path = _legacy_config
    preferences = load_preferences(prefs_path)
    # Ledger (articles, clusters, events, translations cache) stays
    # GLOBAL — same RSS feeds, same stories, no point duplicating per
    # profile. Only per-user filtering (preferences, learned_keywords)
    # moves.
    ledger = Ledger(base / "data")
    # Phase H: learned keywords live under the active profile because
    # they're learned from THAT profile's watchlist buckets.
    learned_store_path = _profile.news_dir / "learned_keywords.json"

    # Optional LLM verifier for the cross-check layer. We build it only
    # when the feature flag is on, so shipping code is decoupled from
    # shipping ollama access — a config file flip enables / disables
    # the whole Phase B path. Failing to build it is fatal-soft: log
    # and continue with cross_verifier=None.
    cross_verifier = None
    try:
        from bridge import features as _features
        if _features.is_enabled("news_cross_verify", default=False):
            from bridge.llm import OllamaLLM
            from bridge.news.cross_verify import OllamaLLMVerifier
            from bridge.utils import load_config
            cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
            lcfg = cfg["llm"]
            verifier_llm = OllamaLLM(
                base_url=lcfg["base_url"], model=lcfg["model"],
                system_prompt="",   # unused; verifier builds its own
                temperature=0.0,
                num_ctx=512,
                options={},
                request_timeout=float(lcfg.get("request_timeout", 60.0)),
                backend=lcfg.get("backend", "ollama"),
            )
            cross_verifier = OllamaLLMVerifier(verifier_llm)
            log.info("cross-verifier enabled: LLM-assisted cluster merge is on")
    except Exception as e:
        log.warning("cross-verifier disabled (build failed): %s", e)
        cross_verifier = None

    # Optional translator for the Phase C translation path. Same
    # feature-flag pattern as cross_verifier: build lazily, fail-soft
    # so a missing Ollama can't break news ingest. When the flag is
    # off, briefs are built with translator=None and behavior matches
    # Phase 4.5 (original-language headlines, no detail translation).
    translator = None
    try:
        from bridge import features as _features
        if _features.is_enabled("news_translate", default=False):
            from bridge.news.translate import Translator
            from bridge.utils import load_config
            cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
            lcfg = cfg["llm"]
            translator = Translator(
                base_url=lcfg["base_url"],
                model=lcfg["model"],
                cache_path=ledger.data_dir / "translations.json",
                request_timeout=float(lcfg.get("request_timeout", 60.0)),
            )
            log.info(
                "translator enabled: target_language=%s model=%s",
                preferences.target_language, lcfg["model"],
            )
    except Exception as e:
        log.warning("translator disabled (build failed): %s", e)
        translator = None

    # Phase E keyword-learning config. Same flag+fail-soft pattern —
    # on failure the pass is just skipped; the read endpoint still
    # works against whatever's already on disk.
    keyword_learn_cfg: "KeywordLearnConfig | None" = None
    try:
        from bridge import features as _features
        if _features.is_enabled("news_keyword_learn", default=False):
            from bridge.utils import load_config
            cfg = load_config(REPO_ROOT / "config" / "assistant.yaml")
            lcfg = cfg["llm"]
            keyword_learn_cfg = KeywordLearnConfig(
                base_url=lcfg["base_url"],
                model=lcfg["model"],
                request_timeout=float(lcfg.get("request_timeout", 60.0)),
            )
            log.info("keyword-learn enabled: model=%s", lcfg["model"])
    except Exception as e:
        log.warning("keyword-learn disabled (build failed): %s", e)
        keyword_learn_cfg = None

    log.info(
        "built default NewsService: %d sources, %d feeds, "
        "preferences.enabled=%s, cross_verify=%s, translate=%s, learn=%s",
        len(sources), len(feeds), preferences.enabled,
        cross_verifier is not None, translator is not None,
        keyword_learn_cfg is not None,
    )
    return NewsService(
        ledger=ledger, sources=sources, feeds=feeds, preferences=preferences,
        cross_verifier=cross_verifier, translator=translator,
        keyword_learn_cfg=keyword_learn_cfg,
        learned_store_path=learned_store_path,
    )
