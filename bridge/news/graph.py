"""Phase F.a: relation-graph data model for the news subsystem.

Turns the current cluster set into a small graph where watchlist items
are nodes and shared clusters are weighted edges. Pure computation —
no I/O, no LLM, no network. Fed to `GET /api/news/graph` for the
future viz layer (F.b).

Node model:

* Each node is one of the user's enabled watchlist items (region /
  topic / event). Node id = ``"{kind}:{item_id}"`` where kind is
  singular ("region" / "topic" / "event"), matching the shape
  WatchlistMatch already emits.
* Node weight = count of distinct clusters matching this item.
* Node carries ``cluster_ids`` so a future panel can draw a convex-hull
  bubble around the stories in each bucket.

Edge model:

* An edge runs between any two items that BOTH matched the same
  cluster. Edge weight = count of clusters they share. An item's
  self-pair is not emitted.
* Same-kind edges (region↔region, topic↔topic) and cross-kind edges
  (region↔topic) are both emitted — the viz can filter if needed.

Everything is returned as plain dicts / lists so the endpoint can
``JSONResponse(payload)`` without a serializer layer.

Design calls:

* Static view. The graph is built from whatever clusters the ledger
  holds at call time. No sliding time window for now — the cluster
  layer itself ages stories out via staleness (see confidence.py),
  so trimming happens one layer up.
* Preferences-gated. ``preferences.enabled=False`` returns an empty
  graph because match_watchlist_items short-circuits there — the UI
  should render a "set up watchlists first" empty state.
* Deterministic output order. Nodes + edges are sorted by (kind,
  label) / (source, target) so snapshot tests don't flap.
"""
from __future__ import annotations

import re
import unicodedata
from itertools import combinations

from bridge.news.models import NormalizedArticle, StoryCluster
from bridge.news.preferences import (
    Preferences,
    _build_haystack,
    match_watchlist_items,
)


def _fold_key(s: str) -> str:
    """Lowercase + fold diacritics so 'González' and 'Gonzalez' dedup
    to the same entity. NFKD decomposes "á" into "a" + combining-acute;
    filtering combining marks (Unicode category ``Mn``) drops the
    accent while keeping the base letter. Also strips whitespace and
    folds spaces to a single space for robustness."""
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip().lower()

# Plural-bucket -> singular-kind used by the graph's node id format.
# Mirrors the convention in learned_store (plural) vs WatchlistMatch
# (singular). Kept here so the graph doesn't reach across modules.
_BUCKET_TO_SINGULAR = {"regions": "region", "topics": "topic", "events": "event"}


def _node_id(watchlist_type: str, item_id: str) -> str:
    return f"{watchlist_type}:{item_id}"


def _node_kind_rank(kind: str) -> int:
    # Deterministic tie-break for sorting. Watchlist kinds first
    # (region/topic/event), learned entities last so any list view
    # shows hubs before their orbiting leaf nodes.
    return {"region": 0, "topic": 1, "event": 2, "entity": 3}.get(kind, 4)


def build_news_graph(
    clusters: dict[str, StoryCluster],
    articles: dict[str, NormalizedArticle],
    preferences: Preferences,
    learned: "dict | None" = None,
    max_entities_per_bucket: int = 8,
) -> dict:
    """Compute the graph payload.

    Returns a dict with three keys:

    * ``nodes``   — list of ``{id, kind, label, weight, cluster_ids}``
    * ``edges``   — list of ``{source, target, weight, cluster_ids}``
    * ``counts``  — quick summary ``{clusters, nodes, edges}``

    Cluster ids on nodes + edges are SORTED so the payload is
    byte-reproducible for a given input (snapshot test friendly).
    """
    # Aggregators.
    node_weight: dict[str, int] = {}
    node_meta: dict[str, dict] = {}
    node_cluster_ids: dict[str, list[str]] = {}
    edge_weight: dict[tuple[str, str], int] = {}
    edge_cluster_ids: dict[tuple[str, str], list[str]] = {}

    # Compute (matches, haystack) per cluster ONCE. Both passes below
    # need this; computing it twice doubled the graph-API latency past
    # 30 seconds on an 800-cluster ledger.
    cluster_info: list[tuple[StoryCluster, list, str]] = []
    for cluster in clusters.values():
        matches = match_watchlist_items(cluster, articles, preferences)
        if not matches:
            continue
        cluster_info.append((cluster, matches, _build_haystack(cluster, articles)))

    # Walk once. For each cluster, find matching watchlist items and
    # record (1) per-item weight + cluster membership, (2) edge weight
    # for every item pair.
    cluster_count = 0
    for cluster, matches, _hay in cluster_info:
        cluster_count += 1

        # Dedup at the cluster level: if a cluster somehow produced two
        # matches with the same item_id (shouldn't happen today, but
        # cheap to guard), count it once.
        node_ids: list[tuple[str, str, str, str]] = []
        seen_ids: set[str] = set()
        for m in matches:
            nid = _node_id(m.watchlist_type, m.item_id)
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            node_ids.append((nid, m.watchlist_type, m.item_id, m.item_label))

        # Per-node bookkeeping.
        for nid, kind, iid, label in node_ids:
            node_weight[nid] = node_weight.get(nid, 0) + 1
            node_meta.setdefault(nid, {"kind": kind, "id": iid, "label": label})
            node_cluster_ids.setdefault(nid, []).append(cluster.cluster_id)

        # Per-edge bookkeeping: every unordered pair of matched items.
        # combinations handles order + dedup; sort the pair so the key
        # is canonical regardless of match iteration order.
        for a, b in combinations(node_ids, 2):
            key = tuple(sorted([a[0], b[0]]))
            edge_weight[key] = edge_weight.get(key, 0) + 1
            edge_cluster_ids.setdefault(key, []).append(cluster.cluster_id)

    # Build deterministic output lists.
    nodes = []
    for nid in node_weight:
        meta = node_meta[nid]
        nodes.append({
            "id": nid,
            "kind": meta["kind"],
            "label": meta["label"],
            "weight": node_weight[nid],
            "cluster_ids": sorted(node_cluster_ids[nid]),
        })

    # Merge entity leaf nodes from two label sources:
    #
    #   * Qwen (semantic) — LLM picks named entities from news text.
    #     Good at canonical names ("Xi Jinping", "ASML"), personas,
    #     organisation acronyms.
    #   * Lexical capitalised-phrase extractor — pure regex over all
    #     cluster headlines with a stopword filter. Catches what
    #     Qwen's budget-limited window missed ("Strait of Hormuz",
    #     "Bank of England") and is fully deterministic.
    #
    # Separation of concerns:
    #   * Both sources only contribute LABELS.
    #   * Counts come from a word-boundary scan of cluster haystacks
    #     per hub — one source of truth grounded in the ledger text.
    #
    # One node per unique label (lowercased). Edges emitted to every
    # hub whose clusters mention it, turning entities like "Iran"
    # into bridges across the hubs that actually discuss them.
    from bridge.news.noun_extract import rank_corpus_phrases

    labels: dict[str, str] = {}  # key -> first-seen display form

    # Only filter entities whose label EXACTLY matches a hub's own
    # label. Broader keyword filtering (dropping "Trump" because
    # "trump" is a keyword of the US hub) would collapse meaningful
    # bridge entities back into their parent hub — but the user's
    # mental model is that Trump is a person who BRIDGES buckets, not
    # a synonym for the US.
    # Diacritic-fold hub labels too so an entity like "Türkiye"
    # (Turkey hub alias) doesn't slip through as a leaf. Matches how
    # we fold entity labels below.
    hub_label_keys: set[str] = {_fold_key(m["label"]) for m in node_meta.values()}

    # Source A: Qwen-learned entity names (semantic filter).
    if isinstance(learned, dict):
        for bucket, groups in learned.items():
            if not isinstance(groups, list):
                continue
            if _BUCKET_TO_SINGULAR.get(bucket) is None:
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                entities = group.get("entities") or []
                if not isinstance(entities, list):
                    continue
                for ent in entities:
                    label = str(ent.get("label") or "").strip()
                    if not label:
                        continue
                    key = _fold_key(label)
                    if not key or key in hub_label_keys:
                        continue
                    labels.setdefault(key, label)

    # Source B: lexical capitalised-phrase extractor (deterministic).
    # Feeds on cluster display_titles — short, editorially consistent,
    # proper nouns already in title case.
    corpus_titles = [c.centroid_display_title for c in clusters.values()]
    for phrase, _count in rank_corpus_phrases(corpus_titles, min_count=2, top_n=60):
        key = _fold_key(phrase)
        if not key or key in hub_label_keys:
            continue
        labels.setdefault(key, phrase)

    # Precompute per-hub (cluster_id, haystack) pairs from cluster_info.
    # Keeping the cluster_id alongside the haystack lets the label-scan
    # below record WHICH cluster produced each match — the node-click
    # drill-down needs those references to show the underlying news.
    hub_cluster_hays: dict[str, list[tuple[str, str]]] = {
        nid: [] for nid in node_meta
    }
    for cluster, matches, hay in cluster_info:
        for m in matches:
            nid = _node_id(m.watchlist_type, m.item_id)
            if nid in hub_cluster_hays:
                hub_cluster_hays[nid].append((cluster.cluster_id, hay))

    # Pass 2: single combined regex over all labels, scanned once per
    # haystack. Records (label, hub, cluster_id) triples so we can
    # emit both counts (edge weight) AND the cluster memberships the
    # UI needs for drill-down.
    #
    # Haystacks are folded the same way as labels so "González" in a
    # headline matches the "Gonzalez" entity key. Without folding,
    # the accented form never hit the regex and the graph showed
    # two separate nodes for the same person.
    #
    # Sort labels by length descending so the regex engine prefers
    # longer alternatives ("Strait of Hormuz" beats "Hormuz" when
    # both match the same position).
    # label_to_hub_to_cids[key][hub_id] -> set of cluster_ids
    label_to_hub_to_cids: dict[str, dict[str, set[str]]] = {}
    if labels:
        sorted_keys = sorted(labels.keys(), key=lambda k: -len(k))
        combined_pat = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
            re.IGNORECASE,
        )
        for hub_id, cluster_hays in hub_cluster_hays.items():
            for cid, hay in cluster_hays:
                folded_hay = _fold_key(hay)
                for m in combined_pat.finditer(folded_hay):
                    matched_key = m.group(0)  # already folded
                    bucket = label_to_hub_to_cids.setdefault(matched_key, {})
                    bucket.setdefault(hub_id, set()).add(cid)

    # For the click-drill-down payload: sort entity cluster_ids by
    # recency (latest_update_at desc) and cap so a dense entity like
    # "Trump" — which can hit 200+ clusters — doesn't bloat the JSON.
    # The user sees the N freshest stories on click, which is what
    # they'd want anyway.
    _MAX_CIDS_PER_EDGE = 10

    def _recent_cids(cids: "set[str] | list[str]") -> list[str]:
        """Top _MAX_CIDS_PER_EDGE cluster_ids by latest_update_at desc."""
        with_time = []
        for cid in cids:
            c = clusters.get(cid)
            if c is None:
                continue
            with_time.append((c.latest_update_at, cid))
        with_time.sort(reverse=True)
        return [cid for _, cid in with_time[:_MAX_CIDS_PER_EDGE]]

    # Emit one node + one edge per hub the entity actually appeared in.
    # Labels with zero matches are silently dropped — keeps ghost
    # entities off the graph.
    for key, label in labels.items():
        hub_cids = label_to_hub_to_cids.get(key)
        if not hub_cids:
            continue
        total_weight = sum(len(s) for s in hub_cids.values())
        ent_id = "entity:" + key.replace(" ", "_")

        # Aggregate top-N recent cluster_ids across ALL hubs for the
        # node's own cluster_ids. Gives a single "show me the stories
        # where this entity came up" list when the user clicks the
        # entity node itself.
        all_cids: set[str] = set()
        for cids in hub_cids.values():
            all_cids.update(cids)
        nodes.append({
            "id": ent_id,
            "kind": "entity",
            "label": label,
            "weight": total_weight,
            "cluster_ids": _recent_cids(all_cids),
        })
        for hub_id, cids in hub_cids.items():
            edge_key = tuple(sorted([hub_id, ent_id]))
            w = len(cids)
            edge_weight[edge_key] = edge_weight.get(edge_key, 0) + w
            # Edge cluster_ids = top-N recent clusters that matched
            # THIS entity in THIS hub. Lets the edge-click UI show
            # "here's what Trump + Defense & Security share".
            edge_cluster_ids.setdefault(edge_key, []).extend(
                _recent_cids(cids)
            )

    nodes.sort(key=lambda n: (_node_kind_rank(n["kind"]), n["label"].lower(), n["id"]))

    edges = []
    for (a, b), w in edge_weight.items():
        edges.append({
            "source": a,
            "target": b,
            "weight": w,
            "cluster_ids": sorted(edge_cluster_ids[(a, b)]),
        })
    edges.sort(key=lambda e: (-e["weight"], e["source"], e["target"]))

    return {
        "nodes": nodes,
        "edges": edges,
        "counts": {
            "clusters": cluster_count,
            "nodes": len(nodes),
            "edges": len(edges),
        },
    }
