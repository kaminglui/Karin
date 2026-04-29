/* NewsPanel — renders /api/news/briefs into a container.
 *
 * Same pattern as AlertsPanel: exported `News.mountNewsPanel(container, options)`
 * takes over the container, returns a controller { refresh, unmount }.
 *
 * Interactive features added (Phase:interactive-widgets):
 *   - Click a news card → opens the news-detail modal with TOC + full
 *     article text (via /api/news/cluster/{cluster_id}).
 *   - "More" button at the bottom grows the grid by N more stories
 *     without re-fetching the ones already shown.
 *   - Footer link to the day's digest (/ui/digest) so users can jump
 *     to the combined passive-class view.
 *
 * Card hierarchy is scan-friendly:
 *   1. headline (h3, primary)
 *   2. state badge + stale badge + watchlist chips
 *   3. source chips
 *   4. "N independent, M syndicated"
 *   5. Reasoning (collapsed in <details>)
 */

(function () {
  "use strict";

  const DEFAULT_MAX_RESULTS = 5;
  const MORE_INCREMENT = 10;
  // Mirror of the server-side cap on /api/news/briefs?max_results.
  // Kept here so the "More" button disables itself once the ceiling
  // is reached instead of letting the user hit the cap mid-click and
  // getting an error card back. Bump both sides together.
  const SERVER_MAX_RESULTS = 100;

  function mountNewsPanel(container, options = {}) {
    const opts = {
      max_results: DEFAULT_MAX_RESULTS,
      topic: null,
      apiBase: "",
      onSubtitle: null,
      ...options,
    };

    // Two separate caps:
    //   `currentCap`    — how many briefs we ASK THE SERVER for.
    //                      Grows when "More" needs more raw data to
    //                      fill the active tab's display cap.
    //   `capByFilter[t]` — how many items we SHOW in tab t. Each tab
    //                      gets its own cap so clicking More on
    //                      Regions only grows Regions, not the other
    //                      tabs. Previously a single cap was shared,
    //                      which made "More" on a filtered tab feel
    //                      disconnected from the count it changed.
    let currentCap = opts.max_results;
    const capByFilter = {
      all:     opts.max_results,
      regions: opts.max_results,
      topics:  opts.max_results,
      events:  opts.max_results,
    };
    // Active filter tab. Values:
    //   "all"     — no filter (default)
    //   "regions" — only briefs that match a regions watchlist item
    //   "topics"  — only briefs that match a topics watchlist item
    //   "events"  — only briefs that match an events watchlist item
    // Legacy ?filter=watched URLs redirect to "regions" so bookmarks
    // that predate the split don't land on an empty grid.
    const VALID_FILTERS = new Set(["all", "regions", "topics", "events"]);
    let filter = "all";
    try {
      const urlFilter = new URLSearchParams(window.location.search).get("filter");
      if (urlFilter === "watched") filter = "regions";
      else if (VALID_FILTERS.has(urlFilter)) filter = urlFilter;
    } catch (_) { /* non-browser environment */ }

    async function load({ preserveOnError = false } = {}) {
      // Clamp the request to the server's limit — asking for more
      // gets us a 422 and wipes the page. Bump both sides together
      // when raising the cap.
      if (currentCap > SERVER_MAX_RESULTS) currentCap = SERVER_MAX_RESULTS;

      // On initial / full reloads we show a spinner. On "More" clicks
      // we pass preserveOnError=true so a network hiccup just flips
      // the button back to "More" without blowing away the grid the
      // user is already reading.
      if (!preserveOnError) {
        container.innerHTML = Panels.stateCard({ kind: "loading" });
        if (opts.onSubtitle) opts.onSubtitle("");
      }

      const params = new URLSearchParams();
      params.set("max_results", String(currentCap));
      if (opts.topic) params.set("topic", opts.topic);

      let data;
      try {
        const res = await Panels.safeFetch(`${opts.apiBase}/api/news/briefs?${params}`);
        data = await res.json();
      } catch (e) {
        if (preserveOnError) {
          // Rethrow so wireMoreButton can restore the button state
          // without wiping the grid.
          throw e;
        }
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Failed to load news: ${e.message}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
        return;
      }

      const briefs = Array.isArray(data.briefs) ? data.briefs : [];
      if (briefs.length === 0) {
        const msg = opts.topic
          ? `No stories matched "${opts.topic}". Try a different keyword or clear the filter.`
          : "No stories yet. News populates after the first get_news call.";
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "no news",
          message: msg,
        });
        if (opts.onSubtitle) opts.onSubtitle("0 stories");
        return;
      }

      // Client-side filter by watchlist type. The backend emits the
      // watchlist_type in singular form ("region" / "topic" / "event")
      // but our tab keys are plural to match the Settings panel. Map
      // plural -> singular once here so filter logic + counts both
      // use the same mapping.
      const TYPE_SINGULAR = { regions: "region", topics: "topic", events: "event" };
      function briefMatches(brief, tabKey) {
        const ms = brief.watchlist_matches;
        if (!Array.isArray(ms) || ms.length === 0) return false;
        const singular = TYPE_SINGULAR[tabKey];
        return ms.some(
          (m) => String(m.watchlist_type || "").toLowerCase() === singular,
        );
      }
      const countByType = { regions: 0, topics: 0, events: 0 };
      for (const b of briefs) {
        for (const t of Object.keys(countByType)) {
          if (briefMatches(b, t)) countByType[t]++;
        }
      }
      const filtered = filter === "all"
        ? briefs
        : briefs.filter((b) => briefMatches(b, filter));
      // Clamp the active tab's display cap to what's actually
      // available — otherwise a tab-switch to a small-match filter
      // would show a blank "More" when there's nothing to grow toward.
      const shown = filtered.slice(0, capByFilter[filter]);

      // "More" disables when:
      //   * server won't give us more briefs (exhausted),
      //   * we've already reached SERVER_MAX_RESULTS (atCeiling), AND
      //   * the active tab has no more matches to reveal.
      // On a filtered tab where all existing matches are already
      // visible, we still offer More IF the server might have
      // additional briefs that could contain new matches.
      const atCeiling = currentCap >= SERVER_MAX_RESULTS;
      const serverExhausted = briefs.length < currentCap;
      const tabShowsAll = shown.length >= filtered.length;
      const disable = atCeiling || (serverExhausted && tabShowsAll);
      const label = (serverExhausted && tabShowsAll)
        ? `No more ${filter === "all" ? "stories" : filter}`
        : atCeiling
          ? "Reached news limit"
          : "More";

      function tab(key, label, count) {
        return `
          <button type="button" class="news-filter-tab"
                  role="tab" data-filter="${key}"
                  aria-selected="${filter === key}">
            ${label} (${count})
          </button>`;
      }
      const filterBar = `
        <div class="news-filter" role="tablist" aria-label="Filter news">
          ${tab("all",     "All",     briefs.length)}
          ${tab("regions", "Regions", countByType.regions)}
          ${tab("topics",  "Topics",  countByType.topics)}
          ${tab("events",  "Events",  countByType.events)}
        </div>
      `;

      const emptyState = shown.length === 0 && filter !== "all"
        ? Panels.stateCard({
            kind: "empty",
            label: `no ${filter} matches`,
            message: `No current stories match your ${filter} watchlist. Add an entry in Settings, or switch back to All.`,
          })
        : "";

      container.innerHTML = `
        ${filterBar}
        ${emptyState}
        <div class="news-grid">
          ${shown.map(renderNewsCard).join("")}
        </div>
        <div class="news-footer">
          <button type="button" class="news-more-btn"
                  ${disable ? "disabled" : ""}>
            ${label}
          </button>
        </div>
      `;
      wireCardHandlers(container, opts);
      wireFilterTabs(container);
      wireMoreButton(container, async () => {
        // Grow the active tab's display cap first. If the tab has
        // already revealed every match it has locally, and the server
        // might have more, also bump the fetch cap. This keeps More
        // tab-isolated: clicking on Regions only adds Regions items
        // (up to MORE_INCREMENT), not a mystery handful that might or
        // might not match.
        const prevTabCap = capByFilter[filter];
        const prevServerCap = currentCap;
        capByFilter[filter] = Math.min(
          capByFilter[filter] + MORE_INCREMENT,
          SERVER_MAX_RESULTS,
        );
        const needsServerFetch =
          shown.length >= filtered.length && !serverExhausted && !atCeiling;
        if (needsServerFetch) {
          currentCap = Math.min(currentCap + MORE_INCREMENT, SERVER_MAX_RESULTS);
        }
        try {
          await load({ preserveOnError: true });
        } catch (e) {
          capByFilter[filter] = prevTabCap;
          currentCap = prevServerCap;
          throw e;
        }
      });

      const n = shown.length;
      const filterSuffix = filter === "all" ? "" : ` · ${filter}`;
      const topicSuffix = opts.topic ? ` · topic="${opts.topic}"` : "";
      if (opts.onSubtitle) {
        opts.onSubtitle(`${n} ${n === 1 ? "story" : "stories"}${filterSuffix}${topicSuffix}`);
      }
    }

    /** Wire the All / Watched filter tabs. Clicking a tab flips the
     *  `filter` closure var and re-renders from the same briefs data
     *  already on screen — we don't re-fetch. Also writes the
     *  selection to the URL so reloads stick. */
    function wireFilterTabs(root) {
      root.querySelectorAll(".news-filter-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
          const next = btn.getAttribute("data-filter");
          if (next === filter) return;
          filter = next;
          try {
            const u = new URL(window.location.href);
            if (filter === "all") u.searchParams.delete("filter");
            else u.searchParams.set("filter", filter);
            window.history.replaceState(null, "", u);
          } catch (_) { /* best-effort */ }
          load();
        });
      });
    }

    load();

    return {
      refresh: () => load(),
      unmount: () => {
        // Also tear down any open modal — prevents orphaned overlays
        // when the caller swaps conversations mid-read.
        NewsModal.close();
        container.innerHTML = "";
      },
    };
  }

  function renderNewsCard(brief) {
    const esc = Panels.escapeHtml;
    const state = Panels.displayFor(
      Panels.DISPLAY.newsState, brief.state, brief.state,
    );

    const staleBadge = brief.is_stale
      ? `<span class="badge state-stale">STALE</span>`
      : "";

    const watchlistChips = (brief.watchlist_matches || [])
      .map((m) => {
        const type = String(m.watchlist_type || "").toLowerCase();
        return `<span class="chip watchlist ${esc(type)}">${esc(m.item_label || m.item_id || "")}</span>`;
      })
      .join("");

    const sourceChips = (brief.top_sources || [])
      .map((s) => `<span class="chip">${esc(s)}</span>`)
      .join("");

    const indep = Number(brief.independent_confirmation_count || 0);
    const synd = Number(brief.syndicated_article_count || 0);
    const confirmationLine = `${indep} independent, ${synd} syndicated`;

    // Timestamps: show when the story first appeared + last update.
    let timeLine = "";
    if (brief.first_seen_at) {
      const first = new Date(brief.first_seen_at);
      const parts = [];
      if (!isNaN(first)) {
        parts.push("first reported " + first.toLocaleString(undefined, {
          month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
        }));
      }
      if (brief.latest_update_at) {
        const latest = new Date(brief.latest_update_at);
        if (!isNaN(latest) && latest - first > 60000) {
          parts.push("updated " + latest.toLocaleString(undefined, {
            month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
          }));
        }
      }
      if (parts.length) timeLine = `<div class="time-line">${esc(parts.join(" · "))}</div>`;
    }

    const reasoning = brief.reasoning || [];
    const reasoningBlock = reasoning.length
      ? `
        <details class="reasoning-block">
          <summary>Details (${reasoning.length})</summary>
          <ul class="reasoning">
            ${reasoning.map((r) => `<li>${esc(r)}</li>`).join("")}
          </ul>
        </details>`
      : "";

    // data-cluster-id is the primary key for opening the detail
    // modal; tabindex=0 + role=button make the card
    // keyboard-navigable for accessibility. Click handling lives in
    // wireCardHandlers() — clicking opens a small action popup
    // (view / copy / google) rather than jumping straight to the
    // modal. That's one more tap, but it lets the user copy or
    // web-search the headline without a separate UI element.
    return `
      <article class="card news-card state-${state.cls}"
               data-cluster-id="${esc(brief.cluster_id)}"
               role="button" tabindex="0"
               aria-label="Actions for: ${esc(brief.headline)}">
        <h3>${esc(brief.headline)}</h3>
        <div class="card-meta">
          <span class="badge state-${state.cls}">${state.label}</span>
          ${staleBadge}
          ${watchlistChips}
        </div>
        ${sourceChips ? `<div class="chips sources">${sourceChips}</div>` : ""}
        <div class="confirmation-line">${esc(confirmationLine)}</div>
        ${timeLine}
        ${reasoningBlock}
      </article>
    `;
  }

  function wireCardHandlers(container, opts) {
    container.querySelectorAll(".news-card").forEach((card) => {
      const titleOf = (c) => (c.querySelector("h3") || {}).textContent?.trim() || "";
      card.addEventListener("click", (e) => {
        // Let the reasoning <details> toggle normally without
        // triggering the action popup.
        if (e.target.closest("details")) return;
        Panels.TitleActionPopup.open(card, titleOf(card));
      });
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          Panels.TitleActionPopup.open(card, titleOf(card));
        }
      });
    });
  }

  function wireMoreButton(container, onMore) {
    const btn = container.querySelector(".news-more-btn");
    if (!btn || btn.disabled) return;
    btn.addEventListener("click", async () => {
      const originalLabel = btn.textContent;
      btn.disabled = true;
      btn.textContent = "Loading…";
      try {
        await onMore();
      } catch (e) {
        // The load call was invoked with preserveOnError=true, so the
        // grid is still visible. Flip the button to a retry state
        // with the actual error message as a tooltip.
        btn.disabled = false;
        btn.textContent = "Retry — load failed";
        btn.title = e && e.message ? String(e.message) : "Load failed";
      }
    });
  }

  // ---- news detail modal -----------------------------------------------
  //
  // Lives as a singleton on `window.NewsModal`. Opens on click of a
  // news card; fetches /api/news/cluster/<id>; renders a sidebar TOC
  // (one anchor per cluster member) + a main panel listing each
  // article with title, byline, date, extracted text (or fallback
  // link). Escape or backdrop click closes.

  const NewsModal = (function () {
    let root = null;   // overlay element; null when closed

    function ensureRoot() {
      if (root) return root;
      root = document.createElement("div");
      root.className = "news-modal-backdrop";
      root.innerHTML = `
        <div class="news-modal" role="dialog" aria-modal="true"
             aria-label="Full article">
          <header class="news-modal-header">
            <h2 class="news-modal-title"></h2>
            <button type="button" class="news-modal-close"
                    aria-label="Close">×</button>
          </header>
          <div class="news-modal-body">
            <aside class="news-modal-toc" aria-label="Article navigation"></aside>
            <main class="news-modal-content"></main>
          </div>
        </div>
      `;
      document.body.appendChild(root);
      root.addEventListener("click", (e) => {
        // Click backdrop (outside the .news-modal panel) → close.
        if (e.target === root) close();
      });
      root.querySelector(".news-modal-close").addEventListener("click", close);
      document.addEventListener("keydown", onKeydown);
      return root;
    }

    function onKeydown(e) {
      if (e.key === "Escape" && root) close();
    }

    async function open(clusterId) {
      ensureRoot();
      const titleEl = root.querySelector(".news-modal-title");
      const tocEl = root.querySelector(".news-modal-toc");
      const contentEl = root.querySelector(".news-modal-content");
      titleEl.textContent = "Loading…";
      tocEl.innerHTML = "";
      contentEl.innerHTML = "";
      root.classList.add("is-open");
      document.body.classList.add("modal-open");
      try {
        const res = await Panels.safeFetch(`/api/news/cluster/${encodeURIComponent(clusterId)}`);
        const data = await res.json();
        render(titleEl, tocEl, contentEl, data);
      } catch (e) {
        contentEl.innerHTML = `
          <div class="news-modal-error">
            Failed to load article: ${Panels.escapeHtml(e.message || String(e))}
          </div>
        `;
      }
    }

    function close() {
      if (!root) return;
      root.classList.remove("is-open");
      document.body.classList.remove("modal-open");
      // Keep root in DOM — reuse on next open. Cheaper than rebuilding.
    }

    function render(titleEl, tocEl, contentEl, data) {
      const esc = Panels.escapeHtml;
      titleEl.textContent = data.headline || "Story";
      const members = Array.isArray(data.members) ? data.members : [];
      if (members.length === 0) {
        contentEl.innerHTML = `<div class="news-modal-empty">No articles in this cluster.</div>`;
        return;
      }
      // TOC: one entry per member, anchored to its section below.
      tocEl.innerHTML = `
        <div class="news-modal-toc-header">Sources (${members.length})</div>
        <ol class="news-modal-toc-list">
          ${members.map((m, i) => `
            <li>
              <a href="#article-${i}">
                <span class="toc-source">${esc(m.source_name)}</span>
                <span class="toc-title">${esc(m.title)}</span>
              </a>
            </li>
          `).join("")}
        </ol>
      `;
      // Main: one section per article. Prefer extracted text; when the
      // extractor failed / wasn't run yet, show a link-out with a
      // reason so the user isn't confused why text is missing.
      contentEl.innerHTML = members.map((m, i) => {
        const date = m.published_at
          ? new Date(m.published_at).toLocaleString()
          : "";
        const byline = m.extracted_author ? ` · ${esc(m.extracted_author)}` : "";
        let body;
        if (m.extracted_text) {
          // Paragraph-split on blank-line or single-line-breaks, with
          // escapeHtml on each paragraph.
          const paras = m.extracted_text
            .split(/\n\n+/)
            .map((p) => p.trim())
            .filter(Boolean)
            .map((p) => `<p>${esc(p)}</p>`)
            .join("");
          body = `<div class="article-body">${paras}</div>`;
        } else {
          const reason = m.extraction_error || "not yet extracted";
          body = `
            <div class="article-body article-body-fallback">
              <p class="muted">Full text not available (${esc(reason)}).</p>
              <p><a href="${esc(m.url)}" target="_blank" rel="noopener noreferrer">
                Read on ${esc(m.source_name)} &rarr;
              </a></p>
            </div>
          `;
        }
        return `
          <section class="article-section" id="article-${i}">
            <h3>${esc(m.title)}</h3>
            <div class="article-meta">
              <span class="article-source">${esc(m.source_name)}</span>
              ${date ? `<span class="muted"> · ${esc(date)}</span>` : ""}
              ${byline}
              <a class="article-outlink" href="${esc(m.url)}"
                 target="_blank" rel="noopener noreferrer">open &#x2197;</a>
            </div>
            ${body}
          </section>
        `;
      }).join("");
    }

    return { open, close };
  })();

  // --- exports -----------------------------------------------------------

  window.News = {
    mountNewsPanel,
    renderNewsCard,
  };
  window.NewsModal = NewsModal;
})();
