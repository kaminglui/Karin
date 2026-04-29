/* websearch.js — DuckDuckGo web-search results widget.
 *
 * Calls a small backend endpoint (/api/web-search?q=...) which wraps
 * the same ddgs library the LLM tool uses, so the widget and the tool
 * see the same results. Renders a list of cards: title, snippet, url.
 */
(function () {
  "use strict";

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function hostFromUrl(u) {
    try { return new URL(u).host.replace(/^www\./, ""); }
    catch { return ""; }
  }

  function render(container, query, results, opts) {
    if (!results.length) {
      container.innerHTML = Panels.stateCard({
        kind: "empty",
        label: "no results",
        message: `No web results for '${query}'.`,
      });
      if (opts.onSubtitle) opts.onSubtitle("0 results");
      return;
    }

    const items = results.map((r, i) => {
      const title = (r.title || "").trim() || "(untitled)";
      const body = (r.body || "").trim();
      const url = (r.href || "").trim();
      const host = hostFromUrl(url);
      return `
        <a class="search-item" href="${escapeHtml(url)}" target="_blank" rel="noopener">
          <div class="search-rank">${i + 1}</div>
          <div class="search-text">
            <div class="search-title">${escapeHtml(title)}</div>
            <div class="search-host">${escapeHtml(host)}</div>
            <div class="search-body">${escapeHtml(body)}</div>
          </div>
        </a>
      `;
    }).join("");

    container.innerHTML = `<div class="search-list">${items}</div>`;
    if (opts.onSubtitle) opts.onSubtitle(`${results.length} result${results.length === 1 ? "" : "s"}`);
  }

  function mountWebSearchPanel(container, options = {}) {
    const opts = { query: null, onSubtitle: null, ...options };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      const q = (opts.query || "").trim();
      if (!q) {
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "no query",
          message: "Search ran without a query.",
        });
        if (opts.onSubtitle) opts.onSubtitle("");
        return;
      }
      try {
        const r = await Panels.safeFetch(`/api/web-search?q=${encodeURIComponent(q)}`);
        const data = await r.json();
        if (aborted) return;
        if (data.error) {
          container.innerHTML = Panels.stateCard({
            kind: "error",
            message: data.error,
          });
          if (opts.onSubtitle) opts.onSubtitle("error");
          return;
        }
        render(container, q, data.results || [], opts);
      } catch (e) {
        if (aborted) return;
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Search failed: ${e.message}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
      }
    }

    load();
    return {
      unmount: () => { aborted = true; container.innerHTML = ""; },
      reload: load,
    };
  }

  if (typeof window !== "undefined") {
    window.WebSearch = { mountWebSearchPanel };
  }
})();
