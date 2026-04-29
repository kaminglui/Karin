/* places.js — location-aware "find places" widget.
 *
 * Shows a header with the resolved location + the search query, then
 * a list of DDG results. Location defaults to IP-detected city when
 * the LLM doesn't pass an explicit one.
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

  function renderResults(container, data, opts) {
    const loc = data.location || "";
    const query = data.query || "";
    const results = data.results || [];

    const header = `
      <div class="places-header">
        <span class="places-pin" aria-hidden="true">\u{1F4CD}</span>
        <div class="places-header-text">
          <div class="places-query">${escapeHtml(query)}</div>
          <div class="places-location">${escapeHtml(loc)}</div>
        </div>
      </div>
    `;

    if (!results.length) {
      container.innerHTML = header + Panels.stateCard({
        kind: "empty",
        label: "no results",
        message: `No places found.`,
      });
      if (opts.onSubtitle) opts.onSubtitle(loc ? `${loc} · 0` : "0");
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

    container.innerHTML = header + `<div class="search-list">${items}</div>`;
    if (opts.onSubtitle) {
      opts.onSubtitle(`${loc} · ${results.length} result${results.length === 1 ? "" : "s"}`);
    }
  }

  function mountPlacesPanel(container, options = {}) {
    const opts = { query: null, location: null, onSubtitle: null, ...options };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      const q = (opts.query || "").trim();
      if (!q) {
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "no query",
          message: "No query given for place search.",
        });
        return;
      }
      try {
        const params = new URLSearchParams({ q });
        if (opts.location) params.set("location", opts.location);
        const r = await Panels.safeFetch(`/api/find-places?${params.toString()}`);
        const data = await r.json();
        if (aborted) return;
        if (data.error && !data.results?.length) {
          container.innerHTML = Panels.stateCard({
            kind: "error",
            message: data.error,
          });
          if (opts.onSubtitle) opts.onSubtitle("error");
          return;
        }
        renderResults(container, data, opts);
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
    window.Places = { mountPlacesPanel };
  }
})();
