/* wiki.js — Wikipedia summary widget.
 *
 * Used for both wiki_search (looks up a specific topic) and
 * wiki_random (random article). Renders title, optional thumbnail,
 * the extract, and a "read more" link to the canonical Wikipedia
 * page. Fetches Wikipedia directly — no backend route needed,
 * Wikipedia's REST API supports CORS.
 */
(function () {
  "use strict";

  const WIKI_BASE = "https://en.wikipedia.org/api/rest_v1";
  const SEARCH_BASE = "https://en.wikipedia.org/w/api.php";
  const UA_HEADERS = { "Api-User-Agent": "karin/0.1 web" };

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  async function resolveTitle(query) {
    // opensearch -> top match -> canonical title.
    const url = `${SEARCH_BASE}?action=opensearch&search=${encodeURIComponent(query)}&limit=1&format=json&origin=*`;
    const resp = await Panels.safeFetch(url);
    const data = await resp.json();
    const titles = data[1] || [];
    return titles[0] || null;
  }

  async function fetchSummary(title) {
    const url = `${WIKI_BASE}/page/summary/${encodeURIComponent(title.replace(/ /g, "_"))}`;
    const resp = await Panels.safeFetch(url, { headers: UA_HEADERS });
    return resp.json();
  }

  async function fetchRandom() {
    const url = `${WIKI_BASE}/page/random/summary`;
    const resp = await Panels.safeFetch(url, { headers: UA_HEADERS });
    return resp.json();
  }

  function render(container, summary, opts) {
    const title = summary.title || "Untitled";
    const extract = (summary.extract || "").trim();
    const thumb = summary.thumbnail && summary.thumbnail.source;
    const pageUrl =
      (summary.content_urls && summary.content_urls.desktop && summary.content_urls.desktop.page)
      || `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, "_"))}`;
    const desc = (summary.description || "").trim();

    const thumbHtml = thumb
      ? `<img class="wiki-thumb" src="${escapeHtml(thumb)}" alt="" loading="lazy">`
      : "";

    container.innerHTML = `
      <div class="wiki-card">
        ${thumbHtml}
        <div class="wiki-body">
          <div class="wiki-title">${escapeHtml(title)}</div>
          ${desc ? `<div class="wiki-desc">${escapeHtml(desc)}</div>` : ""}
          <div class="wiki-extract">${escapeHtml(extract)}</div>
          <a class="wiki-link" href="${escapeHtml(pageUrl)}" target="_blank" rel="noopener">
            Read more on Wikipedia →
          </a>
        </div>
      </div>
    `;

    if (opts.onSubtitle) opts.onSubtitle(title);
  }

  /** Extract the article title from the backend tool result.
   *  Shape: "<Title>: <extract sentence 1>. <extract sentence 2>. ...".
   *  Returns null if the result doesn't match the expected shape (e.g.
   *  an error string starting with "Error:"). */
  function titleFromResult(resultText) {
    if (!resultText || typeof resultText !== "string") return null;
    const s = resultText.trim();
    if (s.startsWith("Error:") || s.startsWith("[tool call suppressed")) return null;
    // Split on the FIRST ": " so titles with colons don't get truncated wrong.
    const idx = s.indexOf(": ");
    if (idx <= 0) return null;
    return s.slice(0, idx);
  }

  function mountWikiPanel(container, options = {}) {
    const opts = {
      query: null,       // when set: search mode; null = random mode
      resultText: null,  // backend tool result — locks widget to the same article the LLM saw
      onSubtitle: null,
      ...options,
    };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      try {
        let summary;
        // Prefer the title embedded in the backend's tool result. That
        // guarantees the widget shows the article the assistant actually
        // summarized — without this, wiki_random would re-roll and the
        // widget would display a different random page than the reply.
        const lockedTitle = titleFromResult(opts.resultText);
        if (lockedTitle) {
          summary = await fetchSummary(lockedTitle);
        } else if (opts.query) {
          const title = await resolveTitle(opts.query);
          if (!title) {
            if (aborted) return;
            container.innerHTML = Panels.stateCard({
              kind: "empty",
              label: "no match",
              message: `No Wikipedia article for '${opts.query}'.`,
            });
            if (opts.onSubtitle) opts.onSubtitle("not found");
            return;
          }
          summary = await fetchSummary(title);
        } else {
          summary = await fetchRandom();
        }
        if (aborted) return;
        render(container, summary, opts);
      } catch (e) {
        if (aborted) return;
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Failed to load Wikipedia: ${e.message}`,
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
    window.Wiki = { mountWikiPanel };
  }
})();
