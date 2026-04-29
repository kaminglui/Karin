/* history.js — On-This-Day historical event widget.
 *
 * Calls Wikipedia's onthisday/events feed for today's MM/DD, picks
 * a random event, and renders a card with the year as a big badge,
 * the event description, and (when available) a thumbnail + link to
 * the most relevant linked Wikipedia article.
 */
(function () {
  "use strict";

  const FEED_BASE = "https://en.wikipedia.org/api/rest_v1/feed/onthisday";

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function pickRandom(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  async function fetchTodayEvents() {
    const now = new Date();
    const mm = String(now.getMonth() + 1).padStart(2, "0");
    const dd = String(now.getDate()).padStart(2, "0");
    const url = `${FEED_BASE}/events/${mm}/${dd}`;
    const resp = await Panels.safeFetch(url);
    const data = await resp.json();
    return data.events || [];
  }

  function render(container, ev, opts) {
    const year = ev.year != null ? String(ev.year) : "";
    const text = (ev.text || "").trim();

    // The most relevant linked page is usually the first entry in
    // ev.pages — Wikipedia orders these by the event's primary subject.
    const page = (ev.pages && ev.pages[0]) || null;
    const thumb = page && page.thumbnail && page.thumbnail.source;
    const pageTitle = page && (page.normalizedtitle || page.title);
    const pageUrl =
      page && page.content_urls && page.content_urls.desktop && page.content_urls.desktop.page;
    const pageDesc = page && (page.description || "");

    const thumbHtml = thumb
      ? `<img class="hist-thumb" src="${escapeHtml(thumb)}" alt="" loading="lazy">`
      : "";

    const linkHtml = pageUrl
      ? `<a class="hist-link" href="${escapeHtml(pageUrl)}" target="_blank" rel="noopener">
           ${escapeHtml(pageTitle || "Read more")} →
         </a>`
      : "";

    const dateLabel = new Date().toLocaleDateString(undefined, {
      month: "long", day: "numeric",
    });

    container.innerHTML = `
      <div class="hist-card">
        <div class="hist-header">
          <span class="hist-year">${escapeHtml(year)}</span>
          <span class="hist-date">${escapeHtml(dateLabel)}</span>
        </div>
        <div class="hist-body">
          ${thumbHtml}
          <div class="hist-text-wrap">
            <div class="hist-text">${escapeHtml(text)}</div>
            ${pageDesc ? `<div class="hist-pagedesc">${escapeHtml(pageDesc)}</div>` : ""}
            ${linkHtml}
          </div>
        </div>
      </div>
    `;

    if (opts.onSubtitle) opts.onSubtitle(year ? `${dateLabel} · ${year}` : dateLabel);
  }

  function mountHistoryPanel(container, options = {}) {
    const opts = { onSubtitle: null, ...options };
    let aborted = false;
    let allEvents = [];

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      try {
        allEvents = await fetchTodayEvents();
        if (aborted) return;
        if (!allEvents.length) {
          container.innerHTML = Panels.stateCard({
            kind: "empty",
            label: "no events",
            message: "No on-this-day events for today.",
          });
          if (opts.onSubtitle) opts.onSubtitle("0 events");
          return;
        }
        render(container, pickRandom(allEvents), opts);
        // Add a "Show another" button so the user can re-roll.
        const reroll = document.createElement("button");
        reroll.className = "hist-reroll";
        reroll.type = "button";
        reroll.textContent = "Another event";
        reroll.addEventListener("click", () => {
          if (aborted || !allEvents.length) return;
          render(container, pickRandom(allEvents), opts);
          container.appendChild(reroll);  // re-attach after re-render
        });
        container.appendChild(reroll);
      } catch (e) {
        if (aborted) return;
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Failed to load on-this-day: ${e.message}`,
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
    window.History = { mountHistoryPanel };
  }
})();
