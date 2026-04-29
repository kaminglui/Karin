/* Shared JS helpers for all UI panels.
 *
 * No framework, no bundler. Loaded with a plain <script> tag before each
 * panel's own script. Provides:
 *   - escapeHtml(s)      XSS-safe HTML string escape (use for ALL untrusted values)
 *   - relativeTime(iso)  "5 minutes ago" etc.
 *   - formatTimeShort(iso)  "16:30"
 *   - formatDate(iso)    "2026-04-10"
 *   - stateCard(...)     Shared empty/loading/error placeholder card
 *   - safeFetch(url)     Wraps fetch with error normalisation
 *
 * Naming convention: everything is attached to the global `Panels` object
 * so panel-specific scripts can do `Panels.escapeHtml(x)` without polluting
 * the window namespace further than necessary.
 */

(function () {
  "use strict";

  const _ESC_MAP = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s).replace(/[&<>"']/g, (c) => _ESC_MAP[c]);
  }

  /** Parse an ISO-8601 string into a Date. Returns null on failure. */
  function parseIso(iso) {
    if (!iso) return null;
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  /**
   * Humanise an ISO timestamp as "5 minutes ago" / "2 hours ago" /
   * "3 days ago". Falls back to the raw date string past ~2 weeks.
   */
  function relativeTime(iso) {
    const d = parseIso(iso);
    if (!d) return "";
    const delta = (Date.now() - d.getTime()) / 1000;
    if (delta < 60)        return "just now";
    if (delta < 3600)      return `${Math.round(delta / 60)} minutes ago`;
    if (delta < 86400)     return `${Math.round(delta / 3600)} hours ago`;
    if (delta < 86400 * 14) return `${Math.round(delta / 86400)} days ago`;
    return d.toISOString().slice(0, 10);
  }

  function formatTimeShort(iso) {
    const d = parseIso(iso);
    if (!d) return "";
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function formatDate(iso) {
    const d = parseIso(iso);
    if (!d) return "";
    return d.toISOString().slice(0, 10);
  }

  /**
   * Render a shared state card (empty / loading / error).
   *
   *   stateCard({ kind: "loading" })
   *   stateCard({ kind: "empty",   message: "No active alerts." })
   *   stateCard({ kind: "error",   message: "HTTP 500", retry: fn })
   *
   * Returns the HTML string. Callers inject it via container.innerHTML.
   * For `error`, if a retry callback is supplied, we attach it by id.
   */
  function stateCard({ kind, message, label }) {
    const kindClass = kind === "error" ? "error" : "";
    const labelText = label || kind.toUpperCase();
    const msg = message || {
      loading: "Loading…",
      empty:   "Nothing to show.",
      error:   "Something went wrong.",
    }[kind];

    return `
      <div class="state-card ${kindClass}">
        <div class="state-label">${escapeHtml(labelText)}</div>
        <div class="state-message">${escapeHtml(msg)}</div>
      </div>
    `;
  }

  /**
   * Wrap fetch so callers get a thrown Error with a useful message on
   * any non-2xx status. Leaves JSON parsing to the caller so non-JSON
   * endpoints (like the eventual /ui/... HTML ones) can reuse it if
   * they ever need to.
   */
  async function safeFetch(url, options = {}) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (e) {
      throw new Error(`network: ${e.message || e}`);
    }
    if (!res.ok) {
      let detail = "";
      try {
        const body = await res.json();
        detail = body.detail ? `: ${body.detail}` : "";
      } catch (_) { /* not JSON, ignore */ }
      throw new Error(`HTTP ${res.status}${detail}`);
    }
    return res;
  }

  // --- shared display mappings --------------------------------------------
  //
  // Single source of display truth for enum-like backend values. Anything
  // that renders a badge or left-border on a card should look the label up
  // here rather than hardcoding strings in per-panel scripts. This keeps
  // AlertCards, NewsCards, and (later) TrackerCards visually consistent
  // and drift-free as more fields show up.
  //
  // Keys use the exact .value the backend serializes:
  //   - AlertLevel is IntEnum -> integer keys 0..3
  //   - ConfidenceState is str Enum -> string keys "confirmed" etc.
  //
  // Each entry has { label, cls }. cls is the CSS class suffix — panels.css
  // defines e.g. .level-critical, .state-confirmed, .state-stale.

  const DISPLAY = {
    alertLevel: {
      0: { label: "INFO",     cls: "info" },
      1: { label: "WATCH",    cls: "watch" },
      2: { label: "ADVISORY", cls: "advisory" },
      3: { label: "CRITICAL", cls: "critical" },
    },
    newsState: {
      confirmed:               { label: "CONFIRMED",   cls: "confirmed" },
      provisionally_confirmed: { label: "PROVISIONAL", cls: "provisional" },
      developing:              { label: "DEVELOPING",  cls: "developing" },
    },
    // Tracker unit suffix keyed on TrackerConfig.category. Mirrors the
    // server-side _UNIT_BY_CATEGORY in bridge/trackers/formatting.py so
    // the UI renders "4759.5950 USD/oz" without the formatter in the loop.
    trackerUnit: {
      fx:         "",
      metal:      "USD/oz",
      food_index: "",
      energy:     "USD/gal",
    },
    // Shock_label values that warrant a visible pill. "None" is the
    // normal case and produces nothing.
    trackerShock: {
      surging:  { label: "SURGING",  cls: "surging" },
      plunging: { label: "PLUNGING", cls: "plunging" },
    },
    // Movement_label values that warrant a pill. "stable" and "moving"
    // are the boring default; only "volatile" is surfaced.
    trackerMovement: {
      volatile: { label: "VOLATILE", cls: "volatile" },
    },
    // Cross-panel freshness badge. Reused by news stale flag and
    // tracker is_stale flag.
    stale:    { label: "STALE", cls: "stale" },
  };

  /** Safe lookup: returns { label, cls } or a fallback. */
  function displayFor(kindTable, key, fallbackLabel) {
    const entry = kindTable[key];
    if (entry) return entry;
    return { label: String(fallbackLabel ?? key ?? "").toUpperCase(), cls: "unknown" };
  }

  /**
   * Shared widget template. Most panels (weather, wiki, places, ...) share
   * the same skeleton: show a loading card, fetch some data, swap in the
   * rendered content on success, show an error card on failure, and return
   * an { unmount, reload } controller. This helper bakes that skeleton so
   * new panels only have to supply fetch + render.
   *
   * Options:
   *   container     DOM element to render into (required).
   *   fetch()       async fn returning the data payload (required). Can
   *                 throw to trigger the error card.
   *   render(c,d)   fn that paints `data` into `container` (required).
   *   emptyWhen(d)  optional predicate — if true, show an empty card
   *                 instead of calling render.
   *   emptyMessage  message to show in the empty card.
   *   emptyLabel    label badge text for the empty card.
   *   onSubtitle    optional cb(text) — fires with "" while loading,
   *                 "error"/"unavailable" on fail, and with
   *                 subtitleFor(data) on success.
   *   subtitleFor(d) optional fn → subtitle text to report via onSubtitle.
   *   autoRefreshMs optional: if > 0, call reload() on an interval
   *                 (cleared on unmount). Use sparingly — most widgets
   *                 are point-in-time.
   *
   * Returns { unmount, reload } — same shape the existing panels return
   * so chat.js and app.js don't need to special-case template users.
   */
  function mountPanel(options) {
    const opts = {
      container: null,
      fetch: null,
      render: null,
      emptyWhen: null,
      emptyMessage: "Nothing to show.",
      emptyLabel: "empty",
      onSubtitle: null,
      subtitleFor: null,
      autoRefreshMs: 0,
      ...options,
    };
    if (!opts.container || typeof opts.fetch !== "function" || typeof opts.render !== "function") {
      throw new Error("mountPanel: container, fetch, and render are required");
    }

    let aborted = false;
    let refreshTimer = null;

    async function load() {
      opts.container.innerHTML = stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");

      let data;
      try {
        data = await opts.fetch();
      } catch (e) {
        if (aborted) return;
        opts.container.innerHTML = stateCard({
          kind: "error",
          message: (e && e.message) ? e.message : String(e),
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
        return;
      }
      if (aborted) return;

      if (opts.emptyWhen && opts.emptyWhen(data)) {
        opts.container.innerHTML = stateCard({
          kind: "empty",
          label: opts.emptyLabel,
          message: opts.emptyMessage,
        });
        if (opts.onSubtitle) opts.onSubtitle("empty");
        return;
      }

      try {
        opts.render(opts.container, data);
      } catch (e) {
        opts.container.innerHTML = stateCard({
          kind: "error",
          message: `Render failed: ${e.message || e}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
        return;
      }
      if (opts.onSubtitle) {
        const sub = opts.subtitleFor ? (opts.subtitleFor(data) || "") : "";
        opts.onSubtitle(sub);
      }
    }

    load();
    if (opts.autoRefreshMs > 0) {
      refreshTimer = setInterval(() => { if (!aborted) load(); }, opts.autoRefreshMs);
    }

    return {
      unmount() {
        aborted = true;
        if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
        opts.container.innerHTML = "";
      },
      reload: load,
    };
  }

  // ---- Title-action popup (copy / search) ---------------------------
  //
  // Tiny singleton menu shown when a news / alert card is clicked:
  //   📋 Copy title  →  writes the card's headline to clipboard
  //   🔎 Search Google → opens a new-tab Google query with the title
  //
  // Used by news.js and alerts.js so the UX matches across panels.

  async function copyText(text) {
    if (!text) return false;
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_e) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-1000px";
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (_) { /* no-op */ }
      document.body.removeChild(ta);
      return ok;
    }
  }

  const TitleActionPopup = (() => {
    let rootEl = null;
    let currentAnchor = null;
    let currentTitle = "";
    let toastResetT = null;

    function ensure() {
      if (rootEl) return rootEl;
      rootEl = document.createElement("div");
      rootEl.className = "news-action-popup";  // reuse news CSS
      rootEl.setAttribute("role", "menu");
      rootEl.hidden = true;
      rootEl.innerHTML = `
        <button type="button" class="news-action-item" data-act="copy"
                role="menuitem">
          <span aria-hidden="true">📋</span> Copy title
        </button>
        <button type="button" class="news-action-item" data-act="google"
                role="menuitem">
          <span aria-hidden="true">🔎</span> Search Google
        </button>
        <div class="news-action-toast" hidden></div>
      `;
      document.body.appendChild(rootEl);
      rootEl.addEventListener("click", onPick);
      document.addEventListener("click", onOutside, true);
      document.addEventListener("keydown", onKey);
      window.addEventListener("resize", close);
      window.addEventListener("scroll", close, true);
      return rootEl;
    }

    function showToast(msg) {
      const t = rootEl.querySelector(".news-action-toast");
      if (!t) return;
      t.textContent = msg;
      t.hidden = false;
      clearTimeout(toastResetT);
      toastResetT = setTimeout(close, 900);
    }

    async function onPick(e) {
      const btn = e.target.closest(".news-action-item");
      if (!btn || !currentAnchor) return;
      const act = btn.dataset.act;
      if (act === "copy") {
        const ok = await copyText(currentTitle);
        showToast(ok ? "Copied" : "Copy failed");
      } else if (act === "google") {
        const url = "https://www.google.com/search?q=" +
                    encodeURIComponent(currentTitle);
        window.open(url, "_blank", "noopener");
        close();
      }
    }

    function onOutside(e) {
      if (!rootEl || rootEl.hidden) return;
      if (rootEl.contains(e.target)) return;
      if (currentAnchor && currentAnchor.contains(e.target)) return;
      close();
    }

    function onKey(e) {
      if (!rootEl || rootEl.hidden) return;
      if (e.key === "Escape") { e.preventDefault(); close(); }
    }

    function positionNear(card) {
      const r = card.getBoundingClientRect();
      const menuW = 200;
      const gap = 4;
      let left = r.right - menuW;
      if (left < 8) left = 8;
      if (left + menuW > window.innerWidth - 8) {
        left = Math.max(8, window.innerWidth - menuW - 8);
      }
      const top = r.top + gap;
      rootEl.style.left = left + "px";
      rootEl.style.top = top + "px";
    }

    function open(card, title) {
      ensure();
      currentAnchor = card;
      currentTitle = title || "";
      const toast = rootEl.querySelector(".news-action-toast");
      if (toast) toast.hidden = true;
      rootEl.hidden = false;
      positionNear(card);
      const first = rootEl.querySelector(".news-action-item");
      if (first) first.focus();
    }

    function close() {
      if (!rootEl) return;
      rootEl.hidden = true;
      currentAnchor = null;
      currentTitle = "";
      clearTimeout(toastResetT);
    }

    return { open, close };
  })();

  // --- cross-frame panel navigation --------------------------------------
  // Any anchor with data-panel-target inside a panel iframe should swap
  // the parent's popup overlay rather than navigating the iframe alone
  // (which leaves the parent's bottom-nav tab indicator stuck on the
  // original panel). Click delegation here intercepts those clicks and
  // postMessages the parent to call UIState.navigatePopup().
  //
  // The href stays valid as a no-JS fallback (navigates the iframe
  // directly to /ui/<name>, which still works — it just doesn't update
  // the tab indicator).
  document.addEventListener("click", (e) => {
    const a = e.target.closest("[data-panel-target]");
    if (!a) return;
    const target = a.getAttribute("data-panel-target");
    if (!target || !target.startsWith("/ui/")) return;
    if (!window.parent || window.parent === window) return;  // standalone visit
    e.preventDefault();
    window.parent.postMessage({ type: "karin:navigate-panel", target }, "*");
  });

  // --- exports -------------------------------------------------------------

  window.Panels = {
    escapeHtml,
    parseIso,
    relativeTime,
    formatTimeShort,
    formatDate,
    stateCard,
    safeFetch,
    mountPanel,
    DISPLAY,
    displayFor,
    TitleActionPopup,
    copyText,
  };
})();
