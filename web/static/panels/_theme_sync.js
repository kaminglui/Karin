/* _theme_sync.js — shared theme propagation for standalone panel pages.
 *
 * The chat surface (index.html) sets --tint-rgb on :root and toggles
 * the `sun-mode` class on html + body when a character with a
 * procedural-sun face is active. Standalone /ui/<name> pages load in
 * an iframe inside that surface, but iframes are separate documents,
 * so the theme doesn't propagate automatically. This script runs in
 * every panel page, reads the parent's theme state once on load, and
 * applies the same vars + class to the panel's own document.
 *
 * Same-origin iframe (parent + child both served from the bridge) so
 * window.parent.document is reachable. If a panel is opened directly
 * in a top-level tab there is no parent document — the script simply
 * no-ops and the panel uses its default theme.
 */
(function () {
  "use strict";

  function syncTheme() {
    let parentDoc;
    try {
      parentDoc = window.parent && window.parent !== window
        ? window.parent.document : null;
    } catch (_) {
      parentDoc = null;  // cross-origin or not embedded
    }
    if (!parentDoc) return;

    const parentRoot = parentDoc.documentElement;
    if (!parentRoot) return;

    // 1. Copy --tint-rgb (and a few related custom vars if set).
    const computed = window.parent.getComputedStyle(parentRoot);
    const TINT_VARS = [
      "--tint-rgb",
      "--accent-glow",
      "--sun-gradient-from",
      "--sun-gradient-to",
      "--sun-aura-color",
    ];
    const myRoot = document.documentElement;
    TINT_VARS.forEach((name) => {
      const val = computed.getPropertyValue(name).trim();
      if (val) myRoot.style.setProperty(name, val);
    });

    // 2. We do NOT mirror the parent's `sun-mode` class. facts.html
    //    (and any future panel that needs the parent style.css for
    //    widget styles) would otherwise inherit `body.sun-mode {
    //    background: linear-gradient(red, ...) }` — making the panel
    //    page-bg red. The user constraint is "background stays
    //    neutral; only buttons + accent text follow theme." Copying
    //    --tint-rgb above is sufficient for that. If a panel ever
    //    actually needs `.sun-mode` (none does today), promote the
    //    branch to a custom data-attribute we control here.

    // 3. Mirror prefers-color-scheme override if the parent set one
    //    explicitly (some apps do `data-theme="dark"`). Cheap forward
    //    compat — current chrome doesn't use it, but worth honoring
    //    if it appears later.
    const parentDark = parentRoot.getAttribute("data-theme");
    if (parentDark) myRoot.setAttribute("data-theme", parentDark);
  }

  // Run as early as possible so the panel paints with the right colors
  // from the first frame. DOMContentLoaded is good enough — the parent
  // document was already fully loaded before the iframe started.
  if (document.readyState !== "loading") {
    syncTheme();
  } else {
    document.addEventListener("DOMContentLoaded", syncTheme);
  }

  // Re-sync when the parent broadcasts a theme change. Parent should
  // do `window.dispatchEvent(new CustomEvent('karin:theme-change'))`
  // after toggling the class. Iframes can't listen on the parent's
  // window directly without messaging, so we also accept postMessage.
  window.addEventListener("message", (e) => {
    if (e.data && e.data.type === "karin:theme-change") syncTheme();
  });
})();
