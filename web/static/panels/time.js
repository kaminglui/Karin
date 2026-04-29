/* time.js — inline clock widget for the get_time tool.
 *
 * Pure client-side: uses Intl.DateTimeFormat so the browser's timezone
 * database does all the work. No backend route needed — which is why
 * the panels.css-bg already looks like what we want (big numbers,
 * muted secondary line).
 *
 * Refreshes once per second so seconds tick live. autoRefreshMs in
 * Panels.mountPanel re-runs the fetch, but our "fetch" here is just
 * `new Date()` so it's free.
 */
(function () {
  "use strict";

  function formatParts(date, tz) {
    const safeTz = tz || undefined;  // undefined → browser local
    const dateFmt = new Intl.DateTimeFormat(undefined, {
      timeZone: safeTz,
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    const timeFmt = new Intl.DateTimeFormat(undefined, {
      timeZone: safeTz,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    // Intl doesn't expose the resolved tz label in a friendly way; use the
    // requested tz when given, otherwise the browser's default.
    const resolvedTz = safeTz || Intl.DateTimeFormat().resolvedOptions().timeZone;
    return {
      date: dateFmt.format(date),
      time: timeFmt.format(date),
      tz: resolvedTz,
    };
  }

  function render(container, data) {
    const p = formatParts(data.now, data.tz);
    container.innerHTML = `
      <div class="time-card">
        <div class="time-clock">${Panels.escapeHtml(p.time)}</div>
        <div class="time-date">${Panels.escapeHtml(p.date)}</div>
        <div class="time-tz">${Panels.escapeHtml(p.tz)}</div>
      </div>
    `;
  }

  function mountTimePanel(container, options = {}) {
    const opts = { tz: null, onSubtitle: null, ...options };
    return Panels.mountPanel({
      container,
      fetch: async () => ({ now: new Date(), tz: opts.tz }),
      render,
      subtitleFor: (d) => {
        const p = formatParts(d.now, d.tz);
        return `${p.time} · ${p.tz}`;
      },
      onSubtitle: opts.onSubtitle,
      // Re-run once per second so the clock ticks. The fetch is a Date
      // construction — effectively free.
      autoRefreshMs: 1000,
    });
  }

  if (typeof window !== "undefined") {
    window.TimePanel = { mountTimePanel };
  }
})();
