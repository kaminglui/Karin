/* AlertsPanel — renders the /api/alerts/active endpoint into a container.
 *
 * Exports a single `Alerts.mountAlertsPanel(container, options)` function
 * that takes over the given container element. The same function is used
 * by the standalone alerts.html page AND (future UI-5) by Karin's chat
 * page when Karin decides an AlertsPanel should surface.
 *
 * Phase G.c — threat visualization:
 *   - each card gets a 0-4 "threat" chip (max score over its signals)
 *   - a threshold control at the top dims cards scoring below the pick
 *   - if the active profile has no user_location, a one-line hint strip
 *     appears explaining why threat scores may all be 0
 *
 * Design choice (per UI-5 pre-planning): in-place render, not iframes.
 * mountAlertsPanel writes into container.innerHTML directly so the caller
 * keeps full control of sizing, layout, and lifecycle.
 */

(function () {
  "use strict";

  // AlertLevel -> display mapping lives in Panels.DISPLAY.alertLevel
  // (see panels.js). renderAlertCard() reads it via Panels.displayFor().

  const DEFAULT_MAX_RESULTS = 20;

  // Threshold slot in localStorage. Per-browser, not per-profile — a
  // visual preference that doesn't need to round-trip the backend.
  // Stored as the numeric score; 0 means "show everything, don't dim."
  const DIM_THRESHOLD_KEY = "alerts.dim_below_threshold";

  // 0..4 threat tier labels + CSS class suffixes. Mirrors the rule-
  // based score rubric in bridge/alerts/threat_llm.py so the UI and
  // LLM stay synchronized: 0=none, 1=awareness, 2=watch,
  // 3=advisory, 4=critical.
  const THREAT_TIERS = [
    { label: "no threat",  cls: "t0", short: "0" },
    { label: "awareness",  cls: "t1", short: "1" },
    { label: "watch",      cls: "t2", short: "2" },
    { label: "advisory",   cls: "t3", short: "3" },
    { label: "critical",   cls: "t4", short: "4" },
  ];

  function getDimThreshold() {
    try {
      const raw = localStorage.getItem(DIM_THRESHOLD_KEY);
      if (raw === null) return 0;
      const n = parseInt(raw, 10);
      if (!Number.isFinite(n) || n < 0 || n > 4) return 0;
      return n;
    } catch {
      return 0;
    }
  }

  function setDimThreshold(n) {
    try {
      localStorage.setItem(DIM_THRESHOLD_KEY, String(n));
    } catch {
      // Private mode / disabled storage -> silently lose the pref.
      // Not worth surfacing to the user.
    }
  }

  /**
   * Mount an AlertsPanel into `container`. Returns a tiny controller
   * object: { refresh(), unmount() }.
   *
   * options:
   *   max_results      default 20
   *   apiBase          default "" (same origin)
   *   onSubtitle       optional callback(text) -- lets the caller reflect
   *                    "3 alerts" / "no active alerts" in their own UI
   *                    without parsing the rendered HTML.
   */
  function mountAlertsPanel(container, options = {}) {
    const opts = {
      max_results: DEFAULT_MAX_RESULTS,
      apiBase: "",
      onSubtitle: null,
      ...options,
    };

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");

      let data;
      try {
        const url = `${opts.apiBase}/api/alerts/active?max_results=${opts.max_results}`;
        const res = await Panels.safeFetch(url);
        data = await res.json();
      } catch (e) {
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Failed to load alerts: ${e.message}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
        return;
      }

      const alerts = Array.isArray(data.alerts) ? data.alerts : [];
      const locationConfigured = data.location_configured !== false;

      // Always render the top strip (threshold + optional hint) even
      // when alerts is empty — the user can set threshold for the next
      // refresh, and the location hint is still relevant.
      const header = renderPanelControls(locationConfigured);

      if (alerts.length === 0) {
        container.innerHTML = header + Panels.stateCard({
          kind: "empty",
          label: "no active alerts",
          message: "Nothing triggered right now. Alert history is in the event log.",
        });
        wireThresholdControl(container);
        if (opts.onSubtitle) opts.onSubtitle("0 alerts");
        return;
      }

      const threshold = getDimThreshold();
      container.innerHTML = header + alerts.map((a) => renderAlertCard(a, threshold)).join("");
      wireCardHandlers(container);
      wireThresholdControl(container);
      if (opts.onSubtitle) {
        opts.onSubtitle(alerts.length === 1 ? "1 alert" : `${alerts.length} alerts`);
      }
    }

    /* Alert-card click → shared title-action popup (copy / google),
     * matching the news panel's interaction. Lets the user exfiltrate
     * an alert title without mousing over to another tool. The <details>
     * block for triggering signals is left alone — clicks inside it
     * should NOT open the popup. Dimmed cards are still clickable —
     * "dim" is a visual hint, not a disable. */
    function wireCardHandlers(container) {
      container.querySelectorAll(".alert-card").forEach((card) => {
        const titleOf = () =>
          (card.querySelector("h3") || {}).textContent?.trim() || "";
        // Mark the card as clickable for keyboard users + screen readers.
        card.setAttribute("role", "button");
        card.setAttribute("tabindex", "0");
        card.addEventListener("click", (e) => {
          if (e.target.closest("details")) return;
          if (!window.Panels || !Panels.TitleActionPopup) return;
          Panels.TitleActionPopup.open(card, titleOf());
        });
        card.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            if (!window.Panels || !Panels.TitleActionPopup) return;
            Panels.TitleActionPopup.open(card, titleOf());
          }
        });
      });
    }

    /* Threshold <select> is re-rendered on every load, so re-wire
     * it each time. Changing the value flips the `.dimmed` class on
     * existing cards without a full re-fetch — cheap DOM pass, no
     * network. */
    function wireThresholdControl(container) {
      const select = container.querySelector("#alerts-dim-threshold");
      if (!select) return;
      select.addEventListener("change", () => {
        const n = parseInt(select.value, 10) || 0;
        setDimThreshold(n);
        applyDimming(container, n);
      });
    }

    // Kick off initial load asynchronously so the caller can chain
    // controller.refresh() / unmount() without awaiting.
    load();

    return {
      refresh: () => load(),
      unmount: () => { container.innerHTML = ""; },
    };
  }

  function applyDimming(container, threshold) {
    container.querySelectorAll(".alert-card").forEach((card) => {
      const score = parseInt(card.getAttribute("data-threat") || "-1", 10);
      // Score of -1 (unknown / no signal score) is always shown —
      // dimming requires a concrete score to compare against.
      if (score >= 0 && score < threshold) {
        card.classList.add("dimmed");
      } else {
        card.classList.remove("dimmed");
      }
    });
  }

  function renderPanelControls(locationConfigured) {
    const current = getDimThreshold();
    const optionMarkup = [0, 1, 2, 3, 4].map((n) => {
      const sel = n === current ? " selected" : "";
      const label = n === 0 ? "show all" : `dim < ${n} (${THREAT_TIERS[n].label})`;
      return `<option value="${n}"${sel}>${label}</option>`;
    }).join("");

    const hint = locationConfigured ? "" : `
      <div class="alerts-hint">
        No <strong>user_location</strong> set — threat scores will all be 0.
        Set one on the <a href="/ui/settings" data-panel-target="/ui/settings">Settings</a> page
        (or in config/assistant.yaml) so news signals can be ranked by
        proximity.
      </div>`;

    return `
      <div class="alerts-controls">
        <label for="alerts-dim-threshold">Threshold:</label>
        <select id="alerts-dim-threshold" aria-label="Dim alerts below threshold">
          ${optionMarkup}
        </select>
        <span class="alerts-controls-hint">below-threshold cards are dimmed, not hidden</span>
      </div>
      ${hint}
    `;
  }

  function renderAlertCard(alert, dimThreshold) {
    const esc = Panels.escapeHtml;
    const levelNum = Number(alert.level);
    const display = Panels.displayFor(Panels.DISPLAY.alertLevel, levelNum, "INFO");
    const levelName = display.label;
    const levelCls = display.cls;
    const inactiveCls = alert.is_active === false ? " inactive" : "";

    // Threat tier badge. `threat_score` may be null for old alerts
    // persisted before Phase G.a — treat that as "unknown" and skip
    // the chip entirely rather than showing a misleading 0.
    const rawScore = alert.threat_score;
    const hasScore = Number.isFinite(rawScore);
    const threatAttr = hasScore ? rawScore : -1;
    const threatBadge = hasScore
      ? (() => {
          const tier = THREAT_TIERS[Math.max(0, Math.min(4, rawScore))];
          return `<span class="badge threat-${tier.cls}" title="${tier.label}">threat ${tier.short}</span>`;
        })()
      : "";
    const dimmedCls = (hasScore && (dimThreshold || 0) > 0 && rawScore < dimThreshold)
      ? " dimmed" : "";

    const domains = (alert.affected_domains || [])
      .map(d => `<span class="chip">${esc(d)}</span>`)
      .join("");

    const bullets = (alert.reasoning_bullets || [])
      .map(b => `<li>${esc(b)}</li>`)
      .join("");

    const sources = (alert.source_attribution || [])
      .map(s => `<span class="chip muted">${esc(s)}</span>`)
      .join("");

    const cooldown = alert.cooldown_until
      ? `cooldown until ${esc(Panels.formatTimeShort(alert.cooldown_until))}`
      : "";

    // Signals detail block — expandable, shows the raw Signal payloads so
    // the "why" is always inspectable from the UI without hitting the log.
    const signals = alert.triggered_by_signals || [];
    const signalsBlock = signals.length
      ? `
        <details>
          <summary>${signals.length} triggering signal${signals.length === 1 ? "" : "s"}</summary>
          <pre>${esc(JSON.stringify(signals, null, 2))}</pre>
        </details>`
      : "";

    return `
      <article class="card alert-card level-${levelCls}${inactiveCls}${dimmedCls}"
               data-alert-id="${esc(alert.alert_id)}"
               data-threat="${threatAttr}">
        <div class="card-meta">
          <span class="badge level-${levelCls}">${levelName}</span>
          ${threatBadge}
          <span class="chip">${esc(alert.category)}</span>
          <time datetime="${esc(alert.created_at)}">
            ${esc(Panels.relativeTime(alert.created_at))}
          </time>
        </div>
        <h3>${esc(alert.title)}</h3>
        ${bullets ? `<ul class="reasoning">${bullets}</ul>` : ""}
        ${domains ? `<div class="chips domains">${domains}</div>` : ""}
        ${signalsBlock}
        <div class="card-footer">
          <div class="attribution">${sources}</div>
          <div class="timing">${esc(cooldown)}</div>
        </div>
      </article>
    `;
  }

  // --- exports -------------------------------------------------------------

  window.Alerts = {
    mountAlertsPanel,
    renderAlertCard,
  };
})();
