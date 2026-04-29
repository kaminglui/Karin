/* TrackersPanel — renders /api/trackers/snapshots (or single snapshot)
 * into a container.
 *
 * Same contract as AlertsPanel / NewsPanel:
 *   Trackers.mountTrackersPanel(container, options)
 *     -> { refresh, unmount }
 *
 * Card priorities (scan-friendly):
 *   1. label (muted header)
 *   2. big value + unit
 *   3. freshness line (as-of date + optional STALE badge)
 *   4. delta row (1d / 1w / 1m) with arrow glyphs
 *   5. shock / volatile pills
 *
 * Direction labels (direction_1d / direction_1w) are intentionally NOT
 * rendered as separate chips — the delta arrows already encode that
 * signal and surfacing them twice would be redundant.
 *
 * Single-tracker view (?id=X) reuses the same renderTrackerCard; the
 * only difference is grid vs. single layout. Empty / error states use
 * the shared Panels.stateCard().
 */

(function () {
  "use strict";

  const DEFAULT_API_BASE = "";

  /**
   * Mount the TrackersPanel into `container`. Returns { refresh, unmount }.
   *
   * options:
   *   id           optional tracker id/alias -> single-card view
   *   apiBase      default "" (same origin)
   *   onSubtitle   callback(text) for chrome subtitle
   */
  function mountTrackersPanel(container, options = {}) {
    const opts = {
      id: null,
      apiBase: DEFAULT_API_BASE,
      onSubtitle: null,
      ...options,
    };

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");

      if (opts.id) {
        await loadSingle(opts.id);
      } else {
        await loadAll();
      }
    }

    async function loadAll() {
      let data;
      try {
        const res = await Panels.safeFetch(`${opts.apiBase}/api/trackers/snapshots`);
        data = await res.json();
      } catch (e) {
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Failed to load trackers: ${e.message}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
        return;
      }

      const snapshots = Array.isArray(data.snapshots) ? data.snapshots : [];
      if (snapshots.length === 0) {
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "no trackers",
          message: "No trackers configured yet. Copy trackers.example.json to trackers.json and enable the ones you want.",
        });
        if (opts.onSubtitle) opts.onSubtitle("0 trackers");
        return;
      }

      container.innerHTML = renderGroupedSections(snapshots);
      wireFxCardHandlers(container);
      wireInfoButtons(container);
      const n = snapshots.length;
      if (opts.onSubtitle) {
        opts.onSubtitle(`${n} ${n === 1 ? "tracker" : "trackers"}`);
      }
    }

    async function loadSingle(id) {
      let data;
      try {
        const url = `${opts.apiBase}/api/trackers/snapshot?id=${encodeURIComponent(id)}`;
        const res = await Panels.safeFetch(url);
        data = await res.json();
      } catch (e) {
        // 404 and other HTTP errors get the same error state card.
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Tracker "${id}" not available: ${e.message}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
        return;
      }

      const snap = data.snapshot;
      if (!snap) {
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "no tracker",
          message: `Tracker "${id}" has no snapshot yet.`,
        });
        if (opts.onSubtitle) opts.onSubtitle("empty");
        return;
      }

      container.innerHTML = `
        <div class="trackers-grid">
          ${renderTrackerCard(snap)}
        </div>
      `;
      wireFxCardHandlers(container);
      wireInfoButtons(container);
      if (opts.onSubtitle) opts.onSubtitle(snap.label || snap.id);
    }

    load();

    return {
      refresh: () => load(),
      unmount: () => { container.innerHTML = ""; },
    };
  }

  /** Parse an FX tracker label like "USD/CNY" into ["USD", "CNY"].
   *  Returns null when the shape doesn't match so callers can fall
   *  back to the generic card layout. */
  function parseFxPair(label) {
    if (typeof label !== "string") return null;
    const parts = label.split("/").map((s) => s.trim().toUpperCase());
    if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
    return parts;
  }

  /** Round a JS number to a readable string with at most 4 decimals
   *  and no trailing zeros. Used by FX conversions — avoids the
   *  "1.23000000000002" floats from JS multiplication/division. */
  function fmtAmount(n) {
    if (typeof n !== "number" || !isFinite(n)) return "—";
    // Scale precision to magnitude: for sub-1 numbers show more
    // decimals so tiny-rate pairs (e.g. JPY→USD) still read cleanly.
    const abs = Math.abs(n);
    const digits = abs >= 100 ? 2 : abs >= 1 ? 4 : 6;
    const s = n.toFixed(digits);
    return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
  }

  /* Category display metadata. Maps the raw `category` strings from
   * the tracker config to a human label + the order we want the
   * grid's sections to appear in. Unknown categories get grouped
   * under "Other" at the end. */
  const CATEGORY_META = {
    fx:         { order: 1, label: "Currencies" },
    metal:      { order: 2, label: "Metals" },
    energy:     { order: 3, label: "Energy" },
    food_index: { order: 4, label: "Food & CPI" },
    crypto:     { order: 5, label: "Crypto" },
  };

  /* PADD (Petroleum Administration for Defense District) regions used
   * by the EIA retail-gas series. Keyed by the short PADD code found
   * in the tracker's label ("PADD 1B" → "1B"). `name` is the
   * official region name; `states` is a reader-friendly state list
   * or `"national"` for the whole-US case. When gas_retail's config
   * `duoarea` changes (and the label updates alongside it), the guide
   * popover adapts automatically — nothing is hard-coded to 1B or to
   * a specific user location. */
  const PADD_REGIONS = {
    "1A": { name: "New England",      states: "CT, ME, MA, NH, RI, VT" },
    "1B": { name: "Central Atlantic", states: "DE, DC, MD, NJ, NY, PA" },
    "1C": { name: "Lower Atlantic",   states: "FL, GA, NC, SC, VA, WV" },
    "2":  { name: "Midwest",          states: "IL, IN, IA, KS, KY, MI, MN, MO, ND, NE, OH, OK, SD, TN, WI" },
    "3":  { name: "Gulf Coast",       states: "AL, AR, LA, MS, NM, TX" },
    "4":  { name: "Rocky Mountain",   states: "CO, ID, MT, UT, WY" },
    "5":  { name: "West Coast",       states: "AK, AZ, CA, HI, NV, OR, WA" },
    "US": { name: "United States",    states: "national" },
  };

  function paddFromLabel(label) {
    // "Retail Gasoline (PADD 1B)" → "1B". Case-insensitive. Returns
    // null when the label doesn't mention a PADD so callers can
    // fall back to a generic description.
    const m = String(label || "").match(/PADD\s+([A-Z0-9]+)/i);
    return m ? m[1].toUpperCase() : null;
  }

  /* Scope / location line derivations. A *function* can be used in
   * place of a string — it receives the TrackerSnapshot and returns
   * the final text. This keeps the data-driven trackers (like gas_retail
   * which can be re-pointed at any PADD via config) from drifting out
   * of sync with the hard-coded descriptions. */
  const TRACKER_SCOPE = {
    gas_retail: (snap) => {
      const padd = paddFromLabel(snap.label);
      const region = PADD_REGIONS[padd];
      if (region && region.states === "national") return "United States (EIA weekly)";
      if (region) return `${region.name} (PADD ${padd}), US`;
      return "US regional (EIA weekly)";
    },
    us_cpi_food:         "United States (CPI, SA)",
    us_cpi_food_at_home: "United States (CPI, SA — groceries only)",
  };

  /* Guide text for indicators that aren't self-explanatory. Shown in
   * a popover when the card's ⓘ button is clicked. Each value is
   * either `{title, body}` or `(snap) => {title, body}` for entries
   * that need to derive their copy from live tracker state. */
  const TRACKER_GUIDE = {
    us_cpi_food: {
      title: "Interpreting US CPI: Food",
      body: (
        "This is the Consumer Price Index for ALL food — groceries plus " +
        "meals out — in the United States. The number itself is an " +
        "index value (base = 100 in 1982-84), not a dollar amount. " +
        "Watch the 1m / 1w deltas rather than the absolute value. " +
        "Positive = food inflation; negative = deflation. Released " +
        "monthly by the BLS."
      ),
    },
    us_cpi_food_at_home: {
      title: "Interpreting US CPI: Food at Home",
      body: (
        "Same idea as CPI Food, but scoped to GROCERIES only — " +
        "restaurant meals are excluded. Use this one when you want to " +
        "know whether supermarket prices are rising or falling. Index " +
        "value (base = 100 in 1982-84); deltas matter more than the " +
        "level. Monthly release, US-wide."
      ),
    },
    gas_retail: (snap) => {
      const padd = paddFromLabel(snap.label);
      const region = PADD_REGIONS[padd];
      const regionPhrase = region && region.states !== "national"
        ? `the ${region.name} region (${region.states})`
        : region
          ? "the entire US"
          : "this EIA region";
      const title = region
        ? `Retail gasoline (PADD ${padd})`
        : "Retail gasoline";
      return {
        title,
        body: (
          `Weekly average pump price for regular gasoline in ${regionPhrase}. ` +
          "Sourced from EIA. Your local price may differ from the " +
          "regional average."
        ),
      };
    },
  };

  /** Resolve a static-or-dynamic scope entry to its final string. */
  function resolveScope(id, snap) {
    const entry = TRACKER_SCOPE[id];
    if (typeof entry === "function") return entry(snap);
    return entry || null;
  }

  /** Resolve a static-or-dynamic guide entry to a {title, body} pair. */
  function resolveGuide(id, snap) {
    const entry = TRACKER_GUIDE[id];
    if (typeof entry === "function") return entry(snap);
    return entry || null;
  }

  function renderGroupedSections(snapshots) {
    // Bucket snapshots by category, then emit one <section> per
    // category in CATEGORY_META order. This is the "merge same stuff
    // together" behavior — FX pairs stay with FX, food indexes stay
    // with food indexes, etc.
    const buckets = new Map();
    for (const s of snapshots) {
      const key = s.category || "other";
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(s);
    }
    const sortedKeys = [...buckets.keys()].sort((a, b) => {
      const ao = (CATEGORY_META[a] || {}).order ?? 99;
      const bo = (CATEGORY_META[b] || {}).order ?? 99;
      return ao - bo;
    });
    const esc = Panels.escapeHtml;
    const sections = sortedKeys.map((key) => {
      const meta = CATEGORY_META[key] || { label: key };
      const cards = buckets.get(key).map(renderTrackerCard).join("");
      return `
        <section class="trackers-section" data-category="${esc(key)}">
          <h2 class="trackers-section-heading">${esc(meta.label)}</h2>
          <div class="trackers-grid">${cards}</div>
        </section>
      `;
    });
    return sections.join("");
  }

  /** Render one TrackerSnapshot as a card. FX-category trackers get
   *  the interactive converter layout; everything else uses the
   *  read-only layout. */
  function renderTrackerCard(snap) {
    const esc = Panels.escapeHtml;
    const unit = Panels.DISPLAY.trackerUnit[snap.category] || "";

    // Empty-data state: tracker config exists, no latest_value yet.
    if (snap.latest_value === null || snap.latest_value === undefined) {
      const note = snap.note ? ` — ${snap.note}` : "";
      return `
        <article class="card tracker-card no-data" data-tracker-id="${esc(snap.id)}">
          <h3>${esc(snap.label)}</h3>
          <div class="no-data-body">no data yet${esc(note)}</div>
        </article>
      `;
    }

    // FX interactive layout — only when category marks this as an FX
    // pair AND the label parses cleanly. Falls through to the generic
    // layout on any mismatch so exotic config never breaks the grid.
    if (snap.category === "fx") {
      const pair = parseFxPair(snap.label);
      if (pair) {
        return renderFxCard(snap, pair);
      }
    }

    const valueStr = Number(snap.latest_value).toFixed(4);
    const unitPart = unit ? `<span class="unit">${esc(unit)}</span>` : "";

    const dateStr = snap.latest_timestamp
      ? Panels.formatDate(snap.latest_timestamp)
      : "";
    const staleBadge = snap.is_stale
      ? `<span class="badge state-stale">${Panels.DISPLAY.stale.label}</span>`
      : "";

    const deltas = [
      deltaRow("1d", snap.change_1d_pct),
      deltaRow("1w", snap.change_1w_pct),
      deltaRow("1m", snap.change_1m_pct),
    ].join("");

    // Pills: shock dominates; movement=volatile fills in independently.
    const pillsHtml = [];
    if (snap.shock_label) {
      const s = Panels.displayFor(Panels.DISPLAY.trackerShock, snap.shock_label);
      pillsHtml.push(`<span class="badge tracker-${s.cls}">${s.label}</span>`);
    }
    if (snap.movement_label && Panels.DISPLAY.trackerMovement[snap.movement_label]) {
      const m = Panels.DISPLAY.trackerMovement[snap.movement_label];
      pillsHtml.push(`<span class="badge tracker-${m.cls}">${m.label}</span>`);
    }
    const pills = pillsHtml.length
      ? `<div class="pills">${pillsHtml.join(" ")}</div>`
      : "";

    const scope = resolveScope(snap.id, snap);
    const scopeLine = scope
      ? `<div class="tracker-scope">${esc(scope)}</div>`
      : "";
    const guide = resolveGuide(snap.id, snap);
    const infoBtn = guide
      ? `<button type="button" class="tracker-info-btn"
                 aria-label="How to interpret this indicator"
                 title="How to interpret this indicator">ⓘ</button>`
      : "";
    return `
      <article class="card tracker-card" data-tracker-id="${esc(snap.id)}">
        <div class="tracker-header">
          <h3>${esc(snap.label)}</h3>
          ${infoBtn}
        </div>
        ${scopeLine}
        <div class="value-row">
          <span class="value">${esc(valueStr)}</span>
          ${unitPart}
        </div>
        <div class="meta-row">
          ${dateStr ? `<time datetime="${esc(snap.latest_timestamp)}">as of ${esc(dateStr)}</time>` : ""}
          ${staleBadge}
        </div>
        <div class="deltas">${deltas}</div>
        ${pills}
      </article>
    `;
  }

  /** Render an FX pair as an interactive converter card.
   *  Uses data attributes to stash the rate + currency pair; the post-
   *  render hook (wireFxCardHandlers) reads these and wires up input
   *  listeners. Math is local JS — no extra backend calls per
   *  keystroke, which is the whole point of the widget. */
  function renderFxCard(snap, pair) {
    const esc = Panels.escapeHtml;
    const [base, quote] = pair;   // e.g. USD, CNY
    const rate = Number(snap.latest_value);
    const dateStr = snap.latest_timestamp
      ? Panels.formatDate(snap.latest_timestamp)
      : "";
    const staleBadge = snap.is_stale
      ? `<span class="badge state-stale">${Panels.DISPLAY.stale.label}</span>`
      : "";
    const deltas = [
      deltaRow("1d", snap.change_1d_pct),
      deltaRow("1w", snap.change_1w_pct),
    ].join("");
    // Initial display: 1 base → <rate> quote.
    const initialResult = fmtAmount(rate);
    return `
      <article class="card tracker-card tracker-fx"
               data-tracker-id="${esc(snap.id)}"
               data-rate="${rate}"
               data-base="${esc(base)}"
               data-quote="${esc(quote)}"
               data-reversed="false">
        <h3>${esc(snap.label)}</h3>
        <div class="fx-rate-line">
          Rate: <strong>${fmtAmount(rate)}</strong>
          <span class="muted">(1 ${esc(base)} → ${esc(quote)})</span>
        </div>
        <div class="fx-convert">
          <input type="number" step="any" class="fx-amount"
                 value="1" inputmode="decimal"
                 aria-label="Amount" />
          <span class="fx-from-ccy">${esc(base)}</span>
          <button type="button" class="fx-reverse" aria-label="Swap direction">⇄</button>
          <span class="fx-result">${esc(initialResult)}</span>
          <span class="fx-to-ccy">${esc(quote)}</span>
        </div>
        <div class="meta-row">
          ${dateStr ? `<time datetime="${esc(snap.latest_timestamp)}">as of ${esc(dateStr)}</time>` : ""}
          ${staleBadge}
        </div>
        ${deltas ? `<div class="deltas">${deltas}</div>` : ""}
      </article>
    `;
  }

  /** Attach input + click handlers to any FX cards in ``container``.
   *  Called after every innerHTML replacement since listeners don't
   *  survive DOM wipes. Idempotent: only operates on freshly-rendered
   *  cards. */
  function wireFxCardHandlers(container) {
    const cards = container.querySelectorAll(".tracker-fx");
    cards.forEach((card) => {
      const rate = Number(card.dataset.rate);
      if (!isFinite(rate) || rate <= 0) return;
      const input = card.querySelector(".fx-amount");
      const result = card.querySelector(".fx-result");
      const reverse = card.querySelector(".fx-reverse");
      const fromCcyEl = card.querySelector(".fx-from-ccy");
      const toCcyEl = card.querySelector(".fx-to-ccy");
      const base = card.dataset.base;
      const quote = card.dataset.quote;

      function compute() {
        const amt = Number(input.value);
        if (!isFinite(amt)) { result.textContent = "—"; return; }
        const reversed = card.dataset.reversed === "true";
        // Forward: amt base * rate = amt*rate quote
        // Reversed: amt quote / rate = amt/rate base
        const converted = reversed ? amt / rate : amt * rate;
        result.textContent = fmtAmount(converted);
      }

      input.addEventListener("input", compute);
      reverse.addEventListener("click", () => {
        const reversed = card.dataset.reversed === "true";
        card.dataset.reversed = reversed ? "false" : "true";
        // Swap display labels. We don't mutate data-base / data-quote —
        // those record the canonical pair from the backend; the
        // direction is purely a UI choice tracked via reversed flag.
        if (reversed) {
          fromCcyEl.textContent = base;
          toCcyEl.textContent = quote;
        } else {
          fromCcyEl.textContent = quote;
          toCcyEl.textContent = base;
        }
        compute();
      });
    });
  }


  /* Singleton info popover attached to the document. Opens when the
   * user clicks a tracker card's ⓘ button — anchors beneath the
   * button, closes on Escape / outside click. */
  const InfoPopover = (() => {
    let rootEl = null;
    let currentAnchor = null;

    function ensure() {
      if (rootEl) return rootEl;
      rootEl = document.createElement("div");
      rootEl.className = "tracker-info-popover";
      rootEl.setAttribute("role", "tooltip");
      rootEl.hidden = true;
      rootEl.innerHTML = `
        <div class="tracker-info-header">
          <strong class="tracker-info-title"></strong>
          <button type="button" class="tracker-info-close"
                  aria-label="Close">×</button>
        </div>
        <p class="tracker-info-body"></p>
      `;
      document.body.appendChild(rootEl);
      rootEl.querySelector(".tracker-info-close")
        .addEventListener("click", close);
      document.addEventListener("click", (e) => {
        if (!rootEl || rootEl.hidden) return;
        if (rootEl.contains(e.target)) return;
        if (currentAnchor && currentAnchor.contains(e.target)) return;
        close();
      }, true);
      document.addEventListener("keydown", (e) => {
        if (!rootEl || rootEl.hidden) return;
        if (e.key === "Escape") { e.preventDefault(); close(); }
      });
      window.addEventListener("resize", close);
      return rootEl;
    }

    function open(anchor, title, body) {
      ensure();
      currentAnchor = anchor;
      rootEl.querySelector(".tracker-info-title").textContent = title || "";
      rootEl.querySelector(".tracker-info-body").textContent = body || "";
      rootEl.hidden = false;
      // Position below the button, clamped inside the viewport.
      const r = anchor.getBoundingClientRect();
      const popW = 320;
      const gap = 6;
      let left = r.left + r.width / 2 - popW / 2;
      if (left < 8) left = 8;
      if (left + popW > window.innerWidth - 8) {
        left = Math.max(8, window.innerWidth - popW - 8);
      }
      const top = r.bottom + gap;
      rootEl.style.left = left + "px";
      rootEl.style.top = top + "px";
    }

    function close() {
      if (!rootEl) return;
      rootEl.hidden = true;
      currentAnchor = null;
    }

    return { open, close };
  })();

  /** Wire ⓘ info buttons on every tracker card that has a guide
   *  defined in TRACKER_GUIDE. Called after every innerHTML wipe
   *  since DOM listeners don't survive those.
   *
   *  Guide entries can be functions — resolveGuide() needs a snapshot-
   *  shaped object. We rebuild the minimal {id, label} from the DOM
   *  rather than threading the full snapshot through innerHTML. */
  function wireInfoButtons(container) {
    container.querySelectorAll(".tracker-info-btn").forEach((btn) => {
      const card = btn.closest(".tracker-card");
      const id = card?.dataset.trackerId;
      if (!id) return;
      const label = card.querySelector("h3")?.textContent?.trim() || "";
      const guide = resolveGuide(id, { id, label });
      if (!guide) return;
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        InfoPopover.open(btn, guide.title, guide.body);
      });
    });
  }

  /** Render one delta-row (1d/1w/1m) with arrow + signed percent.
   *  Null pct becomes a muted em-dash so the row stays aligned. */
  function deltaRow(label, pct) {
    const esc = Panels.escapeHtml;
    if (pct === null || pct === undefined) {
      return `
        <div class="delta-row empty">
          <span class="delta-label">${esc(label)}</span>
          <span class="delta-arrow">—</span>
          <span class="delta-value muted">n/a</span>
        </div>
      `;
    }
    const n = Number(pct);
    let cls, arrow, sign;
    if (n > 0)      { cls = "up";   arrow = "\u2191"; sign = "+"; }
    else if (n < 0) { cls = "down"; arrow = "\u2193"; sign = "";  }
    else            { cls = "flat"; arrow = "\u2014"; sign = "";  }
    return `
      <div class="delta-row ${cls}">
        <span class="delta-label">${esc(label)}</span>
        <span class="delta-arrow">${arrow}</span>
        <span class="delta-value">${sign}${n.toFixed(2)}%</span>
      </div>
    `;
  }

  // --- exports -------------------------------------------------------------

  window.Trackers = {
    mountTrackersPanel,
    renderTrackerCard,
  };
})();
