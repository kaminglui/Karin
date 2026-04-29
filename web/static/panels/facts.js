/* facts.js — year-card aggregator widget.
 *
 * Pulls /api/facts and renders the sections (world_population,
 * region_population, inflation_baseline) as a layout of cards. No
 * chart for v1 — the inflation + population panels already cover
 * time-series visuals; this widget's job is the "snapshot" view.
 */
(function () {
  "use strict";

  function fmtPop(n) {
    if (n == null || !Number.isFinite(n)) return "—";
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return String(n);
  }

  function fmtMoney(v, digits) {
    const d = digits == null ? 2 : digits;
    if (v == null || !Number.isFinite(v)) return "—";
    return v.toLocaleString(undefined, {
      minimumFractionDigits: d, maximumFractionDigits: d,
    });
  }

  function sectionHtml(d) {
    const cells = [];
    const sec = d.sections || {};

    const wp = sec.world_population;
    if (wp && !wp.error && wp.population != null) {
      const now = wp.now;
      const arrow = now ? ` → ${fmtPop(now.population)}` : "";
      const sub = now
        ? `+${(now.change_pct || 0).toFixed(1)}% in ${now.years_span} yrs (today: ${now.year})`
        : "people";
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">World population (${wp.year})</div>
          <div class="infl-metric-value">${fmtPop(wp.population)}${arrow}</div>
          <div class="infl-metric-sub">${sub}</div>
        </div>
      `);
    }

    const rp = sec.region_population;
    if (rp && !rp.error && rp.population != null) {
      const now = rp.now;
      const arrow = now ? ` → ${fmtPop(now.population)}` : "";
      const sub = now
        ? `${now.change_pct >= 0 ? "+" : ""}${(now.change_pct || 0).toFixed(1)}% in ${now.years_span} yrs (today: ${now.year})`
        : "people";
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">${Panels.escapeHtml(rp.country)} (${rp.year})</div>
          <div class="infl-metric-value">${fmtPop(rp.population)}${arrow}</div>
          <div class="infl-metric-sub">${sub}</div>
        </div>
      `);
    }

    const ib = sec.inflation_baseline;
    if (ib && !ib.error && ib.amount_output != null) {
      const cur = ib.currency === "HKD" ? "HK$"
        : ib.currency === "JPY" || ib.currency === "CNY" ? "¥"
        : ib.currency === "KRW" ? "₩"
        : "$";
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">${cur}1 in ${ib.from_year} →</div>
          <div class="infl-metric-value">${cur}${fmtMoney(ib.amount_output)}</div>
          <div class="infl-metric-sub">in ${ib.to_year} (${Panels.escapeHtml(ib.country)} CPI · ${fmtMoney(ib.ratio, 2)}×)</div>
        </div>
      `);
    }

    const co = sec.cohort_age;
    if (co && co.age != null) {
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">Born in ${co.year_born} →</div>
          <div class="infl-metric-value">~${co.age} years old</div>
          <div class="infl-metric-sub">today (${co.current_year})</div>
        </div>
      `);
    }

    const ws = sec.wages_snapshot;
    if (ws && ws.wage_hourly_usd != null) {
      const now = ws.now;
      const arrow = now ? ` → $${fmtMoney(now.wage_hourly_usd)}` : "";
      const sub = now
        ? `${now.ratio.toFixed(2)}× nominal in ${now.years_span} yrs (today: ${now.year})`
        : "avg production worker";
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">US hourly wage (${ws.year})</div>
          <div class="infl-metric-value">$${fmtMoney(ws.wage_hourly_usd)}${arrow}</div>
          <div class="infl-metric-sub">${sub}</div>
        </div>
      `);
    }

    // Item-price highlights — render as a four-column grid (label /
    // price / multiplier / unit). Each row's spans become grid items
    // via `li { display: contents }` in panels.css, so columns align
    // vertically across rows.
    const ih = sec.item_highlights;
    let itemsBlock = "";
    if (ih && Array.isArray(ih.items) && ih.items.length > 0) {
      const rows = ih.items.map((it) => {
        const priceText = (it.price_now != null)
          ? `$${fmtMoney(it.price)} → $${fmtMoney(it.price_now)}`
          : `$${fmtMoney(it.price)}`;
        const multText = (it.price_now != null && it.ratio != null)
          ? `${(it.ratio).toFixed(2)}×`
          : "";
        return `<li>
          <span class="facts-item-label">${Panels.escapeHtml(it.label)}</span>
          <span class="facts-item-price">${priceText}</span>
          <span class="facts-item-mult">${multText}</span>
          <span class="facts-item-unit">${Panels.escapeHtml(it.unit)}</span>
        </li>`;
      }).join("");
      const titleSuffix = ih.now_year ? ` vs ${ih.now_year}` : "";
      itemsBlock = `
        <div class="facts-items">
          <div class="facts-items-title">US prices in ${ih.year}${titleSuffix}</div>
          <ul class="facts-items-list">${rows}</ul>
        </div>
      `;
    }

    // Wiki year-article — v3 prefers an events bullet list when the
    // server-side parser found Events section entries; falls back to
    // the lead summary when only `extract` is present.
    const wy = sec.wiki_year;
    let wikiBlock = "";
    if (wy && Array.isArray(wy.events) && wy.events.length > 0) {
      const items = wy.events.slice(0, 3).map((e) =>
        `<li>${Panels.escapeHtml(e)}</li>`
      ).join("");
      wikiBlock = `<div class="facts-wiki">
        <div class="facts-wiki-title">Notable events in ${Panels.escapeHtml(String(wy.title))}</div>
        <ul class="facts-wiki-events">${items}</ul>
        <div class="infl-source"><a href="${Panels.escapeHtml(wy.source_url)}" target="_blank" rel="noopener">Wikipedia · ${Panels.escapeHtml(wy.title)}</a></div>
      </div>`;
    } else if (wy && wy.extract) {
      wikiBlock = `<div class="facts-wiki">
        <div class="infl-interp">${Panels.escapeHtml(wy.extract)}</div>
        <div class="infl-source"><a href="${Panels.escapeHtml(wy.source_url)}" target="_blank" rel="noopener">Wikipedia · ${Panels.escapeHtml(wy.title)}</a></div>
      </div>`;
    }

    // Source citations — combine all sections that have a source_url
    // (excluding wiki, which has its own block).
    const sources = [];
    [wp, rp, ws].forEach((s) => {
      if (s && s.source_url && !sources.find((x) => x.url === s.source_url)) {
        sources.push({ name: s.source || "Source", url: s.source_url });
      }
    });
    if (ib && ib.source_url && !sources.find((x) => x.url === ib.source_url)) {
      sources.push({ name: ib.source || "BLS", url: ib.source_url });
    }
    if (ih && ih.source_url && !sources.find((x) => x.url === ih.source_url)) {
      sources.push({ name: ih.source || "BLS AP", url: ih.source_url });
    }
    const sourceLine = sources.length === 0 ? "" : sources.map((s) =>
      `<a href="${Panels.escapeHtml(s.url)}" target="_blank" rel="noopener">${Panels.escapeHtml(s.name)}</a>`
    ).join(" · ");

    // Surface section errors so the user sees what's missing.
    const errors = [];
    Object.entries(sec).forEach(([k, v]) => {
      if (v && v.error) errors.push(`${k.replace(/_/g, " ")}: ${v.error}`);
    });
    const errBlock = errors.length === 0 ? ""
      : `<div class="infl-wages-note">${Panels.escapeHtml(errors.join(" · "))}</div>`;

    return `
      <div class="infl-card">
        ${wikiBlock}
        <div class="infl-grid">${cells.join("")}</div>
        ${itemsBlock}
        ${errBlock}
        ${sourceLine ? `<div class="infl-source">${sourceLine}</div>` : ""}
      </div>
    `;
  }

  function mountFactsPanel(container, options = {}) {
    const opts = {
      year: null,
      region: null,
      onSubtitle: null,
      ...options,
    };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      if (opts.year == null) {
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "no year",
          message: "Facts widget needs a year.",
        });
        return;
      }
      try {
        const params = new URLSearchParams({ year: String(opts.year) });
        if (opts.region) params.set("region", opts.region);
        const r = await Panels.safeFetch(`/api/facts?${params.toString()}`);
        const data = await r.json();
        if (aborted) return;
        if (data.error) {
          container.innerHTML = Panels.stateCard({
            kind: "error", message: data.error,
          });
          if (opts.onSubtitle) opts.onSubtitle("error");
          return;
        }
        container.innerHTML = sectionHtml(data);
        if (opts.onSubtitle) {
          const tag = data.region ? `${data.year} · ${data.region}` : `${data.year}`;
          opts.onSubtitle(tag);
        }
      } catch (e) {
        if (aborted) return;
        container.innerHTML = Panels.stateCard({
          kind: "error", message: `Facts lookup failed: ${e.message}`,
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
    window.Facts = { mountFactsPanel };
  }
})();
