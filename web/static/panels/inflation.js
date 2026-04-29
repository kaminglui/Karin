/* inflation.js — historical purchasing-power widget (Plotly).
 *
 * Pulls /api/inflation and renders:
 *   - A summary card (amount-in-from-year → amount-in-to-year, ratio,
 *     real-wage delta when applicable, source citation)
 *   - A Plotly chart of the CPI-equivalent value of the user's $amount
 *     plotted forward from from_year. When measure='wages' or 'both',
 *     the chart also overlays nominal + real (from_year-dollars) hourly
 *     wages so the user can see at a glance whether wages outpaced
 *     prices.
 *
 * Theme matches graph.js (purple-leaning palette, no toolbar, drag/pan
 * + wheel-zoom + dbl-click reset).
 */
(function () {
  "use strict";

  const PALETTE = {
    cpi: "#8B5FBF",
    wages_nominal: "#10B981",
    wages_real: "#F59E0B",
    item: "#3B82F6",
    marker: "#D94F6B",
  };

  function fmtMoney(v, digits) {
    const d = digits == null ? 2 : digits;
    if (v == null || !Number.isFinite(v)) return "—";
    return v.toLocaleString(undefined, {
      minimumFractionDigits: d, maximumFractionDigits: d,
    });
  }

  function summaryHtml(d) {
    const cpi = d.cpi || {};
    const w = d.wages || {};
    const fy = d.from_year, ty = d.to_year;
    const amt = d.amount_input ?? 1;
    const cur = d.currency_symbol || "$";
    // Build the metric grid — CPI always, wages when present without error.
    const cells = [];
    cells.push(`
      <div class="infl-metric">
        <div class="infl-metric-label">${cur}${fmtMoney(amt)} in ${fy} →</div>
        <div class="infl-metric-value">${cur}${fmtMoney(cpi.amount_output)}</div>
        <div class="infl-metric-sub">in ${ty} (CPI · ${fmtMoney(cpi.ratio, 2)}×)</div>
      </div>
    `);
    if (w && !w.error && w.wage_from != null) {
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">Avg hourly wage</div>
          <div class="infl-metric-value">
            $${fmtMoney(w.wage_from)} → $${fmtMoney(w.wage_to)}
          </div>
          <div class="infl-metric-sub">${fmtMoney(w.wage_ratio, 2)}× nominal</div>
        </div>
      `);
    }
    const it = d.item;
    if (it && !it.error && it.price_from != null) {
      const subParts = [];
      if (it.price_to != null) {
        subParts.push(`${fmtMoney(it.nominal_change, 2)}× nominal`);
      }
      subParts.push(`≈ $${fmtMoney(it.today_cpi_equivalent)} in ${ty} (CPI-adj)`);
      const valueLine = it.price_to != null
        ? `$${fmtMoney(it.price_from)} → $${fmtMoney(it.price_to)}`
        : `$${fmtMoney(it.price_from)}`;
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">${Panels.escapeHtml(it.label)}<br>(${Panels.escapeHtml(it.unit)})</div>
          <div class="infl-metric-value">${valueLine}</div>
          <div class="infl-metric-sub">${subParts.join(" · ")}</div>
        </div>
      `);
    }
    if (d.real_wage_delta != null) {
      const pct = (d.real_wage_delta - 1) * 100;
      const dir = pct >= 0 ? "outpaced" : "lagged";
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">Real wages vs prices</div>
          <div class="infl-metric-value">${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%</div>
          <div class="infl-metric-sub">real wages ${dir} prices</div>
        </div>
      `);
    }
    let wagesNote = "";
    if (w && w.error) {
      wagesNote = `<div class="infl-wages-note">${Panels.escapeHtml(w.error)}</div>`;
    }
    const sourceLink = cpi.source_url
      ? `<a href="${Panels.escapeHtml(cpi.source_url)}" target="_blank" rel="noopener">${Panels.escapeHtml(cpi.source || "BLS")}</a>`
      : Panels.escapeHtml(cpi.source || "");
    const dataAsOf = cpi.data_as_of ? ` · data as of ${Panels.escapeHtml(cpi.data_as_of)}` : "";
    const interp = d.interpretation ? `<div class="infl-interp">${Panels.escapeHtml(d.interpretation)}</div>` : "";
    return `
      <div class="infl-card">
        <div class="infl-grid">${cells.join("")}</div>
        ${interp}
        ${wagesNote}
        <div class="infl-source">${sourceLink}${dataAsOf}</div>
      </div>
    `;
  }

  function buildTraces(d) {
    const traces = [];
    const ann = [];
    const series = d.series || {};
    const fy = d.from_year, ty = d.to_year;

    if (series.cpi) {
      traces.push({
        x: series.cpi.years,
        y: series.cpi.values,
        mode: "lines",
        type: "scatter",
        name: "CPI-equivalent $",
        line: { color: PALETTE.cpi, width: 2 },
        hovertemplate: "%{x}: $%{y:,.2f}<extra>CPI-equivalent</extra>",
      });
      // Highlight the to_year value.
      const idx = series.cpi.years.indexOf(ty);
      if (idx >= 0) {
        traces.push({
          x: [ty], y: [series.cpi.values[idx]],
          mode: "markers",
          type: "scatter",
          marker: { color: PALETTE.marker, size: 10, symbol: "circle",
                    line: { color: "#fff", width: 1.5 } },
          name: `${ty}`,
          hovertemplate: `${ty}: $%{y:,.2f}<extra></extra>`,
          showlegend: false,
        });
        ann.push({
          x: ty, y: series.cpi.values[idx], xref: "x", yref: "y",
          text: `$${series.cpi.values[idx].toFixed(2)} in ${ty}`,
          showarrow: true, arrowhead: 0, arrowsize: 0.8, ax: 0, ay: -22,
          font: { size: 10, color: PALETTE.cpi },
          bgcolor: "rgba(255,255,255,0.85)",
          bordercolor: PALETTE.cpi, borderwidth: 1, borderpad: 2,
        });
      }
    }

    if (series.wages_nominal) {
      traces.push({
        x: series.wages_nominal.years,
        y: series.wages_nominal.values,
        mode: "lines",
        type: "scatter",
        name: "Nominal hourly wage",
        line: { color: PALETTE.wages_nominal, width: 2, dash: "dot" },
        yaxis: "y2",
        hovertemplate: "%{x}: $%{y:,.2f}/hr<extra>nominal</extra>",
      });
    }
    if (series.wages_real) {
      traces.push({
        x: series.wages_real.years,
        y: series.wages_real.values,
        mode: "lines",
        type: "scatter",
        name: `Real wage (${fy} $)`,
        line: { color: PALETTE.wages_real, width: 2 },
        yaxis: "y2",
        hovertemplate: `%{x}: $%{y:,.2f}/hr<extra>in ${fy} dollars</extra>`,
      });
    }

    if (series.item) {
      const itemAxis = series.wages_nominal ? "y2" : "y";
      traces.push({
        x: series.item.years,
        y: series.item.values,
        mode: "lines",
        type: "scatter",
        name: series.item.label,
        line: { color: PALETTE.item, width: 2 },
        yaxis: itemAxis,
        hovertemplate: `%{x}: $%{y:,.2f}<extra>${series.item.label}</extra>`,
      });
    }

    return { traces, annotations: ann };
  }

  function comparisonSummaryHtml(d) {
    const cells = [];
    const compare = d.comparison || {};
    const order = d.regions_order || Object.keys(compare);
    order.forEach((r) => {
      const b = compare[r];
      if (!b || b.error) return;
      const sym = b.currency_symbol || "$";
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">${Panels.escapeHtml(b.country)}</div>
          <div class="infl-metric-value">${sym}${fmtMoney(b.amount_input)} → ${sym}${fmtMoney(b.amount_output)}</div>
          <div class="infl-metric-sub">${b.from_year}→${b.to_year} · ${fmtMoney(b.ratio, 2)}× CPI</div>
        </div>
      `);
    });
    const interp = d.interpretation
      ? `<div class="infl-interp">${Panels.escapeHtml(d.interpretation)}</div>` : "";
    // Source citations — dedupe by URL.
    const seen = new Set();
    const links = [];
    Object.values(compare).forEach((b) => {
      if (b && !b.error && b.source_url && !seen.has(b.source_url)) {
        seen.add(b.source_url);
        links.push(`<a href="${Panels.escapeHtml(b.source_url)}" target="_blank" rel="noopener">${Panels.escapeHtml(b.source || "Source")}</a>`);
      }
    });
    const sourceLine = links.length ? `<div class="infl-source">${links.join(" · ")}</div>` : "";
    return `
      <div class="infl-card">
        <div class="infl-grid">${cells.join("")}</div>
        ${interp}
        ${sourceLine}
      </div>
    `;
  }

  function renderComparison(container, d, opts) {
    container.innerHTML = `
      ${comparisonSummaryHtml(d)}
      <div class="infl-plot"></div>
    `;
    const plotEl = container.querySelector(".infl-plot");
    if (!window.Plotly) {
      plotEl.innerHTML = Panels.stateCard({
        kind: "error",
        message: "Plotly didn't load — reload the page.",
      });
      return;
    }
    const palette = ["#8B5FBF", "#10B981", "#3B82F6", "#F59E0B", "#D94F6B"];
    const traces = [];
    const series = d.series_per_region || {};
    const order = d.regions_order || Object.keys(series);
    order.forEach((r, i) => {
      const s = series[r];
      if (!s) return;
      traces.push({
        x: s.years,
        y: s.values,
        mode: "lines",
        type: "scatter",
        name: s.label,
        line: { color: palette[i % palette.length], width: 2 },
        hovertemplate: `%{x}: %{y:.2f}× ${s.currency_symbol || "$"}<extra>${s.label}</extra>`,
      });
    });
    const layout = {
      margin: { l: 50, r: 14, t: 8, b: 32 },
      xaxis: {
        zeroline: false,
        gridcolor: "rgba(139, 95, 191, 0.15)",
        tickfont: { size: 10 },
        showspikes: true, spikemode: "across", spikedash: "dot",
        spikethickness: 1, spikecolor: "rgba(75, 40, 115, 0.55)",
      },
      yaxis: {
        title: { text: `value (=1 at ${d.from_year})`, font: { size: 11 } },
        zeroline: false,
        gridcolor: "rgba(139, 95, 191, 0.15)",
        tickfont: { size: 10 },
        tickformat: ".2f",
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.5)",
      showlegend: true,
      legend: { orientation: "h", y: -0.2, font: { size: 10 } },
      autosize: true,
      dragmode: "pan",
      hovermode: "closest",
    };
    const config = {
      displayModeBar: false,
      responsive: true,
      scrollZoom: true,
      doubleClick: "reset+autosize",
    };
    window.Plotly.newPlot(plotEl, traces, layout, config);
    if (opts.onSubtitle) {
      opts.onSubtitle(`${order.length} regions · ${d.from_year}-${d.to_year}`);
    }
  }

  function render(container, d, opts) {
    // Comparison mode — overlay multiple regions on a normalized scale.
    if (d.comparison && d.series_per_region) {
      return renderComparison(container, d, opts);
    }
    container.innerHTML = `
      ${summaryHtml(d)}
      <div class="infl-plot"></div>
    `;
    const plotEl = container.querySelector(".infl-plot");
    if (!window.Plotly) {
      plotEl.innerHTML = Panels.stateCard({
        kind: "error",
        message: "Plotly didn't load — reload the page.",
      });
      return;
    }
    const { traces, annotations } = buildTraces(d);
    const showWageAxis = traces.some((t) => t.yaxis === "y2");
    const layout = {
      margin: { l: 50, r: showWageAxis ? 50 : 14, t: 8, b: 32 },
      xaxis: {
        zeroline: false,
        gridcolor: "rgba(139, 95, 191, 0.15)",
        tickfont: { size: 10 },
        showspikes: true, spikemode: "across", spikedash: "dot",
        spikethickness: 1, spikecolor: "rgba(75, 40, 115, 0.55)",
      },
      yaxis: {
        title: { text: "$ value", font: { size: 11 } },
        zeroline: false,
        gridcolor: "rgba(139, 95, 191, 0.15)",
        tickfont: { size: 10 },
        tickformat: "$,.2f",
        showspikes: true, spikemode: "across", spikedash: "dot",
        spikethickness: 1, spikecolor: "rgba(75, 40, 115, 0.55)",
      },
      yaxis2: showWageAxis ? {
        title: { text: "$/hour", font: { size: 11 } },
        overlaying: "y",
        side: "right",
        zeroline: false,
        showgrid: false,
        tickfont: { size: 10 },
        tickformat: "$,.2f",
      } : undefined,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.5)",
      showlegend: traces.filter((t) => t.showlegend !== false).length > 1,
      legend: { orientation: "h", y: -0.18, font: { size: 10 } },
      autosize: true,
      dragmode: "pan",
      hovermode: "closest",
      annotations,
    };
    const config = {
      displayModeBar: false,
      responsive: true,
      scrollZoom: true,
      doubleClick: "reset+autosize",
    };
    window.Plotly.newPlot(plotEl, traces, layout, config);

    if (opts.onSubtitle) {
      const cpi = d.cpi || {};
      const sub = `${d.from_year}→${d.to_year} · ${fmtMoney(cpi.ratio, 2)}× CPI`;
      opts.onSubtitle(sub);
    }
  }

  function mountInflationPanel(container, options = {}) {
    const opts = {
      amount: 1.0,
      from_year: null,
      to_year: null,
      measure: "cpi",
      item: null,
      region: "us",
      regions: null,
      onSubtitle: null,
      ...options,
    };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      if (opts.from_year == null) {
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "no year",
          message: "Inflation widget needs a from_year.",
        });
        return;
      }
      try {
        const params = new URLSearchParams({
          amount: String(opts.amount || 1.0),
          from_year: String(opts.from_year),
          measure: opts.measure || "cpi",
        });
        if (opts.to_year != null) params.set("to_year", String(opts.to_year));
        if (opts.item) params.set("item", opts.item);
        if (opts.region && opts.region !== "us") params.set("region", opts.region);
        if (opts.regions) params.set("regions", opts.regions);
        const r = await Panels.safeFetch(`/api/inflation?${params.toString()}`);
        const data = await r.json();
        if (aborted) return;
        if (data.error) {
          container.innerHTML = Panels.stateCard({
            kind: "error", message: data.error,
          });
          if (opts.onSubtitle) opts.onSubtitle("error");
          return;
        }
        render(container, data, opts);
      } catch (e) {
        if (aborted) return;
        container.innerHTML = Panels.stateCard({
          kind: "error", message: `Inflation lookup failed: ${e.message}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
      }
    }

    load();
    return {
      unmount: () => {
        aborted = true;
        try {
          if (window.Plotly) {
            const el = container.querySelector(".infl-plot");
            if (el) window.Plotly.purge(el);
          }
        } catch (_) { /* ignore */ }
        container.innerHTML = "";
      },
      reload: load,
    };
  }

  if (typeof window !== "undefined") {
    window.Inflation = { mountInflationPanel };
  }
})();
