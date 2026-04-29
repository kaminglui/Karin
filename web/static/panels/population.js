/* population.js — World Bank population time-series widget (Plotly).
 *
 * Pulls /api/population and renders:
 *   - A summary card with the queried value (single year) or
 *     before/after + change% (range query)
 *   - A Plotly line chart of the full population history for the
 *     region, with the queried year(s) highlighted
 *
 * Theme matches the inflation panel (purple/red palette, no toolbar,
 * scroll-zoom, drag-pan, double-click reset). Reuses the .infl-*
 * CSS classes via the comma-grouped selectors in style.css.
 */
(function () {
  "use strict";

  const PALETTE = {
    line: "#3B82F6",
    marker: "#D94F6B",
  };

  function fmtPop(n) {
    if (n == null || !Number.isFinite(n)) return "—";
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return String(n);
  }

  function fmtPct(p) {
    if (p == null || !Number.isFinite(p)) return "—";
    return `${p >= 0 ? "+" : ""}${p.toFixed(2)}%`;
  }

  function summaryHtml(d) {
    const cells = [];
    if (d.year != null && d.population != null) {
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">${Panels.escapeHtml(d.country)} (${d.year})</div>
          <div class="infl-metric-value">${fmtPop(d.population)}</div>
          <div class="infl-metric-sub">people</div>
        </div>
      `);
    } else if (d.from_year != null && d.to_year != null) {
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">${Panels.escapeHtml(d.country)}</div>
          <div class="infl-metric-value">${fmtPop(d.population_from)} → ${fmtPop(d.population_to)}</div>
          <div class="infl-metric-sub">${d.from_year} → ${d.to_year}</div>
        </div>
      `);
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">Change</div>
          <div class="infl-metric-value">${fmtPct(d.change_pct)}</div>
          <div class="infl-metric-sub">${(d.change_abs >= 0 ? "+" : "")}${(d.change_abs ?? 0).toLocaleString()} people</div>
        </div>
      `);
    }
    const sourceLink = d.source_url
      ? `<a href="${Panels.escapeHtml(d.source_url)}" target="_blank" rel="noopener">${Panels.escapeHtml(d.source || "World Bank")}</a>`
      : Panels.escapeHtml(d.source || "");
    const dataAsOf = d.data_as_of ? ` · data as of ${Panels.escapeHtml(d.data_as_of)}` : "";
    const interp = d.interpretation ? `<div class="infl-interp">${Panels.escapeHtml(d.interpretation)}</div>` : "";
    return `
      <div class="infl-card">
        <div class="infl-grid">${cells.join("")}</div>
        ${interp}
        <div class="infl-source">${sourceLink}${dataAsOf}</div>
      </div>
    `;
  }

  function buildTraces(d) {
    const traces = [];
    const ann = [];
    const series = d.series;
    if (!series) return { traces, annotations: ann };

    traces.push({
      x: series.years,
      y: series.values,
      mode: "lines",
      type: "scatter",
      name: series.label,
      line: { color: PALETTE.line, width: 2 },
      hovertemplate: "%{x}: %{y:,} people<extra></extra>",
    });

    // Highlight queried years.
    const highlights = [];
    if (d.year != null) highlights.push(d.year);
    if (d.from_year != null) highlights.push(d.from_year);
    if (d.to_year != null && d.to_year !== d.from_year) highlights.push(d.to_year);
    highlights.forEach((y) => {
      const idx = series.years.indexOf(y);
      if (idx < 0) return;
      const v = series.values[idx];
      traces.push({
        x: [y], y: [v],
        mode: "markers",
        type: "scatter",
        marker: {
          color: PALETTE.marker, size: 10, symbol: "circle",
          line: { color: "#fff", width: 1.5 },
        },
        name: `${y}`,
        hovertemplate: `${y}: %{y:,}<extra></extra>`,
        showlegend: false,
      });
      ann.push({
        x: y, y: v, xref: "x", yref: "y",
        text: `${y}: ${fmtPop(v)}`,
        showarrow: true, arrowhead: 0, arrowsize: 0.8, ax: 0, ay: -22,
        font: { size: 10, color: PALETTE.line },
        bgcolor: "rgba(255,255,255,0.85)",
        bordercolor: PALETTE.line, borderwidth: 1, borderpad: 2,
      });
    });
    return { traces, annotations: ann };
  }

  function rankSummaryHtml(d) {
    const cells = [];
    if (d.region_position) {
      const rp = d.region_position;
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">${Panels.escapeHtml(rp.country)} rank</div>
          <div class="infl-metric-value">#${rp.rank}</div>
          <div class="infl-metric-sub">of ${rp.out_of} countries</div>
        </div>
      `);
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">${Panels.escapeHtml(rp.country)} population</div>
          <div class="infl-metric-value">${fmtPop(rp.population)}</div>
          <div class="infl-metric-sub">in ${d.year}</div>
        </div>
      `);
    } else {
      cells.push(`
        <div class="infl-metric">
          <div class="infl-metric-label">Year</div>
          <div class="infl-metric-value">${d.year}</div>
          <div class="infl-metric-sub">${d.total_countries} countries</div>
        </div>
      `);
    }
    const sourceLink = d.source_url
      ? `<a href="${Panels.escapeHtml(d.source_url)}" target="_blank" rel="noopener">${Panels.escapeHtml(d.source || "World Bank")}</a>`
      : Panels.escapeHtml(d.source || "");
    const dataAsOf = d.data_as_of ? ` · data as of ${Panels.escapeHtml(d.data_as_of)}` : "";
    const interp = d.interpretation
      ? `<div class="infl-interp">${Panels.escapeHtml(d.interpretation)}</div>` : "";
    return `
      <div class="infl-card">
        <div class="infl-grid">${cells.join("")}</div>
        ${interp}
        <div class="infl-source">${sourceLink}${dataAsOf}</div>
      </div>
    `;
  }

  function renderRank(container, d, opts) {
    container.innerHTML = `
      ${rankSummaryHtml(d)}
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
    const top = d.top || [];
    // Reverse so rank #1 is at top of the chart.
    const reversed = [...top].reverse();
    const labels = reversed.map((r) => `${r.rank}. ${r.country}`);
    const values = reversed.map((r) => r.population);
    const highlightIso = d.region_position ? d.region_position.iso3 : null;
    const colors = reversed.map((r) =>
      r.iso3 === highlightIso ? PALETTE.marker : PALETTE.line,
    );
    const traces = [{
      x: values,
      y: labels,
      type: "bar",
      orientation: "h",
      marker: { color: colors },
      hovertemplate: "%{y}: %{x:,}<extra></extra>",
    }];
    const layout = {
      margin: { l: 170, r: 20, t: 8, b: 32 },
      xaxis: {
        gridcolor: "rgba(139, 95, 191, 0.15)",
        tickfont: { size: 10 },
        tickformat: ".2s",
      },
      yaxis: {
        tickfont: { size: 10 },
        automargin: true,
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.5)",
      showlegend: false,
      autosize: true,
      // Bar charts are tall when N is high — bump min height proportionally.
    };
    const config = {
      displayModeBar: false,
      responsive: true,
      scrollZoom: false,
      doubleClick: "reset+autosize",
    };
    plotEl.style.height = `${Math.max(220, top.length * 28)}px`;
    window.Plotly.newPlot(plotEl, traces, layout, config);
    if (opts.onSubtitle) {
      const tag = d.region_position
        ? `${d.region_position.country} #${d.region_position.rank} of ${d.region_position.out_of}`
        : `top ${top.length} in ${d.year}`;
      opts.onSubtitle(tag);
    }
  }

  function render(container, d, opts) {
    // Rank mode has a different shape (no `series`, has `top`) — render
    // as a horizontal bar chart instead of the time-series line.
    if (Array.isArray(d.top)) {
      return renderRank(container, d, opts);
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
    const layout = {
      margin: { l: 60, r: 14, t: 8, b: 32 },
      xaxis: {
        zeroline: false,
        gridcolor: "rgba(139, 95, 191, 0.15)",
        tickfont: { size: 10 },
        showspikes: true, spikemode: "across", spikedash: "dot",
        spikethickness: 1, spikecolor: "rgba(75, 40, 115, 0.55)",
      },
      yaxis: {
        title: { text: "Population", font: { size: 11 } },
        zeroline: false,
        gridcolor: "rgba(139, 95, 191, 0.15)",
        tickfont: { size: 10 },
        tickformat: ".2s",
        showspikes: true, spikemode: "across", spikedash: "dot",
        spikethickness: 1, spikecolor: "rgba(75, 40, 115, 0.55)",
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.5)",
      showlegend: false,
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
      const tag = d.from_year && d.to_year
        ? `${d.from_year}→${d.to_year}`
        : (d.year != null ? `${d.year}` : "");
      opts.onSubtitle(`${d.country || "?"} · ${tag}`.trim());
    }
  }

  function mountPopulationPanel(container, options = {}) {
    const opts = {
      region: "world",
      year: null,
      from_year: null,
      to_year: null,
      metric: "value",
      top: 10,
      onSubtitle: null,
      ...options,
    };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      try {
        const params = new URLSearchParams({ region: opts.region || "world" });
        if (opts.year != null) params.set("year", String(opts.year));
        if (opts.from_year != null) params.set("from_year", String(opts.from_year));
        if (opts.to_year != null) params.set("to_year", String(opts.to_year));
        if (opts.metric && opts.metric !== "value") params.set("metric", opts.metric);
        if (opts.top != null && opts.top !== 10) params.set("top", String(opts.top));
        const r = await Panels.safeFetch(`/api/population?${params.toString()}`);
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
          kind: "error", message: `Population lookup failed: ${e.message}`,
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
    window.Population = { mountPopulationPanel };
  }
})();
