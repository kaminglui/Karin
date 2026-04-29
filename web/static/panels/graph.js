/* graph.js — grapher widget powered by Plotly.
 *
 * Pulls /api/graph sampled data and renders one or more line series
 * on a single axis. Layout is restrained (no toolbar, no gridlines
 * heavier than the chat's theme) so it doesn't clash with the rest
 * of the UI.
 */
(function () {
  "use strict";

  function mountGraphPanel(container, options = {}) {
    const opts = {
      expression: null,
      variable: "x",
      xMin: -10, xMax: 10,
      onSubtitle: null,
      ...options,
    };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      const expr = (opts.expression || "").trim();
      if (!expr) {
        container.innerHTML = Panels.stateCard({
          kind: "empty", label: "no expression",
          message: "Graph called without an expression.",
        });
        return;
      }
      try {
        const p = new URLSearchParams({
          expression: expr,
          variable: opts.variable || "x",
          x_min: String(opts.xMin),
          x_max: String(opts.xMax),
        });
        const r = await Panels.safeFetch(`/api/graph?${p.toString()}`);
        const data = await r.json();
        if (aborted) return;
        if (data.error) {
          container.innerHTML = Panels.stateCard({
            kind: "error", message: data.error,
          });
          if (opts.onSubtitle) opts.onSubtitle("error");
          return;
        }
        render(data);
      } catch (e) {
        if (aborted) return;
        container.innerHTML = Panels.stateCard({
          kind: "error", message: `Graph failed: ${e.message}`,
        });
        if (opts.onSubtitle) opts.onSubtitle("error");
      }
    }

    function render(data) {
      container.innerHTML = `<div class="graph-plot"></div>`;
      const el = container.querySelector(".graph-plot");

      if (!window.Plotly) {
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: "Plotly didn't load — reload the page.",
        });
        if (opts.onSubtitle) opts.onSubtitle("plotly missing");
        return;
      }

      const palette = ["#8B5FBF", "#D94F6B", "#3B82F6", "#10B981", "#F59E0B"];
      const traces = [];
      const annotations = [];

      (data.series || []).forEach((s, i) => {
        const color = palette[i % palette.length];
        traces.push({
          x: data.x,
          y: s.y,
          mode: "lines",
          type: "scatter",
          name: s.name,
          line: { color, width: 2 },
          connectgaps: false,
          hovertemplate: `${s.name}<br>${data.variable || "x"}=%{x:.4g}<br>y=%{y:.4g}<extra></extra>`,
        });

        // Max/min markers (skip if the series has no finite y values).
        const xs = data.x || [];
        const ys = s.y || [];
        let iMax = -1, iMin = -1;
        let yMax = -Infinity, yMin = Infinity;
        for (let k = 0; k < ys.length; k++) {
          const v = ys[k];
          if (v == null || !Number.isFinite(v)) continue;
          if (v > yMax) { yMax = v; iMax = k; }
          if (v < yMin) { yMin = v; iMin = k; }
        }
        if (iMax >= 0) {
          traces.push({
            x: [xs[iMax]], y: [yMax],
            mode: "markers",
            type: "scatter",
            marker: { color, size: 8, symbol: "circle",
                      line: { color: "#fff", width: 1.5 } },
            name: `${s.name} max`,
            hovertemplate: `max<br>${data.variable || "x"}=%{x:.4g}<br>y=%{y:.4g}<extra></extra>`,
            showlegend: false,
          });
          annotations.push({
            x: xs[iMax], y: yMax, xref: "x", yref: "y",
            text: `max ${yMax.toFixed(3)}`,
            showarrow: true, arrowhead: 0, arrowsize: 0.8, ax: 0, ay: -18,
            font: { size: 10, color },
            bgcolor: "rgba(255,255,255,0.85)", bordercolor: color, borderwidth: 1,
            borderpad: 2,
          });
        }
        if (iMin >= 0 && iMin !== iMax) {
          traces.push({
            x: [xs[iMin]], y: [yMin],
            mode: "markers",
            type: "scatter",
            marker: { color, size: 8, symbol: "circle-open",
                      line: { color, width: 2 } },
            name: `${s.name} min`,
            hovertemplate: `min<br>${data.variable || "x"}=%{x:.4g}<br>y=%{y:.4g}<extra></extra>`,
            showlegend: false,
          });
          annotations.push({
            x: xs[iMin], y: yMin, xref: "x", yref: "y",
            text: `min ${yMin.toFixed(3)}`,
            showarrow: true, arrowhead: 0, arrowsize: 0.8, ax: 0, ay: 18,
            font: { size: 10, color },
            bgcolor: "rgba(255,255,255,0.85)", bordercolor: color, borderwidth: 1,
            borderpad: 2,
          });
        }
      });

      const layout = {
        margin: { l: 42, r: 14, t: 8, b: 36 },
        // Spike lines give a crosshair-like cursor: horizontal + vertical
        // dotted guides follow the mouse, tied to the nearest data point.
        xaxis: {
          title: { text: data.variable || "x", font: { size: 11 } },
          zeroline: true, zerolinecolor: "rgba(75, 40, 115, 0.28)",
          gridcolor: "rgba(139, 95, 191, 0.15)",
          tickfont: { size: 10 },
          showspikes: true, spikemode: "across", spikedash: "dot",
          spikethickness: 1, spikecolor: "rgba(75, 40, 115, 0.55)",
        },
        yaxis: {
          zeroline: true, zerolinecolor: "rgba(75, 40, 115, 0.28)",
          gridcolor: "rgba(139, 95, 191, 0.15)",
          tickfont: { size: 10 },
          showspikes: true, spikemode: "across", spikedash: "dot",
          spikethickness: 1, spikecolor: "rgba(75, 40, 115, 0.55)",
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(255,255,255,0.5)",
        showlegend: data.series && data.series.length > 1,
        legend: { orientation: "h", y: -0.2, font: { size: 10 } },
        autosize: true,
        // Google-Maps-style interaction: drag to pan, wheel to zoom,
        // double-click to reset. The toolbar stays hidden — users get
        // all the navigation they need through direct gestures.
        dragmode: "pan",
        hovermode: "closest",
        annotations,
      };

      const config = {
        displayModeBar: false,
        responsive: true,
        scrollZoom: true,          // mouse-wheel zoom
        doubleClick: "reset+autosize",  // double-click restores default view
      };

      window.Plotly.newPlot(el, traces, layout, config);

      if (opts.onSubtitle) {
        const names = (data.series || []).map((s) => s.name).join(", ");
        opts.onSubtitle(
          `${names} · ${data.variable} ∈ [${data.x_min}, ${data.x_max}]`,
        );
      }
    }

    load();
    return {
      unmount: () => {
        aborted = true;
        try {
          if (window.Plotly) {
            const el = container.querySelector(".graph-plot");
            if (el) window.Plotly.purge(el);
          }
        } catch (_) { /* ignore */ }
        container.innerHTML = "";
      },
      reload: load,
    };
  }

  if (typeof window !== "undefined") {
    window.Graph = { mountGraphPanel };
  }
})();
