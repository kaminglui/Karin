/* map.js — US choropleth rendering for /ui/map.
 *
 * Phase 1 (state-level rent + curated region overlays + state drill):
 * - Choropleth fill from /api/map driven by --tint-rgb (theme-aware)
 * - Optional region overlay (Rust Belt / Sun Belt / Appalachia / etc.)
 *   applies a colored stroke to member states + dims non-members
 * - Click any state → fetch /api/map/state/<code> and render the
 *   curated profile card (industries, resources, region memberships,
 *   editorial narrative) below the map
 * - Click also emits `karin:focus-region` postMessage for future
 *   cross-panel navigation
 *
 * The us-atlas TopoJSON used (`states-albers-10m.json`) is
 * pre-projected to fit a 975×610 viewBox in Albers USA, so
 * `d3.geoPath()` with no projection arg renders directly.
 */
(function () {
  "use strict";

  const TOPO_URL = "/static/vendor/us-states-10m.json";
  const API_URL = "/api/map";
  const REGIONS_URL = "/api/map/regions";
  const STATE_URL = "/api/map/state/";

  let cachedTopo = null;
  let cachedRegions = null;
  let cachedZoom = null;  // d3.zoom behavior, initialized once

  // SVG viewBox dimensions — the us-atlas albers TopoJSON projects to
  // exactly this rectangle, so we use the same numbers when computing
  // click-to-zoom-bounds transforms.
  const VB_WIDTH = 975;
  const VB_HEIGHT = 610;
  const ZOOM_MIN = 1;
  const ZOOM_MAX = 12;

  async function loadTopo() {
    if (cachedTopo) return cachedTopo;
    const res = await fetch(TOPO_URL, { cache: "force-cache" });
    if (!res.ok) throw new Error(`topojson HTTP ${res.status}`);
    cachedTopo = await res.json();
    return cachedTopo;
  }

  async function loadRegions() {
    if (cachedRegions) return cachedRegions;
    const res = await fetch(REGIONS_URL);
    if (!res.ok) throw new Error(`regions HTTP ${res.status}`);
    const data = await res.json();
    cachedRegions = data.regions || [];
    return cachedRegions;
  }

  function fmtMoney(n) {
    return "$" + Math.round(n).toLocaleString("en-US");
  }

  function buildColorScale(values, tintRgb) {
    const stops = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9];
    return d3.scaleQuantile(values, stops.map((s) =>
      `rgba(${tintRgb}, ${0.18 + s * 0.7})`
    ));
  }

  function getTintRgb() {
    const root = document.documentElement;
    const v = getComputedStyle(root).getPropertyValue("--tint-rgb").trim();
    return v || "139, 95, 191";
  }

  function renderStats(stats, statsEl) {
    statsEl.innerHTML = `
      <div class="map-stat-cell">
        <div class="map-stat-label">Median</div>
        <div class="map-stat-value">${fmtMoney(stats.median)}/mo</div>
      </div>
      <div class="map-stat-cell">
        <div class="map-stat-label">Mean</div>
        <div class="map-stat-value">${fmtMoney(stats.mean)}/mo</div>
      </div>
      <div class="map-stat-cell">
        <div class="map-stat-label">Lowest</div>
        <div class="map-stat-value">${fmtMoney(stats.min)}/mo</div>
      </div>
      <div class="map-stat-cell">
        <div class="map-stat-label">Highest</div>
        <div class="map-stat-value">${fmtMoney(stats.max)}/mo</div>
      </div>
    `;
  }

  function renderLegend(stats, tintRgb) {
    const bar = document.getElementById("map-legend-bar");
    bar.style.background = `linear-gradient(90deg,
      rgba(${tintRgb}, 0.18) 0%,
      rgba(${tintRgb}, 0.88) 100%)`;
    document.getElementById("map-legend-min").textContent = fmtMoney(stats.min);
    document.getElementById("map-legend-max").textContent = fmtMoney(stats.max);
  }

  function renderRegionInfo(region) {
    const el = document.getElementById("map-region-info");
    if (!region) {
      el.hidden = true;
      el.innerHTML = "";
      return;
    }
    el.hidden = false;
    el.style.borderLeftColor = region.color;
    el.innerHTML = `
      <div class="map-region-info-label" style="color: ${escapeHtml(region.color)}">
        ${escapeHtml(region.label)} <span style="color: var(--text-muted); font-weight: 500; font-size: 0.85em;">— ${region.states.length} states</span>
      </div>
      <div class="map-region-info-desc">${escapeHtml(region.description)}</div>
    `;
  }

  function showTooltip(tip, evt, name, value) {
    tip.innerHTML = `
      <span class="map-tooltip-name">${escapeHtml(name)}</span>
      <span class="map-tooltip-value">${fmtMoney(value)}/mo</span>
    `;
    tip.classList.add("visible");
    const container = tip.parentElement;
    const rect = container.getBoundingClientRect();
    const x = evt.clientX - rect.left + 10;
    const y = evt.clientY - rect.top - 28;
    tip.style.left = `${Math.max(4, Math.min(rect.width - 200, x))}px`;
    tip.style.top = `${Math.max(4, y)}px`;
  }

  function hideTooltip(tip) {
    tip.classList.remove("visible");
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  /** Initialize d3.zoom on the SVG once per page. The transform is
   * applied to <g.map-zoom-target> so the SVG element itself stays
   * fixed and only the path coordinates get scaled/translated.
   * Subsequent calls return the same instance. */
  function ensureZoom(svg, target) {
    if (cachedZoom) return cachedZoom;
    cachedZoom = d3.zoom()
      .scaleExtent([ZOOM_MIN, ZOOM_MAX])
      .on("zoom", (event) => {
        target.attr("transform", event.transform);
      });
    svg.call(cachedZoom);
    // Wire the +/-/reset buttons. They use d3.zoom's transition-aware
    // helpers so the camera animates smoothly instead of snapping.
    const zoomBy = (factor) => {
      svg.transition().duration(280)
        .call(cachedZoom.scaleBy, factor);
    };
    const zoomReset = () => {
      svg.transition().duration(420)
        .call(cachedZoom.transform, d3.zoomIdentity);
    };
    document.getElementById("map-zoom-in")
      ?.addEventListener("click", () => zoomBy(1.5));
    document.getElementById("map-zoom-out")
      ?.addEventListener("click", () => zoomBy(1 / 1.5));
    document.getElementById("map-zoom-reset")
      ?.addEventListener("click", zoomReset);
    return cachedZoom;
  }

  /** Smoothly zoom + pan to fit a single feature inside the viewBox.
   * Used by the click handler so picking a state both selects it and
   * fills the visible area with that state's geometry. */
  function zoomToFeature(svg, path, feature) {
    if (!cachedZoom) return;
    const [[x0, y0], [x1, y1]] = path.bounds(feature);
    const dx = x1 - x0;
    const dy = y1 - y0;
    if (!dx || !dy) return;
    // 0.85 leaves a small margin around the focused state.
    const scale = Math.min(
      ZOOM_MAX,
      0.85 / Math.max(dx / VB_WIDTH, dy / VB_HEIGHT),
    );
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const tx = VB_WIDTH / 2 - scale * cx;
    const ty = VB_HEIGHT / 2 - scale * cy;
    svg.transition().duration(700).call(
      cachedZoom.transform,
      d3.zoomIdentity.translate(tx, ty).scale(scale),
    );
  }

  /** FIPS code → 2-letter state code map. We need this when a click
   * yields a feature with id like "06" (CA) — the profile API expects
   * "CA". Reuses the values_by_fips response key map for consistency. */
  function fipsToCode(fips, payload) {
    // The /api/map response includes both values_by_state (keyed by
    // 2-letter code) and values_by_fips. Reverse-engineer by walking
    // values_by_state and matching against values_by_fips on value.
    // Cheap because we only do it on click, not every render.
    const v = payload.values_by_fips[fips];
    if (v == null) return null;
    for (const code in payload.values_by_state) {
      if (payload.values_by_state[code] === v) return code;
    }
    return null;
  }

  async function renderStateProfile(code, payload) {
    const el = document.getElementById("map-state-profile");
    if (!code) {
      el.hidden = true;
      el.innerHTML = "";
      el.removeAttribute("data-state-code");
      return;
    }
    el.hidden = false;
    el.setAttribute("data-state-code", code);
    el.innerHTML = `<div style="grid-column: 1/-1; color: var(--text-muted)">Loading ${escapeHtml(code)}...</div>`;
    try {
      const r = await Panels.safeFetch(`${STATE_URL}${encodeURIComponent(code)}`);
      const profile = await r.json();
      if (profile.error) {
        el.innerHTML = `<div style="grid-column: 1/-1; color: var(--sev-critical)">${escapeHtml(profile.error)}</div>`;
        return;
      }
      const rentLine = profile.rent_context
        ? `<span class="map-profile-rent">${fmtMoney(profile.rent_context.median_2br_rent)}/mo (${profile.rent_context.year} 2br)</span>`
        : "";
      // Region chips are clickable buttons — clicking one activates
      // that region's overlay on the map (sets the dropdown value +
      // dispatches a change event so the existing render path picks
      // up the new overlay choice).
      const activeKey = (window.KMap && window.KMap._activeRegion) || "";
      const regionChips = (profile.region_memberships || []).map((r) => {
        const isActive = r.key === activeKey;
        return `
        <button type="button"
                class="map-region-chip map-region-chip-clickable${isActive ? " is-active" : ""}"
                data-region-key="${escapeHtml(r.key)}"
                aria-pressed="${isActive ? "true" : "false"}"
                title="${isActive ? "Click again to clear overlay" : "Click to highlight " + escapeHtml(r.label) + " on the map"}">
          <span class="map-region-chip-dot" style="background: ${escapeHtml(r.color || '#888')}"></span>
          ${escapeHtml(r.label)}
        </button>
      `;
      }).join("");
      const industriesList = (profile.top_industries || [])
        .map((i) => `<li>${escapeHtml(i)}</li>`).join("");
      const resourcesList = (profile.resources || [])
        .map((i) => `<li>${escapeHtml(i)}</li>`).join("");
      el.innerHTML = `
        <h2>${escapeHtml(profile.name)} (${escapeHtml(profile.code)}) ${rentLine}</h2>
        ${regionChips ? `<div class="map-state-profile-regions">${regionChips}</div>` : ""}
        <div class="map-state-profile-section">
          <h3>Top industries</h3>
          <ul>${industriesList}</ul>
        </div>
        <div class="map-state-profile-section">
          <h3>Resources + infrastructure</h3>
          <ul>${resourcesList}</ul>
        </div>
        ${profile.narrative ? `
          <div class="map-state-profile-narrative">
            ${escapeHtml(profile.narrative)}
          </div>
        ` : ""}
      `;

      // Wire up the chip click handlers — clicking a chip drives the
      // region-overlay dropdown, which the existing change-listener
      // already routes to a re-render. Clicking the active chip
      // toggles back to "no overlay".
      el.querySelectorAll(".map-region-chip-clickable").forEach((btn) => {
        btn.addEventListener("click", () => {
          const key = btn.getAttribute("data-region-key");
          const sel = document.getElementById("map-region-select");
          if (!sel) return;
          // Toggle: clicking the active chip clears the overlay.
          const wasActive = btn.classList.contains("is-active");
          sel.value = wasActive ? "" : key;
          sel.dispatchEvent(new Event("change", { bubbles: true }));
        });
      });
    } catch (e) {
      el.innerHTML = `<div style="grid-column: 1/-1; color: var(--sev-critical)">Error loading profile: ${escapeHtml(e.message || e)}</div>`;
    }
  }

  async function mountMap(opts) {
    opts = opts || {};
    const svg = d3.select("#map-svg");
    const tip = document.getElementById("map-tooltip");
    const statsEl = document.getElementById("map-stats");
    const interpEl = document.getElementById("map-interp");

    const params = new URLSearchParams();
    if (opts.year) params.set("year", String(opts.year));
    if (opts.metric) params.set("metric", String(opts.metric));

    let payload;
    try {
      const r = await Panels.safeFetch(`${API_URL}?${params.toString()}`);
      payload = await r.json();
    } catch (e) {
      svg.html("");
      interpEl.textContent = `Error: ${e.message || e}`;
      return;
    }
    if (payload.error) {
      svg.html("");
      interpEl.textContent = `Error: ${payload.error}`;
      return;
    }

    if (opts.onYearsLoaded && Array.isArray(payload.available_years)) {
      opts.onYearsLoaded(payload.available_years, payload.year);
    }
    if (opts.onSubtitle) {
      opts.onSubtitle(`${payload.metric_label} · ${payload.year}`);
    }

    let topo, states, regions;
    try {
      topo = await loadTopo();
      regions = await loadRegions();
      if (typeof topojson === "undefined") {
        throw new Error("topojson global not loaded");
      }
      if (!topo.objects || !topo.objects.states) {
        throw new Error("topojson missing objects.states");
      }
      states = topojson.feature(topo, topo.objects.states);
      if (!states || !Array.isArray(states.features) || states.features.length === 0) {
        throw new Error("topojson.feature() returned no features");
      }
    } catch (e) {
      svg.html("");
      interpEl.textContent = `Map render error: ${e.message || e}`;
      interpEl.style.color = "var(--sev-critical)";
      return;
    }

    if (opts.onRegionsLoaded) {
      opts.onRegionsLoaded(regions);
    }

    // Pick the region overlay (if any) + the matching FIPS set so
    // the path render can decide member vs non-member at draw time.
    const activeRegion = opts.region_overlay
      ? regions.find((r) => r.key === opts.region_overlay) || null
      : null;
    // Stash the active region key so renderStateProfile can mark the
    // matching chip as active without a parameter chain.
    window.KMap._activeRegion = activeRegion ? activeRegion.key : "";
    renderRegionInfo(activeRegion);

    // If a state profile is currently showing, re-render it so its
    // chips reflect the new overlay state (active vs inactive).
    const profileEl = document.getElementById("map-state-profile");
    if (profileEl && !profileEl.hidden) {
      const currentCode = profileEl.getAttribute("data-state-code");
      if (currentCode && payload) {
        renderStateProfile(currentCode, payload);
      }
    }

    // Build a code→FIPS reverse so we can resolve state codes (which
    // is what regions.json uses) to FIPS (which us-atlas uses).
    // The /api/map response has values_by_fips (FIPS keys) and
    // values_by_state (code keys) sharing the same values; build the
    // mapping from those.
    const fipsForCode = {};
    for (const fips in payload.values_by_fips) {
      const v = payload.values_by_fips[fips];
      for (const code in payload.values_by_state) {
        if (payload.values_by_state[code] === v) {
          fipsForCode[code] = fips;
          break;
        }
      }
    }
    const memberFips = activeRegion
      ? new Set(activeRegion.states.map((c) => fipsForCode[c]).filter(Boolean))
      : null;

    const tintRgb = getTintRgb();
    const values = Object.values(payload.values_by_fips);
    const color = buildColorScale(values, tintRgb);

    renderStats(payload.stats, statsEl);
    renderLegend(payload.stats, tintRgb);
    interpEl.textContent = payload.interpretation || "";
    interpEl.style.color = "";  // clear any prior error styling

    const path = d3.geoPath();

    // All paths render inside <g.map-zoom-target> so a single matrix
    // transform handles pan/zoom for the whole map.
    let zoomTarget = svg.select("g.map-zoom-target");
    if (zoomTarget.empty()) {
      zoomTarget = svg.append("g").attr("class", "map-zoom-target");
    }
    ensureZoom(svg, zoomTarget);

    const sel = zoomTarget.selectAll("path.map-state").data(states.features, (d) => d.id);
    sel.exit().remove();
    const enter = sel.enter().append("path")
      .attr("class", "map-state")
      .attr("d", path);

    enter.merge(sel)
      .attr("d", path)
      .attr("class", (d) => {
        const fips = String(d.id).padStart(2, "0");
        if (!memberFips) return "map-state";
        if (memberFips.has(fips)) return "map-state in-region";
        return "map-state out-region";
      })
      .attr("fill", (d) => {
        const fips = String(d.id).padStart(2, "0");
        const v = payload.values_by_fips[fips];
        if (v == null) return "var(--card-bg)";
        return color(v);
      })
      .attr("stroke", (d) => {
        const fips = String(d.id).padStart(2, "0");
        if (memberFips && memberFips.has(fips)) {
          return activeRegion.color;
        }
        return null;  // fall back to CSS
      })
      .style("color", (d) => {
        // Setting `color` on the path makes `currentColor` in the
        // CSS drop-shadow filter resolve to the region's color, so
        // each highlighted state glows in its region's color.
        const fips = String(d.id).padStart(2, "0");
        if (memberFips && memberFips.has(fips)) {
          return activeRegion.color;
        }
        return null;
      })
      .on("mousemove", (evt, d) => {
        const fips = String(d.id).padStart(2, "0");
        const v = payload.values_by_fips[fips];
        if (v == null) return;
        const name = (d.properties && d.properties.name) || `FIPS ${fips}`;
        showTooltip(tip, evt, name, v);
      })
      .on("mouseleave", () => hideTooltip(tip))
      .on("click", function (evt, d) {
        svg.selectAll(".map-state.is-selected").classed("is-selected", false);
        d3.select(this).classed("is-selected", true);
        const fips = String(d.id).padStart(2, "0");
        const value = payload.values_by_fips[fips];
        const name = (d.properties && d.properties.name) || `FIPS ${fips}`;
        const code = fipsToCode(fips, payload);
        const detail = {
          fips, state_name: name, code,
          metric: payload.metric, value, year: payload.year,
        };
        try {
          if (window.parent && window.parent !== window) {
            window.parent.postMessage(
              { type: "karin:focus-region", detail },
              "*",
            );
          }
        } catch (_) { /* same-origin guard non-critical */ }
        window.dispatchEvent(new CustomEvent("karin:focus-region", { detail }));
        // Smoothly zoom to fit the clicked state. Preserves the
        // existing select/profile behavior; just adds the camera move.
        zoomToFeature(svg, path, d);
        // Fetch + render the profile card.
        if (code) {
          renderStateProfile(code, payload);
        }
      });
  }

  window.KMap = { mountMap, _activeRegion: "" };
})();
