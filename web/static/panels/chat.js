/* chat.js — shared panel mounter.
 *
 * Since UI-6 (unified page), the only job of this module is to map a
 * tool_call event from the NDJSON stream to the appropriate panel mount
 * function and return a small controller the caller can keep around for
 * replacement / unmount.
 *
 * Consumed by:
 *   - web/static/app.js (main unified page at "/")
 *   - any future surface that wants to embed a panel
 *
 * Rules (locked since UI-5):
 *   1. ONE main panel at a time — the CALLER enforces this by calling
 *      prevController.unmount() before calling mountPanelForTool again.
 *      We don't hold global state here.
 *   2. Tool-call-driven only — get_time / get_weather return null (no
 *      panel surface) and the caller skips the mount.
 */

(function () {
  "use strict";

  const PANEL_MOUNTERS = {
    get_alerts: (container, args, callbacks) =>
      Alerts.mountAlertsPanel(container, {
        max_results: args.max_results || 10,
        onSubtitle: callbacks.onSubtitle,
      }),
    get_news: (container, args, callbacks) =>
      News.mountNewsPanel(container, {
        topic: args.topic ? String(args.topic) : null,
        max_results: 5,
        onSubtitle: callbacks.onSubtitle,
      }),
    // New merged tracker: empty id = grid, specific id = single.
    tracker: (container, args, callbacks) =>
      Trackers.mountTrackersPanel(container, {
        id: args.id ? String(args.id) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    get_weather: (container, args, callbacks) =>
      Weather.mountWeatherPanel(container, {
        location: args.location ? String(args.location) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    // New merged wiki: query = lookup, empty = random.
    wiki: (container, args, callbacks, result) =>
      Wiki.mountWikiPanel(container, {
        query: args.query ? String(args.query) : null,
        // Pass the backend's tool result so the widget locks to the
        // SAME article the LLM saw (avoids client-side re-roll on
        // random calls).
        resultText: result || null,
        onSubtitle: callbacks.onSubtitle,
      }),
    // ---- Legacy alias ------------------------------------------------
    // Only `wiki_search` appears in persisted conversation history
    // (audited 2026-04-15). Other retired names (get_tracker/_trackers,
    // wiki_random, today_in_history) had zero references and were
    // removed. Keep this mapping so old wiki_search widget replays
    // still mount correctly.
    wiki_search: (container, args, callbacks, result) =>
      Wiki.mountWikiPanel(container, {
        query: args.query ? String(args.query) : null,
        resultText: result || null,
        onSubtitle: callbacks.onSubtitle,
      }),
    web_search: (container, args, callbacks) =>
      WebSearch.mountWebSearchPanel(container, {
        query: args.query ? String(args.query) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    find_places: (container, args, callbacks) =>
      Places.mountPlacesPanel(container, {
        query: args.query ? String(args.query) : null,
        location: args.location ? String(args.location) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    math: (container, args, callbacks) =>
      Math2.mountMathPanel(container, {
        op: args.op ? String(args.op) : null,
        expression: args.expression ? String(args.expression) : null,
        variable: args.variable ? String(args.variable) : null,
        transformVar: args.transform_var ? String(args.transform_var) : null,
        lower: args.lower != null ? String(args.lower) : null,
        upper: args.upper != null ? String(args.upper) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    graph: (container, args, callbacks) =>
      Graph.mountGraphPanel(container, {
        expression: args.expression ? String(args.expression) : null,
        variable: args.variable ? String(args.variable) : "x",
        xMin: args.x_min != null ? Number(args.x_min) : -10,
        xMax: args.x_max != null ? Number(args.x_max) : 10,
        onSubtitle: callbacks.onSubtitle,
      }),
    circuit: (container, args, callbacks) =>
      Circuit.mountCircuitPanel(container, {
        args: args || {},
        onSubtitle: callbacks.onSubtitle,
      }),
    get_time: (container, args, callbacks) =>
      TimePanel.mountTimePanel(container, {
        tz: args.timezone ? String(args.timezone) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    convert: (container, args, callbacks) =>
      Convert.mountConvertPanel(container, {
        value: args.value != null ? args.value : null,
        fromUnit: args.from_unit ? String(args.from_unit) : null,
        toUnit: args.to_unit ? String(args.to_unit) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    inflation: (container, args, callbacks) =>
      Inflation.mountInflationPanel(container, {
        amount: args.amount != null ? Number(args.amount) : 1.0,
        from_year: args.from_year != null ? Number(args.from_year) : null,
        to_year: args.to_year != null ? Number(args.to_year) : null,
        measure: args.measure ? String(args.measure) : "cpi",
        item: args.item ? String(args.item) : null,
        region: args.region ? String(args.region) : "us",
        regions: args.regions ? String(args.regions) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    population: (container, args, callbacks) =>
      Population.mountPopulationPanel(container, {
        region: args.region ? String(args.region) : "world",
        year: args.year != null ? Number(args.year) : null,
        from_year: args.from_year != null ? Number(args.from_year) : null,
        to_year: args.to_year != null ? Number(args.to_year) : null,
        metric: args.metric ? String(args.metric) : "value",
        top: args.top != null ? Number(args.top) : 10,
        onSubtitle: callbacks.onSubtitle,
      }),
    facts: (container, args, callbacks) =>
      Facts.mountFactsPanel(container, {
        year: args.year != null ? Number(args.year) : null,
        region: args.region ? String(args.region) : null,
        onSubtitle: callbacks.onSubtitle,
      }),
    alice: (container, args, callbacks) =>
      Alice.mountAlicePanel(container, {
        year: args.year != null ? Number(args.year) : null,
        household_size: args.household_size != null ? Number(args.household_size) : 4,
        onSubtitle: callbacks.onSubtitle,
      }),
  };

  // Title formatter per tool. Takes the tool arguments so e.g. a news
  // call with topic="ukraine" gets "News · ukraine" instead of bare "News".
  const PANEL_TITLES = {
    get_alerts:   (_args) => "Alerts",
    get_news:     (args)  => args.topic ? `News · ${args.topic}` : "News",
    tracker:      (args)  => args.id ? `Tracker · ${args.id}` : "Trackers",
    get_weather:  (args)  => args.location ? `Weather · ${args.location}` : "Weather",
    wiki:         (args)  => args.query ? `Wikipedia · ${args.query}` : "Wikipedia · random",
    // Legacy title kept for restored conversations (see PANEL_MOUNTERS)
    wiki_search:  (args)  => args.query ? `Wikipedia · ${args.query}` : "Wikipedia",
    web_search:   (args)  => args.query ? `Web · ${args.query}` : "Web search",
    find_places:  (args)  => args.query ? `Places · ${args.query}` : "Places",
    math:         (args)  => args.op ? `Math · ${args.op}` : "Math",
    graph:        (args)  => args.expression ? `Graph · ${args.expression}` : "Graph",
    circuit:      (args)  => args.op ? `Circuit · ${args.op.replace(/_/g, " ")}` : "Circuit",
    get_time:     (args)  => args.timezone ? `Time · ${args.timezone}` : "Time",
    convert:      (args)  => (args.from_unit && args.to_unit)
      ? `Convert · ${args.from_unit} → ${args.to_unit}` : "Convert",
    inflation:    (args)  => {
      const fy = args.from_year, ty = args.to_year;
      const item = args.item;
      const tag = args.region && args.region !== "us"
        ? ` · ${args.region.toUpperCase().replace("_", " ")}` : "";
      if (item && fy) return `Inflation · ${item} in ${fy}${tag}`;
      const cur = args.region === "hk_sar" ? "HK$" : "$";
      const amt = args.amount != null ? `${cur}${args.amount}` : `${cur}1`;
      if (fy && ty) return `Inflation · ${amt} ${fy}→${ty}${tag}`;
      if (fy)       return `Inflation · ${amt} in ${fy}${tag}`;
      return "Inflation";
    },
    population:   (args)  => {
      if (args.metric === "rank") {
        const n = args.top || 10;
        return args.year
          ? `Population · top ${n} in ${args.year}`
          : `Population · top ${n}`;
      }
      const r = args.region ? args.region.replace("_", " ").toUpperCase() : "WORLD";
      if (args.from_year && args.to_year) return `Population · ${r} ${args.from_year}→${args.to_year}`;
      if (args.year)                      return `Population · ${r} ${args.year}`;
      return `Population · ${r}`;
    },
    facts:        (args)  => {
      const tag = args.region ? ` · ${args.region.replace("_", " ").toUpperCase()}` : "";
      return args.year ? `Facts · ${args.year}${tag}` : "Facts";
    },
    alice:        (args)  => {
      const size = args.household_size != null ? `${args.household_size}p` : "4p";
      return args.year ? `ALICE · ${args.year} · ${size}` : `ALICE · ${size}`;
    },
  };

  /**
   * Mount a panel for a given tool call into `container`.
   *
   * Returns:
   *   null if the tool has no panel surface (get_time, get_weather, unknown)
   *   { controller, title } otherwise, where controller is whatever the
   *   underlying mountXPanel function returned (refresh/unmount methods).
   *
   * callbacks:
   *   onSubtitle(text)  — forwarded to the panel's own onSubtitle so the
   *                       caller can reflect "3 alerts"/"error" in its UI.
   */
  function mountPanelForTool(toolName, args, container, callbacks = {}, result = null) {
    const mounter = PANEL_MOUNTERS[toolName];
    if (!mounter) return null;
    const titleFn = PANEL_TITLES[toolName] || (() => "Panel");
    const title = titleFn(args || {});
    // `result` is the backend's tool result string — panels that need
    // to match the assistant's context (e.g. wiki_random) use it to
    // avoid re-rolling/re-fetching. Most panels ignore it.
    const controller = mounter(container, args || {}, callbacks, result);
    return { controller, title, toolName };
  }

  window.Chat = { mountPanelForTool };
})();
