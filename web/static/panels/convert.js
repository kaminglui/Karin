/* convert.js — inline, interactive unit-conversion widget.
 *
 * Two modes, selected by the caller:
 *   - One-shot (original behavior): caller passes value + fromUnit +
 *     toUnit, widget shows the single result. Used from the chat panel
 *     right after the LLM calls the `convert` tool.
 *   - Interactive: renders <input> fields so the user can edit value
 *     and units directly and see live conversion without a new LLM
 *     turn. Triggered when `interactive: true` in mount options OR
 *     whenever any required arg is missing (so missing-args turns a
 *     blank error card into a usable calculator).
 *
 * In both modes the actual math happens server-side at /api/convert
 * (wraps bridge.tools.convert_data → pint) so the full unit registry
 * including temperature deltas is available. Keeps the widget dumb.
 */
(function () {
  "use strict";

  // Common unit categories for the dropdowns — curated from pint's
  // registry to cover the frequent conversions without overwhelming
  // the dropdown. Users can still type into the input (datalist).
  const UNIT_GROUPS = [
    {
      label: "length",
      units: ["meter", "km", "mile", "yard", "foot", "inch", "nautical_mile"],
    },
    {
      label: "mass",
      units: ["gram", "kg", "pound", "ounce", "ton", "stone"],
    },
    {
      label: "temperature",
      units: ["celsius", "fahrenheit", "kelvin"],
    },
    {
      label: "volume",
      units: ["liter", "milliliter", "gallon", "quart", "pint", "cup", "fluid_ounce"],
    },
    {
      label: "speed",
      units: ["mph", "kph", "meter/second", "knot"],
    },
    {
      label: "energy",
      units: ["joule", "kilojoule", "calorie", "kilocalorie", "kWh", "BTU"],
    },
    {
      label: "power",
      units: ["watt", "kilowatt", "horsepower"],
    },
    {
      label: "pressure",
      units: ["pascal", "kilopascal", "bar", "atmosphere", "psi"],
    },
    {
      label: "time",
      units: ["second", "minute", "hour", "day", "year"],
    },
  ];
  const ALL_UNITS = UNIT_GROUPS.flatMap((g) => g.units);

  function formatMagnitude(m) {
    if (typeof m !== "number" || !isFinite(m)) return String(m);
    const s = m.toPrecision(6);
    return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
  }

  // ---- one-shot (static) render ------------------------------------------

  function renderStatic(container, data) {
    if (data.error) {
      container.innerHTML = Panels.stateCard({
        kind: "error",
        label: "convert",
        message: data.error,
      });
      return;
    }
    const value = formatMagnitude(data.value);
    const mag = formatMagnitude(data.magnitude);
    container.innerHTML = `
      <div class="convert-card">
        <div class="convert-line">
          <span class="convert-from">${Panels.escapeHtml(value)} ${Panels.escapeHtml(data.from_unit)}</span>
          <span class="convert-eq">=</span>
          <span class="convert-to">${Panels.escapeHtml(mag)} ${Panels.escapeHtml(data.to_unit)}</span>
        </div>
      </div>
    `;
  }

  // ---- interactive render ------------------------------------------------

  function optionsHtml(selectedUnit) {
    const esc = Panels.escapeHtml;
    return UNIT_GROUPS.map((g) => {
      const opts = g.units.map((u) => {
        const selAttr = u === selectedUnit ? " selected" : "";
        return `<option value="${esc(u)}"${selAttr}>${esc(u)}</option>`;
      }).join("");
      return `<optgroup label="${esc(g.label)}">${opts}</optgroup>`;
    }).join("");
  }

  /** Produce the interactive widget DOM, with live event handlers
   *  wired to the server on each input change. Stores the last good
   *  result on the widget element so a transient error (network blip,
   *  invalid unit combo) doesn't erase what the user last saw. */
  function mountInteractive(container, opts) {
    const esc = Panels.escapeHtml;
    // Defaults — seeded from mount options when present, else a
    // benign pair (1 meter → foot).
    const initialValue = opts.value != null ? String(opts.value) : "1";
    const initialFrom = opts.fromUnit || "meter";
    const initialTo = opts.toUnit || "foot";

    container.innerHTML = `
      <div class="convert-card convert-interactive">
        <div class="convert-controls">
          <input type="number" step="any" inputmode="decimal"
                 class="convert-value" value="${esc(initialValue)}"
                 aria-label="Value" />
          <select class="convert-from-unit" aria-label="From unit">
            ${optionsHtml(initialFrom)}
          </select>
          <button type="button" class="convert-reverse"
                  aria-label="Swap units">⇄</button>
          <select class="convert-to-unit" aria-label="To unit">
            ${optionsHtml(initialTo)}
          </select>
        </div>
        <div class="convert-result-line">
          <span class="convert-result">…</span>
        </div>
        <div class="convert-error muted" hidden></div>
      </div>
    `;

    const valueInput = container.querySelector(".convert-value");
    const fromSel   = container.querySelector(".convert-from-unit");
    const toSel     = container.querySelector(".convert-to-unit");
    const reverseBtn= container.querySelector(".convert-reverse");
    const resultEl  = container.querySelector(".convert-result");
    const errorEl   = container.querySelector(".convert-error");

    // Debounce rapid input changes — typing "1234" shouldn't fire four
    // fetches. 150 ms is below conscious-perception threshold for
    // interactive UIs.
    let pending = null;
    async function recompute() {
      if (pending) clearTimeout(pending);
      pending = setTimeout(doFetch, 150);
    }
    async function doFetch() {
      const val = valueInput.value;
      const fromU = fromSel.value;
      const toU = toSel.value;
      if (!val || !fromU || !toU) return;
      try {
        const qs = new URLSearchParams({
          value: String(val), from_unit: fromU, to_unit: toU,
        });
        const res = await Panels.safeFetch(`/api/convert?${qs.toString()}`);
        const data = await res.json();
        if (data.error) {
          errorEl.hidden = false;
          errorEl.textContent = data.error;
          return;
        }
        errorEl.hidden = true;
        const mag = formatMagnitude(data.magnitude);
        resultEl.textContent = `${mag} ${data.to_unit}`;
        if (opts.onSubtitle) opts.onSubtitle(`${fromU} → ${toU}`);
      } catch (e) {
        errorEl.hidden = false;
        errorEl.textContent = `convert failed: ${e.message || e}`;
      }
    }

    valueInput.addEventListener("input", recompute);
    fromSel.addEventListener("change", recompute);
    toSel.addEventListener("change", recompute);
    reverseBtn.addEventListener("click", () => {
      // Swap the two <select>s AND keep the current value as-is.
      const a = fromSel.value;
      fromSel.value = toSel.value;
      toSel.value = a;
      recompute();
    });

    // Kick the first compute so the result line isn't stuck on "…".
    doFetch();
  }

  // ---- mount entry point -------------------------------------------------

  function mountConvertPanel(container, options = {}) {
    const opts = {
      value: null,
      fromUnit: null,
      toUnit: null,
      onSubtitle: null,
      interactive: false,
      ...options,
    };

    // When the caller didn't pass full args, drop into interactive mode
    // rather than showing a scary "convert needs value..." error card.
    // That way the widget is useful whether the LLM called convert
    // cleanly or not — a missing to_unit turns into a user-editable UI
    // instead of a dead end.
    const missingArgs = (
      opts.value == null || !opts.fromUnit || !opts.toUnit
    );
    const shouldInteract = opts.interactive || missingArgs;

    if (shouldInteract) {
      mountInteractive(container, opts);
      // Return a minimal controller matching the Panels.mountPanel shape.
      return {
        refresh: () => mountInteractive(container, opts),
        unmount: () => { container.innerHTML = ""; },
      };
    }

    // One-shot path (legacy, used from chat panel for tool results).
    return Panels.mountPanel({
      container,
      fetch: async () => {
        const qs = new URLSearchParams({
          value: String(opts.value),
          from_unit: String(opts.fromUnit),
          to_unit: String(opts.toUnit),
        });
        const res = await Panels.safeFetch(`/api/convert?${qs.toString()}`);
        return res.json();
      },
      render: renderStatic,
      subtitleFor: (d) => {
        if (d.error) return "error";
        return `${d.from_unit} → ${d.to_unit}`;
      },
      onSubtitle: opts.onSubtitle,
    });
  }

  if (typeof window !== "undefined") {
    window.Convert = { mountConvertPanel };
  }
})();
