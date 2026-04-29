/* circuit.js — analog / digital circuit calc widget.
 *
 * Shows the op, the input values, and the result. Truth tables get a
 * proper HTML table rendering.
 */
(function () {
  "use strict";

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function formatOp(op) {
    if (!op) return "";
    return op.replace(/_/g, " ");
  }

  function renderInputsTable(inputs) {
    const entries = Object.entries(inputs || {});
    if (!entries.length) return "";
    const rows = entries.map(([k, v]) =>
      `<tr><td class="ck-key">${escapeHtml(k)}</td><td class="ck-val">${escapeHtml(
        Array.isArray(v) ? v.join(", ") : String(v)
      )}</td></tr>`
    ).join("");
    return `<table class="circuit-inputs"><tbody>${rows}</tbody></table>`;
  }

  function renderTruthTable(inputs, result) {
    const vars = inputs.variables || [];
    const rows = result.rows || [];
    const header = vars.map((v) => `<th>${escapeHtml(v)}</th>`).join("") +
      "<th>out</th>";
    const body = rows.map((r) => {
      const cells = vars.map((v) => `<td>${r[v]}</td>`).join("");
      return `<tr><td class="ttrow-inputs-hidden" hidden></td>${cells}<td class="ck-out">${r.out}</td></tr>`;
    }).join("");
    return `
      <div class="circuit-expr">${escapeHtml(inputs.expression || "")}</div>
      <table class="truth-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>
    `;
  }

  function render(container, data, opts) {
    if (data.op === "truth_table") {
      container.innerHTML = `
        <div class="circuit-card">
          <div class="circuit-op">${escapeHtml(formatOp(data.op))}</div>
          ${renderTruthTable(data.inputs || {}, data.result || {})}
        </div>
      `;
      if (opts.onSubtitle) {
        opts.onSubtitle(`${(data.result.rows || []).length} rows`);
      }
      return;
    }

    const result = data.result || {};
    const formatted = result.formatted || "";
    container.innerHTML = `
      <div class="circuit-card">
        <div class="circuit-op">${escapeHtml(formatOp(data.op))}</div>
        ${renderInputsTable(data.inputs || {})}
        ${formatted ? `<div class="circuit-result">${escapeHtml(formatted)}</div>` : ""}
        <div class="circuit-plain">${escapeHtml(data.plain || "")}</div>
      </div>
    `;
    if (opts.onSubtitle) opts.onSubtitle(formatted || data.op || "");
  }

  function mountCircuitPanel(container, options = {}) {
    const opts = { args: {}, onSubtitle: null, ...options };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      if (!opts.args || !opts.args.op) {
        container.innerHTML = Panels.stateCard({
          kind: "empty", label: "no op",
          message: "Circuit called without an op.",
        });
        return;
      }
      try {
        const params = new URLSearchParams();
        for (const [k, v] of Object.entries(opts.args)) {
          if (v != null && v !== "") params.set(k, String(v));
        }
        const r = await Panels.safeFetch(`/api/circuit?${params.toString()}`);
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
          kind: "error", message: `Circuit failed: ${e.message}`,
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
    window.Circuit = { mountCircuitPanel };
  }
})();
