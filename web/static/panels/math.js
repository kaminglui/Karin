/* math.js — KaTeX-rendered math widget.
 *
 * Fetches /api/math?op=...&expression=... and renders the input and
 * result as pretty equations. Falls back to monospace LaTeX source if
 * KaTeX failed to load.
 */
(function () {
  "use strict";

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function renderLatex(el, latex, displayMode) {
    if (!latex) { el.textContent = ""; return; }
    if (window.katex) {
      try {
        window.katex.render(latex, el, {
          throwOnError: false,
          displayMode: !!displayMode,
          output: "html",
        });
        return;
      } catch (_) { /* fall through to plain text */ }
    }
    // KaTeX unavailable — show raw LaTeX in monospace.
    el.textContent = latex;
    el.classList.add("latex-fallback");
  }

  function render(container, data, opts) {
    container.innerHTML = `
      <div class="math-card">
        <div class="math-op">${escapeHtml(data.op || "")}</div>
        <div class="math-row math-input"></div>
        <div class="math-equals" aria-hidden="true">=</div>
        <div class="math-row math-result"></div>
        ${data.plain ? `<div class="math-plain">${escapeHtml(data.plain)}</div>` : ""}
      </div>
    `;
    const inputEl = container.querySelector(".math-input");
    const resultEl = container.querySelector(".math-result");
    renderLatex(inputEl, data.input_latex || "", true);
    renderLatex(resultEl, data.result_latex || "", true);
    if (opts.onSubtitle) opts.onSubtitle(data.op || "");
  }

  function mountMathPanel(container, options = {}) {
    const opts = {
      op: null, expression: null,
      variable: null, transformVar: null,
      lower: null, upper: null,
      onSubtitle: null,
      ...options,
    };
    let aborted = false;

    async function load() {
      container.innerHTML = Panels.stateCard({ kind: "loading" });
      if (opts.onSubtitle) opts.onSubtitle("");
      if (!opts.op || !opts.expression) {
        container.innerHTML = Panels.stateCard({
          kind: "empty",
          label: "no input",
          message: "Math called without op or expression.",
        });
        return;
      }
      try {
        const params = new URLSearchParams({
          op: opts.op, expression: opts.expression,
        });
        if (opts.variable) params.set("variable", opts.variable);
        if (opts.transformVar) params.set("transform_var", opts.transformVar);
        if (opts.lower != null) params.set("lower", opts.lower);
        if (opts.upper != null) params.set("upper", opts.upper);
        const r = await Panels.safeFetch(`/api/math?${params.toString()}`);
        const data = await r.json();
        if (aborted) return;
        if (data.error) {
          container.innerHTML = Panels.stateCard({
            kind: "error",
            message: data.error,
          });
          if (opts.onSubtitle) opts.onSubtitle("error");
          return;
        }
        render(container, data, opts);
      } catch (e) {
        if (aborted) return;
        container.innerHTML = Panels.stateCard({
          kind: "error",
          message: `Math failed: ${e.message}`,
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
    window.Math2 = { mountMathPanel };  // avoid clobbering built-in Math
  }
})();
