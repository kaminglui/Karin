/* alice.js — ALICE estimator panel.
 *
 * Renders the survival-budget breakdown returned by /api/alice. The
 * v2 model takes a household composition (1A0K, 2A0K, 1A1K, 2A1K,
 * 2A2K, 2A3K) and returns:
 *   - per-composition line items (housing-bdr scaled, food per
 *     adult-equivalent, healthcare coverage tier matched to the
 *     household, etc.)
 *   - bracket-based federal tax with CTC + EITC offsets per filing
 *     status, plus FICA + state-avg
 *   - a 4-person canonical budget for cross-validation against UFA
 *     even when the user picked a different composition
 *
 * Backward-compat: the panel still accepts `household_size` from
 * chat.js's PANEL_MOUNTERS — alice.js translates it to a composition
 * server-side.
 */
(function () {
  "use strict";

  const ENDPOINT = "/api/alice";

  function fmtMoney(n) {
    if (n == null || isNaN(n)) return "—";
    return Math.round(n).toLocaleString("en-US");
  }

  function fmtPct(p) {
    if (p == null || isNaN(p)) return "—";
    return (p * 100).toFixed(1) + "%";
  }

  /** Budget grid: label + dollar amount + one-sentence explanation.
   *  Uses display:contents <li> children inside a 3-col grid (same
   *  trick as the facts items list) so all rows align across columns. */
  function renderBudgetTable(sb) {
    const rows = [
      ["Housing", sb.housing],
      ["Food", sb.food],
      ["Healthcare", sb.healthcare],
      ["Childcare", sb.childcare],
      ["Transport", sb.transport],
      ["Technology", sb.technology],
    ];
    const lineHtml = rows.map(([label, item]) => `
      <li>
        <span class="alice-line-label">${Panels.escapeHtml(label)}</span>
        <span class="alice-line-value">$${fmtMoney(item.annual)}</span>
        <span class="alice-line-hint">${Panels.escapeHtml(item.explanation || "")}</span>
      </li>
    `).join("");
    return `
      <ul class="alice-budget-list">
        ${lineHtml}
        <li class="alice-budget-subtotal">
          <span class="alice-line-label">Subtotal (pre-tax)</span>
          <span class="alice-line-value">$${fmtMoney(sb.subtotal_pretax)}</span>
          <span class="alice-line-hint">${Panels.escapeHtml(sb.subtotal_explanation || "")}</span>
        </li>
      </ul>
    `;
  }

  /** Tax breakdown — federal brackets, CTC, EITC, FICA, state, all
   *  shown per line so the user can audit the numbers. */
  function renderTaxBreakdown(taxes) {
    const rows = [
      ["Federal income tax (pre-credits)", taxes.federal_pre_credits, ""],
      ["Child Tax Credit", -taxes.ctc, taxes.ctc > 0
        ? `${taxes.num_kids} child(ren) × $2,000` : "no qualifying children"],
      ["Earned Income Tax Credit", -taxes.eitc, taxes.eitc > 0
        ? "phase-out range" : "income outside EITC range"],
      ["Federal income tax (after credits)", taxes.federal_after_credits,
        "may go negative when EITC + CTC exceed pre-credit tax"],
      ["Payroll tax (FICA)", taxes.payroll_fica,
        "7.65% Social Security + Medicare (employee share)"],
      ["State income tax", taxes.state_avg,
        "national-average effective rate (~5%)"],
    ];
    const lineHtml = rows.map(([label, val, hint]) => {
      const valStr = val < 0
        ? `−$${fmtMoney(-val)}`
        : `$${fmtMoney(val)}`;
      const cls = val < 0 ? "alice-line-value alice-line-value-credit" : "alice-line-value";
      return `
        <li>
          <span class="alice-line-label">${Panels.escapeHtml(label)}</span>
          <span class="${cls}">${valStr}</span>
          <span class="alice-line-hint">${Panels.escapeHtml(hint)}</span>
        </li>
      `;
    }).join("");
    return `
      <ul class="alice-budget-list">
        ${lineHtml}
        <li class="alice-budget-total">
          <span class="alice-line-label">Total taxes (net)</span>
          <span class="alice-line-value">$${fmtMoney(taxes.net_total)}</span>
          <span class="alice-line-hint">
            ${Panels.escapeHtml(`Effective rate ${taxes.effective_rate_pct}% on gross. Filing as ${taxes.filing_status}.`)}
          </span>
        </li>
      </ul>
    `;
  }

  /** Final-total callout — the gross income required after the
   *  budget + taxes are summed. */
  function renderFinalTotal(sb, compDetails, year) {
    return `
      <div class="alice-scaled-total">
        <div class="alice-scaled-row">
          <span class="alice-scaled-label">
            ALICE threshold for ${Panels.escapeHtml(compDetails.label)} (${year})
          </span>
          <span class="alice-scaled-value">$${fmtMoney(sb.total_for_size)}</span>
        </div>
        <div class="alice-scaled-hint">
          ${Panels.escapeHtml(sb.total_explanation || "")}
        </div>
      </div>
    `;
  }

  function renderComparison(out) {
    const ref = out.reference_united_for_alice || {};
    const ours = out.alice_threshold_4person_us;
    const theirs = ref.alice_threshold_4person_us;
    const ps = out.population_shares || {};
    const deltaThreshold = ref.delta_threshold_4person;
    const deltaPctClass =
      Math.abs(ps.pct_alice - (ref.pct_alice || 0)) < 0.05
        ? "alice-delta-ok" : "alice-delta-warn";
    return `
      <div class="alice-compare">
        <div class="alice-compare-row">
          <div class="alice-compare-cell">
            <div class="alice-compare-label">OUR 4-PERSON BASELINE</div>
            <div class="alice-compare-big">$${fmtMoney(ours)}</div>
            <div class="alice-compare-sub">2A2K canonical (our methodology)</div>
          </div>
          <div class="alice-compare-cell">
            <div class="alice-compare-label">UNITED FOR ALICE</div>
            <div class="alice-compare-big">$${fmtMoney(theirs)}</div>
            <div class="alice-compare-sub">${Panels.escapeHtml(ref.source || "")}</div>
          </div>
          <div class="alice-compare-cell ${deltaPctClass}">
            <div class="alice-compare-label">DELTA</div>
            <div class="alice-compare-big">${deltaThreshold >= 0 ? "+" : ""}$${fmtMoney(deltaThreshold)}</div>
            <div class="alice-compare-sub">${theirs ? ((deltaThreshold/theirs)*100).toFixed(1) : "—"}% — mostly transport + healthcare</div>
          </div>
        </div>

        <div class="alice-compare-row">
          <div class="alice-compare-cell">
            <div class="alice-compare-label">% POVERTY</div>
            <div class="alice-compare-big">${fmtPct(ps.pct_poverty)}</div>
            <div class="alice-compare-sub">vs published ${fmtPct(ref.pct_poverty)} — earning below FPL</div>
          </div>
          <div class="alice-compare-cell">
            <div class="alice-compare-label">% ALICE</div>
            <div class="alice-compare-big">${fmtPct(ps.pct_alice)}</div>
            <div class="alice-compare-sub">vs published ${fmtPct(ref.pct_alice)} — above FPL, below survival</div>
          </div>
          <div class="alice-compare-cell">
            <div class="alice-compare-label">BELOW SURVIVAL</div>
            <div class="alice-compare-big">${fmtPct((ps.pct_poverty || 0) + (ps.pct_alice || 0))}</div>
            <div class="alice-compare-sub">vs published ${fmtPct((ref.pct_poverty || 0) + (ref.pct_alice || 0))} — total stressed</div>
          </div>
        </div>
      </div>
    `;
  }

  function renderCrossCheck(cc) {
    if (!cc) return "";
    const food = cc.bls_food_basket;
    const gas = cc.eia_gasoline;
    if (!food && !gas) return "";
    const lines = [];
    if (food) {
      lines.push(`
        <div class="alice-check-row">
          <strong>BLS food basket cross-check:</strong>
          $${fmtMoney(food.subtotal_annual)}
          <span class="alice-check-note">${Panels.escapeHtml(food.note)}</span>
        </div>
      `);
    }
    if (gas) {
      lines.push(`
        <div class="alice-check-row">
          <strong>EIA gas component:</strong>
          $${fmtMoney(gas.annual_gas_cost)}
          (${gas.gallons_per_year}gal × $${gas.price_per_gallon}/gal)
          <span class="alice-check-note">${Panels.escapeHtml(gas.note)}</span>
        </div>
      `);
    }
    return `
      <details class="alice-checks">
        <summary>Data cross-checks (using cached BLS + EIA series)</summary>
        ${lines.join("")}
      </details>
    `;
  }

  function renderInterpretation(out) {
    return `
      <div class="alice-interp">
        ${Panels.escapeHtml(out.interpretation || "")}
      </div>
    `;
  }

  function mountAlicePanel(container, options) {
    const opts = options || {};
    const params = new URLSearchParams();
    if (opts.year) params.set("year", String(opts.year));
    if (opts.composition) params.set("composition", String(opts.composition));
    if (opts.household_size && !opts.composition) {
      params.set("household_size", String(opts.household_size));
    }
    container.innerHTML = Panels.stateCard({ kind: "loading" });

    (async () => {
      try {
        const r = await Panels.safeFetch(`${ENDPOINT}?${params.toString()}`);
        const out = await r.json();
        if (out.error) {
          container.innerHTML = Panels.stateCard({
            kind: "error",
            label: "Failed to estimate",
            message: out.error,
          });
          return;
        }
        if (opts.onYearsLoaded && Array.isArray(out.available_years)) {
          opts.onYearsLoaded(out.available_years, out.year);
        }
        if (opts.onCompositionsLoaded && Array.isArray(out.available_compositions)) {
          opts.onCompositionsLoaded(out.available_compositions, out.composition);
        }
        if (opts.onSubtitle) {
          opts.onSubtitle(`${out.year} · ${out.composition_details.label}`);
        }

        container.innerHTML = `
          ${renderInterpretation(out)}
          <h3 class="alice-section-title">Survival budget — ${Panels.escapeHtml(out.composition_details.label)}</h3>
          ${renderBudgetTable(out.survival_budget)}
          <h3 class="alice-section-title">Taxes (filing as ${Panels.escapeHtml(out.composition_details.filing_status)})</h3>
          ${renderTaxBreakdown(out.survival_budget.taxes)}
          ${renderFinalTotal(out.survival_budget, out.composition_details, out.year)}
          <h3 class="alice-section-title">Cross-validation vs United for ALICE</h3>
          ${renderComparison(out)}
          ${renderCrossCheck(out.data_cross_check)}
        `;
      } catch (e) {
        container.innerHTML = Panels.stateCard({
          kind: "error",
          label: "Network error",
          message: String(e && e.message || e),
        });
      }
    })();

    return {
      unmount: () => { container.innerHTML = ""; },
    };
  }

  window.Alice = { mountAlicePanel };
})();
