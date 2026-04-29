"""Symbolic math tool (SymPy)."""
from __future__ import annotations

import json
import logging

log = logging.getLogger("bridge.tools")


# ---- Unified math tool (SymPy) --------------------------------------------

# Cap on parsed matrix dimensions — eigenvalue / inverse / RREF on a
# matrix this large would still complete inside the 10 s sympy timeout
# but eats real CPU on the Jetson. Anything bigger is essentially never
# a legitimate user request and is the shape a malicious / hallucinated
# LLM tool call takes. Refuse early with a clear message.
_MAX_MATRIX_ELEMENTS = 100


def _parse_matrix(text: str):
    """Parse a matrix literal in JSON or MATLAB-style into a sympy Matrix.

    Accepted forms:
      - JSON: '[[1,2],[3,4]]'
      - MATLAB: '1 2; 3 4' (semicolons between rows, whitespace or comma between cols)

    DoS guard: rejects matrices with more than ``_MAX_MATRIX_ELEMENTS``
    entries (rows × cols). Without this, the prior 50-element comma cap
    only counted commas in a single row — a 20×5 MATLAB-style matrix
    slipped through and could keep eigenvalue / inverse busy for the
    full 10 s sympy timeout.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor,
    )
    s = (text or "").strip()
    if not s:
        raise ValueError("empty matrix")
    # convert_xor: `^` → `**` so users writing math-notation exponents
    # don't hit sympy's Python-XOR semantics default (which TypeErrors).
    transformations = standard_transformations + (
        implicit_multiplication_application, convert_xor,
    )

    # JSON branch.
    if s.startswith("["):
        try:
            rows = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"matrix JSON parse failed: {e}")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], list):
            raise ValueError("JSON matrix must be a 2-D array like [[1,2],[3,4]]")
        total = sum(len(r) if isinstance(r, list) else 0 for r in rows)
        if total > _MAX_MATRIX_ELEMENTS:
            raise ValueError(
                f"matrix too large ({total} entries; max {_MAX_MATRIX_ELEMENTS})"
            )
        # Allow numeric or string entries (so '[[a, b], [c, d]]' works symbolically).
        parsed = [[parse_expr(str(v), transformations=transformations) for v in row] for row in rows]
        return sp.Matrix(parsed)

    # MATLAB-style branch.
    row_strs = [r.strip() for r in s.split(";") if r.strip()]
    if not row_strs:
        raise ValueError("no matrix rows")
    rows = []
    for rs in row_strs:
        # Split on commas first (allows "1, 2, 3"), fall back to whitespace.
        cells = [c.strip() for c in rs.split(",")] if "," in rs else rs.split()
        if not cells:
            raise ValueError("empty matrix row")
        rows.append([parse_expr(c, transformations=transformations) for c in cells])
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("ragged matrix — rows must be the same length")
    total = len(rows) * width
    if total > _MAX_MATRIX_ELEMENTS:
        raise ValueError(
            f"matrix too large ({total} entries; max {_MAX_MATRIX_ELEMENTS})"
        )
    return sp.Matrix(rows)


def math_data(
    op: str,
    expression: str,
    variable: str = "x",
    transform_var: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
) -> dict:
    """Unified SymPy-backed math backend.

    Returns a structured dict the widget can render directly:
        {
          "op": str,
          "input_latex": str,     # the original input, rendered
          "result_latex": str,    # the result, rendered
          "plain": str,           # plain-text summary for the LLM
          "error": str | None,
        }
    """
    op_lc = (op or "").strip().lower()
    expr_s = (expression or "").strip()

    # Op inference — LoRAs frequently forget the op field but encode
    # the intent as a wrapper (`kl_divergence(...)`, `H(...)`,
    # `integrate(...)`, etc.). Recognise common full-name + shorthand
    # forms, unwrap to the inner args, and continue with the canonical
    # op. Saves a retry / fallback where we'd otherwise bounce the
    # tool call with "missing op".
    _OP_ALIASES = {
        "kl_divergence": "kl_dist", "kl": "kl_dist",
        "kullback_leibler": "kl_dist", "kullback-leibler": "kl_dist",
        "d_kl": "kl_dist", "dkl": "kl_dist",
        "entropy": "entropy_dist", "h": "entropy_dist",
        "shannon_entropy": "entropy_dist", "differential_entropy": "entropy_dist",
        "js_divergence": "js", "js": "js",
        "jensen_shannon": "js", "jensen-shannon": "js",
        "integrate": "integrate", "integral": "integrate", "int": "integrate",
        "differentiate": "differentiate", "deriv": "differentiate",
        "derivative": "differentiate", "d": "differentiate",
        "simplify": "simplify", "factor": "factor", "expand": "expand",
        "solve": "solve",
        "taylor": "taylor", "series": "series", "maclaurin": "maclaurin",
        "limit": "limit", "lim": "limit",
        "extrema": "extrema", "optimize": "optimize",
        "minimize": "minimize", "maximize": "maximize",
        "rref": "rref",
        "det": "det", "determinant": "det",
        "inverse": "inverse", "inv": "inverse",
        "transpose": "transpose",
        "eigenvalues": "eigenvalues", "eig": "eigenvalues",
        "eigenvectors": "eigenvectors",
        "laplace": "laplace", "inverse_laplace": "inverse_laplace",
        "fourier": "fourier", "inverse_fourier": "inverse_fourier",
        "fft": "fft", "ifft": "ifft",
        "dot": "dot", "norm": "norm",
        "softmax": "softmax", "sigmoid": "sigmoid",
        "tanh": "tanh", "relu": "relu",
        "mean": "mean", "avg": "mean", "average": "mean",
        "variance": "variance", "var": "variance",
        "std": "std", "stdev": "std", "standard_deviation": "std",
        "mse": "mse", "mean_squared_error": "mse",
        "mae": "mae", "mean_absolute_error": "mae",
        "cross_entropy": "cross_entropy", "crossentropy": "cross_entropy",
        "evaluate": "evaluate", "eval": "evaluate", "calc": "evaluate",
    }
    if not op_lc and expr_s:
        import re as _re_infer
        # Leading `Name(` — capture the wrapper and try to resolve it.
        m = _re_infer.match(r"\s*([A-Za-z][A-Za-z_0-9\-]*)\s*\(", expr_s)
        if m:
            guess = m.group(1).strip().lower().replace("-", "_")
            canon = _OP_ALIASES.get(guess)
            if canon:
                op_lc = canon
                # Strip the outer wrapper so downstream parsing sees
                # just the args. `rfind(")")` is safe here because
                # the wrapper's matching close paren is the rightmost
                # one for well-formed input; malformed input falls
                # through to the op-specific parsers which have their
                # own error paths.
                end = expr_s.rfind(")")
                if end > m.end():
                    expr_s = expr_s[m.end():end].strip()

    if not op_lc:
        return {"error": "missing op"}
    if not expr_s:
        return {"error": "empty expression"}
    if len(expr_s) > 600:
        return {"error": "expression too long (cap is 600 chars)"}
    # Cap matrix dimensions to prevent sympy eigenvalue/determinant
    # operations from hanging on large inputs.
    _bracket_depth = max((expr_s.count("[") - expr_s.count("]")), 0)
    if expr_s.count(",") > 50:
        return {"error": "expression too complex (>50 elements)"}

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application,
            convert_xor,
        )
    except ImportError:
        return {"error": "sympy not installed"}

    # convert_xor: `^` → `**` (same rationale as in _parse_matrix above).
    transformations = standard_transformations + (
        implicit_multiplication_application, convert_xor,
    )
    var_sym = sp.Symbol(variable or "x")
    local_dict = {variable or "x": var_sym}
    if transform_var:
        local_dict[transform_var] = sp.Symbol(transform_var)

    def _parse(s: str):
        return parse_expr(s, transformations=transformations, local_dict=local_dict)

    # Timeout wrapper — sympy operations on adversarial inputs can
    # hang indefinitely. Cap at 10 seconds per operation.
    import signal as _signal
    import threading as _threading

    class _MathTimeout(Exception):
        pass

    def _run_with_timeout(fn, timeout_sec=10):
        result = [None]
        error = [None]
        def _target():
            try:
                result[0] = fn()
            except Exception as e:
                error[0] = e
        t = _threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            raise _MathTimeout(f"math operation timed out ({timeout_sec}s)")
        if error[0]:
            raise error[0]
        return result[0]

    def _do_math():
        nonlocal op_lc, expr_s, variable, transform_var, lower, upper
        input_latex = ""
        result_latex = ""
        plain = ""

        # ---- scalar expression ops ----
        if op_lc == "evaluate":
            expr = _parse(expr_s)
            input_latex = sp.latex(expr)
            val = expr.evalf()
            result_latex = sp.latex(val)
            plain = f"{expr_s} = {val}"

        elif op_lc == "solve":
            if "=" in expr_s:
                lhs_s, rhs_s = expr_s.split("=", 1)
                eq = sp.Eq(_parse(lhs_s), _parse(rhs_s))
            else:
                eq = sp.Eq(_parse(expr_s), 0)
            input_latex = sp.latex(eq)
            sols = sp.solve(eq, var_sym)
            if not sols:
                return {"op": op_lc, "input_latex": input_latex,
                        "result_latex": "\\varnothing",
                        "plain": f"No solutions for {variable or 'x'}.",
                        "error": None}
            result_latex = (
                f"{sp.latex(var_sym)} = " + ", ".join(sp.latex(s) for s in sols)
            )
            plain = f"{variable or 'x'} = " + ", ".join(str(s) for s in sols)

        elif op_lc in ("simplify", "factor", "expand"):
            expr = _parse(expr_s)
            input_latex = sp.latex(expr)
            fn = {"simplify": sp.simplify, "factor": sp.factor, "expand": sp.expand}[op_lc]
            result = fn(expr)
            result_latex = sp.latex(result)
            plain = f"{expr_s} \u2192 {result}"

        elif op_lc == "differentiate":
            expr = _parse(expr_s)
            d = sp.diff(expr, var_sym)
            input_latex = f"\\frac{{d}}{{d{sp.latex(var_sym)}}} \\left({sp.latex(expr)}\\right)"
            result_latex = sp.latex(d)
            plain = f"d/d{variable or 'x'} of {expr_s} = {d}"

        elif op_lc == "integrate":
            expr = _parse(expr_s)
            if lower is not None and upper is not None:
                lo = _parse(str(lower)); hi = _parse(str(upper))
                val = sp.integrate(expr, (var_sym, lo, hi))
                input_latex = f"\\int_{{{sp.latex(lo)}}}^{{{sp.latex(hi)}}} {sp.latex(expr)} \\, d{sp.latex(var_sym)}"
                result_latex = sp.latex(val)
                try:
                    plain = f"integral from {lower} to {upper} = {val} (≈ {float(val):.6g})"
                except (TypeError, ValueError):
                    plain = f"integral from {lower} to {upper} = {val}"
            else:
                indef = sp.integrate(expr, var_sym)
                input_latex = f"\\int {sp.latex(expr)} \\, d{sp.latex(var_sym)}"
                result_latex = sp.latex(indef) + " + C"
                plain = f"integral of {expr_s} = {indef} + C"

        # ---- Taylor / Maclaurin series ----------------------------------
        # `lower` = expansion center (default 0 → Maclaurin).
        # `upper` = number of terms / series order (default 6).
        elif op_lc in ("taylor", "series", "maclaurin"):
            expr = _parse(expr_s)
            try:
                center = float(lower) if lower is not None else 0.0
            except (TypeError, ValueError):
                # Allow symbolic centres like "pi" or "e"
                center = _parse(str(lower)) if lower is not None else 0
            try:
                order = int(upper) if upper is not None else 6
            except (TypeError, ValueError):
                return {"error": f"taylor 'upper' must be integer order, got {upper!r}"}
            if order < 2:
                return {"error": "taylor order must be >= 2"}
            if order > 20:
                return {"error": "taylor order capped at 20 (was {})".format(order)}
            s = sp.series(expr, var_sym, center, order).removeO()
            input_latex = (
                f"{sp.latex(expr)} \\text{{ around }} "
                f"{sp.latex(var_sym)} = {sp.latex(center)} "
                f"\\text{{ (order {order})}}"
            )
            result_latex = sp.latex(s)
            plain = f"Taylor({expr_s}, {variable or 'x'}={center}, n={order}) = {s}"

        # ---- Limits -----------------------------------------------------
        # `lower` = target value; strings 'inf'/'-inf'/'+inf' mapped to
        # ±∞. `upper` = optional direction ('+' or '-') for one-sided.
        elif op_lc == "limit":
            expr = _parse(expr_s)
            tgt_raw = str(lower) if lower is not None else "0"
            tgt_lc = tgt_raw.strip().lower()
            if tgt_lc in ("inf", "+inf", "infinity", "oo"):
                target = sp.oo
                tgt_display = "\\infty"
            elif tgt_lc in ("-inf", "-infinity", "-oo"):
                target = -sp.oo
                tgt_display = "-\\infty"
            else:
                target = _parse(tgt_raw)
                tgt_display = sp.latex(target)
            dir_arg = str(upper).strip() if upper is not None else "+-"
            if dir_arg not in ("+", "-", "+-"):
                dir_arg = "+-"
            if dir_arg == "+-":
                val = sp.limit(expr, var_sym, target)
                side_label = ""
            else:
                val = sp.limit(expr, var_sym, target, dir_arg)
                side_label = f" (from {dir_arg})"
            input_latex = (
                f"\\lim_{{{sp.latex(var_sym)} \\to {tgt_display}}}"
                f" {sp.latex(expr)}"
            )
            result_latex = sp.latex(val)
            plain = (
                f"lim({expr_s}) as {variable or 'x'} → {tgt_raw}"
                f"{side_label} = {val}"
            )

        # ---- Extrema / optimization -------------------------------------
        # Finds critical points (f'(x) = 0), classifies each via the
        # second-derivative test, and returns [(x, y, type), ...]. If
        # `lower`/`upper` are set, also compares endpoint values so the
        # caller can see the global min/max on the closed interval.
        elif op_lc in ("extrema", "optimize", "minimize", "maximize"):
            expr = _parse(expr_s)
            d1 = sp.diff(expr, var_sym)
            d2 = sp.diff(expr, var_sym, 2)
            try:
                crits = sp.solve(d1, var_sym)
            except Exception as e:
                return {"error": f"extrema: couldn't solve f'=0: {e}"}
            results = []  # list of (x, y, kind)
            for c in crits:
                # Skip non-real roots — they're not useful extrema on R
                if not c.is_real and c.is_real is not None:
                    continue
                try:
                    y = expr.subs(var_sym, c)
                    curv = d2.subs(var_sym, c)
                    if curv > 0:
                        kind = "min"
                    elif curv < 0:
                        kind = "max"
                    else:
                        kind = "inflection/saddle"
                    results.append((c, y, kind))
                except Exception:
                    results.append((c, None, "unknown"))

            # Endpoint comparison for bounded intervals
            endpoints = []
            if lower is not None:
                try:
                    lo = _parse(str(lower))
                    endpoints.append((lo, expr.subs(var_sym, lo), "endpoint"))
                except Exception:
                    pass
            if upper is not None:
                try:
                    hi = _parse(str(upper))
                    endpoints.append((hi, expr.subs(var_sym, hi), "endpoint"))
                except Exception:
                    pass

            all_pts = results + endpoints
            if not all_pts:
                input_latex = f"\\text{{extrema of }} {sp.latex(expr)}"
                result_latex = "\\varnothing"
                plain = f"no real critical points of {expr_s}"
            else:
                # For the 'minimize' / 'maximize' ops, also return a winner
                winner = None
                if op_lc in ("minimize", "maximize") and all_pts:
                    numeric = [(c, y, k) for (c, y, k) in all_pts
                               if y is not None and y.is_number]
                    if numeric:
                        if op_lc == "minimize":
                            winner = min(numeric, key=lambda t: float(t[1]))
                        else:
                            winner = max(numeric, key=lambda t: float(t[1]))

                input_latex = f"\\text{{extrema of }} {sp.latex(expr)}"
                parts_l = []
                parts_p = []
                for (c, y, k) in all_pts:
                    y_latex = sp.latex(y) if y is not None else "?"
                    parts_l.append(
                        f"{sp.latex(var_sym)}={sp.latex(c)},\\ f={y_latex}\\ ({k})"
                    )
                    parts_p.append(f"{variable or 'x'}={c}, f={y} ({k})")
                result_latex = " \\\\ ".join(parts_l)
                if winner is not None:
                    result_latex += (
                        f" \\\\ \\boxed{{{op_lc}: "
                        f"{sp.latex(winner[0])}, f={sp.latex(winner[1])}}}"
                    )
                    plain = ("; ".join(parts_p)
                             + f" | {op_lc}: x={winner[0]}, f={winner[1]}")
                else:
                    plain = "; ".join(parts_p)

        # ---- matrix ops ----
        elif op_lc in ("rref", "det", "inverse", "transpose", "eigenvalues", "eigenvectors"):
            M = _parse_matrix(expr_s)
            input_latex = sp.latex(M)
            if op_lc == "rref":
                rref_M, pivots = M.rref()
                result_latex = sp.latex(rref_M)
                plain = f"rref: {rref_M.tolist()}, pivots {list(pivots)}"
            elif op_lc == "det":
                d = M.det()
                result_latex = sp.latex(d)
                plain = f"det = {d}"
            elif op_lc == "inverse":
                inv = M.inv()
                result_latex = sp.latex(inv)
                plain = f"inverse = {inv.tolist()}"
            elif op_lc == "transpose":
                t = M.T
                result_latex = sp.latex(t)
                plain = f"transpose = {t.tolist()}"
            elif op_lc == "eigenvalues":
                ev = M.eigenvals()
                result_latex = ", ".join(f"{sp.latex(v)} (\\text{{mult }} {m})" for v, m in ev.items())
                plain = "eigenvalues: " + ", ".join(f"{v} (×{m})" for v, m in ev.items())
            elif op_lc == "eigenvectors":
                evs = M.eigenvects()
                parts_l, parts_p = [], []
                for val, mult, vecs in evs:
                    parts_l.append(f"\\lambda={sp.latex(val)}: " +
                                   ", ".join(sp.latex(v) for v in vecs))
                    parts_p.append(f"λ={val}: " + ", ".join(str(v.tolist()) for v in vecs))
                result_latex = " \\\\ ".join(parts_l)
                plain = "; ".join(parts_p)

        elif op_lc == "multiply":
            if "*" not in expr_s:
                return {"error": "multiply needs two matrices separated by '*'"}
            a_s, b_s = expr_s.split("*", 1)
            A = _parse_matrix(a_s)
            B = _parse_matrix(b_s)
            input_latex = f"{sp.latex(A)} \\cdot {sp.latex(B)}"
            prod = A * B
            result_latex = sp.latex(prod)
            plain = f"A * B = {prod.tolist()}"

        # ---- transforms ----
        elif op_lc == "laplace":
            t = sp.Symbol(variable or "t")
            sv = sp.Symbol(transform_var or "s")
            f = parse_expr(expr_s, transformations=transformations, local_dict={str(t): t})
            F = sp.laplace_transform(f, t, sv, noconds=True)
            input_latex = f"\\mathcal{{L}}\\{{{sp.latex(f)}\\}}"
            result_latex = sp.latex(F)
            plain = f"L{{{expr_s}}} = {F}"
        elif op_lc == "inverse_laplace":
            sv = sp.Symbol(variable or "s")
            t = sp.Symbol(transform_var or "t")
            F = parse_expr(expr_s, transformations=transformations, local_dict={str(sv): sv})
            f = sp.inverse_laplace_transform(F, sv, t, noconds=True)
            input_latex = f"\\mathcal{{L}}^{{-1}}\\{{{sp.latex(F)}\\}}"
            result_latex = sp.latex(f)
            plain = f"L^-1{{{expr_s}}} = {f}"
        elif op_lc == "fourier":
            x = sp.Symbol(variable or "x")
            k = sp.Symbol(transform_var or "k")
            f = parse_expr(expr_s, transformations=transformations, local_dict={str(x): x})
            F = sp.fourier_transform(f, x, k)
            input_latex = f"\\mathcal{{F}}\\{{{sp.latex(f)}\\}}"
            result_latex = sp.latex(F)
            plain = f"F{{{expr_s}}} = {F}"
        elif op_lc == "inverse_fourier":
            k = sp.Symbol(variable or "k")
            x = sp.Symbol(transform_var or "x")
            F = parse_expr(expr_s, transformations=transformations, local_dict={str(k): k})
            f = sp.inverse_fourier_transform(F, k, x)
            input_latex = f"\\mathcal{{F}}^{{-1}}\\{{{sp.latex(F)}\\}}"
            result_latex = sp.latex(f)
            plain = f"F^-1{{{expr_s}}} = {f}"

        elif op_lc in ("fft", "ifft"):
            # Numerical DFT / inverse DFT on a sample array.
            # Expression is a comma-separated list of numbers or
            # complex values like '1, 0, -1, 0'.
            try:
                import numpy as np
            except ImportError:
                return {"error": "numpy not installed"}
            parts = [p.strip() for p in expr_s.split(",") if p.strip()]
            if not parts:
                return {"error": "empty sample array"}

            def _parse_num(s: str) -> complex:
                t = s.strip().replace("j", "j").replace("i", "j")
                return complex(t)

            try:
                samples = np.array([_parse_num(p) for p in parts], dtype=complex)
            except ValueError:
                return {"error": "samples must be numbers (real or complex)"}

            if op_lc == "fft":
                X = np.fft.fft(samples)
                label = "FFT"
            else:
                X = np.fft.ifft(samples)
                label = "IFFT"

            mag = np.abs(X)
            phase_deg = np.degrees(np.angle(X))
            # Render output in two rows: complex coefficients and magnitudes.
            fmt_c = lambda c: (
                f"{c.real:.4g}" if abs(c.imag) < 1e-12
                else f"{c.real:.4g}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.4g}j"
            )
            coeffs_s = ", ".join(fmt_c(v) for v in X)
            mags_s = ", ".join(f"{m:.4g}" for m in mag)
            input_latex = "[" + ",\\ ".join(fmt_c(v) for v in samples) + "]"
            result_latex = "[" + ",\\ ".join(fmt_c(v) for v in X) + "]"
            plain = (
                f"{label} of [{', '.join(fmt_c(v) for v in samples)}] = "
                f"[{coeffs_s}]\nmagnitudes: [{mags_s}]"
            )
            return {"op": op_lc,
                    "input_latex": input_latex,
                    "result_latex": result_latex,
                    "plain": plain,
                    "coefficients": [[float(v.real), float(v.imag)] for v in X],
                    "magnitudes": mag.tolist(),
                    "phases_deg": phase_deg.tolist(),
                    "error": None}

        # ---- ML-oriented numeric ops -------------------------------------
        # Vector inputs accept either "[1, 2, 3]" or "1, 2, 3".
        # Two-vector ops (dot, mse, mae, cross_entropy) expect
        # "[1,2,3] * [4,5,6]".
        elif op_lc in ("dot", "norm", "softmax", "sigmoid", "tanh", "relu",
                       "mean", "variance", "std", "mse", "mae",
                       "cross_entropy",
                       "entropy", "kl_divergence", "kl", "js_divergence", "js",
                       "kl_dist", "entropy_dist"):
            try:
                import numpy as np
            except ImportError:
                return {"error": "numpy not installed"}

            # Auto-route to closed-form/numerical distribution ops
            # whenever the expression LOOKS like distribution shorthand
            # rather than a vector pair. Two independent triggers (any
            # one of them fires retarget):
            #
            #   (a) Two or more KNOWN-family distribution calls —
            #       ``N(0,1) || N(1,2)``, ``[Beta(2,5), N(0,1)]``,
            #       ``kl(Beta(2,5), N(0,1))``.
            #   (b) ANY generic ``Name(args)`` pair joined by ``||``,
            #       ``vs``, or ``against`` — catches cases where the
            #       LoRA substituted an unknown family (e.g.
            #       ``dirichlet(...) || normal(...)``). The kl_dist
            #       parser surfaces a useful "unknown distribution
            #       'dirichlet'" error instead of the generic vector
            #       "needs two vectors" bounce.
            #
            # Pure numeric vectors like ``[0.9, 0.1] * [0.5, 0.5]``
            # have zero Name(args) calls and zero ``||`` separators,
            # so they never retarget.
            import re as _re_rt
            _dist_call = _re_rt.compile(
                r"\b(?:N|Normal|Gaussian|Exp|Exponential|Beta|"
                r"Bern|Bernoulli|Uniform|U)\s*\(\s*[^)]*\s*\)",
                flags=_re_rt.IGNORECASE,
            )
            _name_call_pair = _re_rt.compile(
                r"[A-Za-z_][A-Za-z_0-9]*\s*\([^)]*\)\s*(?:\|\||vs\.?|against)\s*"
                r"[A-Za-z_][A-Za-z_0-9]*\s*\([^)]*\)",
                flags=_re_rt.IGNORECASE,
            )
            _name_call_single = _re_rt.compile(
                r"\b[A-Za-z_][A-Za-z_0-9]*\s*\([^)]*\)",
            )
            _n_known = len(_dist_call.findall(expr_s))
            _has_kl_pair = bool(_name_call_pair.search(expr_s))
            _n_name_calls = len(_name_call_single.findall(expr_s))

            if op_lc in ("kl", "kl_divergence", "js", "js_divergence") and (
                _n_known >= 2 or _has_kl_pair
            ):
                op_lc = "kl_dist"
            elif op_lc == "entropy" and (
                _n_known == 1 or _n_name_calls == 1
            ):
                op_lc = "entropy_dist"

            def _parse_vec(s: str) -> list[float]:
                """Parse '[1, 2, 3]' or '1, 2, 3' → list[float]."""
                s = s.strip()
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if not parts:
                    raise ValueError("empty vector")
                return [float(p) for p in parts]

            def _fmt_vec(v, fmt="%.4g") -> str:
                return "[" + ", ".join(fmt % x for x in v) + "]"

            def _parse_two_vectors(s: str) -> tuple[list[float], list[float]]:
                """Extract two bracketed vectors from a free-form string.

                Accepts any separator between them — '*', '||', 'and',
                ',', whitespace, nothing at all. Models frequently
                struggle with our historical '[a,b] * [c,d]' syntax
                and emit variants; be permissive on input so we don't
                bounce their calls.
                """
                import re as _re
                matches = _re.findall(r"\[[^\[\]]*\]", s)
                if len(matches) >= 2:
                    return _parse_vec(matches[0]), _parse_vec(matches[1])
                # Single-bracket fallback: split on the first '*' if any.
                if "*" in s:
                    a_s, b_s = s.split("*", 1)
                    return _parse_vec(a_s), _parse_vec(b_s)
                raise ValueError(
                    "needs two vectors; pass them as '[...] * [...]' or "
                    "just '[...] [...]'"
                )

            if op_lc in ("dot", "mse", "mae", "cross_entropy",
                         "kl_divergence", "kl", "js_divergence", "js"):
                try:
                    a_list, b_list = _parse_two_vectors(expr_s)
                except ValueError as e:
                    return {"error": f"{op_lc}: {e}"}
                a = np.array(a_list, dtype=float)
                b = np.array(b_list, dtype=float)
                if a.shape != b.shape:
                    return {"error": f"shape mismatch: {a.shape} vs {b.shape}"}

                if op_lc == "dot":
                    val = float(np.dot(a, b))
                    input_latex = f"{_fmt_vec(a)} \\cdot {_fmt_vec(b)}"
                    result_latex = f"{val:.6g}"
                    plain = f"{_fmt_vec(a)} · {_fmt_vec(b)} = {val:.6g}"
                elif op_lc == "mse":
                    val = float(np.mean((a - b) ** 2))
                    input_latex = f"\\text{{MSE}}({_fmt_vec(a)}, {_fmt_vec(b)})"
                    result_latex = f"{val:.6g}"
                    plain = f"MSE = {val:.6g}"
                elif op_lc == "mae":
                    val = float(np.mean(np.abs(a - b)))
                    input_latex = f"\\text{{MAE}}({_fmt_vec(a)}, {_fmt_vec(b)})"
                    result_latex = f"{val:.6g}"
                    plain = f"MAE = {val:.6g}"
                elif op_lc == "cross_entropy":
                    # Treats `a` as predicted probabilities, `b` as target
                    # distribution. Clamps predictions to avoid log(0).
                    # For 1-hot `b` this reduces to negative log-likelihood.
                    eps = 1e-12
                    a_c = np.clip(a, eps, 1.0)
                    val = float(-np.sum(b * np.log(a_c)))
                    input_latex = f"\\text{{CE}}(\\hat p={_fmt_vec(a)}, y={_fmt_vec(b)})"
                    result_latex = f"{val:.6g}"
                    plain = (f"cross-entropy = {val:.6g} "
                             f"(predictions {_fmt_vec(a)}, target {_fmt_vec(b)})")
                elif op_lc in ("kl_divergence", "kl"):
                    # D_KL(P || Q) = sum_i p_i log(p_i / q_i).
                    # Non-negative; 0 iff P==Q. Not symmetric.
                    # Terms where p_i == 0 are zero by convention
                    # (0 log 0 = 0); terms where q_i == 0 and p_i > 0
                    # are +inf (clamped via eps).
                    eps = 1e-12
                    p = a / (a.sum() if a.sum() > 0 else 1.0)
                    q = b / (b.sum() if b.sum() > 0 else 1.0)
                    mask = p > 0
                    q_c = np.clip(q[mask], eps, None)
                    val = float(np.sum(p[mask] * np.log(p[mask] / q_c)))
                    input_latex = (f"D_{{\\mathrm{{KL}}}}({_fmt_vec(p)} "
                                   f"\\parallel {_fmt_vec(q)})")
                    result_latex = f"{val:.6g}"
                    plain = (f"KL(P||Q) = {val:.6g} nats "
                             f"(P={_fmt_vec(p)}, Q={_fmt_vec(q)})")
                elif op_lc in ("js_divergence", "js"):
                    # Jensen-Shannon divergence: symmetric, bounded [0, ln 2].
                    # JS(P, Q) = 0.5·KL(P || M) + 0.5·KL(Q || M),  M=(P+Q)/2.
                    eps = 1e-12
                    p = a / (a.sum() if a.sum() > 0 else 1.0)
                    q = b / (b.sum() if b.sum() > 0 else 1.0)
                    m = 0.5 * (p + q)

                    def _kl_term(x, y):
                        mask = x > 0
                        y_c = np.clip(y[mask], eps, None)
                        return float(np.sum(x[mask] * np.log(x[mask] / y_c)))

                    val = 0.5 * _kl_term(p, m) + 0.5 * _kl_term(q, m)
                    input_latex = (f"D_{{\\mathrm{{JS}}}}({_fmt_vec(p)}, "
                                   f"{_fmt_vec(q)})")
                    result_latex = f"{val:.6g}"
                    plain = (f"JS(P, Q) = {val:.6g} nats "
                             f"(bounded [0, ln 2] ≈ [0, 0.693])")

            elif op_lc in ("kl_dist", "entropy_dist"):
                # Distribution-aware closed-form KL and entropy.
                #
                # Input formats:
                #   kl_dist     : "N(0, 1) || N(1, 2)"  (any separator)
                #   entropy_dist: "N(0, 1)"  (single distribution)
                #
                # Supported distributions (case-insensitive, aliases ok):
                #   Normal / N / Gaussian (mu, sigma)   — sigma is std dev
                #   Exponential / Exp (rate)             — rate = 1/mean
                #   Beta (alpha, beta)
                #   Bernoulli / Bern (p)
                #   Uniform / U (a, b)
                import re as _re
                from math import gamma as _gamma, lgamma as _lgamma

                # Digamma approximation — good enough for α,β ≥ 0.5 which
                # covers the common Beta / Gamma parameter ranges. Uses
                # the asymptotic series plus recurrence to lift small
                # arguments. No scipy dependency.
                def _digamma(x: float) -> float:
                    if x < 6:
                        # Recurrence: ψ(x) = ψ(x+1) - 1/x; lift to x≥6
                        n = 0
                        while x < 6:
                            n += 1
                            x += 1
                        asymp = (
                            np.log(x)
                            - 1 / (2 * x)
                            - 1 / (12 * x**2)
                            + 1 / (120 * x**4)
                            - 1 / (252 * x**6)
                        )
                        for k in range(1, n + 1):
                            asymp -= 1 / (x - k)
                        return float(asymp)
                    return float(
                        np.log(x)
                        - 1 / (2 * x)
                        - 1 / (12 * x**2)
                        + 1 / (120 * x**4)
                        - 1 / (252 * x**6)
                    )

                _DIST_ALIASES = {
                    "normal": "normal", "n": "normal", "gaussian": "normal",
                    "exponential": "exponential", "exp": "exponential",
                    "beta": "beta",
                    "bernoulli": "bernoulli", "bern": "bernoulli",
                    "uniform": "uniform", "u": "uniform",
                }
                _DIST_PRETTY = {
                    "normal": "N", "exponential": "Exp", "beta": "Beta",
                    "bernoulli": "Bern", "uniform": "U",
                }

                _DIST_RE = _re.compile(
                    r"([A-Za-z]+)\s*\(\s*([^)]*)\s*\)"
                )

                def _parse_dist(s: str) -> tuple[str, list[float]]:
                    m = _DIST_RE.search(s)
                    if not m:
                        raise ValueError(
                            f"couldn't parse distribution from {s!r}. "
                            "Use 'N(mu, sigma)', 'Exp(rate)', 'Beta(a, b)', "
                            "'Bern(p)', or 'Uniform(a, b)'."
                        )
                    name_raw = m.group(1).lower()
                    name = _DIST_ALIASES.get(name_raw)
                    if name is None:
                        raise ValueError(
                            f"unknown distribution {m.group(1)!r}. "
                            f"Known: {sorted(set(_DIST_ALIASES.values()))}"
                        )
                    args_raw = m.group(2).strip()
                    if not args_raw:
                        return name, []
                    params = [float(p.strip()) for p in args_raw.split(",") if p.strip()]
                    return name, params

                def _entropy_of(name: str, params: list[float]) -> float:
                    if name == "normal":
                        if len(params) != 2:
                            raise ValueError("N needs (mu, sigma)")
                        _, sigma = params
                        if sigma <= 0:
                            raise ValueError("sigma must be > 0")
                        return 0.5 * np.log(2 * np.pi * np.e * sigma**2)
                    if name == "exponential":
                        if len(params) != 1:
                            raise ValueError("Exp needs (rate)")
                        lam = params[0]
                        if lam <= 0:
                            raise ValueError("rate must be > 0")
                        return 1.0 - np.log(lam)
                    if name == "beta":
                        if len(params) != 2:
                            raise ValueError("Beta needs (alpha, beta)")
                        a, b = params
                        if a <= 0 or b <= 0:
                            raise ValueError("alpha, beta must be > 0")
                        log_B = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
                        return (
                            log_B
                            - (a - 1) * _digamma(a)
                            - (b - 1) * _digamma(b)
                            + (a + b - 2) * _digamma(a + b)
                        )
                    if name == "bernoulli":
                        if len(params) != 1:
                            raise ValueError("Bern needs (p)")
                        p = params[0]
                        if not (0 <= p <= 1):
                            raise ValueError("p must be in [0, 1]")
                        if p in (0.0, 1.0):
                            return 0.0
                        return float(-p * np.log(p) - (1 - p) * np.log(1 - p))
                    if name == "uniform":
                        if len(params) != 2:
                            raise ValueError("Uniform needs (a, b)")
                        a, b = params
                        if b <= a:
                            raise ValueError("Uniform requires b > a")
                        return float(np.log(b - a))
                    raise ValueError(f"entropy for {name!r} not implemented")

                # Continuous PDF + support — only used for cross-family
                # numerical KL. Bernoulli is discrete and excluded; mixing
                # it with continuous distributions isn't well-defined on a
                # single measure, so we refuse cross-family KL with Bern.
                def _pdf(name: str, params: list[float], x):
                    if name == "normal":
                        mu, sigma = params
                        return np.exp(-(x - mu) ** 2 / (2 * sigma**2)) / (
                            sigma * np.sqrt(2 * np.pi)
                        )
                    if name == "exponential":
                        lam = params[0]
                        y = np.zeros_like(x, dtype=float)
                        m = x >= 0
                        y[m] = lam * np.exp(-lam * x[m])
                        return y
                    if name == "beta":
                        a, b = params
                        log_B = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
                        y = np.zeros_like(x, dtype=float)
                        m = (x > 0) & (x < 1)
                        y[m] = np.exp(
                            (a - 1) * np.log(x[m])
                            + (b - 1) * np.log(1 - x[m])
                            - log_B
                        )
                        return y
                    if name == "uniform":
                        a, b = params
                        y = np.zeros_like(x, dtype=float)
                        m = (x >= a) & (x <= b)
                        y[m] = 1.0 / (b - a)
                        return y
                    raise ValueError(
                        f"no continuous PDF for {name!r} (bernoulli is discrete)"
                    )

                def _support_interval(name: str, params: list[float]) -> tuple[float, float]:
                    """Effective integration interval for P. For unbounded
                    supports we clip to where the tail is numerically
                    negligible so the trapezoid rule converges."""
                    if name == "normal":
                        mu, sigma = params
                        return (mu - 10 * sigma, mu + 10 * sigma)
                    if name == "exponential":
                        lam = params[0]
                        return (0.0, 20.0 / lam)
                    if name == "beta":
                        return (0.0, 1.0)
                    if name == "uniform":
                        a, b = params
                        return (a, b)
                    raise ValueError(f"no support interval for {name!r}")

                def _q_is_zero_on(nm2, p2, lo: float, hi: float) -> bool:
                    """True if Q has zero density somewhere in [lo, hi],
                    making KL(P||Q) infinite whenever p>0 there."""
                    if nm2 == "normal":
                        return False   # positive everywhere on ℝ
                    if nm2 == "exponential":
                        return lo < 0
                    if nm2 == "beta":
                        return lo < 0 or hi > 1
                    if nm2 == "uniform":
                        a, b = p2
                        return lo < a or hi > b
                    return True   # unknown / bernoulli → refuse

                def _kl_numerical(nm1, p1, nm2, p2) -> float:
                    """Trapezoidal KL over P's effective support. Caller
                    must have already verified Q is positive there."""
                    lo, hi = _support_interval(nm1, p1)
                    xs = np.linspace(lo, hi, 8001)
                    p_vals = _pdf(nm1, p1, xs)
                    q_vals = _pdf(nm2, p2, xs)
                    eps = 1e-15
                    mask = p_vals > eps
                    integrand = np.zeros_like(xs)
                    integrand[mask] = p_vals[mask] * (
                        np.log(p_vals[mask]) - np.log(q_vals[mask] + eps)
                    )
                    return float(np.trapz(integrand, xs))

                def _kl_between(nm1, p1, nm2, p2) -> float:
                    if nm1 != nm2:
                        # Cross-family: fall back to numerical KL over P's
                        # support, but only when both distributions are
                        # continuous and Q has non-zero density everywhere
                        # on P's support. Otherwise KL is infinite or
                        # ill-defined.
                        if nm1 == "bernoulli" or nm2 == "bernoulli":
                            raise ValueError(
                                "cross-family KL with bernoulli isn't supported "
                                "(mixed discrete/continuous); try op='kl' with "
                                "discretised vectors."
                            )
                        lo, hi = _support_interval(nm1, p1)
                        if _q_is_zero_on(nm2, p2, lo, hi):
                            return float("inf")
                        return _kl_numerical(nm1, p1, nm2, p2)
                    if nm1 == "normal":
                        mu1, s1 = p1; mu2, s2 = p2
                        if s1 <= 0 or s2 <= 0:
                            raise ValueError("sigma must be > 0")
                        return float(
                            np.log(s2 / s1)
                            + (s1**2 + (mu1 - mu2) ** 2) / (2 * s2**2)
                            - 0.5
                        )
                    if nm1 == "exponential":
                        l1 = p1[0]; l2 = p2[0]
                        if l1 <= 0 or l2 <= 0:
                            raise ValueError("rate must be > 0")
                        return float(np.log(l1 / l2) + l2 / l1 - 1.0)
                    if nm1 == "beta":
                        a1, b1 = p1; a2, b2 = p2
                        if min(a1, b1, a2, b2) <= 0:
                            raise ValueError("alpha, beta must be > 0")
                        log_B1 = _lgamma(a1) + _lgamma(b1) - _lgamma(a1 + b1)
                        log_B2 = _lgamma(a2) + _lgamma(b2) - _lgamma(a2 + b2)
                        return float(
                            log_B2 - log_B1
                            + (a1 - a2) * _digamma(a1)
                            + (b1 - b2) * _digamma(b1)
                            + (a2 - a1 + b2 - b1) * _digamma(a1 + b1)
                        )
                    if nm1 == "bernoulli":
                        q = p1[0]; r = p2[0]
                        if not (0 <= q <= 1 and 0 <= r <= 1):
                            raise ValueError("p must be in [0, 1]")
                        eps = 1e-12
                        r = min(max(r, eps), 1 - eps)
                        if q in (0.0,):
                            return float(-np.log(1 - r))
                        if q in (1.0,):
                            return float(-np.log(r))
                        return float(
                            q * np.log(q / r)
                            + (1 - q) * np.log((1 - q) / (1 - r))
                        )
                    if nm1 == "uniform":
                        a1, b1 = p1; a2, b2 = p2
                        # KL(U[a1,b1] || U[a2,b2]) is ∞ unless the first
                        # is fully contained in the second, then = ln((b2-a2)/(b1-a1)).
                        if a2 <= a1 and b1 <= b2:
                            return float(np.log((b2 - a2) / (b1 - a1)))
                        return float("inf")
                    raise ValueError(f"KL for {nm1!r} not implemented")

                # Paren-balanced scanner that finds *every* ``Name(args)``
                # call whose name is a known distribution, regardless of
                # how deeply nested. This lets us accept natural LoRA
                # phrasings like ``kl(Beta(2,5), N(0,1))`` or
                # ``D_KL(N(0,1) || N(1,2))`` without bouncing the user's
                # question back with a format-correction error — the
                # answer matters more than the wrapper.
                def _dist_matches(s: str) -> list:
                    found: list[str] = []

                    def scan(text: str, off: int = 0) -> None:
                        i = 0
                        while i < len(text):
                            ch = text[i]
                            if not ch.isalpha():
                                i += 1
                                continue
                            j = i
                            while j < len(text) and text[j].isalpha():
                                j += 1
                            name = text[i:j]
                            k = j
                            while k < len(text) and text[k].isspace():
                                k += 1
                            if k >= len(text) or text[k] != "(":
                                i = j
                                continue
                            depth = 0
                            end = k
                            while end < len(text):
                                if text[end] == "(":
                                    depth += 1
                                elif text[end] == ")":
                                    depth -= 1
                                    if depth == 0:
                                        break
                                end += 1
                            if depth != 0:
                                return   # unbalanced — bail out of this arm
                            if name.lower() in _DIST_ALIASES:
                                found.append(text[i:end + 1])
                            else:
                                # Wrapper like ``kl(...)`` or ``D_KL(...)``:
                                # recurse into the args so we find the
                                # real distributions nested inside.
                                scan(text[k + 1:end])
                            i = end + 1

                    scan(s)

                    # Adapter so the downstream code using ``m.group(0)``
                    # and ``m.group(1)`` keeps working.
                    class _Match:
                        def __init__(self, text: str):
                            self._text = text
                            m = _DIST_RE.match(text)
                            self._name = m.group(1) if m else ""

                        def group(self, i: int) -> str:
                            return self._text if i == 0 else self._name

                    return [_Match(t) for t in found]

                if op_lc == "kl_dist":
                    matches = _dist_matches(expr_s)
                    if len(matches) < 2:
                        # See if the expression tried to use Name(args)
                        # shapes with families we don't have closed
                        # forms for (Dirichlet, Gamma, Chi²-etc.). If
                        # yes, name the offender so the error is
                        # actionable instead of "needs two".
                        import re as _re_err
                        _generic = _re_err.compile(
                            r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\("
                        )
                        known_lower = set(_DIST_ALIASES.keys())
                        offenders: list[str] = []
                        for m in _generic.finditer(expr_s):
                            nm = m.group(1)
                            # Skip the known wrappers we recurse
                            # through (kl, entropy, H, D_KL, etc.)
                            if nm.lower() in {
                                "kl", "kl_divergence", "d_kl", "dkl",
                                "entropy", "h", "js", "js_divergence",
                                "kullback_leibler", "jensen_shannon",
                            }:
                                continue
                            if nm.lower() not in known_lower and nm not in offenders:
                                offenders.append(nm)
                        supported = sorted(set(_DIST_ALIASES.values()))
                        if offenders:
                            return {"error":
                                "kl_dist: unknown distribution "
                                f"{offenders[0]!r}. "
                                f"Supported: {supported}. "
                                "For other families, discretise and "
                                "use op='kl' with vectors."}
                        return {"error":
                            "kl_dist needs two distributions separated by "
                            "'||' / 'vs' / 'and'. Pass them bare — no outer "
                            "wrapper. Example: 'Beta(2,5) || N(0,1)'."}
                    nm1, p1 = _parse_dist(matches[0].group(0))
                    nm2, p2 = _parse_dist(matches[1].group(0))
                    try:
                        val = _kl_between(nm1, p1, nm2, p2)
                    except ValueError as e:
                        return {"error": f"kl_dist: {e}"}
                    d1 = f"{_DIST_PRETTY[nm1]}({', '.join(str(p) for p in p1)})"
                    d2 = f"{_DIST_PRETTY[nm2]}({', '.join(str(p) for p in p2)})"
                    input_latex = f"D_{{\\mathrm{{KL}}}}({d1} \\parallel {d2})"
                    result_latex = f"{val:.6g}"
                    plain = f"KL({d1} || {d2}) = {val:.6g} nats"
                else:  # entropy_dist
                    try:
                        ms = _dist_matches(expr_s)
                        if ms:
                            nm, p = _parse_dist(ms[0].group(0))
                        else:
                            nm, p = _parse_dist(expr_s)
                        val = _entropy_of(nm, p)
                    except ValueError as e:
                        return {"error": f"entropy_dist: {e}"}
                    d = f"{_DIST_PRETTY[nm]}({', '.join(str(x) for x in p)})"
                    input_latex = f"H({d})"
                    result_latex = f"{val:.6g}"
                    plain = f"H({d}) = {val:.6g} nats"

            else:
                # Single-vector ops
                v = np.array(_parse_vec(expr_s), dtype=float)

                if op_lc == "norm":
                    # `variable` field smuggles the order: 1, 2 (default),
                    # "inf", or any positive real. Stays on `variable`
                    # because the schema already exposes it, no new param.
                    order_raw = (variable or "2").strip().lower()
                    if order_raw in ("inf", "infinity", "max"):
                        p_order = np.inf
                        p_label = "\\infty"
                    else:
                        try:
                            p_order = float(order_raw)
                            p_label = order_raw
                        except ValueError:
                            return {"error": f"norm order '{order_raw}' not recognised"}
                    val = float(np.linalg.norm(v, ord=p_order))
                    input_latex = f"\\|{_fmt_vec(v)}\\|_{{{p_label}}}"
                    result_latex = f"{val:.6g}"
                    plain = f"L{p_label} norm of {_fmt_vec(v)} = {val:.6g}"
                elif op_lc == "softmax":
                    # Numerically stable: subtract max before exp.
                    shifted = v - np.max(v)
                    ex = np.exp(shifted)
                    probs = ex / np.sum(ex)
                    input_latex = f"\\text{{softmax}}({_fmt_vec(v)})"
                    result_latex = _fmt_vec(probs, "%.6g")
                    plain = f"softmax({_fmt_vec(v)}) = {_fmt_vec(probs, '%.6f')}"
                elif op_lc == "sigmoid":
                    out = 1.0 / (1.0 + np.exp(-v))
                    input_latex = f"\\sigma({_fmt_vec(v)})"
                    result_latex = _fmt_vec(out, "%.6g")
                    plain = f"sigmoid({_fmt_vec(v)}) = {_fmt_vec(out, '%.6f')}"
                elif op_lc == "tanh":
                    out = np.tanh(v)
                    input_latex = f"\\tanh({_fmt_vec(v)})"
                    result_latex = _fmt_vec(out, "%.6g")
                    plain = f"tanh({_fmt_vec(v)}) = {_fmt_vec(out, '%.6f')}"
                elif op_lc == "relu":
                    out = np.maximum(0.0, v)
                    input_latex = f"\\text{{ReLU}}({_fmt_vec(v)})"
                    result_latex = _fmt_vec(out, "%.4g")
                    plain = f"ReLU({_fmt_vec(v)}) = {_fmt_vec(out, '%.4g')}"
                elif op_lc == "mean":
                    val = float(np.mean(v))
                    input_latex = f"\\bar x\\,({_fmt_vec(v)})"
                    result_latex = f"{val:.6g}"
                    plain = f"mean = {val:.6g}"
                elif op_lc == "variance":
                    val = float(np.var(v, ddof=0))
                    input_latex = f"\\text{{Var}}({_fmt_vec(v)})"
                    result_latex = f"{val:.6g}"
                    plain = f"variance (population, ddof=0) = {val:.6g}"
                elif op_lc == "std":
                    val = float(np.std(v, ddof=0))
                    input_latex = f"\\text{{Std}}({_fmt_vec(v)})"
                    result_latex = f"{val:.6g}"
                    plain = f"std (population, ddof=0) = {val:.6g}"
                elif op_lc == "entropy":
                    # Shannon entropy H(P) = -sum p_i log p_i (nats).
                    # Renormalises so the input doesn't need to sum to 1.
                    # Zero probabilities are skipped (0 log 0 = 0 by
                    # convention).
                    p = v / (v.sum() if v.sum() > 0 else 1.0)
                    mask = p > 0
                    val = float(-np.sum(p[mask] * np.log(p[mask])))
                    # Log-2 equivalent (bits) for convenience.
                    val_bits = val / np.log(2)
                    input_latex = f"H({_fmt_vec(p)})"
                    result_latex = f"{val:.6g}"
                    plain = (f"Shannon entropy = {val:.6g} nats "
                             f"({val_bits:.6g} bits), P={_fmt_vec(p)}")

        else:
            return {"error": f"unknown op '{op}'"}

        return {
            "op": op_lc,
            "input_latex": input_latex,
            "result_latex": result_latex,
            "plain": plain,
            "error": None,
        }

    try:
        return _run_with_timeout(_do_math, timeout_sec=10)

    except _MathTimeout as e:
        return {"error": str(e)}
    except (sp.SympifyError, SyntaxError) as e:
        return {"error": f"couldn't parse '{expr_s}': {e}"}
    except (ValueError, TypeError) as e:
        return {"error": str(e)}
    except Exception as e:
        log.error("math op %s failed: %s", op, e)
        return {"error": f"{op}: {e}"}


def _math(
    op: str,
    expression: str,
    variable: str = "x",
    transform_var: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
) -> str:
    """LLM tool: plain-text summary of a math op. The widget uses the
    structured form via /api/math."""
    data = math_data(op, expression, variable, transform_var, lower, upper)
    if data.get("error"):
        return f"Error: {data['error']}"
    return data["plain"]


# ---- Symbolic math (SymPy) -----------------------------------------------
