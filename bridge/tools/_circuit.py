"""Circuit calculation tool (analog + digital)."""
from __future__ import annotations

import ast
import logging
import math
import re

log = logging.getLogger("bridge.tools")


# ---- Circuits (analog + digital) -----------------------------------------

def _parse_si(s: str) -> float:
    """Parse a value with optional SI prefix (k/M/G/m/u/µ/n/p).

    Raises ValueError on bad input.
    """
    if s is None:
        raise ValueError("missing value")
    t = str(s).strip().replace(",", "")
    if not t:
        raise ValueError("empty value")
    suffixes = {
        "p": 1e-12, "n": 1e-9, "u": 1e-6, "\u00b5": 1e-6, "\u03bc": 1e-6,
        "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9,
    }
    # Strip trailing unit letter if it's one of our known prefixes.
    mult = 1.0
    if t[-1] in suffixes:
        mult = suffixes[t[-1]]
        t = t[:-1]
    # Also accept forms like "1.5k", "100n", "47uF" (F is unit, ignored).
    t = t.rstrip("FHhΩ")  # Farads, Henries, Ohms — any leftover unit
    try:
        return float(t) * mult
    except ValueError:
        raise ValueError(f"couldn't parse value '{s}'")


def _fmt_eng(x: float, unit: str = "") -> str:
    """Format a float with an engineering SI prefix."""
    import math as _m
    if x == 0 or not _m.isfinite(x):
        return f"{x:g}{' ' + unit if unit else ''}"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    prefixes = [
        (1e-12, "p"), (1e-9, "n"), (1e-6, "\u00b5"), (1e-3, "m"),
        (1.0, ""), (1e3, "k"), (1e6, "M"), (1e9, "G"),
    ]
    chosen = prefixes[4]
    for threshold, p in prefixes:
        if ax >= threshold:
            chosen = (threshold, p)
    val = ax / chosen[0]
    return f"{sign}{val:.4g} {chosen[1]}{unit}".rstrip()


def _parse_values_list(values_str: str) -> list[float]:
    if not values_str:
        raise ValueError("missing values")
    parts = [p.strip() for p in str(values_str).split(",") if p.strip()]
    if not parts:
        raise ValueError("no values")
    return [_parse_si(p) for p in parts]


def circuit_data(
    op: str,
    *,
    values: str | None = None,
    R: str | None = None, L: str | None = None, C: str | None = None,
    R1: str | None = None, R2: str | None = None, Vin: str | None = None,
    component: str | None = None, value: str | None = None,
    frequency: str | None = None,
    expression: str | None = None, inputs: str | None = None,
    variables: str | None = None, minterms: str | None = None,
    dontcares: str | None = None,
) -> dict:
    """Structured backend for the circuit tool + widget.

    Returns {op, inputs: {...}, result: {...}, plain, error}.
    """
    import math as _m
    op_lc = (op or "").strip().lower()
    try:
        if op_lc in ("resistance_parallel", "resistance_series"):
            vs = _parse_values_list(values or "")
            if op_lc == "resistance_series":
                total = sum(vs)
                formula = " + ".join(_fmt_eng(v, "\u03a9") for v in vs)
            else:
                total = 1.0 / sum(1.0 / v for v in vs)
                formula = "1 / (" + " + ".join(f"1/{_fmt_eng(v,'Ω')}" for v in vs) + ")"
            plain = f"{op_lc.replace('_', ' ')}: {formula} = {_fmt_eng(total, 'Ω')}"
            return {"op": op_lc,
                    "inputs": {"values": [_fmt_eng(v, "Ω") for v in vs]},
                    "result": {"resistance_ohms": total,
                               "formatted": _fmt_eng(total, "Ω")},
                    "plain": plain, "error": None}

        if op_lc == "rc_time_constant":
            r = _parse_si(R or ""); c = _parse_si(C or "")
            tau = r * c
            return {"op": op_lc,
                    "inputs": {"R": _fmt_eng(r, "Ω"), "C": _fmt_eng(c, "F")},
                    "result": {"tau_s": tau, "formatted": _fmt_eng(tau, "s")},
                    "plain": f"\u03c4 = RC = {_fmt_eng(r,'Ω')} × {_fmt_eng(c,'F')} = {_fmt_eng(tau,'s')}",
                    "error": None}

        if op_lc == "rc_cutoff":
            r = _parse_si(R or ""); c = _parse_si(C or "")
            fc = 1.0 / (2.0 * _m.pi * r * c)
            return {"op": op_lc,
                    "inputs": {"R": _fmt_eng(r,"Ω"), "C": _fmt_eng(c,"F")},
                    "result": {"fc_hz": fc, "formatted": _fmt_eng(fc,"Hz")},
                    "plain": f"fc = 1/(2πRC) = {_fmt_eng(fc,'Hz')}",
                    "error": None}

        if op_lc == "rl_cutoff":
            r = _parse_si(R or ""); l = _parse_si(L or "")
            fc = r / (2.0 * _m.pi * l)
            return {"op": op_lc,
                    "inputs": {"R": _fmt_eng(r,"Ω"), "L": _fmt_eng(l,"H")},
                    "result": {"fc_hz": fc, "formatted": _fmt_eng(fc,"Hz")},
                    "plain": f"fc = R/(2πL) = {_fmt_eng(fc,'Hz')}",
                    "error": None}

        if op_lc == "lc_resonance":
            l = _parse_si(L or ""); c = _parse_si(C or "")
            f0 = 1.0 / (2.0 * _m.pi * _m.sqrt(l * c))
            return {"op": op_lc,
                    "inputs": {"L": _fmt_eng(l,"H"), "C": _fmt_eng(c,"F")},
                    "result": {"f0_hz": f0, "formatted": _fmt_eng(f0,"Hz")},
                    "plain": f"f0 = 1/(2π√LC) = {_fmt_eng(f0,'Hz')}",
                    "error": None}

        if op_lc == "impedance":
            kind = (component or "").strip().upper()
            val = _parse_si(value or "")
            f = _parse_si(frequency or "")
            omega = 2.0 * _m.pi * f
            if kind == "R":
                z = complex(val, 0.0); form = "Z_R = R"
            elif kind == "L":
                z = complex(0.0, omega * val); form = "Z_L = jωL"
            elif kind == "C":
                if val == 0:
                    return {"error": "C must be > 0"}
                z = complex(0.0, -1.0 / (omega * val)); form = "Z_C = 1/(jωC)"
            else:
                return {"error": "component must be R, L, or C"}
            mag = abs(z); phase = _m.degrees(_m.atan2(z.imag, z.real))
            return {"op": op_lc,
                    "inputs": {"component": kind, "value": _fmt_eng(val),
                               "frequency": _fmt_eng(f, "Hz")},
                    "result": {"real": z.real, "imag": z.imag,
                               "magnitude": mag, "phase_deg": phase,
                               "formatted": f"{_fmt_eng(mag,'Ω')} @ {phase:.2f}°"},
                    "plain": f"{form}: |Z| = {_fmt_eng(mag,'Ω')}, ∠{phase:.2f}°",
                    "error": None}

        if op_lc == "voltage_divider":
            r1 = _parse_si(R1 or ""); r2 = _parse_si(R2 or "")
            vin = _parse_si(Vin or "")
            vout = vin * r2 / (r1 + r2)
            return {"op": op_lc,
                    "inputs": {"R1": _fmt_eng(r1,"Ω"), "R2": _fmt_eng(r2,"Ω"),
                               "Vin": _fmt_eng(vin,"V")},
                    "result": {"vout_v": vout, "formatted": _fmt_eng(vout,"V"),
                               "ratio": r2/(r1+r2)},
                    "plain": f"Vout = Vin × R2/(R1+R2) = {_fmt_eng(vout,'V')}",
                    "error": None}

        if op_lc == "logic_eval":
            expr = (expression or "").strip()
            if not expr:
                return {"error": "missing expression"}
            ins = {}
            for pair in [p.strip() for p in (inputs or "").split(",") if p.strip()]:
                if "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                ins[k.strip()] = bool(int(v.strip()))
            py_expr = _logic_to_python(expr)
            result = _safe_bool_eval(py_expr, ins)
            return {"op": op_lc,
                    "inputs": {"expression": expr, **{k: int(v) for k, v in ins.items()}},
                    "result": {"value": int(result)},
                    "plain": f"{expr} = {int(result)}",
                    "error": None}

        if op_lc == "synthesize":
            var_names = [v.strip() for v in (variables or "").split(",") if v.strip()]
            if not var_names:
                return {"error": "missing 'variables' (e.g. 'A,B,C')"}
            if len(var_names) > 8:
                return {"error": "up to 8 variables"}
            try:
                mts = [int(m.strip()) for m in (minterms or "").split(",") if m.strip()]
            except ValueError:
                return {"error": "minterms must be integers"}
            try:
                dcs = [int(d.strip()) for d in (dontcares or "").split(",") if d.strip()]
            except ValueError:
                return {"error": "dontcares must be integers"}
            N = 1 << len(var_names)
            if any(m < 0 or m >= N for m in mts + dcs):
                return {"error": f"minterm/dontcare indices must be in [0,{N-1}]"}
            import sympy as sp
            from sympy.logic import SOPform, POSform
            syms = [sp.Symbol(v) for v in var_names]
            sop = SOPform(syms, mts, dcs) if mts or dcs else sp.false
            pos = POSform(syms, mts, dcs) if mts or dcs else sp.false

            def _count_gates(expr) -> int:
                if expr in (sp.true, sp.false):
                    return 0
                total = 0
                for node in sp.preorder_traversal(expr):
                    if isinstance(node, (sp.And, sp.Or, sp.Not, sp.Xor)):
                        total += 1
                return total

            sop_str = str(sop)
            pos_str = str(pos)
            plain = (
                f"SOP: {sop_str}\n"
                f"POS: {pos_str}\n"
                f"gates: SOP\u2248{_count_gates(sop)}, POS\u2248{_count_gates(pos)}"
            )
            return {"op": op_lc,
                    "inputs": {"variables": var_names,
                               "minterms": mts,
                               "dontcares": dcs},
                    "result": {"sop": sop_str, "pos": pos_str,
                               "sop_gates": _count_gates(sop),
                               "pos_gates": _count_gates(pos),
                               "sop_latex": sp.latex(sop),
                               "pos_latex": sp.latex(pos)},
                    "plain": plain, "error": None}

        if op_lc == "truth_table":
            expr = (expression or "").strip()
            if not expr:
                return {"error": "missing expression"}
            # Extract variables: uppercase single letters A-Z appearing outside keywords.
            vars_set: list[str] = []
            import re as _re
            for token in _re.findall(r"\b[A-Z]\b", expr):
                if token not in vars_set:
                    vars_set.append(token)
            if not vars_set:
                return {"error": "no variables found (use A, B, C, ...)"}
            if len(vars_set) > 5:
                return {"error": "up to 5 variables (32 rows max)"}
            py_expr = _logic_to_python(expr)
            rows: list[dict] = []
            for i in range(1 << len(vars_set)):
                env = {v: bool((i >> (len(vars_set) - 1 - j)) & 1)
                       for j, v in enumerate(vars_set)}
                out = _safe_bool_eval(py_expr, env)
                rows.append({**{k: int(v) for k, v in env.items()}, "out": int(out)})
            plain_rows = ["  ".join(f"{v}={r[v]}" for v in vars_set) + f"  |  out={r['out']}"
                          for r in rows]
            return {"op": op_lc,
                    "inputs": {"expression": expr, "variables": vars_set},
                    "result": {"rows": rows},
                    "plain": f"Truth table for {expr}:\n" + "\n".join(plain_rows),
                    "error": None}

        return {"error": f"unknown op '{op}'"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        log.error("circuit op %s failed: %s", op, e)
        return {"error": f"{op}: {e}"}


def _safe_bool_eval(py_expr: str, env: dict[str, bool]) -> bool:
    """Evaluate a boolean expression via AST walking only — no eval.

    Accepts the Python-ified form produced by ``_logic_to_python``
    (``and/or/not`` + ``!=`` for XOR + numeric constants) and walks
    the parsed tree with a fixed whitelist. Any construct outside the
    whitelist raises ValueError rather than reaching the interpreter,
    so a malicious ``expression`` argument can't touch builtins.
    """
    import ast as _ast
    try:
        tree = _ast.parse(py_expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"couldn't parse boolean expression: {e}") from None

    def _walk(node):
        if isinstance(node, _ast.Expression):
            return _walk(node.body)
        if isinstance(node, _ast.BoolOp):
            vals = [_walk(v) for v in node.values]
            if isinstance(node.op, _ast.And):
                return all(vals)
            if isinstance(node.op, _ast.Or):
                return any(vals)
            raise ValueError(f"unsupported boolop: {type(node.op).__name__}")
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.Not):
            return not _walk(node.operand)
        if isinstance(node, _ast.Compare):
            # Only != / == are meaningful for booleans (XOR / XNOR).
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("chained comparisons not allowed")
            left = _walk(node.left)
            right = _walk(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, _ast.NotEq):
                return bool(left) != bool(right)
            if isinstance(op, _ast.Eq):
                return bool(left) == bool(right)
            raise ValueError(f"unsupported compare op: {type(op).__name__}")
        if isinstance(node, _ast.Name):
            if node.id not in env:
                raise ValueError(f"unknown variable: {node.id!r}")
            return bool(env[node.id])
        if isinstance(node, _ast.Constant):
            if isinstance(node.value, (bool, int)):
                return bool(node.value)
            raise ValueError(f"unsupported constant: {node.value!r}")
        raise ValueError(f"unsupported expression: {type(node).__name__}")

    return bool(_walk(tree))


def _logic_to_python(expr: str) -> str:
    """Translate common boolean syntaxes to Python's and/or/not."""
    import re as _re
    out = expr
    # Unary NOT variants: !A, ~A, ¬A, NOT A
    out = _re.sub(r"!", " not ", out)
    out = _re.sub(r"~", " not ", out)
    out = _re.sub(r"\u00ac", " not ", out)
    out = _re.sub(r"\bNOT\b", " not ", out, flags=_re.IGNORECASE)
    # Binary AND: &, &&, ·, AND
    out = _re.sub(r"&&?", " and ", out)
    out = _re.sub(r"\u00b7", " and ", out)
    out = _re.sub(r"\bAND\b", " and ", out, flags=_re.IGNORECASE)
    # Binary OR: |, ||, +, OR
    out = _re.sub(r"\|\|?", " or ", out)
    out = _re.sub(r"\bOR\b", " or ", out, flags=_re.IGNORECASE)
    # XOR: ^, XOR -> !=
    out = _re.sub(r"\bXOR\b", " != ", out, flags=_re.IGNORECASE)
    out = _re.sub(r"\^", " != ", out)
    return out


def _circuit(
    op: str,
    values: str | None = None,
    R: str | None = None, L: str | None = None, C: str | None = None,
    R1: str | None = None, R2: str | None = None, Vin: str | None = None,
    component: str | None = None, value: str | None = None,
    frequency: str | None = None,
    expression: str | None = None, inputs: str | None = None,
    variables: str | None = None, minterms: str | None = None,
    dontcares: str | None = None,
) -> str:
    """LLM tool: short text summary."""
    data = circuit_data(
        op, values=values, R=R, L=L, C=C, R1=R1, R2=R2, Vin=Vin,
        component=component, value=value, frequency=frequency,
        expression=expression, inputs=inputs,
        variables=variables, minterms=minterms, dontcares=dontcares,
    )
    if data.get("error"):
        return f"Error: {data['error']}"
    return data["plain"]


