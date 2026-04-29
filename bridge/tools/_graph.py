"""Graph/plotting tool."""
from __future__ import annotations

import logging
import re

log = logging.getLogger("bridge.tools")


# ---- Preset expressions ---------------------------------------------------
#
# Named distributions / canonical functions users ask for by name ("plot
# Gaussian"). Without these, the LoRA has to invent the formula AND pick
# a sensible x-range, which it's unreliable at — we've seen Gaussians
# emitted as `exp(-0.5*(x-3)**2)/sqrt(2*pi)` (centered off-origin) or
# with x ∈ [-1, 1] (missing most of the bell). Presets give a canonical
# form + a sensible default range.
#
# Matching is keyword-based on the user-provided expression string:
# if ``expression`` (lowercased, stripped of common filler words)
# matches any key, we substitute. Otherwise we fall through to the
# normal sympy parser.
_PRESETS: dict[str, tuple[str, float, float]] = {
    # key                expression (uses `**`)            x_min  x_max
    # --- Continuous distributions (normalised PDFs) ---
    "gaussian":         ("exp(-x**2/2)/sqrt(2*pi)",        -4.0,  4.0),
    "standard normal":  ("exp(-x**2/2)/sqrt(2*pi)",        -4.0,  4.0),
    "normal":           ("exp(-x**2/2)/sqrt(2*pi)",        -4.0,  4.0),
    "bell curve":       ("exp(-x**2/2)/sqrt(2*pi)",        -4.0,  4.0),
    # Beta(2, 5): PDF = x*(1-x)**4 / B(2,5) = 30·x·(1-x)**4 on [0,1]
    "beta":             ("30*x*(1-x)**4",                   0.0,  1.0),
    # Gamma(k=2, θ=2): PDF = x·exp(-x/2) / (θ^k · Γ(k)) = x·exp(-x/2)/4
    "gamma":            ("x*exp(-x/2)/4",                   0.0, 15.0),
    "gamma distribution":("x*exp(-x/2)/4",                  0.0, 15.0),
    # Chi-squared with k=3 d.f.
    "chi squared":      ("sqrt(x)*exp(-x/2)/(sqrt(2*pi))",  0.0, 12.0),
    "chi-squared":      ("sqrt(x)*exp(-x/2)/(sqrt(2*pi))",  0.0, 12.0),
    "chisq":            ("sqrt(x)*exp(-x/2)/(sqrt(2*pi))",  0.0, 12.0),
    # Student-t with ν=5 d.f. (unnormalised shape)
    "student t":        ("(1 + x**2/5)**(-3)",             -5.0,  5.0),
    "student-t":        ("(1 + x**2/5)**(-3)",             -5.0,  5.0),
    "t distribution":   ("(1 + x**2/5)**(-3)",             -5.0,  5.0),
    # Log-normal(μ=0, σ=1)
    "log normal":       ("exp(-log(x)**2/2)/(x*sqrt(2*pi))", 0.01, 5.0),
    "lognormal":        ("exp(-log(x)**2/2)/(x*sqrt(2*pi))", 0.01, 5.0),
    # Note: Poisson / Binomial are DISCRETE distributions — they're
    # PMFs on integers, not continuous curves. The current grapher
    # only plots continuous functions. To compute PMF values, use
    # the math tool:
    #   Poisson(λ=3) at k=2:    op=evaluate, expression='3**2 * exp(-3) / factorial(2)'
    #   Binomial(n=10, p=0.5):  op=evaluate, expression='binomial(10,k) * 0.5**10'
    # --- Activations ---
    "sigmoid":          ("1/(1+exp(-x))",                  -6.0,  6.0),
    "tanh":             ("tanh(x)",                        -4.0,  4.0),
    "relu":             ("Max(0, x)",                      -5.0,  5.0),
    "softplus":         ("log(1+exp(x))",                  -5.0,  5.0),
    # --- Simple distributions ---
    "exponential":      ("exp(-x)",                         0.0,  5.0),
    "laplace":          ("exp(-abs(x))/2",                 -5.0,  5.0),
    "cauchy":           ("1/(pi*(1+x**2))",                -6.0,  6.0),
    "logistic":         ("exp(-x)/(1+exp(-x))**2",         -6.0,  6.0),
    "uniform":          ("1",                               0.0,  1.0),
    # --- Trig / polynomials ---
    "sine":             ("sin(x)",                          0.0,  6.283185307),
    "sin":              ("sin(x)",                          0.0,  6.283185307),
    "cosine":           ("cos(x)",                          0.0,  6.283185307),
    "cos":              ("cos(x)",                          0.0,  6.283185307),
    "parabola":         ("x**2",                           -5.0,  5.0),
    "cubic":            ("x**3",                           -3.0,  3.0),
}

# Filler words stripped as WHOLE WORDS before preset lookup so "plot
# Gaussian distribution" and "the Gaussian" both match "gaussian". Word-
# boundary matching is required — earlier a substring `an` was eating
# the interior of "gaussian" and killing the match.
_PRESET_FILLER_WORDS = (
    "plot", "graph", "draw", "show",
    "the", "a", "an",
    "distribution", "curve", "function",
    "standard", "unit",
)
_PRESET_FILLER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _PRESET_FILLER_WORDS) + r")\b",
    re.IGNORECASE,
)


def _resolve_preset(expression: str) -> tuple[str, float, float] | None:
    """Map a user-supplied name like 'Gaussian distribution' to a
    canonical (expression, x_min, x_max) triple, or None if no match.
    """
    if not expression:
        return None
    s = expression.strip().lower()
    s = _PRESET_FILLER_RE.sub(" ", s)
    s = " ".join(s.split())
    if not s:
        return None
    if s in _PRESETS:
        return _PRESETS[s]
    # Prefix match for qualified names like "gaussian blah".
    for key, val in _PRESETS.items():
        if s == key or s.startswith(key + " "):
            return val
    return None


# ---- Graph (sampled plot data) --------------------------------------------

def graph_data(
    expression: str,
    variable: str = "x",
    x_min: float = -10.0,
    x_max: float = 10.0,
    num_samples: int = 400,
) -> dict:
    """Sample an expression (or ';'-separated list) for plotting.

    Returns a dict with {x, series: [{name, y}], variable, error}.
    One entry per sub-expression so the widget can render multiple
    curves in one axis.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor,
    )
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not installed"}

    expr_s = (expression or "").strip()
    if not expr_s:
        return {"error": "empty expression"}
    if len(expr_s) > 600:
        return {"error": "expression too long"}

    # Preset lookup: if the expression names a well-known curve/
    # distribution ("gaussian", "sigmoid", "tanh", …) substitute the
    # canonical formula + a sensible x-range. Only overrides x_min/x_max
    # when the caller left them at the default (-10, 10) so an explicit
    # user-provided range still wins.
    preset = _resolve_preset(expr_s)
    preset_name = None
    if preset is not None:
        canonical, p_xmin, p_xmax = preset
        preset_name = expr_s
        expr_s = canonical
        if x_min == -10.0 and x_max == 10.0:
            x_min, x_max = p_xmin, p_xmax

    try:
        xmin = float(x_min)
        xmax = float(x_max)
    except (TypeError, ValueError):
        return {"error": "x_min and x_max must be numbers"}
    if xmax <= xmin:
        return {"error": "x_max must be greater than x_min"}

    n = max(50, min(int(num_samples) if num_samples else 400, 2000))
    var_sym = sp.Symbol(variable or "x")
    # convert_xor maps `^` → `**` during parse so math-notation exponents
    # like `x^2` don't blow up (sympy defaults to Python-XOR semantics
    # for `^`, which fails with a TypeError on expressions that naturally
    # use `^`, e.g. `exp(-x^2 / 2)` emitted by the LoRA).
    transformations = standard_transformations + (
        implicit_multiplication_application, convert_xor,
    )
    x_samples = np.linspace(xmin, xmax, n)

    series: list[dict] = []
    for piece in [p.strip() for p in expr_s.split(";") if p.strip()]:
        try:
            expr = parse_expr(piece, transformations=transformations,
                              local_dict={variable or "x": var_sym})
            fn = sp.lambdify(var_sym, expr, modules=["numpy"])
            y = fn(x_samples)
            # Coerce scalars (e.g. the expression is a constant) to an array.
            if np.ndim(y) == 0:
                y = np.full_like(x_samples, float(y))
            # Replace non-finite values with None so JSON is valid.
            y_list: list[float | None] = []
            for v in np.asarray(y, dtype=float).tolist():
                if v is None or not np.isfinite(v):
                    y_list.append(None)
                else:
                    y_list.append(v)
            series.append({"name": piece, "y": y_list})
        except Exception as e:
            series.append({"name": piece, "y": [], "error": str(e)})

    result = {
        "variable": variable or "x",
        "x_min": xmin,
        "x_max": xmax,
        "x": x_samples.tolist(),
        "series": series,
    }
    if preset_name is not None:
        # Widget header can show "Gaussian — exp(-x**2/2)/sqrt(2*pi)".
        result["preset"] = preset_name
    return result


def _graph(
    expression: str,
    variable: str = "x",
    x_min: float | None = None,
    x_max: float | None = None,
) -> str:
    """LLM tool: short text summary. The widget renders the plot."""
    data = graph_data(
        expression, variable,
        x_min if x_min is not None else -10.0,
        x_max if x_max is not None else 10.0,
    )
    if data.get("error"):
        return f"Error: {data['error']}"
    names = ", ".join(s["name"] for s in data["series"])
    return (
        f"Plotted {names} over {data['variable']} \u2208 "
        f"[{data['x_min']:g}, {data['x_max']:g}]."
    )

