"""Smoke tests for the tools added after the initial voice-loop build:

- ``math``     (SymPy-backed: evaluate, solve, transforms, matrix ops, fft/ifft)
- ``convert``  (pint unit conversion)
- ``graph``    (sampled plot data for the widget)
- ``circuit``  (analog + digital, including the synthesize / safe bool evaluator)

These are smoke tests — one happy path + one failure mode per operation —
not exhaustive symbolic-math correctness tests. Their job is to catch
regressions when someone changes the dispatch table, the arg plumbing,
or the tool-result shape.
"""
from __future__ import annotations

import math

import pytest

from bridge import tools


# --- math --------------------------------------------------------------------

class TestMathTool:
    def test_evaluate(self):
        d = tools.math_data("evaluate", "2**10")
        assert d["error"] is None
        assert "1024" in d["plain"]

    def test_solve_equation(self):
        d = tools.math_data("solve", "x**2 - 4 = 0", variable="x")
        assert d["error"] is None
        # Roots should be -2 and 2, order not guaranteed.
        assert "-2" in d["plain"] and "2" in d["plain"]

    def test_integrate_definite(self):
        d = tools.math_data("integrate", "x**2", variable="x", lower="0", upper="1")
        assert d["error"] is None
        assert "1/3" in d["plain"] or "0.333" in d["plain"]

    def test_differentiate(self):
        d = tools.math_data("differentiate", "x**3", variable="x")
        assert d["error"] is None
        assert "3*x**2" in d["plain"]

    def test_rref(self):
        d = tools.math_data("rref", "1 2 3; 4 5 6; 7 8 9")
        assert d["error"] is None
        assert "pivots" in d["plain"]

    def test_inverse(self):
        d = tools.math_data("inverse", "[[1,2],[3,4]]")
        assert d["error"] is None
        assert "-2" in d["plain"]

    def test_laplace(self):
        d = tools.math_data("laplace", "exp(-a*t)", variable="t", transform_var="s")
        assert d["error"] is None
        assert "a + s" in d["plain"]

    def test_inverse_laplace(self):
        d = tools.math_data("inverse_laplace", "1/(s**2 + 1)", variable="s", transform_var="t")
        assert d["error"] is None
        assert "sin(t)" in d["plain"]

    def test_fft_cosine(self):
        # A 4-point sample of a cos wave at k=1: energy concentrated
        # at bins ±1. Real DFT returns symmetric spectrum.
        d = tools.math_data("fft", "1, 0, -1, 0")
        assert d["error"] is None
        mags = d["magnitudes"]
        assert abs(mags[1] - 2.0) < 1e-9 and abs(mags[3] - 2.0) < 1e-9
        assert mags[0] < 1e-9 and mags[2] < 1e-9

    def test_unknown_op(self):
        d = tools.math_data("foo", "x")
        assert d.get("error")


# --- convert -----------------------------------------------------------------

class TestConvertTool:
    def test_length(self):
        d = tools.convert_data("5", "mile", "km")
        assert d["error"] is None
        assert abs(d["magnitude"] - 8.04672) < 1e-3

    def test_absolute_temperature(self):
        # Must be absolute, not delta — 100°C = 212°F.
        d = tools.convert_data("100", "celsius", "fahrenheit")
        assert d["error"] is None
        assert abs(d["magnitude"] - 212.0) < 1e-6

    def test_incompatible_units(self):
        d = tools.convert_data("5", "mile", "kg")
        assert d.get("error")

    def test_nonnumeric_value(self):
        d = tools.convert_data("hello", "meter", "km")
        assert d.get("error")


# --- graph -------------------------------------------------------------------

class TestGraphTool:
    def test_single_curve(self):
        # graph_data clamps num_samples to at least 50 (too few samples
        # produces ugly plots), so explicit smaller counts get bumped.
        d = tools.graph_data("sin(x)", "x", -3.14, 3.14, 50)
        assert d.get("error") is None
        assert len(d["x"]) == 50
        assert len(d["series"]) == 1
        assert d["series"][0]["name"] == "sin(x)"
        assert len(d["series"][0]["y"]) == 50

    def test_multiple_curves(self):
        d = tools.graph_data("sin(x); cos(x)", "x", 0, 6.28, 5)
        assert d.get("error") is None
        assert len(d["series"]) == 2

    def test_bad_range(self):
        d = tools.graph_data("x", "x", 1, 1)  # x_max not > x_min
        assert d.get("error")

    def test_divide_by_zero_samples_as_nulls(self):
        # linspace(-1, 1, 51) lands exactly on 0 at the midpoint, so
        # 1/x there is ±inf -> rendered as None in the JSON so Plotly
        # draws a gap instead of a spike to infinity.
        d = tools.graph_data("1/x", "x", -1, 1, 51)
        assert d.get("error") is None
        ys = d["series"][0]["y"]
        # The middle sample (index 25) is exactly at x=0.
        assert ys[25] is None


# --- circuit -----------------------------------------------------------------

class TestCircuitTool:
    def test_resistance_parallel(self):
        d = tools.circuit_data("resistance_parallel", values="100, 220, 330")
        assert d["error"] is None
        # Parallel combo of 100, 220, 330 ≈ 56.9 Ω
        assert abs(d["result"]["resistance_ohms"] - 56.96) < 0.1

    def test_rc_cutoff(self):
        d = tools.circuit_data("rc_cutoff", R="10k", C="100n")
        assert d["error"] is None
        # fc = 1/(2π·10e3·100e-9) ≈ 159.15 Hz
        assert abs(d["result"]["fc_hz"] - 159.15) < 0.5

    def test_lc_resonance(self):
        d = tools.circuit_data("lc_resonance", L="10u", C="1n")
        assert d["error"] is None
        # f0 = 1/(2π√(10e-6·1e-9)) ≈ 1.59 MHz
        assert abs(d["result"]["f0_hz"] - 1.5915e6) < 1e4

    def test_impedance_capacitor(self):
        d = tools.circuit_data(
            "impedance", component="C", value="100n", frequency="1k",
        )
        assert d["error"] is None
        # |Z_C| at 1 kHz with 100 nF = 1 / (2π·1000·100e-9) ≈ 1591.55 Ω
        assert abs(d["result"]["magnitude"] - 1591.55) < 1.0
        # Pure capacitor: phase = -90°
        assert abs(d["result"]["phase_deg"] - (-90.0)) < 0.01

    def test_voltage_divider(self):
        d = tools.circuit_data("voltage_divider", R1="10k", R2="10k", Vin="5")
        assert d["error"] is None
        assert abs(d["result"]["vout_v"] - 2.5) < 1e-9

    def test_logic_eval(self):
        d = tools.circuit_data(
            "logic_eval", expression="(A and B) or not C",
            inputs="A=1,B=0,C=1",
        )
        assert d["error"] is None
        assert d["result"]["value"] == 0

    def test_truth_table_xor(self):
        d = tools.circuit_data("truth_table", expression="A xor B")
        assert d["error"] is None
        rows = d["result"]["rows"]
        assert len(rows) == 4
        # XOR: (0,0)=0, (0,1)=1, (1,0)=1, (1,1)=0
        outs = {(r["A"], r["B"]): r["out"] for r in rows}
        assert outs == {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}

    def test_logic_eval_rejects_code_injection(self):
        # Attempted sandbox escape via builtins / dunder access —
        # the AST walker must refuse these.
        for attack in [
            "__import__('os')",
            "A.__class__",
            "A + 1",           # BinOp — not a boolean op
            "A and __builtins__",  # Name that isn't a declared input
        ]:
            d = tools.circuit_data(
                "logic_eval", expression=attack, inputs="A=1",
            )
            assert d.get("error"), f"should reject: {attack!r}"

    def test_synthesize_xor(self):
        # XOR: true when exactly one input is 1 -> minterms 1, 2
        d = tools.circuit_data(
            "synthesize", variables="A,B", minterms="1,2",
        )
        assert d["error"] is None
        sop = d["result"]["sop"]
        # (A & ~B) | (~A & B) — exact form can vary, but both terms
        # should appear.
        assert "~A" in sop or "~B" in sop

    def test_synthesize_out_of_range(self):
        # 2 variables -> minterm indices must be 0..3.
        d = tools.circuit_data(
            "synthesize", variables="A,B", minterms="1,7",
        )
        assert d.get("error")

    def test_unknown_op(self):
        d = tools.circuit_data("no_such_op")
        assert d.get("error")


# --- dispatch integration ----------------------------------------------------

class TestDispatchIntegration:
    """Sanity-check that new tools are reachable through tools.execute()
    (the entry point the LLM actually hits)."""

    def test_math_via_execute(self):
        result = tools.execute("math", {"op": "evaluate", "expression": "2+2"})
        assert "4" in result

    def test_convert_via_execute(self):
        result = tools.execute(
            "convert",
            {"value": "1", "from_unit": "meter", "to_unit": "cm"},
        )
        assert "100" in result

    def test_circuit_via_execute(self):
        result = tools.execute(
            "circuit",
            {"op": "resistance_series", "values": "10,20,30"},
        )
        assert "60" in result

    def test_graph_via_execute(self):
        result = tools.execute(
            "graph",
            {"expression": "x", "variable": "x", "x_min": 0, "x_max": 1},
        )
        assert "x" in result and "Plotted" in result

    def test_unknown_tool(self):
        # Unknown tool names now come back as a bracketed friendly
        # directive the LLM can gracefully paraphrase, instead of a raw
        # "Error:" the model sometimes parrots verbatim.
        result = tools.execute("no_such_tool", {})
        assert result.startswith("[unknown tool")
