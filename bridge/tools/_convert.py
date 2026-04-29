"""Unit conversion tool (pint)."""
from __future__ import annotations

import logging

log = logging.getLogger("bridge.tools")


# ---- Unit conversion (pint) ----------------------------------------------

# One shared UnitRegistry — first use takes ~50 ms to load the defaults,
# subsequent conversions are fast. The registry is stateless for reads.
_unit_registry = None


def _get_unit_registry():
    global _unit_registry
    if _unit_registry is None:
        try:
            import pint
            _unit_registry = pint.UnitRegistry()
        except ImportError:
            return None
    return _unit_registry


def convert_data(value: str, from_unit: str, to_unit: str) -> dict:
    """Structured unit conversion for the tool + any future widget."""
    ureg = _get_unit_registry()
    if ureg is None:
        return {"error": "pint not installed"}
    # Parse the magnitude separately from the unit so pint treats
    # temperature as absolute (not a delta), which is what users mean
    # with "100 celsius -> fahrenheit".
    try:
        mag_in = float(str(value).strip())
    except ValueError:
        return {"error": f"value must be numeric, got '{value}'"}
    try:
        qty = ureg.Quantity(mag_in, str(from_unit).strip())
    except Exception as e:
        return {"error": f"couldn't parse unit '{from_unit}': {e}"}
    try:
        converted = qty.to(str(to_unit).strip())
    except Exception as e:
        return {"error": f"can't convert to {to_unit}: {e}"}
    mag = float(converted.magnitude)
    plain = f"{value} {from_unit} = {mag:.6g} {to_unit}"
    return {
        "value": float(qty.magnitude),
        "from_unit": str(qty.units),
        "to_unit": str(converted.units),
        "magnitude": mag,
        "plain": plain,
        "error": None,
    }


def _convert(value: str, from_unit: str, to_unit: str) -> str:
    data = convert_data(value, from_unit, to_unit)
    if data.get("error"):
        return f"Error: {data['error']}"
    return data["plain"]


