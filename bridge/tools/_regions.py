"""Shared region/year helpers for the data tools.

The data tools (`inflation`, `population`, `facts`, `alice`,
`county_metrics`, `analyze`) each carry their own per-region config
(file paths, supports flags, currency, etc.) because the *data* differs
per tool — alice is US-only, inflation has 5 regions but no world,
population has 6 including world, county_metrics is US-county-keyed.

What's *not* tool-specific is the year-resolution policy: pick the
requested year if it's available, else the closest year ≤ requested,
else the latest year. That helper used to be reimplemented in 4 places.

Region alias maps and natural-language detection patterns are NOT
unified here on purpose — they're intentionally divergent (e.g.,
inflation's classifier omits "world"/"global" because the tool has no
world series, so a "global inflation" query falls back to the US
default rather than failing). If you find yourself wanting to share
those, the right move is to filter the detection result against the
calling tool's `_REGIONS` keys, not to merge the maps.
"""
from __future__ import annotations


# Canonical region keys used by the data tools that DO support
# multi-region. Each tool's `_REGIONS` dict is a (possibly proper)
# subset of this set.
REGION_LABELS: dict[str, str] = {
    "us": "United States",
    "world": "World",
    "hk_sar": "Hong Kong SAR, China",
    "cn_mainland": "China (Mainland)",
    "japan": "Japan",
    "south_korea": "South Korea",
}


def resolve_year(requested: int | None, available: list[int]) -> int | None:
    """Pick a year from ``available`` to honor ``requested``.

    Policy: exact match if present, else the closest year ≤ requested,
    else the latest available year. Returns ``None`` only when the
    available list is empty.

    Used by `inflation`, `population`, `facts`, `alice`, and
    `county_metrics` so a user prompt like "wages in 2017" gracefully
    degrades to 2016 when 2017 isn't in the dataset, rather than
    raising or returning the latest year out of the blue.
    """
    if not available:
        return None
    if requested is None:
        return max(available)
    if requested in available:
        return requested
    older = [y for y in available if y <= requested]
    return max(older) if older else max(available)
