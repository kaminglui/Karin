"""Snapshot -> short human-readable string.

Pure presentation layer. No service access, no store access, no I/O.
Takes TrackerSnapshot objects and produces strings suitable for the
LLM to paraphrase (single tracker) or the LLM to read verbatim (multi).

Design choices:

- Category-driven unit suffix (metal -> "USD/oz", everything else "").
  Kept here rather than on the TrackerSnapshot so Phase 5.1 doesn't
  touch the data model.

- Single delta per snapshot, priority 1m > 1w > 1d. Picks the longest
  informative window that's available. Monthly series naturally get
  1m; freshly-seeded daily series degrade gracefully as history grows.

- 4 decimal places for values across all categories. FX and gold are
  user-specified; CPI values have enough magnitude that 4 d.p. reads
  naturally too.

- Rounded percentage delta of exactly 0.00 renders as "Unchanged" to
  avoid the "Up 0.00%" / "Down 0.00%" uncanny-valley wording.
"""
from __future__ import annotations

from bridge.trackers.models import TrackerSnapshot

# Category -> unit suffix shown after the numeric value. Centralized
# here so the tracker data model stays untouched. Extend as new
# categories are added.
_UNIT_BY_CATEGORY: dict[str, str] = {
    "fx": "",
    "metal": "USD/oz",
    "food_index": "",
    "energy": "USD/gal",   # RBOB gasoline futures quoted in $/gallon
}

# Category -> cadence, used to pick the honest "as of" vs "latest close"
# wording for the date line on single-tracker voice output. Daily market
# data is always a close snapshot (not live intraday), so the "latest
# close" label is more accurate than "as of" for fx/metal/energy.
# Monthly series (CPI) are published once per month for a reference
# period, so "as of" reads correctly there.
_CADENCE_BY_CATEGORY: dict[str, str] = {
    "fx": "daily",
    "metal": "daily",
    "energy": "daily",
    "food_index": "monthly",
}


def format_tracker_voice(snapshot: TrackerSnapshot) -> str:
    """Full one-sentence summary for single-tracker voice output.

    Example outputs:
      "USD/CNY is 6.8284 as of 2026-04-10. Up 0.12% vs 1 week ago."
      "Gold is 4759.5950 USD/oz as of 2026-04-10. Up 3.42% vs 1 day ago. Surging."
      "USD/JPY is 159.1900 as of 2026-04-10. Down 0.05% vs 1 day ago. Volatile."
      "US CPI: Food is 346.7960 as of 2026-03-31. This reading is stale."

    Phase 5.2: appends AT MOST ONE flag sentence from the derived
    labels. Direction is omitted deliberately — the delta line already
    conveys direction. Shock beats movement; "moving"/"stable"/"flat"
    are considered too boring to merit extra text.
    """
    if snapshot.latest_value is None:
        # Phase-7 polish: parenthesize the note so we don't get an
        # awkward double colon ("USD/CNY: frankfurter: status 503.").
        if snapshot.note:
            return f"{snapshot.label}: no data ({snapshot.note})."
        return f"{snapshot.label}: no data yet."

    value_str = _format_value(snapshot.latest_value)
    unit = _UNIT_BY_CATEGORY.get(snapshot.category, "")
    unit_part = f" {unit}" if unit else ""
    date_str = _format_date(snapshot.latest_timestamp)
    # Daily market data: "latest close 2026-04-10" (honest — Stooq/Frankfurter
    # only publish closes, never intraday). Monthly CPI: keep "as of".
    cadence = _CADENCE_BY_CATEGORY.get(snapshot.category, "daily")
    date_prefix = "latest close" if cadence == "daily" else "as of"

    parts = [f"{snapshot.label} is {value_str}{unit_part} {date_prefix} {date_str}."]
    if snapshot.is_stale:
        parts.append("This reading is stale.")
    delta_line = _pick_delta_line(snapshot)
    if delta_line:
        parts.append(delta_line)
    flag = _extract_flag_sentence(snapshot)
    if flag:
        parts.append(flag)
    return " ".join(parts)


def format_tracker_summary_line(snapshot: TrackerSnapshot) -> str:
    """Compact one-liner for multi-tracker listings.

    Example:
      "USD/CNY: 6.8284 (2026-04-10, fresh)"
      "Gold: 4759.5950 USD/oz (2026-04-10, fresh, surging)"
      "USD/JPY: 159.1900 (2026-04-10, fresh, up)"

    Phase 5.2: appends AT MOST ONE short tag inside the parens,
    chosen by priority (shock > volatile > up/down). `flat`/`stable`/
    `moving` don't produce a tag — they're the normal state.
    """
    if snapshot.latest_value is None:
        # Same parenthesize-the-note polish as the single-tracker form.
        if snapshot.note:
            return f"{snapshot.label}: no data ({snapshot.note})"
        return f"{snapshot.label}: no data yet"

    value_str = _format_value(snapshot.latest_value)
    unit = _UNIT_BY_CATEGORY.get(snapshot.category, "")
    unit_part = f" {unit}" if unit else ""
    date_str = _format_date(snapshot.latest_timestamp)
    freshness = "stale" if snapshot.is_stale else "fresh"
    tag = _extract_tag(snapshot)
    tag_part = f", {tag}" if tag else ""
    return f"{snapshot.label}: {value_str}{unit_part} ({date_str}, {freshness}{tag_part})"


def format_trackers_voice(snapshots: list[TrackerSnapshot]) -> str:
    """Multi-tracker output. One line per snapshot, newline-joined.

    Returns a neutral message for an empty input so callers don't need
    to special-case the "no data" path before formatting.
    """
    if not snapshots:
        return "No trackers available."
    return "\n".join(format_tracker_summary_line(s) for s in snapshots)


# --- helpers --------------------------------------------------------------

def _format_value(value: float) -> str:
    return f"{value:.4f}"


def _format_date(ts) -> str:
    return ts.strftime("%Y-%m-%d") if ts is not None else "unknown"


def _pick_delta_line(snapshot: TrackerSnapshot) -> str | None:
    """Return the sentence for the longest available delta, or None.

    Priority: 1m > 1w > 1d. First non-null delta wins.
    """
    candidates: list[tuple[float | None, str]] = [
        (snapshot.change_1m_pct, "1 month ago"),
        (snapshot.change_1w_pct, "1 week ago"),
        (snapshot.change_1d_pct, "1 day ago"),
    ]
    for pct, phrase in candidates:
        if pct is None:
            continue
        return _format_delta_phrase(pct, phrase)
    return None


def _format_delta_phrase(pct: float, phrase: str) -> str:
    if round(pct, 2) == 0.0:
        return f"Unchanged vs {phrase}."
    direction = "Up" if pct > 0 else "Down"
    return f"{direction} {abs(pct):.2f}% vs {phrase}."


# --- Phase 5.2 derived-label rendering ------------------------------------

def _extract_flag_sentence(snapshot: TrackerSnapshot) -> str | None:
    """Pick the strongest derived label worth rendering as its own sentence.

    Priority: shock_label > volatile movement. Other labels (direction,
    stable/moving) are NOT rendered here — direction is already in the
    delta line, and 'stable'/'moving' aren't interesting enough to
    warrant extra text per user's "only when it improves readability"
    guidance.
    """
    if snapshot.shock_label:
        return f"{snapshot.shock_label.capitalize()}."
    if snapshot.movement_label == "volatile":
        return "Volatile."
    return None


def _extract_tag(snapshot: TrackerSnapshot) -> str | None:
    """Pick the single most informative tag for a compact listing line.

    Priority: shock > volatile > direction_1d if up/down (not flat).
    Returns None when there's nothing interesting — the line stays short.
    """
    if snapshot.shock_label:
        return snapshot.shock_label
    if snapshot.movement_label == "volatile":
        return "volatile"
    if snapshot.direction_1d in ("up", "down"):
        return snapshot.direction_1d
    return None
