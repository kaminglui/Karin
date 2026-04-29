"""Structured tracker subsystem for numeric benchmarks.

V1: FX (Frankfurter), gold (Stooq), US food CPI (BLS). JSON persistence,
lazy-refresh TTL gate, deterministic delta computation.

Deliberately parallel to bridge/news — no shared state, no cross-imports.
Trackers live on a different cadence (daily to monthly) and answer
different questions ("what's the number now / vs. last week?") than
news clustering ("what happened today?").
"""
