"""Rule-based alert subsystem.

Consumes outputs from bridge.trackers and bridge.news plus one external
feed (State Dept travel advisories). Emits structured Alert objects
via deterministic rules with per-rule cooldowns. No LLM in the decision
path — the LLM only narrates alerts already generated here.

One-way dependency: alerts → trackers, alerts → news. Never the reverse.

Layered pipeline:
  detectors  — convert subsystem outputs into typed Signals
  rules      — evaluate Signals against explicit thresholds, produce Alerts
  cooldowns  — per-rule, per-scope-key suppression
  engine     — runs rules, applies cooldowns, persists outcomes
  service    — orchestrator + TTL gate on full scan
"""
