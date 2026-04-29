"""AlertEngine: run rules over signals, respect cooldowns, persist outcomes.

The engine is the only component that knows about all three of:
  - the rule registry
  - the cooldown ledger
  - the alert store

Rules don't know about cooldowns. Cooldowns don't know about rules.
They're composed here.
"""
from __future__ import annotations

import logging
from datetime import datetime

from bridge.alerts.cooldown import CooldownLedger
from bridge.alerts.models import (
    Alert,
    ScanResult,
    Signal,
    SuppressedAlertRecord,
)
from bridge.alerts.rules import AlertRule
from bridge.alerts.store import AlertStore

log = logging.getLogger("bridge.alerts.engine")


class AlertEngine:
    """Pure orchestration. Doesn't fetch signals (that's service.py's job);
    just consumes them."""

    def __init__(
        self,
        *,
        rules: list[AlertRule],
        cooldown_ledger: CooldownLedger,
        store: AlertStore,
    ) -> None:
        self._rules = rules
        self._cooldown_ledger = cooldown_ledger
        self._store = store

    def run(self, signals: list[Signal], now: datetime) -> ScanResult:
        """Evaluate every rule against `signals`. Persist fires and
        suppressions. Return a structured ScanResult.

        Rules are run in registration order, which only matters for log
        readability (cooldowns are scoped per-rule, so ordering doesn't
        affect firing decisions).
        """
        fired: list[Alert] = []
        suppressed: list[SuppressedAlertRecord] = []
        for rule in self._rules:
            try:
                candidates = rule.evaluate(signals, now)
            except Exception as e:
                # Defensive: a single broken rule must not halt the scan.
                log.exception("rule %s crashed during evaluate: %s", rule.id, e)
                self._store.append_event(
                    "rule_crashed",
                    {"rule_id": rule.id, "error": str(e)},
                )
                continue
            for candidate in candidates:
                if self._cooldown_ledger.is_in_cooldown(
                    rule.id, candidate.scope_key, rule.cooldown_hours, now,
                ):
                    last = self._cooldown_ledger.last_fired(
                        rule.id, candidate.scope_key,
                    ) or now
                    record = SuppressedAlertRecord(
                        rule_id=rule.id,
                        scope_key=candidate.scope_key,
                        candidate_title=candidate.title,
                        last_fired_at=last,
                        suppressed_at=now,
                    )
                    self._store.append_suppression(record)
                    suppressed.append(record)
                    continue
                self._store.append_alert(candidate)
                self._cooldown_ledger.mark_fired(
                    rule.id, candidate.scope_key, now,
                )
                fired.append(candidate)
        return ScanResult(
            alerts=fired,
            suppressed=suppressed,
            signals_considered=len(signals),
            skipped_due_to_cache=False,
        )
