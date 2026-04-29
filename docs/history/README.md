# Historical engineering notes

This folder contains dated training audits and postmortems. These files
explain why the current routing stack looks the way it does, but they are
not setup instructions and they do not describe the default runtime path.

Use the current docs first:

- [../system-overview.md](../system-overview.md) for the system map.
- [../routing-pipeline.md](../routing-pipeline.md) for the current routing
  pipeline and the major decisions that led to it.
- [../../sft/README.md](../../sft/README.md) for the active SFT workflow.

## Files

| File | Purpose |
|---|---|
| [routing-evolution.md](routing-evolution.md) | Historical routing iterations, backend experiments, runtime rescue changes, and archived next-step notes. |
| [lora-training-audit-2026-04-21.md](lora-training-audit-2026-04-21.md) | Audit of the LoRA training setup and known risks. |
| [iter5-postmortem-2026-04-22.md](iter5-postmortem-2026-04-22.md) | Why iter-5 was not shipped. |
| [iter6-postmortem-2026-04-22.md](iter6-postmortem-2026-04-22.md) | Why iter-6 was not shipped. |
| [iter7-postmortem-2026-04-23.md](iter7-postmortem-2026-04-23.md) | Why the targeted DPO attempt was not shipped. |
