# Karin docs index

All supplementary documentation for the Karin project. Start with
[README.md](../README.md) for the product overview, [RUNBOOK.md](../RUNBOOK.md)
for live operations, and [system-overview.md](system-overview.md) for the
short technical map of how the codebase fits together.

## Getting started

Pick the doc that matches your situation:

- **[jetson-setup.md](jetson-setup.md)** — Flash JetPack 6 and
  provision a Jetson Orin Nano end-to-end. SDK Manager, Docker, GUI
  strip, Ollama tuning, repo + voice weights transfer, first bring-up,
  HTTPS via Tailscale. Start here if you have a Jetson in hand.
- **[pc-setup.md](pc-setup.md)** — Run the bridge on a PC (Windows /
  Linux) using host venvs instead of Docker. Mostly for development.
- **[pc-testing.md](pc-testing.md)** — Smoke-test the full voice loop
  on a PC before committing to a Jetson flash.
- **[training-guide.md](training-guide.md)** — Train your own voice
  clone (`gpt_model.ckpt` + `sovits_model.pth`) via the GPT-SoVITS
  Colab workflow.

## Reference

For understanding or extending the codebase:

- **[system-overview.md](system-overview.md)** — The codebase by layer:
  UI/API runtime, LLM routing, tools, voice, characters, persistence, and
  deployment. Best first read before deeper architecture docs.
- **[architecture.md](architecture.md)** — Component diagram and how
  audio / text flow through the system (VAD → STT → LLM → tools → TTS).
- **[routing-pipeline.md](routing-pipeline.md)** — The LLM-side pipeline:
  classifier, rescue, two-phase compose, the full experimental journey
  (iter-3 → iter-4 rollback → Phase-0 → round 2 → two-phase), and why
  each layer exists. Read this when you want the *why* of the LoRA /
  routing stack.
- **[design.md](design.md)** — 12 project-level design principles
  (active vs passive, code-is-truth, fail-soft, etc.).
- **[api-reference.md](api-reference.md)** — The `assistant.yaml`
  schema, the main browser-facing `/api/*` routes, the upstream
  service APIs the bridge calls, and the systemd templates.
- **[tools.md](tools.md)** — Catalog of the 21 tool schemas
  (20 active by default), which ones render widgets, and the recipe
  for adding a new tool.
- **[deployment.md](deployment.md)** — `setup.sh`, Docker Compose layout,
  systemd service wiring, and first-boot troubleshooting.

## Research + historical analysis

- **[eval_135_manual_review.md](eval_135_manual_review.md)** — Per-iteration
  manual review of replies against a held-out 135-case set. The large
  iter-3 per-case table is split into [evals/iter3-reference.md](evals/iter3-reference.md).
- **[routing_eval_comparison.md](routing_eval_comparison.md)** —
  Cross-backend + cross-iteration metric table (iter-3 → two-phase
  compose), with the narrative for each step.
- **[history/routing-evolution.md](history/routing-evolution.md)** —
  Historical training iterations, backend experiments, runtime rescue
  changes, and the former iter-5 plan split out of the current pipeline
  reference.
- **[history/iter5-postmortem-2026-04-22.md](history/iter5-postmortem-2026-04-22.md)**,
  **[history/iter6-postmortem-2026-04-22.md](history/iter6-postmortem-2026-04-22.md)**,
  **[history/iter7-postmortem-2026-04-23.md](history/iter7-postmortem-2026-04-23.md)** —
  Failed training attempts kept as historical engineering records.
- **[history/lora-training-audit-2026-04-21.md](history/lora-training-audit-2026-04-21.md)** —
  Audit notes for the LoRA training setup.
- **[evals/](evals/)** — Raw per-case JSONs + logs for every eval run
  referenced above. See [evals/README.md](evals/README.md) for the file
  catalog + JSON schema.
- **[ideas.md](ideas.md)** — Roadmap / future-work notes.

## Operations

For running / maintaining a live deploy:

- **[../RUNBOOK.md](../RUNBOOK.md)** — Updating, stopping, resetting,
  hardware sizing, performance tuning, troubleshooting.
