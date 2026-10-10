# BraidFormer research rules

These rules apply to every change and experiment in this repository.

## Workspace isolation

- This repository is `/home/timbushnell/braidformer_lab`.
- It is independent from `/home/timbushnell/projectbrain_dev` and every other project.
- Do not read, modify, import from, or write into `projectbrain_dev` or any unrelated repository.
- Keep source, environments, datasets, checkpoints, logs, measurements, and results under this repository unless the user explicitly authorizes another location.
- Do not use external data, model artifacts, or services as hidden dependencies.

## Scientific posture

- BraidFormer is an empirical falsification program, not a claim that the proposed architecture is superior.
- Treat every architectural addition as a hypothesis to test against a controlled baseline.
- Do not claim improvement without measured evidence from comparable experiments.
- Record negative, null, failed, and infrastructure results; a failed experiment is still a result.
- Do not silently change an architecture, data pipeline, tokenizer, optimizer, schedule, seed, budget, or evaluation procedure.
- Every material experiment change must be represented in a versioned configuration, decision record, or experiment log.

## Matched controls and reproducibility

Where applicable, comparisons must match:

- dataset and token stream
- tokenizer and vocabulary
- random seeds
- optimizer and schedule
- training-token budget
- batch and effective batch size
- context length
- evaluation data and metrics
- approximate training FLOPs

Record actual parameter count, active parameter count where meaningful, wall-clock time, peak VRAM, tokens per second, training loss, validation loss, and evaluation metrics. Preserve raw measurements before producing summaries. Use fixed seeds and record software and hardware versions.

## Experiment discipline

- Start with small tests and smoke checks before expensive tests.
- Do not begin long training runs without an explicit, versioned configuration and a successful cheap validation.
- The initial Phase 0 bootstrap does not download large datasets or checkpoints
  and does not train. A later explicitly approved A0 readiness step may fetch
  the pinned dataset and tokenizer artifacts; it still must not train or fetch
  checkpoints.
- The initial target is BraidNano at approximately 40M–60M parameters; do not jump to 1B or 14B.
- Do not use Qwen distillation in Phase 0. Qwen may be introduced later as a clearly labeled teacher or baseline.
- Keep the architecture ladder explicit. Do not implement or report later stages as complete merely because scaffolding exists.
- Preserve raw hardware and experiment reports in `results/` when they are safe and lightweight; keep large artifacts out of git.

## Review standard

Before declaring a task complete, check:

1. The change stays inside this repository and its stated scope.
2. Tests or smoke checks pass, or failures are documented precisely.
3. Configurations are reproducible and no behavior changed silently.
4. Claims are limited to what the measurements support.
5. The next experiment and its falsification criteria are clear.
