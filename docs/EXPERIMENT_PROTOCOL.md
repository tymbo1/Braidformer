# Experiment protocol

## Purpose

This protocol defines the minimum controls for testing BraidFormer-H1. The goal is to find where the hypothesis fails as readily as where it helps. Each result must be reproducible from a versioned configuration and must preserve raw measurements.

## Planned experimental ladder

| Stage | Planned change | Current status |
| --- | --- | --- |
| A0 | Conventional Transformer control | Implemented; synthetic smoke-tested only |
| A1 | Recurrent/state-space majority backbone | Implemented; CPU-verified; not trained |
| A2 | A1 plus periodic/selective attention | Planned |
| A3 | A2 plus a small sparse latent MoE | Planned |
| A4 | A3 plus conditional depth | Planned |
| A5 | A4 plus a unified compute controller | Planned |
| A6 | Later compressed/landmark memory | Later |
| A7 | Later multi-token prediction | Later |
| A8 | Later conditional/static memory | Later |

The Phase 0 bootstrap did not begin training. The A0 control was subsequently
trained only in its separately authorized seed-17 pilot; the A1 pilot remains
unstarted.

The A0 smoke boundary is deliberately small: validate configuration, construct
the model, check causal logits and parameter count, and run one synthetic
forward/backward pass on CPU. A real experiment still requires a versioned
dataset/tokenizer choice, matched optimizer settings, a working device memory
smoke test, and an experiment record.

The A1 architecture boundary is similarly limited: construct the recurrent
model, verify causal logits and shifted loss, run a synthetic CPU backward pass,
and verify its parameter count against the proposal. The generalized runner's
A1 dry-run and the one-update host CUDA smoke test have now passed. This still
does not authorize the 1,000-step pilot.

## Required matching

For every comparison where the mechanism permits it, hold constant:

- dataset, preprocessing, and exact token stream;
- tokenizer, vocabulary, and sequence packing policy;
- training and evaluation splits;
- random seeds and determinism settings;
- optimizer, learning-rate schedule, warmup, regularization, and initialization policy;
- training-token budget and number of evaluation tokens;
- context length, micro-batch size, effective batch size, and gradient accumulation;
- precision policy and gradient clipping;
- approximate training FLOPs and checkpoint/evaluation cadence.

If a change makes exact matching impossible, record the reason, the alternative control, and the resulting limitation before running it.

## Measurements

Record both planned and actual values. At minimum:

- git commit and configuration hash;
- machine, driver, CUDA, Python, and PyTorch versions;
- model parameter count and active parameter count where meaningful;
- peak allocated and reserved VRAM;
- wall-clock training time, warm-up time, and tokens per second;
- training loss and validation loss against token count;
- task-specific evaluation metrics and uncertainty where practical;
- approximate training FLOPs and the accounting method;
- failures, interruptions, OOMs, NaNs, and skipped evaluations.

Raw logs and step-level measurements belong in ignored run directories under `results/` or another explicitly recorded repository-local path. Concise summaries and configurations should remain versionable.

## BraidNano starting envelope

BraidNano targets roughly 40M–60M parameters. Until a CUDA memory smoke test is available, use the conservative planning envelope in `configs/braidnano_phase0_planning.json`:

- context length: 512 tokens;
- micro-batch size: 2;
- gradient accumulation: 8;
- mixed precision when supported;
- gradient checkpointing enabled;
- no experts, conditional depth, or controller in the A0 planning config.

This is an estimate, not a measured safe maximum. Reduce context or micro-batch size after the first OOM and record the change. Do not use the config as authorization to start training in Phase 0.

## Evaluation order

1. Validate configuration and tensor shapes with unit tests.
2. Run a CPU or tiny-device forward/backward smoke test on synthetic data.
3. Run a short fixed-token pilot with instrumentation.
4. Check memory, throughput, loss curves, and reproducibility.
5. Only then run the next controlled comparison.

No long run should begin before steps 1–4 pass and the experiment record is complete.

## Bounded A0 integrity safeguards

The 1,000-step A0 pilot uses the policy in
[`configs/braidnano_a0_integrity_policy.json`](../configs/braidnano_a0_integrity_policy.json)
and the low-risk utilities in `braidformer.training.integrity`. Before a run,
the local TinyStories artifact verifier must pass. During a run, the A0
training entry point records a durable JSONL record for every optimizer step
containing train loss, validation loss when evaluated, gradient norm, learning
rate, token count, throughput, and peak VRAM. It stops on
non-finite losses or gradient norms, an excessive configured loss ceiling,
skipped optimizer steps, or a non-monotonic token counter. Checkpoints must be
written atomically and accompanied by a SHA-256 sidecar containing recovery
metadata.

This is an engineering safety boundary for the pilot, not a new benchmark or
an architectural feature. Hyperparameter search, distributed fault testing,
security review, and long-run resilience experiments are deferred.

## Phase gates

- **Gate 0:** hardware and software audit completed; CUDA state is explicit.
- **Gate 1:** A0 and A1 have matched data, optimizer, tokens, context, and seeds; the A1 runner and CUDA smoke test must pass before training authorization.
- **Gate 2:** A2 retrieval probes demonstrate whether selective attention corrects a known failure mode.
- **Gate 3:** A3–A5 report quality, active compute, routing statistics, and controller stability together.
- **Gate 4:** memory and multi-token variants are introduced only after the core architecture has a measured baseline.

Qwen distillation is out of scope for Phase 0 and must not be introduced before the core architecture demonstrates basic viability.

## Minimal experiment record

Each future run should include:

```text
experiment_id:
stage:
git_commit:
config_path:
config_hash:
seed:
data_and_tokenizer:
training_token_budget:
context_length:
effective_batch_size:
optimizer_and_schedule:
precision:
parameter_count:
active_parameter_count:
approximate_training_flops:
wall_clock_seconds:
tokens_per_second:
peak_vram_bytes:
train_loss:
validation_loss:
evaluation_metrics:
failure_or_deviation_notes:
```
