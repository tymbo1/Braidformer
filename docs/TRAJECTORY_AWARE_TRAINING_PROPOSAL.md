# Trajectory-aware predictive training proposal

Status: **deferred proposal only; no implementation or training is authorized**.

This is a training-policy side track, not a change to A0, A1, or the frozen
solvability experiments. It records the idea of using the model's cumulative
training history to predict which future examples are already well understood,
which are likely to produce useful learning, and where the training process is
surprised by the data.

## Core hypothesis

Ordinary pretraining applies the same expensive update path to every batch:

```text
batch -> forward -> loss -> backward -> optimizer update
```

Some batches may be both predictable from the accumulated training trajectory
and already well handled by the model. A lightweight controller could assess a
batch before the expensive backward pass and skip, downweight, or fully train
it according to predicted learning utility.

The testable hypothesis is not that the model can guess the literal next text.
With a well-shuffled stream, exact next-batch prediction should contain little
signal. The first experiment should instead predict a future batch's learning
state, for example:

\[
  \widehat{u}_{t+1} =
  \text{expected reducible loss or learning utility of batch }x_{t+1}
\]

conditioned on a compact cumulative state \(z_t\) and the candidate batch:

\[
  \widehat{u}_{t+1}=P_\phi(z_t,x_{t+1}).
\]

The cumulative state may contain detached summaries of recent and historical
training loss, gradient statistics, data-bucket exposure, validation loss,
learning-rate position, optimizer progress, and recent improvement or
forgetting. It is a record of the training trajectory, not an unrestricted
second copy of the model.

## Required distinction: predictability is not mastery

The gate must not skip a batch merely because its topic or difficulty was
predicted correctly. A predictable batch can still expose a weakness. A first
version should therefore use both a trajectory surprise signal and a current
performance signal:

\[
  \text{full update if}
  \quad S(x_{t+1}\mid z_t)\ge\tau_S
  \quad\text{or}\quad
  L(x_{t+1})\ge\tau_L.
\]

Otherwise it may take a reduced path, subject to an exploration/audit floor.
The audit floor is necessary to detect false skips and to prevent the
controller from declaring a data region mastered simply because it stopped
measuring it.

The controller's cost must be included in the measured CUDA and wall-clock
budget. A skipped backward pass is only a saving if the assessment and
predictor overhead do not consume it.

## Deferred experimental schedule

The stages below are a future schedule, not authorization to run them.

| Stage | Test | Purpose | Gate to next stage |
| --- | --- | --- | --- |
| T0 | Define targets and replay traces | Establish data-bucket, loss, gradient, exposure, and forgetting summaries; test calibration without changing model weights | Predictor has an out-of-sample target and leakage audit |
| T1 | Tiny synthetic redundancy/novelty task | Verify that the controller does not skip high-loss or rare useful examples | Full-update and audit controls recover the known task; false-skip rate is measured |
| T2 | One-seed A1 TinyStories pilot | Compare normal training, current-loss gating, and trajectory-predictive gating | All three arms complete with matched data, seeds, token budget, and instrumentation |
| T3 | Three-seed replication | Test whether any compute/quality effect survives initialization variation | Pre-registered quality and CUDA criteria are met |
| T4 | Optional A0 replication or architectural integration | Determine whether the policy is architecture-specific or useful independently of A1 | Separate authorization and a new versioned protocol |

T2 is the first meaningful comparison. The initial arms should be:

1. `A1-normal`: the existing full-update training policy.
2. `A1-loss-gated`: a strong simple baseline using only current loss or a
   cheap current-gradient proxy.
3. `A1-trajectory-gated`: the same policy augmented with the cumulative
   trajectory predictor.

The predictive arm must not be credited for gains that are explained by a
different data order, extra tokens, a different optimizer schedule, or an
unmeasured assessment pass.

## Controls and measurement requirements

All first-pass arms must use the same model, data stream, tokenizer,
initialization seed, optimizer, context, effective batch size, precision, and
validation set. A gated arm consumes the same candidate examples as its
control; it changes only whether the expensive update is applied.

The first comparison should report both views:

- **Token-matched:** validation loss and task metrics after the same available
  token stream.
- **Compute-matched:** validation loss after the same measured CUDA budget,
  allowing a gated arm to process additional data only in a separately
  declared condition.

At minimum, preserve:

- validation loss and task metrics versus consumed tokens, optimizer updates,
  CUDA seconds, and wall-clock seconds;
- fraction of batches fully updated, downweighted, or skipped;
- predictor calibration and out-of-sample utility error;
- false skips: batches predicted as low utility that an audit/full update shows
  had high loss, high gradient utility, or caused useful validation change;
- exploration/audit rate and data-bucket coverage;
- forgetting or regression on previously improved validation slices;
- predictor assessment cost, peak VRAM, throughput, and all update counts.

The primary comparison is trajectory-gated versus loss-gated, not merely
trajectory-gated versus an intentionally weak baseline. A promising result
would preserve baseline validation quality while reducing measured CUDA and
wall-clock cost, or achieve better validation quality at the same measured
cost. The quality tolerance, minimum useful compute reduction, and seed count
must be frozen in a dedicated protocol before T2 begins; this proposal does
not declare those thresholds.

## Leakage and stability safeguards

- The predictor may use only information available before the candidate batch's
  training update. Future batches, future validation outcomes, and the
  candidate batch's post-update gradient are forbidden inputs.
- The candidate batch may be summarized for assessment, but the controller
  must distinguish topic predictability from expected reducible loss.
- Begin with a full-update warm-up and retain a fixed random audit fraction.
- Maintain a minimum update rate and minimum per-bucket coverage.
- Freeze the predictor's version and feature schema before the comparison.
- Record predictor checkpoints and trajectory-state hashes alongside the main
  model checkpoint.
- Stop on non-finite predictor or model values, update starvation, coverage
  collapse, or a configured false-skip ceiling.
- Include the predictor's forward pass, state update, audits, and any recovery
  work in the resource accounting.

## Interpretation boundaries

This proposal tests adaptive training compute allocation. It does not yet test
literal next-batch generation, consciousness, introspection, or a complete
self-model. A later provenance/creation-ledger project could expose immutable
architecture, data, checkpoint, and experiment history to a model, but that is
a separate hypothesis and must not be smuggled into T2.

A positive result would show that a cumulative trajectory signal adds value
beyond a current-loss gate under the declared controls. A null result would be
useful: it would indicate that the extra self-trajectory machinery does not
justify its overhead on the tested corpus and budget. Neither result would
establish general compute savings without replication across data and
architectures.

## Authorization boundary

No T0–T4 implementation, configuration, training run, data reorder, or
architecture change is authorized by this document. Any future execution needs
a new versioned protocol, cheap validation, explicit approval, and a fresh
decision entry.
