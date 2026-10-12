# BraidFormer A0/A1 Mechanism Campaign Status

Status: **active**

This ledger is the authoritative continuation point for the bounded mechanism
campaign. It is updated and committed after every queue item. The campaign
does not modify the frozen A0 Transformer or original A1 recurrent
state-space definition.

## Baseline

- Campaign opened from commit `70ba2cd` (`Record target-length solvability results`).
- The worktree was clean at campaign start: `main...origin/main [ahead 36]`.
- Existing target-length checkpoints, sidecars, reports, and aggregate JSON
  are present and tracked/ignored according to the repository policy.
- The existing full test suite had previously passed after the target-length
  work; it is rerun as part of the first campaign-item verification.
- No GitHub push is authorized.

## Frozen evidence before this campaign

The unchanged 42M controls both memorize the fixed synthetic training sets at
100% exact match, while held-out retrieval degrades with answer-space size,
association count, distractor distance, and (for A1) target length. The
distance axis is especially diagnostic: A0 remains broadly flat, while A1
held-out target loss worsens from `+0.291780` A1-minus-A0 at distance 0 to
`+1.239856` at distance 64. Original A1 forensics found median learned
channel time constant about `8.9` tokens and no channels with `tau >= 16`.
The frozen 75%/25% long-decay-tail initialization created long-lived channels
but did not recover long-distance probe or task performance. Target-length
checkpoints show A1 matching A0 token accuracy at length 2 but falling below
A0 at lengths 4 and 8; both still memorize training perfectly.

These observations support a distance-sensitive internal failure but do not
yet distinguish temporal decay, recurrent transformation/interference,
query-conditioned corruption, binding, readout/routing, autoregressive
target-state degradation, or optimization.

## Current hypothesis set

| ID | Hypothesis | Current status |
| --- | --- | --- |
| H1 | Temporal decay erases the required value trace. | Supported by distance curve, short learned taus, and fixed-probe collapse; long-decay-tail negative result means decay alone is insufficient. |
| H2 | Recurrent-state transformation/interference destroys the binding while the sequence advances. | Open; requires layer/position evidence. |
| H3 | Query-conditioned processing corrupts or overwrites a retained binding. | Open; requires before/after-query comparison. |
| H4 | Binding/key-value identity is not represented compositionally under the synthetic load. | Supported by association/value-space collapse; not enough to explain the distance-specific A1 divergence alone. |
| H5 | The required value survives internally but the output route/readout fails. | Open; requires probe/readout localization. |
| H6 | Multi-token autoregressive target state degrades after the first output. | Open; target-position decomposition is next. |
| H7 | A1 needs more optimization steps or a different optimization trajectory. | Not ruled out, but existing controls show the key failures are present despite perfect training memorization; no new training is authorized before diagnostics. |

## Queue

1. **Target-position decomposition — complete.** Frozen target-length checkpoints;
   no retraining. Measure per-position teacher-forced CE, top-1 accuracy,
   correct-token probability, correct-vs-best-incorrect logit margin, and
   autoregressive accuracy where supported.
2. **Layerwise representation forensics — next.** Frozen checkpoints; probe key,
   value/binding identity, and target position before query, after query, and
   at every target-output position; include every A1 recurrent layer where
   practical.
3. **Query/readout localization.** Use frozen diagnostic probes and logits to
   locate any gap between retained information and output use.
4. **One reversible inference-time causal intervention.** Predeclare its
   prediction from items 1–3; no weight changes.
5. **Mechanism adjudication.** Rank hypotheses and reject contradicted ones.
6. **Optional single minimal training intervention.** Unlock only if at least
   two independent diagnostics converge and a falsifiable mechanism-specific
   prediction is frozen first. Otherwise record that the gate stayed locked.

## Stop gates

Stop only for a completed queue, unresolved protocol contradiction,
integrity/test/checkpoint failure, unexplained worktree changes, a required
frozen-architecture modification outside the single intervention rule, two
failed mechanism-targeted interventions, or genuinely contradictory evidence
requiring arbitrary scientific choice. A successful item is not itself a stop
condition.

## Completed campaign items

### Item 1 — target-position decomposition

The six frozen target-length checkpoints passed source-run and checkpoint
verification. At target lengths 4 and 8, A1 was already at zero held-out
teacher-forced top-1 accuracy on position 1, with mean correct-token
probability about `4.8e-5` and `6.1e-5`, respectively. A0 retained position-1
teacher-forced top-1 of `0.0234` and `0.0312`. Therefore autoregressive error
accumulation cannot be the sole explanation for the A1 collapse at those
lengths. At length 2 both models were near the measurement floor at position 1
and the A1 small edge reversed to a deficit at position 2. A0 also loses some
signal under autoregressive feedback at longer targets, but A1's length-4/8
deficit is present before generated target tokens are fed back.

The full position-wise CE, top-1, correct-token probability, logit margin, and
greedy autoregressive metrics are in
`results/a0_a1_target_position_decomposition_values8000.json` and
`docs/A0_A1_TARGET_POSITION_DECOMPOSITION_RESULT.md`. This item weakens a
pure H6 explanation and leaves H1/H2/H3/H4/H5 open for representation
forensics. The current next item is layerwise representation forensics.

Integrity: source checkpoint hashes and machine-readable finite-value checks
passed; the full test suite passed (`67 passed`); `git diff --check` passed;
raw logs/checkpoints remain ignored and no architecture/config files changed.

## Commit ledger

| Queue state | Commit | Evidence |
| --- | --- | --- |
| Campaign baseline | `70ba2cd` | Existing target-length result and clean repository state. |
| Campaign/Item 1 protocol | `e82f342` | Ledger and frozen target-position analysis plan. |
| Item 1 analyzer | `e031c2d` | Read-only decomposition implementation, committed before execution. |
| Item 1 result | `6d19ee4` | Frozen-checkpoint decomposition result/report and ledger update. |

