# A0/A1 mechanism adjudication result

Status: **complete; evidence synthesis only**.

Adjudication plan: `docs/A0_A1_MECHANISM_ADJUDICATION_PLAN.md` (sha256 `451ac300e36e8af602404ec7cfd6d176f30775692e9429d5fabfc781b10c5c95`).

## Strongest supported mechanism

A1's weakness is best described as a composite recurrent retrieval failure: the task requires compositional key/value binding, and A1's learned recurrent transformation/retention path makes that binding increasingly unstable as temporal separation grows. The evidence does not isolate a single decay constant, layer, gate, or output-head defect.

## Evidence chain

1. Both A0 and A1 memorize the fixed training associations, but heldout generalization collapses as value-space size and association count increase.
2. A1 diverges from A0 as distractor distance grows; its loss gap reaches about `+1.240` at distance 64 while A1 uses more CUDA time.
3. Original A1 learns median state time constants around `8.9` tokens with no tau>=16 channels; frozen value probes collapse at matching distances.
4. Adding a long-timescale tail creates 768 long-lived channels but does not repair long-distance task/probe performance.
5. Target-position decomposition shows the A1 length-4/8 deficit already on the first teacher-forced output, ruling out autoregressive accumulation as the sole cause.
6. Layerwise and four-fold query/readout probes do not reveal a stable heldout value code or a strong hidden-value/native-readout gap.
7. The predeclared layer-7 stored-value state replay produces no top-1 gain and slightly worsens CE, rejecting simple state preservation as sufficient.

## Hypothesis ranking

| rank | ID | status | conclusion |
| ---: | --- | --- | --- |
| 1 | H1 | partially_supported_but_insufficient | A temporal component is real, but simple decay or direct state preservation is not the complete mechanism. |
| 1 | H2 | leading_composite_hypothesis | The best current mechanism family is recurrent transformation/interference combined with temporal retention and binding demands, not a proven single-layer defect. |
| 2 | H3 | surviving_alternative | Not separated from H2 or H4 by the completed campaign. |
| 2 | H4 | strongly_supported_shared_failure | A shared compositional-binding/generalization failure is established; A1 adds a stronger temporal/state penalty. |
| 3 | H7 | unresolved_background_factor | Not ruled out, but not enough evidence to unlock an optimization intervention inside this campaign. |
| 4 | H5 | weakened | A pure final-readout failure is not supported; nonlinear or task-specific routes remain possible but are secondary. |
| 5 | H6 | rejected_as_sole_cause | Autoregressive feedback may contribute later, but cannot be the primary explanation. |

## Optional training gate

**Locked.** Two independent diagnostics support a broad recurrent/temporal/binding family, but they do not converge on one minimal mechanism-specific factor: long decay was already falsified as sufficient, layer-7 replay was null, and cross-validated pre-query value generalization is weak. Starting a training intervention now would require an arbitrary choice between state dynamics, query conditioning, binding curriculum, and optimization. No item-6 training run was performed.

## Surviving alternatives and best next experiment

The main alternatives are query-conditioned corruption (H3), a task/curriculum-induced binding failure shared by both architectures (H4), and an optimization factor (H7). The single best next experiment outside this completed campaign is a matched fresh A0/A1 run with a staged synthetic relational/binding curriculum, preserving architecture, optimizer, compute budget, and final heldout tests. It directly tests whether the current boundary is primarily developmental/curriculum-induced before any architectural patch is attempted.

## Integrity and scope

All source result JSON files were present and complete; the four campaign diagnostics were read-only; source file hashes and checkpoint hashes are preserved in the machine-readable result. The full test suite passed after each campaign item (`67 passed`), `git diff --check` passed, raw checkpoints/logs remain ignored, no A0/original-A1 definitions were changed, and no GitHub push occurred.

## Commits

| stage | commit |
| --- | --- |
| campaign_baseline | `70ba2cd` |
| item1_protocol | `e82f342` |
| item1_analyzer | `e031c2d` |
| item1_result | `6d19ee4` |
| item2_protocol | `9202c1d` |
| item2_analyzer | `338a249` |
| item2_result | `9856a8b` |
| item3_protocol | `75c4efe` |
| item3_analyzer | `a0555b0` |
| item3_result | `b534b0e` |
| item4_protocol | `6e299a7` |
| item4_analyzer | `961d452` |
| item4_result | `dbf4706` |
| item5_protocol | `cddcd83` |
| item5_analyzer | `861854b3068e1a6b9284cca034bf1f912513868f` |

Final Git status is recorded in the machine-readable result and must be clean except for this result/ledger commit while the campaign handoff is made.
