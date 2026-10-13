# A0/A1 relational-curriculum mechanism adjudication

Status: **complete; one mechanism-targeted training intervention unlocked**.

This adjudication combines the frozen behavioural ladder, decay/state forensics,
curriculum comparison, layerwise probes, query/readout localization, and the
single reversible state-replay intervention. It does not modify A0 or original
A1.

## Evidence chain

1. **Behavioural solvability ladder.** Both architectures reached 100% training
   exact match while held-out retrieval collapsed as value-space size,
   association count, distance, and target length increased. This supports a
   systematic binding/generalization failure rather than inability to memorize
   the training examples.
2. **Distance and decay forensics.** A1 performance degraded with distance;
   learned time constants were short and frozen-state value probes degraded at
   matching distances. The long-decay-tail intervention and later state-replay
   test did not restore the task, so temporal retention is a contributor or
   correlate, not a sufficient mechanism.
3. **Target-position decomposition.** A1 was already impaired at the first
   requested teacher-forced target token at longer target lengths. This rejects
   autoregressive accumulation as the sole explanation.
4. **Curriculum result.** Matched staged/shuffled training changed A1 native
   confidence and held-out CE, while hard exact retrieval remained absent.
   Training history is a modifier, not a complete repair.
5. **Layerwise curriculum probes.** Staged A1 had a much larger
   held-out-fitted value ceiling in later recurrent states than shuffled A1,
   while train-fitted held-out and fixed pre-query transfer remained near floor.
   The staged advantage is therefore not a stable cross-example binding code.
6. **Query/readout localization.** Four-fold fixed and site-specific target
   readouts were near floor for A1. Native staged A1 CE improved, but a fresh
   diagnostic readout could not recover the value. A pure final readout failure
   is not sufficient.
7. **State replay intervention.** Copying layer-7 stored-value states into
   matching target positions changed the state but produced no top-1 rescue and
   worsened aggregate CE in every run. Simple state preservation is not
   sufficient.

## Hypothesis status

| hypothesis | status | adjudication |
| --- | --- | --- |
| H1 temporal decay contributes | partial | Distance/decay evidence supports contribution; long-tail and replay nulls reject sufficiency. |
| H2 recurrent transformation/interference amplifies failure | leading | A1 distance sensitivity, later-layer curriculum separation, cross-validated readout failure, and replay null converge on a distributed recurrent transformation/binding problem. |
| H3 query-conditioned corruption contributes | surviving, unlocalized | The pre-query cross-validated coordinate is already weak, so the query cannot be assigned the primary loss yet. |
| H4 shared compositional binding/generalization failure | strongly supported | Train EM is perfect while held-out binding and probe transfer collapse across both architectures. |
| H5 pure final readout failure | weakened as sole cause | Site-specific diagnostic readouts do not rescue A1 target values; replay also fails. |
| H6 autoregressive accumulation is sole cause | rejected | First-target teacher-forced failures and target-position results contradict it. |
| H7 optimization/training dynamics contribute | partial modifier | Staged versus shuffled A1 changes competence and confidence, but does not remove the boundary. |
| H8 curriculum/developmental history is a major contributor | partial modifier | A1 is strongly order-sensitive at easier/CE measures; hard exact retrieval remains broken. |

## Unlocked intervention

Two independent diagnostic families converge on the same actionable mechanism:
A1's recurrent state does not provide a stable train-to-held-out key→value
binding coordinate, and a post-hoc state copy is not enough. The smallest
mechanism-specific training test is therefore unlocked:

> Add a temporary, no-new-parameter auxiliary loss during A1 curriculum
> training that makes the existing layer-7 recurrent state at target-prediction
> positions directly rank the correct value token among the existing 8,000-value
> embedding range.

The original A1 forward output, parameterization, optimizer, data schedule,
and checkpoint controls remain unchanged; the auxiliary term is a training
objective only and is applied to a separate A1 branch. The untouched existing
A1 staged/shuffled checkpoints are the controls. The intervention is falsified
if it does not improve held-out target loss/rank or if it only improves the
auxiliary signal without native retrieval improvement.

The intervention protocol must be frozen before any training. If it fails its
predeclared prediction, no second mechanism-targeted intervention is authorized.
