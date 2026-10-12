# Relational curriculum campaign status

Status: **protocol frozen; execution pending**.

This is the continuation campaign selected by the completed mechanism
adjudication. It is separate from, and does not reopen, the completed
`MECHANISM_CAMPAIGN_STATUS.md` queue.

## Current hypothesis set

| ID | hypothesis | current status entering this campaign |
| --- | --- | --- |
| H1 | temporal decay contributes to A1 weakness | partially supported; long-decay tail was insufficient |
| H2 | recurrent transformation/interference amplifies retrieval failure | leading composite mechanism |
| H3 | query-conditioned corruption contributes | surviving alternative |
| H4 | shared compositional binding/generalization failure | strongly supported |
| H5 | pure final readout failure | weakened by frozen probes |
| H6 | autoregressive accumulation is the sole cause | rejected |
| H7 | optimization/training dynamics contribute | unresolved background factor |
| H8 | curriculum/developmental history is a major contributor | newly tested here |

## Frozen experiment

Protocol: `docs/A0_A1_RELATIONAL_CURRICULUM_V1_PROTOCOL.md`.

The experiment has four fresh runs: A0-staged, A0-shuffled, A1-staged, and
A1-shuffled. The two arms within each architecture have identical stage
exposure, fixed 29-token batches, 2,000 optimizer steps, and the same optimizer
controls. Only stage order changes.

Execution commit: **to be recorded from the exact pre-run commit**.

## Queue

1. **Run and verify the four matched curriculum arms.**
2. **Aggregate the frozen held-out retrieval results.**
3. **Adjudicate H8 against H2/H4/H7, record nulls or gains, and select one
   next experiment without changing frozen A0/A1.**

No A2, scaling, longer training, persistent memory, predictive-learning,
inheritance, SNAIL, Woven Neural Fabric, post-training, or new-dataset work is
in scope.

## Stop conditions

Stop and report rather than silently adapting if a run fails numerical or
checkpoint integrity, the worktree contains unexplained changes, the frozen
architectures/config controls drift, or the results cannot be compared because
the exposure hash differs. A positive or negative result alone is not a stop
condition; it must be recorded and adjudicated.

## Completed evidence

The preceding mechanism campaign is complete at commit `040736a`, with the
strongest supported mechanism and its alternatives documented in
`docs/A0_A1_MECHANISM_ADJUDICATION_RESULT.md`. Its recommended next
experiment is this campaign.

## Continuation log

| state | commit | evidence |
| --- | --- | --- |
| protocol/code freeze | pending | this protocol, deterministic generator, runner, configs, and tests |
