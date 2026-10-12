# Relational curriculum campaign status

Status: **complete; result adjudicated**.

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

1. **Complete.** Four matched curriculum arms ran and passed integrity checks.
2. **Complete.** Frozen held-out retrieval results were aggregated.
3. **Complete.** H8 was adjudicated against H2/H4/H7 in the result report.

No A2, scaling, longer training, persistent memory, predictive-learning,
inheritance, SNAIL, Woven Neural Fabric, post-training, or new-dataset work is
in scope.

## Stop conditions

Stop and report rather than silently adapting if a run fails numerical or
checkpoint integrity, the worktree contains unexplained changes, the frozen
architectures/config controls drift, or the results cannot be compared because
the exposure hash differs. A positive or negative result alone is not a stop
condition; it must be recorded and adjudicated.

## Next queued experiment

Use the existing frozen representation-probe methodology on the four new
curriculum checkpoints, beginning with A1 staged versus A1 shuffled. Test
whether the staged A1 cross-entropy gains correspond to a more decodable held-
out value or binding identity before the query and at target output positions.
This is a read-only diagnostic; it must receive its own analysis plan/commit
before execution and must not modify the checkpoints.

## Completed evidence

The preceding mechanism campaign is complete at commit `040736a`, with the
strongest supported mechanism and its alternatives documented in
`docs/A0_A1_MECHANISM_ADJUDICATION_RESULT.md`. Its recommended next
experiment is this campaign.

The campaign result is documented in
`docs/A0_A1_RELATIONAL_CURRICULUM_V1_RESULT.md`; the aggregate evidence is in
`results/a0_a1_relational_curriculum_v1.json`. H8 is partially supported as a
training-history modifier, but curriculum order did not repair A1's hard
8,000-value or long-distance exact retrieval. H2 remains the leading composite
mechanism and H4 remains strongly supported.

## Continuation log

| state | commit | evidence |
| --- | --- | --- |
| protocol/code freeze | `30488e4f6b95994f02e15de9fab5710ce9af2aa4` | frozen protocol, deterministic generator, runner, configs, and tests |
| four-arm result | `73898c8cc0e1b16e2dd77ec8ab38a3692ee442ab` | four completed summaries, verified checkpoints, and aggregate result |
