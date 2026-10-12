# Relational curriculum campaign status

Status: **complete; result adjudicated; representation-probe continuation complete**.

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
controls. Only stage order changes. The source protocol commit is
`30488e4f6b95994f02e15de9fab5710ce9af2aa4`; the four-arm result commit is
`73898c8cc0e1b16e2dd77ec8ab38a3692ee442ab`.

Execution provenance: the source summaries pin the protocol/code-freeze commit
`30488e4f6b95994f02e15de9fab5710ce9af2aa4`; they do not contain a separate
runner execution-commit field.

## Queue

1. **Complete.** Four matched curriculum arms ran and passed integrity checks.
2. **Complete.** Frozen held-out retrieval results were aggregated.
3. **Complete.** H8 was adjudicated against H2/H4/H7 in the result report.
4. **Complete.** Frozen layerwise representation probes ran on all four
   curriculum checkpoints at target lengths 2/4/8, A1 first, with all source
   checkpoint hashes verified.
5. **Next.** Query/readout localization on these same frozen checkpoints.

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

Run the query/readout localization diagnostic on the frozen curriculum
checkpoints. Compare value/key decodability and geometry before and after the
query, then compare those representations with native output logits and
correct-token rank. This is still read-only; it requires its own committed
analysis plan before execution and must not modify A0/A1 checkpoints.

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

The representation-probe continuation is documented in
`docs/A0_A1_RELATIONAL_CURRICULUM_REPRESENTATION_PROBE_RESULT.md`; its
machine-readable evidence is in
`results/a0_a1_relational_curriculum_representation_probes.json`. Staged A1
has substantially higher held-out-fitted value decodability in later recurrent
states than shuffled A1, but train-fitted held-out and fixed pre-query transfer
remain near floor. This supports a curriculum-sensitive recurrent
representation/transformation effect without demonstrating a stable arbitrary
binding coordinate.

## Continuation log

| state | commit | evidence |
| --- | --- | --- |
| protocol/code freeze | `30488e4f6b95994f02e15de9fab5710ce9af2aa4` | frozen protocol, deterministic generator, runner, configs, and tests |
| four-arm result | `73898c8cc0e1b16e2dd77ec8ab38a3692ee442ab` | four completed summaries, verified checkpoints, and aggregate result |
| representation-probe plan/code freeze | `4effe8e442e32255d9fd1556a4895a963999ede8` | frozen read-only probe plan and wrapper |
| checkpoint-digest correction | `ba3b48a0213a0809056d9dbf92cc5319f9c4f23f` | corrected A1-shuffled source digest; integrity gate stopped before loading on the earlier attempt |
| unique-comparison correction | `19a2ef68db8ab6f43b725167dc4b63b961091ddc` | deduplicated aggregate comparison rows; tests passed |
| representation-probe result | `4b3d4cdd859ac7711d8d3b1f67e08cc5bb4b4fe6` | 12 frozen analyses, verified aggregate, and result report |
