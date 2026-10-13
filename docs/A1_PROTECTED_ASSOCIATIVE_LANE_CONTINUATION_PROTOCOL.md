# A1 protected associative-lane storage continuation protocol

Status: **frozen for continuation execution**.

This document is an operational continuation of the frozen scientific
protocol in `docs/A1_PROTECTED_ASSOCIATIVE_LANE_PROTOCOL.md`. It does not
change the hypothesis, model, data generator, optimizer, seed, budget, or
evaluation. The original protocol commit remains `32fd4d052d6ff1c5783472b9a673885c4e61985b`; it is passed to the runner so the resumed
checkpoint retains its original provenance.

## Reason for continuation

The first execution stopped on a root-filesystem checkpoint serialization
failure, not on a numerical or scientific result. Storage remediation has now
been completed and independently verified:

- root filesystem: approximately 429 GiB free;
- second NVMe: `/mnt/ssd2`, approximately 597 GiB free;
- all 63 checkpoints from the completed/partial arms verified before storage
  remediation;
- no original A0 or A1 checkpoint was modified.

The failed ordinary distance-4 arm has valid checkpoints through step 300. Its
step-400 checkpoint is absent and is not treated as valid evidence.

## Frozen continuation sequence

Run the following arms in order, using fresh run directories:

1. Resume ordinary distance-4 from
   `results/runs/a1_protected_lane_distance4_seed17_ordinary/checkpoint_step_000300.pt`
   into `a1_protected_lane_distance4_seed17_ordinary_resume300`.
2. Fresh protected and ordinary distance-16 arms.
3. Fresh protected and ordinary distance-64 arms.

The completed distance-0 protected/ordinary and distance-4 protected arms are
not rerun. The failed source directory is never overwritten. Every new arm
uses the existing matching config, seed 17, and original protocol commit.

## Integrity gates

Before each arm, verify a clean worktree, CUDA availability, the source
checkpoint sidecar where applicable, and at least 20 GiB free on `/`. After
each arm, verify its final checkpoint and sidecar, finite summary metrics,
expected parameter count (`41,969,346`), expected token budget, and no
unexpected source changes. Stop before the next arm on any checkpoint,
numerical, provenance, or filesystem-integrity failure.

The continuation is complete only when all six required comparisons are
available: protected/ordinary at distances 0, 4, 16, and 64, with the
distance-4 ordinary result supplied by the verified resume. The final
aggregate must keep the interrupted source failure separately identified and
must not silently treat its missing step-400 file as a result.

No A0/A1 definition, protected-lane equation, or original result is changed.
No A2, scaling, persistent-memory, curriculum, predictive-learning, or
unrelated architecture work is authorized by this continuation.
