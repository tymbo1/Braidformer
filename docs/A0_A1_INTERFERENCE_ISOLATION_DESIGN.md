# A0/A1 inference-only interference-isolation design

Status: **design complete; execution not started**.

This document defines the next mechanistic follow-up after the completed
query-conditioned binding branch. It is deliberately an inference-only
diagnostic. No A0/A1 checkpoint, architecture, optimizer, or training
objective is changed by this design.

## Question

Does A1 lose the one-association binding because recurrent updates written by
the neutral distractors overwrite the stored value, or because the state
passively decays while the distractors are processed?

The existing A1 mixer uses, for each channel,

\[
\alpha=\operatorname{sigmoid}(\ell),
\qquad
u_t=(1-\alpha)\,\operatorname{sigmoid}(g_t)\,\tanh(c_t),
\]

\[
s_t=\alpha s_{t-1}+u_t,
\qquad
y_t=W_o\left(\operatorname{sigmoid}(o_t)\odot s_t\right).
\]

The test will change only the state recurrence at distractor positions. The
query, query-key, answer marker, output projection, feed-forward path, and
all other timesteps remain native.

## Frozen scope and source controls

Use the original, non-curriculum A1 checkpoints from the completed
one-association, 8,000-value distractor-distance axis:

- distance `0`;
- distance `4`;
- distance `16`;
- distance `64`.

Each source run is seed 17, 2,000 optimizer steps, one association, 8,000
candidate values, the fixed 128-example held-out split, and target length 1.
The original A1 is the only patched model. A0 remains an untouched native
reference from the completed distance aggregate; it is not passed through an
artificial recurrent intervention.

Pinned source artifacts at design time:

- frozen execution commit: `3e8e8623d596d0878d0e7f49dea280d37335fcd7`;
- distance aggregate:
  `results/a0_a1_solvability_distances_values8000.json`, sha256
  `3a40736a783fd1ac977fe2b6651a6e015891d85699f6fee2e3698cc0917a8261`;
- synthetic runner:
  `scripts/run_synthetic_overfit_calibration.py`, sha256
  `2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5`;
- distance verifier:
  `scripts/analyze_synthetic_solvability_distances_values8000.py`, sha256
  `79b3abd5f40fab4646c3308b1b8daf6ec54b44653b08ce81ef471e435d386ff3`.

The implementation must verify the existing aggregate, each source summary,
configuration hash, protocol hash, generator/data hash, checkpoint sidecar,
parameter count, seed, step count, and numerical-failure fields before any
checkpoint is loaded.

## Exact intervention geometry

For distance `d`, the input layout is:

```text
TASK, RECORD, KEY, VALUE, DISTRACTOR×d, QUERY, KEY, ANSWER, VALUE
```

With zero-based token positions:

- stored value: position `3`;
- distractors: positions `4` through `3+d` inclusive;
- query marker: position `4+d`;
- query key: position `5+d`;
- answer marker / teacher-forced target-logit position: position `6+d`.

The intervention applies only at positions `4` through `3+d`. At distance
zero the protected interval is empty, so every intervention must reproduce
the native forward pass within the declared floating-point tolerance. This is
the negative implementation control.

The rule is applied to every one of A1's eight recurrent state-space mixers,
not just an arbitrarily chosen layer. It is therefore a test of the complete
distractor interval rather than another single-layer replay. It does not add
state channels, change the learned decay parameters, or patch the query
itself.

## Predeclared variants

Every held-out example receives a native baseline and both functional variants
in the same process.

### Native control

At every timestep:

\[
s_t^{\mathrm{native}}=\alpha s_{t-1}+u_t.
\]

### Write-protected / decay-only variant

At distractor positions only:

\[
s_t^{\mathrm{write\text{-}protected}}=\alpha s_{t-1}.
\]

This suppresses new distractor writes while retaining the learned passive
decay. The native recurrence resumes at the query marker and remains native
afterward.

### Fully frozen distractor-state variant

At distractor positions only:

\[
s_t^{\mathrm{frozen}}=s_{t-1}.
\]

This suppresses both new writes and passive decay for the interval. Output
gates and the normal residual/feed-forward computation still run at the
distractor positions. The native recurrence resumes at the query marker.

The write-protected variant is the primary interference test. The fully frozen
variant is the stronger retention control and distinguishes an update/write
effect from an additional decay effect. No query-freeze, target-position
replay, layer selection, or other variant may be added after inspecting the
results.

## Measurements

For every distance and variant, report the existing native metrics over both
the full vocabulary and the 8,000-value candidate range:

- teacher-forced target cross-entropy;
- correct-token probability;
- top-1 accuracy;
- correct-token logit margin;
- mean and median target rank;
- greedy answer-token accuracy and exact match where the frozen evaluator
  supports them.

Also record, for every recurrent layer and each relevant site:

- state norm;
- recurrent update norm;
- cosine similarity to the post-value state;
- state values at the post-value, post-distractor, post-query, and target-logit
  positions;
- intervention-induced state and logit deltas.

Reuse the existing fixed pre-query value-probe methodology as a secondary
measurement: fit the deterministic probe on the training split at the
post-value site and evaluate the unchanged probe at held-out post-distractor,
post-query, and target-logit sites for all three A1 variants. No probe may be
refit separately to a patched held-out representation. Native and patched
metrics must be paired on the same examples.

The result must include absolute metrics and deltas from native at every
distance. Continuous loss, probability, margin, and rank are primary because
the held-out exact-match sample contains only 128 examples and may remain at
the floor.

## Predeclared predictions and adjudication

The distance-zero intervention deltas must be zero within tolerance. If not,
the implementation is invalid and no scientific result is accepted.

| Outcome | Interpretation |
| --- | --- |
| Write-protected improves long-distance value CE/rank/top-1, especially at 16/64 | Active distractor writes are causally damaging to the binding. This is evidence for recurrent interference, not merely passive decay. |
| Fully frozen improves, while write-protected does not | Passive decay or a decay/update interaction remains necessary; pure distractor-write interference is not isolated. |
| Both improve, with fully frozen at least as strong as write-protected | The interval contains both a destructive-write component and a retention/decay component. |
| Neither improves native value metrics or fixed-probe transfer | Distractor-interval state evolution is not sufficient to explain the failure; query-conditioned corruption, pre-query binding/generalisation, or another recurrent transformation remains primary. |
| State/probe metrics improve but native output does not | State preservation is not enough; query use, binding composition, routing, or readout still fails. |
| Native metrics change only at short distance or only outside the value range | Treat as a non-specific sequence-state effect, not as a retrieval rescue. |

No outcome unlocks a training intervention automatically. A positive result
would justify a separately frozen localization or minimal training protocol;
a null result would not justify stacking another inference patch.

## Integrity boundary

The eventual analysis script must be read-only with respect to source runs and
must fail nonzero on missing artifacts, hash drift, non-finite tensors,
parameter changes, or checkpoint writes. It must record parameter digests
before and after each functional forward pass and verify that they are
identical.

After execution, the repository must contain:

- a committed analysis script whose source hash is recorded in the result;
- `results/a0_a1_interference_isolation_values8000.json`;
- `docs/A0_A1_INTERFERENCE_ISOLATION_RESULT.md`;
- an updated mechanism-campaign continuation entry only if the diagnostic is
  actually executed and passes verification.

The execution must also pass `git diff --check`, the full test suite, and the
repository's checkpoint/hash checks. Raw logs and checkpoints remain ignored
and untracked. No GitHub push is permitted.

This design document records the protocol boundary only. No training,
checkpoint modification, or causal execution is authorized by this design
pass.
