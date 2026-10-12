# A0/A1 Target-Position Decomposition Plan

Status: **frozen analysis plan; read-only execution**

## Question

The target-length result shows A1 matches A0 at target length 2 but collapses
at lengths 4 and 8, while both models retain 100% training exact match. This
analysis determines whether A1 already fails on the first requested target
token or whether performance degrades across target positions.

## Scope and controls

- Use only the six existing final target-length checkpoints from the frozen
  values8000, one-association, distance-0, seed-17 runs.
- Analyze target lengths `2`, `4`, and `8` for both frozen A0 and original A1.
- Use the existing deterministic train and held-out example construction from
  `scripts/run_synthetic_overfit_calibration.py`.
- Do not retrain, alter checkpoint tensors, alter configs, alter A0/A1 model
  definitions, or write into `results/runs/`.
- Verify every source checkpoint through its existing sidecar/hash metadata
  before loading it. Keep temporary diagnostic tensors/probes in memory only.

## Position convention

For an input of length `S` and target length `L`, the target suffix occupies
the final `L` token positions. Causal logits at input positions
`S-L-1` through `S-2` predict target positions `1` through `L` under teacher
forcing. The autoregressive prefix is the same example truncated before the
target suffix; each next token is generated greedily and appended before the
next prediction.

## Measurements

For each architecture, split (`train`, `heldout`), target length, and target
position `j in 1..L`, record:

1. teacher-forced cross-entropy for the correct target token;
2. teacher-forced top-1 token accuracy;
3. teacher-forced correct-token probability after softmax;
4. teacher-forced logit margin: correct-token logit minus the largest
   incorrect-token logit;
5. greedy autoregressive token accuracy at position `j` when the preceding
   target tokens are generated, not teacher-forced;
6. greedy autoregressive whole-target exact match and per-position accuracy
   when the framework permits the full generated sequence.

Aggregate by the existing 128 examples. Preserve finite-value checks and
include sample counts, target lengths, source checkpoint paths and hashes,
config/protocol/generator hashes, and an explicit `read_only: true` field in
the machine-readable result.

## Predeclared interpretation rules

- If A1 is already worse at position 1 under teacher forcing, a purely
  autoregressive error-accumulation explanation is insufficient; prioritize
  storage, binding, query, or readout diagnostics.
- If A1 is close to A0 at position 1 but degrades progressively with position,
  target-state degradation or autoregressive error propagation becomes more
  plausible.
- If teacher-forced metrics remain strong while autoregressive metrics fall,
  generation feedback is implicated; if both fall together, the failure is
  upstream of autoregressive feedback.
- A near-floor token metric is reported as a measurement, not treated as
  evidence of a mechanism by itself. Subsequent representation forensics must
  distinguish retained information from unusable readout.

## Outputs

- `results/a0_a1_target_position_decomposition_values8000.json`
- `docs/A0_A1_TARGET_POSITION_DECOMPOSITION_RESULT.md`
- a committed update to `docs/MECHANISM_CAMPAIGN_STATUS.md`

The analysis script must fail nonzero on missing artifacts, hash mismatch,
non-finite metrics, unexpected parameter counts, or any source-checkpoint
integrity mismatch.

