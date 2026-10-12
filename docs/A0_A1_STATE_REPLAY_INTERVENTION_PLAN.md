# A0/A1 state-replay intervention plan

Status: **frozen causal diagnostic plan; inference-only execution**

## Question

Does restoring the recurrent state associated with each stored target value at
its corresponding target-output position improve A1 retrieval? This tests
whether the value trace is being lost or destructively transformed before the
native readout, without changing any trained weight.

## Rationale and scope

Items 1–3 establish: A1 fails on the first target token under teacher forcing
at target lengths 4 and 8; the distance curve and earlier time-constant probe
support a temporal/state hypothesis; cross-validated fixed and site-specific
value probes do not show a strong retained-value/native-readout gap. The
intervention therefore targets recurrent-state preservation, not the output
head.

Use only the six frozen values8000 target-length checkpoints, the deterministic
heldout examples, and A1's original `uniform_0.9` recurrence. A0 remains an
untouched control and no checkpoint or model parameter is modified.

## Intervention

For each target length `L` and target position `j` (1-indexed), use A1 layer 7
only. During a fresh inference forward pass, replace the internal diagonal
recurrent state at the target-output position with the internal state from the
earlier stored-value token for that same target:

\[
  s_{7,\,4+L+j} \leftarrow s_{7,\,2+j}.
\]

The target-output gate, output projection, residual path, feed-forward path,
all other layers, and all other timesteps remain unchanged. The intervention is
implemented functionally in the diagnostic forward pass and is discarded
afterward. Unpatched inference on the same heldout examples is the control.

Measure native teacher-forced metrics at every target position over the full
vocabulary and the 8,000-value candidate range: cross-entropy, correct-token
probability, top-1, logit margin, and target rank. Also record the patched
state norm/cosine relative to the unpatched target state and verify that the
source/target positions and all tensors are finite.

## Predeclared prediction

If temporal retention or recurrent-state transformation is a causal part of
the A1 weakness, state replay should improve the heldout target-value rank and
top-1 at one or more positions, with the clearest improvement at lengths 4 and
8 and especially at the first target position. If replay does not improve the
metrics, the current evidence does not support state preservation at this
layer/position as the missing operation; do not stack another intervention.

An improvement is mechanistic evidence, not a production fix. A null result
does not disprove every temporal/state hypothesis, because this is one
predeclared layer-7 replay and the state may need query-conditioned binding.

## Integrity requirements

The analysis must verify all six source checkpoint sidecars/hashes, expected A1
parameter counts, unchanged source model state, exact intervention positions,
finite baseline/patched tensors and metrics, and `read_only: true`. It must
fail nonzero on missing artifacts, hash mismatch, non-finite values, or any
checkpoint write. No architecture/config/protocol file may be modified by the
run.

## Outputs

- `results/a0_a1_state_replay_intervention_values8000.json`
- `docs/A0_A1_STATE_REPLAY_INTERVENTION_RESULT.md`
- updated `docs/MECHANISM_CAMPAIGN_STATUS.md`
