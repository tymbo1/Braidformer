# A1 protected associative-lane falsification protocol

Status: **frozen for execution; fresh training branch**.

This is a new architectural falsification branch after the completed A0/A1
mechanism campaign. The original A0 Transformer and original A1 recurrent
state-space model remain untouched controls. This protocol does not authorize
A2, persistent memory, scaling, new data, or any modification to the frozen
models.

## Question

Can a small parallel associative lane preserve and retrieve one bound
key/value pair when the ordinary A1 recurrent path is still free to process
the sequence normally?

The specific falsifiable prediction is:

> If the missing capability is protected compositional persistence rather than
> simply more parameters, the protected lane should improve held-out retrieval
> as distractor distance grows, while the parameter-matched ordinary lane
> should not show the same distance profile.

This first branch intentionally tests one binding only. It does not claim to
solve the already-completed multi-association axis.

## Frozen controls and scope

Run two fresh lane variants at each of the existing one-association,
8,000-value distance rungs:

- distance `0`;
- distance `4`;
- distance `16`;
- distance `64`.

Variants:

1. `protected`: the associative lane commits a key/value pair at the value
   token and does not rewrite that pair at distractor positions;
2. `ordinary`: the same lane, with the same parameters and initialization,
   updates its memory state at every token using the same learned decay.

The original frozen A1 distance results remain the primary historical control.
The fresh ordinary-lane variant is the capacity/parameterization control for
the new branch. Protected and ordinary runs use the same seed, data generator,
optimizer, schedule, effective batch, step count, evaluation cadence, and
checkpoint integrity rules as the completed distance axis.

Each arm uses seed `17`, 2,000 optimizer steps, one association, target
length `1`, the 8,000-value candidate range, the fixed 128-example training
split, and the disjoint 128-example held-out split. Only the lane mode and
distractor distance vary.

## Protected lane definition

The base is an unchanged copy of the existing eight-layer, 512-wide A1
recurrent state-space model. A parallel lane of width `64` is added after the
ordinary A1 blocks and before final normalization and the tied language-model
head.

The lane has exactly these trainable components:

- `key_projection: Linear(512, 64)`;
- `value_projection: Linear(512, 64)`;
- `readout: Linear(64, 512)`;
- one 64-channel memory-decay logit;
- one scalar query-read scale;
- one scalar query-read bias.

The lane adds `99,010` parameters, giving an expected total of `41,969,346`
for both variants. The ordinary and protected lane modes must have identical
parameter names, shapes, initial values, and parameter count; only the state
transition rule differs.

The lane uses only causal hidden states and the existing reserved task-marker
tokens. It never reads the target value or labels. For the frozen one-binding
layout

```text
TASK, RECORD, KEY, VALUE, DISTRACTOR×d, QUERY, KEY, ANSWER, VALUE
```

the lane identifies the key event from a token immediately following
`RECORD`, commits the pending key together with the current value at the next
token, and reads at the `ANSWER` marker using the preceding query-key hidden
state. The resulting lane readout is added only at the answer-marker hidden
position, which is the teacher-forced position that predicts the target value.
The marker schedule supplies event boundaries; it does not supply key or value
identities.

Let `k_t` and `v_t` be the projected key/value candidates and let `m_k,m_v`
be the lane memory. At a value commit, both modes use the same learned decay
to write the pending key and current value:

\[
m_k\leftarrow\alpha m_k+(1-\alpha)k_{\text{pending}},
\qquad
m_v\leftarrow\alpha m_v+(1-\alpha)v_t.
\]

In the protected mode, the pair is then held unchanged at all non-commit
positions until another explicit value commit. In the ordinary mode, the lane
also applies the same decay/update recurrence at every other token, allowing
distractors to rewrite the pair.

At the answer marker, the previous query-key representation is compared with
the stored key:

\[
q=\operatorname{normalize}(k_{t-1}),
\qquad
r=\operatorname{normalize}(m_k),
\qquad
g=\sigma(s(q\cdot r)+b),
\qquad
h_{\text{lane}}=W_{\text{readout}}(g,m_v).
\]

The base A1 hidden state receives `h_lane` only at that answer marker. No
ordinary A1 state transition, gate, feed-forward block, embedding, tokenizer,
or output head is altered.

## Training and evaluation controls

- Same AdamW hyperparameters, cosine schedule, warmup, FP16 autocast,
  gradient clipping, micro-batch, accumulation, and 2,000-step budget as the
  completed distance protocol.
- Same deterministic generator and data hashes for distances 0, 4, 16, and
  64.
- Fresh initialization per arm, with seed reset before construction.
- Checkpoint every 100 steps, atomically, with optimizer/RNG state and SHA-256
  sidecars. Interrupted runs must resume into a fresh directory only.
- Original A1 checkpoints are loaded only for comparison and are never
  modified or overwritten.

Report for every arm and validation checkpoint:

- training and held-out target loss;
- answer-token accuracy and exact match;
- full-vocabulary and 8,000-value-range target metrics at the final step;
- lane key/query cosine, read gate, memory norms, and lane readout norm on the
  held-out split;
- CUDA time, wall time, token exposure, and peak memory;
- parameter count, config/data/protocol/code hashes, and failure status.

The primary comparison is held-out exact match and target loss versus the
fresh ordinary-lane control and the frozen original A1 distance curve. The
lane diagnostics verify that a protected branch is actually carrying and
reading the binding rather than merely perturbing the output head.

## Predeclared outcomes

| Result | Interpretation |
| --- | --- |
| Protected lane improves held-out distance-16/64 retrieval, ordinary lane does not | Strong support for protected associative persistence as the missing primitive. |
| Both lane variants improve similarly | Added capacity/extra nonlinear path may explain the gain; protected persistence is not isolated. |
| Neither improves, and lane diagnostics remain weak | The proposed one-slot protected lane is insufficient; do not stack another lane modification. |
| Lane diagnostics show strong key/value storage and read gating but output remains weak | The bottleneck is downstream integration/readout or binding formation in the base representation. |
| Protected lane improves only distance 0, not long distances | It learned an answer shortcut or local binding, not robust protected persistence. |

No result authorizes a second architectural modification automatically. A
positive result would be followed by a separately frozen multi-association
generalization test; a null result closes this minimal lane branch.

## Integrity boundary and outputs

The new model and runner must not import the original A1 under a changed
definition or modify existing A0/A1 files. The source code, configs, and
training protocol must be committed before any fresh training starts.

Expected outputs:

- `results/a1_protected_lane_distance_axis_values8000.json`;
- `docs/A1_PROTECTED_ASSOCIATIVE_LANE_RESULT.md`;
- ignored raw run directories under `results/runs/`;
- an evidence update in `docs/MECHANISM_CAMPAIGN_STATUS.md` after successful
  execution.

The result must fail nonzero on non-finite values, checkpoint/hash drift,
parameter-count mismatch, unexpected data identity, or unexplained training
failure. After execution, run `git diff --check`, the full test suite, and
verify that raw logs/checkpoints/secrets remain untracked. Do not push to
GitHub.
