# A1 decay-tail causal intervention protocol

Status: **frozen before execution**.

This protocol tests whether the distance-sensitive A1 retrieval failure is
caused in part by the absence of long-lived recurrent channels. It is a
fresh-initialization intervention against the completed A1 one-association,
8000-value distractor-distance runs.

## Matched control

The control is the completed A1 arm at each of distractor distances 4, 16,
and 64 under the original `uniform_0.9` state-transition initialization.
The intervention uses the same:

- architecture and 41,870,336 trainable parameters;
- tokenizer, synthetic generator, fixed seed 17, and train/held-out examples;
- sequence geometry, 2,000 optimizer steps, effective batch size 16, and
  optimizer schedule;
- precision, parallel-scan implementation, evaluation cadence, and checkpoint
  policy.

The only intended difference is the fresh initialization of
`state_transition_logit`.

## Decay-tail initialization

For every A1 layer with hidden size 512, channels are initialized as:

| Channels per layer | Initial characteristic τ | Initial decay `a = exp(-1/τ)` |
| ---: | ---: | ---: |
| 384 | original initialization | 0.900000 |
| 32 | 16 tokens | `exp(-1/16)` |
| 32 | 32 tokens | `exp(-1/32)` |
| 32 | 64 tokens | `exp(-1/64)` |
| 32 | 128 tokens | `exp(-1/128)` |

Across the eight layers this gives 3,072 original channels and 256 channels
at each long time scale. The long channels are an initialization tail, not a
fixed mask: all transition coefficients remain trainable under the ordinary
optimizer.

## Runs

| Run | Distance | Sequence length | Training tokens |
| --- | ---: | ---: | ---: |
| A1 decay-tail distance 4 | 4 | 12 | 384,000 |
| A1 decay-tail distance 16 | 16 | 24 | 768,000 |
| A1 decay-tail distance 64 | 64 | 72 | 2,304,000 |

The intervention is deliberately limited to the predeclared one-association
distance axis. It does not alter the frozen original A0/A1 results and does
not introduce a target-length, relational-curriculum, or A2 branch.

## Predictions

The primary prediction is that the intervention increases fixed-probe value
decodability and held-out retrieval at distances 16 and 64 while preserving
100% training memorization.

| Observation | Interpretation |
| --- | --- |
| Probe retention and task performance both improve | Strong causal support for decay-limited retention. |
| Probe retention improves but task performance does not | Storage was repaired; query interaction, binding, or readout remains limiting. |
| Neither improves | The short learned time constants were a correlate rather than the dominant cause. |
| Distance-0-like behavior degrades in a later control | Long-lived channels may trade off against local language dynamics; this branch must not be called a free improvement. |

## Measurements

For every run, retain the standard training/held-out target loss, greedy exact
match, training memorization, CUDA time, and checkpoint integrity. After all
three runs complete, apply the existing read-only A1 state/decay forensic
procedure to the intervention checkpoints using the same fixed linear probe
fit at distance 0 and evaluation distances 0, 4, 8, 16, 32, and 64.

No result is interpreted as causal from the initialization change alone unless
the post-training state signature moves in the predicted direction as well as
the retrieval curve.
