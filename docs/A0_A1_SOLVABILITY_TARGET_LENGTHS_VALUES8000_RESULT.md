# A0/A1 synthetic solvability target-length result

Status: **complete; all six runs and final checkpoints verified**.

Frozen execution commit: `e133e5279761b73b390f8d52feacd63c5289368c`.

The target is a one-key binding to a sequence of 2, 4, or 8 value tokens drawn from the 8,000-token reserved pool. All other architecture and optimizer controls are unchanged.

| Target length | A0 token acc. | A1 token acc. | A1−A0 token acc. | A0 loss | A1 loss | A1−A0 loss | A1/A0 CUDA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.0039 | 0.0039 | +0.0000 | 10.375900 | 10.486862 | +0.110962 | 1.721× |
| 4 | 0.0293 | 0.0000 | -0.0293 | 9.825911 | 10.535301 | +0.709391 | 1.686× |
| 8 | 0.0088 | 0.0000 | -0.0088 | 10.036486 | 10.535526 | +0.499041 | 1.839× |

## Result

All six models reached 100% training exact match. Held-out whole-sequence exact match was 0% for every arm, so the informative measurements are target loss and per-token accuracy.

A0 retained held-out token accuracy of 0.3906%, 2.9297%, and 0.8789% at target lengths 2, 4, and 8. A1 matched the length-2 token accuracy but fell to 0% at lengths 4 and 8. A1 target loss was higher at every length and required approximately 1.72×, 1.69×, and 1.84× the A0 CUDA time.

This is consistent with a target-length-sensitive A1 retrieval/readout weakness beyond the one-token distance result. It remains one model size and one seed, and the data are synthetic; it does not by itself distinguish recurrent storage from binding, output routing, or autoregressive error accumulation.

## Integrity

Every arm has 2,000 contiguous step records, the expected 100-step validation cadence, matching generator/data/config/protocol hashes, and a sidecar-verified final checkpoint.
