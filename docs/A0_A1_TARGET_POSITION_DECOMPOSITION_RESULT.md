# A0/A1 target-position decomposition result

Status: **complete; read-only frozen-checkpoint analysis**.

Analysis plan: `docs/A0_A1_TARGET_POSITION_DECOMPOSITION_PLAN.md` (sha256 `95ed330cedc75d0a82bee2b018802ff68a523b0c53e3e500d050fedc75a09d7e`).
Source protocol: `e133e5279761b73b390f8d52feacd63c5289368c`; source runner sha256 `2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5`.

No model parameters or source checkpoints were modified. The tables below use the held-out split; the machine-readable result also preserves the training split.

## Held-out teacher-forced and greedy autoregressive positions

| L | pos | A0 TF CE | A1 TF CE | A0 TF top1 | A1 TF top1 | A0 TF p(correct) | A1 TF p(correct) | A0 AR top1 | A1 AR top1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 10.353274 | 10.396515 | 0.0000 | 0.0078 | 0.000121 | 0.001031 | 0.0000 | 0.0078 |
| 2 | 2 | 10.398792 | 10.576988 | 0.0078 | 0.0000 | 0.000416 | 0.000035 | 0.0078 | 0.0000 |
| 4 | 1 | 9.707314 | 10.389627 | 0.0234 | 0.0000 | 0.003424 | 0.000048 | 0.0234 | 0.0000 |
| 4 | 2 | 9.301262 | 10.361156 | 0.0781 | 0.0000 | 0.007980 | 0.000049 | 0.0312 | 0.0000 |
| 4 | 3 | 10.144904 | 10.796622 | 0.0078 | 0.0000 | 0.000137 | 0.000034 | 0.0000 | 0.0000 |
| 4 | 4 | 10.150028 | 10.594551 | 0.0078 | 0.0000 | 0.000322 | 0.000035 | 0.0000 | 0.0000 |
| 8 | 1 | 9.888000 | 10.473614 | 0.0312 | 0.0000 | 0.000202 | 0.000061 | 0.0312 | 0.0000 |
| 8 | 2 | 9.885854 | 10.560410 | 0.0234 | 0.0000 | 0.000425 | 0.000036 | 0.0000 | 0.0000 |
| 8 | 3 | 10.090995 | 10.594731 | 0.0000 | 0.0000 | 0.000084 | 0.000032 | 0.0000 | 0.0000 |
| 8 | 4 | 10.067081 | 10.626375 | 0.0078 | 0.0000 | 0.000089 | 0.000033 | 0.0078 | 0.0078 |
| 8 | 5 | 10.190956 | 10.536533 | 0.0000 | 0.0000 | 0.000053 | 0.000034 | 0.0000 | 0.0000 |
| 8 | 6 | 10.064111 | 10.419757 | 0.0000 | 0.0000 | 0.000070 | 0.000036 | 0.0000 | 0.0000 |
| 8 | 7 | 10.038659 | 10.537083 | 0.0000 | 0.0000 | 0.000069 | 0.000039 | 0.0000 | 0.0000 |
| 8 | 8 | 10.066218 | 10.535351 | 0.0078 | 0.0000 | 0.000090 | 0.000036 | 0.0000 | 0.0000 |

## Held-out endpoint summaries

| L | architecture | TF token accuracy | TF exact match | AR token accuracy | AR exact match |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | A0 | 0.003906 | 0.000000 | 0.003906 | 0.000000 |
| 2 | A1 | 0.003906 | 0.000000 | 0.003906 | 0.000000 |
| 4 | A0 | 0.029297 | 0.000000 | 0.013672 | 0.000000 |
| 4 | A1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | A0 | 0.008789 | 0.000000 | 0.004883 | 0.000000 |
| 8 | A1 | 0.000000 | 0.000000 | 0.000977 | 0.000000 |

## Interpretation boundary

The position-resolved metrics are diagnostic rather than a new training result. A1 performance at position 1 is compared directly with later positions under both teacher forcing and greedy generation. A first-position teacher-forced deficit cannot be attributed solely to autoregressive error accumulation; a teacher-forced/autoregressive divergence supports a target-state feedback contribution. Layerwise and query/readout forensics remain required before assigning a mechanism.

## Integrity

All six existing target-length runs were sidecar/hash verified before analysis. The analysis required the expected parameter counts, finite logits and metrics, and used evaluation mode with gradients disabled. Raw logs and checkpoints remain under ignored `results/runs/` paths.
