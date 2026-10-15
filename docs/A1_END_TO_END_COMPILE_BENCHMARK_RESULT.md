# A1 end-to-end compile benchmark result

Status: complete; engineering-only benchmark.

Device: NVIDIA GeForce RTX 3060 Ti; PyTorch 2.12.0+cu130; CUDA build 13.0.

## Isolated component

| path | mode | warm-up wall s | measured CUDA s/call |
| --- | --- | ---: | ---: |
| eager forward | eager | 0.114 | 0.023058 |
| compiled forward | reduce-overhead | 4.164 | 0.005412 |

Isolated forward speedup: 4.260x.
Isolated accumulated-training speedup: 8.423x.

## Complete A1 update

| path | mode | warm-up wall s | steady CUDA s/update | steady tok/s |
| --- | --- | ---: | ---: | ---: |
| eager full model | eager | 0.376 | 0.374572 | 21870.3 |
| mixers compiled only | default | 0.374 | 0.223333 | 36680.6 |
| full model compiled | default | 1.459 | 0.183020 | 44760.2 |

Mixer-only end-to-end speedup: 1.677x.
Full-model end-to-end speedup: 2.047x.
Mixer-only speedup retention versus isolated forward: 0.394.
Full-model speedup retention versus isolated forward: 0.480.

## Numerical and state checks

Mixer-only complete loss max relative difference: 1.24098e-05.
Mixer-only complete gradient max relative difference: 0.00165895.
Full-compiled loss max relative difference: 1.78174e-05.
Full-compiled gradient max relative difference: 0.00188772.
Full-compiled final parameter relative L2: 0.005306.
Compiled complete-model paths stayed within declared short-horizon tolerances: True.
Complete-model state-key invariants preserved: True.
Complete-model optimizer-state structures match: True.

No checkpoint, model artifact, or quality-training result was written.
