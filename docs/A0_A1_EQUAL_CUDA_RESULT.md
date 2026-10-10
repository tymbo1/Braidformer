# A0/A1 equal-CUDA-time result

Status: **completed under authorized protocol `de41ef2`**. This is a
single-seed exploratory result; it is not a general architectural superiority
claim.

Both runs started from fresh seed-17 initialization in new run directories.
The original A0/A1 pilots and checkpoints were not loaded or overwritten.

## Outcome

| Metric | A0 Transformer | A1 parallel scan |
| --- | ---: | ---: |
| Optimizer steps | 1,000 | 502 |
| Training tokens | 8,192,000 | 4,112,384 |
| Final training loss | 3.6099218 | 3.7728949 |
| Final validation loss | 3.7396816 | 3.5777884 |
| CUDA execution time | 209.3490 s | 205.3696 s |
| CUDA tokens/s | 39,130.83 | 20,024.30 |
| Training wall time | 210.4063 s | 205.9029 s |
| Total elapsed wall time | 233.6988 s | 225.1602 s |
| Peak allocated VRAM | 1,329,467,904 bytes | 1,657,272,320 bytes |
| Peak reserved VRAM | 1,593,835,520 bytes | 1,931,476,992 bytes |
| Numerical instability | None | None |

The target was `202.1068954` CUDA seconds. A0 was `+3.5833%` and A1 was
`+1.6144%` from the target, so both are within the predeclared ±5% tolerance.
The actual A1-minus-A0 CUDA-time difference was `-3.9794` seconds (`-1.9689%`
of the target). The comparison therefore satisfies the committed matched-CUDA
time condition.

The predeclared quality difference is:

`Delta = validation_loss_A1 - validation_loss_A0 = -0.1618933`.

Because `Delta <= -0.05`, this run is classified as **A1 advantage under the
predeclared exploratory rule**. It does not establish statistical significance,
general capability, or superiority beyond this seed-17 TinyStories comparison.

## Validation curves

The complete concise curve record is in
[`results/a0_a1_equal_cuda_comparison.json`](../results/a0_a1_equal_cuda_comparison.json).
Validation was run at every 100 steps and at the final step:

| Step | A0 validation loss | A1 validation loss |
| ---: | ---: | ---: |
| 100 | 4.9811054 | 4.3334316 |
| 200 | 4.3750186 | 3.8676986 |
| 300 | 4.2953775 | 3.7487871 |
| 400 | 4.0970245 | 3.6181645 |
| 500 | 4.0156197 | 3.5777913 |
| 502 | — | 3.5777884 |
| 600 | 3.9048442 | — |
| 700 | 3.8498941 | — |
| 800 | 3.7642651 | — |
| 900 | 3.7459791 | — |
| 1000 | 3.7396816 | — |

Raw step logs and checkpoints remain under ignored `results/runs/` paths. Final
checkpoint sidecars were verified:

- A0: `59b3dd03095bd37487423c93b2d958323b9679bd321fc4661f47dc9c5a5b3152`
- A1: `10ca50c2aa30cb856eeb63c9a3feccc9943ddfd4f74723d129f0157ec3221fea`

Estimated FLOPs and energy remain separate from the decision: the committed
analytical FLOP counter is a lower bound and was not used to define equality;
energy was not measured because no synchronized power telemetry was collected.

No A2, scaling, or persistent-memory work was started.
