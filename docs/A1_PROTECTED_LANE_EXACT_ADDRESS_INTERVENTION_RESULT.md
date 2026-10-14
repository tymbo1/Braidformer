# A1 protected-lane exact-address intervention result

Status: **complete; inference-only frozen-checkpoint intervention**.

Protocol: `docs/A1_PROTECTED_LANE_EXACT_ADDRESS_PROTOCOL.md` (sha256 `9a7a7b0105eeefe8032e2b77d24b2e382876445c444ee269fece2d2bdd95cc43`).

Execution: host CUDA on the NVIDIA GeForce RTX 3060 Ti; no checkpoint or
weight was written.

The exact-address path replaced only the learned key-similarity/read gate at the answer marker. It retained each checkpoint's original stored value state and trained readout.

| distance | mode | native value CE | exact-address value CE | native value rank | exact-address value rank | native top-1 | exact top-1 | CE delta | rank delta | target-logit delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | protected | 9.4943 | 9.4941 | 2744.9 | 2744.9 | 0.0391 | 0.0391 | -0.0002 | 0.0 | 0.0002 |
| 0 | ordinary | 9.5206 | 9.5200 | 2743.2 | 2743.0 | 0.0391 | 0.0391 | -0.0006 | -0.2 | 0.0008 |
| 4 | protected | 9.6847 | 9.6848 | 3095.0 | 3094.4 | 0.0078 | 0.0078 | 0.0000 | -0.6 | -0.0001 |
| 4 | ordinary | 9.9585 | 9.9658 | 3089.4 | 3087.9 | 0.0078 | 0.0078 | 0.0072 | -1.4 | -0.0078 |
| 16 | protected | 10.4107 | 10.4101 | 3948.7 | 3947.7 | 0.0000 | 0.0000 | -0.0006 | -0.9 | 0.0006 |
| 16 | ordinary | 9.9049 | 10.0224 | 3839.7 | 3837.7 | 0.0000 | 0.0000 | 0.1175 | -2.1 | -0.1467 |
| 64 | protected | 10.3191 | 10.3172 | 4000.0 | 3999.3 | 0.0000 | 0.0000 | -0.0019 | -0.7 | 0.0028 |
| 64 | ordinary | 10.1938 | 10.3569 | 3942.8 | 3951.1 | 0.0000 | 0.0000 | 0.1630 | 8.3 | -0.1804 |

## Integrity

All 8 final checkpoints and sidecars were verified. Canonical exact-address gates were one for every held-out example. Native functional logits matched the model forward within the recorded tolerance, all tensors and metrics were finite, parameter digests were unchanged, and no checkpoint was written.

Interpretation is governed by the frozen protocol: this is a mechanistic oracle-like address isolation, not a production architecture or a training result.

## Interpretation

The exact address did not rescue the protected lane. Its value-range top-1
accuracy stayed at `3.90625%`, `0.78125%`, `0%`, and `0%` at distances 0, 4,
16, and 64; protected value cross-entropy changed by only `-0.0002`,
`+0.0000`, `-0.0006`, and `-0.0019`. The stored protected value remained
cosine `1.0000` to its post-write state, so the null is not a storage
persistence failure or a missing exact key match.

The ordinary lane also had no top-1 rescue and was worse at distances 4, 16,
and 64 under exact addressing. The result therefore closes the narrow
learned-address-gate hypothesis: even an exact record-key/query-key oracle
does not make the preserved value usable. Binding/representation and
query-conditioned or downstream value integration remain the stronger target;
no training or architecture intervention is unlocked.
