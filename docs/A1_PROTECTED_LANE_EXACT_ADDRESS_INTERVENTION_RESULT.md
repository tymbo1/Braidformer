# A1 protected-lane exact-address intervention result

Status: **complete; inference-only frozen-checkpoint intervention**.

Protocol: `docs/A1_PROTECTED_LANE_EXACT_ADDRESS_PROTOCOL.md` (sha256 `9a7a7b0105eeefe8032e2b77d24b2e382876445c444ee269fece2d2bdd95cc43`).

The exact-address path replaced only the learned key-similarity/read gate at the answer marker. It retained each checkpoint's original stored value state and trained readout.

| distance | mode | native value CE | exact-address value CE | native value rank | exact-address value rank | native top-1 | exact top-1 | CE delta | rank delta | target-logit delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | protected | 9.4943 | 9.4941 | 2749.1 | 2749.0 | 0.0391 | 0.0391 | -0.0002 | -0.1 | 0.0002 |
| 0 | ordinary | 9.5205 | 9.5199 | 2747.0 | 2746.9 | 0.0391 | 0.0391 | -0.0007 | -0.1 | 0.0008 |
| 4 | protected | 9.6847 | 9.6848 | 3099.2 | 3098.9 | 0.0078 | 0.0078 | 0.0001 | -0.3 | -0.0001 |
| 4 | ordinary | 9.9586 | 9.9659 | 3094.0 | 3093.7 | 0.0078 | 0.0078 | 0.0073 | -0.3 | -0.0079 |
| 16 | protected | 10.4107 | 10.4102 | 3953.0 | 3952.0 | 0.0000 | 0.0000 | -0.0005 | -1.0 | 0.0006 |
| 16 | ordinary | 9.9048 | 10.0224 | 3844.0 | 3841.9 | 0.0000 | 0.0000 | 0.1176 | -2.1 | -0.1467 |
| 64 | protected | 10.3191 | 10.3172 | 4004.3 | 4003.4 | 0.0000 | 0.0000 | -0.0019 | -0.9 | 0.0028 |
| 64 | ordinary | 10.1936 | 10.3566 | 3947.1 | 3955.6 | 0.0000 | 0.0000 | 0.1630 | 8.5 | -0.1804 |

## Integrity

All 8 final checkpoints and sidecars were verified. Canonical exact-address gates were one for every held-out example. Native functional logits matched the model forward within the recorded tolerance, all tensors and metrics were finite, parameter digests were unchanged, and no checkpoint was written.

Interpretation is governed by the frozen protocol: this is a mechanistic oracle-like address isolation, not a production architecture or a training result.

## Interpretation

The exact address did not rescue the protected lane. Its value-range top-1
accuracy stayed at `3.90625%`, `0.78125%`, `0%`, and `0%` at distances 0, 4,
16, and 64; protected value cross-entropy changed by only `-0.0002`,
`+0.0001`, `-0.0005`, and `-0.0019`. The stored protected value remained
cosine `1.0000` to its post-write state, so the null is not a storage
persistence failure or a missing exact key match.

The ordinary lane also had no top-1 rescue and was worse at distances 4, 16,
and 64 under exact addressing. The result therefore closes the narrow
learned-address-gate hypothesis: even an exact record-key/query-key oracle
does not make the preserved value usable. Binding/representation and
query-conditioned or downstream value integration remain the stronger target;
no training or architecture intervention is unlocked.
