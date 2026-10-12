# A1 decay-tail state/decay forensics

Status: **complete; read-only analysis**. No checkpoint or model parameter was updated.

The same fixed linear value-vector probe used for the original A1 forensics was applied to the three decay-tail checkpoints. The probe is fit at the post-value state for distance 0 and evaluated without refitting after neutral-marker distances.

## Learned time scales

| Checkpoint | Training distance | Median τ | P90 τ | Max τ | Channels τ≥16 | Channels τ≥64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decaytail_distance4 | 4 | 8.926 | 56.600 | 111.648 | 768 | 256 |
| decaytail_distance16 | 16 | 8.936 | 56.781 | 112.002 | 768 | 256 |
| decaytail_distance64 | 64 | 8.905 | 56.845 | 112.155 | 768 | 256 |

The tail survived optimization in a measurable form: every intervention checkpoint retained 768 channels with τ≥16 and 256 channels with τ≥64, whereas the original A1 forensic checkpoints had zero channels at either threshold. The median τ nevertheless remained about 8.9 tokens, so the intervention created a minority long-lived population rather than changing the whole recurrent stack.

## Fixed-probe curve

The table shows the best held-out probe top-1 across recurrent layers. Actual answer top-1 is the model's value-token readout within the reserved 8,000-value pool.

| Checkpoint | Distance | Best layer | Probe top-1 | Probe top-5 | State cosine to d0 | Actual answer top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decaytail_distance4 | 0 | 0 | 0.9961 | 0.9961 | 1.0000 | 0.0020 |
| decaytail_distance4 | 4 | 0 | 0.1953 | 0.3770 | 0.5079 | 0.0039 |
| decaytail_distance4 | 8 | 0 | 0.0098 | 0.0352 | 0.0978 | 0.0000 |
| decaytail_distance4 | 16 | 2 | 0.0039 | 0.0039 | 0.0288 | 0.0000 |
| decaytail_distance4 | 32 | 6 | 0.0020 | 0.0020 | -0.1359 | 0.0000 |
| decaytail_distance4 | 64 | 6 | 0.0020 | 0.0020 | -0.1500 | 0.0000 |
| decaytail_distance16 | 0 | 1 | 0.9883 | 0.9922 | 1.0000 | 0.0000 |
| decaytail_distance16 | 4 | 0 | 0.5352 | 0.7578 | 0.8770 | 0.0000 |
| decaytail_distance16 | 8 | 0 | 0.0273 | 0.0840 | 0.5907 | 0.0000 |
| decaytail_distance16 | 16 | 7 | 0.0020 | 0.0020 | -0.1675 | 0.0000 |
| decaytail_distance16 | 32 | 7 | 0.0020 | 0.0020 | -0.2179 | 0.0000 |
| decaytail_distance16 | 64 | 7 | 0.0000 | 0.0020 | -0.2000 | 0.0000 |
| decaytail_distance64 | 0 | 1 | 0.9961 | 0.9961 | 1.0000 | 0.0000 |
| decaytail_distance64 | 4 | 0 | 0.8438 | 0.9551 | 0.9047 | 0.0000 |
| decaytail_distance64 | 8 | 0 | 0.0332 | 0.1562 | 0.6640 | 0.0000 |
| decaytail_distance64 | 16 | 7 | 0.0020 | 0.0020 | -0.2535 | 0.0020 |
| decaytail_distance64 | 32 | 7 | 0.0020 | 0.0020 | -0.3454 | 0.0000 |
| decaytail_distance64 | 64 | 7 | 0.0020 | 0.0020 | -0.3763 | 0.0000 |

## Interpretation boundary

The intervention clearly changed the learned time-scale distribution, but it did not produce a corresponding fixed-probe recovery at the target long distances: the distance-64-trained checkpoint still has probe top-1 0.0020 at distances 16, 32, and 64. The probe improves at short distances in some arms, especially distance 4 for the distance-64-trained model, while the long-distance trace still collapses. Combined with unchanged held-out exact match, this says that long-lived channels alone are insufficient to solve the present retrieval failure; query interaction, binding, later-layer transformation, or optimization remains limiting.

All measurements are one model size and one seed. They are diagnostic evidence, not a license to replace the A1 architecture or to claim language-model improvement.

## Integrity

Source checkpoints were sidecar-verified, loaded in evaluation mode, and analyzed with gradients disabled. Probe weights were temporary in-memory ridge solutions; source artifacts remain unchanged.
