# A1 state/decay forensics on the 8000-value retrieval task

Status: **complete; read-only analysis**. No checkpoint or model parameter was updated.

The analysis uses the completed distance-0, distance-4, distance-16, and distance-64 A1 checkpoints. It extracts the learned diagonal decay scales, reconstructs internal recurrent states from forward hooks, fits a fixed ridge probe only at the post-value distance-0 state, and evaluates that same probe after neutral-marker distances.

## Learned time scales

| Checkpoint | Training distance | Median τ | P90 τ | Max τ | Channels τ≥16 | Channels τ≥64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| distance0 | 0 | 8.896 | 8.929 | 8.979 | 0 | 0 |
| distance4 | 4 | 8.919 | 8.943 | 8.987 | 0 | 0 |
| distance16 | 16 | 8.935 | 8.956 | 9.013 | 0 | 0 |
| distance64 | 64 | 8.896 | 8.930 | 9.017 | 0 | 0 |

The τ values are characteristic decay times for `state[t] = a state[t−1] + ...`, computed as `−1/log(a)`. They describe the learned transition coefficients, not a guarantee that information survives for exactly τ tokens because input gates, residual paths, feed-forward blocks, and later layers also contribute.

## Frozen-state probe summary

The table reports the best held-out value-vector probe accuracy across the eight recurrent layers at each distance. The probe is fit once at distance 0 and is not refit at later distances; this prevents later positions from hiding decay through a newly adapted decoder.

| Checkpoint | Distance | Best layer | Probe top-1 | Probe top-5 | Mean state cosine to d0 | Actual answer top-1 | Mean answer rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distance0 | 0 | 0 | 0.9980 | 0.9980 | 1.0000 | 0.0176 | 2832.4 |
| distance0 | 4 | 0 | 0.0938 | 0.2012 | 0.6459 | 0.0000 | 3556.3 |
| distance0 | 8 | 2 | 0.0039 | 0.0098 | 0.4570 | 0.0000 | 3637.4 |
| distance0 | 16 | 7 | 0.0000 | 0.0020 | 0.4424 | 0.0000 | 3838.7 |
| distance0 | 32 | 7 | 0.0000 | 0.0000 | 0.3875 | 0.0000 | 3924.0 |
| distance0 | 64 | 7 | 0.0000 | 0.0000 | 0.3708 | 0.0000 | 3896.8 |
| distance4 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0020 | 3506.4 |
| distance4 | 4 | 0 | 0.1836 | 0.4668 | 0.4823 | 0.0059 | 3200.0 |
| distance4 | 8 | 0 | 0.0137 | 0.0371 | 0.0740 | 0.0000 | 3586.7 |
| distance4 | 16 | 1 | 0.0020 | 0.0020 | -0.0067 | 0.0020 | 3896.7 |
| distance4 | 32 | 7 | 0.0000 | 0.0000 | -0.0605 | 0.0000 | 3990.9 |
| distance4 | 64 | 7 | 0.0000 | 0.0000 | -0.0972 | 0.0000 | 4022.8 |
| distance16 | 0 | 1 | 0.9922 | 0.9941 | 1.0000 | 0.0000 | 3824.4 |
| distance16 | 4 | 0 | 0.4707 | 0.7324 | 0.8702 | 0.0000 | 3776.1 |
| distance16 | 8 | 0 | 0.0098 | 0.0332 | 0.5794 | 0.0020 | 3763.4 |
| distance16 | 16 | 7 | 0.0020 | 0.0020 | -0.2719 | 0.0020 | 3841.4 |
| distance16 | 32 | 7 | 0.0000 | 0.0020 | -0.3607 | 0.0020 | 3960.9 |
| distance16 | 64 | 7 | 0.0000 | 0.0000 | -0.3554 | 0.0000 | 3934.2 |
| distance64 | 0 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 3990.7 |
| distance64 | 4 | 0 | 0.5195 | 0.7324 | 0.8979 | 0.0000 | 3938.4 |
| distance64 | 8 | 0 | 0.0078 | 0.0117 | 0.6615 | 0.0000 | 3959.9 |
| distance64 | 16 | 0 | 0.0059 | 0.0059 | 0.2410 | 0.0000 | 4063.3 |
| distance64 | 32 | 0 | 0.0020 | 0.0039 | 0.0374 | 0.0000 | 4038.9 |
| distance64 | 64 | 5 | 0.0020 | 0.0020 | -0.1404 | 0.0000 | 3949.4 |

## Interpretation boundary

A falling fixed-probe score together with falling state cosine would support loss or transformation of the binding trace inside the recurrent stack. A stable probe score with poor actual answer readout would instead point toward query interaction or readout failure. These measurements are diagnostic signatures, not causal proof: the probe is linear, its decoder is fit at distance 0, and this remains one model size and one seed.

No longer-decay initialization, architectural change, target-length run, relational curriculum, or A2 intervention was performed. The next action should be selected from the measured signature rather than assumed in advance.

## Integrity

Each source checkpoint was sidecar-verified before loading. The model was placed in evaluation mode, gradients were disabled, and only temporary probe weights were fitted in memory. Source checkpoint files remain unchanged.
