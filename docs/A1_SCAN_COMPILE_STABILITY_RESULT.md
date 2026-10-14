# A1 scan compilation stability result

Status: complete; engineering-only ten-update in-memory comparison.

Device: NVIDIA GeForce RTX 3060 Ti; PyTorch 2.12.0+cu130; CUDA build 13.0.

| path | warm-up CUDA s | steady CUDA s/update | steady tok/s | peak reserved VRAM |
| --- | ---: | ---: | ---: | ---: |
| eager parallel scan | 0.666718 | 0.364185 | 22494.1 | 1931476992 |
| torch.compile default | 2.103275 | 0.176296 | 46467.2 | 2099249152 |

Steady-state speedup: 2.066x.
Final-parameter relative L2 difference: 0.006791.

Per-update loss and gradient comparisons are in the machine-readable aggregate. This result does not authorize normal-runner integration.

Integrity: both paths completed ten finite updates; no checkpoint or model artifact was written.

## Interpretation

Both paths passed the operational gate: all ten updates completed with finite
losses and gradients, and no CUDA-graph output-lifetime error occurred. The
compiled path passed the 10% performance gate with 2.066x steady-state
speedup, increasing throughput from 22,494 to 46,467 tokens per second.

Numerical drift stayed small over the ten updates: the largest relative loss
difference was about 1.4e-5, the largest relative gradient-norm difference was
about 2.4e-3, and final-parameter relative L2 difference was 0.006791. The
compiled path's peak reserved VRAM included compiler warm-up and was
2,099,249,152 bytes versus 1,931,476,992 for eager, so the preflight's
steady-state memory observation is not generalized to full-run peak memory.

This validates the engineering candidate for a separately gated optional
runner path; it does not change the default A1 path or authorize quality
training with compilation enabled.
