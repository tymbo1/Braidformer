# A1 scan compilation preflight result

Status: complete; engineering-only in-memory benchmark.

Device: NVIDIA GeForce RTX 3060 Ti; PyTorch 2.12.0+cu130; CUDA build 13.0.

| path | warm-up wall s | measured CUDA s/update | measured tok/s | mean loss | mean grad norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager parallel scan | 0.662 | 0.361625 | 22653.3 | 9.446936 | 1.622054 |
| torch.compile default | 23.265 | 0.171341 | 47811.2 | 9.446775 | 1.619777 |

Steady-state speedup: 2.111x.
Final-parameter relative L2 difference: 0.003981.

This preflight does not authorize runner integration. A separate ten-update stability protocol is required before adopting compilation for normal training.

Integrity: no checkpoint was written, no model artifact was modified, and all losses/gradients were finite.

## Interpretation

The preflight clears the exploratory engineering threshold: default-mode
compilation reduced steady-state update time by 2.111x and reserved VRAM by
about 20%, while keeping all values finite. The mean measured loss delta was
-0.000161 and the final-parameter relative L2 difference was 0.003981 after
the short in-memory comparison.

This is not yet a training result. The 23.265-second compiler warm-up and
trajectory drift require a separate ten-update stability check before adding
an optional compiled path to the normal runner. The CUDA-graph
reduce-overhead mode remains rejected for this eight-microbatch accumulation
pattern because its outputs were overwritten before autograd consumed them.
