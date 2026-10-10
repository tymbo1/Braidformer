# Decisions and environment findings

## 2026-10-10 — D000: isolate the research workspace

All BraidFormer source, environments, datasets, checkpoints, logs, and results stay under `/home/timbushnell/braidformer_lab`. The unrelated `projectbrain_dev` repository is out of scope and must not be inspected or used as a dependency.

## 2026-10-10 — D001: use falsification-first comparisons

The project will not assume BraidFormer-H1 is better. The A0 control and later stages will be compared with matched data, tokenizer, seed policy, optimizer, token budget, batch, context, evaluation, and approximate FLOPs wherever applicable. Claims must stay within measured evidence.

## 2026-10-10 — D002: start with BraidNano

The initial target is approximately 40M–60M parameters. This keeps architecture falsification affordable on the local machine and avoids premature scaling to 1B or 14B parameters.

## 2026-10-10 — D003: defer Qwen distillation

Qwen is excluded from Phase 0. It may later be used as a clearly labeled teacher or baseline only after the core architecture demonstrates basic viability.

## 2026-10-10 — D004: avoid heavyweight Phase 0 setup

Phase 0 uses the already available Python, PyTorch, and pytest installation. It adds no large dependencies, datasets, model checkpoints, or training run.

## 2026-10-10 — D005: audited machine differs from expectation

Read-only PCI metadata identifies an NVIDIA GeForce RTX 3060 Ti GDDR6X (GA104, ASUS subsystem `1043:88ae`, PCI device `10de:24c9`), not an RTX 3070. The loaded kernel driver reports version 580.126.18. `nvidia-smi` fails to communicate with the driver, and PyTorch 2.12.0+cu130 reports `torch.cuda.is_available() == False` with zero visible CUDA devices. The GPU SKU is commonly an 8-GiB device, but VRAM was not treated as measured because the runtime query failed.

## 2026-10-10 — D006: use JSON configs in the bootstrap

The bootstrap uses standard-library JSON configuration loading and validation so Phase 0 does not require a YAML dependency. Later experiment tooling may add a dependency only when its need is recorded and tested.

## 2026-10-10 — D007: implement the A0 control before architectural variants

The next implementation slice is a conventional decoder-only Transformer using the existing BraidNano dimensions. Its model type, dropout, normalization epsilon, bias policy, and tied-embedding policy are explicit configuration fields. The model rejects non-A0 stages, and validation is limited to synthetic CPU smoke tests; no dataset, checkpoint, or training run is authorized by this change.

## 2026-10-10 — D008: keep training gated on data and hardware evidence

The A0 control now has explicit synthetic checks for causal attention and exact next-token loss shifting. Training remains gated: a dataset/tokenizer and deterministic split manifest must be approved, and the device/memory smoke test must pass. The audited machine currently has no usable CUDA device, so no training-time performance or model-quality claim is permitted.

## 2026-10-10 — D009: approve TinyStories preparation, not training

TinyStories is approved as the A0 preparation source at immutable Hugging Face revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, under the source-declared CDLA-Sharing-1.0 terms. A 32,000-token byte-level BPE tokenizer was trained on the deduplicated train split only. Deterministic train/validation/test assignments, source hashes, tokenizer hash, and packed token-stream hashes are recorded and pass local verification. The license is not treated as unrestricted; any redistribution must preserve its terms and attribution.

## 2026-10-10 — D010: bounded training-integrity review

For the 1,000-step pilot, low-risk integrity controls were added for deterministic seeding, provenance hashes, durable step records, non-finite/loss/progress circuit breakers, and atomic checkpoint sidecars. Hyperparameter search, distributed fault testing, security review, and architectural changes are deferred. At the time of this review, CUDA was the sole infrastructure blocker identified by the readiness audit.

## 2026-10-10 — D011: complete the primary A0 pilot before A1

The explicitly authorized seed-17 A0 pilot completed 1,000 optimizer steps and
processed exactly 8,192,000 TinyStories token positions using the approved
configuration. The host RTX 3060 Ti completed the run without OOM, non-finite
losses or gradients, or CUDA errors. A durable step log, validation records,
and SHA-256-verified final checkpoint are preserved under
`results/runs/a0_seed17_pilot/`. Gradient checkpointing remains explicitly
unimplemented and was not claimed as enabled. No A1 work began.

## 2026-10-10 — D012: implement a narrow A1 recurrent control

A1 changes only the sequence-mixing path relative to A0: eight gated diagonal
state-space recurrent mixers replace attention, with zero retained attention
layers, while embeddings, context length, feed-forward widths, normalization,
tied output weights, and the proposed training protocol remain matched. The
implementation has 41,870,336 unique trainable parameters versus A0's
41,866,240, a difference of 4,096 parameters. The generalized runner's A1
dry-run and one approved host-GPU smoke test passed. The smoke test processed
8,192 synthetic token positions in 11.5694 seconds (708.08 tokens/sec), with
1,294,976,512 allocated and 1,526,726,656 reserved bytes, and no non-finite
losses or gradients. This is not a training result or matched A0 benchmark;
the observed sequential recurrence bottleneck is recorded without optimizing
it. A1 training remains blocked pending review and explicit pilot
authorization. No A1 quality or efficiency claim is made.

## 2026-10-10 — D013: preserve A1 reference recurrence and add associative scan

Inspection confirmed that A1's actual gates depend only on the current token
representation and its learned decay is independent of the evolving state.
Therefore its update has the exact affine form `s[t] = a[t] * s[t-1] + b[t]`
and admits an associative prefix scan. The original sequential loop remains the
default reference path; the scan is opt-in and uses the same parameters. CPU
tests, including context 512, verified forward and gradient agreement. The
matched one-update host benchmark measured 981.85 tokens/sec for the reference
path and 22,055.47 tokens/sec for the scan, a 22.46x implementation-path
speedup, with higher scan VRAM and recorded FP16 rounding differences. No A1
training was started, and no architectural features were added.

## 2026-10-10 — D014: select the verified A1 parallel path for the pilot

The bounded ten-update stability comparison initialized the sequential
reference and associative scan from identical A1 weights and empty AdamW
states, then used the same seed-17 batches, FP16 settings, schedule, and
gradient accumulation. Both paths completed all ten updates with finite losses
and gradients. Losses differed by at most `6.795e-5`; final parameter
relative-L2 difference was `0.141%`; final full-gradient relative-L2 difference
was `4.96%`, recorded as FP16 operation-order variation. The reference path
measured `998.59 tokens/sec` and the scan `20,433.84 tokens/sec` over the
ten-update run, with separate peak allocated VRAM of `1,392,904,192` and
`1,656,395,776` bytes respectively. The explicit A1 configuration now selects
`parallel_scan`; the sequential loop remains available for diagnostics. This
was a bounded stability check, not A1 training, and final pilot authorization
remains outstanding.

## 2026-10-10 — D015: complete the authorized A1 seed-17 pilot

The explicitly authorized A1 pilot used the committed `parallel_scan` path,
the approved TinyStories revision and tokenizer, seed 17, context 512,
micro-batch 2, accumulation 8, AdamW, cosine learning rate with 10 warmup
steps, and 1,000 optimizer steps. It processed exactly 8,192,000 token
positions in `404.41` seconds of training time (`20,256.62 tokens/sec`) and
`436.41` seconds elapsed. Final training loss was `3.1820`; final validation
loss was `3.2635`. Peak allocated/reserved VRAM was `1,657,214,976 /
1,931,476,992` bytes. No non-finite values, CUDA failures, or OOM events
occurred. The final checkpoint and optimizer/RNG state were SHA-256 verified.

At equal tokens, A1's validation loss was `0.4877` below A0's seed-17 result,
but A1 took `1.94x` the training wall time and had `51.5%` of A0's measured
throughput. The result therefore supports only a single-seed equal-token
observation; no equal-compute efficiency claim is made. A0 artifacts remain
unchanged, and A2, scaling, and persistent-memory work remain deferred.

## 2026-10-10 — D016: evaluate A1 generation and define an equal-compute test

The preserved A1 checkpoint was evaluated with the exact four A0 prompts,
greedy argmax decoding, and 64-token generation limit. The run completed in
`1.5696` seconds with `221,171,200 / 241,172,480` allocated/reserved bytes;
A0 took `1.0436` seconds under the same method. The outputs are recorded in
`results/a1_prompt_eval.json` and are qualitative only. A bounded follow-up
proposal now separates equal token positions, equal host training wall time,
and equal measured CUDA-event GPU execution time. No additional training was
started. The subsequent bounded CUDA calibration and protocol are recorded in
D017 below.

## 2026-10-10 — D017: establish a measured equal-CUDA-compute protocol

A bounded host-GPU calibration used two warm-up and twenty measured in-memory
optimizer updates per arm with device-resident approved batches. It preserved
and verified the existing A0/A1 checkpoint hashes, produced finite losses and
gradients, and discarded all updates. Mean CUDA-event time was `0.202106895`
seconds for A0 and `0.402105196` seconds for A1; mean host time for the same
scope was `0.202128821` and `0.402126285` seconds respectively. The resulting
fixed planning budget is `202.106895447` CUDA seconds: A0 uses 1,000 steps and
A1 uses 502 steps, processing 8,192,000 and 4,112,384 token positions. The
calibration record is `results/a0_a1_cuda_timing_calibration.json`.

This is a proposal only. Any authorized equal-compute run must start both arms
from fresh seed-17 initialization, use normalized cosine schedules with 10 A0
warmup steps and 6 A1 warmup steps, validate at the declared 100-step
checkpoints plus each final step, and record actual CUDA time, host time,
estimated FLOPs, and checkpoint integrity separately. Estimated FLOPs are not
the budget, and energy was not measured. Existing A0/A1 pilots, checkpoints,
configurations, and results remain unchanged; no additional training began.

## 2026-10-10 — D018: complete the authorized equal-CUDA-time comparison

The committed protocol `de41ef2` was executed with fresh seed-17 A0 and A1
initialization in isolated run directories. A0 completed 1,000 steps and
8,192,000 token positions; A1 completed 502 steps and 4,112,384 positions.
Accumulated CUDA-event time was `209.3490` seconds for A0 and `205.3696`
seconds for A1, both within the predeclared ±5% of the `202.1069`-second
target. Final validation loss was `3.7396816` for A0 and `3.5777884` for A1,
giving `Delta = -0.1618933`. A1 therefore meets the predeclared exploratory
advantage rule for this single-seed TinyStories equal-CUDA-time comparison.

Both runs completed with finite values, no CUDA/OOM failures, and verified final
checkpoints. The original A0/A1 pilots and checkpoints remain unchanged. This
does not establish statistical significance or general architectural
superiority; replication and broader evaluation remain separate future work.

## 2026-10-10 — D019: complete seed-23 and seed-29 replications

Two independent fresh-initialization replications were executed with the
accepted equal-CUDA-time protocol. Seeds 23 and 29 completed both A0 (1,000
steps, 8,192,000 tokens) and A1 (502 steps, 4,112,384 tokens), with all four
actual CUDA totals within the predeclared ±5% budget tolerance and all final
checkpoints verified. Their validation-loss differences were `-0.2007560` and
`-0.1503594`; including accepted seed 17 (`-0.1618933`), the mean difference
was `-0.1710029` with sample standard deviation `0.0264044`. A1 met the
`Delta <= -0.05` advantage threshold in all three seeds. This is reproducible
evidence for the specified TinyStories equal-CUDA-time protocol, not a general
architectural-superiority claim. No A2, scaling, or persistent-memory work
started.

## 2026-10-10 — D020: freeze the first A0/A1 milestone and assess A2 options

The three-seed A0/A1 equal-CUDA-time milestone is frozen. Commit `00bb31a` was
audited for tracked checkpoints, raw logs, and secrets and pushed to
`origin/main`. No A0/A1 source, configuration, checkpoint, or result was
modified. A focused equation-level mapping against S4D, Mamba, RetNet, and
RWKV found established family-level similarities but did not establish
architectural novelty for A1. The bounded next-step assessment is recorded in
`docs/A2_OPTIONS_ASSESSMENT.md`: longer matched-CUDA training, mathematics-
preserving A1 execution work, and a dual-state design review. No A2
implementation, new training, scaling, or persistent-memory integration was
authorized.

## 2026-10-10 — D021: freeze the 4× seed-17 matched-CUDA protocol

Option 1 was frozen for one fresh seed-17 A0/A1 experiment before execution.
The protocol targets `808.427581787` CUDA seconds per arm, with A0 at 4,000
steps / 32,768,000 tokens and A1 at 2,010 steps / 16,465,920 tokens. It keeps
the approved TinyStories data, tokenizer, context, effective batch, AdamW
settings, normalized cosine schedules, A1 `parallel_scan`, and ±5% CUDA-time
acceptance tolerance. A0 quarter-budget validation anchors are steps 1,000,
2,000, 3,000, and 4,000; A1 anchors are 500, 1,000, 1,500, and 2,010, with
the existing 100-step validation interval retained. Protocol-specific derived
configs preserve the original A0/A1 config files and hashes. No additional
seeds or later architecture work is authorized by this decision.
