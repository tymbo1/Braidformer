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

## 2026-10-10 — D022: complete the 4× seed-17 matched-CUDA experiment

The frozen Option 1 protocol completed one fresh run per architecture. A0
processed 32,768,000 tokens in 4,000 steps and accumulated `839.6201` CUDA
seconds; A1 processed 16,465,920 tokens in 2,010 steps and accumulated
`817.6418` CUDA seconds. Both totals were within the predeclared ±5% of the
`808.4276`-second target. Final validation losses were `3.1149070` for A0 and
`3.0030669` for A1, giving `A1 - A0 = -0.1118402`, which meets the exploratory
`<= -0.05` rule for this single seed. Both runs completed with finite values,
no OOM/CUDA failures, and SHA-256-verified final checkpoints. Full curves and
resource records are in `docs/A0_A1_LONGER_EQUAL_CUDA_RESULT.md` and
`results/a0_a1_longer_equal_cuda_seed17.json`. No additional seed was started.

## 2026-10-10 — D023: complete the 4× seed-23 and seed-29 replications

The frozen 4× matched-CUDA protocol was replicated with fresh seed-23 and
seed-29 A0/A1 initializations. All four new arms completed their declared
budgets, and all six seed-17/23/29 arm totals were within the predeclared ±5%
CUDA-time tolerance. Final validation-loss differences `A1 - A0` were
`-0.1118402`, `-0.1256531`, and `-0.1242971`, respectively. The three-seed
mean was `-0.1205968` with sample standard deviation `0.0076137`; A1 met the
`<= -0.05` criterion in all three seeds. Every run had finite values and a
SHA-256-verified optimizer/RNG checkpoint. A1 remained slower per token and
used more VRAM. This supports the result only for the specified TinyStories
42M-parameter measured-CUDA protocol; it is not a general superiority,
energy-efficiency, FLOP-efficiency, or novelty claim. No 8× run, A2, scaling,
or persistent-memory work started.

## 2026-10-11 — D024: complete the 8× matched-CUDA-time replication

The frozen 8× protocol in commit `188327e` was executed with fresh seed-17,
seed-23, and seed-29 A0/A1 initialization. A0 used 8,000 steps / 65,536,000
tokens and A1 used 4,020 steps / 32,931,840 tokens, with measured CUDA-time
calibration determining the A1 budget. All six runs completed within the
predeclared ±5% of the `1,616.855164`-second target, with finite values, no
CUDA/OOM failures, and complete SHA-256-verified optimizer/RNG checkpoints.
Final A1-minus-A0 validation-loss differences were `-0.2422227`, `-0.2474567`,
and `-0.2461745`; the mean was `-0.2452846` with sample standard deviation
`0.0027281`, meeting the `<= -0.05` criterion in all three seeds.

The read-only curve analysis used cumulative CUDA time, piecewise-linear
interpolation between observed validation points, and the frozen common-
interval AUC definition. All nine budget/seed AUCs were valid. Relative to
the prior means, the advantage narrowed from 1× (`-0.1710029`) to 4×
(`-0.1205968`) and widened at 8× (`-0.2452846`), so the three-budget pattern
is non-monotonic rather than evidence for a crossover or asymptotic trend.
Raw logs and checkpoints remain ignored and local; the tracked report and
machine-readable results are `docs/A0_A1_EIGHT_EQUAL_CUDA_RESULT.md` and
`results/a0_a1_eight_equal_cuda_replications.json`. No A2, scaling,
persistent-memory, or unrelated architecture work started, and nothing was
pushed to GitHub.

## 2026-10-11 — D025: freeze the schedule-disentanglement protocol

The 8× forensic analysis found that A1's advantage widens progressively across
the later CUDA-time regions in all three seeds, while also identifying a
learning-rate trajectory confound. A token-normalized and optimizer-step-
normalized schedule are not independent under the frozen 8,192-token-per-step
geometry. The bounded follow-up therefore defines three conditions: the
completed native 8× reference, a fresh 8× CUDA-fraction schedule control, and
a fresh equal-token/equal-step 4,020-step shared schedule control.

The new CUDA-referenced scheduler and four planned-only configs are recorded in
`docs/A0_A1_SCHEDULE_DISENTANGLEMENT_PROTOCOL.md`. It preserves both
architectures and all approved data/training controls. The protocol is frozen
for review, but no new training has started; execution requires explicit
authorization after the exact conditions are presented. No 16×, A2, scaling,
persistent-memory, or unrelated architecture work is included.

## 2026-10-11 — D026: complete schedule disentanglement

The frozen protocol in `eb1cb6a` was executed with fresh seeds 17, 23, and 29
for both controls: twelve runs in total. The CUDA-fraction schedule control
used 8,000 A0 steps / 65,536,000 tokens and 4,020 A1 steps / 32,931,840
tokens under a shared CUDA-progress cosine schedule. All six CUDA totals passed
the ±5% tolerance; the mean final delta was `-0.2325530` with sample SD
`0.0104430`, and A1 met the `<= -0.05` criterion in all three seeds.

The equal-token/equal-step control used 4,020 steps / 32,931,840 tokens for
both arms under the identical 41-step cosine schedule. The mean final delta
was `-0.2975742` with sample SD `0.0240075`, again meeting the criterion in
all three seeds. A1 used approximately 1.96× A0's CUDA time in this condition,
so it is an equal-token/update result rather than an equal-compute result.

All twelve final optimizer/RNG checkpoints passed the existing SHA-256
integrity and metadata checks; no run reported numerical, CUDA, or OOM failure.
Relative to native 8× mean delta `-0.2452846`, the CUDA-fraction control
retained a slightly smaller but still consistent advantage. The complete
evidence is recorded in
`docs/A0_A1_SCHEDULE_DISENTANGLEMENT_RESULT.md` and
`results/a0_a1_schedule_disentanglement_results.json`. These results remain
specific to the declared TinyStories 42M-parameter protocols and do not
establish general architectural superiority, matched FLOP/energy efficiency,
or novelty. No further experiment was started.

## 2026-10-11 — D027: complete the synthetic recall and retrieval battery

The frozen synthetic battery protocol in commit
`a0f2a95a65c696053d9b2d98b4f6106d421c0daa` was executed with fresh A0/A1
initialization for seeds 17, 23, and 29. Each of the six runs used 1,000
optimizer steps and 8,192,000 synthetic token positions with the unchanged
42M-parameter architectures, context 512, micro-batch 2, accumulation 8,
AdamW, and the identical 10-step-warmup cosine schedule. The tasks were exact
copying, eight-record associative recall, and thirty-two-record selective
retrieval.

The primary autoregressive greedy test produced zero exact whole-answer
matches for both architectures on every task and seed. Mean answer-token
accuracy was `0.000109` for A0 and `0.000000` for A1; this budget therefore
does not demonstrate successful exact recall or a primary-metric winner.
As a secondary diagnostic, A0 had lower target-only test loss in all nine
seed/task pairs: A0 mean `9.341637`, A1 mean `9.686330`, paired delta
`A1 - A0 = +0.344694` with sample SD `0.129459`. A1 used mean `385.549`
CUDA seconds versus A0's `195.319` seconds, a `1.974x` ratio, for the same
token/update budget. All six runs completed with finite values, no CUDA/OOM
failure, and SHA-256-verified optimizer/RNG checkpoints; gradient checkpointing
remains unimplemented and disabled.

This is a bounded negative result for this synthetic generator and budget. It
does not generalize beyond the declared tasks or overturn the prior TinyStories
results. The tracked evidence is in
`docs/A0_A1_SYNTHETIC_RECALL_RESULT.md` and
`results/a0_a1_synthetic_recall_battery.json`; raw logs and checkpoints remain
ignored. No A2, long-context, second-corpus, scaling, post-training,
persistent-memory, WNF, or SNAIL/SSRE work started.

## 2026-10-11 — D028: complete the seed-17 solvability calibration

The frozen first-rung calibration protocol in commit `8a34cda` was executed
with fresh seed-17 initialization for the unchanged 41,866,240-parameter A0
Transformer and 41,870,336-parameter A1 `parallel_scan` model. Each run used
2,000 optimizer steps and 256,000 sequence-token positions on the fixed
one-key/one-value task. Both models reached `1.0000` autoregressive greedy
exact match on the repeated 128-example training set, so neither model failed
the trivial memorization calibration.

The held-out set was diagnostic only: it used disjoint keys and mappings while
retaining the earlier record value needed for the same copy/retrieve operation.
A0 reached `0.8828` exact match and target loss `1.222888`; A1 reached
`0.8516` and `1.509632`. The A1-minus-A0 held-out target-loss difference was
`+0.286744`, with A1 using `1.519x` A0's CUDA time (`283.746` versus
`186.852` seconds). This calibration was not a matched-compute comparison and
does not isolate model capacity, recurrent state capacity, or optimization.

Both runs completed with finite values, no CUDA/OOM failure, and SHA-256-
verified optimizer/RNG checkpoints. The tracked evidence is in
`docs/A0_A1_SOLVABILITY_CALIBRATION_RESULT.md`,
`results/a0_a1_solvability_calibration.json`, and the read-only verifier
`scripts/analyze_synthetic_overfit_calibration.py`; raw logs and checkpoints
remain ignored. No seeds 23/29, harder ladder rung, scaling, A2, persistent
memory, WNF, or SNAIL/SSRE work started.

## 2026-10-12 — D029: freeze the 128-value solvability-ladder rung for review

The completed first-rung calibration showed that both unchanged 42M models can
memorize the one-association task, while A0 had lower held-out target loss than
A1 at seed 17. The next bounded rung therefore keeps the sequence, one
association, 128 fixed training examples, optimizer, 2,000-step budget,
architecture, and full 32,000-token output vocabulary unchanged while expanding
the reserved answer-token set from 64 to 128. No winner threshold is declared;
the primary observation remains separate training-set autoregressive exact
match, with held-out performance diagnostic only.

The planned A0/A1 configurations are
`configs/braidnano_a0_solvability_values128.json` and
`configs/braidnano_a1_solvability_values128.json`. The protocol is recorded in
`docs/A0_A1_SOLVABILITY_VALUES128_PROTOCOL.md`; it requires atomic checkpoints
every 100 steps and resumes from a verified interior checkpoint into a fresh
run directory after an interruption. Unit tests and dry-run validation pass.
This freezes the protocol for review only: no seed-17 training, additional
seeds, higher value pools, multiple associations, distractors, longer answers,
scaling, A2, or memory integration is authorized by this decision.

## 2026-10-12 — D030: complete the 128-value solvability-ladder rung

After the GPU device nodes were restored following the host power outage, the
frozen protocol in commit `68fbfd3` was executed with fresh seed-17 A0 and A1
initialization. Both unchanged 42M models completed 2,000 optimizer steps and
256,000 token positions, reached `1.0000` autoregressive greedy exact match on
the repeated 128-example training set, and produced SHA-256-verified final
checkpoints with finite values and no CUDA/OOM failure.

On the diagnostic held-out set, A0 reached `0.6484` exact match and target loss
`3.488070`; A1 reached `0.5938` and `4.252668`. The paired A1-minus-A0
held-out target-loss delta was `+0.764598`, while A1 used `1.583x` A0's CUDA
time (`288.644` versus `182.390` seconds). This is evidence of an A0 advantage
on this explicit one-association retrieval diagnostic at one seed and budget,
not a matched-compute or general architectural claim. The held-out set is
diagnostic because keys and mappings are disjoint while the earlier record
value remains in the input.

The tracked evidence is in
`docs/A0_A1_SOLVABILITY_VALUES128_RESULT.md`,
`results/a0_a1_solvability_values128.json`, and the read-only verifier
`scripts/analyze_synthetic_solvability_values128.py`; raw logs and checkpoints
remain ignored. No higher value pool, additional seed, multiple-association
rung, scaling, A2, persistent memory, WNF, or SNAIL/SSRE work started.

## 2026-10-12 — D031: record a deferred trajectory-aware training-policy test

The project will later test whether a compact cumulative training-state
predictor can estimate the learning utility of a candidate batch and reduce
unnecessary backward passes. The first useful target is not literal next-batch
text prediction: a shuffled stream makes that target mostly unidentifiable.
The initial target is expected reducible loss, learning utility, surprise, or
future forgetting conditioned on prior training history and the candidate
batch.

The proposal is recorded in
`docs/TRAJECTORY_AWARE_TRAINING_PROPOSAL.md` and added as the deferred T1
training-policy side track in `docs/EXPERIMENT_PROTOCOL.md`. Its first
comparison is planned as A1-normal, current-loss-gated, and
trajectory-predictive-gated arms with the same model, data, seed, token
budget, optimizer, validation set, and measured resource accounting. It must
retain an exploration/audit floor and report false skips, coverage, predictor
overhead, validation quality, CUDA time, and wall time.

This entry authorizes no implementation, data reordering, training, model
self-description, provenance-ledger integration, or architecture change. A
future execution requires a versioned protocol, leakage audit, cheap synthetic
validation, explicit quality/compute thresholds, and separate approval.

## 2026-10-12 — D032: freeze the 256-value solvability-ladder rung

The 64-value calibration and 128-value rung both passed the primary training
memorization criterion, while A0 remained ahead of A1 on the diagnostic
held-out copy/retrieve measure. The next test therefore continues the same
calibrated synthetic-recall ladder: expand only the reserved answer-token pool
from 128 to 256, preserving the one association, 128 fixed examples, model
architectures, seed 17, optimizer, 2,000-step budget, 256,000 token
positions, and evaluation/checkpoint cadence.

The frozen configurations are
`configs/braidnano_a0_solvability_values256.json` and
`configs/braidnano_a1_solvability_values256.json`; the protocol is
`docs/A0_A1_SOLVABILITY_VALUES256_PROTOCOL.md`. Only the two fresh seed-17
arms are in scope. Values1024/8000, association-count changes,
distractors, target-length changes, decay analysis, T1 predictive training,
scaling, A2, and memory work remain outside this authorization boundary.

## 2026-10-12 — D033: complete the 256-value solvability-ladder rung

The frozen protocol in commit `5b02db5` was executed with fresh seed-17 A0 and
A1 initialization. Both unchanged 42M models completed 2,000 optimizer steps
and 256,000 token positions, reached `1.0000` autoregressive greedy exact
match on the repeated 128-example training set, and produced finite values
with SHA-256-verified final checkpoints and no CUDA/OOM failure.

On the diagnostic held-out set, A0 reached `0.4297` exact match and target
loss `5.813058`; A1 reached `0.3516` and `6.850369`. The paired A1-minus-A0
held-out target-loss delta was `+1.037312`, and the exact-match delta was
`-0.0781`. Relative to values128, held-out exact match fell from `0.6484` to
`0.4297` for A0 and from `0.5938` to `0.3516` for A1, while both still
memorized training perfectly. A1 used `1.544x` A0's CUDA time (`285.427`
versus `184.821` seconds).

This locates a stronger difficulty effect and a wider A0 diagnostic advantage,
but not a complete capability boundary: neither model failed the training
memorization criterion and held-out exact match remained nonzero. The tracked
evidence is in `docs/A0_A1_SOLVABILITY_VALUES256_RESULT.md`,
`results/a0_a1_solvability_values256.json`, and the read-only verifier
`scripts/analyze_synthetic_solvability_values256.py`; raw logs and checkpoints
remain ignored. No values1024/8000 rung, association-count change,
distractor-distance change, target-length change, decay analysis, predictive
training, scaling, A2, or memory work started.

## 2026-10-12 — D034: refine the deferred T1 hypothesis

The deferred trajectory-aware training-policy idea is refined as **predictive
learning-state modeling**. Its stronger target is not literal next-batch text
prediction, which can learn dataset order rather than learning dynamics. A
future controller should instead predict a compact future learning state,
expected reducible loss, useful parameter/gradient change, competence change,
or forgetting from cumulative training history and candidate data.

The longer-horizon “learning inheritance” concept—passing a validated
curriculum map, competence/failure map, or future-state predictor from one
model generation to a successor—remains a separate extension. It must be
tested only after the single-generation normal/loss-gated/trajectory-gated
comparison, with shuffled-order and withheld/scrambled-inheritance controls.
This refinement authorizes no implementation, training, self-model, or
cross-generation experiment.

## 2026-10-12 — D035: freeze the 1024-value solvability-ladder rung

The completed values256 rung preserved perfect training memorization for both
models while reducing held-out exact match to `0.4297` for A0 and `0.3516` for
A1. The next calibrated test therefore expands only the reserved answer-token
pool from 256 to 1,024, keeping the one association, 128 fixed examples,
architectures, seed 17, optimizer, 2,000-step budget, 256,000 token
positions, and checkpoint/evaluation cadence unchanged.

The frozen configurations are
`configs/braidnano_a0_solvability_values1024.json` and
`configs/braidnano_a1_solvability_values1024.json`; the protocol is
`docs/A0_A1_SOLVABILITY_VALUES1024_PROTOCOL.md`. Only the two fresh seed-17
arms are in scope. Values8000, association-count changes, distractors,
target-length changes, decay analysis, predictive learning-state/inheritance,
scaling, A2, and memory work remain outside this authorization boundary.

## 2026-10-12 — D036: complete the 1024-value solvability-ladder rung

The frozen protocol in commit `18025c1` was executed with fresh seed-17 A0 and
A1 initialization. Both unchanged 42M models completed 2,000 optimizer steps
and 256,000 token positions, reached `1.0000` autoregressive greedy exact
match on the repeated 128-example training set, and produced finite values
with SHA-256-verified final checkpoints and no CUDA/OOM failure.

On the diagnostic held-out set, A0 reached `0.1406` exact match and target
loss `8.802002`; A1 reached `0.0703` and `9.573342`. The paired A1-minus-A0
held-out target-loss delta was `+0.771340`, with an exact-match delta of
`-0.0703`. Relative to values256, held-out exact match fell from `0.4297` to
`0.1406` for A0 and from `0.3516` to `0.0703` for A1. A1 used `1.573x` A0's
CUDA time (`287.715` versus `182.948` seconds).

This is a substantial shared difficulty effect, but not yet a training-
memorization boundary: both models still reached perfect training exact match.
The A0 held-out advantage remained, while narrowing slightly from values256.
The tracked evidence is in
`docs/A0_A1_SOLVABILITY_VALUES1024_RESULT.md`,
`results/a0_a1_solvability_values1024.json`, and the read-only verifier
`scripts/analyze_synthetic_solvability_values1024.py`; raw logs and
checkpoints remain ignored. No values8000 rung, association-count change,
distractor-distance change, target-length change, decay analysis, predictive
learning-state, competence-ledger, scaling, A2, or memory work started.

## 2026-10-12 — D037: define the future competence-ledger safeguards

The later predictive-learning/inheritance branch may maintain a cumulative
competence ledger covering successes, failures, mastery, forgetting,
fail-to-pass transitions, pass-to-fail transitions, and useful interventions.
This must use `Train | Meta/Development | Final Test`: the development split
may update a curriculum or inherited artifact, while the final test remains
sealed until evaluation. Future successor experiments require no-inheritance
and scrambled/withheld-inheritance controls and must judge transfer on the
untouched final test. This is a deferred design constraint, not authorization
for implementation or training.

## 2026-10-12 — D038: freeze the 8000-value solvability-ladder rung

The completed values1024 rung preserved perfect training memorization for both
models while reducing held-out exact match to `0.1406` for A0 and `0.0703` for
A1. The next calibrated test therefore expands only the reserved answer-token
pool from 1,024 to 8,000, keeping the one association, 128 fixed examples,
architectures, seed 17, optimizer, 2,000-step budget, 256,000 token
positions, and checkpoint/evaluation cadence unchanged. The reserved values
fit inside the existing 32,000-token vocabulary.

The frozen configurations are
`configs/braidnano_a0_solvability_values8000.json` and
`configs/braidnano_a1_solvability_values8000.json`; the protocol is
`docs/A0_A1_SOLVABILITY_VALUES8000_PROTOCOL.md`. Only the two fresh seed-17
arms are in scope. Association-count changes, distractors, target-length
changes, decay analysis, predictive learning-state/inheritance, scaling, A2,
and memory work remain outside this authorization boundary.

## 2026-10-12 — D039: complete the 8000-value solvability-ladder rung

The frozen protocol in commit `eee5e58` was executed with fresh seed-17 A0 and
A1 initialization. Both unchanged 42M models completed 2,000 optimizer steps
and 256,000 token positions, reached `1.0000` autoregressive greedy exact
match on the repeated 128-example training set, and produced finite values
with SHA-256-verified final checkpoints and no CUDA/OOM failure.

On the diagnostic held-out set, both A0 and A1 reached `0.0391` exact match
(`5/128`). A0's target loss was `9.947532`; A1's was `10.239311`, for an A1
minus A0 delta of `+0.291780`. Thus exact-match separation disappeared at
this rung, but A0 remained closer to the correct answer by target loss.
Relative to values1024, held-out exact match fell from `0.1406` to `0.0391`
for A0 and from `0.0703` to `0.0391` for A1. A1 used `1.550x` A0's CUDA time
(`287.884` versus `185.684` seconds).

This completes the reserved value-space axis without reaching a training-
memorization boundary: both models still memorized the fixed training set.
The next separately controlled rung is association count. The tracked
evidence is in `docs/A0_A1_SOLVABILITY_VALUES8000_RESULT.md`,
`results/a0_a1_solvability_values8000.json`, and the read-only verifier
`scripts/analyze_synthetic_solvability_values8000.py`; raw logs and
checkpoints remain ignored. No association-count change, distractor-distance
change, target-length change, decay analysis, predictive learning-state,
competence-ledger, scaling, A2, or memory work started.

## 2026-10-12 — D040: freeze the two-association 8000-value solvability rung

The completed one-association ladder through 8,000 values still showed perfect
training memorization but severe held-out collapse. The next controlled test
therefore changes only the association count from one to two while retaining
the 8,000-value pool, 128 examples, seed 17, model sizes, optimizer, 2,000
optimizer steps, effective batch 16, and evaluation/checkpoint cadence.

Two explicit records require an 11-token example rather than the prior
8-token example. The resulting run processes 352,000 sequence-token positions
at the same optimizer-step budget; this unavoidable sequence-length change is
recorded as a limitation rather than treated as equal-token evidence.

The frozen configurations are
`configs/braidnano_a0_solvability_associations2_values8000.json` and
`configs/braidnano_a1_solvability_associations2_values8000.json`; the protocol
is `docs/A0_A1_SOLVABILITY_ASSOCIATIONS2_VALUES8000_PROTOCOL.md`. Only the
two fresh seed-17 arms are in scope. Four/eight associations, additional
seeds, distractors, target-length changes, decay analysis, predictive
learning-state/inheritance, scaling, A2, and memory work remain outside this
authorization boundary.

## 2026-10-12 — D041: complete the two-association 8000-value solvability rung

The frozen protocol in commit `3079c74` was executed with fresh seed-17 A0 and
A1 initialization. Both unchanged 42M models completed 2,000 optimizer steps
and 352,000 sequence-token positions, reached `1.0000` autoregressive greedy
exact match on the repeated 128-example training set, and produced finite
values with SHA-256-verified final checkpoints and no CUDA/OOM failure.

On the diagnostic held-out set, A0 reached `0.0156` exact match (`2/128`) and
A1 reached `0.0000` (`0/128`). A0's target loss was `10.385956`; A1's was
`10.544830`, for an A1 minus A0 delta of `+0.158875`. A0 therefore retained
the diagnostic advantage. Relative to the one-association 8000-value rung,
held-out exact match fell from `0.0391` to `0.0156` for A0 and from `0.0391`
to `0.0000` for A1 under the same 2,000-optimizer-step recipe.

This is a held-out retrieval/generalization effect, not a training-
memorization boundary: both models still memorized the fixed training set.
The two-association examples necessarily used 11 tokens, so the rung used
352,000 token positions and is not an equal-token comparison with the
one-association result. A1 used `1.736x` A0's CUDA time (`316.698` versus
`182.455` seconds). The tracked evidence is in
`docs/A0_A1_SOLVABILITY_ASSOCIATIONS2_VALUES8000_RESULT.md`,
`results/a0_a1_solvability_associations2_values8000.json`, and the read-only
verifier `scripts/analyze_synthetic_solvability_associations2_values8000.py`;
raw logs and checkpoints remain ignored. Four/eight associations, distractors,
target-length changes, decay analysis, predictive learning-state,
competence-ledger, scaling, A2, and memory work remain unfrozen.

## 2026-10-12 — D044: freeze the eight-association 8000-value solvability rung

The completed four-association rung reached perfect training memorization but
held-out exact match was at the floor for both architectures. The next and
final predeclared association-count test therefore changes only the count from
four to eight while retaining the 8,000-value pool, 128 examples, seed 17,
model sizes, optimizer, 2,000 optimizer steps, effective batch 16, and
evaluation/checkpoint cadence.

Eight explicit records require a 29-token example. The resulting run
processes 928,000 sequence-token positions at the same optimizer-step budget;
this unavoidable sequence-length change is recorded as a limitation rather
than treated as equal-token evidence.

The frozen configurations are
`configs/braidnano_a0_solvability_associations8_values8000.json` and
`configs/braidnano_a1_solvability_associations8_values8000.json`; the protocol
is `docs/A0_A1_SOLVABILITY_ASSOCIATIONS8_VALUES8000_PROTOCOL.md`. Only the
two fresh seed-17 arms are in scope. Additional seeds, distractors,
target-length changes, decay analysis, predictive learning-state/inheritance,
scaling, A2, and memory work remain outside this authorization boundary.

## 2026-10-12 — D042: freeze the four-association 8000-value solvability rung

The completed two-association rung still showed perfect training memorization
but near-floor held-out retrieval. The next controlled test therefore changes
only the association count from two to four while retaining the 8,000-value
pool, 128 examples, seed 17, model sizes, optimizer, 2,000 optimizer steps,
effective batch 16, and evaluation/checkpoint cadence.

Four explicit records require a 17-token example. The resulting run processes
544,000 sequence-token positions at the same optimizer-step budget; this
unavoidable sequence-length change is recorded as a limitation rather than
treated as equal-token evidence.

The frozen configurations are
`configs/braidnano_a0_solvability_associations4_values8000.json` and
`configs/braidnano_a1_solvability_associations4_values8000.json`; the protocol
is `docs/A0_A1_SOLVABILITY_ASSOCIATIONS4_VALUES8000_PROTOCOL.md`. Only the
two fresh seed-17 arms are in scope. Eight associations, additional seeds,
distractors, target-length changes, decay analysis, predictive
learning-state/inheritance, scaling, A2, and memory work remain outside this
authorization boundary.

## 2026-10-12 — D043: complete the four-association 8000-value solvability rung

The frozen protocol in commit `5c4dfb1` was executed with fresh seed-17 A0 and
A1 initialization. Both unchanged 42M models completed 2,000 optimizer steps
and 544,000 sequence-token positions, reached `1.0000` autoregressive greedy
exact match on the repeated 128-example training set, and produced finite
values with SHA-256-verified final checkpoints and no CUDA/OOM failure.

On the diagnostic held-out set, both A0 and A1 reached `0.0000` exact match
(`0/128`). A0's target loss was `10.506805`; A1's was `10.871704`, for an A1
minus A0 delta of `+0.364899`. Exact-match separation is at the floor, while
target loss still favors A0. Relative to the two-association 8000-value rung,
A0 fell from `0.0156` to `0.0000` held-out exact match and A1 remained at
`0.0000` under the same 2,000-optimizer-step recipe.

This is a held-out retrieval/generalization floor, not a training-
memorization boundary: both models still memorized the fixed training set.
The four-association examples necessarily used 17 tokens, so the rung used
544,000 token positions and is not an equal-token comparison with the one- or
two-association results. A1 used `1.873x` A0's CUDA time (`349.060` versus
`186.286` seconds). The tracked evidence is in
`docs/A0_A1_SOLVABILITY_ASSOCIATIONS4_VALUES8000_RESULT.md`,
`results/a0_a1_solvability_associations4_values8000.json`, and the read-only
verifier `scripts/analyze_synthetic_solvability_associations4_values8000.py`;
raw logs and checkpoints remain ignored. Eight associations, distractors,
target-length changes, decay analysis, predictive learning-state,
competence-ledger, scaling, A2, and memory work remain unfrozen.
