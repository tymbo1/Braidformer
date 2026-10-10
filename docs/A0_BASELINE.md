# A0 verification and baseline proposal

Status: the A0 implementation, approved TinyStories preparation, host CUDA
path, and primary seed-17 pilot are verified. The pilot is a bounded
TinyStories measurement, not a general capability benchmark.

## Reproduce the verification

From the repository root:

```bash
PYTHONPATH=src python3 scripts/verify_a0.py
PYTHONPATH=src python3 scripts/verify_tinystories.py
python3 -m pytest -q
python3 -m compileall -q src tests scripts
```

The verification script is read-only with respect to experiment artifacts. It
loads the JSON config, seeds PyTorch, constructs the model, checks the target
parameter range, and reports a SHA-256 hash of the config. It does not perform
an optimizer step or a training forward pass.

The verified result for the planning configuration is:

| Item | Result |
| --- | --- |
| Config | `configs/braidnano_phase0_planning.json` |
| Config SHA-256 | `f11ba3bb84f884d7565793998c11f2eafeb93e9e2df5d44abc81e8cda9fd6aa2` |
| Stage | A0 |
| Unique trainable parameters | 41,866,240 |
| Target range | 40,000,000–60,000,000 |
| Word embeddings | tied to the LM head |
| Device used for verification | CPU |
| CUDA state | host RTX 3060 Ti available; smoke test and pilot passed |
| Training started | Yes — seed-17 A0 pilot completed |

## Architecture record

| Component | Fixed value |
| --- | --- |
| Model | conventional causal decoder-only Transformer control |
| Layers | 8 pre-layer-normalized Transformer blocks |
| Hidden size | 512 |
| Attention heads | 8 |
| Head dimension | 64 |
| Feed-forward width | 2,048 |
| Vocabulary | 32,000 |
| Context length | 512 tokens |
| Attention | causal scaled dot-product attention |
| Feed-forward activation | GELU |
| Dropout | 0.0 |
| LayerNorm epsilon | 1e-5 |
| Linear biases | enabled |
| Word embeddings | tied input/output weights |
| Recurrent/selective/MoE/controller paths | disabled; later stages only |

Parameter accounting is unique-parameter accounting, so the tied LM head is
not counted twice:

| Group | Parameters |
| --- | ---: |
| Token embedding | 16,384,000 |
| Position embedding | 262,144 |
| Eight Transformer blocks | 25,219,072 |
| Final LayerNorm | 1,024 |
| Tied LM head increment | 0 |
| **Total** | **41,866,240** |

## Test coverage

At verification time, `python3 -m pytest -q` reports **23 passed**. The tests
cover:

- JSON configuration loading and validation, including architecture and
  training-field constraints;
- output shape and finite next-token loss on a synthetic batch;
- exact next-token loss shifting, including the ignored first label;
- a CPU backward pass with a populated embedding gradient;
- causal isolation of earlier logits from changed future tokens;
- rejection of a later-stage config by the A0 model;
- parameter-count target-range and tied-embedding checks;
- context-length enforcement; and
- existing hardware-report parser behavior.
- data-preparation helper determinism;
- fail-fast numerical/progress integrity monitoring; and
- atomic checkpoint sidecar verification.

The earlier A0 verification at commit `1a55157` used the Python
standard-library `trace` module and reported 461/461 executable lines (100%)
for the pre-runner `src/braidformer` code. That is line coverage, not branch
coverage, runner coverage, throughput validation, numerical stability
validation, or training performance evidence. The historical coverage command
was:

```bash
PYTHONPATH=src python3 -m trace --count --summary \
  --coverdir=/tmp/braidformer-trace \
  --ignore-dir=/usr --ignore-dir=/usr/local \
  --module pytest -q
```

## Executed training configuration

The machine-readable proposal used for the run is
[`configs/braidnano_a0_baseline_proposal.json`](../configs/braidnano_a0_baseline_proposal.json).
It remains marked `planned_only: true` in source control; explicit run
authorization was supplied through the training entrypoint.

| Setting | Proposed value |
| --- | --- |
| Primary seed | 17 |
| Replication seeds | 23 and 29, if the primary pilot is approved |
| Dataset/tokenizer | TinyStories at revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, 32,000-token byte-level BPE; verified artifact record in `configs/braidnano_a0_data.json` |
| Training budget | 8,192,000 tokens (1,000 optimizer steps at 8,192 tokens/step) |
| Evaluation budget | 131,072 tokens |
| Context | 512 tokens |
| Micro-batch | 2 sequences |
| Gradient accumulation | 8 steps; effective batch 16 sequences / 8,192 tokens |
| Optimizer | AdamW, learning rate 3e-4, betas (0.9, 0.999), epsilon 1e-8 |
| Weight decay | 0.1 |
| Schedule | cosine, 1% warmup |
| Gradient clipping | global norm 1.0 |
| Precision | fp16 only after a CUDA memory smoke test passes |
| Gradient checkpointing | not implemented; not enabled |
| Evaluation cadence | every 100 optimizer steps |
| Checkpoint cadence | at the end of the 1,000-step pilot |
| Distillation | disabled; no Qwen teacher |

The dataset/tokenizer was selected, pinned, prepared, and locally verified
before the run. The measured result is recorded in
[`A0_TRAINING_RESULT.md`](A0_TRAINING_RESULT.md). Replication seeds 23 and 29
remain unstarted.

## Benchmark methodology

The first empirical comparison should be a fixed-token A0 control run, then a
matched A1 run. For every run:

1. Pin the Git commit, configuration hash, dataset revision, tokenizer,
   preprocessing, packing policy, and seed.
2. Keep context length, token budget, micro/effective batch, optimizer,
   schedule, precision, clipping, evaluation split, and checkpoint cadence
   identical across A0 and A1 wherever the architecture permits.
3. Gate execution on a successful device and memory smoke test. The A0 host
   CUDA gate passed before the seed-17 pilot.
4. Preserve step-level raw logs before summarizing them.
5. Report parameter count, active parameters where applicable, approximate
   training FLOPs and accounting method, wall-clock time, warm-up time, tokens
   per second, peak allocated/reserved VRAM, training loss, validation loss,
   and task metrics with uncertainty where practical.
6. Compare quality at matched tokens and approximate FLOPs, not only at equal
   wall-clock time. Any deviation must be recorded before interpreting a
   result.

The seed-17 result is a bounded TinyStories pilot measurement. It is not a
general capability claim or an A0-versus-A1 comparison; A1 architecture
implementation is now complete, but A1 training has not started.
