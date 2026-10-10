# A1 seed-17 training result

Status: **completed**. The authorized A1 pilot used the verified associative
`parallel_scan` path. It is an empirical TinyStories result, not evidence of
general reasoning ability or architectural superiority.

## Reproducibility record

| Item | Value |
| --- | --- |
| Launch commit | `4ff8379` |
| Configuration | `configs/braidnano_a1_recurrent_state_space_proposal.json` |
| Configuration SHA-256 | `0ddd4e1aa69e24dc33bd6846aaed645199ef26990d5eee33dfa911098056e7ab` |
| Dataset | TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| Tokenizer | 32,000-token byte-level BPE; SHA-256 `9d2b84ebf358729cf548a2962f10fe15eebc512a0b6f42b3f90a010d44b589f3` |
| Seed | 17 |
| Architecture | 8-block gated diagonal state-space decoder; parallel associative scan |
| Parameters | 41,870,336 |
| Context / micro-batch / effective batch | 512 / 2 / 16 |
| Optimizer | AdamW, learning rate `3e-4`, weight decay `0.1` |
| Schedule | cosine with 10 warmup steps |
| Precision | FP16 CUDA autocast with FP32 parameters |
| Validation | 131,072 tokens every 100 optimizer steps |
| Gradient checkpointing | **Not implemented; not enabled** |

## Measurements

- Optimizer steps: **1,000**
- Tokens processed: **8,192,000**
- Training wall time excluding validation/checkpoint overhead: **404.41 s**
- Total elapsed wall time: **436.41 s**
- Mean training throughput: **20,256.62 tokens/s**
- Peak allocated/reserved VRAM: **1,657,214,976 / 1,931,476,992 bytes**
- Training loss: `10.5403` at the first step; `3.1820` at step 1,000
- Validation loss: `4.3308` at step 100; `3.2635` at step 1,000
- No non-finite loss/gradients, CUDA failures, or out-of-memory events occurred.

The complete validation curve and concise machine-readable record are in
[`results/a1_seed17_pilot_summary.json`](../results/a1_seed17_pilot_summary.json).
The raw step log remains locally at
`results/runs/a1_seed17_pilot/steps.jsonl` and is excluded from Git.

## Comparison with A0 seed 17

Both pilots used the same TinyStories revision, tokenizer, seed, context,
effective batch, optimizer settings, validation budget, and exactly 8,192,000
training token positions.

### Equal-token comparison

| Metric | A0 Transformer | A1 parallel scan |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Final training loss | 3.6161 | 3.1820 |
| Final validation loss | 3.7512 | 3.2635 |
| Training throughput | 39,320.88 tokens/s | 20,256.62 tokens/s |
| Training wall time | 208.55 s | 404.41 s |
| Peak allocated VRAM | 1,329,410,560 bytes | 1,657,214,976 bytes |
| Peak reserved VRAM | 1,591,738,368 bytes | 1,931,476,992 bytes |

At the matched token budget, A1's final validation loss was `0.4877` lower
than A0's, a `13.0%` relative reduction in this single pilot. This is the
equal-token result; it does not hold compute or wall time constant.

### Equal-compute comparison

No direct equal-compute quality comparison was run. The pilots matched token
positions, not measured FLOPs or wall-clock time. On the same RTX 3060 Ti, A1
took `1.94x` the training wall time and achieved `51.5%` of A0's measured
throughput. Therefore this result does not support an equal-compute efficiency
claim, even though A1 reached lower validation loss at equal tokens.

The A0 checkpoint and records remain unchanged. A1's final checkpoint is
locally preserved with optimizer and Python/Torch/CUDA RNG state; its verified
SHA-256 is
`0cac9f52b532f9edb44bffb11051843b4423877b28615cb3378bd73965a69a1d`.

No A2 implementation, model scaling, or persistent-memory integration was
started.

## Fixed-prompt generation evaluation

The preserved A1 checkpoint was evaluated with the same four prompts used for
A0, greedy `argmax` decoding, no sampling, and a maximum of 64 new tokens per
prompt. The evaluation took **1.5696 s**; peak allocated/reserved VRAM was
`221,171,200 / 241,172,480` bytes. A0 took `1.0436 s` under the same method,
so this A1 evaluation was approximately `1.50x` slower. This is a qualitative
inspection, not a capability benchmark.

The exact token IDs and outputs are preserved in
[`results/a1_prompt_eval.json`](../results/a1_prompt_eval.json). The decoded
outputs were:

1. **Fox and key:** “He was very excited and wanted to play with it. One day,
   a little girl named Lily went to the park. She saw a big, big, a big, big,
   and a little girl named Lily. She loved to play with her friends. One day,
   she went to the park to play with”
2. **Lucy and tower:** “She was very excited to see the big, and she was very
   happy. The little girl was so happy that she had to go to the park with her
   mom. She was so happy to have a new friend. She was so happy that she had
   never seen a new friend.”
3. **Sam and dog:** “He was very happy and he had a big, a big, big, and a
   little girl named Lily. She loved to play with her friends and play with her
   friends. One day, Lily's mom gave her a big hug and a big hug.

   \"I'm sorry, Lily. I'm sorry,”
4. **Bird and nest:** “he was very excited. He was so excited to see the little
   girl. The little girl was so happy that she had to go to the park with her
   friends. She was so happy that she had to be a new friend.”
