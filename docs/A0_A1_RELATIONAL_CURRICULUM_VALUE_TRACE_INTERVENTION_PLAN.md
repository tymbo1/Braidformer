# A1 relational-curriculum value-trace intervention plan

Status: **frozen mechanism-targeted training protocol**.

## Motivation and authorization

The mechanism adjudication found two convergent facts: later A1 recurrent
states do not provide a stable train-to-held-out key→value coordinate, and
post-hoc layer-7 state replay does not restore native retrieval. One and only
one minimal training intervention is therefore authorized to test whether
directly shaping the existing recurrent state during learning can repair the
binding operation.

## Intervention

Train fresh A1 models under the exact relational-curriculum v1 schedule, once
with `staged` order and once with `shuffled` order. During training only, add a
fixed auxiliary value-trace loss to the ordinary causal language-model loss.

For every non-masked target label, use the existing layer-7 recurrent state at
the prediction position and score it against the existing 8,000 value-token
embeddings:

```text
value_logits = cosine(normalize(layer7_state),
                      normalize(token_embedding[VALUE_BASE:VALUE_BASE+8000]))
                 / 0.1
loss = native_causal_loss + 0.1 * cross_entropy(value_logits, target_value)
```

The auxiliary objective adds no parameter, head, state channel, or model
module. It uses the existing tied embedding table and is disabled in evaluation
mode, so all validation/final metrics remain native causal-LM metrics. The
original A1 definition and A0 are untouched. The implementation is a training
wrapper/subclass with the same parameter/state-dict keys; it does not edit
`src/braidformer/models/` or the frozen configs.

## Controls and matched methodology

- Controls: the already verified A1 staged and A1 shuffled curriculum
  checkpoints from `results/a0_a1_relational_curriculum_v1.json`.
- Seed: 17 for both fresh intervention runs.
- Same A1 curriculum config, deterministic schedule, 2,000 optimizer steps,
  928,000 training tokens, microbatch, gradient accumulation, AdamW, learning
  rate, warmup, cosine schedule, clipping, evaluation cadence, and checkpoint
  cadence as curriculum v1.
- Same held-out frozen retrieval battery and checkpoint sidecar verification.
- Record native final metrics against the matched control, auxiliary training
  branch metadata, CUDA time, checkpoint digest, and all failure state.

## Predeclared prediction

If the missing capability is learnable binding in the existing recurrent state,
the intervention should improve native held-out retrieval relative to the
matched control, especially on target lengths 4/8 and the hard values8000
tests. Primary evidence is lower native target loss and better held-out value
rank/top-1; an auxiliary signal without native improvement is a failed repair.

The intervention fails if neither staged nor shuffled A1 shows a coherent
native held-out improvement on the hard retrieval axes. If it fails, record the
negative result and do not stack another mechanism-targeted training change in
this campaign.

## Integrity and outputs

The run must require the exact committed protocol hash, preserve parameter
count and state-dict structure, verify all source controls, fail on non-finite
loss/gradient/checkpoint, and leave the original A0/A1 definitions and source
checkpoints unchanged. Raw intervention runs remain ignored under
`results/runs/`.

Outputs:

- `results/a0_a1_relational_curriculum_value_trace_intervention.json`
- `docs/A0_A1_RELATIONAL_CURRICULUM_VALUE_TRACE_INTERVENTION_RESULT.md`
- updated `docs/RELATIONAL_CURRICULUM_CAMPAIGN_STATUS.md`
