# A0/A1 Layerwise Representation Forensics Plan

Status: **frozen analysis plan; read-only execution**

## Question

The target-position decomposition shows that A1 is already impaired under
teacher forcing at the first requested token for target lengths 4 and 8.
This analysis asks where the required information is lost or becomes unusable:
before the query, after the query, during recurrent/block transformations, or
only at the final output route.

## Scope and controls

- Reuse only the six final target-length values8000 checkpoints already
  verified for queue item 1: A0/A1, target lengths 2/4/8, seed 17.
- Use the same deterministic 128-example train and held-out examples.
- Do not retrain, modify source checkpoints, change A0/A1 definitions, or
  alter configs. Diagnostic probe weights are temporary in-memory objects.
- Evaluate in `model.eval()` with gradients disabled. Verify every checkpoint
  sidecar/hash before loading.

## Capture sites

For an example of target length `L`, with sequence positions defined by the
existing generator, capture the following token positions:

- `pre_query`: the final stored-value token, position `2 + L`;
- `post_query`: the query-key token, position `4 + L`;
- `target_1 ... target_L`: the hidden state used to predict each target token,
  positions `4 + L + j` for `j=1..L` (the answer marker predicts target 1).

At every A0 Transformer block, capture the block output. At every A1
recurrent block, capture both the block output and the diagonal recurrent
state before output gating/projection, reusing the state reconstruction
method from `scripts/analyze_a1_state_decay_forensics.py`. Also capture the
final-normalized representation. This keeps A0/A1 comparisons explicit while
exposing the A1-specific state path.

## Probes and metrics

For each capture site, architecture, layer/path, split, and target position:

1. **Key identity:** ridge regression to the model's key-token embedding,
   followed by nearest-candidate key top-1/top-5 accuracy. Report both a
   train-fitted probe evaluated on held-out examples and a held-out-fitted
   linear decodability ceiling; the latter is not a generalization claim.
2. **Associated value:** ridge regression to the target value-token embedding,
   followed by nearest-candidate value top-1/top-5 accuracy and target cosine.
   Use a separate probe for each target position so a pre-query state is not
   incorrectly asked to map one vector to several different target tokens.
3. **Binding identity:** operationalize the one-association binding as the
   paired key→value retrieval probe. Record whether both key and the requested
   value are decoded correctly at the same capture site.
4. **Target position:** fit a ridge one-hot classifier over the output-position
   labels `1..L` using all output-position states, then report held-out
   position accuracy. This is a positional-state diagnostic, not a claim of
   semantic knowledge.

For value/binding retention, additionally fit a fixed probe on the `pre_query`
train representation for each requested target position and transfer that
same probe to later held-out sites. A sharp fall in this fixed-probe metric
with stable site-fitted held-out decodability indicates a changed coordinate
system or readout-use problem; a fall in both fixed and site-fitted
decodability supports information loss or destructive transformation.

The aggregate must preserve finite metrics, sample counts, source checkpoint
hashes, plan/protocol/generator hashes, capture-position metadata, and an
explicit `read_only: true` field.

## Predeclared interpretation rules

- If A1 recurrent states lose value decodability before the query while later
  states cannot recover it, temporal decay or recurrent transformation is
  implicated; layer/depth localization determines whether loss is early or
  late.
- If value decodability survives in an A1 state or block representation after
  the query but model logits remain wrong, storage loss is insufficient; query
  interaction or readout/routing becomes primary.
- If A1 value probes remain strong but key/value pair accuracy or target
  position use fails, binding or routing remains implicated.
- If A1 and A0 both lose value decodability at the same sites, the result is a
  shared task/generalization limit rather than an A1-specific mechanism.
- These probes are correlational diagnostics. They do not authorize a weight
  change or establish causality; queue item 4 requires a separate reversible
  intervention with a predeclared prediction.

## Outputs

- `results/a0_a1_layerwise_representation_forensics_values8000.json`
- `docs/A0_A1_LAYERWISE_REPRESENTATION_FORENSICS_RESULT.md`
- an updated `docs/MECHANISM_CAMPAIGN_STATUS.md`

The analysis must fail nonzero on missing artifacts, source hash mismatch,
unexpected parameter count, missing capture sites/layers, non-finite tensors,
or malformed probe metrics.

