# A0/A1 query/readout localization plan

Status: **frozen analysis plan; read-only execution**

## Question

Does the required value remain available in a representation after the query
and at the target-output positions, while the native output route fails, or is
the value already lost/corrupted during query-conditioned processing?

## Scope and controls

- Reuse only the six final values8000 target-length checkpoints already used by
  queue items 1 and 2: A0/A1, target lengths 2/4/8, seed 17.
- Use the deterministic heldout examples from the frozen target-length
  generator. The 128-example heldout set is never used for training the model.
- Do not retrain, modify source checkpoints, change A0/A1 definitions, or
  alter configs. Diagnostic probe weights are temporary in-memory objects.
- Evaluate in `model.eval()` with gradients disabled and verify each source
  checkpoint sidecar/hash before loading.

## Measurements

For each heldout example, capture the same representation sites as item 2:
`pre_query`, `post_query`, every target-output position, every A0 block output,
every A1 block output, every A1 recurrent state, and `final_norm`.

At every target-output position, also record the native teacher-forced output
over the full vocabulary and over the 8,000-value candidate range:

- cross-entropy/log probability of the correct value;
- correct-token probability;
- top-1 accuracy;
- correct-token logit margin against the strongest incorrect candidate;
- target rank.

Use deterministic four-fold cross-validation within the heldout set. For each
target position and representation site:

1. Fit a temporary ridge probe from the pre-query representation on three
   folds to the target value embedding and evaluate it on the fourth fold at
   `pre_query`, `post_query`, and the later target-output sites. This is the
   fixed-pre-query transfer test and measures whether the same value code
   remains usable through the query/readout path.
2. Fit a separate temporary ridge probe at each site on three folds and
   evaluate it on the fourth fold. This is the site-specific diagnostic
   decodability test; it measures whether a linearly usable value code is
   present even if the native route does not use it.
3. Aggregate predictions over all four heldout folds rather than reporting a
   same-sample fitted ceiling.

The diagnostic probe is compared with the native value-range output, not only
the full-vocabulary output, so the comparison does not confuse an incorrect
non-value token with a failure to rank among candidate values.

## Predeclared interpretation rules

- If pre-query cross-validated decodability is already at floor, later failure
  does not localize to query/readout; binding is absent or not linearly
  transferable before the query.
- If pre-query transfer is above floor but drops sharply at `post_query` or the
  first target site, query-conditioned corruption or recurrent transformation
  is implicated.
- If fixed pre-query transfer remains above floor at a target site but the
  native output is poor, the value survives the representation path and
  query/readout/routing becomes primary. A substantially stronger
  site-specific diagnostic readout than the native value-range output is
  additional evidence for a readout gap, not causal proof.
- If fixed transfer is weak but site-specific decodability remains strong, the
  representation may preserve the value in a changed coordinate system; this
  separates coordinate transformation from outright information loss.
- If both fixed transfer and site-specific decodability fall together after
  the query, destructive state transformation/retention is favoured.
- These are frozen-checkpoint correlational results. They do not authorize a
  weight change or establish causality; queue item 4 must predeclare a
  reversible inference-time intervention.

## Integrity requirements

The machine-readable output must preserve source checkpoint hashes, plan and
runner hashes, fold assignments, capture positions, native metrics, fixed and
site-specific cross-validated probe metrics, finite-value checks, and
`read_only: true`. It must fail nonzero on missing artifacts, hash mismatch,
unexpected parameter count, missing capture sites/layers, non-finite tensors,
or malformed metrics.

## Outputs

- `results/a0_a1_query_readout_localization_values8000.json`
- `docs/A0_A1_QUERY_READOUT_LOCALIZATION_RESULT.md`
- updated `docs/MECHANISM_CAMPAIGN_STATUS.md`
