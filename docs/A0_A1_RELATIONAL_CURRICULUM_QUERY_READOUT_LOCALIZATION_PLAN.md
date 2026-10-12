# A0/A1 relational-curriculum query/readout localization plan

Status: **frozen read-only analysis plan**.

## Question

The preceding probes found a large staged-versus-shuffled difference in A1's
later-state held-out-fitted decodability, but little train-to-held-out or fixed
pre-query transfer. This diagnostic asks whether the native query/output route
loses the value, whether a value remains in a changed representation, or
whether no linearly transferable value code is present at all.

## Frozen scope and provenance

- Use the four final relational-curriculum checkpoints only: A0/A1 × staged/
  shuffled, seed 17.
- Evaluate target lengths 2, 4, and 8 on the fixed 128-example held-out split.
- Use the existing query/readout localization equations and four-fold
  held-out cross-validation from `scripts/analyze_query_readout_localization.py`.
- Capture `pre_query`, `post_query`, every target-output site, every block
  output, every A1 recurrent state, and `final_norm`.
- Record native teacher-forced logits over the full vocabulary and the 8,000
  candidate-value range at every target position.
- Diagnostic probes are temporary in-memory objects. No checkpoint, model
  parameter, architecture, or configuration is modified.

The source curriculum aggregate is pinned to sha256
`66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71`. The
preceding representation aggregate is pinned to sha256
`b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3` and
analysis commit `19a2ef68db8ab6f43b725167dc4b63b961091ddc`.

The reused query/readout implementation is pinned to sha256
`4e9788c1f2598a278d37e36c93df35ce86d03f6ada17094c63c75b7a2e6699d4`.
Its historical call site supplied the distractor-distance argument twice;
the new wrapper normalizes that call to the existing deterministic generator's
signature and records the compatibility shim. The generated examples must
remain the same fixed held-out examples used by the preceding probes.

## Measurements

For each checkpoint, target length, output position, and capture path, record:

- native full-vocabulary cross-entropy, correct-token probability, top-1,
  logit margin, and target rank;
- native value-candidate-range versions of those metrics;
- four-fold cross-validated fixed pre-query probe at pre-query, post-query,
  and target-output sites;
- four-fold cross-validated site-specific diagnostic probe at the same sites.

The fixed probe tests whether a shared pre-query value coordinate survives the
query/readout path. The site-specific probe tests whether a linearly usable
value code is present in a changed coordinate system. Cross-validation keeps
the diagnostic distinct from a same-sample fitted ceiling.

## Predeclared interpretation rules

- If fixed transfer is already at floor before the query, the result does not
  localize the failure to query/readout; stable train-to-held-out binding is
  absent upstream.
- If fixed transfer is above floor before the query and collapses at post-query
  or target sites, query-conditioned corruption or recurrent transformation is
  implicated.
- If fixed transfer is weak but site-specific transfer is strong, the value may
  survive in a changed coordinate system; native output failure then remains a
  readout/use candidate but is not causal proof.
- If both fixed and site-specific probes fall after the query while native
  logits fail, destructive state transformation/retention is favoured.
- Compare staged and shuffled A1 first, then A0 controls. A1-only effects
  strengthen an A1 recurrent learning-history interaction; matched A0 effects
  indicate a shared task or curriculum effect.
- No result here unlocks a weight change. The next queue item, if warranted,
  is a separate reversible inference-time intervention with a predeclared
  prediction.

## Integrity and output

Before loading any checkpoint, verify the source aggregate, preceding probe
aggregate, summary fields, config hashes, checkpoint sidecars/digests,
parameter counts, seed, token budget, and exposure hash. Require balanced
folds, finite logits/representations/metrics, complete capture paths, and
`read_only: true`. Raw checkpoints and logs remain ignored under
`results/runs/`.

Outputs:

- `results/a0_a1_relational_curriculum_query_readout_localization.json`
- `docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_READOUT_LOCALIZATION_RESULT.md`
- updated `docs/RELATIONAL_CURRICULUM_CAMPAIGN_STATUS.md`
