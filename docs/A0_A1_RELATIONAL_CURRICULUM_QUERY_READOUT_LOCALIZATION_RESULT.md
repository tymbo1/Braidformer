# A0/A1 relational-curriculum query/readout localization result

Status: **complete; read-only frozen-checkpoint analysis**.

## Result in one sentence

Staged A1 has better native value-range cross-entropy and target ranking than
shuffled A1, but four-fold cross-validated fixed pre-query and site-specific
diagnostic value readouts remain at or near floor. The evidence does not show a
hidden linearly usable value that the final output head simply fails to read;
the stable cross-example binding is already missing or transformed upstream.

## Provenance and integrity

- Analysis plan: `docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_READOUT_LOCALIZATION_PLAN.md`.
- Analysis commit: `cc7d7cfbde079fc8af2bb25e588b11a913b34e9f`.
- Reused query/readout script sha256:
  `4e9788c1f2598a278d37e36c93df35ce86d03f6ada17094c63c75b7a2e6699d4`.
- Source curriculum aggregate sha256:
  `66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71`.
- Preceding representation aggregate sha256:
  `b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3`.
- Query aggregate: `results/a0_a1_relational_curriculum_query_readout_localization.json`
  (sha256 `73dc7df64d9d57099d3f31ede2628ba673ed8cf716249fbf17b83164a8d77e50`).
- Twelve analyses completed: A1/A0 × staged/shuffled × target lengths 2/4/8.
- All source checkpoint sidecars/digests, configuration hashes, balanced folds,
  capture paths, logits, representations, and diagnostic metrics passed
  integrity checks. The output contains 60,312 finite floating-point values;
  no checkpoint or model parameter was modified.

## Method

The existing query/readout diagnostic was reused with four-fold cross-validation
within the fixed 128-example held-out split. Native teacher-forced logits were
measured over both the full vocabulary and the 8,000-value candidate range at
each target position. Temporary ridge readouts were trained from the pre-query
representation and transferred to post-query/target sites; separate
site-specific readouts were trained at each site. The legacy script's duplicated
distractor-distance argument was normalized by the committed wrapper without
changing the deterministic example generator.

## Native value-range output

The table averages native value-candidate top-1, cross-entropy, and target rank
over target positions. Rank is out of 8,000 candidates.

| arch | arm | L | top-1 | cross-entropy | mean target rank |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | staged | 2 | 0.0000 | 9.4654 | 3727.6 |
| A1 | shuffled | 2 | 0.0000 | 10.3576 | 4113.2 |
| A1 | staged | 4 | 0.0020 | 9.2660 | 3515.6 |
| A1 | shuffled | 4 | 0.0000 | 10.5782 | 3814.0 |
| A1 | staged | 8 | 0.0000 | 9.2352 | 3678.1 |
| A1 | shuffled | 8 | 0.0000 | 10.8192 | 3884.9 |
| A0 | staged | 2 | 0.0547 | 8.5043 | 2848.7 |
| A0 | shuffled | 2 | 0.0547 | 8.2250 | 2256.0 |
| A0 | staged | 4 | 0.0391 | 8.7690 | 2935.2 |
| A0 | shuffled | 4 | 0.0371 | 8.8824 | 2966.4 |
| A0 | staged | 8 | 0.0312 | 8.9534 | 3222.3 |
| A0 | shuffled | 8 | 0.0195 | 9.1560 | 3264.7 |

Staged A1 improves native confidence/ranking at every target length, but exact
candidate top-1 remains effectively absent. The improvement is therefore a
trajectory/confidence effect, not a solved retrieval operation.

## Diagnostic value readouts

The following means are over every captured path and target position. “Fixed” is
the pre-query probe transferred to the target site; “site-specific” fits a
separate cross-validated probe at the target site. The maxima show the largest
single target-site top-1 observed across all paths/positions.

| arch | arm | L | fixed mean | fixed max | site-specific mean | site-specific max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A1 | staged | 2 | 0.0000 | 0.0000 | 0.0016 | 0.0156 |
| A1 | shuffled | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | staged | 4 | 0.0001 | 0.0078 | 0.0001 | 0.0078 |
| A1 | shuffled | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | staged | 8 | 0.0002 | 0.0078 | 0.0001 | 0.0078 |
| A1 | shuffled | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A0 | staged | 2 | 0.0004 | 0.0078 | 0.0004 | 0.0078 |
| A0 | shuffled | 2 | 0.0004 | 0.0078 | 0.0043 | 0.0156 |
| A0 | staged | 4 | 0.0002 | 0.0078 | 0.0000 | 0.0000 |
| A0 | shuffled | 4 | 0.0037 | 0.0156 | 0.0039 | 0.0156 |
| A0 | staged | 8 | 0.0005 | 0.0078 | 0.0017 | 0.0156 |
| A0 | shuffled | 8 | 0.0002 | 0.0078 | 0.0007 | 0.0078 |

The staged A1 representation advantage seen under held-out-fitted probes in the
previous item does not survive this stricter cross-validated diagnostic. There
is no strong site-specific A1 readout at the target sites while native output
fails. That weakens the pure final-readout/routing hypothesis.

## Adjudication

1. **Stable arbitrary binding/generalization (H4): strongly supported as the
   upstream failure.** Fixed and site-specific value readouts are near floor
   before/after the query and at target sites.
2. **Recurrent transformation/interference (H2): remains leading.** A1's
   staged curriculum improves native confidence but does not create a
   cross-validated value coordinate; the earlier layerwise separation is a
   representation ceiling rather than a transferable binding algorithm.
3. **Pure final readout failure (H5): weakened further.** A separate diagnostic
   readout cannot recover the target value at useful rates from the frozen A1
   target representations.
4. **Query-conditioned corruption (H3): not isolated.** The fixed probe is
   already at floor at pre-query in this held-out cross-validation, so this
   experiment cannot assign the primary loss specifically to the query.
5. **Temporal decay (H1): unresolved as a contributor, not established here.**
   The result is compatible with the earlier decay-tail negative and does not
   provide a new causal test.
6. **Curriculum history (H8): remains a modifier.** Staging changes native
   confidence/ranking and the representation ceiling, but not robust retrieval.

The next queue item is therefore the predeclared non-training causal state
intervention. It must be reversible, must alter no weights, and must state in
advance which representation/output rescue would support or reject the leading
recurrent-transformation hypothesis.
