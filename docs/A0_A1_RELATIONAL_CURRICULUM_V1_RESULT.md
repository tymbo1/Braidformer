# A0/A1 staged relational-binding curriculum result

Status: **complete; matched four-arm result**.

Protocol: `docs/A0_A1_RELATIONAL_CURRICULUM_V1_PROTOCOL.md`.

Machine-readable aggregate: `results/a0_a1_relational_curriculum_v1.json`.

## Result in one sentence

Curriculum order materially changed A1's learning trajectory and improved its
held-out cross-entropy on several retrieval axes, but it did not restore exact
retrieval on the 8,000-value, long-distance, or multi-token hard cases; A0's
order effect was mixed and often favoured the shuffled control.

This is evidence that training history contributes to A1's behaviour, not
evidence that curriculum alone explains or repairs the recurrent retrieval
weakness.

## Integrity and controls

All four fresh runs used protocol commit
`30488e4f6b95994f02e15de9fab5710ce9af2aa4`, seed 17, the original A0/A1
definitions, 41,866,240 A0 parameters or 41,870,336 A1 parameters, 2,000
optimizer steps, 928,000 tokens, fixed 29-token batches, effective batch 16,
AdamW, and the same cosine schedule. The order-independent exposure hash was
identical across all arms:

`fbbc2ade9f57566ecc18a1752b828880d2218b7ff42692638c6ecf74ea4e4d78`.

| arm | CUDA seconds | checkpoint SHA-256 prefix | status |
| --- | ---: | --- | --- |
| A0 staged | 183.70 | `8c15b235dd53` | complete |
| A0 shuffled | 184.34 | `e6c998e3d90a` | complete |
| A1 staged | 343.66 | `0b6ba5eff90f` | complete |
| A1 shuffled | 347.45 | `be7a71017ea6` | complete |

All final checkpoint sidecars verified. No numerical failure was recorded.

## Staged minus shuffled held-out deltas

Positive exact-match deltas favour staged order. Negative target-loss deltas
favour staged order.

| frozen test | A0 ΔEM | A0 ΔCE | A1 ΔEM | A1 ΔCE |
| --- | ---: | ---: | ---: | ---: |
| values128 | −11.72 pts | +2.481 | +28.13 pts | −2.791 |
| values256 | −2.34 pts | +0.990 | +15.63 pts | −2.700 |
| values1024 | +5.47 pts | +0.090 | +5.47 pts | −2.428 |
| values8000 | −6.25 pts | +1.177 | 0.00 pts | +0.016 |
| associations2 | 0.00 pts | +0.303 | +0.78 pts | −0.076 |
| associations4 | +4.69 pts | −0.347 | 0.00 pts | −0.254 |
| associations8 | 0.00 pts | −0.078 | +2.34 pts | −0.295 |
| distance4 | −0.78 pts | +0.772 | 0.00 pts | +0.141 |
| distance16 | −0.78 pts | +0.645 | 0.00 pts | −2.282 |
| distance64 | −2.34 pts | +0.527 | 0.00 pts | −1.830 |
| target length2 | 0.00 pts | +0.241 | 0.00 pts | −0.574 |
| target length4 | 0.00 pts | −0.086 | 0.00 pts | −0.885 |
| target length8 | 0.00 pts | −0.147 | 0.00 pts | −1.206 |

The corresponding absolute EM values are available for every arm in the JSON
aggregate. For orientation, A1 staged reached 32.03%, 17.97%, and 7.03% on
values128/256/1024, versus 3.91%, 2.34%, and 1.56% for A1 shuffled. Both A1
arms were at 0% on values8000 and all distance/target-length exact-match tests;
the small non-zero association values are not a hard-case repair.

## Interpretation

1. **H8 curriculum/developmental history: partially supported.** A1's staged
   order produced a large, consistent improvement on smaller value pools and a
   lower CE on distance16/64 and target lengths 2/4/8. The identical exposure
   hash makes a simple data-volume explanation unlikely. This is still one
   seed and one schedule, so it is not a general curriculum law.
2. **H2 recurrent transformation/interference: remains leading.** The staged
   curriculum did not recover A1's exact answers on the hard 8,000-value and
   long-distance tests. Changing developmental order altered the trajectory but
   did not remove the recurrent retrieval boundary.
3. **H4 shared binding/generalization failure: remains supported.** A0 also
   failed to show a uniform staged benefit, and both architectures remain
   limited on simultaneous bindings and long targets.
4. **H7 optimization/training dynamics: strengthened as a modifier, not a
   sole explanation.** A1 is sensitive to order, but the A1 shuffled arm did
   not catch up on hard tasks merely by seeing the same examples in a different
   order. No optimizer or architecture intervention was made.
5. **H1/H3/H5/H6:** the prior adjudication stands. Curriculum order does not
   overturn the decay-tail negative, state-replay negative, probe evidence, or
   the rejection of autoregressive accumulation as the sole cause.

The strongest updated description is therefore:

> A1's composite recurrent retrieval failure is training-history-sensitive,
> but staged relational exposure changes competence and confidence more readily
> than it creates a robust arbitrary-address binding algorithm.

## Scope and next experiment

This campaign does not authorize an A2 store, a decay change, scaling, longer
training, new data, predictive-learning/inheritance, or post-training. The
single best next experiment is a diagnostic follow-up that separates the two
remaining interpretations of the A1 CE gains: repeat the staged/shuffled A1
comparison at a second seed, or use the existing frozen probes on the two
curriculum checkpoints. The probe comparison is the cheaper and more directly
mechanistic next step; it should test whether staged order preserves a usable
value/binding code or merely changes output confidence.

## Verification

The aggregate script verified all four completed summaries, all checkpoint
sidecars, equal exposure hash, seed, token budget, optimizer-step budget, and
protocol commit. The full suite passed with `71 passed`; `git diff --check`
passed; A0/original-A1 source definitions and the frozen historical generator
were unchanged; raw run artifacts remain ignored; and no GitHub push occurred.
