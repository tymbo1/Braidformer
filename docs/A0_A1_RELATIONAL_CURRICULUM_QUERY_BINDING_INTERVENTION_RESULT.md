# A1 query-conditioned binding intervention result

Status: **complete; native loss improved selectively, but the preregistered
binding-repair prediction failed**.

## Result in one sentence

Scoring the existing layer-7 state at the query-key position with an explicit
key/state joint objective improved staged A1's native hard-task loss and some
easy-rung exact match, but hard exact retrieval remained at floor and the
postdiagnostics found no stable transferable key→value code.

## Provenance and integrity

- Protocol/code freeze: `e7259e7b56023d4f4e83e9e17b7eefe194b25da1`.
- Multi-token correction: `8484cd608f4a5ac6860fb4e7f8a23c2149563512`.
- Corrected execution: `4ca85c7ac389d91f164874e6e80914b322da2357`.
- Aggregate commit: `2df07ef23d4d992a46bbc78397d3f0a150569271`.
- Aggregate sha256:
  `5f63c688985ac049047f5e8b34b6c359b1eeeea28de5069728d71a80dcbad9a2`.
- Both fresh A1 arms completed 2,000 steps / 928,000 tokens with 41,870,336
  parameters. Checkpoint sidecars and digests passed.
- Original A0/A1 definitions, configs, and controls were untouched.

The initial gated attempt stopped at step 1,754 because the implementation
incorrectly assumed one target label. The existing stress curriculum includes
target lengths 2/4/8. The partial raw run was preserved outside the repository;
the corrected implementation scores every target token against the same
pre-answer query-conditioned state. No scientific result was accepted from the
failed attempt.

## Native held-out outcome

Negative loss deltas are improvements relative to the original A1 curriculum
control. Hard exact match remained zero in both arms on values8000, association
loads, long distances, and target lengths 4/8.

| arm | test | Δ target loss | Δ exact match | Δ answer-token accuracy |
| --- | --- | ---: | ---: | ---: |
| staged | values8000 | −0.4912 | 0.0000 | 0.0000 |
| staged | distance16 | −0.4188 | 0.0000 | 0.0000 |
| staged | distance64 | −0.6825 | 0.0000 | 0.0000 |
| staged | target length 2 | −0.9064 | 0.0000 | +0.0078 |
| staged | target length 4 | −0.6057 | 0.0000 | +0.0020 |
| staged | target length 8 | −0.4666 | 0.0000 | 0.0000 |
| shuffled | values8000 | −0.0336 | 0.0000 | 0.0000 |
| shuffled | distance16 | −0.1722 | +0.0078 | +0.0078 |
| shuffled | distance64 | −0.0577 | +0.0078 | +0.0078 |
| shuffled | target length 2 | −0.7840 | 0.0000 | 0.0000 |
| shuffled | target length 4 | −0.6724 | 0.0000 | 0.0000 |
| shuffled | target length 8 | −0.3494 | 0.0000 | 0.0000 |

Against the unconditioned value-trace branch, staged query-binding loss was
lower on the hard tests by roughly 0.05–0.38 nats, while shuffled query-binding
was generally worse on association and long-target tests. Neither branch
produced a decisive exact-retrieval repair.

## Postdiagnostic result

The read-only postdiagnostic plan/code was frozen at
`ed55b28f4b29dd4ec7cb3ad5c0a61ba68e7fc0ef`. The aggregate was produced at
`2df07ef23d4d992a46bbc78397d3f0a150569271` and has sha256
`03ddd4e90b8cbe78fd264180b99669a36d33bf9ae707e667ef71981509bcbeb9`.

- Six layerwise and six query/readout analyses completed; all outputs were
  finite and source/checkpoint hashes passed.
- Staged A1 layerwise fixed pre-query value transfer changed by approximately
  −0.0065, −0.0020, and −0.0006 at target lengths 2/4/8 versus original A1.
- Shuffled A1's held-out-fitted value ceiling rose by approximately +0.163,
  +0.210, and +0.270, but fixed pre-query transfer changed only +0.0021,
  +0.0001, and +0.0002.
- Query/readout native value top-1 gains were small and localized: staged
  averaged +0.0078/+0.0020/0 at target lengths 2/4/8; shuffled was 0 at all
  three. Fixed and site-specific target transfer remained near zero.

Therefore the intervention changed representation geometry/confidence in places
but did not establish a transferable query-conditioned binding representation.

## Adjudication

This falsifies the repair prediction for this fixed, parameter-free
query-conditioned objective. It does not falsify the broader binding
hypothesis. The surviving interpretation remains recurrent
transformation/interference combined with weak compositional key→value
generalisation. Temporal decay, curriculum order, and optimization remain
modifiers; a pure final-readout explanation remains insufficient.

No additional auxiliary objective is stacked in this branch. The campaign is
stopped pending a separately preregistered decision about whether to test a
different causal mechanism or leave the result as a negative intervention.
