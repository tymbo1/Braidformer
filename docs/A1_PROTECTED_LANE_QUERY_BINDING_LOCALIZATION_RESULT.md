# A1 protected-lane query/binding localization result

Status: **complete; read-only frozen-checkpoint analysis**.

Protocol: `docs/A1_PROTECTED_LANE_QUERY_BINDING_LOCALIZATION_PROTOCOL.md` (sha256 `63456020d89b2f5003559b233d196894d2cf8b35752e279e4c0b839123b49da9`).
Source aggregate: `results/a1_protected_lane_distance_axis_values8000.json` (sha256 `afdeeb42b2379f38ac7a592b805dcad1b2acadbeace0eaa57c43472f4472a158`).

The primary stages are post-value, post-distractor, post-query-key, and answer-marker. Fixed probes are fit on the 128-example training split and evaluated on the disjoint held-out split; within-held-out cross-validation is retained as a local decodability diagnostic.
The train and held-out value draws share 6 of 128 value tokens, so 6/128 is an explicit fixed-probe overlap floor; the fixed top-1 columns are not interpreted as transfer above that floor. The key ranges are disjoint.

## Summary

| distance | mode | native value top-1 | native value CE | memory value post-write | post-distractor | post-query | answer | pair answer | query-conditioned answer | output-representation answer | lane target-logit delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | protected | 0.0391 | 9.4943 | 0.0469 | 0.0469 | 0.0469 | 0.0469 | 0.0312 | 0.0234 | 0.0391 | -0.0031 |
| 0 | ordinary | 0.0391 | 9.5206 | 0.0469 | 0.0469 | 0.0234 | 0.0312 | 0.0078 | 0.0156 | 0.0391 | 0.0163 |
| 4 | protected | 0.0078 | 9.6847 | 0.0469 | 0.0469 | 0.0469 | 0.0469 | 0.0312 | 0.0234 | 0.0078 | -0.0206 |
| 4 | ordinary | 0.0078 | 9.9585 | 0.0391 | 0.0469 | 0.0234 | 0.0156 | 0.0078 | 0.0078 | 0.0078 | -0.0579 |
| 16 | protected | 0.0000 | 10.4107 | 0.0469 | 0.0469 | 0.0469 | 0.0469 | 0.0469 | 0.0078 | 0.0000 | 0.0035 |
| 16 | ordinary | 0.0000 | 9.9049 | 0.0156 | 0.0156 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0246 |
| 64 | protected | 0.0000 | 10.3191 | 0.0469 | 0.0469 | 0.0469 | 0.0469 | 0.0391 | 0.0156 | 0.0000 | 0.0521 |
| 64 | ordinary | 0.0000 | 10.1938 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0238 |

## Key observations

The protected lane's memory-value cosine to its post-value state was `1.0000`
at every later stage and every distance. Its answer read gate remained high,
between `0.9644` and `0.9756`, but native value-range exact retrieval stayed at
`3.90625%`, `0.78125%`, `0%`, and `0%` for distances 0, 4, 16, and 64. The
stored value therefore survives; storage persistence is not sufficient.

The ordinary lane provides the expected interference contrast. At distance 64
its memory-value cosine to post-value was `-0.0636` after distractors, `0.0192`
after the query key, and `0.0661` at the answer marker; its answer read gate
was only `0.0366`.

Within-heldout four-fold probes give the more useful local representation
measure. At the answer marker, protected `memory_value` mean target rank was
`896.7`, `974.9`, `1,082.6`, and `1,070.2` at distances 0, 4, 16, and 64.
The corresponding `query_conditioned` ranks were `3,078.3`, `2,973.4`,
`3,592.8`, and `3,503.6`; `pair_state` ranks were `1,967.3`, `2,315.9`,
`2,365.6`, and `1,900.7`. Adding the key and then the query did not produce a
usable transferable selection code. The final normalized answer readout also
remained weak, with mean ranks ranging from about `3,315` to `4,333`.

The fixed train-to-held-out top-1 values must be read cautiously: the
deterministic splits share 6 of 128 value tokens (`0.046875`), exactly the
fixed-probe overlap floor, while their key ranges are disjoint. That overlap
and the `1/8000` candidate-space chance level are recorded in the JSON; fixed
top-1 is not treated as transfer above the floor.

## Mechanism adjudication

| Hypothesis | Status |
| --- | --- |
| Passive temporal decay alone | Rejected as sufficient: protected state is unchanged while retrieval remains poor. |
| Distractor recurrent-state interference | Supported for ordinary state evolution, but not the complete protected-lane explanation. |
| Query-conditioned transformation corrupts or misuses the state | Strengthened: the protected value survives the query, while query-conditioned selection is less decodable than value-only state. |
| Weak compositional key/value binding or associative addressing | Strongly supported: pair and query-conditioned probes do not produce a stable usable value selection. |
| Pure final-head/readout failure | Weakened as the sole explanation: the stored value is not converted into a strong answer representation, so the failure spans binding/use and downstream integration. |
| Optimization trajectory | Remains a background alternative, not isolated by this frozen analysis. |

The supported sequence is:

```text
protected storage succeeds
        -> transferable key/value binding is weak
        -> query-conditioned selection degrades the usable code
        -> answer integration does not restore retrieval
```

No training intervention is unlocked. A new architectural or training test
requires a separately frozen protocol outside this completed campaign.

## Reading boundary

A fixed-probe drop before the query localizes a cross-example storage/coordinate problem. Survival through post-distractor followed by a drop at post-query-key supports query-conditioned transformation. A strong pair/query-conditioned diagnostic with weak native output supports downstream integration or readout failure. These are frozen-state localization results, not causal claims about a new intervention.

## Integrity

Analyzed arms: 8; every final checkpoint and sidecar was hash verified. Parameter digests were unchanged for every arm. All tensors, probes, logits, and aggregate metrics were finite. No checkpoint was written.
