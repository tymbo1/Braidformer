# A0/A1 inference-only interference-isolation result

Status: **complete; reversible inference-only intervention**.

Protocol: `docs/A0_A1_INTERFERENCE_ISOLATION_DESIGN.md` (sha256 `a1c24d7b61601674c7092e5a6f8c86e0e8189a40c9140840fc337d3c020f78b9`).
Analyzer: `scripts/analyze_a1_interference_isolation.py` (sha256 `07261b756e4fd2122677f37cf67c7796e6452e13738714d27e4a2789d9a131ee`).
Source distance aggregate sha256: `3a40736a783fd1ac977fe2b6651a6e015891d85699f6fee2e3698cc0917a8261`.

The write-protected variant suppressed new recurrent writes only between the stored value and query. The fully frozen variant additionally suppressed passive decay during that interval. Query/key/answer processing remained native.

## Native and intervention metrics

Negative CE deltas are improvements relative to native A1. Primary rows use the 8,000-value candidate range.

| distance | variant | value CE | Δ CE | value top-1 | Δ top-1 | mean rank | Δ rank | exact | Δ exact |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | native | 9.453497 | +0.000000 | 0.0391 | +0.0000 | 2792.1 | +0.0 | 0.0391 | +0.0000 |
| 0 | write_protected | 9.453497 | +0.000000 | 0.0391 | +0.0000 | 2792.1 | +0.0 | 0.0391 | +0.0000 |
| 0 | fully_frozen | 9.453497 | +0.000000 | 0.0391 | +0.0000 | 2792.1 | +0.0 | 0.0391 | +0.0000 |
| 4 | native | 9.766390 | +0.000000 | 0.0156 | +0.0000 | 3160.5 | +0.0 | 0.0156 | +0.0000 |
| 4 | write_protected | 10.005584 | +0.239194 | 0.0078 | -0.0078 | 3375.1 | +214.6 | 0.0078 | -0.0078 |
| 4 | fully_frozen | 9.906113 | +0.139723 | 0.0078 | -0.0078 | 3162.7 | +2.2 | 0.0078 | -0.0078 |
| 16 | native | 10.193719 | +0.000000 | 0.0000 | +0.0000 | 3839.3 | +0.0 | 0.0000 | +0.0000 |
| 16 | write_protected | 10.159332 | -0.034387 | 0.0000 | +0.0000 | 3895.1 | +55.8 | 0.0000 | +0.0000 |
| 16 | fully_frozen | 10.244604 | +0.050885 | 0.0000 | +0.0000 | 3855.1 | +15.8 | 0.0000 | +0.0000 |
| 64 | native | 10.467443 | +0.000000 | 0.0000 | +0.0000 | 3844.8 | +0.0 | 0.0000 | +0.0000 |
| 64 | write_protected | 10.515527 | +0.048084 | 0.0000 | +0.0000 | 3846.7 | +1.9 | 0.0000 | +0.0000 |
| 64 | fully_frozen | 10.369745 | -0.097697 | 0.0078 | +0.0078 | 4059.1 | +214.2 | 0.0078 | +0.0078 |

## Predeclared interpretation

The machine-readable deltas are authoritative. Write-protected improvement at distances 16/64 supports active distractor-write interference; improvement only under full freezing supports a decay or decay/update interaction; a null result leaves query-conditioned corruption, pre-query binding/generalisation, and other recurrent transformations as the leading explanations.

## Observed adjudication

The protected variants preserve the fixed layer-7 value probe through the distractor interval at distances 4, 16, and 64, whereas native A1 loses that transfer. This is causal evidence that ongoing distractor-state transformation contributes to trace loss. However, the preserved trace is still degraded after the query, and native output retrieval is not coherently restored: write protection worsens distance 4 and 64 value CE, improves distance 16 CE only slightly without rank/top-1 rescue, and full freezing produces a partial distance-64 CE/top-1 improvement with worse mean rank. The result therefore supports a composite mechanism—distractor interference plus query-conditioned/downstream transformation—not a sufficient production repair.

## Integrity

All source checkpoints and hashes were verified. The distance-zero protected variants reproduced native metrics within tolerance, functional native logits matched the model forward, all tensors were finite, and parameter digests were unchanged. No checkpoint was written.
