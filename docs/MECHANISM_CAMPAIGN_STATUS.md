# BraidFormer A0/A1 Mechanism Campaign Status

Status: **mechanism campaign complete; protected-lane localization and exact-address intervention complete; no training intervention unlocked**

This ledger is the authoritative continuation point for the bounded mechanism
campaign. It is updated and committed after every queue item. The campaign
does not modify the frozen A0 Transformer or original A1 recurrent
state-space definition.

## Baseline

- Campaign opened from commit `70ba2cd` (`Record target-length solvability results`).
- The worktree was clean at campaign start: `main...origin/main [ahead 36]`.
- Existing target-length checkpoints, sidecars, reports, and aggregate JSON
  are present and tracked/ignored according to the repository policy.
- The existing full test suite had previously passed after the target-length
  work; it is rerun as part of the first campaign-item verification.
- No GitHub push is authorized.

## Frozen evidence before this campaign

The unchanged 42M controls both memorize the fixed synthetic training sets at
100% exact match, while held-out retrieval degrades with answer-space size,
association count, distractor distance, and (for A1) target length. The
distance axis is especially diagnostic: A0 remains broadly flat, while A1
held-out target loss worsens from `+0.291780` A1-minus-A0 at distance 0 to
`+1.239856` at distance 64. Original A1 forensics found median learned
channel time constant about `8.9` tokens and no channels with `tau >= 16`.
The frozen 75%/25% long-decay-tail initialization created long-lived channels
but did not recover long-distance probe or task performance. Target-length
checkpoints show A1 matching A0 token accuracy at length 2 but falling below
A0 at lengths 4 and 8; both still memorize training perfectly.

These observations support a distance-sensitive internal failure but do not
yet distinguish temporal decay, recurrent transformation/interference,
query-conditioned corruption, binding, readout/routing, autoregressive
target-state degradation, or optimization.

## Current hypothesis set

| ID | Hypothesis | Current status |
| --- | --- | --- |
| H1 | Temporal decay erases the required value trace. | Partially supported but insufficient: distance, taus, probe collapse, and the full-freeze partial distance-64 gain support a temporal component; long-tail initialization and native retrieval results reject decay alone as sufficient. |
| H2 | Recurrent-state transformation/interference destroys the binding while the sequence advances. | Partially supported but insufficient: suppressing distractor writes preserves fixed-probe value decodability, and the protected lane preserves its own state, yet neither restores held-out retrieval. |
| H3 | Query-conditioned processing corrupts or overwrites a retained binding. | Strengthened: protected traces/lane state survive distractors but native output remains at floor, leaving query-conditioned/downstream transformation as a surviving component. |
| H4 | Binding/key-value identity is not represented compositionally under the synthetic load. | Strongly supported shared failure; it does not alone explain A1's extra distance sensitivity. |
| H5 | The required value survives internally but the output route/readout fails. | Reopened for the protected lane: its stored state and read gate survive long distance, but the output does not recover. This is not sufficient to explain native A1 alone. |
| H6 | Multi-token autoregressive target state degrades after the first output. | Weakened: A1 is already impaired on the first teacher-forced target at lengths 4 and 8, so autoregressive feedback is not the sole cause. |
| H7 | A1 needs more optimization steps or a different optimization trajectory. | Unresolved background factor; not enough evidence to justify a training intervention. |

## Queue

1. **Target-position decomposition — complete.** Frozen target-length checkpoints;
   no retraining. Measure per-position teacher-forced CE, top-1 accuracy,
   correct-token probability, correct-vs-best-incorrect logit margin, and
   autoregressive accuracy where supported.
2. **Layerwise representation forensics — complete.** Frozen checkpoints; probe
   key, value/binding identity, and target position before query, after query,
   and at every target-output position across all A1 recurrent layers. The
   result is correlational and its cross-split fixed-probe metric conflates
   retention with representation transfer; item 3 explicitly repairs that
   limitation.
3. **Query/readout localization — complete.** Added within-heldout fixed-probe
   transfer and direct logit/rank/probability localization through the query
   and target-output stages. The probes stayed near floor and did not reveal a
   strong retained-value/native-readout gap; item 4 is the next step.
4. **One reversible inference-time causal intervention — complete.** State
   replay at A1 layer 7 produced no top-1 gain and slightly worsened mean CE;
   no second intervention is authorized.
5. **Mechanism adjudication — complete.** The strongest supported explanation
   is a composite recurrent retrieval failure: compositional binding is weak,
   and A1's recurrent transformation/retention path amplifies the weakness
   with temporal separation. The optional training gate is locked.
6. **Optional single minimal training intervention — not unlocked.** No item-6
   training run was performed because the diagnostics did not converge on one
   minimal mechanism-specific factor.

## Stop gates

Stop only for a completed queue, unresolved protocol contradiction,
integrity/test/checkpoint failure, unexplained worktree changes, a required
frozen-architecture modification outside the single intervention rule, two
failed mechanism-targeted interventions, or genuinely contradictory evidence
requiring arbitrary scientific choice. A successful item is not itself a stop
condition.

## Completed campaign items

### Item 1 — target-position decomposition

The six frozen target-length checkpoints passed source-run and checkpoint
verification. At target lengths 4 and 8, A1 was already at zero held-out
teacher-forced top-1 accuracy on position 1, with mean correct-token
probability about `4.8e-5` and `6.1e-5`, respectively. A0 retained position-1
teacher-forced top-1 of `0.0234` and `0.0312`. Therefore autoregressive error
accumulation cannot be the sole explanation for the A1 collapse at those
lengths. At length 2 both models were near the measurement floor at position 1
and the A1 small edge reversed to a deficit at position 2. A0 also loses some
signal under autoregressive feedback at longer targets, but A1's length-4/8
deficit is present before generated target tokens are fed back.

The full position-wise CE, top-1, correct-token probability, logit margin, and
greedy autoregressive metrics are in
`results/a0_a1_target_position_decomposition_values8000.json` and
`docs/A0_A1_TARGET_POSITION_DECOMPOSITION_RESULT.md`. This item weakens a
pure H6 explanation and leaves H1/H2/H3/H4/H5 open for representation
forensics. Layerwise representation forensics is now complete; the next item
is query/readout localization.

Integrity: source checkpoint hashes and machine-readable finite-value checks
passed; the full test suite passed (`67 passed`); `git diff --check` passed;
raw logs/checkpoints remain ignored and no architecture/config files changed.

### Item 2 — layerwise representation forensics

The six frozen target-length checkpoints passed sidecar/hash verification and
were analyzed without retraining or checkpoint writes. Every A0 block output,
every A1 block output, every reconstructed A1 recurrent state, and final
normalization were captured at pre-query, post-query, and each target position.
Temporary ridge probes and geometry measurements were finite.

The site-fitted heldout key/value probes are essentially perfect, but that is a
same-split diagnostic ceiling and must not be treated as learned
generalization. Train-fitted-to-heldout value probes are near floor at the
longer target lengths, as are the fixed pre-query transfer probes. The current
cross-split transfer therefore supports a failure of stable transferable
binding, but cannot distinguish actual trace erasure from a changed coordinate
system or output-use failure. A1 recurrent-state target-position decodability
also weakens with depth at length 4, but target position is not the requested
value and this is not causal evidence.

This item leaves H2 recurrent transformation/interference, H3 query-conditioned
corruption, H4 binding failure, and H5 readout/routing failure open. H1 remains
compatible with the result but is not isolated by it. The full per-site
metrics, capture positions, geometry, and checkpoint hashes are in
`results/a0_a1_layerwise_representation_forensics_values8000.json`; the
interpretation caveat and detailed table are in
`docs/A0_A1_LAYERWISE_REPRESENTATION_FORENSICS_RESULT.md`.

Integrity: all six source checkpoint hashes, expected parameter counts, capture
sites, recurrent layers, finite tensors, and finite probe metrics passed. Raw
checkpoints/logs remain ignored; no A0/A1 architecture or frozen protocol was
changed. The full test suite and `git diff --check` passed in the item-2 result
commit. The next queued item is the reversible inference-time causal
intervention.

### Item 3 — query/readout localization

The six frozen checkpoints passed sidecar/hash verification. Native
teacher-forced logits were measured at every requested target position over the
full vocabulary and the 8,000-value candidate range. Temporary value readouts
were fit with deterministic four-fold cross-validation inside the heldout set;
no same-sample fitted ceiling was used.

A1's native value-range top-1 was zero at every target position for lengths 4
and 8. Fixed pre-query probes stayed at or near floor through post-query and
target sites, and site-specific probes did not produce a strong retained-value
signal or a diagnostic readout advantage. This weakens the pure H5
readout/routing hypothesis. The result favours representation/binding or
recurrent transformation over a final-head-only failure, while H1 temporal
decay, H2 state transformation/interference, H3 query-conditioned corruption,
and H4 binding failure remain jointly viable. It is correlational and requires
the predeclared reversible intervention in item 4.

The complete native metrics, fold assignments, probe ranks/cosines, capture
positions, and source checkpoint hashes are in
`results/a0_a1_query_readout_localization_values8000.json`; the concise tables
and interpretation are in
`docs/A0_A1_QUERY_READOUT_LOCALIZATION_RESULT.md`.

Integrity: source hashes, expected parameter counts, capture sites, balanced
folds, finite logits/representations, and finite diagnostic metrics passed. Raw
checkpoints/logs remain ignored; no A0/A1 architecture or frozen protocol was
changed. The full test suite and `git diff --check` are required before the
item result commit. The next queued item is the reversible inference-time
causal intervention.

### Item 4 — layer-7 state-replay intervention

The predeclared inference-only intervention copied each target value's
layer-7 recurrent state from its stored-value position to the matching target
output position. All six A0/A1 source runs were verified; the three A1 runs
were evaluated on the heldout split. Parameter digests were unchanged before
and after the functional intervention.

The prediction failed: replay produced no top-1 improvement at any target
position and slightly worsened mean value-range cross-entropy at every
position. This rejects the narrow claim that simple layer-7 state preservation
is sufficient to restore retrieval. It does not disprove every temporal,
query-conditioned, or recurrent-transformation mechanism, because the test
used one layer and one post-query state replacement. The campaign will not
stack another intervention; item 5 is mechanism adjudication.

The complete baseline/patched metrics, state norms/cosines, parameter digests,
positions, and source hashes are in
`results/a0_a1_state_replay_intervention_values8000.json`; the report is in
`docs/A0_A1_STATE_REPLAY_INTERVENTION_RESULT.md`.

Integrity: all source hashes, expected A1 parameter counts, exact positions,
finite baseline/patched tensors, and unchanged parameter digests passed. Raw
checkpoints/logs remain ignored; no architecture/config/protocol was changed.
The full test suite and `git diff --check` are required before the item result
commit. The next queued item is mechanism adjudication.

### Item 5 — mechanism adjudication

The completed evidence was audited without model execution. The strongest
supported mechanism is a composite recurrent retrieval failure: the synthetic
task already exposes a shared compositional key/value-binding failure, while
A1's learned recurrent transformation/retention path makes that failure more
severe as temporal separation grows. The evidence does not isolate one decay
constant, layer, gate, or output-head defect.

H1 temporal decay is partially supported but insufficient; H2 recurrent
transformation/interference is the leading mechanism family; H3
query-conditioned corruption and H4 shared binding/generalization remain
viable; H5 pure readout failure is weakened; H6 autoregressive accumulation is
rejected as the sole cause; H7 optimization remains unresolved but unsupported
as the primary explanation. The layer-7 replay intervention is negative and
was not stacked.

The optional training gate is **locked**. Two independent diagnostics support
a broad recurrent/temporal/binding family, but they do not identify one
minimal mechanism-specific factor. The single best follow-up outside this
campaign is a matched fresh A0/A1 run with a staged synthetic relational/
binding curriculum, keeping architecture, optimizer, compute budget, and
final heldout tests fixed.

The complete machine-readable synthesis and source hashes are in
`results/a0_a1_mechanism_adjudication.json`; the report is in
`docs/A0_A1_MECHANISM_ADJUDICATION_RESULT.md`. This queue is now complete.

Integrity: all 16 source result artifacts were present and complete; the four
campaign diagnostics were read-only; source/checkpoint hashes were preserved;
the full test suite passed after each item (`67 passed`); `git diff --check`
passed; raw logs/checkpoints remain ignored; A0 and original A1 definitions
remain untouched; no GitHub push occurred.

## Post-campaign follow-up — distractor-interval interference isolation

Status: **complete; inference-only causal diagnostic; no training intervention
unlocked**.

The committed protocol tested the original seed-17 A1 checkpoints at distances
0, 4, 16, and 64 with one association and 8,000 values. During distractors
only, every recurrent layer was run under native recurrence, decay-only
write-protection, or full state freezing. Query/key/answer processing remained
native. The distance-zero protected variants reproduced native output and
state metrics exactly within tolerance; functional native logits matched the
model forward; all outputs were finite; parameter digests were unchanged; no
checkpoint was written.

The intervention produced a useful separation. Native A1 lost the fixed
layer-7 value probe through the distractor interval, while both protected
variants preserved it at distances 4, 16, and 64. Therefore ongoing
distractor-state transformation causally contributes to intermediate trace
loss. However, the protected trace still degraded after the query. Native
retrieval was not coherently rescued: write-protection worsened distance-4 and
distance-64 value cross-entropy, improved distance-16 cross-entropy only
slightly without rank/top-1 rescue, and full freezing produced only a partial
distance-64 cross-entropy/top-1 improvement with worse mean rank.

The strongest current mechanism is therefore a composite:

```text
weak compositional key/value binding
        + distractor-state transformation/interference
        + query-conditioned/downstream transformation
```

This follow-up does not justify another stacked inference patch or a training
intervention. The complete aggregate and report are
`results/a0_a1_interference_isolation_values8000.json` and
`docs/A0_A1_INTERFERENCE_ISOLATION_RESULT.md`. The protocol is
`docs/A0_A1_INTERFERENCE_ISOLATION_DESIGN.md`; the analyzer was frozen before
execution and its source hash is recorded in the aggregate.

## Commit ledger

| Queue state | Commit | Evidence |
| --- | --- | --- |
| Campaign baseline | `70ba2cd` | Existing target-length result and clean repository state. |
| Campaign/Item 1 protocol | `e82f342` | Ledger and frozen target-position analysis plan. |
| Item 1 analyzer | `e031c2d` | Read-only decomposition implementation, committed before execution. |
| Item 1 result | `6d19ee4` | Frozen-checkpoint decomposition result/report and ledger update. |
| Item 2 protocol | `9202c1d` | Frozen layerwise representation-forensics plan. |
| Item 2 analyzer | `338a249` | Read-only layerwise capture/probe implementation. |
| Item 2 result | `9856a8b` | Frozen-checkpoint layerwise result, caveats, and ledger update. |
| Item 3 protocol | `75c4efe` | Frozen query/readout localization plan. |
| Item 3 analyzer | `a0555b0` | Cross-validated native-logit and diagnostic-readout analyzer. |
| Item 3 result | `b534b0e` | Frozen query/readout localization result and hypothesis update. |
| Item 4 protocol | `6e299a7` | Frozen layer-7 state-replay intervention plan. |
| Item 4 analyzer | `961d452` | Inference-only functional state-replay analyzer. |
| Item 4 result | `dbf4706` | Negative layer-7 replay result and integrity evidence. |
| Item 5 protocol | `cddcd83` | Frozen evidence-synthesis and intervention-gate rubric. |
| Item 5 analyzer | `861854b` | Read-only evidence audit and hypothesis synthesis. |
| Item 5 result | `ebbcddb` | Final adjudication result and completed-campaign handoff. |
| Follow-up protocol/design | `665c614` | Inference-only distractor-interval interference-isolation design. |
| Follow-up analyzer freeze | `1e88deb` | Hash-complete read-only analyzer with no-op and parameter-digest controls. |
| Follow-up result | `31fafd0` | Protected-state causal result, aggregate, report, and evidence update. |

## Protected associative lane follow-up — initial integrity stop

The fresh architectural falsification branch was frozen in
`docs/A1_PROTECTED_ASSOCIATIVE_LANE_PROTOCOL.md`. It added a separate 64-wide
protected/ordinary associative lane around an unchanged A1 base; original A0
and A1 remained untouched controls. The implementation and four distance
configs were committed before training. Three of the eight predeclared arms
completed: protected and ordinary at distance 0, and protected at distance 4.
All three reached 100% train EM; held-out EM was `3.90625%`, `3.90625%`, and
`0.78125%`, respectively. These partial numbers are not sufficient to
adjudicate the protected-lane prediction because the ordinary distance-4
comparator failed before completion.

The ordinary distance-4 arm stopped at a checkpoint serialization error after
valid checkpoints through step 300. The filesystem had only about 497 MB free,
less than the approximately 569 MB atomic checkpoint. Distances 16 and 64
were never started. No retry, deletion, or storage workaround was performed.
The read-only audit verified 63 intact checkpoint sidecars and recorded the
failure in `results/a1_protected_lane_distance_axis_interrupted.json`; its
report is `docs/A1_PROTECTED_ASSOCIATIVE_LANE_STOP_RESULT.md`.

The initial execution stopped by the campaign integrity gate, not by a
scientific null or positive result. Storage remediation and the continuation
protocol are recorded below; the initial partial metrics remain non-results.

## Current continuation point

The protected-lane distance axis, lane-state/query-output localization, and
the exact-address intervention are now complete. The localization confirms
that protected state survives but does not become a transferable
query-conditioned binding or a usable answer representation; exact record-key
addressing does not rescue it. Do not begin A2, scaling, persistent-memory,
predictive-learning, curriculum, post-training, or unrelated architecture work
under this campaign. A new training or architectural intervention would
require a separately frozen protocol outside this completed queue.

## Storage remediation — 2026-10-14

The mounted second NVMe (`/dev/sda2` at `/mnt/ssd2`) now holds the redundant
checkpoint archive at
`/mnt/ssd2/braidformer_checkpoint_archive_20261014/`. The committed retention
tool (`11ac56d`) planned and executed a hash-verified move of 808 redundant
checkpoint/sidecar pairs, totaling `459082746849` bytes. It retained 140
checkpoints on the root filesystem: the newest valid checkpoint in each run and
every checkpoint referenced by a tracked report or result. All result files,
logs, configs, reports, and provenance remain in the repository; no checkpoint
was discarded. The machine-readable retention record is
`results/checkpoint_retention_manifest.json`.

Post-move verification found 140 root checkpoints, 808 archive checkpoints,
zero missing sidecars, zero partial archive files, and approximately 429 GiB
free on `/`. This is storage housekeeping only and does not change the
protected-lane scientific status or authorize resuming training.

## Protected-lane commit ledger

| State | Commit | Evidence |
| --- | --- | --- |
| Protected-lane protocol | `32fd4d0` | Frozen protocol and predeclared prediction. |
| Protected-lane implementation | `d6358bf` | New A1P model, configs, runner, and tests; original A0/A1 untouched. |
| Runner correction | `b48ddc2` | Fixed provenance seed typo; `78 passed`. |
| Stop audit | `8bef743`, `36f5aa9` | Read-only checkpoint audit and exact failure classifier. |
| Stop result | `2f14a30` | Partial-arm report, interruption aggregate, and ledger update. |
| Retention tool | `11ac56d` | Hash-verified checkpoint archiver retaining latest/referenced root checkpoints. |
| Continuation protocol | `289b397` | Storage-resume protocol; original scientific settings unchanged. |
| Continuation analysis/result | `6a2f35c` | Complete eight-arm axis aggregate, report, and negative adjudication. |

## Protected-lane continuation result — complete negative

The continuation resumed the ordinary distance-4 comparator from its verified
step-300 checkpoint into a fresh directory, then completed fresh protected and
ordinary arms at distances 16 and 64. The eight-arm axis is now complete:

| Distance | Protected held-out EM | Ordinary held-out EM | Protected loss | Ordinary loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3.90625% | 3.90625% | 10.248867 | 10.265778 |
| 4 | 0.78125% | 0.78125% | 10.414185 | 10.554665 |
| 16 | 0% | 0% | 10.949066 | 10.671906 |
| 64 | 0% | 0% | 10.891357 | 10.842102 |

All eight arms reached 100% training EM. Protected never improved held-out EM;
its target-loss advantage at distances 0 and 4 reversed at distances 16 and
64. The lane diagnostics confirm that protected state itself was held through
the long distractor intervals, so the null is not a simple persistence failure.
The preserved state nevertheless did not yield transferable retrieval. The
minimal protected-lane hypothesis is therefore closed negative: protected
state alone is insufficient, with binding formation and query-conditioned
integration/use remaining the leading loci.

The aggregate is `results/a1_protected_lane_distance_axis_values8000.json` and
the report is `docs/A1_PROTECTED_ASSOCIATIVE_LANE_RESULT.md`. The aggregate
structurally verified 158 checkpoint records (18 for the resumed arm, whose
step-300 source checkpoint is separately retained, and 20 for each other
completed arm). No A0/original-A1 source changed and no checkpoint was
discarded by this continuation.

The continuation added 97 new root checkpoint files for its completed arms;
they remain on root because the aggregate records each checkpoint path for
provenance. Current post-campaign storage is approximately 377 GiB free on
root and 597 GiB free on `/mnt/ssd2`; the verified 428 GiB redundant archive
remains intact.

## 2026-10-14 — D053: localize protected-lane binding and query use

The read-only protocol was frozen in
`docs/A1_PROTECTED_LANE_QUERY_BINDING_LOCALIZATION_PROTOCOL.md` at commit
`5d60713`; the analyzer was committed before execution and then corrected only
for source-arm ordering and the intentional distance-zero stage alias in
commits `6bfd0a6`, `79416f6`, and `8cfe0d2`. The final analyzer also records the
train/held-out value overlap caveat in `90ca381`. It loaded the eight verified
final protected/ordinary lane checkpoints at distances 0, 4, 16, and 64.
No weights or checkpoints were modified.

The primary stages were post-value, post-distractor, post-query-key, and
answer-marker, with the query-marker stage retained as an auxiliary site. The
protected lane's memory-value cosine to its post-value state was `1.0000` at
all later stages and all distances. Its answer read gate stayed high
(`0.9644`–`0.9756`), yet native value-range exact retrieval remained at the
existing floor: `3.90625%` at distance 0, `0.78125%` at distance 4, and `0%`
at distances 16 and 64. This rejects storage persistence as a sufficient
repair.

The ordinary lane showed the expected destructive state evolution: at distance
64 its memory-value cosine to post-value was `-0.0636` after distractors,
`0.0192` after the query key, and `0.0661` at the answer marker; its answer
read gate fell to `0.0366`. Thus the diagnostic independently reproduces the
distractor/interference contribution without confusing it with the protected
lane's failure.

The stronger within-held-out cross-validated probes show what survives in the
protected state but is not usable as a binding. At the answer marker, the
protected memory-value readout had mean target ranks `896.7`, `974.9`,
`1,082.6`, and `1,070.2` at distances 0, 4, 16, and 64, respectively, while
the query-conditioned readout ranks were `3,078.3`, `2,973.4`, `3,592.8`, and
`3,503.6`. Adding the stored key (`pair_state`) worsened the corresponding
ranks to `1,967.3`, `2,315.9`, `2,365.6`, and `1,900.7`. The native/final
answer representation remained weak, with answer readout ranks from about
`3,315` to `4,333` and zero top-1 in the long-distance protected arms.

The fixed train-to-held-out top-1 probe is not treated as transfer above its
floor: the deterministic train and held-out value draws share 6 of 128 values
(`0.046875`), while the key ranges are disjoint. The machine-readable result
records this overlap explicitly. The local rank/cosine pattern, exact state
cosine, native output, and ordinary-mode comparison are the primary evidence.

### Adjudication

| Hypothesis | Status after D053 |
| --- | --- |
| Passive temporal decay alone | Rejected as sufficient: protected state survives unchanged while retrieval remains poor. |
| Distractor recurrent-state interference | Supported for ordinary A1/lane evolution; not the complete protected-lane explanation. |
| Query-conditioned transformation corrupts or misuses the state | Strengthened: protected value survives the query, while query-conditioned selection becomes substantially less decodable than value-only state. |
| Weak compositional key/value binding or associative addressing | Strongly supported: pair and query-conditioned probes do not produce a transferable/useful value selection. |
| Pure final-head/readout failure | Weakened as the sole explanation: the stored value is not converted into a strong final answer representation, so the failure is upstream-plus-downstream rather than only the final head. |
| Autoregressive target feedback | Not implicated by this one-token lane diagnostic; prior target-position work already rejects it as the sole cause. |
| Optimization trajectory | Remains a background alternative, not isolated by this frozen analysis. |

The protected-lane branch is therefore closed as a composite negative:

```text
protected storage succeeds
        -> transferable key/value binding is weak
        -> query-conditioned selection degrades the usable code
        -> answer integration does not restore retrieval
```

No training intervention was unlocked. A new architectural or training test
would require a separately frozen protocol outside this completed campaign.
The aggregate is
`results/a1_protected_lane_query_binding_localization.json`, the report is
`docs/A1_PROTECTED_LANE_QUERY_BINDING_LOCALIZATION_RESULT.md`, and the
read-only analyzer is
`scripts/analyze_a1_protected_lane_query_binding_localization.py`.

Integrity: all eight final checkpoint sidecars and hashes verified; every
tensor, probe, logit, and metric was finite; all eight parameter digests were
unchanged; no checkpoint was written; no A0 or original-A1 definition changed.
The full test suite, `git diff --check`, and untracked-artifact audit are
required before the result commit.

## 2026-10-14 — D054: exact address does not make preserved value usable

The exact-address protocol was frozen in
`docs/A1_PROTECTED_LANE_EXACT_ADDRESS_PROTOCOL.md` at commit `f9badbf`; the
read-only analyzer was committed at `9e97ec2`. It replaced only the learned
key-similarity/read gate at the answer marker with exact record-key/query-key
token equality. The stored protected value, trained lane readout, base hidden
state, output head, and all other recurrence remained unchanged.

The eight verified protected/ordinary checkpoints were evaluated on the
held-out 8,000-value battery with host CUDA on the visible RTX 3060 Ti; no
checkpoint or weight was written. An earlier sandboxed CPU fallback was
superseded by this CUDA execution before the result was treated as final.
Source hashes, checkpoint sidecars, canonical exact-key assertions,
native-forward equivalence, finite tensors/metrics, and unchanged parameter
digests all passed.

Exact addressing did not rescue protected retrieval. Protected value-range
top-1 remained `3.90625%`, `0.78125%`, `0%`, and `0%` at distances 0, 4, 16,
and 64. Its value cross-entropy deltas (exact minus native) were only
`-0.0002`, `+0.0000`, `-0.0006`, and `-0.0019`, with mean-rank changes of at
most one rank and no long-distance top-1 recovery. Protected memory-value
cosine remained `1.0000` and the exact gate was one for every example, so this
null is not explained by failed storage persistence or missing exact key
matches.

The ordinary lane also had no top-1 recovery; exact addressing worsened value
cross-entropy at distances 4, 16, and 64 by `+0.0072`, `+0.1175`, and
`+0.1630`. The intervention therefore closes the narrow learned-address-gate
hypothesis: even an exact address oracle does not make the preserved value
usable. Weak binding/representation plus query-conditioned or downstream
value integration remain the stronger loci. No second inference patch,
training run, or architecture change is authorized by this result.

The machine-readable result is
`results/a1_protected_lane_exact_address_intervention.json` (sha256
`f9f1bca66c3e25a9c2a8c402d108aa8a0e6c27a7b29675e36b60146fe6918220`); the
report is
`docs/A1_PROTECTED_LANE_EXACT_ADDRESS_INTERVENTION_RESULT.md`; the analyzer
source hash is
`7a57101badf7130c3b14c49bb977e83bf8fe88fa58031d0eb437fd3d43c7f894`.

