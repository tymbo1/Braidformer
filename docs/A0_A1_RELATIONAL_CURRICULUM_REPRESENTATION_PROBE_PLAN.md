# A0/A1 relational-curriculum representation-probe plan

Status: **frozen read-only analysis plan**.

## Question

The relational-curriculum experiment changed the learning order while holding
architecture, exposure, optimizer controls, seed, and token budget fixed. This
analysis tests whether the resulting A1 staged-versus-shuffled performance
difference is accompanied by a change in the decodability or persistence of
keys, associated values, binding identity, and output-position state. A0 is run
with the same measurements as an architectural control.

The execution order is A1 staged, A1 shuffled, A0 staged, and A0 shuffled. No
result is interpreted as a curriculum or architecture effect until the matched
arms have passed the same integrity checks.

## Frozen scope and provenance

- Use only the four completed final checkpoints from relational-curriculum v1.
- Architectures: frozen A0 and original A1; no model or checkpoint changes.
- Curriculum arms: `staged` and `shuffled` within each architecture.
- Seed: `17`.
- Target lengths: `2`, `4`, and `8`.
- Evaluation task: values8000, one association, distractor distance zero,
  using the existing deterministic train/held-out examples.
- Capture method: reuse
  `scripts/analyze_layerwise_representation_forensics.py` without changing its
  probe equations, capture sites, ridge strength, or target-position labels.
- Analysis is in `model.eval()` with gradients disabled. Probe weights exist
  only in memory and are not written into checkpoints.

The source aggregate is pinned to:

```text
results/a0_a1_relational_curriculum_v1.json
sha256: 66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71
protocol commit: 30488e4f6b95994f02e15de9fab5710ce9af2aa4
protocol sha256: eb3e172d6057d070ad4d5c6a64529246e9831c688f7a7d609db6abc5942696e5
curriculum runner sha256: 56e0d0f7361b86c1d0fb3deef031cf8203c75fabd4fd764e943f131f9b7062dd
order-independent exposure sha256: fbbc2ade9f57566ecc18a1752b828880d2218b7ff42692638c6ecf74ea4e4d78
```

The expected checkpoint digests are:

| architecture | arm | checkpoint sha256 |
| --- | --- | --- |
| A1 | staged | `0b6ba5eff90ff477a0dc8dc810e4bcfa1b396f2b31f5b2a72c100c5645eb5e8b` |
| A1 | shuffled | `be7a71017ea6ea88dbbb99476611e4bfef98c398b3788e74d2a57fd40a5fec8b` |
| A0 | staged | `8c15b235dd5329ef349c64e96f24476dcfdaa904cdf4e654f3ad92ca6427c105` |
| A0 | shuffled | `e6c998e3d90acb5b102ddf4f1f3dd4cfb892400f90543e798001fe814126ebff` |

The curriculum configuration hashes are A1
`cf4c8b387c9c9d9bdc01df66382514131ac5953e85c8bd8776ff66e1b988d1fa` and A0
`39ed06dc15a4d8260dd9074e3816f0d08ac12d81edf89a7453ea942e511ded01`.

## Measurements

At `pre_query`, `post_query`, and every target-output position, capture every
A0 block output, every A1 block output, every A1 recurrent state, and the final
normalized representation. For each path and site, record the existing
method's:

- key identity train-fit/held-out and held-out-fit/held-out top-1 metrics;
- associated-value metrics for each target position;
- fixed pre-query train-fit transfer to held-out examples;
- paired key/value binding top-1 masks;
- output-position train-fit/held-out accuracy;
- held-out representation geometry.

The primary comparison is staged minus shuffled within architecture, target
length, representation path, and metric. Site-fitted held-out probes are
decodability ceilings; fixed pre-query transfer is the main coordinate/
persistence diagnostic.

## Predeclared interpretation rules

- Higher staged fixed-transfer value accuracy at later sites, together with
  higher site-fitted value accuracy, supports curriculum-dependent retention or
  representation formation.
- Higher site-fitted value accuracy without higher fixed-transfer accuracy
  indicates that information may remain decodable but its coordinate or use has
  changed.
- Similar value probes but different native task loss/accuracy leave query use,
  routing, readout, or optimization as the leading explanation; this analysis
  does not establish causality.
- A1-specific changes that are absent in A0 strengthen an A1 training-history
  interaction; matched A0 changes indicate a shared curriculum effect.
- No representation-probe result unlocks a training intervention by itself.
  Any causal test remains a separate, predeclared queue item.

## Integrity and outputs

Before loading any checkpoint, the analysis must verify the source aggregate
hash, summary fields, configuration hashes, checkpoint sidecars and checkpoint
digests, parameter counts, seed, token budget, and exposure hash. It must fail
nonzero on missing artifacts, hash drift, malformed summaries, missing capture
sites/layers, or non-finite tensors/metrics. It must record the exact analysis
commit and hashes of this plan and the reused forensic script.

The analysis writes:

- `results/a0_a1_relational_curriculum_representation_probes.json`
- `docs/A0_A1_RELATIONAL_CURRICULUM_REPRESENTATION_PROBE_RESULT.md`
- an updated `docs/RELATIONAL_CURRICULUM_CAMPAIGN_STATUS.md`

Raw checkpoints and logs remain under the repository's ignored run directories
and are not copied or modified.
