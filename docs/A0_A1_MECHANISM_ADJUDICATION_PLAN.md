# A0/A1 mechanism adjudication plan

Status: **frozen evidence-synthesis plan; no new model execution**

## Purpose

Adjudicate the mechanism behind A1's synthetic retrieval weakness using only
the completed frozen behavioral, decay, target-position, representation,
query/readout, and state-replay evidence. This item does not authorize a new
model, intervention, training run, dataset, or architecture change.

## Evidence sources

The synthesis must verify and cite the repository artifacts for:

1. value-space, association-count, distractor-distance, and target-length
   behavior;
2. original A1 decay/time-constant and frozen-state probe forensics;
3. long-decay-tail intervention;
4. target-position decomposition;
5. layerwise representation forensics;
6. query/readout localization; and
7. the predeclared layer-7 state-replay intervention.

For every source, preserve its commit/hash identity and distinguish direct
measurement from interpretation. Do not treat same-split fitted probe
ceilings as generalization evidence.

## Adjudication rubric

For each hypothesis H1–H7, record:

- evidence supporting it;
- evidence against it;
- whether it is contradicted, weakened, supported, or remains unresolved;
- the strongest surviving alternative; and
- whether the evidence is sufficient to unlock the one optional
  mechanism-targeted training intervention.

The final ranking must explicitly separate:

- temporal decay from generic recurrent transformation/interference;
- query-conditioned corruption from pre-query binding/generalization failure;
- retained-information/readout failure from actual representation loss; and
- optimization effects from architectural/task effects.

## Gate rule

Unlock the optional training intervention only if at least two independent
diagnostics converge on one mechanism-specific factor and a single minimal
factor change has a predeclared, falsifiable prediction. If the evidence only
supports a composite or underspecified mechanism, keep the gate locked and
record the exact missing discriminator. A negative inference-time intervention
must not be followed by an unplanned second intervention.

## Outputs

- `results/a0_a1_mechanism_adjudication.json`
- `docs/A0_A1_MECHANISM_ADJUDICATION_RESULT.md`
- final update to `docs/MECHANISM_CAMPAIGN_STATUS.md`
