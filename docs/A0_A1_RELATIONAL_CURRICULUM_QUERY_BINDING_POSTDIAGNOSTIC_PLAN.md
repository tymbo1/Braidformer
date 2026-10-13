# A1 query-conditioned binding postdiagnostic plan

Status: **frozen read-only postdiagnostic plan**.

## Purpose

Rerun the exact layerwise representation and query/readout localization
analyses used for the original A1 curriculum controls and the unconditioned
value-trace intervention. The purpose is to test the preregistered mechanistic
prediction, not to tune or select the new checkpoints.

## Inputs

- query-conditioned staged and shuffled A1 checkpoints from
  `results/a0_a1_relational_curriculum_query_binding_intervention.json`;
- original A0/A1 curriculum probe aggregates as immutable controls;
- the completed unconditioned value-trace postdiagnostic aggregate as a
  matched intervention comparator.

All source and checkpoint hashes must be verified before loading. No model
parameters, checkpoint files, configs, or source controls may be modified.

## Analyses

Run the existing frozen analyses without changing their metrics:

1. layerwise probes at target lengths 2/4/8, including key/value held-out-fit,
   fixed pre-query transfer, and target-position transfer;
2. native query/readout localization at every target position, including
   native value top-1, fixed pre-query transfer, and site-specific transfer.

Report both:

- new checkpoint minus original A1 curriculum control;
- new checkpoint minus the unconditioned value-trace checkpoint.

The decisive mechanistic signal is improvement in fixed or site-specific
cross-validated transfer. A higher held-out-fitted ceiling without transfer is
not sufficient evidence of a stable query-conditioned binding code.

## Integrity

The wrapper must require the committed analysis hash, verify all source
aggregate hashes and checkpoint sidecars, reject non-finite outputs, and write
one machine-readable aggregate. The analysis is read-only and may run on CUDA;
it must not alter the source checkpoints or architecture definitions.

## Outputs

- `results/a0_a1_relational_curriculum_query_binding_postdiagnostics.json`
- `docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_BINDING_POSTDIAGNOSTIC_RESULT.md`
