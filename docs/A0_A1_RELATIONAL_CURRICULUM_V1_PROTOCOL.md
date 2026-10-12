# A0/A1 staged relational-binding curriculum protocol v1

Status: **frozen for execution**.

## Question

The completed mechanism campaign established a shared compositional binding
failure and a stronger distance-sensitive recurrent retrieval failure in A1,
but did not identify one sufficient state or readout intervention. This
experiment tests the recorded next hypothesis: the present boundary may be
partly developmental/curriculum-induced rather than solely architectural.

The controlled question is:

> Does a staged synthetic relational/binding curriculum improve held-out
> retrieval over the same examples presented in a deterministic shuffled order,
> while A0 and the original A1 architecture remain unchanged?

## Arms

Four fresh runs are required:

| architecture | `staged` | `shuffled` |
| --- | --- | --- |
| A0 | staged curriculum | identical exposure, shuffled stage order |
| A1 | staged curriculum | identical exposure, shuffled stage order |

The shuffled arm is the primary matched control. It contains the same number
of examples from every stage and consumes the same fixed-shape batches; only
the order of stage presentation changes. Historical TinyStories-only and
frozen values8000 runs remain contextual references, not substitutes for the
fresh matched control.

## Curriculum

Each optimizer step consumes 16 examples of exactly 29 tokens. Shorter tasks
are left-padded with a reserved neutral marker so the compute shape and token
counter are identical across all stages and arms. The answer remains at the
right edge; only answer-token labels contribute to the supervised loss.

The 2,000-step staged schedule is:

| steps | stage | task |
| ---: | --- | --- |
| 250 | `direct64` | one direct key/value binding, 64-value pool |
| 250 | `chain64` | two-hop relational chain, 64-value pool |
| 250 | `direct128` | one direct binding, 128-value pool |
| 250 | `direct1024` | one direct binding, 1,024-value pool |
| 250 | `associations2_1024` | two simultaneous bindings, 1,024-value pool |
| 250 | `associations4_1024` | four simultaneous bindings, 1,024-value pool |
| 250 | `associations8_8000` | eight simultaneous bindings, 8,000-value pool |
| 250 | `stress8000` | deterministic mixture of direct, multi-binding, distractor, and target-length tasks |

The chain stage uses two explicit relations, `key_a -> key_b` and
`key_b -> value`, and asks for the value associated with `key_a`. The stress
stage cycles through direct values8000, associations8, distractor distances 4
and 16, and target lengths 2, 4, and 8. It does not introduce a new final
test or alter the frozen generator.

The shuffled arm uses the exact same stage schedule multiset and the exact
same within-stage example stream. A fixed seed-derived permutation changes
only the global stage order. The protocol records both the schedule hash and
an order-independent exposure hash; the latter must match across arms.

## Controls held fixed

- original A0 and original A1 model definitions and parameterization;
- seed `17`, with fresh model and optimizer state per run;
- 8 layers, hidden size 512, vocabulary 32,000, context length 512;
- micro-batch 2 and gradient accumulation 8 (effective batch 16);
- AdamW, learning rate `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.1`;
- one-indexed 20-step warmup plus cosine decay over 2,000 optimizer steps;
- gradient clipping at 1.0, fp16 autocast, deterministic seed controls;
- 928,000 training tokens (`2,000 * 16 * 29`), checkpoint cadence 100 steps;
- the existing atomic checkpoint, numerical-integrity, and recovery tooling;
- no A0/A1 source edits, no decay-tail initialization, no architecture patch.

The A1 initialization remains the original `uniform_0.9` definition. The
curriculum changes data order/content only; it is not an A1 modification.

## Final evaluation

Each completed checkpoint is evaluated without further training on the
existing frozen generators and seed-17 held-out split:

- value pools 128, 256, 1,024, and 8,000 with one binding;
- 2, 4, and 8 simultaneous bindings at values8000;
- distractor distances 4, 16, and 64 at values8000;
- target lengths 2, 4, and 8 at values8000.

For every axis, record target cross-entropy, greedy answer-token accuracy, and
greedy exact match. The final aggregate also records optimizer steps, tokens,
CUDA seconds, checkpoint hashes, config/script/protocol hashes, and numerical
status. No final test result is fed back into training.

## Predictions and falsification

Primary prediction: if curriculum/developmental history is a major contributor,
the staged arm will improve held-out retrieval relative to the shuffled arm,
especially on associations, distance, and target-length tests. A0 and A1 may
benefit differently, but the comparison is within architecture.

Null result: no meaningful staged-vs-shuffled improvement on either
architecture. This would strengthen the case that the observed weakness is not
fixed by ordering these relational experiences alone.

Architecture interaction: if A1 improves disproportionately, its recurrent
state may require appropriate developmental examples; if A0 improves while A1
does not, the shared data failure is separable from A1's recurrent penalty.

The experiment does not authorize A2, a relational store, decay changes,
scaling, longer training, new datasets, or predictive-learning/inheritance
work. A positive result is curriculum evidence, not proof of a mechanism.

## Integrity gate

Execution must be invoked with the exact commit containing this protocol,
configs, generator, tests, and runner. Before execution, the cheap generator
tests and full test suite must pass. After each arm and after aggregation:

1. verify every machine-readable summary and checkpoint sidecar;
2. verify all four run statuses are complete and numerical failure status is empty;
3. run `git diff --check` and the full pytest suite;
4. inspect Git status/diff for accidental architecture or protocol changes;
5. confirm raw runs/checkpoints/logs remain ignored and untracked.

No GitHub push is permitted.
