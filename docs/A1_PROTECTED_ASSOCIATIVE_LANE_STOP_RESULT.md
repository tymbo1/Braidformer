# Protected associative lane follow-up — integrity-stop result

Status: **stopped before scientific completion; no lane conclusion authorized**.

The protected-associative-lane protocol was frozen at
`32fd4d052d6ff1c5783472b9a673885c4e61985b`. The implementation was frozen at
`d6358bf` and the provenance-seed fix was committed at
`b48ddc24658eacfd93a98f4a3c97c66bc6ca7f0d`. The branch kept the original A0
and original A1 definitions untouched and used a 64-wide lane with
`41,969,346` total parameters.

The campaign began sequentially with the predeclared protected/ordinary arms
at distances 0, 4, 16, and 64. Three arms completed:

| Distance | Mode | Train EM | Held-out EM | Held-out target loss | CUDA s | Checkpoints |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | protected | 100% | 3.90625% | 10.248867 | 452.00 | 20 |
| 0 | ordinary | 100% | 3.90625% | 10.265778 | 447.84 | 20 |
| 4 | protected | 100% | 0.78125% | 10.414185 | 568.65 | 20 |

The ordinary distance-4 arm reached step 400 and had valid checkpoints through
step 300. Its next atomic checkpoint write failed with PyTorch's
`inline_container.cc:672` unexpected-position error. The filesystem was full:
`/home/timbushnell` and `/tmp` reported 100% use with approximately 497 MB
free, while one checkpoint is approximately 569 MB. The failed arm is not a
scientific result, and no further training was attempted. Distances 16 and 64
were not started.

The read-only stop audit verified all 63 intact checkpoint sidecars from the
three completed and one partial arms, confirmed the expected provenance and
parameter counts, and found no source-checkpoint mutation. The machine-readable
record is
`results/a1_protected_lane_distance_axis_interrupted.json` (SHA-256
`ac0e3ce0f3c9331c894b0cc04c83dbc90558410d3911e8f706035386bf6dc3e1`). The
audit implementation was committed at `8bef743`, with the failure-classifier
fix at `36f5aa9`.

These partial observations must not be used to claim that protected
persistence helps or fails. The distance-0 pair is a null separation, and the
distance-4 protected result has no parameter-matched ordinary comparator. The
best next action is storage remediation followed by a separately verified
restart/resume protocol review; it is not another architecture modification.

Integrity at stop:

- full test suite: `78 passed`;
- `git diff --check`: passed;
- worktree: clean at audit time before this report/ledger commit;
- raw checkpoints/logs: local and ignored, not pushed;
- no A0/original-A1 source changes;
- no GitHub push.
