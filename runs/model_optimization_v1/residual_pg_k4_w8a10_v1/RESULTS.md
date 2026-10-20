# K=4 residual PPO: completed run and reserved test

The run improves on its initial RL policy on the model-selection validation
seeds, but the selected RL policy remains substantially worse than HEUR. The
predeclared reserved test confirms that the final performance objective has
not been achieved. This review changes no training code or task parameters.

## Natural deterministic validation (102000-102009)

| Model | Success | Mean return | Violations per environment step |
|---|---:|---:|---:|
| Initial RL | 2/10 (20%) | -9041.4 | 8.7249 |
| Selected RL, iter 3 (fourth update) | 4/10 (40%) | -7657.8 | 8.0563 |
| Latest RL, iter 29 | 4/10 (40%) | -8628.2 | 8.3409 |
| HEUR | 9/10 (90%) | -3436.5 | 7.2592 |

Selected RL improves success by 20 percentage points, return by 1383.6 and
violations/step by 0.6686 relative to the initial policy. It remains 50
percentage points behind HEUR, with lower return and more violations.
Iter 3 is the selected checkpoint, not iter 29. The latter has the same
success count but return is 970.4 worse than the selected model. No later
validation exceeds the fourth update's selection rank.

New successes are seeds 102006 and 102008. Seed 102006 has no competing
allocation candidates before or after training; seed 102008 goes from one
event to zero. These gains establish that task residual adaptation can change
previously unreachable bid-only trajectories. They do not establish that
improved winner selection caused the gains. A causal attribution requires a
separate task-residual versus bid ablation.

## Predeclared reserved test (162000-162029)

The best checkpoint was fixed by validation before testing; no model or
hyperparameter selection used this test. Natural requests and full penalties
were used; the forced conflict probe was disabled.

| Controller | Success | Mean return | Violations per environment step |
|---|---:|---:|---:|
| Selected RL | 11/30 (36.7%) | -8125.9 | 7.8982 |
| HEUR | 25/30 (83.3%) | -3550.0 | 7.3999 |

RL is behind by 46.7 percentage points, 4575.9 return and 0.4983
violations/step. Paired outcomes include 2 RL-only successes and 16 HEUR-only
successes. These thirty seeds are now consumed as test evidence and must not
be reused as fresh held-out data for future tuning. The initial RL checkpoint
was not tested on this set, so no claim of before/after generalization gain
is made.

There were 57 natural RL multi-candidate allocation events and 14 winners
different from the fixed-score comparator. Winner changes alone are not proof
of improved conflict utility or coverage of all thirteen rule-conflict pairs.
The two controllers produce different trajectories and opportunities.

## Execution and learning signal

- Completed 30 updates, 240 stochastic training episodes, seeds 160000-160239,
  156731 environment steps; elapsed training time 3962.594 seconds.
- K=4, W=8, A=10, C04=80, sector-axis limit 60 degrees; full penalties and
  natural requests, with no training-only conflict exposure.
- 177 informative environment-copy events and 354 candidate branches across
  240 episodes, about 0.738 informative events per episode. Arbitration
  credit remains sparse; iter 19 had no informative event.
- Task residuals and bid tensors updated, candidate output weights became
  trainable, and all thirty iterations kept the base actor/heads, task
  exploration and protected radar-axis residual row exactly fixed.
- Maximum logged initial-policy KL is only 3.75e-6. This records small
  distribution changes; it does not prove that increasing the learning rate
  would improve task success.
- Source snapshot, original/effective checkpoint hashes, best/latest hashes
  and training artifact references passed independent verification. Test
  source and checkpoint also remained unchanged throughout evaluation.
- Teacher, rule, scenario and red-agent files were not modified. HEUR was
  used only for independent paired performance comparison in this review.

## Decision

Retain iter 3 as this experiment's selected candidate, along with effective
initial and final weights, logs and source snapshots. Do not extend iter 29
or present this result as RL exceeding HEUR. The current method remains
unsuccessful on the stated final objective. The evidence does not show that
the environment is impossible: HEUR succeeds on 25 of 30 reserved seeds.

Before another long run, diagnose task-level long-horizon credit, the
expressiveness of aggregate conflict features, and the contribution of task
residuals versus allocation bids. The current one-step arbitration signal
does not by itself establish stable-window or mission success. No new
training is launched as part of this review.

Raw evidence: pilot_log.json, heldout_162000_162029.json and its provenance
sidecar; computed comparisons: audit_summary.json; independent source and
parameter checks: integrity_audit.json. Any checkpoint pruning is recorded
separately in artifact_retention.json; original provenance describes files
as captured during training, before that explicit pruning.

The attempted intermediate-checkpoint cleanup was rejected by automatic
approval review with `blocked by policy`. No checkpoint was deleted. The
retention record identifies the six useful checkpoints included in the
archive commit; the other 32 checkpoint copies remain locally and are
excluded from that commit.
