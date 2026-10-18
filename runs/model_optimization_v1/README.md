# Model Optimization Record

## Current Candidate

The strongest task-policy starting point remains
`Aircraft_mumt/runs/joint_w8_a10_lr3e5_diag_loadfix/curr_k2_best.pt`.
Its earlier validation used K=2, W8/A10 and seeds 200-209 (RL 10/10,
HEUR 9/10). It is a relaxed-configuration starting point, not a final model.
The paired validation returns were -1034.9 (RL) and -1704.3 (HEUR), with
1.9627 and 3.6659 violations per step, respectively. Under strict W15/A5,
seeds 220-229, RL reached 3/10 versus HEUR 1/10, but its mean return was
-5634.5 versus -6949.2. Keep both outcomes in view; the strict result is not
evidence that the deployment goal has been met.

The source snapshot for this older candidate predates the provenance system.
Its checkpoint, training log, relaxed validation and strict paired evaluation
are retained together in `Aircraft_mumt/runs/joint_w8_a10_lr3e5_diag_loadfix/`,
but the exact training source/configuration cannot be reconstructed from that
run alone. New optimization runs must include the generated provenance bundle.

## Retained Comparison Runs

These runs contain substantive training, paired evaluation or conflict-probe
evidence and are retained as historical comparisons, including negative
results:

- `Aircraft_mumt/runs/fullppo_w8a10_k3/`: K=3 full-PPO training, four paired
  task-configuration evaluations and forced conflict probes.
- `Aircraft_mumt/runs/detect_bid_cf_long_12_7/` and
  `Aircraft_mumt/runs/detect_bid_cf_long_12_7_loadfix/`: separate K=2 runs
  with strict paired evaluations; the latter also evaluates best/final
  checkpoints and several bid-weight probes.
- `Aircraft_mumt/runs/bid_directional_k4_w5/`: K=4 comparison with paired
  evaluation and deterministic conflict probe.
- `Aircraft_mumt/runs/conflict_exposure_smoke_v6/`: despite its name, this
  contains a 10-seed paired BC/current-policy comparison and is not disposable
  smoke output.

Only exact duplicate run outputs and model checkpoints produced solely by
short smoke tests are eligible for cleanup. Their useful logs remain as local
development records. Real paired evaluations and their corresponding
checkpoints are never removed based on directory names or low scores.

## Event-Normalized Bid Ablation

Run bundle:
`runs/model_optimization_v1/entropy_norm_bidonly_lr1e4_w8a10_w5_h005_retry2/`

The PPO bid loss, bid entropy and counterfactual scale now give equal weight to
each arbitration event instead of weighting an event by the number of active
candidate slots. The conflict probe also has a separate seeded random stream,
so paired checkpoints see the same forced conflict opportunities.

Training used K=2, 10 iterations x 4 episodes, W8/A10, C04=80 km,
W8 scale 0.2, bid weight 5, hysteresis 0.05, bid-only PPO, conflict exposure
1.0, learning rate 1e-4 and 2 PPO epochs. The input was the candidate above.
The training provenance record, source snapshot, input checkpoint and every
iteration checkpoint are kept in the run bundle.

| Evaluation | RL | HEUR | Result |
| --- | ---: | ---: | --- |
| W8/A10 natural paired, seeds 210-219 | 8/10, return -1809.3, violations/step 2.9116 | 10/10, return -1489.6, violations/step 3.4666 | Fails success and return criteria; fewer violations |
| W15/A5 strict, seeds 200-209 | 2/10, return -6469.7, violations/step 5.0956 | 4/10, return -5121.1, violations/step 6.0283 | Fails success and return criteria |
| Forced W8/A10 conflict probe, seeds 300-309 | 10 multi-candidate events, 0 winner changes at bid weight 5 | n/a | No demonstrated arbitration improvement |

At bid weights 5, 20, 50, 100 and 200, each 10-seed probe produced zero winner
changes. A rerun at weights 5 and 200 confirmed this on the same deterministic
probe events. The learned bid differences generally reinforced the base-score
winner. Training rollouts did show counterfactual bid updates and winner
changes, but that behavior did not transfer to deterministic evaluation. This
checkpoint is retained as a negative-result ablation, not as a deployment
candidate.

The detailed results and their provenance sidecars are in the run bundle.
`natural_best_210_219.json`, `strict_w15a5_200_209.json`,
`deterministic_details_w5_300_309.json` and
`deterministic_details_w200_300_309.json` are the primary evaluation records.
The separate `runs/model_optimization_v1/` records provide the same-seed
baseline conflict probe at weights 5 and 200 and the strict W15/A5 baseline.

## Acceptance Rule

Compare RL and HEUR on the same seeds and identical task/arbitration settings.
Keep a candidate only when RL success rate is at least HEUR, mean return is
better, violations per step are no higher, and the conflict probe demonstrates
task-useful winner changes. Forced-probe results are diagnostic and are not a
substitute for natural paired evaluation.
