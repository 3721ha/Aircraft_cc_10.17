# K=4 conflict-conditioned residual PPO: fixed next experiment

Prepared on 2026-10-20 after correcting arbitration credit and auditing the
previous bid-only protocol. This is an experiment plan, not a success claim.

## Why the next experiment changes the trainable scope

The old continuous-bid credit was independent of the sampled winner. It is
replaced by the actual winner's environment-copy utility minus the mean
candidate utility, shared by all sampled bids in that event. Candidate scores
are summed inside an event and events are averaged; utility scaling is fixed.
The implementation uses factorized PPO ratios, not exact joint PPO ratios.

The inherited candidate branch had zero input weights/bias and zero output
weights. An explicit training-start migration resets only its hidden linear
layer, leaving output weights zero. Initial policy outputs are identical.
Every new curriculum now archives its effective initial weights and hash
after all loading, exploration overrides, anchor refresh and migration.

The fixed real-event audit improved held-out stochastic expected regret
0.04440094 -> 0.03993602 (10.06%) and optimal-winner probability
55.10% -> 58.93%, on 19 events from three held-out episodes. Its deterministic
held-out winners did not change, and training deterministic regret slightly
worsened. This only establishes conditional one-step learning, not improved
deployment success. See the adjacent fixed_event_learning_audit_k4_20261021
reports for raw frozen cases, paired changes and checkpoint/source hashes.

Two failed natural-validation episodes (102006 and 102007) have no competing
candidates. With a frozen task policy, bids cannot affect either trajectory.
The old bid-only protocol therefore has a conditional upper bound of 80%
success on these ten validation seeds, below HEUR's 90%. Retaining bid-only
long training cannot meet the declared objective on that validation set.

## Controlled task adaptation

- Freeze the shared actor, base flight/role/gear/pointer heads and task log std.
- Train the existing 13-feature conflict-conditioned task residual, arbitration
  bid branches and critic. Protect radar-axis residual row 2 exactly, including
  against existing Adam momentum. Task actions can change without a bid event.
- Keep an initial-RL-policy KL penalty of 1. No HEUR winner labels are used.
- Clip residual, bid and critic gradients separately to 0.5.
- Use natural requests with exposure probability 0 and full violation penalty.
- Deployment remains actor -> continuous safety shield -> direct RL bids.
  The residual itself is part of the actor; HEUR only supplies paired metrics.

This differs from the previous failed full-joint plus anchor experiments:
the backbone, base heads, task exploration and radar-axis output cannot drift.
The aggregate features still have limited directional information. No claim
is made that the residual can recover every failure or cover all 13 conflicts.

## Fixed protocol

- K=4; W=8; A=10; C04 range=80; sector axis maximum=60 degrees.
- Start from curriculum_k4_w8a10/curr_k4_best.pt, best iter=24.
- Source SHA256: 386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797.
- 30 updates, 8 complete episodes/update, LR=1e-5, 2 PPO epochs/update.
- Training seeds 160000-160239. Candidate repair seed 20261021.
- Natural deterministic validation every 2 updates on 102000-102009.
- No early stopping; retain the initial candidate and use the existing
  success -> mean return -> fewer violations ordering to save best.
- These validation seeds have been examined previously and are not a fresh
  final test set. A separately reserved test (162000-162029) is evaluated
  only after selection. Do not tune this run from those test results.
- Keep task/rule/red/teacher files unchanged. Do not add the old forced-truth
  conflict probe to task performance evaluation.

## Acceptance and interpretation

Implementation acceptance: correct expected policy-gradient direction, restored
candidate trainability, effective-start/source hashes, exact protected weights,
separate gradient clipping, and a real natural-request update.

Model acceptance is separate: compare the selected deterministic RL checkpoint
with paired HEUR success, return and violations on natural episodes. RL has
not yet been shown to exceed HEUR after these repairs. If there is no model
improvement, report that outcome; do not relabel probe or stochastic-training
metrics as task success, select a more favorable validation set, or extend
the budget after reading test data.

## Run from the repository root in PowerShell

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe _run_best_refine.py `
  --k 4 --source-run runs/model_optimization_v1/curriculum_k4_w8a10 `
  --output runs/model_optimization_v1/residual_pg_k4_w8a10_v1 `
  --mode conflict-residual --iterations 30 --episodes-per-iter 8 `
  --learning-rate 0.00001 --conflict-exposure 0 --penalty-scale 1 `
  --policy-anchor-coef 1 --patience 0 --validation-every 2 `
  --training-seed-start 160000 `
  --validation-seeds 102000,102001,102002,102003,102004,102005,102006,102007,102008,102009 `
  --repair-dead-candidate-head --candidate-repair-seed 20261021
```

Append --preflight-only to check without starting training. This command has
passed preflight; the long run has not been started by the agent.

Expected result: runs/model_optimization_v1/residual_pg_k4_w8a10_v1/pilot_log.json.
