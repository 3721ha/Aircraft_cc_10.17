# Predeclared next stage: diverse real-event arbitration learning

This is an RL arbitration training diagnostic aimed at resolving the failed
generalization check. It is not a paper ablation or a task-success acceptance
run. Do not promote its final bid-only model as the best mission policy.

Run from the repository root in PowerShell:

```powershell
& .\runs\model_optimization_v1\window_geometry_protocol_20261020\train_diverse_events.ps1
```

The same command with `-PreflightOnly` checks arguments, source checkpoint
hash and output safety without creating output or collecting trajectories.
The implementation was preflighted before handing the command to the user.

The fixed source is residual PPO iter 3 (SHA256 begins 38815f375df4).
The task remains K4/W8/A10, C04=80, axis limit=60 degrees, full penalties.
Training-only conflict exposure uses existing lawful onboard sensor tracks;
RL supplies bids and H8 continuation. No HEUR winner labels or truth-track
fixtures participate. Deployment and natural mission evaluation have no
exposure intervention. The geometry adapter remains zero/frozen in this
stage so a change in its task behavior cannot be mistaken for bid learning.

The data budget is fixed before execution: 64 training episodes, seeds
184000-184063, and 16 held-out diagnostic episodes, seeds 185000-185015.
Retain at most four events per episode, spaced at least 24 decisions apart,
without ranking events by utility or observed winners. Cap each split at 256
events. Learning requires informative events from at least 16 training and
8 held-out episodes. Insufficient coverage writes learning_audit.json with
an explicit failure status and stops before updating weights. Do not silently
lower those thresholds if collection is insufficient.

Training samples episodes uniformly, then an informative event uniformly
within that episode. Run exactly 48 updates, 32 events/update, two PPO epochs,
learning rate 1e-4, utility scale 0.1, seed 20261021. Evaluate before and after
with the same 4096 Monte Carlo samples/event. Do not choose an intermediate
checkpoint or change this budget using held-out results.

Review the final conditional objective using episode-balanced held-out
metrics: expected regret should fall by at least 5% and deterministic regret
should not increase. Source and frozen task tensors must remain intact.
This is a diagnostic gate for the next design decision, not statistical
evidence that RL exceeds HEUR. Both event-weighted and episode-weighted
metrics and every episode's metrics are retained.

If the gate passes, separately test task adaptation and then a jointly
trained candidate on natural paired validation, preserving the selected
checkpoint and hashes. Mission success, return and violations relative to
HEUR remain the final criteria. If the gate fails despite adequate episode
coverage, do not extend this run; revisit the bid objective and information
available to the arbiter. A failed gate is not proof that the task is
impossible. No future test result or success advantage is assumed.

After completion, review:
`runs/model_optimization_v1/window_diverse_event_learning_k4_w8a10_v1/learning_audit.json`.
The previous reserved task-test seeds are excluded. This user-run stage has
not been launched by the agent.
