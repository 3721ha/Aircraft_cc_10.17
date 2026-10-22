# RL Bid Arbitration Training Protocol v1

## Objective

Train only the RL bid arbitration head from environment-copy candidate-winner
utilities. The deployed controller remains an RL policy. HEUR is not used as a
label, target, reward, or conflict resolver. No files under `teacher/`, rule
predicates, conflict-pair inventory, or red-agent implementation may be changed.

## Fixed task profile

- Red aircraft: K=2.
- Scenario profile: `k2_lane_center_50`.
- Sensor model: Bernoulli radar hits.
- Numeric relaxations: `STABLE_WINDOW=8`, `AGE_STD_TRACK=10`.
- Existing arbitration setup: `C04_RANGE=80`, `ARB_BID_WEIGHT=0.35`,
  `ARB_HYSTERESIS=0.2`.
- Candidate-utility horizon: H=4, with a frozen deterministic RL continuation
  and independent branch shield.
- Training-only overlap exposure probability: 0.5. Exposed candidates are
  labeled from the collected action mask; natural conflicts remain in training.
- Development and final confirmation events must be natural conflicts only.

The task profile changes numeric scenario configuration only. Rule and red-agent
source files are included in the source snapshot and must remain unchanged.

## Data partitions

- Training event collection: episode seeds 910000-910199.
- Natural-conflict development bank: episode seeds 920000-920099.
- One-time paired episode confirmation: seeds 930000-930299, reserved and not
  used by this training command or for model selection.

Training uses source-balanced, episode-balanced event sampling. The collection
gate requires at least 20 informative training episodes, at least 10
informative episodes from each training source (natural and exposed), and at
least 10 informative natural development episodes. Insufficient coverage stops
before PPO updates and writes the event bank and coverage report.

## Fixed training schedule

- Bid-only PPO, learning rate 1e-5, 2 epochs per update.
- 48 fixed updates, 32 conflict events per update.
- Utility scale 0.1, entropy coefficient 0, policy anchor coefficient 0.
- No checkpoint selection on development events. The final checkpoint after
  update 48 is the sole candidate for the reserved episode confirmation.

The event bank records checkpoint and source hashes, effective scenario
configuration, utility definition, seed partitions, collection RNG seeds,
exposure probabilities, event sources, episode coverage, and Python/Torch/NumPy
runtime versions. Reuse is rejected if any of these fields differ. Collection
actor RNG uses independent recorded train and development streams; each
environment episode keeps its own scenario seed.

## Interpretation

An improvement in conditional event regret is evidence about arbitration only.
It does not establish an episode-success advantage. Final claims require the
fixed final checkpoint to be evaluated against HEUR on the same reserved
930000-930299 seeds, with identical task settings, and episode-level paired
uncertainty reported. The final block must be evaluated once after all training
and development decisions are frozen.
