# Sensor Shaping 2x2 Screening

## Protocol

- Task: K=4, W=8, A=10, deterministic beam, W8 scale=1.0.
- Arms: control (geometry 0.5, alignment legacy), geometry-only (1.0, legacy), alignment-only (0.5, C04 usable sector), both (1.0, C04 usable sector).
- All arms started from `sensor_joint_adapt_k4_w8a10_v1/initial.pt`, SHA-256 `ef86e5734d2a6466694371a0931386d7f1937355f279eed28f1560dcae27499b`.
- Each arm used 4 updates x 8 episodes, training seeds 260000-260031, validation seeds 261000-261009, optimizer seed 20261021, learning rate 1e-5, and 2 PPO epochs.
- The independent comparison used fixed update 4 for every arm and seeds 262100-262109. Reserved test seeds 208000-208029 were not run.
- Training episodes were collected by the RL policy. HEUR was used only as an evaluation comparator. RL conflict arbitration remained `rl_direct`.

## Independent Evaluation

| Policy | Success | Mean return | Violations/step | RL-only / HEUR-only successes |
|---|---:|---:|---:|---:|
| HEUR | 6/10 | -4734.66 | 8.1316 | - |
| Initial RL | 3/10 | -8636.80 | 8.0472 | 0 / 3 |
| Control update 4 | 4/10 | -7282.70 | 7.6754 | 2 / 4 |
| Geometry-only update 4 | 3/10 | -7872.28 | 8.3699 | 1 / 4 |
| Alignment-only update 4 | 3/10 | -8147.81 | 7.6225 | 1 / 4 |
| Both update 4 | 1/10 | -8808.42 | 7.9050 | 0 / 5 |

All RL arms remained below HEUR in success rate and mean return. Control had the best RL success rate; alignment-only had the lowest RL violations/step. Neither result establishes an advantage over HEUR. The combined arm had the weakest success rate.

RL arbitration was active during evaluation. The counts of multi-candidate events and winners differing from the fixed-score choice were: initial RL 86/5, control 108/7, geometry-only 138/8, alignment-only 87/3, both 82/2. A changed winner is evidence that RL made an arbitration decision; it does not by itself show that the decision improved task success.

## Validation And Selection

All arms began validation at 7/10. At update 2, control, geometry-only, alignment-only, and both reached 3/10, 4/10, 3/10, and 4/10. At update 4 they reached 7/10, 6/10, 6/10, and 3/10. The strict success-then-return selection rule retained update 0 for all four arms. Therefore the independent update 4 table is a pre-specified fixed-budget training comparison, not the selected checkpoint result.

## Interpretation

This screening does not support extending the combined shaping configuration to long training. Neither single factor improved success over the control on these seeds, and both factors together degraded it. Each arm had one optimizer seed and only 10 independent evaluation episodes; the estimates are noisy and do not establish population-level causal effects. The selected model for every arm remains the unchanged initial RL checkpoint.

The result files preserve the full suite manifest, per-episode independent evaluation, and per-arm training and provenance logs. Checkpoints and source archives remain in the local run directory and are not included in this commit.
