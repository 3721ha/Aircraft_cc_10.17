# K4 Event-Wise Counterfactual Training Audit

## Run and provenance

- Run ID: `20261020T102001.444436Z_22824_c50494bc`; code commit `4fee7a0` on `experiments/bid-entropy-normalized-20261019`.
- Source: `curriculum_k4_w8a10/curr_k4_best.pt`, SHA256 `386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797`.
- K=4, stable window 8 s, track age threshold 10 s; C04 range 80 km, sector-axis limit 60 degrees, full violation penalties.
- Direct RL arbitration, bid-only PPO, conflict exposure 1.0, learning rate `1e-5`, two epochs, eight episodes per iteration. Recorded exploration override `MUMT_BID_LOG_STD=-0.7` was applied at startup; the source checkpoint was approximately `-0.6999`.
- Training completed 10 iterations / 80 episodes (seeds 140000-140079) in 1006.219 s. It stopped at iter 9 after five validations without improving the preserved best.
- Source integrity passed with no changed files; recorded source hashes also match the files at audit time. The run used the requested initial checkpoint.

## Natural paired validation

Seeds 102000-102009; no forced conflict exposure during validation.

| Policy | Success | Mean return | Violations/step |
| --- | ---: | ---: | ---: |
| Initial RL | 2/10 (20%) | -9041.4 | 8.7249 |
| Selected RL (initial, iter -1) | 2/10 (20%) | -9041.4 | 8.7249 |
| Latest RL (iter 9) | 2/10 (20%) | -9041.4 | 8.7249 |
| HEUR | 9/10 (90%) | -3436.5 | 7.2592 |

All five validations at iterations 1, 3, 5, 7, and 9 match the baseline's per-seed recorded results exactly, not merely the rounded aggregate means. Natural conflict counts also remain at 18 events and seven winner changes versus the fixed-score diagnostic. This does not assert identity of every unlogged action or trajectory.

The preceding smoke used validation seeds 101000-101009 and reported RL 1/10 versus HEUR 6/10. Its task configuration hash and recorded source-file hashes match this run. Those percentages belong to different seed sets and are not a learning comparison.

## Training signal and checkpoint checks

- 448 multi-candidate events, including 389 exposed events (86.8%) and 59 unexposed events.
- 413 completed counterfactual events, 412 informative events, 827 candidate branches, and 16 events marked uncontrollable. The complete event totals need not sum to 448: non-direct events are skipped by the counterfactual helper.
- Weighted mean utility spread `0.190618`; weighted mean actual regret `0.088277`, with no consistent downward trend across iterations.
- Nonzero bid gradient norms in every iteration (`0.748466` to `3.754179`); per-update bid parameter deltas `0.00005228` to `0.00023663`.
- Latest versus initial selected model changes only bid-related parameters; their combined L2 difference is `0.000710067`. Task-policy parameters remain identical.
- Training success is 45/80 (56.25%), measured with stochastic policies and forced conflict exposure. It is not the natural deterministic deployment success rate.

## Structural problems found in this audit

1. **Utility credit does not match the sampled continuous bid action.** The helper assigns candidate `i` the advantage `U(force i wins) - mean(U)`, then PPO applies that value to the likelihood of its originally sampled continuous bid. For a fixed event, the winner replays overwrite those bids; the candidate advantage therefore does not depend on its own sampled bid. Positive utility credit can reinforce a sampled low bid. For a Normal latent bid at the on-policy parameters, the score-function direction is proportional to `A_i * (z_i - mu_i) / sigma^2`, whose expectation is zero when `A_i` is fixed. Nonzero finite-batch gradients, clipping, and parameter updates do not demonstrate a useful arbitration preference. Small autograd examples confirm both the wrong-way update for a low sample and cancellation for symmetric samples.
2. **The candidate feature branch is dead in the inherited checkpoint.** The input weight, input bias, and output weight of `bid_candidate_head` are all zero in the source, initial selected, and latest checkpoints. With `Linear -> tanh -> Linear`, those tensors receive zero gradients; only the output bias can change. A nonzero-output-loss gradient check confirms this. The constructor's intended initialization (random first layer, zero output layer) differs from the inherited checkpoint; ordinary loading preserves the dead state. Other bid branches still update, so this does not mean the entire bid policy is frozen.

The prior smoke established executable replay plumbing and nonzero parameter updates, but its tests did not establish the correct expected utility-gradient direction or candidate-feature trainability. Describing those checks as sufficient method validation was premature.

## Decision and next step

This run fails the success, return, and violation comparisons to HEUR. Preserve the initial selected policy; do not scale up the same loss for another long run. This result does not establish task impossibility: HEUR succeeds on 9/10 of these seeds, while the audited RL implementation has identifiable training defects.

Before the next long run, repair the inherited candidate-branch initialization and align the policy-gradient advantage with the realized sampled winner (or use an explicit categorical winner objective). First verify the expected gradient favors the higher-utility contender and that regret decreases on the same frozen conflict events. Separately test whether the one-step utility predicts later stable tracking and task success. All candidate utility feedback can continue to come from environment copies without HEUR winner labels.

No training implementation, task, rule, red-agent, or teacher files were changed during this audit. Detailed machine-readable checks are in `audit_summary.json`; the original log, source snapshot, provenance, initial selected checkpoint, and final iteration checkpoint are retained beside this report. Nine unselected intermediate checkpoints (iterations 0-8, 14,524,551 bytes) were pruned after checking their recorded hashes. Their iteration metrics and hashes remain in the original log and audit record; the old provenance entries document the original run rather than claiming those pruned files remain on disk.
