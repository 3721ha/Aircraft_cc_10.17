# K4 Arbitration-Only Smoke Test (2026-10-20)

## Change

- The conflict residual used by task heads and the bid conflict residual are now separate parameters.
- `bid-only` PPO explicitly freezes every non-bid policy parameter, including the task residual and task action standard deviation. Only the bid head, target-conditioned bid head, bid conflict residual, and bid standard deviation can update.
- Legacy checkpoints with one combined 21-row conflict residual load by splitting the final bid row from the task rows.
- `_run_best_refine.py` now defaults to `bid-only`; `--mode joint` remains available for comparison runs.

## Verification

- Focused regression suite: 99 passed. It includes a test with pre-existing nonzero Adam momentum and an out-of-range task `log_std`; bid-only update leaves the task policy unchanged.
- K4 iter=24 input checkpoint SHA256: `386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797`.
- One-iteration, eight-episode smoke used `direct_rl_arbitration=true`, `bid_only_ppo=true`, conflict exposure `1.0`, and penalty scale `1.0`.
- Smoke rollout produced 49 counterfactual arbitration events and 95 active bid slots; informative fraction `0.979381`, bid gradient norm `1.89918602`, and bid parameter delta `0.00023663`.
- Smoke `pilot_log.json` SHA256: `b0048bef87b3695a8c7d3d0e05f063cf456ac11a6aea5b064a29ef5e9b1126ca`.

This smoke test verifies that RL arbitration receives an environment-derived learning signal and changes bid parameters while the task policy stays fixed. It is not a performance result: the run kept the initial checkpoint as best, and its single-seed validation did not beat HEUR.
