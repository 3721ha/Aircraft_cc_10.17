# K4 Event-Wise Counterfactual Credit Smoke

## Configuration

- Source policy: `curriculum_k4_w8a10/curr_k4_best.pt`, SHA256 `386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797`.
- Task: K=4, W8/A10; direct RL arbitration; bid-only PPO; conflict exposure 1.0.
- Training: one PPO update from seeds 100000-100007, learning rate `1e-5`, two epochs.
- Validation: paired seeds 101000-101009; no conflict exposure during evaluation.
- Run ID: `20261020T084345.030480Z_35080_8a09703b`.

## Signal Check

- 26 controllable direct-RL conflict events produced 52 candidate branches; all 26 had nonzero candidate utility contrasts. Four additional events were uncontrollable under the bounded bid action.
- Mean candidate utility spread was `0.157075`; mean regret of the selected winner against the best forced-winner branch was `0.063976`.
- The bid head had gradient norm `2.89526534` and parameter delta `0.00023408`.
- Unit tests verify that candidate-specific advantages favor the higher-utility winner and reverse when the replayed utility ordering reverses.

## Task Performance

- Source and post-update validation were identical: RL success `1/10`, mean return `-8307.5`, and `7.8546` violations/step; HEUR success `6/10`, mean return `-6216.1`, and `9.0991` violations/step.
- The source checkpoint remained selected. This smoke verifies environment-derived arbitration credit and a nonzero policy update, not an improvement over HEUR.

The detailed run log and provenance are preserved beside this report. The initial one-episode smoke used a seed with no controllable conflict and was not used as method evidence.
