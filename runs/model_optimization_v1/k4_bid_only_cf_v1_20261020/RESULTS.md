# K4 Bid-Only Counterfactual Fine-Tuning

## Configuration

- Source: K4 W8/A10 checkpoint, iter 24, SHA256 `386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797`.
- Training: K=4, W8/A10, direct RL arbitration, bid-only PPO, conflict exposure 1.0, 10 completed iterations, 8 episodes per iteration, training seeds 100000-100079.
- Validation: seeds 101000-101009; natural conflict exposure disabled during evaluation.
- Code: commit `405e6b3`; provenance reports unchanged source during the run.

## Result

- The run stopped at iter 9 after five validations without improving the preserved best. Best remained the initial checkpoint (`iter=-1`); latest is iter 9, SHA256 `c23087e700272ba2b9d8b3ac9f024af44a7b71aa539af165fe410c787caf2ef5`.
- Best validation: RL 1/10 success, mean return -8307.5, 7.8546 violations/step; HEUR 6/10, -6216.1, 9.0991. Acceptance failed; only the violation-rate comparison favored RL.
- Training generated 544 exposed multi-candidate events and nonzero bid gradients/parameter updates. The bid policy was updated from environment-copy counterfactual utility; no HEUR winner labels were used.
- Natural paired evaluation on seeds 250-279 was identical for the source and iter-9 checkpoints: RL 13/30, mean return -7877.5, 7.8993 violations/step; HEUR 23/30, -4614.8, 8.1064. Both RL checkpoints recorded 32 natural conflict events and 6 winner changes versus fixed scores.
- Forced-conflict probe on seeds 310-319 produced 1016 events for each checkpoint. Winner changes versus fixed scores rose from 386 (37.99%) at the source checkpoint to 393 (38.68%) at iter 9. This is a small behavior change, not evidence that the selected winners had higher task utility.

## Conclusion

This run verifies that RL arbitration receives environment-derived learning signal, but it does not improve natural task performance or beat HEUR. Keep the source checkpoint as the selected model. The saved JSON logs, provenance, source snapshot, and checkpoints preserve the run for audit.
