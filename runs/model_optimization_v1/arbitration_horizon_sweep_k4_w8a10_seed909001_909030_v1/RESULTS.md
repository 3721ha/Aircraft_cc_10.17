# Frozen arbitration horizon sweep

This is a read-only, same-snapshot counterfactual audit of the frozen K=4,
W=8, A=10 policy. It uses the same 30 seeds and reference replay as the
earlier arbitration audit. No training, teacher labels, rule changes, or
scenario changes were used.

The audit compares the learned RL winner with the reconstructed fixed-score
winner at counterfactual horizons 1, 4, and 8. The horizon only changes the
finite-horizon utility used to rank the same candidate branches; utilities are
never compared across horizons.

| Horizon | Informative events | RL better when winners differ | Fixed-score better | Mean RL minus fixed utility when winners differ |
|---:|---:|---:|---:|---:|
| 1 | 85 | 0/8 | 8/8 | -0.0438125 |
| 4 | 85 | 6/8 | 2/8 | 0.7761220 |
| 8 | 85 | 6/8 | 2/8 | 0.7682967 |

H=8 is the aligned training target for W=8: it spans one stable window while
keeping the task policy frozen. This result supports a controlled H=8 bid-only
training trial, followed by natural-exposure evaluation on fresh seeds. It
does not establish an episode-level success-rate improvement by itself.

Reproduction:

```powershell
Aircraft_mumt/.venv/Scripts/python.exe _diagnose_arbitration_event_regret.py `
  --reference-ablation runs/model_optimization_v1/frozen_arbitration_k4_w8a10_seed909001_909030_v1/arbitration_ablation.json `
  --output runs/model_optimization_v1/arbitration_horizon_sweep_k4_w8a10_seed909001_909030_v1 `
  --seed-start 909001 --episodes 30 --counterfactual-horizons 1 4 8
```
