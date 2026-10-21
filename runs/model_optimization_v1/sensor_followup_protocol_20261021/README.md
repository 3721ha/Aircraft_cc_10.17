# Fixed K4 sensor adaptation: independent development follow-up

The previous short joint PPO run selected update 8 on 12 validation seeds:
initial RL succeeded in 5/12 episodes, update 8 in 6/12, and HEUR in 11/12.
These validation results do not establish a reliable improvement. This follow-up
checks both frozen RL policies against HEUR on new development seeds before
deciding whether additional training is justified.

## Frozen comparison

- Task: K=4, stable tracking window W=8 seconds, tracking freshness threshold
  A=10 seconds, `RADAR_HIT_MODEL=deterministic_beam`, `W8_SCALE=1.0`.
- The deterministic profile counts a hit only within range and the actual
  instantaneous small radar beam. Scanning geometry and periods remain intact.
- Conditions: `initial_rl`, `best_rl` (update 8), and `heur`.
- Initial checkpoint SHA256:
  `ef86e5734d2a6466694371a0931386d7f1937355f279eed28f1560dcae27499b`.
- Update-8 checkpoint SHA256:
  `b30c53060f0374b204ca95fd6797bbf9a7ec5b1fb00d6d628e2afe5998f0e801`.
- Full task configuration SHA256:
  `256d20dadddbf0b400462a23a5f7c846156732131a7ef1b542f2235555cf2f3d`.
- Development seeds: **210000–210029**, 30 seeds x 3 conditions = 90 episodes.
- No training, conflict injection, checkpoint repair, or selection of another
  update. RL actions are deterministic; every observed RL conflict must use
  `rl_direct` arbitration. HEUR is evaluated separately.
- `teacher/`, rule definitions, red model and simulator implementation are
  unchanged by this follow-up.

The seed ranges are separate: diagnosis 205000–205029, training
206000–206095, selection validation 207000–207011, and still-unused reserved
final test 208000–208029. The old test 162000–162029 is already consumed.
Smoke seed 219900 verifies this runner only. The new 30 development seeds are
consumed once inspected; they must not later be described as unseen final test.
Shared seeds provide the same initial scenario. Stochastic trajectories can
diverge after the controllers take different actions.

## Run from the repository root

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe _evaluate_sensor_adaptation.py --output runs/model_optimization_v1/sensor_joint_adapt_followup_k4_w8a10_v1
```

Optional read-only preflight (no simulator/PyTorch imports or output directory):

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe _evaluate_sensor_adaptation.py --output runs/model_optimization_v1/sensor_joint_adapt_followup_k4_w8a10_v1 --plan-only
```

The output directory must be new. The runner checks pinned source logs and
checkpoints, the original source archive and member manifest, unchanged current
source, the full task configuration, saved policy configurations, and effective
loaded tensors. Inputs and current sources are archived; hashes are checked
throughout evaluation. Interruptions retain incomplete episode progress and a
failed lifecycle sidecar; a fresh output directory is required to rerun.

Results: `followup_result.json`; audit sidecar: `followup.provenance.json`.

## Interpretation fixed before the run

Report success counts/rates with Wilson intervals, mean raw environment return,
total violations / total steps, and mean per-episode violations / steps. Keep
per-seed return and violation-rate differences, success gains/losses, radar dwell
counts, verdict failure causes, rule-family totals and all 13 conflict-pair
proxies. Proxies alone are not evidence of causal conflict-resolution benefit.

For best-minus-initial, best-minus-HEUR and initial-minus-HEUR, report exact
two-sided paired success p values and paired percentile bootstrap 95% intervals
(10,000 resamples, RNG seed 20261027). Resample whole matched seeds and recompute
pooled violation ratios per resample. These are descriptive comparisons without
a correction for multiple comparisons.

Two joint descriptive screens are recorded independently: update 8 must have
at least the baseline success count, strictly higher mean return, and no higher
pooled violation rate, first against initial RL and then against HEUR. Passing
the initial-RL screen would support further investigation of adaptation, but
does not establish superiority to HEUR. Failure against HEUR leaves the project
goal unmet. Uncertain or regressing results call for a focused diagnosis before
another long run. No outcome automatically launches training or authorizes a
claim of final acceptance or arbitration benefit. The reserved final test stays
untouched for the eventual fixed candidate and protocol.

## Implementation validation

109 distinct tests passed: 61 existing runner regression cases and 48 new
follow-up cases. The new tests cover successful and failed lifecycle records,
input/source changes, reserved seeds, wrong effective weights/configurations,
forbidden fixed-score fallback, stochastic misses and accidental training calls.

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe -m pytest Aircraft_mumt/tests/test_sensor_profile_runner.py Aircraft_mumt/tests/test_sensor_adaptation_runner.py -q --basetemp runs/pytest_sensor_followup_regression_20261021
.\Aircraft_mumt\.venv\Scripts\python.exe -m pytest Aircraft_mumt/tests/test_sensor_adaptation_followup.py -q --basetemp .pytest_sensor_followup_review_20261021_v2
.\Aircraft_mumt\.venv\Scripts\python.exe -m pytest Aircraft_mumt/tests/test_sensor_adaptation_followup.py -q -k runtime_source_or_input --basetemp .pytest_sensor_followup_review_20261021_v3
```

The second command passed 46 cases before the final two drift cases were added;
the third command passed those two cases with 46 deselected. A fresh run of the
complete follow-up test file now collects all 48 cases. Use a new basetemp path
when rerunning to preserve existing local test artifacts.

One real three-condition smoke run at seed 219900 completed in 20.985 seconds.
Initial RL succeeded (return -1249.82), update 8 failed (return -15515.50), and
HEUR succeeded (return -1511.70). This single seed checks implementation and
does not establish aggregate model performance; it supplies no positive evidence
that the adaptation generalizes. Proceed with the fixed 30-seed comparison.

Smoke artifacts are retained in
`runs/model_optimization_v1/sensor_joint_adapt_followup_smoke_20261021/`.
`validation.json` records 30 passing checks, including all input/source hashes,
effective policy tensors/configurations, exact episode counts, recomputed
statistics and untouched final test. The official 30-seed run has not started.
