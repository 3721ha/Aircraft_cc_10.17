# Frozen sensor profile diagnosis — K4/W8/A10

This experiment separates the immediate effect of removing random radar misses
from any benefit of policy retraining. It does not train, select a checkpoint,
inject conflict requests, or claim that RL exceeds HEUR.

## Fixed model and single changed factor

- Frozen task-policy checkpoint:
  `runs/model_optimization_v1/residual_pg_k4_w8a10_v1/curr_k4_best.pt`
- SHA256: `38815f375df492ee320d9a0d4a9105efaf3944a6694165b295f646c9917aea18`.
- K=4, stable tracking window W=8 seconds, tracking age standard A=10 seconds.
- Profiles: `RADAR_HIT_MODEL=bernoulli` and `deterministic_beam`.
- Both RL and HEUR run both profiles on the same diagnostic seeds.
- Every other configuration field is identical and checked by the runner.
- Teacher files, 21 rule families, 13 conflict pairs, red behavior, sigma,
  instantaneous FOV, range attenuation, scan dwell/revisit and success
  conditions are unchanged.

The deterministic profile records a detection only when the actual current
instantaneous beam reaches a target within the attenuated distance limit. It
does not treat the whole reachable scan sector as simultaneously visible and
does not turn one detection into a completed target.

The default profile remains the original Bernoulli model. A legacy radar
trajectory and next random draw captured before editing are regression-tested.
Both profiles consume one random draw per eligible target-dwell, avoiding a
shared link-RNG shift caused solely by skipping draws. Different subsequent
actions can still change geometry and later RNG consumption; common seeds
are not an assertion of identical link histories after trajectories diverge.

RL loads its saved policy configuration, uses deterministic actor actions and
direct learned arbitration, and runs the same continuous shield as HEUR.
Training exposure is disabled; no HEUR labels or arbitration fallback enter
the RL path. Old zero-output geometry-adapter migration is recorded, not
trained or repaired. Loaded policy and both anchor tensors are hashed before
and after each episode and across all conditions.

## Run from the repository root in PowerShell

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe _diagnose_sensor_profile.py --k 4 --stable-window 8 --age-std-track 10 --seed-start 205000 --episodes 30 --output runs/model_optimization_v1/sensor_profile_diagnostic_k4_w8a10_v1
```

This is 30 seeds × 2 profiles × 2 controllers = 120 episodes. The default
checkpoint and its expected SHA256 are built into the runner; a mismatch
stops before generating an output directory. Existing output directories are
refused. Run the command once; do not edit source files while it runs.

Primary result:
`runs/model_optimization_v1/sensor_profile_diagnostic_k4_w8a10_v1/sensor_diagnostic.json`.
The same directory retains the exact source ZIP, copied input checkpoint,
full task and policy configurations, per-episode rows, hashes and completed/
failed lifecycle sidecar. An incomplete run retains its episode progress;
it is not a completed comparison and this version does not resume it.

Seeds 205000–205029 are diagnostic/development seeds. They must not later be
reported as an untouched final test. The smoke seed 204900 is separate. Old
test seeds 162000–162029 are not reused for this experiment. Training,
checkpoint-selection validation and any final test must receive separate
seed ranges if retraining proceeds.

## Interpret measurements

The report includes success rate (with Wilson intervals), mean return,
violations/step, judged targets, radar opportunities/hits/misses, owner-gate
failure target-steps, actual stable-progress resets, violation families,
13 conflict-pair observation proxies and natural multi-candidate events.

Radar counts use target-dwell units; completed targets are also reported
separately from unfinished targets. Violations/step means total violations
divided by total environment steps. The mean per-episode rate is retained
under a different name. Conflict-pair activity counts agent-steps with encoded
pressure >0.1, not verified arbitration events or proof of beneficial choices.
Completion normally clears stable progress and is not counted as a failure
reset. Reset causes are exclusive: outside zone, missing owner track, then
stale owner track. Missing tracks and age failures are descriptive; they are
not proof that a particular random miss caused the whole task failure.

Compare deterministic minus Bernoulli for each controller, then compare RL
minus HEUR within each profile. If both improve but the RL gap remains, the
experiment supports simplifying the sensing assumption but does not establish
an algorithmic improvement. Only decide on retraining after reviewing this
diagnosis. A subsequent retrained model needs new validation/test seeds and
conflict-arbitration evidence; old/new sensor results are different benchmarks.

## Implementation verification

- 136 distinct relevant checks passed: radar profile and legacy mechanics,
  diagnostic neutrality, task environment, rules, shield, MAPPO loading and
  evaluation-runner safeguards. A Windows system-TEMP fixture permission error
  was resolved by rerunning the affected check with workspace `--basetemp`.
- Full-episode smoke: seed 204900, four conditions, exact frozen weights,
  no source drift, deterministic in-beam misses = 0.

| Profile | Controller | Result | Return | Violations/step |
| --- | --- | --- | ---: | ---: |
| Bernoulli | RL | failure | -7543.52 | 8.0650 |
| Bernoulli | HEUR | failure | -9343.86 | 9.2487 |
| Deterministic beam | RL | success | -7043.98 | 6.7146 |
| Deterministic beam | HEUR | success | -1523.74 | 4.4635 |

This single seed validates execution and instrumentation, not a success-rate
estimate or a model-selection decision. Full smoke evidence is retained in
`../sensor_profile_smoke_k4_w8a10_20261021/`.
