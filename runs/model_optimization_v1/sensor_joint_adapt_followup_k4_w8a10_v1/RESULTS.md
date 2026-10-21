# Fixed update-8 independent development follow-up

The selected update-8 policy did not reproduce its validation improvement.
It failed both predeclared joint descriptive screens: against initial RL and
against HEUR. Do not promote this checkpoint or extend this joint adaptation
solely on the previous 12-seed validation results.

Configuration: K4/W8/A10, deterministic instantaneous small-beam hits,
`W8_SCALE=1.0`, 30 new matched scenario seeds 210000--210029, 90 episodes.
Source commit: `1d887e4`. Runtime: 307.907 seconds. Policies were frozen;
update 8 was fixed before this run. These are development follow-up seeds,
not the reserved final test.

| Fixed policy | Success | Mean raw return | Violations / step | Unfinished target-dwell hits / step |
| --- | ---: | ---: | ---: | ---: |
| Initial RL | 15/30 (50.00%) | -8602.37 | 7.9120 | 0.2115 |
| Update-8 RL (old validation best) | 8/30 (26.67%) | -9408.61 | 8.3481 | 0.2038 |
| HEUR | 28/30 (93.33%) | -2172.43 | 6.1164 | 1.1287 |

Violations / step means total violation events divided by total decision steps,
not a percentage. Hits count each platform and dwell separately and can exceed
one per decision step. All geometrically eligible beam opportunities hit; all
radar dwells were active. Lack of radar activation and stochastic Pd misses do
not explain the illumination gap in this configuration.

## Paired results

Update 8 gains success on two seeds (210013, 210020), loses it on nine
(210001, 210004, 210005, 210009, 210014, 210015, 210023, 210024, 210025), and
has 19 unchanged outcomes. Its net success difference is -7/30, or -23.33
percentage points. The exact paired two-sided success p value is 0.06543;
do not claim a decline significant at the 5% level. The descriptive success
bootstrap interval is [-43.33, -3.33] percentage points, but its exclusion of
zero must not be selectively used to override the exact test.

Update-8 minus initial mean return is -806.24, with paired percentile 95%
interval [-2734.12, +1031.25]. Pooled violations / step changes by +0.4362,
interval [-0.3619, +1.2688]. All three point estimates worsen, and no
generalization improvement is supported.

Against HEUR, update 8 has zero success gains and 20 losses. Success difference
is -66.67 percentage points (exact paired p=0.00000190735). Return difference
is -7236.18, paired interval [-8745.86, -5672.56]; pooled violations / step
difference is +2.2318, interval [+1.3066, +3.2283]. The RL goal remains unmet.
Intervals are descriptive and unadjusted for multiple comparisons.

## Diagnosis and limits

Update 8 has 3.65% fewer unfinished-target hits per step than initial RL;
HEUR achieves 5.54 times the update-8 rate. Missing-owner-track gate failures
are 9319 target-steps versus initial 8917 and HEUR 2702; unowned tracking-zone
target-steps are 6979 versus 5998 and 1075. Episode durations differ, so raw
counts are associations. They cannot identify a unique causal mechanism.
Lower stable-progress reset counts do not demonstrate better tracking when
successful verdicts also fall (88 versus initial 98 and HEUR 118 targets).

All 326 update-8 multi-candidate events use direct RL arbitration. Ten winners
differ from the fixed-score reference (initial: 307 events, ten differences).
The 13 conflict-pair proxies do not establish causal arbitration benefit.

Code inspection and isolated method calls establish two training-objective
mismatches, recorded in `integrity_audit.json`:

1. `_track_potential` uses `0.15 * max(0, 1 - d/40)` per owned target, despite
   documenting denominator 80. With MAR40, sigma4 and K_SIGMA3, d60 and d55
   both satisfy the shield's instantaneous MAR margin but have zero geometric
   potential. Positive values require d<40, already inside nominal MAR.
   The term is enabled in the actual saved training configuration. This does
   not prove it caused the observed regression; the shield is a recovery
   mechanism and does not guarantee safe states throughout an episode.
2. Alignment shaping uses `max(1, gear-abs(axis))`, while C04 uses
   `min(gear, 60-abs(axis))`. At gear45/axis30, a target 20 degrees off axis
   passes the actual 30-degree C04 sector gate but receives zero alignment
   potential under the shaping term's 15-degree width. The witness isolates
   the alignment term by setting stable progress to zero.

These findings concern training rewards. They are not errors in the fixed rule
definitions and do not justify changing C04, MAR, the teacher or the red model.
They suggest a focused next investigation of reward support in the feasible
sensor geometry and radar-axis drift, rather than another unmodified long PPO
run. Scan timing observations and multi-head credit remain separate hypotheses.
Any later correction must be tested as a new, versioned experiment with fresh
validation data. Initial RL remains the existing diagnostic reference, not a
newly selected winning model. Neither checkpoint currently beats HEUR.

The HEUR result demonstrates that this fixed simplified sensor/task setup can
achieve high success with an existing controller. It does not establish that
the current RL architecture can learn that behavior, nor that RL is impossible.

## Audit and retention

`audit_result.py` recomputes 32 integrity/statistics checks and the two isolated
training-potential witnesses. It passed against the completed run. Two separate
read-only reviews independently recomputed the episode summaries, exact paired
tests and all 10,000-resample bootstrap intervals. No task rerun was needed.

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe runs/model_optimization_v1/sensor_joint_adapt_followup_k4_w8a10_v1/audit_result.py
```

Keep the result, provenance sidecar, audit script/output and original input
snapshots. The two policy checkpoints and selected update-8 alias are archived
unchanged; they are provenance inputs, not new models. The original source run
and frozen weights remain intact. No model training, checkpoint repair,
algorithm change, rule/scenario/red-model change or teacher edit occurred in
this result review. Reserved final-test seeds 208000--208029 were not run.
The inspected follow-up seeds 210000--210029 are now consumed.
