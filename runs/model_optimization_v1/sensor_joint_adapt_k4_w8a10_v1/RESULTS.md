# Short joint adaptation: K4, W8/A10

Source commit: `5d7713b`. Sensor profile: `deterministic_beam`.
Source checkpoint: `residual_pg_k4_w8a10_v1/curr_k4_best.pt`.
Selected checkpoint: `best.pt`, copied from `checkpoints/update_0008.pt`.
Selected SHA-256: `b30c53060f0374b204ca95fd6797bbf9a7ec5b1fb00d6d628e2afe5998f0e801`.

## Validation results

All rows use the same 12 validation seeds, 207000--207011. These seeds were
repeatedly used for model selection; this is **not a final test**. Metrics were
recalculated from episode records in `adaptation_log.json`. Violations per step
means total violations divided by total environment decision steps; it is not
a percentage.

| Controller / checkpoint | Successes | Success rate | Mean raw return | Violations / step | Unfinished-target illumination / step |
| --- | ---: | ---: | ---: | ---: | ---: |
| RL initial, update 0 | 5/12 | 41.67% | -9187.18 | 8.5491 | 0.2096 |
| RL selected best, update 8 | 6/12 | 50.00% | -7210.00 | 8.0375 | 0.2613 |
| RL latest, update 12 | 6/12 | 50.00% | -9707.46 | 8.5188 | 0.2313 |
| HEUR comparator | 11/12 | 91.67% | -2653.81 | 6.7868 | 1.0312 |

Illumination counts target-dwell hits across all platforms. It can exceed one
per environment step. Under this profile every valid beam opportunity is a hit.

The selection rule is successes, then mean raw return, then lower violations
per step, with strict improvement required. Update 8 is the correct selected
checkpoint. Validation successes at updates 0/2/4/6/8/10/12 were 5/5/5/4/6/4/6.

## Interpretation

Relative to the initial model, the selected model gains one net successful
episode: three seeds improve (207001, 207006, 207008), while two regress
(207000, 207009). Mean return improves by 1977.18, and violations per step fall
by 5.98%. Nine of twelve episode returns improve. This small, repeatedly
selected validation sample does not establish a statistically reliable task
gain or generalization. The selected success rate has a descriptive 95% Wilson
interval of 25.38%--74.62%; selection effects are not accounted for by that
interval. The success-rate gap to HEUR remains 41.67 percentage points.

The selected model increases unfinished-target illumination by 24.67%, but
HEUR still obtains about 3.95 times as many hits per step. Missing-owner-track
failure counts fall from 4637 to 3925 target-steps, and unowned tracking-zone
counts fall from 2248 to 1991; HEUR records 1373 and 302 respectively. These raw
counts have different episode lengths and are diagnostic, not causal effects.
Stable-progress resets increase from 1 to 7, so freshness does not improve
uniformly.

At update 8, pressure proxies for all 13 rule-conflict pairs are observed. All 208 natural
multi-candidate events use direct RL arbitration; four winners differ from the
fixed-score reference. Neither event coverage nor winner differences establish
an arbitration benefit without a controlled comparison.

Training completes 12 updates, 96 episodes and 58205 environment steps, with
226 natural multi-candidate events and 188 informative counterfactual events.
Task heads, radar axis, bid parameters and critic have nonzero parameter
changes, independently checked against every checkpoint. The geometry adapter
and both reference models remain bitwise frozen as intended. There is
no sustained validation improvement, and updates after the selected checkpoint
lose return and coverage quality.

## Next decision

Preserve update 8; do not directly continue from `latest` or launch a longer
run on the strength of these results. Check whether the task gain persists on
fresh independent diagnostic validation seeds before expanding training.
Further work should diagnose available sensor observations and credit for
coverage, pointing, ownership and sustained observation, with controlled
comparisons for the RL arbitration contribution.

Reserved final-test seeds 208000--208029 have not been run and remain reserved.
No source code, task settings, red-force model, rule library or `teacher/`
contents were changed during this result audit.
