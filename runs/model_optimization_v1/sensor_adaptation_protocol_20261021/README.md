# Short joint RL adaptation: deterministic beam, K4/W8/A10

The preceding frozen sensor diagnosis did not close the RL/HEUR gap: RL
succeeded on 10/30 and HEUR on 25/30 under both radar profiles. The complete
negative result and independent audit are retained in
`../sensor_profile_diagnostic_k4_w8a10_v1/`.

This experiment tests whether the current RL policy can adapt its task
behavior to deterministic in-beam detection. It is a short adaptation trial,
not a final comparison, and does not assume that more training will succeed.

## Fixed inputs and scope

- Source: `runs/model_optimization_v1/residual_pg_k4_w8a10_v1/curr_k4_best.pt`.
- Required SHA256:
  `38815f375df492ee320d9a0d4a9105efaf3944a6694165b295f646c9917aea18`.
- K=4; W=8s; A=10s; `RADAR_HIT_MODEL=deterministic_beam`; full penalties.
- Teacher files, 21 rule families/13 conflict pairs, red dynamics, range,
  beam geometry/revisit timing, sigma and verdicts are unchanged.
- The source task-policy checkpoint initializes this run; the newer
  fixed-event bid diagnostic checkpoint is not silently substituted. This
  preserves the model used in the frozen sensor comparison as the baseline.
- No BC/DAgger, expert action labels, HEUR winner labels or forced request
  injection occur. HEUR runs only as a fixed independent validation comparator.

## What becomes trainable

The inherited source was residual-only: the actor backbone was frozen,
direct radar-axis PPO had coefficient zero, and a radar-axis reference loss
restricted movement. This run explicitly enables joint task PPO: flight
controls, radar-axis, role, gear, pointer, conflict residual and critic. The
candidate bid branch retains environment-copy counterfactual utility credit
and direct RL arbitration. The pre-existing optional geometry adapter stays
frozen; no new policy architecture is introduced.

Existing flight-control intervention masking remains enabled. An opt-in
`mask_intervened_radar_axis=False` lets the unchanged sampled radar-axis
action receive ordinary PPO return credit when the shield projects flight
controls. Training-exposure rows remain masked. The default setting is True,
so historical runs keep their original behavior. This change does not use
safe/HEUR radar labels or feed diagnostic truth into the actor.

Effective settings:

| Setting | Value |
| --- | --- |
| learning rate / PPO epochs | 1e-5 / 2, inherited |
| joint task PPO | bid-only=False; residual-only=False |
| radar-axis PPO / old axis reference | 1.0 / 0.0 |
| task-policy reference KL coefficient | 0.5 |
| task-policy reference | this explicit initial RL policy, frozen |
| counterfactual bid continuation | H=1, inherited default |
| training conflict exposure | 0 |
| geometry-adapter training | False |
| reward shaping / optimizer / clipping | existing implementation, unchanged |

The source's task-reference coefficient was 1.0; 0.5 allows limited task
adaptation while keeping an explicit RL trust reference. Shared global
gradient clipping remains a known limitation of this trial. Per-family
parameter deltas, active PPO fractions and post-update reference KL are
logged; a nonzero delta is not evidence of improved behavior.

## Run from the repository root

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe _train_sensor_adaptation.py --output runs/model_optimization_v1/sensor_joint_adapt_k4_w8a10_v1
```

Defaults: 12 updates × 8 complete training episodes = 96 episodes, with
deterministic validation initially and after every two updates. This is the
authorized short trial, not another 100-update long run. The runner refuses
existing output directories and a mismatched source checkpoint hash. It
creates source/weight snapshots and records complete effective configurations.
No final reserved test is launched by this command.

Optional read-only preflight, producing no experiment artifacts:

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe _train_sensor_adaptation.py --output runs/model_optimization_v1/sensor_joint_adapt_k4_w8a10_v1 --plan-only
```

Training seeds: **206000–206095**. Selection validation: **207000–207011**.
Reserved future test: **208000–208029**, not run or used for selection here.
The prior diagnostic seeds 205000–205029 and old test 162000–162029 are
explicitly rejected for training/selection. Smoke uses separate seeds
209900–209903 and 209990–209991. The four seed roles must remain distinct.
Repeated validation is model-selection evidence, not an untouched test.

Read `adaptation_log.json` and `adaptation.provenance.json` afterward. The
effective initial model is validated and included in best selection. Ranking
uses validation success count, mean raw task return and lower pooled
violations/step, in that order; exact ties retain the earlier candidate.
`best.pt` is the selected candidate, `checkpoints/update_XXXX.pt` are immutable
update weights, and `latest` is separately identified. The source SHA is
checked again after completion. Validation restores the training Torch RNG
and does not change training weights.

Review success, return and violations against both the initial RL and HEUR,
plus illumination, unowned/missing-track periods, request outcomes, natural
conflict frequency and bid learning signal. Only consider a longer run after
this review. Do not call a task-level improvement evidence of useful conflict
arbitration without a separate causal comparison.

## Smoke verification

**175 distinct related tests passed** in this implementation iteration:
7 gradient/masking checks, 60 existing MAPPO/residual/reference regressions,
39 adaptation-runner safeguards and 69 sensor/profile regressions. Initial
Windows system-TEMP fixture setup failures were rerun with workspace temporary
directories and passed. Detailed counts are in `validation.json`; the sensor
regression run also has a retained JUnit report. Independent checkpoint/source
verification of the real smoke passed 23 additional integrity checks.

Two genuine updates, two training episodes per update and two independent
validation episodes completed in 55.609 seconds. All source/artifact hashes
verified; task heads, radar-axis, critic and bid parameters changed; the two
anchors and geometry adapter were bitwise unchanged. Five informative natural
arbitration events produced environment-copy credit. The radar-axis PPO
active fraction was 1.0; flight fractions were 0.724 and 0.682.

| Validation | Success | Mean return | Violations/step | Selected best |
| --- | ---: | ---: | ---: | --- |
| initial | 1/2 | -2354.37 | 5.1998 | initially yes |
| update 1 | 1/2 | -6046.42 | 6.8099 | no |
| update 2 | 1/2 | -2271.45 | 5.0955 | yes |

This small check verifies training and selection wiring. Success did not
increase; the modest return difference is not evidence of generalization.
The real 96-episode trial starts from the fixed source, not this smoke's
selected weights. Exact smoke records remain in
`../sensor_joint_adapt_smoke_k4_w8a10_20261021/`.
