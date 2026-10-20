# K=4 frozen real-event arbitration learning audit

## Scope and result

The repaired bid PPO improves the sampled bidding distribution's expected
one-step utility on this small held-out event bank. It does **not** establish
improved deterministic deployment decisions or episode success. No HEUR labels
were used. All utilities were collected by forcing each feasible candidate to
win from the same real simulator snapshot.

| Split | Events | Expected regret before | Expected regret after | Optimal winner probability before | After | Deterministic regret before | After |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 40 | 0.136360 | 0.121204 | 52.0343% | 55.9210% | 0.065810 | 0.067202 |
| Held out | 19 | 0.044401 | 0.039936 | 55.0974% | 58.9317% | 0.000000 | 0.000000 |

Expected regret decreases by 11.11% on training events and 10.06% on held-out
events. Held-out expected regret improves in 17 events and worsens in 2;
deterministic winners change in zero held-out events. Training deterministic
regret becomes slightly worse. The held-out bank's deterministic baseline
already selects an optimal one-step candidate in every case.

The event counts do not represent independent episodes: training retains 38
events from seed 100001 and 2 from seed 100002; held out retains 2 from 100004,
7 from 100005, and 10 from 100007. Seed 100000/100006 has no controllable event.
Seeds 100002/100003 generated 3 events each, but the predeclared 40-event cap
retains only the first 2 from 100002. No confidence or task success claim is
made from these clustered diagnostic events.

## Fixed protocol

- Environment: K=4, W8/A10, sector axis maximum 60 degrees, C04 range 80,
  bid weight 0.35, hysteresis 0.2, original rules/scenario/red/teacher files.
- Source checkpoint:
  `runs/model_optimization_v1/curriculum_k4_w8a10/curr_k4_best.pt`
- Source SHA256:
  `386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797`
- Real episode collection seeds: training 100000-100003, held out
  100004-100007. Collection uses the existing training-only request exposure.
- Fixed budget: 24 PPO updates, 32 resampled informative training events per
  update, learning rate 1e-4, 2 epochs, utility scale 0.1, entropy disabled.
- Update credit is the sampled winner's utility minus the mean candidate
  utility, assigned to the joint sampled bid action for that event.
- 4096 paired Monte Carlo samples per event before/after, identical noise
  seed 9017. Held-out results are read only after the fixed update budget.
- Candidate branch repair only initializes its dead hidden linear weights;
  the zero output weight preserves initial policy outputs exactly.
- Only bid tensors change during PPO. Hidden candidate weight L2 change is
  0.257562 and output candidate weight L2 change is 0.022987.
- Source snapshot comparison reports zero changed source files during run.

## Artifacts

- `event_bank.json`: frozen real local observations, unchanged task actions,
  event candidate slots and environment-copy utilities.
- `learning_audit.json`: source/config hashes, repair metadata, all update
  statistics and complete before/after event results.
- `paired_deltas.json`: event identity, paired regret deltas, deterministic
  decisions and parameter change verification.
- `initial_repaired.pt`, `final.pt`: diagnostic initial/final policy weights.
  The final checkpoint is a conditional-event diagnostic, not a selected
  deployment model.

## Reproduction

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe _audit_arbitration_learning.py `
  --checkpoint runs/model_optimization_v1/curriculum_k4_w8a10/curr_k4_best.pt `
  --expected-sha256 386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797 `
  --output runs/model_optimization_v1/fixed_event_learning_audit_k4_reproduce `
  --updates 24 --events-per-update 32 --mc-samples 4096
```

The script refuses nonempty output paths and verifies checkpoint hashes.
Use `--event-bank` with this run's `event_bank.json` to repeat bid learning
on exactly these cases; task config, utility definition and seed splits must
match the frozen collection record.
