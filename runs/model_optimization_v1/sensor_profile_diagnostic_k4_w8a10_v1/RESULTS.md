# Frozen sensor-profile diagnostic: K4 / W8 / A10

The deterministic small-beam profile did **not improve the frozen RL policy's success rate**: both profiles achieved 10/30, while HEUR achieved 25/30 in both. No training was performed. This negative result does not establish that adaptation training cannot help or that K4 is physically infeasible.

| Sensor profile | Controller | Successes | Mean return | Violations / environment step |
| --- | --- | ---: | ---: | ---: |
| Bernoulli | RL | 10/30 (33.3%) | -7810.4843 | 8.006122 |
| Bernoulli | HEUR | 25/30 (83.3%) | -4812.2747 | 8.361945 |
| Deterministic beam | RL | 10/30 (33.3%) | -8358.0148 | 8.125602 |
| Deterministic beam | HEUR | 25/30 (83.3%) | -4232.0649 | 7.932783 |

Violations / step is the pooled ratio `sum(violations) / sum(environment steps)`, not the mean of episode ratios. The frozen RL success-rate gap to HEUR is 50 percentage points in each profile. For RL, five seeds switched from failure to success and five switched from success to failure. For HEUR, two switched in each direction.

## Interpretation and limits

- The only task-configuration difference was `RADAR_HIT_MODEL`. Deterministic detection retained effective range, actual instantaneous beam geometry, scanning/revisit timing, ownership, freshness and stable-window verdicts. It did not grant a detection for every target in the whole reachable sector or equate one hit with task completion.
- For unfinished targets in the deterministic profile, geometrically eligible target-dwell opportunities per environment step were **0.227774 for RL versus 0.908499 for HEUR**. Both converted every opportunity to a hit. These counts include separate platforms and dwell substeps; they describe illumination and are not unique-target coverage probabilities or a causal attribution of failure.
- Natural multi-candidate RL arbitration events increased from **37 to 193**; all were recorded as `rl_direct`, with forced conflict exposure disabled. All 13 conflict-pair proxies were observed for RL in each profile. More events and proxy coverage did not demonstrate useful arbitration or better task performance.
- The closed-loop profiles share initial seeds and a separately seeded red process. One random draw per geometrically eligible target was retained in both profiles, but changed actions and eligibility can still change subsequent shared radar/link RNG alignment. This is not a perfectly isolated matched-randomness causal experiment.
- HEUR's 25/30 success demonstrates feasible policies under both profiles. The result does not prove the sensor has no bottlenecks, nor does it show that deterministic sensing is sufficient for RL to surpass HEUR.

The next experiment should measure short adaptation training under the deterministic profile, with independent training and validation seeds, before any long training or final held-out comparison. This frozen diagnostic is useful evidence against treating random missed detections as the sole explanation for the current gap.

## Provenance and audit

- Completed **120 episodes**: two profiles × two controllers × 30 seeds.
- Diagnostic seeds **205000–205029 are now consumed**. Do not reuse them for new model selection or a final held-out test.
- Source commit: `c2640a991ae461b9f91d3435741d9bc092201209`; branch: `experiments/bid-entropy-normalized-20261019`.
- Frozen checkpoint: `runs/model_optimization_v1/residual_pg_k4_w8a10_v1/curr_k4_best.pt`.
- Checkpoint SHA256: `38815f375df492ee320d9a0d4a9105efaf3944a6694165b295f646c9917aea18`. An identical input copy is retained at `provenance/frozen_policy.pt`.
- `diagnostic.provenance.json` records source, task/algorithm configuration, runtime parameters, dependencies and artifacts. Its completed source check reports no drift. All **37 archived source members**, the source ZIP and all input/output artifact hashes and sizes were independently verified. At audit time, working sources also matched the snapshot.
- `integrity_audit.json` contains **26 passing checks**, independently recomputed episode summaries and paired differences, and artifact hashes. All 60 RL episode records report unchanged weights and the same loaded tensor fingerprint. The standard-library audit verifies those records and checkpoint bytes; it does not independently deserialize runtime tensors.

The original diagnostic, provenance, source snapshot and checkpoint copy are retained. No source or teacher files were modified by this archival audit.
