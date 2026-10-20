# Bid-only validation ceiling and rule-conflict scope

This is a read-only audit of existing logs, archived sources and checkpoint
tensors. No experiment or training was run for this audit. Existing results
were not modified. The machine-readable evidence is
`structural_scope_audit.json` in this directory.

## Result: this fixed bid-only protocol cannot reach HEUR

The previous run uses frozen task heads and direct RL responsibility bids.
Its ten natural-validation episodes contain 18 multi-candidate events, so
there are some genuine opportunities for bids to change allocation. However,
two failed episodes have no multi-candidate event at all. With the task policy
frozen, no bid update can change those two trajectories under the conditions
below. The maximum possible success rate is therefore 8/10 = **80%**, below
the recorded HEUR rate of 9/10 = **90%**.

This is a limitation of the chosen checkpoint, intervention surface and fixed
validation protocol. It does not show that the task itself is impossible, or
that a method allowed to learn task actions has the same ceiling. Changing
validation seeds would not resolve this methodological limitation.

| Seed | Starting RL terminal | Natural multi-candidate events | Winner differs from fixed score | HEUR terminal |
|---:|---|---:|---:|---|
| 102000 | failure | 3 | 0 | success |
| 102001 | failure | 1 | 0 | success |
| 102002 | success | 0 | 0 | success |
| 102003 | failure | 1 | 1 | success |
| 102004 | failure | 6 | 0 | success |
| 102005 | failure | 1 | 0 | success |
| 102006 | failure | 0 | 0 | success |
| 102007 | failure | 0 | 0 | success |
| 102008 | failure | 1 | 1 | failure |
| 102009 | success | 5 | 5 | success |

The baseline and all five later validation checkpoints have exactly the
same per-seed metrics. Recorded aggregate RL success remains 20%, return
-9041.4 and violations/step 8.7249; HEUR remains 90%, -3436.5 and 7.2592.

### Conditions and argument

1. Use the same simulator and task parameters, reset seeds and deterministic
   execution path. Do not change the scene, rules, red model or shield.
2. Keep all task policy parameters fixed. Only `bid_head`,
   `bid_candidate_head`, `bid_conflict_residual` and `bid_log_std` may change.
3. In the archived `detect_env.py` direct mode, C04 determines the feasible
   candidate pool and `max(pool, key=(bid, -agent_id))` selects its winner.
   A pool with at most one candidate has the same winner for every bid.
4. Given identical state and task actions, the same random stream produces
   the same next state when no multi-candidate event occurs. The next local
   observation and deterministic task action are therefore identical. By
   induction, a complete reference episode with zero multi-candidate events
   remains unchanged under any bid-only update.
5. Seeds 102006 and 102007 stay failures. Even if every other episode could
   become a success, at most eight successes are possible.

Checkpoint comparison confirms frozen task parameters in the old run and
both new smoke runs. The only non-bid shape migration is the legacy
`conflict_residual.weight`: 21 x 13 becomes 20 x 13, with the first twenty
rows exactly preserved and the final bid row moved to its separate branch.

## New smoke runs have no natural arbitration opportunity

| Smoke | Validation seeds | Natural multi-candidate events | Starting/final RL success |
|---|---|---:|---:|
| `eventwise_pg_smoke_k4_20261021` | 103000, 103001 | 0 | 0/2 |
| `eventwise_pg_conflict_smoke_k4_20261021` | 103002 | 0 | 1/1 |

All per-seed results match their respective starting checkpoint. Under the
bid-only conditions this is required by the structure of the environment.
Training-time exposed conflicts do not imply that deterministic natural
validation contains a conflict. Unchanged task metrics here establish
neither successful arbitration learning nor failed arbitration learning.
The one-seed RL-versus-HEUR return advantage already exists at initialization
and must not be presented as a training improvement.

## Twenty-one rules, thirteen pairs, narrower bid control

The archived authoritative definitions contain **21 rule families** in
`RULE_LAYER` and **13 conflict pairs** in `DETECT_CONFLICT_PAIRS`. The file
header saying eighteen predicates is an outdated comment; no rule change
was made or requested.

The encoder supplies thirteen local-observation soft conflict-regime
proxies. They are neither the exact referee predicates nor a feasibility
test proving that both obligations cannot be satisfied together. The legacy
evaluation counts a proxy as active above 0.1; this does not demonstrate
that the corresponding rule conflict was encountered or resolved.

The bid branch chooses the responsibility owner for an already contested
target among existing C04-feasible candidates. That decision can affect
later obligations and task utility, but it does not directly arbitrate all
thirteen rule conflicts. Pair-specific evidence would require recording
the actual obligations, feasible resolutions, chosen action, priority
violations and downstream effects. Current bid results must not be described
as proof that all thirteen pairs are solved.

The frozen real-event learning audit in this directory avoids fabricated
truth tracks. It measures conditional, one-step responsibility arbitration,
and its randomized expected-regret improvement does not imply improved
deterministic deployment success or full conflict-pair coverage.

## Legacy truth-fixture probe is not natural task evaluation

`_eval_conflict_paired.py:_install_probe_fixture` reads true current red
positions, temporarily inserts confirmed tracks into two local files, and
forces overlapping responsibility requests. It then calls `env.step` and
restores only selected fields in `_restore_probe_fixture`.

The returned reward and rule report were calculated before restoration.
The restoration does not restore all changed state: task role/gear and
kinematics, terminal state, and parts of sensing/history/random state are
not included in that restoration. Its docstring claim that the fixture
cannot alter the task trajectory or rewards is therefore not supported by
the implementation. Restoring some arbitration fields does not undo the
intervened step.

Consequently, success, return and violation metrics from the legacy forced
probe cannot serve as natural-task performance evidence. A changed winner
only proves that bids can change allocation; it does not prove that the
changed allocation has higher utility. This finding does not invalidate
the ordinary natural evaluation, where the probe is disabled.

## Source and artifact identification

- Previous log:
  `runs/model_optimization_v1/eventwise_cf_k4_w8a10_v1_20261020/pilot_log.json`
- Previous log SHA256:
  `a770715696b1a702ffcacf9c272af3d66062bf0653cbd713ea41c3e7ce38737e`
- Archived source bundle SHA256:
  `39e0de9249505bb1bc6831fe89e11a7bbc0920a86172f93d4aa5d29415097134`
- Archived source manifest SHA256:
  `84e54ed59a92a33e86fee4900c634d520c66aa2c90787cd72939509cd8c2de32`
- Initial checkpoint SHA256:
  `386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797`
- Archived source anchors: `detect_env.py:460` and `:466` direct arbitration;
  `detect_mappo.py:1730` bid-only parameter filter; `detect_rules.py:56` and
  `:73` rule/pair definitions; `detect_encoder.py:49` soft proxies;
  `_p3_pilot.py:147` deterministic validation.

The JSON audit includes hashes of both smoke logs, relevant old source
files, final checkpoints and the inspected legacy probe source. The old
log files were rehashed after inspection and remained unchanged.

## Next step

Keep the repaired bid learner for mechanism checks and introduce a controlled
RL task-residual path to broaden the decisions the method can change, while
anchoring the starting task policy. Preserve the current validation seeds,
separate natural task performance from conditional arbitration diagnostics,
and keep teacher, rules, scenario and red model unchanged. The main agent
will provide the separate formal protocol for that mode.
