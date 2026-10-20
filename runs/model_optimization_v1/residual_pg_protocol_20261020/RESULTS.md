# Implementation verification before user-run training

Date: 2026-10-20. Scope: learning-direction repair and controlled training
implementation, not a claim that the task-performance objective is achieved.

- Final related regression: **130 passed**, one existing tensor-construction
  performance warning. Full command and output: pytest_result.json, pytest.txt.
- Python compilation and git diff whitespace checks passed.
- Both expected Gaussian bid gradients and actual policy.update() directions
  are covered for two/three candidates, reversed utilities and utility scaling.
- Candidate hidden repair preserves every initial output exactly and records
  seed, tensor hashes and effective initial checkpoint. Ordinary load/evaluation
  does not repair or reset weights.
- Tests cover real rollout likelihoods, task/base freezing under existing Adam
  state, protected radar-axis row, separate gradient clipping, no-event task
  updates, startup configuration and best-checkpoint/hash verification.

## Real simulator smoke checks

1. Bid-only, seeds 150000-150001: zero multi-candidate events, zero bid update.
   This confirms absent events do not invent bid credit. It does not assess
   whether the learned arbitration improves task performance.
2. Bid-only, seed 100001 with existing training-only request exposure: 27
   informative real events, 54 candidate replays; bid gradient norm 12.801259,
   parameter delta L2 0.00754835, post-update bid KL 0.000632. Only bid tensors
   changed. Natural validation contained no multi-candidate event, so its
   unchanged trajectory was structurally expected.
3. Conflict residual, seed 100001, natural requests and full penalties:
   zero multi-candidate events, but residual gradient norm 0.16309404 and
   residual parameter delta L2 0.00022954. Only task residual and critic changed;
   all base actor/heads/log std and radar-axis residual row stayed exactly fixed.
   Validation seed 103003 remains failed before/after; the rounded return
   -4365.1 -> -4365.0 is not evidence of a useful task gain.

Each run completed with its source-integrity check unchanged. Machine-readable
weight checks, log hashes and initial artifacts: implementation_audit.json.
The three smoke directories retain their logs, effective starts and final/best
models. Only byte-identical same-run checkpoint copies were removed after hash
comparison, as recorded in each artifact_retention.json. Existing provenance
documents the files captured during the run before this explicit pruning.

## Scientific boundaries and next run

The independent frozen-real-event audit reports a 10.06% reduction in held-out
stochastic one-step expected regret. Deterministic held-out winners did not
change. This is separate from task success, return and violations.

The old fixed bid-only validation has a conditional 80% success upper bound
against HEUR's 90% because two failed episodes have no competing candidates.
The next experiment opens only task conflict residuals to escape that fixed
intervention bound while protecting the inherited base controller. It may
still fail because of limited feature expressiveness, sparse natural bid
events or misalignment of one-step utility with long-horizon task success.

Teacher, rule, scenario, red-agent and simulator task/config files have no
changes in this work. RL supplies both task residuals and allocation bids.
HEUR is not used for arbitration labels. No final-model success improvement
or advantage over HEUR has yet been established.

The fixed long-run protocol is PLAN.md; run train.ps1 from PowerShell. The
script's preflight passed and was saved as launch_preflight.json. The agent
did not start the 30-iteration run. verification.json records current code
hashes and test evidence.
