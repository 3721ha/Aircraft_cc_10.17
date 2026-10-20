# Window replay and local task adaptation: diagnostic results

The final goal remains an RL deployment policy that exceeds HEUR. This round
does not meet that goal. No candidate is promoted; the retained task candidate
is still residual PPO iter 3, SHA256
`38815f375df492ee320d9a0d4a9105efaf3944a6694165b295f646c9917aea18`.
Teacher, rule, environment, scenario, red-agent and numeric task/config files
have no changes. K=4/W=8/A=10, C04=80, axis limit=60 degrees remain fixed.

The implementation adds an optional H-decision forced-winner return on
independent simulator copies. H=8 uses deterministic RL continuation and a
copied shield. Its target is discounted environment return plus the existing
terminal-corrected tracking/stability potential difference. This is a
conditional finite-horizon auxiliary objective, not an unbiased full-episode
success objective. Factorized event-bid PPO remains an approximation.

An optional 169-observation, 64-hidden-unit task adapter adds bounded local
residuals. Its final layer starts at zero; the radar-axis output is always
masked to zero. The base actor, task heads and task exploration remain frozen
in this training mode. The gate is a soft pressure proxy, not a detector of
all thirteen actual rule-conflict pairs. The original linear residual is
still unbounded. BC does not train the new adapter or arbitration heads.

Natural stochastic collection, seeds 180000-180007, produced zero training
events and only one held-out event. Training stopped before any update. The
original event bank and a clearly marked post-failure review are retained.
These data cannot establish conditional arbitration learning.

Training-only request exposure on separate seeds 182000-182007 produced 31
training events and 33 held-out events. Only three episodes per split had
events; many were temporally adjacent. A fixed 24-update diagnostic and a
matched 24-update episode-balanced diagnostic gave the following results.
These are conditional utility regrets, not mission success rates.

| Sampling | Training expected regret | Held-out expected regret | Held-out deterministic regret |
|---|---:|---:|---:|
| Event weighted | 0.69628 -> 0.68221 | 1.09172 -> 1.12155 | 0.33377 -> 0.94372 |
| Episode balanced | 0.69628 -> 0.67357 | 1.09172 -> 1.09746 | 0.33377 -> 0.77448 |

Neither diagnostic establishes generalization. Episode balancing alone does
not repair this small event bank. Both use the same previously inspected
diagnostic hold-out; it is not fresh confirmation of a selected method.
All utilities came from environment copies, with no HEUR winner labels.

A two-update, four-episode natural task smoke verified executable learning,
not performance. The adapter's gradient and parameter changes were nonzero;
its final maximum absolute output was 0.004885. On two separate diagnostic
validation seeds 181100-181101, RL success stayed 0/2, return declined from
-7106.5 to -10185.0 and violations/step rose from 7.4687 to 9.1061. HEUR was
2/2, return -1608.9, violations/step 5.9311. The initial candidate was correctly
retained as best. Two seeds and two updates do not support a general task
performance estimate or a claim of impossibility.

Training-seed decision margins also limit the adapter-only interpretation:
with its fixed scale 1, fewer than 17% of observed categorical decisions had
a top-two margin below 2. This is a capacity/exploration concern, not proof
that the new adapter can change those decisions usefully.

Implementation checks verify exact inherited initial action heads, actual
PPO gradients, frozen base/anchors/axis, whole-group legacy migration,
partial-group rejection, serialization and live-simulator replay isolation.
The latter exercises real sensor RNG, red dynamics and copied QP shields;
it uses constructed requests from genuine sensor tracks and is not a natural
task-performance benchmark. Tests and exact commands are in pytest.txt,
pytest.xml and pytest_result.json. integrity_audit.json verifies actual smoke
weights and source hashes; results_summary.json preserves raw comparisons.

The next bounded research stage is specified in PLAN.md. It collects more
episode coverage before training; no full joint long run is started or
recommended on the basis of these failed diagnostics. Reserved task-test
seeds 162000-162029 were not used. Checkpoints remain local; the archive keeps
useful starts/finals/logs and excludes redundant copies, without deleting
files, as recorded in artifact_retention.json.
