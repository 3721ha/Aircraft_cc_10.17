# Coverage-stop review and fixed held-out extension

The original diverse-event audit stopped before any PPO update. Training
coverage passed (37 informative events from 26 episodes, minimum 16).
Held-out coverage did not pass (7 events from 5 episodes, minimum 8).
The original train/held-out collection contained 64/16 episodes respectively.
The source/checkpoint hashes passed; no new model checkpoint was written.

This is a collection-coverage result, not a mission-performance result. It
does not show that the task is impossible or that arbitration training works.
Sparse events in the held-out episodes explain the stop. Changing the event
spacing would not increase the number of independent episodes with events.

The next budget is fixed before collection: append all 32 episode seeds
185016-185047 to the existing held-out split. Reuse every original training
and held-out case without modification. Do not move training cases into the
held-out split, lower its minimum eight informative episodes, select seeds
using reward/winner quality, or stop collection at the first passing count.
The training settings, H8 utilities, numeric task parameters, per-episode
event cap of four and minimum spacing of 24 decisions remain unchanged.
The new stochastic action collection stream has explicit seed 20261023.

The extension tool verifies the original report and event-bank hashes, source
archive and current source files, checkpoint, split membership, coverage and
selection policy. It refuses an original run with learning updates or model
checkpoints. A separate merged event bank retains the original source snapshot
and records the extension snapshot, seed block and collection RNG seed.
Original artifacts are never overwritten; nonempty outputs are refused.

The launcher then uses the merged bank for the original fixed 48-update
bid-only training protocol. It does not recollect the original 80 episodes.
Coverage is checked again by the original audit before training. If still
insufficient after this entire fixed extension, it saves a report and stops;
do not repeatedly add seed blocks until a favorable result appears. If
coverage passes, perform the fixed training and evaluate before/after without
intermediate model selection. No held-out performance has yet been evaluated
for this experiment. The original gate (episode-balanced expected regret
reduction at least 5%, deterministic regret does not increase) remains a
diagnostic decision criterion, not evidence of superiority to HEUR.

From the repository root:

```powershell
& .\runs\model_optimization_v1\window_geometry_protocol_20261020\resume_diverse_events.ps1
```

`-PreflightOnly` verifies the source artifacts and arguments without collecting
episodes or creating the formal output directories. The long extension and
training are left for the user. A separate one-episode implementation smoke
uses seed 186000, outside the formal splits, and has no learning updates.

Formal merged bank:
`runs/model_optimization_v1/window_diverse_event_extension_k4_w8a10_v1/event_bank.json`

Formal final report:
`runs/model_optimization_v1/window_diverse_event_learning_k4_w8a10_v2/learning_audit.json`

Teacher, rules, task/configuration, red-agent and environment source files
remain unchanged. Final deployment/arbitration remain RL; the event utilities
come from environment copies and deterministic RL continuation, without HEUR
winner labels. The reserved mission test is not used.
