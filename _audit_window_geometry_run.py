"""Verify the frozen backbone and source of a window/geometry pilot run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run = (ROOT / args.run).resolve()
    log_path = run / "pilot_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    record = log["run_provenance"][-1]
    initial_artifact = record["candidate_head_training_repair"]["effective_checkpoint"]
    source = record["curriculum_start"]
    initial_path = Path(initial_artifact["path"])
    latest_artifact = next(item for item in record["outputs"]
                           if item["role"] == "curriculum_latest_checkpoint")
    latest_path = Path(latest_artifact["path"])
    initial = torch.load(initial_path, map_location="cpu")
    latest = torch.load(latest_path, map_location="cpu")
    changed = [name for name, value in initial["model"].items()
               if not torch.equal(value, latest["model"][name])]
    allowed = ("conflict_residual.", "task_geometry_adapter.", "bid_head.",
               "bid_candidate_head.", "bid_conflict_residual.", "critic.")
    unexpected = [name for name in changed
                  if name != "bid_log_std" and not name.startswith(allowed)]
    anchors_fixed = {
        name: all(torch.equal(value, latest[name][key])
                  for key, value in initial[name].items())
        for name in ("xi_anchor_model", "policy_anchor_model")
    }
    from aircraft_sim.detect_config import DetectSimConfig
    from aircraft_sim.detect_env import DetectEnv
    from aircraft_sim.detect_mappo import DetectMAPPOPolicy
    from aircraft_sim.detect_shield import DetectQPShield
    env = DetectEnv(DetectSimConfig())
    policy = DetectMAPPOPolicy(env, DetectQPShield(env.cfg), seed=0)
    policy.load(source["path"])
    source_state = {key: value.clone() for key, value in policy.model.state_dict().items()}
    # This is a tensor-equivalence check, not a task-performance evaluation.
    rng = np.random.default_rng(20261022)
    obs = torch.tensor(rng.normal(size=(32, 169)), dtype=torch.float32)
    obs[:, -13:] = torch.tensor(rng.uniform(size=(32, 13)), dtype=torch.float32)
    with torch.no_grad():
        original_outputs = tuple(x.clone() for x in policy.model.action_parameters(obs))
    policy.load(str(initial_path))
    with torch.no_grad():
        initial_outputs = policy.model.action_parameters(obs)
    source_preserved = all(torch.equal(value, initial["model"][key])
                           for key, value in source_state.items())
    protected_axis = torch.equal(initial["model"]["conflict_residual.weight"][2],
                                 latest["model"]["conflict_residual.weight"][2])
    checks = {
        "completed": record["status"] == "completed",
        "source_unchanged_during_run": record["source_integrity"]["unchanged"],
        "source_checkpoint_hash": sha256(source["path"]) == source["sha256"],
        "effective_initial_hash": sha256(initial_path) == initial_artifact["sha256"],
        "latest_hash": sha256(latest_path) == latest_artifact["sha256"],
        "source_tensors_preserved_at_initialization": source_preserved,
        "all_initial_action_heads_identical": all(torch.equal(a, b) for a, b in
                                                   zip(original_outputs, initial_outputs)),
        "only_allowed_tensors_changed": not unexpected,
        "both_anchors_fixed": all(anchors_fixed.values()),
        "protected_axis_row_fixed": protected_axis,
        "geometry_learned": any(name.startswith("task_geometry_adapter.")
                                 for name in changed),
    }
    report = {
        "scope": "implementation_integrity_not_success_improvement",
        "pilot_log": str(log_path), "pilot_log_sha256": sha256(log_path),
        "source_checkpoint": source,
        "effective_initial_checkpoint": initial_artifact,
        "latest_checkpoint": latest_artifact,
        "checks": checks, "all_passed": all(checks.values()),
        "changed_tensors": changed, "unexpected_changed_tensors": unexpected,
        "anchor_checks": anchors_fixed,
    }
    from _experiment_provenance import write_json_atomic
    write_json_atomic(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
