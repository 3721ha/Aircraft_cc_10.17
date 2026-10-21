"""Paired frozen-policy K4 audit: RL direct arbitration vs fixed-score arbitration.

Task heads, weights, observations, requests, shields and sensor model are held
constant. Only the arbitration branch changes; no labels or training are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _experiment_provenance import (
    add_output, artifact_record, capture_run_provenance,
    capture_source_snapshot, changed_source_files, register_active_run,
    track_run_lifecycle, write_json_atomic,
)
from _diagnose_sensor_profile import (
    baseline_winner, build_configs, evaluation_policy_config,
    model_fingerprint, wilson_interval,
)

DEFAULT_CHECKPOINT = (
    ROOT / "runs/model_optimization_v1/sensor_joint_adapt_k4_w8a10_v1/initial.pt")
DEFAULT_CHECKPOINT_SHA256 = (
    "ef86e5734d2a6466694371a0931386d7f1937355f279eed28f1560dcae27499b")


def apply_arbitration_treatment(actions: dict, direct_rl: bool) -> dict:
    """Change only arbitration fields; fixed-score mode ignores policy bids."""
    if direct_rl:
        if not all(action.rl_direct_arbitration for action in actions.values()):
            raise RuntimeError("direct RL treatment requires direct-arbitration actions")
        return dict(actions)
    return {
        agent_id: replace(
            action, arbitration_bid=0.0,
            arbitration_bids=(tuple(0.0 for _ in action.arbitration_bids)
                              if action.arbitration_bids is not None else None),
            rl_direct_arbitration=False)
        for agent_id, action in actions.items()
    }


def run_episode(cfg, checkpoint: Path, saved_config: dict, seed: int,
                direct_rl: bool) -> dict:
    from aircraft_sim.detect_mappo import DetectMAPPOPolicy
    from aircraft_sim.detect_sensor_diagnostics import SensorDiagnosticEnv
    from aircraft_sim.detect_shield import DetectQPShield

    env = SensorDiagnosticEnv(cfg)
    policy_cfg = evaluation_policy_config(saved_config)
    # Both arms generate actions from the same policy mode. The treatment is
    # applied after the shield, immediately before the environment receives it.
    policy_cfg.direct_rl_arbitration = True
    policy = DetectMAPPOPolicy(env, DetectQPShield(cfg), policy_cfg, seed=0)
    policy.load(str(checkpoint))
    policy.config = policy_cfg
    for name in ("model", "xi_anchor_model", "policy_anchor_model"):
        getattr(policy, name).eval()
    fingerprint_before = model_fingerprint(policy)
    env.reset({"seed": seed, "k_red": cfg.K_RED})
    violations_by_family = Counter()
    request_outcomes = Counter()
    arbitration_modes = Counter()
    steps = total_return = violations = conflicts = changed = interventions = 0
    while env.terminal is None:
        observations = env.observe()
        actions, _ = policy.act_batch(observations, deterministic=True)
        if not all(a.rl_direct_arbitration for a in actions.values()):
            raise RuntimeError("policy did not produce direct-arbitration actions")
        safe_actions, corrections = policy.shield.filter(env, actions)
        if not all(a.rl_direct_arbitration for a in safe_actions.values()):
            raise RuntimeError("shield changed the policy arbitration mode")
        executed_actions = apply_arbitration_treatment(safe_actions, direct_rl)
        if direct_rl and not all(a.rl_direct_arbitration
                                 for a in executed_actions.values()):
            raise RuntimeError("direct RL treatment flag mismatch")
        if not direct_rl and any(
                a.arbitration_bid != 0.0 or
                (a.arbitration_bids is not None and
                 any(value != 0.0 for value in a.arbitration_bids)) or
                a.rl_direct_arbitration
                for a in executed_actions.values()):
            raise RuntimeError("fixed-score treatment retained a learned bid")
        interventions += len(corrections)
        result = env.step(executed_actions)
        steps += 1
        total_return += float(result.reward)
        report = result.info["report"]
        violations += len(report.violations)
        violations_by_family.update(key.split(":", 1)[0] for key in report.violations)
        request_outcomes.update(item.get("reason", "unknown")
                                for item in result.info.get("request_outcomes", []))
        for event in result.info.get("arbitration", []):
            conflicts += 1
            mode = event["arbitration_mode"]
            expected_mode = "rl_direct" if direct_rl else "fixed_score_plus_bid"
            if mode != expected_mode:
                raise RuntimeError(f"expected {expected_mode}, observed {mode}")
            if not direct_rl and any(
                    float(value) != 0.0 for value in event["bids"].values()):
                raise RuntimeError("fixed-score conflict received a nonzero RL bid")
            arbitration_modes[mode] += 1
            changed += int(int(event["winner"]) != baseline_winner(event))
    fingerprint_after = model_fingerprint(policy)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError("frozen policy weights changed during evaluation")
    return {
        "seed": seed, "arbitration": "rl_direct" if direct_rl else "fixed_score",
        "terminal": env.terminal, "return": total_return, "steps": steps,
        "violations": violations, "violations_per_step": violations / max(1, steps),
        "judged_targets": sum(env.judged.values()), "shield_interventions": interventions,
        "violations_by_family": dict(violations_by_family),
        "multi_candidate_events": conflicts,
        "winners_different_from_fixed_score": changed,
        "arbitration_modes": dict(arbitration_modes),
        "request_outcomes": dict(request_outcomes),
        "sensor_diagnostics": env.sensor_diagnostics(),
        "loaded_tensor_sha256": fingerprint_after, "weights_unchanged": True,
        "effective_policy_config": asdict(policy_cfg),
    }


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    steps = sum(row["steps"] for row in rows)
    violations = sum(row["violations"] for row in rows)
    successes = sum(row["terminal"] == "success" for row in rows)
    modes = Counter()
    for row in rows:
        modes.update(row["arbitration_modes"])
    return {
        "episodes": count, "successes": successes,
        "success_rate": successes / count,
        "success_rate_wilson_95": wilson_interval(successes, count),
        "mean_return": sum(row["return"] for row in rows) / count,
        "total_steps": steps, "total_violations": violations,
        "violations_per_step": violations / max(1, steps),
        "mean_episode_violations_per_step": sum(
            row["violations_per_step"] for row in rows) / count,
        "mean_judged_targets": sum(row["judged_targets"] for row in rows) / count,
        "multi_candidate_events": sum(row["multi_candidate_events"] for row in rows),
        "winners_different_from_fixed_score": sum(
            row["winners_different_from_fixed_score"] for row in rows),
        "arbitration_modes": dict(modes),
    }


def paired(left: list[dict], right: list[dict]) -> dict:
    lmap, rmap = ({row["seed"]: row for row in group} for group in (left, right))
    if lmap.keys() != rmap.keys() or len(lmap) != len(left):
        raise ValueError("paired arms require identical unique seeds")
    a, b = summarize(left), summarize(right)
    return {
        "direction": "rl_direct_minus_fixed_score",
        "episodes": len(left), "success_rate_delta": b["success_rate"] - a["success_rate"],
        "mean_return_delta": b["mean_return"] - a["mean_return"],
        "violations_per_step_delta": b["violations_per_step"] - a["violations_per_step"],
        "rl_only_successes": sum(lmap[s]["terminal"] != "success" and
                                  rmap[s]["terminal"] == "success" for s in lmap),
        "fixed_only_successes": sum(lmap[s]["terminal"] == "success" and
                                     rmap[s]["terminal"] != "success" for s in lmap),
    }


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--plan-only", action="store_true")
    return ap


@track_run_lifecycle
def main():
    ap = parser()
    args = ap.parse_args()
    source_path, output = args.checkpoint.resolve(), args.output.resolve()
    if output.exists():
        ap.error(f"output already exists: {output}")
    if args.seed_start < 0 or args.episodes < 1 or args.torch_threads < 1:
        ap.error("seed-start must be nonnegative; episodes and torch-threads positive")
    checkpoint_record = artifact_record(source_path, "frozen_checkpoint")
    if (not checkpoint_record["exists"] or
            checkpoint_record["sha256"] != args.expected_checkpoint_sha256):
        ap.error("checkpoint missing or differs from its declared SHA256")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    if args.plan_only:
        print(json.dumps({"checkpoint": checkpoint_record, "seeds": seeds,
                          "arms": ["rl_direct", "fixed_score"],
                          "task": {"k": 4, "stable_window": 8, "age_std_track": 10,
                                   "radar_hit_model": "deterministic_beam", "w8_scale": 1.0},
                          "fixed_score_bids": "all_zero",
                          "training": False, "teacher": False, "output": str(output)}, indent=2))
        return 0
    source = capture_source_snapshot(ROOT, output, [Path(__file__),
        ROOT / "_diagnose_sensor_profile.py", ROOT / "Aircraft_mumt/aircraft_sim/detect_env.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_mappo.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_sensor_diagnostics.py"])
    import torch
    torch.set_num_threads(args.torch_threads)
    cfg = build_configs(4, 8, 10.0)["deterministic_beam"]
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "provenance" / "frozen_policy.pt"
    shutil.copy2(source_path, frozen_path)
    frozen_record = artifact_record(frozen_path, "archived_frozen_checkpoint")
    if frozen_record["sha256"] != checkpoint_record["sha256"]:
        raise RuntimeError("checkpoint changed while archiving")
    checkpoint_data = torch.load(frozen_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint_data.get("config")
    policy_config = evaluation_policy_config(saved_config)
    base_config = asdict(cfg)
    base_config["W8_SCALE"] = 1.0
    cfg = type(cfg)(**base_config)
    prov = capture_run_provenance(ROOT, output, [Path(__file__),
        ROOT / "_diagnose_sensor_profile.py", ROOT / "Aircraft_mumt/aircraft_sim/detect_env.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_mappo.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_sensor_diagnostics.py"],
        "frozen_arbitration_ablation", task_config=asdict(cfg),
        algorithm_config={"saved": saved_config, "effective": asdict(policy_config)},
        runtime_parameters={"arguments": vars(args), "seeds": seeds,
                            "training": False, "teacher": False,
                            "only_treatment": "direct RL vs fixed-score arbitration",
                            "fixed_score_bid_override": "all arbitration bids set to zero after shield"},
        inputs=[checkpoint_record, frozen_record], source_snapshot=source)
    prov_path, result_path = output / "diagnostic.provenance.json", output / "arbitration_ablation.json"
    register_active_run(prov, lambda value: write_json_atomic(prov_path, value))
    write_json_atomic(prov_path, prov)
    rows = []
    for seed in seeds:
        for direct in (False, True):
            if changed_source_files(source):
                raise RuntimeError("source changed during diagnostic")
            row = run_episode(cfg, frozen_path, saved_config, seed, direct)
            rows.append(row)
            write_json_atomic(output / "episode_progress.json", {"episodes": rows})
            print(f"seed={seed} {row['arbitration']} terminal={row['terminal']} "
                  f"return={row['return']:.1f} conflicts={row['multi_candidate_events']}", flush=True)
        fixed = [r for r in rows if r["arbitration"] == "fixed_score"]
    direct = [r for r in rows if r["arbitration"] == "rl_direct"]
    fingerprints = {r["loaded_tensor_sha256"] for r in rows}
    if len(fingerprints) != 1:
        raise RuntimeError("different policy tensors across treatment arms")
    report = {
        "status": "completed_frozen_arbitration_ablation", "episodes_per_arm": args.episodes,
        "seeds": seeds, "task_config": asdict(cfg),
        "summaries": {"fixed_score": summarize(fixed), "rl_direct": summarize(direct)},
        "paired_rl_direct_minus_fixed_score": paired(fixed, direct),
        "loaded_tensor_sha256": next(iter(fingerprints)),
        "weights_unchanged": all(r["weights_unchanged"] for r in rows),
        "training": False, "teacher_used": False, "rule_or_scenario_changes": False,
        "treatment": {
            "rl_direct": "highest learned bid among C04-feasible candidates",
            "fixed_score": "fixed quality score plus incumbent hysteresis; all learned bids are zero",
        },
        "limits": ["Arbitration trajectories diverge after a winner differs.",
                   "This isolates arbitration mode, not every possible policy adaptation.",
                   "Fixed-score arbitration is a diagnostic baseline, not HEUR."],
        "episodes": rows, "provenance_sidecar": str(prov_path),
    }
    if changed_source_files(source):
        raise RuntimeError("source changed before diagnostic completion")
    write_json_atomic(result_path, report)
    (output / "episode_progress.json").unlink()
    add_output(prov, result_path, "arbitration_ablation")
    prov["checkpoint_unchanged_during_evaluation"] = True
    print(json.dumps({"summaries": report["summaries"],
                      "paired": report["paired_rl_direct_minus_fixed_score"]}, indent=2))
    print(f"Completed: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
