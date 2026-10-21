"""Short joint RL adaptation to deterministic in-beam sensing, K4/W8/A10.

Fresh runs only. The initial RL checkpoint participates in validation-based
selection, HEUR is an independent comparator, and no reserved test is run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _experiment_provenance import (
    add_output, artifact_record, canonical_sha256, capture_run_provenance,
    capture_source_snapshot, changed_source_files, register_active_run,
    track_run_lifecycle, write_json_atomic,
)
from _diagnose_sensor_profile import (
    DEFAULT_CHECKPOINT, DEFAULT_CHECKPOINT_SHA256, build_configs,
    evaluation_policy_config, model_fingerprint, paired_difference,
    run_episode, summarize,
)

RESERVED_TEST_SEEDS = tuple(range(208000, 208030))
CONSUMED_DIAGNOSTIC_SEEDS = set(range(205000, 205030))
CONSUMED_TEST_SEEDS = set(range(162000, 162030))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--episodes-per-iter", type=int, default=8)
    ap.add_argument("--validation-every", type=int, default=2)
    ap.add_argument("--train-seed-start", type=int, default=206000)
    ap.add_argument("--validation-seed-start", type=int, default=207000)
    ap.add_argument("--validation-episodes", type=int, default=12)
    ap.add_argument("--optimizer-seed", type=int, default=20261021)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--plan-only", action="store_true",
                    help="validate inputs and print the protocol without creating outputs")
    return ap


def preflight(args) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    for name in ("iterations", "episodes_per_iter", "validation_every",
                 "validation_episodes", "epochs", "torch_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("train_seed_start", "validation_seed_start", "optimizer_seed"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if (len(args.expected_checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.expected_checkpoint_sha256)):
        raise ValueError("expected checkpoint SHA256 must be 64 lowercase hex characters")
    checkpoint = artifact_record(args.checkpoint.resolve(), "source_best_checkpoint")
    if not checkpoint["exists"] or checkpoint["sha256"] != args.expected_checkpoint_sha256:
        raise ValueError("source checkpoint missing or SHA256 differs from declared best model")
    train = list(range(args.train_seed_start,
                       args.train_seed_start + args.iterations * args.episodes_per_iter))
    validation = list(range(args.validation_seed_start,
                            args.validation_seed_start + args.validation_episodes))
    forbidden = set(RESERVED_TEST_SEEDS) | CONSUMED_DIAGNOSTIC_SEEDS | CONSUMED_TEST_SEEDS
    if set(train) & set(validation):
        raise ValueError("training and validation seeds overlap")
    if (set(train) | set(validation)) & forbidden:
        raise ValueError("training/validation overlaps a reserved test or consumed diagnostic/test seed")
    return {"source_checkpoint": checkpoint, "training_seeds": train,
            "validation_seeds": validation,
            "reserved_test_seeds_not_run": list(RESERVED_TEST_SEEDS),
            "task": {"K_RED": 4, "STABLE_WINDOW": 8, "AGE_STD_TRACK": 10.0,
                     "RADAR_HIT_MODEL": "deterministic_beam", "W8_SCALE": 1.0},
            "training": {"mode": "joint", "learning_rate": args.learning_rate,
                         "epochs": args.epochs, "iterations": args.iterations,
                         "episodes_per_iter": args.episodes_per_iter,
                         "xi_ppo_coef": 1.0, "xi_anchor_coef": 0.0,
                         "policy_anchor_coef": 0.5,
                         "mask_intervened_continuous": True,
                         "mask_intervened_radar_axis": False,
                         "direct_rl_arbitration": True,
                         "conflict_exposure_prob": 0.0,
                         "cf_bid_horizon": 1,
                         "train_geometry_adapter": False},
            "selection": "validation successes, then mean raw return, then lower total violations/steps; strict improvement only",
            "no_training_teacher": True, "no_conflict_injection": True,
            "no_final_test": True}


def adaptation_config(saved: dict, learning_rate: float, epochs: int):
    config = evaluation_policy_config(saved)
    config = replace(
        config, learning_rate=learning_rate, epochs=epochs,
        bid_only_ppo=False, task_residual_only_ppo=False,
        train_geometry_adapter=False, direct_rl_arbitration=True,
        conflict_exposure_prob=0.0, xi_ppo_coef=1.0, xi_anchor_coef=0.0,
        policy_anchor_coef=0.5, policy_anchor_monitor=True,
        mask_intervened_continuous=True, mask_intervened_radar_axis=False,
        cf_bid_horizon=1)
    config.validate_training_mode()
    return config


def selection_rank(summary: dict) -> tuple:
    return (int(summary["successes"]), float(summary["mean_return"]),
            -float(summary["violations_per_step"]))


def tensor_state(module) -> dict:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def module_digest(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}/{tensor.dtype}/{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_changes(before: dict, after: dict) -> dict:
    """Measure actual task-head changes, including the radar-axis row."""
    groups = {
        "actor": [], "flight_controls": [], "radar_axis": [],
        "role": [], "gear": [], "pointer": [], "bid": [],
        "task_conflict_residual": [], "critic": [], "geometry_adapter": []}
    for name, old in before.items():
        difference = after[name].detach() - old
        if name.startswith("cont_head."):
            groups["flight_controls"].append(difference[:2])
            groups["radar_axis"].append(difference[2:3])
        elif name == "log_std":
            groups["flight_controls"].append(difference[:2])
            groups["radar_axis"].append(difference[2:3])
        elif name == "conflict_residual.weight":
            groups["task_conflict_residual"].append(difference)
            groups["radar_axis"].append(difference[2:3])
        elif name.startswith(("bid_head.", "bid_candidate_head.", "bid_conflict_residual.")) or name == "bid_log_std":
            groups["bid"].append(difference)
        else:
            for label, prefix in (("actor", "actor."), ("role", "role_head."),
                                  ("gear", "gear_head."), ("pointer", "pointer_head."),
                                  ("critic", "critic."),
                                  ("geometry_adapter", "task_geometry_adapter.")):
                if name.startswith(prefix):
                    groups[label].append(difference)
                    break
    return {name: math.sqrt(sum(float(value.square().sum()) for value in values))
            for name, values in groups.items()}


def evaluate_checkpoint(cfg, checkpoint: Path, saved_config: dict,
                        seeds: list[int]) -> list[dict]:
    import torch
    # run_episode constructs a policy using manual_seed(0). Restore the
    # training stream after all evaluation construction and action calls.
    with torch.random.fork_rng(devices=[]):
        return [run_episode(cfg, "rl", seed, checkpoint, saved_config)
                for seed in seeds]


def write_checkpoint(policy, path: Path) -> dict:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(path))
    return artifact_record(path, "adaptation_checkpoint")


def check_source(source):
    drift = changed_source_files(source)
    if drift:
        raise RuntimeError(f"source files changed during training: {drift}")


def ensure_finite(value, path="metrics"):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            ensure_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_finite(item, f"{path}[{index}]")


def training_episode_record(policy, data: list[dict], seed: int) -> dict:
    if not data or not data[-1]["done"]:
        raise RuntimeError("PPO requires a complete training episode")
    if any(row.get("n_forced_overlap_pairs", 0) or row.get("n_pointer_exposed", 0)
           for row in data):
        raise RuntimeError("unexpected training conflict injection")
    return {
        "seed": seed, "terminal": data[-1]["terminal"], "steps": len(data),
        "environment_return": sum(row["environment_reward"] for row in data),
        "shaped_return_scaled": sum(row["reward"] for row in data),
        "violations": sum(row["violations"] for row in data),
        "judged_targets": sum(policy.env.judged.values()),
        "multi_candidate_events": sum(row["n_multi_candidate_events"] for row in data),
        "cf_informative_events": sum(row["n_cf_bid_informative_events"] for row in data),
        "cf_candidate_branches": sum(row["n_cf_bid_candidate_branches"] for row in data),
        "cf_actual_regrets": [v for row in data for v in row["bid_cf_actual_regrets"]],
        "requests": sum(row["n_requests"] for row in data),
        "c04_rejected_requests": sum(row["n_c04_rejected_requests"] for row in data),
        "sensor_diagnostics": policy.env.sensor_diagnostics(),
    }


@track_run_lifecycle
def main():
    ap = parser()
    args = ap.parse_args()
    try:
        protocol = preflight(args)
    except ValueError as error:
        ap.error(str(error))
    if args.plan_only:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    output = args.output.resolve()
    source_path = args.checkpoint.resolve()
    source = capture_source_snapshot(
        ROOT, output, [Path(__file__), ROOT / "_diagnose_sensor_profile.py"])
    import torch
    from aircraft_sim.detect_mappo import DetectMAPPOPolicy
    from aircraft_sim.detect_sensor_diagnostics import SensorDiagnosticEnv
    from aircraft_sim.detect_shield import DetectQPShield

    torch.set_num_threads(args.torch_threads)
    copied_source = output / "provenance" / "source_best.pt"
    shutil.copy2(source_path, copied_source)
    copied_record = artifact_record(copied_source, "archived_source_best_checkpoint")
    if copied_record["sha256"] != protocol["source_checkpoint"]["sha256"]:
        raise RuntimeError("source checkpoint changed during copying")
    data = torch.load(copied_source, map_location="cpu", weights_only=False)
    config = adaptation_config(data.get("config"), args.learning_rate, args.epochs)
    cfg = build_configs(4, 8, 10.0)["deterministic_beam"]
    record = capture_run_provenance(
        ROOT, output, [Path(__file__), ROOT / "_diagnose_sensor_profile.py"],
        "short_joint_sensor_adaptation", cfg,
        algorithm_config={"source_saved": data.get("config"), "effective": asdict(config)},
        runtime_parameters={"protocol": protocol, "arguments": vars(args)},
        inputs=[protocol["source_checkpoint"], copied_record], source_snapshot=source)
    provenance_path = output / "adaptation.provenance.json"
    log_path = output / "adaptation_log.json"
    register_active_run(record, lambda value: write_json_atomic(provenance_path, value))
    write_json_atomic(provenance_path, record)
    env = SensorDiagnosticEnv(cfg)
    policy = DetectMAPPOPolicy(env, DetectQPShield(cfg), config=config, seed=0)
    policy.load(str(copied_source))
    # Refresh only the task trust-region reference to this chosen RL start.
    # Neither HEUR labels nor a head repair change the starting actor.
    policy.refresh_policy_anchor()
    source_model = data["model"]
    if not all(torch.equal(value, policy.model.state_dict()[name])
               for name, value in source_model.items()):
        raise RuntimeError("effective initial actor differs from the source checkpoint")
    initial_artifact = write_checkpoint(policy, output / "initial.pt")
    fixed = {"xi_anchor": module_digest(policy.xi_anchor_model),
             "policy_anchor": module_digest(policy.policy_anchor_model),
             "geometry_adapter": module_digest(policy.model.task_geometry_adapter)}
    log = {
        "status": "running", "protocol": protocol,
        "task_config": asdict(cfg), "task_config_sha256": canonical_sha256(asdict(cfg)),
        "policy_config": asdict(config), "source_checkpoint": protocol["source_checkpoint"],
        "initial_checkpoint": initial_artifact,
        "checkpoint_migration": policy.checkpoint_migration_record,
        "anchor_reference": "explicit chosen RL source; refreshed before initial validation",
        "fixed_module_sha256": fixed,
        "training": [], "validations": [],
        "reserved_test_seeds_not_run": list(RESERVED_TEST_SEEDS),
        "selection_limitations": "Validation is used repeatedly for selection; it is not a final test. No arbitration-benefit claim follows from task success alone.",
    }
    write_json_atomic(log_path, log)
    seeds = protocol["validation_seeds"]
    with torch.random.fork_rng(devices=[]):
        heur_rows = [run_episode(cfg, "heur", seed, copied_source, data["config"])
                     for seed in seeds]
    heur_summary = summarize(heur_rows)
    log["heur_validation"] = {"summary": heur_summary, "episodes": heur_rows}
    initial_rows = evaluate_checkpoint(cfg, Path(initial_artifact["path"]), asdict(config), seeds)
    initial_summary = summarize(initial_rows)
    best = {"update": 0, "checkpoint": initial_artifact,
            "validation_summary": initial_summary}
    log["validations"].append({"update": 0, "checkpoint": initial_artifact,
                               "summary": initial_summary, "episodes": initial_rows,
                               "rl_minus_heur": paired_difference(heur_rows, initial_rows)})
    log["best"] = best
    log["latest"] = {"update": 0, "checkpoint": initial_artifact}
    record["effective_initial_checkpoint"] = initial_artifact
    add_output(record, Path(initial_artifact["path"]), "effective_initial_checkpoint")
    write_json_atomic(log_path, log)
    write_json_atomic(provenance_path, record)
    print(f"initial validation RL={initial_summary['successes']}/{len(seeds)} "
          f"HEUR={heur_summary['successes']}/{len(seeds)}", flush=True)
    torch.manual_seed(args.optimizer_seed)
    for update in range(1, args.iterations + 1):
        check_source(source)
        rollout, episodes = [], []
        policy.model.train()
        for episode in range(args.episodes_per_iter):
            seed = args.train_seed_start + (update - 1) * args.episodes_per_iter + episode
            collected = policy.collect(k_red=4, seed=seed)
            episodes.append(training_episode_record(policy, collected, seed))
            rollout.extend(collected)
        before = tensor_state(policy.model)
        stats = policy.update(rollout)
        changes = parameter_changes(before, policy.model.state_dict())
        if fixed != {"xi_anchor": module_digest(policy.xi_anchor_model),
                     "policy_anchor": module_digest(policy.policy_anchor_model),
                     "geometry_adapter": module_digest(policy.model.task_geometry_adapter)}:
            raise RuntimeError("a frozen anchor or geometry adapter changed")
        if cfg.RADAR_HIT_MODEL != "deterministic_beam" or env.cfg.W8_SCALE != 1.0:
            raise RuntimeError("training task configuration changed")
        ensure_finite(stats)
        ensure_finite(changes)
        if changes["actor"] == 0 or changes["radar_axis"] == 0:
            raise RuntimeError("joint adaptation did not update the actor and radar-axis parameters")
        artifact = write_checkpoint(policy, output / "checkpoints" / f"update_{update:04d}.pt")
        total_steps = sum(item["steps"] for item in episodes)
        item = {"update": update, "checkpoint": artifact, "episodes": episodes,
                "mean_environment_return": sum(x["environment_return"] for x in episodes) / len(episodes),
                "success_rate": sum(x["terminal"] == "success" for x in episodes) / len(episodes),
                "violations_per_step": sum(x["violations"] for x in episodes) / total_steps,
                "environment_steps": total_steps, "ppo_stats": stats,
                "parameter_delta_l2": changes, "frozen_modules_unchanged": True}
        log["training"].append(item)
        log["latest"] = {"update": update, "checkpoint": artifact}
        write_json_atomic(log_path, log)
        print(f"update={update}/{args.iterations} train_success={item['success_rate']:.3f} "
              f"return={item['mean_environment_return']:.1f} "
              f"xi_delta={changes['radar_axis']:.6g} "
              f"CF_events={sum(x['cf_informative_events'] for x in episodes)}", flush=True)
        if update % args.validation_every == 0 or update == args.iterations:
            check_source(source)
            fingerprint = model_fingerprint(policy)
            rng = torch.get_rng_state().clone()
            rows = evaluate_checkpoint(cfg, Path(artifact["path"]), asdict(config), seeds)
            if not torch.equal(rng, torch.get_rng_state()) or model_fingerprint(policy) != fingerprint:
                raise RuntimeError("validation changed training weights or sampling RNG")
            summary = summarize(rows)
            entry = {"update": update, "checkpoint": artifact, "summary": summary,
                     "episodes": rows, "rl_minus_heur": paired_difference(heur_rows, rows)}
            improved = selection_rank(summary) > selection_rank(best["validation_summary"])
            if improved:
                best = {"update": update, "checkpoint": artifact, "validation_summary": summary}
            entry["selected_as_best"] = improved
            log["validations"].append(entry)
            log["best"] = best
            write_json_atomic(log_path, log)
            print(f"validation update={update} RL={summary['successes']}/{len(seeds)} "
                  f"return={summary['mean_return']:.1f}; best_update={best['update']}", flush=True)
    for artifact in (protocol["source_checkpoint"], copied_record, initial_artifact,
                     best["checkpoint"], log["latest"]["checkpoint"]):
        if artifact_record(artifact["path"], "after")["sha256"] != artifact["sha256"]:
            raise RuntimeError(f"checkpoint changed: {artifact['path']}")
    check_source(source)
    best_path = output / "best.pt"
    shutil.copy2(best["checkpoint"]["path"], best_path)
    selected = artifact_record(best_path, "selected_best_checkpoint")
    if selected["sha256"] != best["checkpoint"]["sha256"]:
        raise RuntimeError("selected best checkpoint copy differs")
    log["best"]["selected_checkpoint"] = selected
    log["status"] = "completed_short_adaptation"
    log["training_episodes_completed"] = len(protocol["training_seeds"])
    log["initial_to_best_validation_success_delta"] = best["validation_summary"]["success_rate"] - initial_summary["success_rate"]
    write_json_atomic(log_path, log)
    add_output(record, log_path, "adaptation_log")
    add_output(record, best_path, "selected_best_checkpoint")
    add_output(record, Path(log["latest"]["checkpoint"]["path"]), "latest_checkpoint")
    record["checkpoint_inputs_unchanged"] = True
    record["frozen_modules_unchanged"] = True
    record["selected_best_update"] = best["update"]
    record["reserved_test_run"] = False
    print(f"Completed. Selected best update={best['update']}. Result: {log_path}", flush=True)


if __name__ == "__main__":
    main()
