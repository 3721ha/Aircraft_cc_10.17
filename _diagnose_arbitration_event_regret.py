"""Same-snapshot counterfactual audit of RL and fixed-score conflict winners."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))
sys.path.insert(0, str(ROOT))

from _diagnose_arbitration_ablation import (
    DEFAULT_CHECKPOINT, DEFAULT_CHECKPOINT_SHA256, baseline_winner,
    build_configs, evaluation_policy_config, model_fingerprint,
)
from _experiment_provenance import (
    add_output, artifact_record, capture_run_provenance,
    capture_source_snapshot, changed_source_files, register_active_run,
    track_run_lifecycle, write_json_atomic,
)


def compare_winners(event: dict, counterfactual: dict) -> dict:
    utilities = {int(agent): float(value)
                 for agent, value in counterfactual["utilities"].items()}
    candidates = sorted(int(agent) for agent in event["candidates"])
    if set(candidates) != set(utilities):
        raise ValueError("Counterfactual utilities do not match conflict candidates")
    learned_winner = int(event["winner"])
    fixed_winner = int(baseline_winner(event))
    if learned_winner not in utilities or fixed_winner not in utilities:
        raise ValueError("Conflict winner is outside the feasible candidate set")
    best_utility = max(utilities.values())
    return {
        "tid": int(event["tid"]), "candidates": candidates,
        "previous_owner": event.get("previous_owner"),
        "scores": {str(int(agent)): float(value)
                   for agent, value in event["scores"].items()},
        "hysteresis": float(event["hysteresis"]),
        "rl_winner": learned_winner, "fixed_score_winner": fixed_winner,
        "winners_differ": learned_winner != fixed_winner,
        "utilities": {str(agent): utilities[agent] for agent in candidates},
        "utility_spread": best_utility - min(utilities.values()),
        "rl_utility_minus_fixed_score": (
            utilities[learned_winner] - utilities[fixed_winner]),
        "rl_regret": best_utility - utilities[learned_winner],
        "fixed_score_regret": best_utility - utilities[fixed_winner],
        "branch_results": counterfactual.get("branch_results", {}),
    }


def _cluster_summary(rows: list[dict], key: str) -> dict:
    selected = [row for row in rows if row["utility_spread"] > 1e-8]
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        by_seed[int(row["seed"])].append(row)
    episode_means = [float(np.mean([row[key] for row in items]))
                     for items in by_seed.values()]
    if len(episode_means) >= 2:
        rng = np.random.default_rng(42021)
        values = np.asarray(episode_means, dtype=float)
        samples = rng.choice(values, size=(50000, len(values)), replace=True).mean(axis=1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    else:
        interval = [None, None]
    return {
        "informative_events": len(selected),
        "episode_clusters": len(by_seed),
        "mean_event_value": (float(np.mean([row[key] for row in selected])
                               if selected else 0.0)),
        "mean_episode_value": (float(np.mean(episode_means))
                               if episode_means else 0.0),
        "episode_cluster_bootstrap_95": interval,
        "positive_events": sum(row[key] > 1e-8 for row in selected),
        "negative_events": sum(row[key] < -1e-8 for row in selected),
        "tie_events": sum(abs(row[key]) <= 1e-8 for row in selected),
    }


def _normalize_horizons(horizons: list[int]) -> list[int]:
    normalized = list(dict.fromkeys(horizons))
    if not normalized or any(type(horizon) is not int or not 1 <= horizon <= 32
                             for horizon in normalized):
        raise ValueError("counterfactual horizons must be unique integers in [1, 32]")
    return normalized


def _summarize_horizons(rows: list[dict], horizons: list[int]) -> dict:
    summaries = {}
    for horizon in horizons:
        events = []
        for row in rows:
            for event in row["event_regrets"]:
                comparison = event["counterfactual_by_horizon"][str(horizon)]
                if comparison["counterfactual_available"]:
                    events.append({**comparison, "seed": event["seed"]})
        changed_events = [event for event in events if event["winners_differ"]]
        grouped = defaultdict(list)
        for event in changed_events:
            grouped[int(event["seed"])].append(event)
        summaries[str(horizon)] = {
            "counterfactual_events": len(events),
            "winner_disagreement_events_with_counterfactual": len(changed_events),
            "winner_disagreement_episodes_with_counterfactual": len(grouped),
            "rl_minus_fixed_score_utility_all_informative": _cluster_summary(
                events, "rl_utility_minus_fixed_score"),
            "rl_minus_fixed_score_utility_when_winners_differ": _cluster_summary(
                changed_events, "rl_utility_minus_fixed_score"),
            "rl_better_changed_events": sum(
                event["utility_spread"] > 1e-8 and
                event["rl_utility_minus_fixed_score"] > 1e-8
                for event in changed_events),
            "fixed_score_better_changed_events": sum(
                event["utility_spread"] > 1e-8 and
                event["rl_utility_minus_fixed_score"] < -1e-8
                for event in changed_events),
        }
    return summaries


def _run_episode(cfg, checkpoint: Path, saved_config: dict, seed: int,
                 horizons: list[int]) -> dict:
    from aircraft_sim.detect_mappo import DetectMAPPOPolicy
    from aircraft_sim.detect_sensor_diagnostics import SensorDiagnosticEnv
    from aircraft_sim.detect_shield import DetectQPShield

    env = SensorDiagnosticEnv(cfg)
    policy_cfg = evaluation_policy_config(saved_config)
    policy_cfg.direct_rl_arbitration = True
    policy = DetectMAPPOPolicy(env, DetectQPShield(cfg), policy_cfg, seed=0)
    policy.load(str(checkpoint))
    policy.config = policy_cfg
    for name in ("model", "xi_anchor_model", "policy_anchor_model"):
        getattr(policy, name).eval()
    fingerprint_before = model_fingerprint(policy)
    env.reset({"seed": seed, "k_red": cfg.K_RED})
    steps = total_return = conflicts = changed = uncontrollable = 0
    violations = interventions = 0
    violations_by_family = defaultdict(int)
    event_rows = []

    while env.terminal is None:
        observations = env.observe()
        actions, _ = policy.act_batch(observations, deterministic=True)
        if not all(action.rl_direct_arbitration for action in actions.values()):
            raise RuntimeError("frozen policy lost direct-RL arbitration mode")
        safe_actions, corrections = policy.shield.filter(env, actions)
        if not all(action.rl_direct_arbitration for action in safe_actions.values()):
            raise RuntimeError("shield changed direct-RL arbitration mode")

        phi_before = policy._track_potential()
        phi_stable_before = policy._stable_potential()
        replay_env = (deepcopy(env) if policy._may_have_arbitration_conflict(safe_actions)
                      else None)
        result = env.step(safe_actions)
        arbitration = result.info.get("arbitration", [])
        cf_by_horizon = {}
        for horizon in horizons:
            policy.config.cf_bid_horizon = horizon
            advantages = [[0.0] * 8 for _ in range(cfg.N_WINGMEN)]
            masks = [[False] * 8 for _ in range(cfg.N_WINGMEN)]
            cf = policy._counterfactual_bid_credit(
                replay_env, safe_actions, arbitration, phi_before,
                phi_stable_before, advantages, masks)
            cf_by_horizon[horizon] = {
                (int(item["tid"]), tuple(sorted(int(agent)
                                                for agent in item["candidates"]))): item
                for item in cf["event_records"]
            }
            uncontrollable += int(cf["uncontrollable_events"])
        for event in arbitration:
            conflicts += 1
            fixed_winner = baseline_winner(event)
            changed += int(int(event["winner"]) != fixed_winner)
            key = (int(event["tid"]), tuple(sorted(int(agent)
                                                   for agent in event["candidates"])))
            comparisons = {}
            for horizon in horizons:
                counterfactual = cf_by_horizon[horizon].get(key)
                if counterfactual is None:
                    comparisons[str(horizon)] = {
                        "counterfactual_available": False}
                else:
                    comparisons[str(horizon)] = {
                        "counterfactual_available": True,
                        **compare_winners(event, counterfactual),
                    }
            event_row = {
                "seed": seed, "step": steps, "tid": int(event["tid"]),
                "candidates": key[1], "rl_winner": int(event["winner"]),
                "fixed_score_winner": int(fixed_winner),
                "winners_differ": int(event["winner"]) != fixed_winner,
                "counterfactual_by_horizon": comparisons,
            }
            if "1" in comparisons:
                event_row.update(comparisons["1"])
            event_rows.append(event_row)

        steps += 1
        total_return += float(result.reward)
        violations += len(result.info["report"].violations)
        for key in result.info["report"].violations:
            violations_by_family[key.split(":", 1)[0]] += 1
        interventions += len(corrections)

    fingerprint_after = model_fingerprint(policy)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError("frozen policy weights changed during event audit")
    return {
        "seed": seed, "terminal": env.terminal, "return": total_return,
        "steps": steps, "violations": violations,
        "violations_per_step": violations / max(1, steps),
        "judged_targets": sum(env.judged.values()),
        "shield_interventions": interventions,
        "violations_by_family": dict(violations_by_family),
        "multi_candidate_events": conflicts,
        "winners_different_from_fixed_score": changed,
        "uncontrollable_events": uncontrollable,
        "event_regrets": event_rows,
        "loaded_tensor_sha256": fingerprint_after,
        "weights_unchanged": True,
    }


def _verify_reference(reference: dict, seeds: list[int],
                      tensor_hash: str | None = None) -> dict:
    if reference.get("status") != "completed_frozen_arbitration_ablation":
        raise ValueError("Reference ablation is not a completed run")
    if reference.get("seeds") != seeds:
        raise ValueError("Event audit seeds must exactly match the reference ablation")
    if tensor_hash and reference.get("loaded_tensor_sha256") != tensor_hash:
        raise ValueError("Reference ablation loaded different policy tensors")
    direct = {row["seed"]: row for row in reference["episodes"]
              if row["arbitration"] == "rl_direct"}
    if set(direct) != set(seeds):
        raise ValueError("Reference direct-RL arm is incomplete")
    if any(not row.get("weights_unchanged") or
           row.get("loaded_tensor_sha256") != reference.get("loaded_tensor_sha256")
           or not row.get("effective_policy_config", {}).get("direct_rl_arbitration")
           for row in direct.values()):
        raise ValueError("Reference RL arm has changed weights or inconsistent config")
    return direct


def _task_config():
    cfg = build_configs(4, 8, 10.0)["deterministic_beam"]
    task = asdict(cfg)
    task["W8_SCALE"] = 1.0
    return type(cfg)(**task)


def _assert_reference_match(row: dict, reference: dict) -> None:
    for key in ("terminal", "steps", "multi_candidate_events",
                "winners_different_from_fixed_score"):
        if row[key] != reference[key]:
            raise RuntimeError(f"replay differs from reference for {key} at seed {row['seed']}")
    if not np.isclose(row["return"], reference["return"], rtol=0.0, atol=1e-7):
        raise RuntimeError(f"replay return differs from reference at seed {row['seed']}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    ap.add_argument("--reference-ablation", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--counterfactual-horizons", type=int, nargs="+",
                    default=[1, 4, 8])
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--plan-only", action="store_true")
    return ap


@track_run_lifecycle
def main():
    ap = parser()
    args = ap.parse_args()
    source_path, output = args.checkpoint.resolve(), args.output.resolve()
    reference_path = args.reference_ablation.resolve()
    try:
        horizons = _normalize_horizons(args.counterfactual_horizons)
    except ValueError as exc:
        ap.error(str(exc))
    if output.exists():
        ap.error(f"output already exists: {output}")
    if args.seed_start < 0 or args.episodes < 1 or args.torch_threads < 1:
        ap.error("seed-start must be nonnegative; episodes and torch-threads positive")
    checkpoint_record = artifact_record(source_path, "frozen_checkpoint")
    reference_record = artifact_record(reference_path, "reference_ablation")
    if (not checkpoint_record["exists"] or
            checkpoint_record["sha256"] != args.expected_checkpoint_sha256):
        ap.error("checkpoint missing or differs from its declared SHA256")
    if not reference_record["exists"]:
        ap.error("reference ablation report does not exist")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    reference_direct = _verify_reference(reference, seeds)
    cfg = _task_config()
    # The reference report has gone through JSON serialization, so tuples and
    # non-string mapping keys are represented as lists/strings.  Compare the
    # canonical JSON forms rather than the in-memory dataclass representation.
    expected_task_config = json.loads(json.dumps(asdict(cfg), sort_keys=True))
    if reference.get("task_config") != expected_task_config:
        ap.error("reference ablation task configuration differs from K4/W8/A10 deterministic beam")
    reference_prov_path = Path(reference.get("provenance_sidecar", ""))
    if not reference_prov_path.is_absolute():
        reference_prov_path = reference_path.parent / reference_prov_path
    if not reference_prov_path.is_file():
        ap.error("reference ablation provenance sidecar is missing")
    reference_prov = json.loads(reference_prov_path.read_text(encoding="utf-8"))
    reference_checkpoint = next((item for item in reference_prov.get("inputs", [])
                                 if item.get("role") == "frozen_checkpoint"), None)
    reference_output = next((item for item in reference_prov.get("outputs", [])
                             if item.get("role") == "arbitration_ablation"), None)
    if (reference_prov.get("status") != "completed" or
            not reference_prov.get("source_integrity", {}).get("unchanged") or
            reference_checkpoint is None or
            reference_checkpoint.get("sha256") != checkpoint_record["sha256"] or
            reference_output is None or
            reference_output.get("sha256") != reference_record["sha256"]):
        ap.error("reference provenance does not pin the same checkpoint and intact source")
    if args.plan_only:
        print(json.dumps({"checkpoint": checkpoint_record,
                          "reference_ablation": reference_record,
                          "seeds": seeds, "episodes": len(seeds),
                          "task": {"k": 4, "stable_window": 8,
                                   "age_std_track": 10,
                                   "radar_hit_model": "deterministic_beam",
                                   "w8_scale": 1.0},
                          "counterfactual_horizons": horizons,
                          "training": False, "teacher": False,
                          "output": str(output)}, indent=2))
        return 0

    source = capture_source_snapshot(ROOT, output, [Path(__file__),
        ROOT / "_diagnose_arbitration_ablation.py",
        ROOT / "_diagnose_sensor_profile.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_env.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_mappo.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_sensor_diagnostics.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_shield.py"])
    import torch
    torch.set_num_threads(args.torch_threads)
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "provenance" / "frozen_policy.pt"
    shutil.copy2(source_path, frozen_path)
    frozen_record = artifact_record(frozen_path, "archived_frozen_checkpoint")
    if frozen_record["sha256"] != checkpoint_record["sha256"]:
        raise RuntimeError("checkpoint changed while archiving")
    checkpoint_data = torch.load(frozen_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint_data.get("config")
    policy_cfg = evaluation_policy_config(saved_config)
    for horizon in horizons:
        policy_cfg.cf_bid_horizon = horizon
        policy_cfg.validate_counterfactual_horizon()
    prov = capture_run_provenance(ROOT, output, [Path(__file__),
        ROOT / "_diagnose_arbitration_ablation.py",
        ROOT / "_diagnose_sensor_profile.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_env.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_mappo.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_sensor_diagnostics.py",
        ROOT / "Aircraft_mumt/aircraft_sim/detect_shield.py"],
        "frozen_same_snapshot_arbitration_horizon_sweep", task_config=asdict(cfg),
        algorithm_config={"saved": saved_config, "effective": asdict(policy_cfg)},
        runtime_parameters={"arguments": vars(args), "seeds": seeds,
                            "training": False, "teacher": False,
                            "counterfactual_horizons": horizons,
                            "only_treatment": "same-snapshot RL winner vs reconstructed fixed-score winner"},
        inputs=[checkpoint_record, frozen_record, reference_record],
        source_snapshot=source)
    prov_path = output / "diagnostic.provenance.json"
    result_path = output / "arbitration_event_regret.json"
    register_active_run(prov, lambda value: write_json_atomic(prov_path, value))
    write_json_atomic(prov_path, prov)

    rows = []
    for seed in seeds:
        if changed_source_files(source):
            raise RuntimeError("source changed during event audit")
        row = _run_episode(cfg, frozen_path, saved_config, seed, horizons)
        _assert_reference_match(row, reference_direct[seed])
        rows.append(row)
        write_json_atomic(output / "episode_progress.json", {"episodes": rows})
        print(f"seed={seed} conflicts={row['multi_candidate_events']} "
              f"changed={row['winners_different_from_fixed_score']} "
              f"cf_by_horizon=" + ",".join(
                  f"{h}:{sum(e['counterfactual_by_horizon'][str(h)]['counterfactual_available']
                            for e in row['event_regrets'])}"
                  for h in horizons),
              flush=True)

    fingerprints = {row["loaded_tensor_sha256"] for row in rows}
    if len(fingerprints) != 1:
        raise RuntimeError("policy tensors changed across episodes")
    if any(not row["weights_unchanged"] for row in rows):
        raise RuntimeError("frozen policy weights changed during event audit")
    by_horizon = _summarize_horizons(rows, horizons)
    disagreements = [event for row in rows for event in row["event_regrets"]
                     if event["winners_differ"]]
    report = {
        "status": "completed_frozen_same_snapshot_arbitration_horizon_sweep",
        "episodes": len(rows), "seeds": seeds, "task_config": asdict(cfg),
        "counterfactual_horizons": horizons,
        "checkpoint_sha256": checkpoint_record["sha256"],
        "reference_report_sha256": reference_record["sha256"],
        "reference_replay_matched": True,
        "loaded_tensor_sha256": next(iter(fingerprints)),
        "weights_unchanged": True, "training": False,
        "teacher_used": False, "rule_or_scenario_changes": False,
        "summary": {
            "multi_candidate_events": sum(row["multi_candidate_events"] for row in rows),
            "uncontrollable_events": sum(row["uncontrollable_events"] for row in rows),
            "winner_disagreement_events": len(disagreements),
            "winner_disagreement_episodes": len({
                int(event["seed"]) for event in disagreements}),
            "by_horizon": by_horizon,
        },
        "interpretation_limits": [
            "Counterfactual utility is a finite-horizon conditional objective, not full-episode success.",
            "The forced-winner branches share the same pre-conflict state and task actions; continuation is frozen deterministic RL.",
            "Episode seeds, not individual conflicts, are the bootstrap clusters.",
            "Utilities and regret are compared only within the same horizon."],
        "episodes_detail": rows,
        "provenance_sidecar": str(prov_path),
    }
    if changed_source_files(source):
        raise RuntimeError("source changed before event audit completion")
    write_json_atomic(result_path, report)
    (output / "episode_progress.json").unlink()
    add_output(prov, result_path, "arbitration_event_regret")
    prov["checkpoint_unchanged_during_evaluation"] = True
    print(json.dumps(report["summary"], indent=2))
    print(f"Completed: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
