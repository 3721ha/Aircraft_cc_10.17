"""Frozen-policy, single-factor radar diagnosis; never trains or injects requests.

Run from the repository root. Each seed runs both controllers in both sensor
profiles. Outputs are immutable and include full configurations, exact input
weights, source archives, per-episode measurements and paired differences.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields, replace
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

PROFILES = ("bernoulli", "deterministic_beam")
CONTROLLERS = ("rl", "heur")
CONFLICT_PAIRS = (
    "S04-M02", "S05-M02", "C01-M02", "C03-M02", "C02-M01",
    "M01-M02", "S02-M02", "S03-M02", "I02-M02", "I03-M02",
    "M04-S04", "C01-C03", "M05-E03",
)
DEFAULT_CHECKPOINT = (
    ROOT / "runs/model_optimization_v1/residual_pg_k4_w8a10_v1/curr_k4_best.pt")
DEFAULT_CHECKPOINT_SHA256 = (
    "38815f375df492ee320d9a0d4a9105efaf3944a6694165b295f646c9917aea18")


def build_configs(k: int, stable_window: int, age_std_track: float):
    from aircraft_sim.detect_config import DetectSimConfig

    if k < 1 or k > 12:
        raise ValueError("k must be in [1, 12]")
    if stable_window < 1 or not math.isfinite(age_std_track) or age_std_track <= 0:
        raise ValueError("stable window and age standard must be positive and finite")
    base = replace(DetectSimConfig(), K_RED=k, STABLE_WINDOW=stable_window,
                   AGE_STD_TRACK=age_std_track)
    configs = {profile: replace(base, RADAR_HIT_MODEL=profile)
               for profile in PROFILES}
    left, right = (asdict(configs[profile]) for profile in PROFILES)
    differences = [key for key in left if left[key] != right[key]]
    if differences != ["RADAR_HIT_MODEL"]:
        raise RuntimeError(f"not a single-factor comparison: {differences}")
    return configs


def evaluation_policy_config(saved_config: dict):
    from aircraft_sim.detect_mappo import DetectMAPPOConfig

    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint must contain its saved policy configuration")
    known = {item.name for item in fields(DetectMAPPOConfig)}
    unknown = set(saved_config) - known
    if unknown:
        raise ValueError(f"unsupported checkpoint configuration fields: {sorted(unknown)}")
    config = DetectMAPPOConfig(**saved_config)
    if not config.direct_rl_arbitration:
        raise ValueError("this protocol requires a checkpoint with direct RL arbitration")
    # Exposure is a training intervention; act_batch itself never injects it.
    config.conflict_exposure_prob = 0.0
    return config


def model_fingerprint(policy) -> str:
    """Hash all loaded policy and anchor tensors without rewriting a checkpoint."""
    digest = hashlib.sha256()
    for label in ("model", "xi_anchor_model", "policy_anchor_model"):
        for name, value in sorted(getattr(policy, label).state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{label}/{name}/{tensor.dtype}/{tuple(tensor.shape)}".encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def baseline_winner(event: dict) -> int:
    scores = {int(key): float(value) for key, value in event["scores"].items()}
    owner = event.get("previous_owner")
    winner = max(scores, key=lambda i: (scores[i], i == owner, -i))
    if (owner in scores and winner != owner
            and scores[winner] <= scores[owner] + float(event["hysteresis"])):
        return int(owner)
    return winner


def run_episode(cfg, controller: str, seed: int, checkpoint: Path,
                saved_policy_config: dict) -> dict:
    from aircraft_sim.detect_baselines import HeurPolicy
    from aircraft_sim.detect_encoder import DetectObservationEncoder
    from aircraft_sim.detect_mappo import DetectMAPPOPolicy, N_CONFLICT_FEATURES
    from aircraft_sim.detect_sensor_diagnostics import SensorDiagnosticEnv
    from aircraft_sim.detect_shield import DetectQPShield

    if controller not in CONTROLLERS:
        raise ValueError(controller)
    env = SensorDiagnosticEnv(cfg)
    shield = DetectQPShield(cfg)
    policy = None
    if controller == "rl":
        policy = DetectMAPPOPolicy(
            env, shield, evaluation_policy_config(saved_policy_config), seed=0)
        policy.load(str(checkpoint))
        for name in ("model", "xi_anchor_model", "policy_anchor_model"):
            getattr(policy, name).eval()
        fingerprint_before = model_fingerprint(policy)
        encoder = policy.encoder
    else:
        heur = HeurPolicy()
        encoder = DetectObservationEncoder(cfg)
    env.reset({"seed": seed, "k_red": cfg.K_RED})
    violations_by_family = Counter()
    pair_active_agent_steps = Counter({name: 0 for name in CONFLICT_PAIRS})
    request_outcomes = Counter()
    arbitration_modes = Counter()
    total_return = 0.0
    steps = violations = conflicts = changed_winners = interventions = 0
    while env.terminal is None:
        observations = env.observe()
        for observation in observations.values():
            vector = encoder.encode(observation)
            for name, pressure in zip(CONFLICT_PAIRS, vector[-N_CONFLICT_FEATURES:]):
                pair_active_agent_steps[name] += int(pressure > 0.1)
        if policy is None:
            actions = heur(env)
        else:
            actions, _ = policy.act_batch(observations, deterministic=True)
            if not all(action.rl_direct_arbitration for action in actions.values()):
                raise RuntimeError("RL direct arbitration flag lost before shield")
        safe_actions, corrections = shield.filter(env, actions)
        if policy is not None and not all(
                action.rl_direct_arbitration for action in safe_actions.values()):
            raise RuntimeError("shield removed RL direct arbitration flag")
        interventions += len(corrections)
        result = env.step(safe_actions)
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
            arbitration_modes[mode] += 1
            if policy is not None and mode != "rl_direct":
                raise RuntimeError("an RL conflict fell back to fixed-score arbitration")
            changed_winners += int(int(event["winner"]) != baseline_winner(event))
    row = {
        "sensor_profile": cfg.RADAR_HIT_MODEL, "controller": controller,
        "seed": seed, "k": cfg.K_RED, "terminal": env.terminal,
        "return": total_return, "steps": steps, "violations": violations,
        "violations_per_step": violations / max(1, steps),
        "judged_targets": sum(env.judged.values()),
        "shield_interventions": interventions,
        "violations_by_family": dict(violations_by_family),
        "active_agent_steps_by_conflict_pair": dict(pair_active_agent_steps),
        "multi_candidate_events": conflicts,
        "arbitration_modes": dict(arbitration_modes),
        "winners_different_from_fixed_score": changed_winners,
        "request_outcomes": dict(request_outcomes),
        "sensor_diagnostics": env.sensor_diagnostics(),
        "effective_task_config_sha256": canonical_sha256(asdict(env.cfg)),
    }
    if policy is not None:
        after = model_fingerprint(policy)
        if fingerprint_before != after:
            raise RuntimeError("frozen evaluation mutated policy or anchor weights")
        row["effective_policy_config"] = asdict(policy.config)
        row["checkpoint_migration"] = policy.checkpoint_migration_record
        row["loaded_tensor_sha256"] = after
        row["weights_unchanged"] = True
    return row


def wilson_interval(successes: int, episodes: int) -> list[float]:
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    centre = (p + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize zero episodes")
    count = len(rows)
    steps = sum(row["steps"] for row in rows)
    violations = sum(row["violations"] for row in rows)
    successes = sum(row["terminal"] == "success" for row in rows)
    radar = {scope: {key: sum(row["sensor_diagnostics"]["radar"][scope][key]
                              for row in rows)
                     for key in ("opportunities", "hits", "misses")}
             for scope in ("all_targets", "unfinished_targets")}
    for counters in radar.values():
        counters["hit_fraction_per_opportunity"] = (
            counters["hits"] / counters["opportunities"]
            if counters["opportunities"] else None)
    reset_causes = Counter()
    gate_causes = Counter()
    families = Counter()
    pairs = Counter({name: 0 for name in CONFLICT_PAIRS})
    for row in rows:
        verdict = row["sensor_diagnostics"]["verdict"]
        reset_causes.update(verdict["reset_causes"])
        gate_causes.update(verdict["owner_gate_failure_causes"])
        families.update(row["violations_by_family"])
        pairs.update(row["active_agent_steps_by_conflict_pair"])
    return {
        "episodes": count, "successes": successes, "success_rate": successes / count,
        "success_rate_wilson_95": wilson_interval(successes, count),
        "mean_return": sum(row["return"] for row in rows) / count,
        "total_steps": steps, "total_violations": violations,
        "violations_per_step": violations / max(1, steps),
        "mean_episode_violations_per_step": sum(row["violations_per_step"] for row in rows) / count,
        "mean_judged_targets": sum(row["judged_targets"] for row in rows) / count,
        "radar_target_dwell_counts": radar,
        "stable_progress_resets": sum(row["sensor_diagnostics"]["verdict"]["stable_progress_resets"]
                                      for row in rows),
        "stable_reset_causes": dict(reset_causes),
        "owner_gate_failure_target_step_causes": dict(gate_causes),
        "unowned_tracking_target_steps": sum(
            row["sensor_diagnostics"]["verdict"]["unowned_tracking_target_steps"] for row in rows),
        "violations_by_family": dict(families),
        "active_agent_steps_by_conflict_pair": dict(pairs),
        "conflict_pairs_observed": sum(value > 0 for value in pairs.values()),
        "multi_candidate_events": sum(row["multi_candidate_events"] for row in rows),
        "winners_different_from_fixed_score": sum(
            row["winners_different_from_fixed_score"] for row in rows),
    }


def paired_difference(left: list[dict], right: list[dict]) -> dict:
    """Report right minus left; success change is paired by episode seed."""
    left_by_seed = {row["seed"]: row for row in left}
    right_by_seed = {row["seed"]: row for row in right}
    if (len(left_by_seed) != len(left) or len(right_by_seed) != len(right)
            or left_by_seed.keys() != right_by_seed.keys()):
        raise ValueError("paired comparison requires identical unique seeds")
    a, b = summarize(left), summarize(right)
    return {
        "direction": "right_minus_left", "episodes": len(left),
        "success_rate_delta": b["success_rate"] - a["success_rate"],
        "mean_return_delta": b["mean_return"] - a["mean_return"],
        "violations_per_step_delta": b["violations_per_step"] - a["violations_per_step"],
        "right_only_successes": sum(
            row["terminal"] != "success" and right_by_seed[seed]["terminal"] == "success"
            for seed, row in left_by_seed.items()),
        "left_only_successes": sum(
            row["terminal"] == "success" and right_by_seed[seed]["terminal"] != "success"
            for seed, row in left_by_seed.items()),
    }


def make_report(rows: list[dict]) -> dict:
    grouped = {(profile, controller): [row for row in rows
               if row["sensor_profile"] == profile and row["controller"] == controller]
               for profile in PROFILES for controller in CONTROLLERS}
    fingerprints = {row["loaded_tensor_sha256"] for row in rows if row["controller"] == "rl"}
    if len(fingerprints) != 1:
        raise RuntimeError("different effective RL weights used across conditions")
    return {
        "summaries": {profile: {controller: summarize(grouped[profile, controller])
                                for controller in CONTROLLERS} for profile in PROFILES},
        "sensor_effect_deterministic_minus_bernoulli": {
            controller: paired_difference(grouped["bernoulli", controller],
                                          grouped["deterministic_beam", controller])
            for controller in CONTROLLERS},
        "rl_minus_heur": {profile: paired_difference(grouped[profile, "heur"],
                                                     grouped[profile, "rl"])
                          for profile in PROFILES},
        "same_loaded_rl_tensors_all_conditions": True,
        "loaded_tensor_sha256": next(iter(fingerprints)),
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    ap.add_argument("--output", type=Path, required=True, help="new directory; existing paths are refused")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--stable-window", type=int, default=8)
    ap.add_argument("--age-std-track", type=float, default=10.0)
    ap.add_argument("--seed-start", type=int, default=205000)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--torch-threads", type=int, default=1)
    return ap


@track_run_lifecycle
def main() -> None:
    ap = parser()
    args = ap.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        ap.error(f"output already exists (will not overwrite): {output}")
    if args.episodes < 1 or args.seed_start < 0 or args.torch_threads < 1:
        ap.error("episodes/torch-threads must be positive; seed-start must be nonnegative")
    if not 1 <= args.k <= 12 or args.stable_window < 1:
        ap.error("k must be in [1, 12]; stable-window must be positive")
    if not math.isfinite(args.age_std_track) or args.age_std_track <= 0:
        ap.error("age-std-track must be positive and finite")
    if not args.expected_checkpoint_sha256 or len(args.expected_checkpoint_sha256) != 64:
        ap.error("an explicit 64-character expected checkpoint SHA256 is required")
    input_record = artifact_record(checkpoint, "policy_checkpoint")
    if not input_record["exists"] or input_record["sha256"] != args.expected_checkpoint_sha256:
        ap.error("checkpoint missing or SHA256 differs from the declared frozen model")
    # Capture before importing simulator code. No module-import output side effects.
    source = capture_source_snapshot(ROOT, output, [Path(__file__)])
    import torch
    torch.set_num_threads(args.torch_threads)
    configs = build_configs(args.k, args.stable_window, args.age_std_track)
    snapshot_checkpoint = output / "provenance" / "frozen_policy.pt"
    shutil.copy2(checkpoint, snapshot_checkpoint)
    archived = artifact_record(snapshot_checkpoint, "policy_checkpoint_snapshot")
    if archived["sha256"] != input_record["sha256"]:
        raise RuntimeError("checkpoint changed while archiving")
    data = torch.load(snapshot_checkpoint, map_location="cpu", weights_only=False)
    saved_config = data.get("config")
    policy_config = evaluation_policy_config(saved_config)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    record = capture_run_provenance(
        ROOT, output, [Path(__file__)], "frozen_sensor_profile_diagnostic",
        task_config={profile: asdict(configs[profile]) for profile in PROFILES},
        algorithm_config={"saved": saved_config, "effective": asdict(policy_config)},
        runtime_parameters={"arguments": vars(args), "seeds": seeds,
                            "training": False, "conflict_injection": False,
                            "checkpoint_repair": False},
        inputs=[input_record, archived], source_snapshot=source)
    provenance_path = output / "diagnostic.provenance.json"
    progress_path = output / "episode_progress.json"
    result_path = output / "sensor_diagnostic.json"
    register_active_run(record, lambda value: write_json_atomic(provenance_path, value))
    write_json_atomic(provenance_path, record)
    rows = []
    for seed in seeds:
        for profile in PROFILES:
            for controller in CONTROLLERS:
                drift = changed_source_files(source)
                if drift:
                    raise RuntimeError(f"source changed during diagnosis: {drift}")
                row = run_episode(configs[profile], controller, seed,
                                  snapshot_checkpoint, saved_config)
                rows.append(row)
                write_json_atomic(progress_path, {"status": "incomplete", "episodes": rows})
                print(f"seed={seed} {profile}/{controller}: {row['terminal']} "
                      f"return={row['return']:.1f} violations/step={row['violations_per_step']:.4f}", flush=True)
    for path in (checkpoint, snapshot_checkpoint):
        if artifact_record(path, "after")["sha256"] != input_record["sha256"]:
            raise RuntimeError(f"checkpoint changed during evaluation: {path}")
    drift = changed_source_files(source)
    if drift:
        raise RuntimeError(f"source changed during diagnosis: {drift}")
    report = {
        "status": "completed_diagnostic", "scope": "frozen model response; no training",
        "checkpoint": input_record, "seeds": seeds,
        "task_configs": {profile: asdict(configs[profile]) for profile in PROFILES},
        "task_config_sha256": {profile: canonical_sha256(asdict(configs[profile]))
                               for profile in PROFILES},
        "single_changed_config_field": "RADAR_HIT_MODEL",
        "units": {"radar": "target-dwell", "gate_failure": "target-step",
                  "conflict_pair_activity": "agent-step, encoded pressure > 0.1",
                  "violations_per_step": "sum violations / sum environment steps"},
        "limitations": [
            "This diagnostic does not measure retraining benefit or prove learned arbitration utility.",
            "One draw per geometrically eligible target is retained in both profiles; changed actions may still change later shared radar/link RNG alignment.",
            "Common seeds fix initial conditions and the separately seeded red process; closed-loop observations/actions differ across conditions.",
            "Gate and reset counts are descriptive, not a causal attribution of every task failure.",
            "Conflict-pair proxies describe coverage; a changed arbitration winner alone does not establish benefit.",
        ],
        **make_report(rows), "episodes": rows,
        "provenance_sidecar": str(provenance_path),
    }
    write_json_atomic(result_path, report)
    # The completed report supersedes the redundant resumeless progress file.
    progress_path.unlink()
    add_output(record, result_path, "sensor_diagnostic")
    record["checkpoint_unchanged_during_evaluation"] = True
    record["same_loaded_rl_tensors_all_conditions"] = True
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2), flush=True)
    print(f"Completed: {result_path}", flush=True)


if __name__ == "__main__":
    main()
