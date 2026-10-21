"""Independent development check of fixed initial RL, update-8 RL and HEUR.

No training, checkpoint selection, forced conflict requests or final test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _experiment_provenance import (
    add_output, artifact_record, canonical_sha256, capture_run_provenance,
    capture_source_snapshot, changed_source_files, register_active_run,
    track_run_lifecycle, write_json_atomic,
)
from _diagnose_sensor_profile import (
    build_configs, evaluation_policy_config, paired_difference,
    run_episode, summarize,
)

DEFAULT_RUN = ROOT / "runs/model_optimization_v1/sensor_joint_adapt_k4_w8a10_v1"
EXPECTED_LOG = "9018f87365c90d9ee7cdc627d83480a28b6c069cad7ae29c8222bbac543c7475"
EXPECTED_PROVENANCE = "e7f4ced938b4fcaeeb6f998cf1d93b4dbb62d42a1462f516c0b58ba57a7989c5"
EXPECTED_INITIAL = "ef86e5734d2a6466694371a0931386d7f1937355f279eed28f1560dcae27499b"
EXPECTED_BEST = "b30c53060f0374b204ca95fd6797bbf9a7ec5b1fb00d6d628e2afe5998f0e801"
LABELS = ("initial_rl", "best_rl", "heur")
RESERVED_TEST = set(range(208000, 208030))
PREVIOUSLY_USED = {204900, *range(205000, 205030), *range(162000, 162030),
                   *range(209900, 209904), 209990, 209991}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=210000)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--plan-only", action="store_true")
    return ap


def verified_artifact(path: Path, expected: str, role: str) -> dict:
    record = artifact_record(path, role)
    if not record["exists"] or record["sha256"] != expected:
        raise ValueError(f"{role} missing or SHA256 differs: {path}")
    return record


def rank(summary):
    return (summary["successes"], summary["mean_return"],
            -summary["violations_per_step"])


def verify_source_snapshot(provenance):
    source = provenance["effective_source"]
    files = source["files"]
    if not files or canonical_sha256(files) != source["manifest_sha256"]:
        raise ValueError("source snapshot manifest differs")
    snapshot = Path(source["snapshot_path"])
    if not snapshot.is_absolute():
        snapshot = ROOT / snapshot
    artifact = verified_artifact(snapshot, source["snapshot_sha256"], "source_training_snapshot")
    with zipfile.ZipFile(snapshot) as archive:
        names = [item["path"] for item in files]
        if len(names) != len(set(names)) or sorted(archive.namelist()) != sorted(names):
            raise ValueError("source snapshot members differ from manifest")
        for item in files:
            content = archive.read(item["path"])
            if (hashlib.sha256(content).hexdigest() != item["sha256"]
                    or len(content) != item["size_bytes"]):
                raise ValueError(f"source snapshot member differs: {item['path']}")
    drift = changed_source_files(source)
    if drift:
        raise ValueError(f"source differs from the audited training run: {drift}")
    return artifact


def checkpoint_fingerprint(data):
    """Use the exact runtime fingerprint format on all saved policy/anchor tensors."""
    digest = hashlib.sha256()
    for label in ("model", "xi_anchor_model", "policy_anchor_model"):
        if not isinstance(data.get(label), dict) or not data[label]:
            raise ValueError(f"fixed checkpoint lacks complete {label} state")
        for name, value in sorted(data[label].items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{label}/{name}/{tensor.dtype}/{tuple(tensor.shape)}".encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def preflight(args):
    if args.output.resolve().exists():
        raise ValueError(f"output already exists: {args.output.resolve()}")
    if args.episodes < 1 or args.seed_start < 0 or args.torch_threads < 1:
        raise ValueError("episodes/torch-threads must be positive; seed-start nonnegative")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    source = args.source_run.resolve()
    log_artifact = verified_artifact(source / "adaptation_log.json", EXPECTED_LOG, "source_log")
    provenance_artifact = verified_artifact(
        source / "adaptation.provenance.json", EXPECTED_PROVENANCE, "source_provenance")
    log = json.loads((source / "adaptation_log.json").read_text(encoding="utf-8"))
    provenance = json.loads((source / "adaptation.provenance.json").read_text(encoding="utf-8"))
    if (log["status"] != "completed_short_adaptation"
            or provenance["status"] != "completed"
            or not provenance["source_integrity"]["unchanged"]
            or provenance["reserved_test_run"]):
        raise ValueError("source run is not a completed, intact adaptation with reserved test unused")
    if (log["best"]["update"] != 8 or provenance["selected_best_update"] != 8
            or max(log["validations"], key=lambda row: rank(row["summary"]))["update"] != 8
            or log["best"]["validation_summary"] != next(
                row["summary"] for row in log["validations"] if row["update"] == 8)):
        raise ValueError("source selection does not identify fixed update 8")
    declared = {item["role"]: item["sha256"] for item in provenance["outputs"]}
    if any(declared[role] != expected for role, expected in (
            ("adaptation_log", EXPECTED_LOG), ("effective_initial_checkpoint", EXPECTED_INITIAL),
            ("selected_best_checkpoint", EXPECTED_BEST))):
        raise ValueError("source provenance output declarations differ")
    training_snapshot = verify_source_snapshot(provenance)
    expected_task = {"K_RED": 4, "STABLE_WINDOW": 8, "AGE_STD_TRACK": 10.0,
                     "RADAR_HIT_MODEL": "deterministic_beam", "W8_SCALE": 1.0}
    if any(log["task_config"][key] != value for key, value in expected_task.items()):
        raise ValueError("source task is not deterministic K4/W8/A10 with full penalties")
    if canonical_sha256(log["task_config"]) != log["task_config_sha256"]:
        raise ValueError("source task configuration hash differs")
    initial = verified_artifact(source / "initial.pt", EXPECTED_INITIAL, "initial_rl")
    best = verified_artifact(source / "best.pt", EXPECTED_BEST, "best_rl")
    update8 = verified_artifact(source / "checkpoints/update_0008.pt", EXPECTED_BEST, "selected_update8")
    if (log["initial_checkpoint"]["sha256"] != EXPECTED_INITIAL
            or log["best"]["checkpoint"]["sha256"] != EXPECTED_BEST
            or log["best"]["selected_checkpoint"]["sha256"] != EXPECTED_BEST):
        raise ValueError("source checkpoint declarations differ from fixed initial/update8")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    consumed = set(log["protocol"]["training_seeds"]) | set(log["protocol"]["validation_seeds"])
    consumed |= {episode["seed"] for row in log["training"] for episode in row["episodes"]}
    consumed |= {episode["seed"] for row in log["validations"] for episode in row["episodes"]}
    if set(seeds) & (consumed | RESERVED_TEST | PREVIOUSLY_USED):
        raise ValueError("follow-up seeds overlap training, selection, consumed diagnostics or reserved test")
    return {
        "source_log": log,
        "inputs": [log_artifact, provenance_artifact, initial, best, update8, training_snapshot],
        "protocol": {
            "scope": "independent development follow-up; not a final test",
            "seeds": seeds, "conditions": list(LABELS), "best_update_fixed": 8,
            "task_config": log["task_config"], "task_config_sha256": log["task_config_sha256"],
            "training": False, "checkpoint_selection": False,
            "conflict_injection": False, "reserved_test_run": False,
            "reserved_test_seeds_not_run": sorted(RESERVED_TEST),
            "screening": {
                "adaptation": "best success >= initial, mean return > initial, pooled violations/step <= initial",
                "heur_comparison": "best success >= HEUR, mean return > HEUR, pooled violations/step <= HEUR",
                "meaning": "descriptive joint screen; not proof of significance, final acceptance or arbitration benefit",
            },
        },
    }


def paired_statistics(left, right, samples=10000):
    """Paired episodes; return differences and pooled-rate bootstrap intervals."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    difference = paired_difference(left, right)
    left_by_seed = {row["seed"]: row for row in left}
    right_by_seed = {row["seed"]: row for row in right}
    seeds = sorted(left_by_seed)
    count = len(seeds)
    wins, losses = difference["right_only_successes"], difference["left_only_successes"]
    discordant = wins + losses
    p = (min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
             / 2**discordant) if discordant else 1.0)
    pairs = [(left_by_seed[seed], right_by_seed[seed]) for seed in seeds]
    rng = random.Random(20261027)
    draws = {name: [] for name in ("mean_return_delta", "success_rate_delta",
                                  "violations_per_step_delta", "mean_episode_violations_per_step_delta")}
    for _ in range(samples):
        selected = [pairs[rng.randrange(count)] for _ in range(count)]
        draws["mean_return_delta"].append(sum(b["return"] - a["return"] for a, b in selected) / count)
        draws["success_rate_delta"].append(sum(
            int(b["terminal"] == "success") - int(a["terminal"] == "success")
            for a, b in selected) / count)
        draws["violations_per_step_delta"].append(
            sum(b["violations"] for a, b in selected) / sum(b["steps"] for a, b in selected)
            - sum(a["violations"] for a, b in selected) / sum(a["steps"] for a, b in selected))
        draws["mean_episode_violations_per_step_delta"].append(sum(
            b["violations_per_step"] - a["violations_per_step"] for a, b in selected) / count)
    intervals = {}
    for name, values in draws.items():
        values.sort()
        intervals[name] = [values[int((samples - 1) * .025)], values[int((samples - 1) * .975)]]
    return {
        **difference, "exact_paired_success_two_sided_p": p,
        "paired_percentile_bootstrap_95": intervals,
        "bootstrap_samples": samples, "bootstrap_seed": 20261027,
        "mean_episode_violations_per_step_delta": sum(
            b["violations_per_step"] - a["violations_per_step"] for a, b in pairs) / count,
        "per_seed_differences": [
            {"seed": seed, "return_delta": b["return"] - a["return"],
             "violations_per_step_delta": b["violations_per_step"] - a["violations_per_step"],
             "success_delta": int(b["terminal"] == "success") - int(a["terminal"] == "success")}
            for seed, (a, b) in zip(seeds, pairs)],
        "success_flip_seeds": {
            "right_only_successes": [seed for seed in seeds if left_by_seed[seed]["terminal"] != "success"
                                     and right_by_seed[seed]["terminal"] == "success"],
            "left_only_successes": [seed for seed in seeds if left_by_seed[seed]["terminal"] == "success"
                                    and right_by_seed[seed]["terminal"] != "success"],
        },
        "limits": "Intervals and exact p describe fixed checkpoints on these seeds; no multiplicity adjustment or arbitration attribution.",
    }


def make_report(rows, seeds, bootstrap_samples):
    expected = {(label, seed) for label in LABELS for seed in seeds}
    observed = [(row["model"], row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds) or len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("requires exactly one episode per condition and unique matching seed")
    grouped = {label: [row for row in rows if row["model"] == label] for label in LABELS}
    summaries = {label: summarize(group) for label, group in grouped.items()}
    fingerprints = {}
    for label in LABELS[:2]:
        group = grouped[label]
        values = {row["loaded_tensor_sha256"] for row in group}
        if len(values) != 1 or not all(row["weights_unchanged"] for row in group):
            raise ValueError(f"{label} evaluation used changing policy/anchor weights")
        fingerprints[label] = next(iter(values))
    def screen(reference):
        best, base = summaries["best_rl"], summaries[reference]
        conditions = {
            "success_not_lower": best["successes"] >= base["successes"],
            "mean_return_higher": best["mean_return"] > base["mean_return"],
            "pooled_violations_not_higher": best["violations_per_step"] <= base["violations_per_step"],
        }
        return {"conditions": conditions, "passed": all(conditions.values()),
                "scope": "descriptive screen only; no automatic long-training authorization or final acceptance"}
    return {
        "summaries": summaries,
        "best_minus_initial": paired_statistics(grouped["initial_rl"], grouped["best_rl"], bootstrap_samples),
        "best_minus_heur": paired_statistics(grouped["heur"], grouped["best_rl"], bootstrap_samples),
        "initial_minus_heur": paired_statistics(grouped["heur"], grouped["initial_rl"], bootstrap_samples),
        "descriptive_screen_against_initial": screen("initial_rl"),
        "descriptive_screen_against_heur": screen("heur"),
        "loaded_tensor_sha256": fingerprints,
    }


def require_unchanged(source, inputs):
    drift = changed_source_files(source)
    if drift:
        raise RuntimeError(f"evaluation source changed: {drift}")
    for item in inputs:
        current = artifact_record(item["path"], item["role"])
        if current["sha256"] != item["sha256"]:
            raise RuntimeError(f"evaluation input changed: {item['path']}")


@track_run_lifecycle
def main():
    ap = parser()
    args = ap.parse_args()
    try:
        plan = preflight(args)
    except (ValueError, KeyError, TypeError, StopIteration, OSError, zipfile.BadZipFile) as error:
        ap.error(str(error))
    if args.plan_only:
        print(json.dumps(plan["protocol"], ensure_ascii=False, indent=2))
        return
    output = args.output.resolve()
    source = capture_source_snapshot(
        ROOT, output, [Path(__file__), ROOT / "_diagnose_sensor_profile.py"])
    log = plan["source_log"]
    record = capture_run_provenance(
        ROOT, output, [Path(__file__), ROOT / "_diagnose_sensor_profile.py"],
        "fixed_sensor_adaptation_independent_development_followup",
        log["task_config"], algorithm_config={"source_saved_config": log["policy_config"]},
        runtime_parameters={"arguments": vars(args), "protocol": plan["protocol"]},
        inputs=list(plan["inputs"]), source_snapshot=source)
    provenance_path = output / "followup.provenance.json"
    result_path = output / "followup_result.json"
    progress_path = output / "episode_progress.json"
    register_active_run(record, lambda value: write_json_atomic(provenance_path, value))
    write_json_atomic(provenance_path, record)
    archived = {}
    for item in plan["inputs"]:
        original = Path(item["path"])
        target = output / "provenance" / f"{item['role']}_{original.name}"
        shutil.copy2(original, target)
        archived_item = verified_artifact(target, item["sha256"], f"archived_{item['role']}")
        record["inputs"].append(archived_item)
        archived[item["role"]] = target
    write_json_atomic(provenance_path, record)
    import torch
    torch.set_num_threads(args.torch_threads)
    cfg = build_configs(4, 8, 10.0)["deterministic_beam"]
    if canonical_sha256(asdict(cfg)) != log["task_config_sha256"]:
        raise RuntimeError("current full task configuration differs from the source experiment")
    saved_configs, expected_fingerprints = {}, {}
    for label in LABELS[:2]:
        data = torch.load(archived[label], map_location="cpu", weights_only=False)
        saved_configs[label] = data["config"]
        if saved_configs[label] != log["policy_config"]:
            raise RuntimeError(f"{label} saved policy configuration differs from source")
        evaluation_policy_config(data["config"])
        expected_fingerprints[label] = checkpoint_fingerprint(data)
    record["effective_evaluation_policy_configs"] = {
        label: asdict(evaluation_policy_config(config)) for label, config in saved_configs.items()}
    record["expected_loaded_tensor_sha256"] = expected_fingerprints
    write_json_atomic(provenance_path, record)
    rows = []
    seeds = plan["protocol"]["seeds"]
    for seed in seeds:
        for label in LABELS:
            require_unchanged(source, record["inputs"])
            controller = "heur" if label == "heur" else "rl"
            weight_label = "initial_rl" if label == "heur" else label
            row = run_episode(cfg, controller, seed, archived[weight_label], saved_configs[weight_label])
            if row["effective_task_config_sha256"] != log["task_config_sha256"]:
                raise RuntimeError("episode used a different task configuration")
            if any(row["sensor_diagnostics"]["radar"][scope]["misses"]
                   for scope in ("all_targets", "unfinished_targets")):
                raise RuntimeError("deterministic-beam condition produced random misses")
            if controller == "rl" and set(row["arbitration_modes"]) - {"rl_direct"}:
                raise RuntimeError("RL arbitration used a fixed-score fallback")
            if controller == "rl":
                if row["loaded_tensor_sha256"] != expected_fingerprints[label]:
                    raise RuntimeError(f"{label} loaded tensors differ from the fixed saved checkpoint")
                if row["effective_policy_config"] != record["effective_evaluation_policy_configs"][label]:
                    raise RuntimeError(f"{label} effective policy configuration differs")
                if not row["weights_unchanged"]:
                    raise RuntimeError(f"{label} weights changed during evaluation")
            rows.append({**row, "model": label})
            write_json_atomic(progress_path, {"status": "incomplete", "episodes": rows})
            print(f"seed={seed} {label}: {row['terminal']} return={row['return']:.1f} "
                  f"violations/step={row['violations_per_step']:.4f}", flush=True)
    require_unchanged(source, record["inputs"])
    report = {
        "status": "completed_independent_development_followup",
        "protocol": plan["protocol"], "task_config": asdict(cfg),
        "task_config_sha256": log["task_config_sha256"],
        "checkpoints": {item["role"]: item for item in plan["inputs"] if item["role"] in LABELS},
        **make_report(rows, seeds, args.bootstrap_samples), "episodes": rows,
        "limitations": [
            "These independent development seeds are now consumed; they are not a new final test pool after inspection.",
            "The fixed initial and update-8 checkpoints are compared without selecting a new model.",
            "Coverage proxies and changed winners do not demonstrate causal arbitration benefit.",
            "Paired seeds share the initial scenario; subsequent stochastic trajectories can diverge after policy actions differ.",
        ],
        "provenance_sidecar": str(provenance_path),
    }
    write_json_atomic(result_path, report)
    progress_path.unlink()
    add_output(record, result_path, "independent_followup_result")
    record["inputs_unchanged"] = True
    record["reserved_test_run"] = False
    record["training_run"] = False
    record["checkpoint_selection_run"] = False
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2), flush=True)
    print(f"Completed: {result_path}", flush=True)


if __name__ == "__main__":
    main()
