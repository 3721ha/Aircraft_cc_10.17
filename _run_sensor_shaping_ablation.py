"""Run the isolated 2x2 training-only sensor-shaping screening protocol.

Each arm starts from the same immutable RL checkpoint and uses identical
training, validation, and optimizer seeds.  The child runner owns checkpoint
and provenance creation; this launcher only records the suite manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "_train_sensor_adaptation.py"
RESERVED_TEST_SEEDS = set(range(208000, 208030))
CONSUMED_SEEDS = {
    *range(162000, 162030), *range(205000, 205030), *range(206000, 206096),
    *range(207000, 207012), *range(209900, 209904), 209990, 209991,
    *range(210000, 210030), *range(219900, 219911), *range(250000, 250006),
}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--expected-checkpoint-sha256", required=True)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--episodes-per-iter", type=int, default=8)
    ap.add_argument("--validation-every", type=int, default=2)
    ap.add_argument("--train-seed-start", type=int, default=260000)
    ap.add_argument("--validation-seed-start", type=int, default=261000)
    ap.add_argument("--validation-episodes", type=int, default=10)
    ap.add_argument("--evaluation-seed-start", type=int, default=262100)
    ap.add_argument("--evaluation-episodes", type=int, default=10)
    ap.add_argument("--optimizer-seed", type=int, default=20261021)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--plan-only", action="store_true")
    return ap


def make_arms(root: Path):
    return [
        ("control", 0.5, "legacy"),
        ("geometry_only", 1.0, "legacy"),
        ("alignment_only", 0.5, "c04_usable_sector"),
        ("both", 1.0, "c04_usable_sector"),
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child_args(args, output: Path, geo: float, alignment: str):
    return [
        sys.executable, str(RUNNER), "--output", str(output),
        "--checkpoint", str(args.checkpoint),
        "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
        "--iterations", str(args.iterations),
        "--episodes-per-iter", str(args.episodes_per_iter),
        "--validation-every", str(args.validation_every),
        "--train-seed-start", str(args.train_seed_start),
        "--validation-seed-start", str(args.validation_seed_start),
        "--validation-episodes", str(args.validation_episodes),
        "--optimizer-seed", str(args.optimizer_seed),
        "--learning-rate", str(args.learning_rate),
        "--epochs", str(args.epochs), "--torch-threads", str(args.torch_threads),
        "--geo-shaping-range-fraction", str(geo),
        "--stable-alignment-geometry", alignment,
    ]


def protocol(args):
    suite = args.suite_output.resolve()
    checkpoint = args.checkpoint.resolve()
    if suite.exists():
        raise ValueError(f"suite output already exists: {suite}")
    arms = []
    training_seeds = set(range(
        args.train_seed_start,
        args.train_seed_start + args.iterations * args.episodes_per_iter))
    validation_seeds = set(range(
        args.validation_seed_start,
        args.validation_seed_start + args.validation_episodes))
    evaluation_seeds = set(range(
        args.evaluation_seed_start,
        args.evaluation_seed_start + args.evaluation_episodes))
    all_seeds = training_seeds | validation_seeds | evaluation_seeds
    if len(all_seeds) != (len(training_seeds) + len(validation_seeds)
                          + len(evaluation_seeds)):
        raise ValueError("training, validation, and independent evaluation seeds overlap")
    if all_seeds & (RESERVED_TEST_SEEDS | CONSUMED_SEEDS):
        raise ValueError("suite seeds overlap reserved test or consumed seed pools")
    for name, geo, alignment in make_arms(suite):
        output = suite / name
        if output.exists():
            raise ValueError(f"arm output already exists: {output}")
        command = child_args(args, output, geo, alignment)
        preflight = subprocess.run(
            [*command, "--plan-only"], cwd=ROOT, check=False,
            capture_output=True, text=True)
        if preflight.returncode != 0:
            raise ValueError(
                f"arm {name} preflight failed: {preflight.stderr.strip()}")
        arms.append({
            "name": name, "output": str(output),
            "geo_shaping_range_fraction": geo,
            "stable_alignment_geometry": alignment,
            "argv": command,
            "child_preflight": json.loads(preflight.stdout),
        })
    return {
        "protocol": "sensor_shaping_2x2_screening",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "launcher_sha256": sha256(Path(__file__).resolve()),
        "training_runner_sha256": sha256(RUNNER),
        "common_training": {
            "iterations": args.iterations,
            "episodes_per_iter": args.episodes_per_iter,
            "validation_every": args.validation_every,
            "train_seed_start": args.train_seed_start,
            "validation_seed_start": args.validation_seed_start,
            "validation_episodes": args.validation_episodes,
            "evaluation_seed_start": args.evaluation_seed_start,
            "evaluation_episodes": args.evaluation_episodes,
            "optimizer_seed": args.optimizer_seed,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "torch_threads": args.torch_threads,
        },
        "reserved_test_seeds_not_run": list(range(208000, 208030)),
        "independent_evaluation_seeds": sorted(evaluation_seeds),
        "primary_comparison": "initial RL, each fixed final update, and HEUR on independent evaluation seeds; validation-selected best is exploratory",
        "arms": arms,
        "no_training_teacher": True,
        "direct_rl_arbitration": True,
        "no_final_test": True,
    }


def main():
    args = parser().parse_args()
    plan = protocol(args)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    suite = args.suite_output.resolve()
    suite.mkdir(parents=True, exist_ok=False)
    manifest_path = suite / "suite_manifest.json"
    plan["status"] = "running"
    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    completed = []
    try:
        for arm in plan["arms"]:
            result = subprocess.run(arm["argv"], cwd=ROOT, check=False)
            arm["returncode"] = result.returncode
            completed.append(arm["name"])
            if result.returncode != 0:
                raise RuntimeError(f"arm {arm['name']} exited with code {result.returncode}")
        evaluation = evaluate_final_updates(args, plan, suite)
        evaluation_path = suite / "independent_evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        plan["independent_evaluation"] = str(evaluation_path)
        plan["status"] = "completed"
    except Exception as exc:
        plan["status"] = "failed"
        plan["error"] = str(exc)
        raise
    finally:
        plan["completed_arms"] = completed
        manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed 2x2 sensor-shaping suite: {manifest_path}")


def evaluate_final_updates(args, plan, suite: Path):
    import torch
    from _diagnose_sensor_profile import (
        build_configs, paired_difference, run_episode, summarize,
    )

    cfg = build_configs(4, 8, 10.0)["deterministic_beam"]
    seeds = plan["independent_evaluation_seeds"]
    arms = {}
    reference_config = None
    for arm in plan["arms"]:
        run_dir = Path(arm["output"])
        log_path = run_dir / "adaptation_log.json"
        log = json.loads(log_path.read_text(encoding="utf-8"))
        if log.get("status") != "completed_short_adaptation":
            raise RuntimeError(f"arm {arm['name']} did not complete")
        if (log["protocol"]["source_checkpoint"]["sha256"]
                != plan["expected_checkpoint_sha256"]):
            raise RuntimeError(f"arm {arm['name']} source checkpoint hash differs")
        policy_config = log["policy_config"]
        if (policy_config["geo_shaping_range_fraction"]
                != arm["geo_shaping_range_fraction"]
                or policy_config["stable_alignment_geometry"]
                != arm["stable_alignment_geometry"]):
            raise RuntimeError(f"arm {arm['name']} effective shaping config differs")
        if reference_config is None:
            reference_config = policy_config
        elif policy_config != reference_config:
            differing = [key for key in policy_config
                         if policy_config[key] != reference_config[key]
                         and key not in {"geo_shaping_range_fraction",
                                         "stable_alignment_geometry"}]
            if differing:
                raise RuntimeError(f"arms differ beyond the planned factors: {differing}")
        final_update = args.iterations
        checkpoint = run_dir / "checkpoints" / f"update_{final_update:04d}.pt"
        record = log["training"][-1]["checkpoint"]
        if record["sha256"] != sha256(checkpoint):
            raise RuntimeError(f"arm {arm['name']} final checkpoint hash differs")
        arms[arm["name"]] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": record["sha256"],
            "policy_config": policy_config,
        }

    source_checkpoint = Path(args.checkpoint).resolve()
    all_rows = {"initial_rl": [], "heur": [], **{name: [] for name in arms}}
    with torch.random.fork_rng(devices=[]):
        for seed in seeds:
            all_rows["initial_rl"].append(run_episode(
                cfg, "rl", seed, source_checkpoint, reference_config))
            all_rows["heur"].append(run_episode(
                cfg, "heur", seed, source_checkpoint, reference_config))
            for name, arm in arms.items():
                all_rows[name].append(run_episode(
                    cfg, "rl", seed, Path(arm["checkpoint"]), arm["policy_config"]))
    heur_summary = summarize(all_rows["heur"])
    summaries = {"heur": heur_summary,
                 "initial_rl": summarize(all_rows["initial_rl"])}
    paired = {
        "initial_rl": paired_difference(all_rows["heur"], all_rows["initial_rl"])}
    for name in arms:
        summaries[name] = summarize(all_rows[name])
        paired[name] = paired_difference(all_rows["heur"], all_rows[name])
    return {
        "status": "completed_independent_screening_evaluation",
        "task": {"K_RED": 4, "STABLE_WINDOW": 8, "AGE_STD_TRACK": 10.0,
                 "RADAR_HIT_MODEL": "deterministic_beam", "W8_SCALE": 1.0},
        "seeds": seeds,
        "checkpoint_rule": f"fixed update {args.iterations} for each arm",
        "summaries": summaries, "rl_minus_heur": paired,
        "checkpoints": arms,
        "episodes": all_rows,
        "reserved_test_seeds_not_run": list(range(208000, 208030)),
        "no_training_teacher": True,
        "direct_rl_arbitration": True,
    }


if __name__ == "__main__":
    main()
