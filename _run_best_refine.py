"""Launch a new, auditable curriculum run from a verified best checkpoint.

This module intentionally imports no training code: preflight performs only reads.
Use the project's Python interpreter when invoking this launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
TASK_CONFIG = {
    "stable_window": 8,
    "age_std_track": 10.0,
    "c04_range": 80.0,
    "sector_axis_max_deg": 60.0,
    "arb_bid_weight": 0.35,
    "arb_hysteresis": 0.2,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integer seeds") from error
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("validation seeds must be unique and nonnegative")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=_positive_int, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=_positive_int, default=20)
    parser.add_argument("--mode", choices=("joint", "bid-only"), default="joint")
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-5)
    parser.add_argument("--episodes-per-iter", type=_positive_int, default=8)
    parser.add_argument("--validation-every", type=_positive_int, default=2)
    parser.add_argument("--patience", type=_nonnegative_int, default=3,
                        help="Stop after this many validations without improvement; 0 disables.")
    parser.add_argument("--validation-seeds", type=_seeds,
                        default=list(range(230, 240)))
    parser.add_argument("--training-seed-start", type=_nonnegative_int)
    parser.add_argument("--conflict-exposure", type=_unit_interval, default=1.0,
                        help="Training conflict exposure probability; evaluation remains natural.")
    parser.add_argument("--penalty-scale", type=_unit_interval, default=0.2,
                        help="Training violation penalty scale; does not change task thresholds.")
    parser.add_argument("--policy-anchor-coef", type=_nonnegative_float, default=0.0,
                        help="KL penalty coefficient against the fixed initial policy.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _project_path(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _same_number(value: Any, expected: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{label} mismatch: expected {expected}, got {value!r}")


def _recorded_path(value: Any, source_run: Path, expected: Path, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing")
    recorded = Path(value)
    if recorded.name != expected.name:
        raise ValueError(f"{label} must identify {expected.name}, got {value!r}")
    resolved = recorded.resolve() if recorded.is_absolute() else (source_run / recorded).resolve()
    if resolved != expected:
        raise ValueError(f"{label} points outside the selected best checkpoint: {value!r}")


def verify_source(source_run: Path, k: int) -> dict[str, Any]:
    log_path = source_run / "pilot_log.json"
    if not log_path.is_file():
        raise ValueError(f"source pilot log does not exist: {log_path}")
    log_bytes = log_path.read_bytes()
    data = _mapping(json.loads(log_bytes.decode("utf-8")), "pilot_log")
    best = _mapping(data.get(f"curr_k{k}_best"), f"curr_k{k}_best")
    iteration = best.get("iter")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < -1:
        raise ValueError("best iter must be an integer >= -1 (initial candidate is -1)")
    checkpoint = source_run / f"curr_k{k}_best.pt"
    _recorded_path(best.get("checkpoint"), source_run, checkpoint, "best checkpoint")
    if not checkpoint.is_file():
        raise ValueError(f"best checkpoint does not exist: {checkpoint}")
    expected_hash = best.get("checkpoint_sha256")
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is None:
        raise ValueError("best checkpoint_sha256 is missing or invalid")
    expected_hash = expected_hash.lower()
    with checkpoint.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"best checkpoint SHA256 mismatch: expected {expected_hash}, got {actual_hash}")

    validation = _mapping(best.get("validation"), "best validation")
    if validation.get("k") != k:
        raise ValueError(f"best validation K mismatch: expected {k}")
    configs = []
    if "task_config" in validation:
        configs.append(("best validation.task_config", _mapping(validation["task_config"], "task_config"), str.upper))
    experiment = _mapping(data.get("experiment", {}), "experiment")
    if "task_numeric_config" in experiment:
        configs.append(("experiment.task_numeric_config", _mapping(experiment["task_numeric_config"], "task_numeric_config"), str.lower))
    if not configs:
        raise ValueError("source is missing the recorded numeric task configuration")
    for label, config, transform in configs:
        for key, expected in TASK_CONFIG.items():
            _same_number(config.get(transform(key)), expected, f"{label}.{transform(key)}")

    recorded_artifact = validation.get("best_checkpoint")
    if recorded_artifact is not None:
        artifact = _mapping(recorded_artifact, "validation best_checkpoint")
        _recorded_path(artifact.get("path"), source_run, checkpoint, "validation best_checkpoint.path")
        if artifact.get("sha256", "").lower() != expected_hash:
            raise ValueError("validation best_checkpoint SHA256 differs from best checkpoint_sha256")

    direct_flags = []
    ppo = _mapping(experiment.get("ppo_config", {}), "experiment.ppo_config")
    if "direct_rl_arbitration" in ppo:
        direct_flags.append(ppo["direct_rl_arbitration"] is True)
    records = data.get("run_provenance", [])
    if not isinstance(records, list):
        raise ValueError("run_provenance must be a list")
    for record in records:
        if not isinstance(record, dict) or record.get("run_id") != validation.get("run_id"):
            continue
        env = _mapping(record.get("execution_environment", {}), "source execution_environment")
        if "MUMT_RL_DIRECT_ARBITRATION" in env:
            direct_flags.append(env["MUMT_RL_DIRECT_ARBITRATION"] == "1")
    if not direct_flags or not all(direct_flags):
        raise ValueError("source must explicitly record direct_rl_arbitration=true")
    return {
        "source_run": str(source_run),
        "source_log": str(log_path),
        "source_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "source_best_iter": iteration,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_hash,
    }


def prepare_launch(args: argparse.Namespace, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    source_run = _project_path(args.source_run)
    output = _project_path(args.output)
    if source_run == output or source_run in output.parents or output in source_run.parents:
        raise ValueError("output must be separate from the source run, with no ancestor/descendant overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output must be new or empty; refusing to overwrite: {output}")
    source = verify_source(source_run, args.k)
    training_seed_start = args.training_seed_start
    if training_seed_start is None:
        training_seed_start = {3: 10000, 4: 20000}.get(args.k, args.k * 10000)
    training_end = training_seed_start + args.iterations * args.episodes_per_iter
    if any(training_seed_start <= seed < training_end for seed in args.validation_seeds):
        raise ValueError("training seeds overlap the validation seeds")

    inherited = os.environ if environment is None else environment
    child_env = {key: value for key, value in inherited.items() if not key.upper().startswith("MUMT_")}
    overrides = {
        "MUMT_PILOT_OUT": str(output),
        "MUMT_CURR_INIT": source["checkpoint"],
        "MUMT_CURR_START_MODE": "best",
        "MUMT_CURR_EXPECTED_SHA256": source["checkpoint_sha256"],
        "MUMT_W8_FIXED": str(args.penalty_scale),
        "MUMT_STABLE_WINDOW": "8",
        "MUMT_AGE_STD_TRACK": "10",
        "MUMT_C04_RANGE": "80",
        "MUMT_SECTOR_AXIS_MAX_DEG": "60",
        "MUMT_ARB_BID_WEIGHT": "0.35",
        "MUMT_ARB_HYSTERESIS": "0.2",
        "MUMT_RL_DIRECT_ARBITRATION": "1",
        "MUMT_CONFLICT_EXPOSURE_PROB": str(args.conflict_exposure),
        "MUMT_POLICY_ANCHOR_COEF": str(args.policy_anchor_coef),
        "MUMT_BID_ONLY_PPO": "1" if args.mode == "bid-only" else "0",
        "MUMT_PPO_LR": str(args.learning_rate),
        "MUMT_PPO_EPOCHS": "2",
        "MUMT_EPISODES_PER_ITER": str(args.episodes_per_iter),
        "MUMT_VALIDATION_EVERY": str(args.validation_every),
        "MUMT_VALIDATION_SEEDS": ",".join(map(str, args.validation_seeds)),
        "MUMT_TRAIN_SEED_START": str(training_seed_start),
        "MUMT_CURR_PATIENCE": str(args.patience),
    }
    child_env.update(overrides)
    return {
        "source": source,
        "output": str(output),
        "task_numeric_config": TASK_CONFIG.copy(),
        "training_controls": {
            "conflict_exposure": args.conflict_exposure,
            "penalty_scale": args.penalty_scale,
            "policy_anchor_coef": args.policy_anchor_coef,
        },
        "mode": args.mode,
        "command": [sys.executable, str(ROOT / "_p3_pilot.py"), "curr", str(args.k), "0", str(args.iterations)],
        "cwd": str(ROOT),
        "training_environment": overrides,
        "environment": child_env,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        launch = prepare_launch(args)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in launch.items() if key != "environment"},
                     ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return 0
    completed = subprocess.run(launch["command"], cwd=ROOT, env=launch["environment"], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
