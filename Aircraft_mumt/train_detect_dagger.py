"""One-round DAgger warm start for the learned detection policy.

HEUR labels are queried only on training trajectories. Evaluation runs the
learned policy and production shield without consulting the teacher.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _experiment_provenance import (add_output, artifact_record,
                                    capture_run_provenance,
                                    capture_source_snapshot,
                                    relocate_source_snapshot,
                                    register_active_run,
                                    track_run_lifecycle,
                                    write_json_atomic)

_startup_parser = argparse.ArgumentParser(add_help=False)
_startup_parser.add_argument("--output", type=Path)
_startup_args, _ = _startup_parser.parse_known_args()
_startup_output = (_startup_args.output.resolve()
                   if _startup_args.output else
                   ROOT / "runs" / "dagger_warm_start")
_startup_temp = None
_startup_output_is_empty = (
    not _startup_output.exists() or not any(_startup_output.iterdir()))
if ("--help" not in sys.argv and "-h" not in sys.argv
        and _startup_args.output and _startup_output_is_empty):
    _startup_temp = tempfile.TemporaryDirectory(
        dir=ROOT / "runs")
    SOURCE_SNAPSHOT = capture_source_snapshot(
        ROOT, _startup_temp.name, [Path(__file__).resolve()])
else:
    SOURCE_SNAPSHOT = None

from aircraft_sim.detect_baselines import HeurPolicy
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import DetectMAPPOPolicy
from aircraft_sim.detect_shield import DetectQPShield


def _archive_input(run_record: dict, source: Path, role: str) -> tuple[Path, dict]:
    before = artifact_record(source, role)
    if not before["exists"]:
        raise FileNotFoundError(source)
    archive_path = (Path(run_record["output_dir"]) / "provenance"
                    / f"{run_record['run_id']}_input_{role}.pt")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archive_path)
    archived = artifact_record(archive_path, f"{role}_archive")
    if before["sha256"] != archived["sha256"]:
        raise RuntimeError(f"checkpoint changed while archiving: {source}")
    after = artifact_record(source, role)
    if before["sha256"] != after["sha256"]:
        raise RuntimeError(f"checkpoint changed while archiving: {source}")
    before["archived_snapshot"] = archived
    inputs = run_record.setdefault("inputs", [])
    existing = next((item for item in inputs
                     if item.get("path") == before["path"]
                     and item.get("role") == role), None)
    if existing is None:
        inputs.append(before)
    else:
        existing.update(before)
    return archive_path, before


def _archive_output(run_record: dict, source: Path, role: str) -> dict:
    before = artifact_record(source, role)
    if not before["exists"]:
        raise FileNotFoundError(source)
    archive_path = (Path(run_record["output_dir"]) / "provenance"
                    / f"{run_record['run_id']}_output_{source.name}")
    shutil.copy2(source, archive_path)
    archived = artifact_record(archive_path, f"{role}_archive")
    if before["sha256"] != archived["sha256"]:
        raise RuntimeError(f"checkpoint changed while archiving: {source}")
    archived["source_path"] = str(source)
    run_record.setdefault("checkpoint_archives", []).append(archived)
    add_output(run_record, source, role)
    return archived


def _teacher_demos(policy, shield, seeds: list[int], k: int) -> list[tuple]:
    demos = []
    teacher = HeurPolicy()
    for seed in seeds:
        observations = policy.env.reset({"seed": seed, "k_red": k})
        env = policy.env
        while env.terminal is None:
            nominal = teacher(env)
            safe, _ = shield.filter(env, nominal)
            demos.extend((policy.encoder.encode(observations[i]), safe[i])
                         for i in observations)
            result = env.step(safe)
            observations = result.observations
    return demos


def _student_demos(policy, shield, seeds: list[int], k: int) -> list[tuple]:
    demos = []
    teacher = HeurPolicy()
    for seed in seeds:
        observations = policy.env.reset({"seed": seed, "k_red": k})
        env = policy.env
        while env.terminal is None:
            student, _ = policy.act_batch(observations, deterministic=True)
            teacher_safe, _ = shield.filter(env, teacher(env))
            demos.extend((policy.encoder.encode(observations[i]), teacher_safe[i])
                         for i in observations)
            student_safe, _ = shield.filter(env, student)
            result = env.step(student_safe)
            observations = result.observations
    return demos


def _evaluate(policy, shield, seeds: list[int], k: int) -> list[dict]:
    rows = []
    for seed in seeds:
        env = policy.env
        observations = env.reset({"seed": seed, "k_red": k})
        total_return = 0.0
        violations = 0
        while env.terminal is None:
            actions, _ = policy.act_batch(observations, deterministic=True)
            safe, _ = shield.filter(env, actions)
            result = env.step(safe)
            total_return += float(result.reward)
            violations += len(result.info["report"].violations)
            observations = result.observations
        rows.append({"seed": seed, "terminal": env.terminal,
                     "return": round(total_return, 3),
                     "steps": env.step_count, "violations": violations,
                     "max_stable_progress": max(
                         env.stable_progress.values(), default=0)})
    return rows


@track_run_lifecycle
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--replay-checkpoint", action="append", type=Path,
                        default=[], help="additional historical student policy; repeatable")
    parser.add_argument("--output", required=True, type=Path,
                        help="new output directory; existing checkpoints are not overwritten")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--teacher-seeds", default="100,101,102,103,104,105,106,107")
    parser.add_argument("--student-seeds", default="400,401,402,403")
    parser.add_argument("--eval-seeds", default="200,201,202,203,204")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bc-learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be empty: {output}")
    if args.k < 1 or args.epochs < 1 or args.bc_learning_rate <= 0:
        parser.error("--k, --epochs and --bc-learning-rate must be positive")

    parse_seeds = lambda text: [int(value) for value in text.split(",") if value.strip()]
    teacher_seeds = parse_seeds(args.teacher_seeds)
    student_seeds = parse_seeds(args.student_seeds)
    eval_seeds = parse_seeds(args.eval_seeds)
    replay_checkpoints = [path.resolve() for path in args.replay_checkpoint]
    missing_replay = [path for path in replay_checkpoints if not path.is_file()]
    if missing_replay:
        parser.error(f"replay checkpoint not found: {missing_replay[0]}")

    env = DetectEnv()
    shield = DetectQPShield(env.cfg)
    policy = DetectMAPPOPolicy(env, shield, seed=0)
    policy.config.bc_learning_rate = args.bc_learning_rate
    if SOURCE_SNAPSHOT is None:
        parser.error("could not capture startup source snapshot")
    relocate_source_snapshot(SOURCE_SNAPSHOT, output)
    _startup_temp.cleanup()
    run_record = capture_run_provenance(
        repo_root=ROOT,
        output_dir=output,
        entry_scripts=[Path(__file__).resolve()],
        phase="dagger_warm_start",
        task_config=env.cfg,
        algorithm_config=policy.config,
        runtime_parameters={
            "environment": {key: value for key, value in os.environ.items()
                            if key.startswith("MUMT_")},
            "k": args.k,
            "teacher_seeds": teacher_seeds,
            "student_seeds": student_seeds,
            "eval_seeds": eval_seeds,
            "epochs": args.epochs,
            "bc_learning_rate": args.bc_learning_rate,
            "continuous_objective": "latent_regression",
            "replay_checkpoints": [str(path) for path in replay_checkpoints],
        },
        inputs=[artifact_record(checkpoint, "source_checkpoint")]
                + [artifact_record(path, f"replay_checkpoint_{index}")
                   for index, path in enumerate(replay_checkpoints)],
        source_snapshot=SOURCE_SNAPSHOT,
    )
    run_record["output_dir"] = str(output)
    run_record["status"] = "running"
    manifest_path = output / "provenance" / f"{run_record['run_id']}.json"
    run_record["manifest_path"] = str(manifest_path.resolve())
    write_json_atomic(manifest_path, {"run_provenance": run_record})
    register_active_run(run_record, lambda record: write_json_atomic(
        manifest_path, {"run_provenance": record}))
    archived_checkpoint, _ = _archive_input(
        run_record, checkpoint, "source_checkpoint")
    archived_replays = []
    for index, source in enumerate(replay_checkpoints):
        archived, _ = _archive_input(
            run_record, source, f"replay_checkpoint_{index}")
        archived_replays.append(archived)
    write_json_atomic(manifest_path, {"run_provenance": run_record})
    policy.load(str(archived_checkpoint))

    teacher_demos = _teacher_demos(policy, shield, teacher_seeds, args.k)
    student_demos = []
    replay_sources = [archived_checkpoint, *archived_replays]
    replay_counts = {}
    source_labels = [checkpoint, *replay_checkpoints]
    for source, source_label in zip(replay_sources, source_labels):
        policy.load(str(source))
        samples = _student_demos(policy, shield, student_seeds, args.k)
        student_demos.extend(samples)
        replay_counts[str(source_label)] = len(samples)
    policy.load(str(archived_checkpoint))
    demos = teacher_demos + student_demos
    bc_stats = policy.behavior_clone(
        demos, epochs=args.epochs, continuous_objective="latent_regression")
    evaluation = _evaluate(policy, shield, eval_seeds, args.k)

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_output = output / "dagger_bc.pt"
    policy.save(str(checkpoint_output))
    checkpoint_archive = _archive_output(
        run_record, checkpoint_output, "dagger_bc_checkpoint")
    payload = {
        "source_checkpoint": str(checkpoint), "k": args.k,
        "teacher_seeds": teacher_seeds,
        "student_seeds": student_seeds,
        "eval_seeds": eval_seeds,
        "bc_learning_rate": args.bc_learning_rate,
        "teacher_demos": len(teacher_demos),
        "student_state_demos": len(student_demos),
        "student_demos_by_checkpoint": replay_counts,
        "bc": bc_stats,
        "evaluation": evaluation,
        "success_rate": sum(row["terminal"] == "success" for row in evaluation)
                        / max(1, len(evaluation)),
        "mean_return": sum(row["return"] for row in evaluation)
                       / max(1, len(evaluation)),
        "run_id": run_record["run_id"],
        "checkpoint": checkpoint_archive,
    }
    run_record["status"] = "completed"
    run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["provenance"] = {"run_id": run_record["run_id"],
                             "sidecar": str(manifest_path.resolve())}
    write_json_atomic(output / "dagger_log.json", payload)
    run_record["associated_log"] = artifact_record(
        output / "dagger_log.json", "dagger_log")
    write_json_atomic(manifest_path, {"run_provenance": run_record})
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
