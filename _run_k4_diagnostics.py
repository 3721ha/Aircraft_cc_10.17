"""Run paired K4 training controls from one verified RL best checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from _curriculum_guard import file_sha256
from _experiment_provenance import canonical_sha256, write_json_atomic
from _run_best_refine import (ROOT, _nonnegative_int, _positive_int, _seeds,
                              build_parser as refine_parser, prepare_launch)


CASES = {
    "control": (1.0, 0.2, 0.0),
    "natural_requests": (0.0, 0.2, 0.0),
    "full_penalty": (1.0, 1.0, 0.0),
    "natural_full_penalty": (0.0, 1.0, 0.0),
    "natural_full_penalty_anchor": (0.0, 1.0, 1.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path,
                        default=Path("runs/model_optimization_v1/curriculum_k4_w8a10"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("runs/model_optimization_v1/k4_diagnostics_v2"))
    parser.add_argument("--iterations", type=_positive_int, default=8)
    parser.add_argument("--episodes-per-iter", type=_positive_int, default=8)
    parser.add_argument("--validation-every", type=_positive_int, default=2)
    parser.add_argument("--training-seed-start", type=_nonnegative_int, default=30000)
    parser.add_argument("--validation-seeds", type=_seeds, default=list(range(230, 240)))
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _code_hashes() -> dict[str, str]:
    sources = list((ROOT / "Aircraft_mumt" / "aircraft_sim").glob("*.py"))
    sources += [ROOT / name for name in (
        "_p3_pilot.py", "_curriculum_guard.py", "_run_best_refine.py",
        "_run_k4_diagnostics.py", "_experiment_provenance.py")]
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(sources)}


def prepare_suite(args: argparse.Namespace) -> dict:
    output = (ROOT / args.output_root).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Suite output must be new or empty: {output}")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("Diagnostic cases must be unique")
    jobs = []
    for name in args.cases:
        exposure, penalty, anchor = CASES[name]
        child_args = refine_parser().parse_args([
            "--k", "4", "--source-run", str(args.source_run),
            "--output", str(output / name),
            "--iterations", str(args.iterations),
            "--episodes-per-iter", str(args.episodes_per_iter),
            "--validation-every", str(args.validation_every), "--patience", "0",
            "--training-seed-start", str(args.training_seed_start),
            "--validation-seeds", ",".join(map(str, args.validation_seeds)),
            "--conflict-exposure", str(exposure), "--penalty-scale", str(penalty),
            "--policy-anchor-coef", str(anchor),
        ])
        job = {"name": name, **prepare_launch(child_args)}
        # Monitor all groups against their initial RL policy, including the
        # unregularized controls; a disabled loss must not mean fake zero drift.
        job["environment"]["MUMT_POLICY_ANCHOR_MONITOR"] = "1"
        job["training_environment"]["MUMT_POLICY_ANCHOR_MONITOR"] = "1"
        jobs.append(job)
    sources = [job["source"] for job in jobs]
    if any(source != sources[0] for source in sources[1:]):
        raise ValueError("Source changed while preparing paired cases")
    return {
        "output_root": str(output), "jobs": jobs,
        "source_code_sha256": _code_hashes(),
        "budget": {"iterations": args.iterations,
                   "episodes_per_iter": args.episodes_per_iter,
                   "validation_every": args.validation_every,
                   "training_seed_start": args.training_seed_start,
                   "validation_seeds": args.validation_seeds,
                   "early_stopping_patience": 0},
    }


def public_plan(plan: dict) -> dict:
    return {**plan, "jobs": [
        {key: value for key, value in job.items() if key != "environment"}
        for job in plan["jobs"]]}


def _check_inputs(plan: dict, job: dict) -> None:
    source = job["source"]
    if (file_sha256(Path(source["checkpoint"])) != source["checkpoint_sha256"]
            or file_sha256(Path(source["source_log"])) != source["source_log_sha256"]):
        raise ValueError("Selected source checkpoint or log changed during suite")
    if _code_hashes() != plan["source_code_sha256"]:
        raise ValueError("Training source changed during suite; refusing to mix code versions")


def _rank(validation: dict) -> tuple:
    rl = validation["rl"]
    return (rl["success_rate"], rl["mean_return"], -rl["violations_per_step"])


def verify_completed_run(job: dict, budget: dict) -> dict:
    output = Path(job["output"])
    log = json.loads((output / "pilot_log.json").read_text(encoding="utf-8"))
    rows, experiment = log["curr_k4"], log["experiment"]
    record = log["run_provenance"][-1]
    if (record["status"] != "completed" or not record["source_integrity"]["unchanged"]
            or len(rows) != budget["iterations"]
            or record["completed_iterations"] != len(rows)
            or record["early_stopping"]["triggered"]):
        raise ValueError("Incomplete or inconsistent fixed-budget diagnostic run")
    if record["curriculum_start"]["sha256"] != job["source"]["checkpoint_sha256"]:
        raise ValueError("Run loaded a different input checkpoint")
    controls = job["training_controls"]
    ppo = experiment["ppo_config"]
    if (experiment["validation_seeds"] != budget["validation_seeds"]
            or experiment["fixed_w8_scale"] != controls["penalty_scale"]
            or experiment["training_seed_start"] != budget["training_seed_start"]
            or experiment["early_stopping_patience"] != 0
            or ppo["conflict_exposure_prob"] != controls["conflict_exposure"]
            or ppo["policy_anchor_coef"] != controls["policy_anchor_coef"]
            or not ppo["direct_rl_arbitration"] or ppo["bid_only_ppo"]):
        raise ValueError("Actual training controls differ from the suite plan")
    baseline, best = log["curr_k4_baseline"], log["curr_k4_best"]
    validations = [baseline] + [row["validation"] for row in rows if "validation" in row]
    if any(v["seeds"] != budget["validation_seeds"] or v["k"] != 4
           or v["task_config_sha256"] != baseline["task_config_sha256"]
           or v["natural_arbitration"]["forced_probe_enabled"]
           for v in validations):
        raise ValueError("Validation task, seeds or natural evaluation mode changed")
    chosen = max(validations, key=_rank)  # First candidate wins ties, like training.
    if best["iter"] != chosen["iter"] or tuple(best["rank"]) != _rank(chosen):
        raise ValueError("Best checkpoint selection does not match validation ranking")
    if (file_sha256(output / "curr_k4_best.pt") != best["checkpoint_sha256"]
            or file_sha256(output / "curr_k4.pt") != rows[-1]["checkpoint"]["sha256"]):
        raise ValueError("Best/latest checkpoint differs from its logged SHA256")
    for iteration, row in enumerate(rows):
        seeds = [budget["training_seed_start"] + iteration * budget["episodes_per_iter"] + ep
                 for ep in range(budget["episodes_per_iter"])]
        if (row["iter"] != iteration or row["episode_seeds"] != seeds
                or row["w8_scale"] != controls["penalty_scale"]):
            raise ValueError("Training iteration, seeds or penalty scale mismatch")
        if not math.isfinite(row["post_update_policy_anchor_kl"]):
            raise ValueError("Nonfinite policy anchor KL")
        if controls["conflict_exposure"] == 0 and (
                row["forced_overlap_pairs"] or row["exposed_agent_steps"]):
            raise ValueError("Natural training unexpectedly used action exposure")
    return {
        "best_iter": best["iter"], "best_checkpoint_sha256": best["checkpoint_sha256"],
        "baseline": {"rl": baseline["rl"], "heur": baseline["heur"]},
        "baseline_by_seed_sha256": canonical_sha256(baseline["by_seed"]),
        "best": {"rl": chosen["rl"], "heur": chosen["heur"],
                 "natural_arbitration": chosen["natural_arbitration"]},
        "last": {"rl": log["curr_k4_last_validation"]["rl"],
                 "heur": log["curr_k4_last_validation"]["heur"]},
        "total_environment_steps": sum(row["environment_steps"] for row in rows),
        "total_agent_steps": sum(row["agent_steps"] for row in rows),
        "total_exposed_agent_steps": sum(row["exposed_agent_steps"] for row in rows),
        "last_post_update_policy_anchor_kl": rows[-1]["post_update_policy_anchor_kl"],
    }


def run_suite(plan: dict) -> int:
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "diagnostic_suite.json"
    manifest = {"status": "running", "plan": public_plan(plan), "results": []}
    write_json_atomic(manifest_path, manifest)
    baseline_digest = None
    for job in plan["jobs"]:
        result = {"name": job["name"], "status": "running",
                  "stdout": str(output / f"{job['name']}.stdout.txt")}
        manifest["results"].append(result)
        write_json_atomic(manifest_path, manifest)
        print(f"[K4 diagnostic] {job['name']}; log={result['stdout']}", flush=True)
        try:
            _check_inputs(plan, job)
            environment = dict(job["environment"])
            environment["MUMT_DIAGNOSTIC_CASE"] = job["name"]
            environment["MUMT_DIAGNOSTIC_SUITE_SHA256"] = plan[
                "source_code_sha256"]["_run_k4_diagnostics.py"]
            with Path(result["stdout"]).open("w", encoding="utf-8") as stream:
                process = subprocess.run(job["command"], cwd=ROOT, env=environment,
                                         stdout=stream, stderr=subprocess.STDOUT, check=False)
            result["returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"Training process exited {process.returncode}")
            _check_inputs(plan, job)
            result["summary"] = verify_completed_run(job, plan["budget"])
            digest = result["summary"]["baseline_by_seed_sha256"]
            if baseline_digest is not None and digest != baseline_digest:
                raise ValueError("Paired cases have different initial evaluation results")
            baseline_digest = digest
            result["status"] = "completed"
            print(json.dumps({"case": job["name"], **result["summary"]}), flush=True)
        except (Exception, KeyboardInterrupt) as error:
            result["status"] = manifest["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            write_json_atomic(manifest_path, manifest)
            print(result["error"], file=sys.stderr, flush=True)
            return result.get("returncode") or 1
        write_json_atomic(manifest_path, manifest)
    manifest["status"] = "completed"
    write_json_atomic(manifest_path, manifest)
    print(f"[K4 diagnostics completed] {manifest_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = prepare_suite(args)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(public_plan(plan), indent=2), flush=True)
    return 0 if args.preflight_only else run_suite(plan)


if __name__ == "__main__":
    raise SystemExit(main())
