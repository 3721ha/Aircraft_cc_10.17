"""Strict staged MAPPO experiment for the conflict-resolution policy.

This runner keeps the task, rule predicates, red model and safety chain
unchanged.  It only changes which policy parameters receive gradients during
each training stage:

  A (discrete): role, gear and pointer heads + the central value critic.
  B (continuous): bounded flight-control head and log standard deviation +
                  the central value critic.
  C (joint): all actor and critic parameters, with a small learning rate.

The shared actor trunk stays frozen in stages A and B.  This is deliberate:
the current PPO update computes all head losses in one pass, so allowing the
trunk to move while a head is nominally frozen would silently change the
frozen head's output.  Stage C provides the joint representation adaptation.

Run from the project root, for example::

  Aircraft_mumt/.venv/Scripts/python.exe _p3_staged.py a 0 20
  Aircraft_mumt/.venv/Scripts/python.exe _p3_staged.py b 0 20
  Aircraft_mumt/.venv/Scripts/python.exe _p3_staged.py c 0 40
  Aircraft_mumt/.venv/Scripts/python.exe _p3_staged.py eval

The default source for stage A is the current shield-aware BC checkpoint.  B
loads A's checkpoint and C loads B's checkpoint.  Set ``MUMT_STAGED_OUT`` to
keep separate runs.  The evaluation path always uses the simulator's standard
stable window (no training-only override) and only calls the RL policy followed
by the QP shield.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _experiment_provenance import (  # noqa: E402
    add_output,
    artifact_record,
    capture_run_provenance,
    capture_source_snapshot,
    canonical_sha256,
    register_active_run,
    track_run_lifecycle,
    write_json_atomic,
)


OUT = Path(os.environ.get(
    "MUMT_STAGED_OUT",
    str(ROOT / "Aircraft_mumt" / "runs" / "detect_staged")))
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_SNAPSHOT = capture_source_snapshot(
    ROOT, OUT, [ROOT / "_p3_staged.py"])

from aircraft_sim.detect_env import DetectEnv  # noqa: E402
from aircraft_sim.detect_mappo import DetectMAPPOPolicy  # noqa: E402
from aircraft_sim.detect_shield import DetectQPShield  # noqa: E402

STAGE_ALIASES = {
    "a": "a", "discrete": "a", "disc": "a",
    "b": "b", "continuous": "b", "cont": "b",
    "c": "c", "joint": "c", "all": "c",
}

DEFAULT_ITERS = {"a": 20, "b": 20, "c": 40}
STAGE_LABEL = {
    "a": "discrete",
    "b": "continuous",
    "c": "joint",
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _stage_arg(value: str) -> str:
    try:
        return STAGE_ALIASES[value.lower()]
    except KeyError as exc:
        raise SystemExit(
            f"unknown stage {value!r}; use a/discrete, b/continuous, "
            "c/joint or eval") from exc


def _stage_source(stage: str) -> Path:
    """Resolve a source checkpoint without changing any existing checkpoint."""
    explicit = os.environ.get(f"MUMT_STAGED_SOURCE_{stage.upper()}")
    if explicit:
        return Path(explicit)

    if stage == "a":
        candidates = [
            ROOT / "Aircraft_mumt" / "runs" / "detect_pilot" / "pilot_bc.pt",
            OUT / "stage_a_final.pt",
            OUT / "stage_a.pt",
        ]
    elif stage == "b":
        candidates = [OUT / "stage_a_final.pt", OUT / "stage_a.pt"]
    else:
        candidates = [OUT / "stage_b_final.pt", OUT / "stage_b.pt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    names = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No source checkpoint for stage {stage}; looked at: {names}")


def _set_stage(pol: DetectMAPPOPolicy, stage: str) -> List[str]:
    """Apply an exact parameter freeze and rebuild Adam for active parameters."""
    model = pol.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # The critic is needed for a useful value baseline in every stage.  It is
    # not part of the deployed policy and therefore does not violate the
    # actor-head freeze.
    for parameter in model.critic.parameters():
        parameter.requires_grad_(True)

    if stage == "a":
        modules = (model.role_head, model.gear_head, model.pointer_head)
    elif stage == "b":
        modules = (model.cont_head,)
        model.log_std.requires_grad_(True)
    elif stage == "c":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        modules = ()
    else:  # pragma: no cover - guarded by _stage_arg
        raise ValueError(stage)

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    # Do not retain stale Adam moments from a different objective.  The
    # checkpoint contains model weights only, so this is also deterministic
    # when resuming a stage in a new process.
    lr_default = {"a": 5e-5, "b": 5e-5, "c": 2e-5}[stage]
    lr = _env_float(f"MUMT_STAGED_LR_{stage.upper()}", lr_default)
    active = [p for p in model.parameters() if p.requires_grad]
    pol.optimizer = __import__("torch").optim.Adam(active, lr=lr)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"[stage {stage}] lr={lr:g} active={','.join(names)}", flush=True)
    return names


def _snapshot(model) -> Dict[str, object]:
    return {name: parameter.detach().clone()
            for name, parameter in model.named_parameters()}


def _parameter_deltas(before: Mapping[str, object], model) -> Dict[str, float]:
    """Return grouped L2 parameter movement for a frozen-gradient audit."""
    import torch

    groups = {
        "trunk": ("actor.",),
        "continuous": ("cont_head.", "log_std"),
        "discrete": ("role_head.", "gear_head.", "pointer_head."),
        "critic": ("critic.",),
    }
    result = {}
    for group, prefixes in groups.items():
        total = torch.zeros(())
        for name, parameter in model.named_parameters():
            if not any(name.startswith(prefix) or name == prefix
                       for prefix in prefixes):
                continue
            total = total + (parameter.detach() - before[name]).pow(2).sum()
        result[group] = float(total.sqrt())
    return result


def _evaluate(pol: DetectMAPPOPolicy, shield: DetectQPShield,
              seeds: Sequence[int], k_red: int) -> List[dict]:
    """Evaluate deterministic RL actions with the production shield only."""
    rows = []
    for seed in seeds:
        env = DetectEnv()
        env.reset({"seed": int(seed), "k_red": int(k_red)})
        pol.env = env
        ret = 0.0
        steps = 0
        violations = 0
        max_progress = 0
        while env.terminal is None:
            obs = env.observe()
            nominal = {
                i: pol.act_with_log_prob(obs[i], deterministic=True)[0]
                for i in obs
            }
            safe, _ = shield.filter(env, nominal)
            result = env.step(safe)
            ret += result.reward
            steps += 1
            violations += len(result.info["report"].violations)
            max_progress = max(max_progress,
                               max(env.stable_progress.values(), default=0))
        rows.append({
            "seed": int(seed),
            "k": int(k_red),
            "terminal": env.terminal,
            "return": round(ret, 3),
            "steps": steps,
            "violations": violations,
            "max_stable_progress": max_progress,
        })
    return rows


def _load_log() -> dict:
    path = OUT / "staged_log.json"
    if not path.exists():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_log(log: dict) -> None:
    write_json_atomic(OUT / "staged_log.json", log)


def _persist_run_record(log: dict, run_record: dict) -> None:
    manifest_path = OUT / "provenance" / f"{run_record['run_id']}.json"
    run_record["manifest_path"] = str(manifest_path.resolve())
    records = log.setdefault("run_provenance", [])
    for index, record in enumerate(records):
        if record.get("run_id") == run_record["run_id"]:
            records[index] = run_record
            break
    else:
        records.append(run_record)
    _write_log(log)
    write_json_atomic(manifest_path, {
        "run_provenance": run_record,
        "associated_log": artifact_record(OUT / "staged_log.json", "staged_log"),
    })


def _load_policy_checkpoint(pol, path: Path, run_record: dict, role: str) -> None:
    import torch

    source = Path(path).resolve()
    before = artifact_record(source, role)
    if not before["exists"]:
        raise FileNotFoundError(source)
    snapshot_path = (OUT / "provenance" /
                     f"{run_record['run_id']}_input_{role}_{source.name}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot_path)
    snapshot = artifact_record(snapshot_path, f"{role}_snapshot")
    if before["sha256"] != snapshot["sha256"]:
        raise RuntimeError(f"checkpoint changed while archiving: {source}")
    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    pol.load(str(snapshot_path))
    after = artifact_record(source, role)
    if before["sha256"] != after["sha256"]:
        raise RuntimeError(f"checkpoint changed while loading: {source}")
    saved_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    before["checkpoint_saved_config"] = saved_config
    before["checkpoint_config_sha256"] = (
        canonical_sha256(saved_config) if saved_config is not None else None)
    before["archived_snapshot"] = snapshot
    run_record.setdefault("inputs", []).append(before)


def _archive_checkpoint(run_record: dict, path: Path, role: str,
                        archive_name: str) -> dict:
    source = Path(path).resolve()
    source_record = artifact_record(source, role)
    if not source_record["exists"]:
        raise FileNotFoundError(source)
    archive_path = OUT / "provenance" / f"{run_record['run_id']}_{archive_name}"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archive_path)
    archived = artifact_record(archive_path, f"{role}_archive")
    if source_record["sha256"] != archived["sha256"]:
        raise RuntimeError(f"checkpoint changed while archiving: {source}")
    archived["source_path"] = str(source)
    run_record.setdefault("checkpoint_archives", []).append(archived)
    return archived


def _save_checkpoint(pol, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    pol.save(str(temporary))
    temporary.replace(path)


def _evaluation_rank(evaluation: dict) -> tuple[float, float, float]:
    rows = evaluation["rows"]
    violations = sum(row["violations"] for row in rows)
    steps = sum(row["steps"] for row in rows)
    return (float(evaluation["success"]), float(evaluation["mean_return"]),
            -violations / max(1, steps))


def _train(stage: str, iter0: int, n_iters: int) -> None:
    import numpy as np

    t0 = time.time()
    train_env = DetectEnv()
    shield = DetectQPShield(train_env.cfg)
    pol = DetectMAPPOPolicy(train_env, shield, seed=0)

    stage_ckpt = OUT / f"stage_{stage}.pt"
    best_ckpt = OUT / f"stage_{stage}_best.pt"
    final_ckpt = OUT / f"stage_{stage}_final.pt"
    if iter0 > 0:
        source = stage_ckpt
        if not source.exists():
            raise FileNotFoundError(
                f"Cannot resume stage {stage} at iter {iter0}: {source}")
    else:
        source = _stage_source(stage)
    active_names = _set_stage(pol, stage)

    episodes = _env_int("MUMT_STAGED_EPISODES_PER_ITER", 4)
    k_red = _env_int("MUMT_STAGED_K", 2)
    eval_every = _env_int("MUMT_STAGED_EVAL_EVERY", 5)
    seed_base = _env_int("MUMT_STAGED_SEED_BASE", 70000)
    sw_start = _env_int("MUMT_STAGED_SW_START", 10)
    sw_final = _env_int("MUMT_STAGED_SW_FINAL", 30)
    sw_span = max(1, _env_int("MUMT_STAGED_SW_SPAN", 50))
    eval_seeds = [int(x) for x in os.environ.get(
        "MUMT_STAGED_EVAL_SEEDS", "300,301,302").split(",") if x.strip()]

    run_record = capture_run_provenance(
        repo_root=ROOT,
        output_dir=OUT,
        entry_scripts=[ROOT / "_p3_staged.py"],
        phase=f"staged_{stage}",
        task_config=train_env.cfg,
        algorithm_config=pol.config,
        runtime_parameters={
            "environment": {key: value for key, value in os.environ.items()
                            if key.startswith("MUMT_")},
            "stage": stage,
            "iteration_start": iter0,
            "iterations": n_iters,
            "episodes_per_iter": episodes,
            "k_red": k_red,
            "eval_every": eval_every,
            "seed_base": seed_base,
            "stable_window_curriculum": {
                "start": sw_start,
                "final": sw_final,
                "span": sw_span,
            },
            "evaluation_seeds": eval_seeds,
            "source_checkpoint": str(source.resolve()),
            "active_parameters": active_names,
            "effective_learning_rate": pol.optimizer.param_groups[0]["lr"],
        },
        inputs=[],
        source_snapshot=SOURCE_SNAPSHOT,
    )
    run_record["training_state_semantics"] = {
        "mode": "weights_only_warm_start",
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "note": "Checkpoint loading restores model weights only; optimizer and RNG are initialized for this invocation.",
    }
    log = _load_log()
    register_active_run(
        run_record, lambda record: _persist_run_record(_load_log(), record))
    _persist_run_record(log, run_record)
    _load_policy_checkpoint(
        pol, source, run_record,
        "stage_resume_checkpoint" if iter0 > 0 else "stage_initial_checkpoint")
    _persist_run_record(log, run_record)
    print(f"[stage {stage}] loaded {source}", flush=True)
    pol.env = train_env

    stage_log = log.setdefault("stages", {}).setdefault(stage, [])
    best_rank = None
    for iteration in range(iter0, iter0 + n_iters):
        fraction = min(1.0, max(0.0, iteration / float(sw_span)))
        stable_window = int(round(
            sw_start + (sw_final - sw_start) * fraction))
        rollout = []
        episode_returns: List[float] = []
        episode_terminal: List[str] = []
        episode_steps: List[int] = []
        episode_violations: List[int] = []
        for episode in range(episodes):
            seed = seed_base + ord(stage) * 10000 + iteration * 100 + episode
            data = pol.collect(k_red=k_red, seed=seed,
                               stable_window=stable_window)
            rollout.extend(data)
            episode_returns.append(
                sum(item["reward"] for item in data) / pol.config.reward_scale)
            episode_terminal.append(data[-1]["terminal"])
            episode_steps.append(len(data))
            episode_violations.append(sum(item["violations"] for item in data))

        before = _snapshot(pol.model)
        stats = pol.update(rollout)
        deltas = _parameter_deltas(before, pol.model)
        n_agents = train_env.cfg.N_WINGMEN
        agent_steps = len(rollout) * n_agents
        corrections = [magnitude for item in rollout
                       for magnitude in item.get("correction_magnitudes", [])]
        entry = {
            "iter": iteration,
            "run_id": run_record["run_id"],
            "stage": STAGE_LABEL[stage],
            "stable_window": stable_window,
            "k": k_red,
            "mean_return": round(float(np.mean(episode_returns)), 3),
            "success": round(sum(t == "success" for t in episode_terminal)
                              / max(1, len(episode_terminal)), 4),
            "mean_steps": round(float(np.mean(episode_steps))),
            "vio_per_step": round(sum(episode_violations)
                                   / max(1, sum(episode_steps)), 4),
            "track_action_rate": round(sum(item.get("n_track_actions", 0)
                                            for item in rollout)
                                       / max(1, agent_steps), 4),
            "pointer_rate": round(sum(item.get("n_nonzero_pointer", 0)
                                      for item in rollout)
                                  / max(1, agent_steps), 4),
            "valid_request_rate": round(sum(item.get("n_valid_requests", 0)
                                            for item in rollout)
                                       / max(1, agent_steps), 4),
            "intervention_rate": round(sum(item.get("n_interventions", 0)
                                           for item in rollout)
                                      / max(1, agent_steps), 4),
            "mean_correction": round(float(np.mean(corrections)), 6)
            if corrections else 0.0,
            "max_stable_progress": max(
                (item.get("max_stable_progress", 0) for item in rollout),
                default=0),
            "active_parameters": active_names,
            "parameter_delta_l2": deltas,
            "policy_loss": round(float(stats.get("policy_loss", 0.0)), 6),
            "value_loss": round(float(stats.get("value_loss", 0.0)), 6),
            "approx_kl": round(float(stats.get("approx_kl", 0.0)), 6),
            "cf_nonzero_frac": round(float(stats.get("cf_nonzero_frac", 0.0)), 6),
        }
        if iteration % max(1, eval_every) == 0 or iteration == iter0 + n_iters - 1:
            evaluation = _evaluate(pol, shield, eval_seeds, k_red)
            entry["eval"] = {
                "success": round(sum(row["terminal"] == "success"
                                      for row in evaluation)
                                 / max(1, len(evaluation)), 4),
                "mean_return": round(float(np.mean(
                    [row["return"] for row in evaluation])), 3),
                "max_stable_progress": max(
                    (row["max_stable_progress"] for row in evaluation),
                    default=0),
                "rows": evaluation,
            }
            pol.env = train_env
        stage_log.append(entry)
        _save_checkpoint(pol, stage_ckpt)
        stage_archive = _archive_checkpoint(
            run_record, stage_ckpt, "stage_iteration_checkpoint",
            f"stage_{stage}_iter_{iteration:06d}.pt")
        entry["checkpoint"] = artifact_record(
            stage_ckpt, "stage_iteration_checkpoint")
        entry["checkpoint"]["run_id"] = run_record["run_id"]
        entry["checkpoint"]["archive"] = stage_archive
        add_output(run_record, stage_ckpt, "stage_latest_checkpoint")

        if "eval" in entry:
            rank = _evaluation_rank(entry["eval"])
            entry["eval"]["rank"] = list(rank)
            if best_rank is None or rank > best_rank:
                _save_checkpoint(pol, best_ckpt)
                best_archive = _archive_checkpoint(
                    run_record, best_ckpt, "stage_best_checkpoint",
                    f"stage_{stage}_best_iter_{iteration:06d}.pt")
                entry["eval"]["best_checkpoint"] = artifact_record(
                    best_ckpt, "stage_best_checkpoint")
                entry["eval"]["best_checkpoint"]["run_id"] = run_record["run_id"]
                entry["eval"]["best_checkpoint"]["archive"] = best_archive
                add_output(run_record, best_ckpt, "stage_best_checkpoint")
                best_rank = rank
        _persist_run_record(log, run_record)
        print(
            f"[{stage} {iteration:03d}] train_ret={entry['mean_return']:8.1f} "
            f"succ={entry['success']:.0%} valid_ptr={entry['valid_request_rate']:.3f} "
            f"int={entry['intervention_rate']:.3f} "
            f"progress={entry['max_stable_progress']:02d} "
            f"deltas={deltas}", flush=True)

    _save_checkpoint(pol, final_ckpt)
    final_archive = _archive_checkpoint(
        run_record, final_ckpt, "stage_final_checkpoint",
        f"stage_{stage}_final.pt")
    add_output(run_record, final_ckpt, "stage_final_checkpoint")
    run_record["final_checkpoint"] = final_archive
    _persist_run_record(log, run_record)
    print(f"[stage {stage}] saved {final_ckpt}; wall "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def _eval_only() -> None:
    stage = _stage_arg(os.environ.get("MUMT_STAGED_EVAL_STAGE", "c"))
    env = DetectEnv()
    shield = DetectQPShield(env.cfg)
    pol = DetectMAPPOPolicy(env, shield, seed=0)
    source = os.environ.get("MUMT_STAGED_EVAL_SOURCE")
    if source:
        checkpoint = Path(source)
    else:
        checkpoint = OUT / f"stage_{stage}_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing evaluation checkpoint: {checkpoint}")
    seeds = [int(x) for x in os.environ.get(
        "MUMT_STAGED_EVAL_SEEDS", "300,301,302").split(",") if x.strip()]
    k_red = _env_int("MUMT_STAGED_K", 2)
    run_record = capture_run_provenance(
        repo_root=ROOT,
        output_dir=OUT,
        entry_scripts=[ROOT / "_p3_staged.py"],
        phase="staged_eval",
        task_config=env.cfg,
        algorithm_config=pol.config,
        runtime_parameters={
            "environment": {key: value for key, value in os.environ.items()
                            if key.startswith("MUMT_")},
            "stage": stage,
            "k_red": k_red,
            "evaluation_seeds": seeds,
            "source_checkpoint": str(checkpoint.resolve()),
        },
        inputs=[],
        source_snapshot=SOURCE_SNAPSHOT,
    )
    log = _load_log()
    register_active_run(
        run_record, lambda record: _persist_run_record(_load_log(), record))
    _persist_run_record(log, run_record)
    _load_policy_checkpoint(pol, checkpoint, run_record,
                            "evaluation_checkpoint")
    rows = _evaluate(pol, shield, seeds, k_red)
    result = {
        "run_id": run_record["run_id"],
        "checkpoint": artifact_record(checkpoint, "evaluation_checkpoint"),
        "rows": rows,
    }
    report_path = OUT / "provenance" / f"{run_record['run_id']}_evaluation.json"
    write_json_atomic(report_path, result)
    add_output(run_record, report_path, "staged_evaluation_report")
    result["report"] = artifact_record(report_path, "staged_evaluation_report")
    log.setdefault("evaluations", []).append(result)
    _persist_run_record(log, run_record)
    print(json.dumps(result, ensure_ascii=False, indent=1))


@track_run_lifecycle
def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: _p3_staged.py a|b|c|eval [iter0] [n_iters]")
    command = sys.argv[1].lower()
    if command == "eval":
        _eval_only()
        return
    stage = _stage_arg(command)
    iter0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_ITERS[stage]
    _train(stage, iter0, n_iters)


if __name__ == "__main__":
    main()
