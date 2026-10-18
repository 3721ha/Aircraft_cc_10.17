# -*- coding: utf-8 -*-
"""P3 pilot: BC warm start (HEUR) + short MAPPO run + Gate-4 metrics.

Phases:
  1. BC   -- imitate HeurPolicy demos (plan §9.4, 50-100 epochs), then eval
  2. MAPPO pilot -- iterations of (collect N episodes -> PPO update)
  3. Gate 4 readout -- intervention-type distribution (continuous QP share
     must be >= 30% of interventions) + reward/violation trend

Outputs: runs/detect_pilot/pilot_log.json, pilot_bc.pt, pilot_final.pt
Run from workspace root:  Aircraft_mumt/.venv/Scripts/python.exe _p3_pilot.py
"""
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))
from _experiment_provenance import (add_input, add_output, artifact_record,
                                    canonical_sha256, capture_run_provenance,
                                    capture_source_snapshot,
                                    register_active_run, track_run_lifecycle,
                                    write_json_atomic)

OUT = Path(os.environ.get(
    "MUMT_PILOT_OUT",
    str(ROOT / "Aircraft_mumt" / "runs" / "detect_pilot")))
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_SNAPSHOT = capture_source_snapshot(
    ROOT, OUT, [ROOT / "_p3_pilot.py"])

from aircraft_sim.detect_baselines import HeurPolicy              # noqa: E402
from aircraft_sim.detect_env import DetectEnv                     # noqa: E402
from aircraft_sim.detect_config import DetectSimConfig             # noqa: E402
from aircraft_sim.detect_mappo import DetectMAPPOPolicy           # noqa: E402
from aircraft_sim.detect_shield import DetectQPShield             # noqa: E402

# CLI: _p3_pilot.py [phase] ; phase = bc | pilot <iter0> <n_iters> | eval
#      | curr <k> <iter0> <n_iters>  (curriculum: train at K=k, eval each 10)
PHASE = sys.argv[1] if len(sys.argv) > 1 else "all"
ITERS0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
CURR_K = int(sys.argv[2]) if PHASE == "curr" and len(sys.argv) > 2 else 2
if PHASE == "curr":
    ITERS0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    N_ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 50


def _env_config() -> DetectSimConfig:
    """Build the task config with explicit, reproducible numeric overrides."""
    values = {}
    for name, caster in (
            ("MUMT_STABLE_WINDOW", int),
            ("MUMT_AGE_STD_TRACK", float),
            ("MUMT_SECTOR_AXIS_MAX_DEG", float),
            ("MUMT_C04_RANGE", float),
            ("MUMT_ARB_BID_WEIGHT", float),
            ("MUMT_ARB_HYSTERESIS", float)):
        text = os.environ.get(name)
        if text is not None:
            values[name.removeprefix("MUMT_")] = caster(text)
    return replace(DetectSimConfig(), **values) if values else DetectSimConfig()


def _make_env() -> DetectEnv:
    return DetectEnv(_env_config())

BC_EPISODES = 8          # K=4, seeds 100+
BC_EPOCHS = 60
BC_CONTINUOUS_OBJECTIVE = os.environ.get(
    "MUMT_BC_CONTINUOUS_OBJECTIVE", "likelihood")
BC_K = int(os.environ.get("MUMT_BC_K", "4"))
BC_SHIELD_LABELS = os.environ.get("MUMT_BC_SHIELD_LABELS", "1") != "0"
BC_SEED_START = int(os.environ.get("MUMT_BC_SEED_START", "100"))
CURRICULUM_START = int(os.environ.get("MUMT_W8_START", "0"))
CURRICULUM_SPAN = int(os.environ.get("MUMT_W8_SPAN", "100"))
FIXED_W8_TEXT = os.environ.get("MUMT_W8_FIXED")
FIXED_W8_SCALE = None if FIXED_W8_TEXT is None else float(FIXED_W8_TEXT)
if FIXED_W8_SCALE is not None and not 0.0 <= FIXED_W8_SCALE <= 1.0:
    raise ValueError("MUMT_W8_FIXED must be in [0, 1]")
if CURRICULUM_SPAN <= 0:
    raise ValueError("MUMT_W8_SPAN must be positive")
EVAL_SEEDS = [200, 201, 202, 203, 204]
EVAL_K = int(os.environ.get("MUMT_EVAL_K", "4"))
PPO_K = int(os.environ.get("MUMT_PPO_K", "4"))
EPISODES_PER_ITER = int(os.environ.get("MUMT_EPISODES_PER_ITER", "4"))
VALIDATION_EVERY = int(os.environ.get("MUMT_VALIDATION_EVERY", "10"))
VALIDATION_SEEDS = [int(value) for value in os.environ.get(
    "MUMT_VALIDATION_SEEDS", "200,201,202,203,204,205,206,207,208,209"
).split(",") if value.strip()]
CURR_INIT = os.environ.get("MUMT_CURR_INIT")

if EPISODES_PER_ITER < 1:
    raise ValueError("MUMT_EPISODES_PER_ITER must be at least 1")
if VALIDATION_EVERY < 1:
    raise ValueError("MUMT_VALIDATION_EVERY must be at least 1")
if not VALIDATION_SEEDS:
    raise ValueError("MUMT_VALIDATION_SEEDS must contain at least one seed")


def evaluate(pol, shield, seeds, k=EVAL_K):
    rows = []
    for s in seeds:
        env = _make_env()
        env.reset({"seed": s, "k_red": k})
        pol.env = env
        ret, steps, vio = 0.0, 0, 0
        while env.terminal is None:
            obs = env.observe()
            nom = {i: pol.act_with_log_prob(obs[i], deterministic=True)[0] for i in obs}
            safe, _ = shield.filter(env, nom)
            res = env.step(safe)
            ret += res.reward
            vio += len(res.info["report"].violations)
            steps += 1
        rows.append({"seed": s, "terminal": env.terminal, "return": round(ret, 1),
                     "steps": steps, "violations": vio})
    return rows


def evaluate_heuristic(shield, seeds, k):
    rows = []
    for seed in seeds:
        env = _make_env()
        env.reset({"seed": seed, "k_red": k})
        heur = HeurPolicy()
        ret, steps, vio = 0.0, 0, 0
        while env.terminal is None:
            nominal = heur(env)
            safe, _ = shield.filter(env, nominal)
            result = env.step(safe)
            ret += result.reward
            vio += len(result.info["report"].violations)
            steps += 1
        rows.append({"seed": seed, "terminal": env.terminal,
                     "return": round(ret, 1), "steps": steps,
                     "violations": vio})
    return rows


def _eval_summary(rows):
    steps = sum(row["steps"] for row in rows)
    return {
        "success_rate": sum(row["terminal"] == "success" for row in rows)
                        / max(1, len(rows)),
        "mean_return": round(sum(row["return"] for row in rows)
                             / max(1, len(rows)), 1),
        "violations_per_step": round(
            sum(row["violations"] for row in rows) / max(1, steps), 4),
    }


def _w8_scale_for_iter(iteration):
    if FIXED_W8_SCALE is not None:
        return min(1.0, max(0.0, FIXED_W8_SCALE))
    return 0.2 + 0.8 * min(
        1.0, max(0.0, (iteration - CURRICULUM_START) / CURRICULUM_SPAN))


def evaluate_paired(pol, shield, seeds, k):
    rl_rows = evaluate(pol, shield, seeds, k=k)
    heur_rows = evaluate_heuristic(shield, seeds, k=k)
    return {
        "k": k,
        "seeds": list(seeds),
        "stable_window": _env_config().STABLE_WINDOW,
        "task_config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(_env_config()).items()
        },
        "task_config_sha256": canonical_sha256(_env_config()),
        "rl": _eval_summary(rl_rows),
        "heur": _eval_summary(heur_rows),
        "by_seed": [
            {"seed": seed, "rl": rl, "heur": heur}
            for seed, rl, heur in zip(seeds, rl_rows, heur_rows)
        ],
    }


def _validation_rank(paired):
    """Prioritize task success, then return, then fewer rule violations."""
    rl = paired["rl"]
    return (float(rl["success_rate"]), float(rl["mean_return"]),
            -float(rl["violations_per_step"]))


def _save_checkpoint(pol, path):
    """Write a complete checkpoint before replacing the visible filename."""
    path = Path(path)
    temp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    pol.save(str(temp_path))
    temp_path.replace(path)


def _best_validation_record(log, key, best_path):
    record = log.get(key)
    if not best_path.exists() or not isinstance(record, dict):
        return None
    validation = record.get("validation")
    if not isinstance(validation, dict):
        return None
    return _validation_rank(validation)


def _load_log():
    f = OUT / "pilot_log.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"iters": []}


def _write_log(log):
    write_json_atomic(OUT / "pilot_log.json", log)


def _run_metrics_snapshot(log, run_record):
    """Return the metrics attributable to this invocation only."""
    run_id = run_record["run_id"]
    metrics = {
        "run_id": run_id,
        "phase": run_record["phase"],
        "task_config_sha256": run_record["task_config_sha256"],
        "algorithm_config_sha256": run_record["algorithm_config_sha256"],
        "source_manifest_sha256": run_record["effective_source"][
            "manifest_sha256"],
        "records": {},
    }
    for key in ("bc", "final_eval_run"):
        value = log.get(key)
        if isinstance(value, dict) and value.get("run_id") == run_id:
            metrics["records"][key] = value
    for key in ("iters", "pilot"):
        values = log.get(key, [])
        if isinstance(values, list):
            rows = [row for row in values
                    if isinstance(row, dict) and row.get("run_id") == run_id]
            if rows:
                metrics["records"][key] = rows
    for key, values in log.items():
        if not key.startswith("curr_k") or not isinstance(values, list):
            continue
        rows = [row for row in values
                if isinstance(row, dict) and row.get("run_id") == run_id]
        if rows:
            metrics["records"][key] = rows
    for key, value in log.items():
        if not key.endswith("_best") or not isinstance(value, dict):
            continue
        validation = value.get("validation")
        if (isinstance(validation, dict)
                and validation.get("run_id") == run_id):
            metrics["records"][key] = value
    for key in ("pilot_validation", "curr_k2_validation", "curr_k3_validation",
                "curr_k4_validation"):
        value = log.get(key)
        if isinstance(value, dict) and value.get("run_id") == run_id:
            metrics["records"][key] = value
    metrics["training_state_semantics"] = run_record.get(
        "training_state_semantics")
    metrics["inputs"] = run_record.get("inputs", [])
    metrics["checkpoint_archives"] = run_record.get(
        "checkpoint_archives", [])
    return metrics


def _archive_shared_log(run_record):
    """Preserve the previous shared log before this invocation can reset it."""
    source = OUT / "pilot_log.json"
    if not source.is_file():
        return
    before = artifact_record(source, "previous_shared_log")
    archive = (OUT / "provenance" /
               f"{run_record['run_id']}_previous_pilot_log.json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archive)
    snapshot = artifact_record(archive, "previous_shared_log_snapshot")
    if before["sha256"] != snapshot["sha256"]:
        raise RuntimeError(f"shared log changed while archiving: {source}")
    before["archived_snapshot"] = snapshot
    run_record.setdefault("inputs", []).append(before)


def _reconcile_run_manifests(log):
    """Repair sidecars when a previous process stopped after writing the log."""
    changed = False
    for record in log.get("run_provenance", []):
        if not isinstance(record, dict) or not record.get("run_id"):
            continue
        manifest_path = Path(record.get(
            "manifest_path", OUT / "provenance" / f"{record['run_id']}.json"))
        metrics_path = (OUT / "provenance" /
                        f"{record['run_id']}_metrics.json")
        if metrics_path.is_file():
            current_metrics = artifact_record(metrics_path, "run_metrics")
            if record.get("metrics_artifact") != current_metrics:
                record["metrics_artifact"] = current_metrics
                changed = True
        payload = {
            "run_provenance": record,
            "run_record_sha256": canonical_sha256(record),
            "metrics_artifact": record.get("metrics_artifact"),
        }
        current = None
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if current != payload:
            write_json_atomic(manifest_path, payload)
    if changed:
        _write_log(log)


def _persist_run_record(log, run_record):
    manifest_path = (OUT / "provenance" /
                     f"{run_record['run_id']}.json")
    run_record["manifest_path"] = str(manifest_path.resolve())
    metrics_path = (OUT / "provenance" /
                    f"{run_record['run_id']}_metrics.json")
    run_record["metrics_path"] = str(metrics_path.resolve())
    records = log.setdefault("run_provenance", [])
    for index, record in enumerate(records):
        if record.get("run_id") == run_record["run_id"]:
            records[index] = run_record
            break
    else:
        records.append(run_record)
    write_json_atomic(metrics_path, _run_metrics_snapshot(log, run_record))
    run_record["metrics_artifact"] = artifact_record(
        metrics_path, "run_metrics")
    for index, record in enumerate(records):
        if record.get("run_id") == run_record["run_id"]:
            records[index] = run_record
            break
    _write_log(log)
    write_json_atomic(manifest_path, {
        "run_provenance": run_record,
        "run_record_sha256": canonical_sha256(run_record),
        "metrics_artifact": run_record["metrics_artifact"],
    })


def _load_policy_checkpoint(pol, path, run_record, role):
    path = Path(path).resolve()
    before = artifact_record(path, role)
    if not before["exists"]:
        raise FileNotFoundError(path)
    snapshot_path = (OUT / "provenance" /
                     f"{run_record['run_id']}_input_{role}_{path.name}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, snapshot_path)
    snapshot = artifact_record(snapshot_path, f"{role}_snapshot")
    if before["sha256"] != snapshot["sha256"]:
        raise RuntimeError(f"checkpoint changed while archiving: {path}")
    checkpoint = torch.load(
        snapshot_path, map_location="cpu", weights_only=False)
    pol.load(str(snapshot_path))
    after = artifact_record(path, role)
    if before["sha256"] != after["sha256"]:
        raise RuntimeError(f"checkpoint changed while loading: {path}")
    saved_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    before["checkpoint_saved_config"] = saved_config
    before["checkpoint_config_sha256"] = (
        canonical_sha256(saved_config) if saved_config is not None else None)
    before["archived_snapshot"] = snapshot
    run_record.setdefault("inputs", []).append(before)


def _archive_checkpoint(run_record, path, role, archive_name):
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


def _configure_policy(pol):
    """Apply experiment-only PPO knobs without changing the task contract."""
    lr_text = os.environ.get("MUMT_PPO_LR")
    if lr_text:
        lr = float(lr_text)
        pol.config.learning_rate = lr
        for group in pol.optimizer.param_groups:
            group["lr"] = lr
    epochs_text = os.environ.get("MUMT_PPO_EPOCHS")
    if epochs_text:
        pol.config.epochs = max(1, int(epochs_text))
    exposure_text = os.environ.get("MUMT_CONFLICT_EXPOSURE_PROB")
    if exposure_text:
        pol.config.conflict_exposure_prob = min(
            1.0, max(0.0, float(exposure_text)))
    if os.environ.get("MUMT_BID_ONLY_PPO"):
        pol.config.bid_only_ppo = os.environ["MUMT_BID_ONLY_PPO"] != "0"
    if os.environ.get("MUMT_RL_DIRECT_ARBITRATION"):
        pol.config.direct_rl_arbitration = (
            os.environ["MUMT_RL_DIRECT_ARBITRATION"] != "0")
    _apply_bid_std_override(pol)
    print(f"[PPO] lr={pol.config.learning_rate:g} epochs={pol.config.epochs} "
          f"shield_aware={pol.config.shield_aware_continuous} "
          f"cf_request_credit={pol.config.cf_request_credit:g} "
          f"conflict_exposure={pol.config.conflict_exposure_prob:g} "
          f"bid_only={pol.config.bid_only_ppo} "
          f"direct_rl_arbitration={pol.config.direct_rl_arbitration}", flush=True)


def _apply_bid_std_override(pol):
    bid_std_text = os.environ.get("MUMT_BID_LOG_STD")
    if bid_std_text:
        # Training-time exploration knob for the arbitration bid only.  The
        # deployed action remains the deterministic actor mean; this does not
        # alter the task rules or arbitration formula.
        pol.model.bid_log_std.data.fill_(float(bid_std_text))
        pol.model.bid_log_std.data.clamp_(-2.5, 0.0)


@track_run_lifecycle
def main():
    t_start = time.time()
    env = _make_env()
    shield = DetectQPShield(env.cfg)
    pol = DetectMAPPOPolicy(env, shield, seed=0)
    _configure_policy(pol)
    run_record = capture_run_provenance(
        repo_root=ROOT,
        output_dir=OUT,
        entry_scripts=[ROOT / "_p3_pilot.py"],
        phase=PHASE,
        task_config=env.cfg,
        algorithm_config=pol.config,
        runtime_parameters={
            "environment": {key: value for key, value in os.environ.items()
                            if key.startswith("MUMT_")},
            "bc": {"episodes": BC_EPISODES, "epochs": BC_EPOCHS,
                   "k": BC_K, "seed_start": BC_SEED_START,
                   "shield_labels": BC_SHIELD_LABELS,
                   "continuous_objective": BC_CONTINUOUS_OBJECTIVE},
            "pilot": {"k": PPO_K, "episodes_per_iter": EPISODES_PER_ITER,
                      "validation_every": VALIDATION_EVERY,
                      "validation_seeds": EVAL_SEEDS},
            "curriculum": {"k": CURR_K, "iter_start": ITERS0,
                           "iterations": N_ITERS,
                           "validation_seeds": VALIDATION_SEEDS,
                           "fixed_w8_scale": FIXED_W8_SCALE,
                           "w8_start": CURRICULUM_START,
                           "w8_span": CURRICULUM_SPAN},
        },
        source_snapshot=SOURCE_SNAPSHOT,
    )
    run_record["training_state_semantics"] = {
        "mode": "weights_only_warm_start",
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "note": "Checkpoint loading restores model weights only; optimizer and RNG are initialized for this invocation.",
    }
    log = _load_log()
    _reconcile_run_manifests(log)
    register_active_run(
        run_record, lambda record: _persist_run_record(_load_log(), record))
    _archive_shared_log(run_record)
    log.setdefault("run_provenance", []).append(run_record)
    _persist_run_record(log, run_record)

    # ---------------- Phase 1: BC ----------------
    if PHASE in ("all", "bc"):
        heur = HeurPolicy()
        demos = []
        for ep in range(BC_EPISODES):
            env.reset({"seed": BC_SEED_START + ep, "k_red": BC_K})
            obs = env.observe()
            while env.terminal is None:
                acts = heur(env)
                # The deployed actor is followed by the shield, so train the
                # continuous heads on the same executed controls.  Discrete
                # role/gear/pointer labels remain the teacher's decisions;
                # HEUR is still confined to this training-time data path.
                labels = acts
                executed = acts
                if BC_SHIELD_LABELS:
                    executed, _ = shield.filter(env, acts)
                    labels = executed
                for i in labels:
                    demos.append((pol.encoder.encode(obs[i]), labels[i]))
                env.step(executed)
                obs = env.observe()
        bc_stats = pol.behavior_clone(
            demos, epochs=BC_EPOCHS,
            continuous_objective=BC_CONTINUOUS_OBJECTIVE)
        pol.save(str(OUT / "pilot_bc.pt"))
        bc_archive = _archive_checkpoint(
            run_record, OUT / "pilot_bc.pt", "behavior_clone_checkpoint",
            "pilot_bc.pt")
        bc_eval = evaluate(pol, shield, EVAL_SEEDS, k=BC_K)
        print(f"[BC] {bc_stats} | eval: "
              f"{sum(r['terminal'] == 'success' for r in bc_eval)}/{len(bc_eval)} success, "
              f"mean return {sum(r['return'] for r in bc_eval) / len(bc_eval):.0f}",
              flush=True)
        log = _load_log()
        log["bc"] = {"k": BC_K, "seed_start": BC_SEED_START,
                     "run_id": run_record["run_id"],
                     "checkpoint": bc_archive,
                     "stats": bc_stats, "eval": bc_eval}
        add_output(run_record, OUT / "pilot_bc.pt", "behavior_clone_checkpoint")
        _persist_run_record(log, run_record)
        if PHASE == "bc":
            return

    # ---------------- Phase 2: MAPPO pilot ----------------
    if PHASE in ("all", "pilot"):
        pilot_latest_path = OUT / "pilot_iter.pt"
        if ITERS0 > 0 and pilot_latest_path.exists():
            _load_policy_checkpoint(
                pol, pilot_latest_path, run_record, "pilot_resume_checkpoint")
            print(f"[pilot] resume from {pilot_latest_path.name}", flush=True)
        else:
            pilot_init = OUT / "pilot_bc.pt"
            _load_policy_checkpoint(
                pol, pilot_init, run_record, "pilot_initial_checkpoint")
            print("[pilot] fresh start from pilot_bc.pt", flush=True)
        _persist_run_record(_load_log(), run_record)
        _apply_bid_std_override(pol)
        env = _make_env()
        pol.env = env
        log = _load_log()
        pilot_best_path = OUT / "pilot_best.pt"
        pilot_best_key = "pilot_best"
        if ITERS0 == 0:
            log["pilot"] = []
            log[pilot_best_key] = None
            pilot_best_rank = None
        else:
            log.setdefault("pilot", [])
            pilot_best_rank = _best_validation_record(
                log, pilot_best_key, pilot_best_path)
            if pilot_best_path.exists():
                inherited_best = _archive_checkpoint(
                    run_record, pilot_best_path, "pilot_best_checkpoint",
                    f"pilot_best_inherited_iter_{ITERS0:06d}.pt")
                if isinstance(log.get(pilot_best_key), dict):
                    validation = log[pilot_best_key].get("validation")
                    if isinstance(validation, dict):
                        validation.setdefault("best_checkpoint", {})[
                            "archive"] = inherited_best
                _persist_run_record(log, run_record)
        log.setdefault("experiment", {}).update({
            "pilot_episodes_per_iter": EPISODES_PER_ITER,
            "pilot_validation_every": VALIDATION_EVERY,
            "pilot_validation_seeds": EVAL_SEEDS,
        })
        for it in range(ITERS0, ITERS0 + N_ITERS):
            rollout, ep_rets, ep_term, ep_steps, ep_vio = [], [], [], [], []
            n_qp = n_fb = 0
            for ep in range(EPISODES_PER_ITER):
                data = pol.collect(
                    k_red=PPO_K,
                    seed=1000 + it * EPISODES_PER_ITER + ep)
                rollout.extend(data)
                ep_rets.append(sum(d["reward"] for d in data) / pol.config.reward_scale)
                ep_term.append(data[-1]["terminal"])
                ep_steps.append(len(data))
                ep_vio.append(sum(d["violations"] for d in data))
                n_qp += sum(d["n_qp"] for d in data)
                n_fb += sum(d["n_fallback"] for d in data)
            stats = pol.update(rollout)
            entry = {
                "iter": it,
                "run_id": run_record["run_id"],
                "mean_return": round(sum(ep_rets) / len(ep_rets), 1),
                "success": sum(t == "success" for t in ep_term) / len(ep_term),
                "mean_steps": round(sum(ep_steps) / len(ep_steps)),
                "vio_per_step": round(sum(ep_vio) / max(1, sum(ep_steps)), 3),
                "qp_share": round(n_qp / max(1, n_qp + n_fb), 3),
                "interventions_per_ep": round((n_qp + n_fb) / EPISODES_PER_ITER, 1),
                "policy_loss": round(stats["policy_loss"], 4),
                "value_loss": round(stats["value_loss"], 2),
                "approx_kl": round(stats["approx_kl"], 4),
            }
            log["iters"].append(entry)
            print(f"[{it:02d}] ret={entry['mean_return']:>8} succ={entry['success']:.0%} "
                  f"vio/step={entry['vio_per_step']:.2f} qp_share={entry['qp_share']:.2f} "
                  f"kl={entry['approx_kl']:.3f}", flush=True)
            _save_checkpoint(pol, pilot_latest_path)
            latest_archive = _archive_checkpoint(
                run_record, pilot_latest_path, "pilot_iteration_checkpoint",
                f"pilot_iter_{it:06d}.pt")
            add_output(run_record, pilot_latest_path, "pilot_latest_checkpoint")
            log["iters"][-1]["checkpoint"] = artifact_record(
                pilot_latest_path, "pilot_iteration_checkpoint")
            log["iters"][-1]["checkpoint"]["run_id"] = run_record["run_id"]
            log["iters"][-1]["checkpoint"]["archive"] = latest_archive
            should_validate = (
                (it + 1) % VALIDATION_EVERY == 0
                or it == ITERS0 + N_ITERS - 1)
            if should_validate:
                paired = evaluate_paired(
                    pol, shield, EVAL_SEEDS, k=PPO_K)
                candidate_rank = _validation_rank(paired)
                is_best = (pilot_best_rank is None
                            or candidate_rank > pilot_best_rank)
                log["iters"][-1]["validation"] = paired
                paired["run_id"] = run_record["run_id"]
                paired["source_manifest_sha256"] = run_record[
                    "effective_source"]["manifest_sha256"]
                paired["checkpoint"] = artifact_record(
                    pilot_latest_path, "pilot_validation_checkpoint")
                if is_best:
                    _save_checkpoint(pol, pilot_best_path)
                    best_archive = _archive_checkpoint(
                        run_record, pilot_best_path,
                        "pilot_best_checkpoint", f"pilot_best_iter_{it:06d}.pt")
                    add_output(run_record, pilot_best_path,
                               "pilot_best_checkpoint")
                    paired["best_checkpoint"] = artifact_record(
                        pilot_best_path, "pilot_best_checkpoint")
                    paired["best_checkpoint"]["archive"] = best_archive
                    pilot_best_rank = candidate_rank
                    log[pilot_best_key] = {
                        "iter": it,
                        "checkpoint": pilot_best_path.name,
                        "checkpoint_sha256": paired["best_checkpoint"]["sha256"],
                        "rank": list(candidate_rank),
                        "validation": paired,
                    }
                print(
                    f"  [paired validation pilot] "
                    f"RL={paired['rl']['success_rate']:.0%}/"
                    f"{paired['rl']['mean_return']:.0f} vs HEUR="
                    f"{paired['heur']['success_rate']:.0%}/"
                    f"{paired['heur']['mean_return']:.0f}; "
                    f"best={'updated' if is_best else 'kept'} "
                    f"({pilot_best_path.name})",
                    flush=True)
            _persist_run_record(log, run_record)
        if not pilot_best_path.exists():
            raise RuntimeError(
                f"No paired validation checkpoint available: {pilot_best_path}")
        pol.load(str(pilot_best_path))
        _save_checkpoint(pol, OUT / "pilot_final.pt")
        final_archive = _archive_checkpoint(
            run_record, OUT / "pilot_final.pt", "pilot_final_checkpoint",
            "pilot_final.pt")
        add_output(run_record, OUT / "pilot_final.pt", "pilot_final_checkpoint")
        log["pilot_validation"] = log[pilot_best_key]["validation"]
        log["pilot_final_checkpoint"] = final_archive
        _persist_run_record(log, run_record)

    # ---------------- Phase 2b: curriculum long run ----------------
    if PHASE == "curr":
        # A fresh curriculum run must start from the current BC policy.  An
        # old curr_k*.pt may contain a degraded policy from an earlier
        # experiment, so it is only valid when resuming with ITERS0 > 0.
        ckpt = OUT / f"curr_k{CURR_K}.pt"
        if ITERS0 == 0:
            loaded = None
            # pilot_bc.pt is the only preferred starting point for a fresh
            # curriculum run.  The other checkpoints are fallback sources,
            # never an existing curr_k*.pt.
            candidates = ([Path(CURR_INIT)] if CURR_INIT else []) + [
                OUT / "pilot_bc.pt", OUT / "dagger_latest.pt", OUT / "pilot_final.pt"]
            for candidate in candidates:
                if not candidate.is_absolute():
                    candidate = ROOT / candidate
                if candidate.exists():
                    _load_policy_checkpoint(
                        pol, candidate, run_record,
                        "curriculum_initial_checkpoint")
                    loaded = candidate
                    break
            if loaded is None:
                raise FileNotFoundError(
                    "No fresh curriculum checkpoint found; expected pilot_bc.pt")
            print(f"[curr K={CURR_K}] fresh start from {loaded.name}", flush=True)
        else:
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Cannot resume curr K={CURR_K} at iter {ITERS0}: "
                    f"missing {ckpt.name}")
            _load_policy_checkpoint(
                pol, ckpt, run_record, "curriculum_resume_checkpoint")
            print(f"[curr K={CURR_K}] resume from {ckpt.name}", flush=True)
        _persist_run_record(_load_log(), run_record)
        _apply_bid_std_override(pol)
        env = _make_env()
        pol.env = env
        log = _load_log()
        ck = f"curr_k{CURR_K}"
        log.setdefault("experiment", {}).update({
            "curriculum_k": CURR_K,
            "bc_k": BC_K,
            "fixed_w8_scale": FIXED_W8_SCALE,
            "w8_start": CURRICULUM_START,
            "w8_span": CURRICULUM_SPAN,
            "episodes_per_iter": EPISODES_PER_ITER,
            "validation_every": VALIDATION_EVERY,
            "validation_seeds": VALIDATION_SEEDS,
            "task_numeric_config": {
                "stable_window": env.cfg.STABLE_WINDOW,
                "age_std_track": env.cfg.AGE_STD_TRACK,
                "sector_axis_max_deg": env.cfg.SECTOR_AXIS_MAX_DEG,
                "c04_range": env.cfg.C04_RANGE,
                "arb_bid_weight": env.cfg.ARB_BID_WEIGHT,
                "arb_hysteresis": env.cfg.ARB_HYSTERESIS,
            },
            "ppo_config": {
                "learning_rate": pol.config.learning_rate,
                "epochs": pol.config.epochs,
                "bid_only_ppo": pol.config.bid_only_ppo,
                "direct_rl_arbitration": pol.config.direct_rl_arbitration,
                "conflict_exposure_prob": (
                    pol.config.conflict_exposure_prob),
            },
        })
        best_path = OUT / f"curr_k{CURR_K}_best.pt"
        best_key = f"{ck}_best"
        if ITERS0 == 0:
            log[ck] = []
            log[best_key] = None
            best_rank = None
        else:
            log.setdefault(ck, [])
            best_rank = _best_validation_record(log, best_key, best_path)
            if best_path.exists():
                inherited_best = _archive_checkpoint(
                    run_record, best_path, "curriculum_best_checkpoint",
                    f"curr_k{CURR_K}_best_inherited_iter_{ITERS0:06d}.pt")
                if isinstance(log.get(best_key), dict):
                    validation = log[best_key].get("validation")
                    if isinstance(validation, dict):
                        validation.setdefault("best_checkpoint", {})[
                            "archive"] = inherited_best
                _persist_run_record(log, run_record)
        for it in range(ITERS0, ITERS0 + N_ITERS):
            w8_scale = _w8_scale_for_iter(it)
            rollout, ep_rets, ep_term, ep_steps, ep_vio = [], [], [], [], []
            n_qp = n_fb = 0
            for ep in range(EPISODES_PER_ITER):
                data = pol.collect(
                                   k_red=CURR_K,
                                   seed=2000 + it * EPISODES_PER_ITER + ep,
                                   w8_scale=w8_scale)
                rollout.extend(data)
                ep_rets.append(sum(d["reward"] for d in data) / pol.config.reward_scale)
                ep_term.append(data[-1]["terminal"])
                ep_steps.append(len(data))
                ep_vio.append(sum(d["violations"] for d in data))
                n_qp += sum(d["n_qp"] for d in data)
                n_fb += sum(d["n_fallback"] for d in data)
            stats = pol.update(rollout)
            # Rollout diagnostics are aggregated from agent-level records.
            # They make it possible to distinguish an invalid request policy
            # from a tracking/progress problem without changing the scenario.
            n_agents = pol.env.cfg.N_WINGMEN
            agent_steps = len(rollout) * n_agents
            n_track_actions = sum(
                int(item.get("n_track_actions", 0)) for item in rollout)
            n_nonzero_pointer = sum(
                int(item.get("n_nonzero_pointer", 0)) for item in rollout)
            n_valid_requests = sum(
                int(item.get("n_valid_requests", 0)) for item in rollout)
            n_requests = sum(int(item.get("n_requests", 0)) for item in rollout)
            n_c04_pass_requests = sum(
                int(item.get("n_c04_pass_requests", 0)) for item in rollout)
            n_c04_rejected_requests = sum(
                int(item.get("n_c04_rejected_requests", 0)) for item in rollout)
            n_request_winners = sum(
                int(item.get("n_request_winners", 0)) for item in rollout)
            n_new_assignments = sum(
                int(item.get("n_new_assignments", 0)) for item in rollout)
            n_handoffs = sum(int(item.get("n_handoffs", 0)) for item in rollout)
            owned_target_steps = sum(
                int(item.get("owned_count", 0)) for item in rollout)
            stable_progress_ticks = sum(
                int(item.get("stable_progress_total", 0)) for item in rollout)
            stable_target_steps = sum(
                int(item.get("stable_targets_in_progress", 0))
                for item in rollout)
            n_interventions = sum(
                int(item.get("n_interventions", 0)) for item in rollout)
            n_bid_events = sum(
                int(item.get("n_bid_events", 0)) for item in rollout)
            n_bid_participants = sum(
                sum(bool(active) for active in item.get("bid_mask", []))
                for item in rollout)
            n_forced_overlap_pairs = sum(
                int(item.get("n_forced_overlap_pairs", 0)) for item in rollout)
            n_pointer_exposed = sum(
                int(item.get("n_pointer_exposed", 0)) for item in rollout)
            n_exposed_multi_candidate_events = sum(
                int(item.get("n_exposed_multi_candidate_events", 0))
                for item in rollout)
            n_bid_changed_winner = sum(
                int(item.get("n_bid_changed_winner", 0))
                for item in rollout)
            n_cf_bid_events = sum(
                int(item.get("n_cf_bid_events", 0)) for item in rollout)
            n_cf_bid_changed_winner = sum(
                int(item.get("n_cf_bid_changed_winner", 0))
                for item in rollout)
            n_cf_bid_directional_slots = sum(
                int(item.get("n_cf_bid_directional_slots", 0))
                for item in rollout)
            cf_bid_advantages = [
                abs(float(value))
                for item in rollout
                for agent_adv, agent_mask in zip(
                    item.get("bid_cf_adv_slots", []),
                    item.get("bid_cf_mask_slots", []))
                for value, active in zip(agent_adv, agent_mask)
                if active]
            conflict_feature_rows = [
                np.asarray(vector[-13:], dtype=float)
                for item in rollout
                for vector in item["obs"].values()
            ]
            conflict_feature_mean = np.mean(conflict_feature_rows, axis=0)
            conflict_feature_nonzero = np.mean(
                np.abs(conflict_feature_rows) > 1e-6, axis=0)
            correction_values = [
                float(magnitude)
                for item in rollout
                for magnitude in item.get("correction_magnitudes", [])
            ]
            max_stable_progress = max(
                (int(item.get("max_stable_progress", 0)) for item in rollout),
                default=0)
            entry = {
                "iter": it,
                "run_id": run_record["run_id"],
                "mean_return": round(sum(ep_rets) / len(ep_rets), 1),
                "success": sum(t == "success" for t in ep_term) / len(ep_term),
                "vio_per_step": round(sum(ep_vio) / max(1, sum(ep_steps)), 3),
                "qp_share": round(n_qp / max(1, n_qp + n_fb), 3),
                "approx_kl": round(stats["approx_kl"], 4),
                "bid_approx_kl": round(
                    stats.get("bid_approx_kl", 0.0), 6),
                "bid_post_update_kl": round(
                    stats.get("bid_post_update_kl", 0.0), 6),
                "bid_active_slot_count": int(
                    stats.get("bid_active_slot_count", 0)),
                "bid_active_slot_count_by_index": stats.get(
                    "bid_active_slot_count_by_index", [0] * 8),
                "cf_bid_informative_fraction": round(
                    stats.get("cf_bid_informative_fraction", 0.0), 6),
                "bid_gradient_norm": round(
                    stats.get("bid_gradient_norm", 0.0), 8),
                "bid_parameter_delta_l2": round(
                    stats.get("bid_parameter_delta_l2", 0.0), 8),
                "w8_scale": round(w8_scale, 2),
                "k": CURR_K,
                "agent_steps": agent_steps,
                "track_actions": n_track_actions,
                "track_action_rate": round(
                    n_track_actions / max(1, agent_steps), 4),
                "nonzero_pointer": n_nonzero_pointer,
                "nonzero_pointer_rate": round(
                    n_nonzero_pointer / max(1, agent_steps), 4),
                "valid_requests": n_valid_requests,
                "valid_request_rate": round(
                    n_valid_requests / max(1, agent_steps), 4),
                "requests": n_requests,
                "c04_pass_requests": n_c04_pass_requests,
                "c04_rejected_requests": n_c04_rejected_requests,
                "request_winners": n_request_winners,
                "new_assignments": n_new_assignments,
                "handoffs": n_handoffs,
                "mean_owned_targets": round(
                    owned_target_steps / max(1, len(rollout)), 4),
                "stable_target_steps": stable_target_steps,
                "stable_progress_ticks": stable_progress_ticks,
                "interventions": n_interventions,
                "intervention_rate": round(
                    n_interventions / max(1, agent_steps), 4),
                "mean_correction": round(
                    sum(correction_values) / max(1, len(correction_values)), 6),
                "max_stable_progress": max_stable_progress,
                "intervention_loss": round(
                    stats.get("intervention_loss", 0.0), 6),
                "intervention_fraction": round(
                    stats.get("intervention_fraction", 0.0), 6),
                "cf_nonzero_frac": round(
                    stats.get("cf_nonzero_frac", 0.0), 6),
                "gear_gate_active_frac": round(
                    stats.get("gear_gate_active_frac", 0.0), 6),
                "gear_gate_credit_mean": round(
                    stats.get("gear_gate_credit_mean", 0.0), 6),
                "bid_event_frac": round(
                    stats.get("bid_event_frac", 0.0), 6),
                "bid_action_mean": round(
                    stats.get("bid_action_mean", 0.0), 6),
                "bid_action_std": round(
                    stats.get("bid_action_std", 0.0), 6),
                "bid_events": n_bid_events,
                "bid_events_per_episode": round(
                    n_bid_events / max(1, EPISODES_PER_ITER), 3),
                "bid_participants": n_bid_participants,
                "bid_participation_rate": round(
                    n_bid_participants / max(1, agent_steps), 6),
                "bid_event_step_rate": round(
                    n_bid_events / max(1, len(rollout)), 6),
                "forced_overlap_pairs": n_forced_overlap_pairs,
                "pointer_exposed_agents": n_pointer_exposed,
                "exposed_multi_candidate_events": (
                    n_exposed_multi_candidate_events),
                "bid_changed_winner": n_bid_changed_winner,
                "cf_bid_events": n_cf_bid_events,
                "cf_bid_changed_winner": n_cf_bid_changed_winner,
                "cf_bid_directional_slots": n_cf_bid_directional_slots,
                "cf_bid_mean_abs_advantage": round(
                    float(np.mean(cf_bid_advantages))
                    if cf_bid_advantages else 0.0, 6),
                "rule_conflict_feature_mean": [
                    round(float(v), 6) for v in conflict_feature_mean],
                "rule_conflict_feature_nonzero_rate": [
                    round(float(v), 6) for v in conflict_feature_nonzero],
                "rule_conflict_any_rate": round(
                    float(np.mean(np.any(
                        np.abs(conflict_feature_rows) > 1e-6, axis=1))), 6),
            }
            log[ck].append(entry)
            print(f"[{ck} {it:02d}] episodes={EPISODES_PER_ITER} "
                  f"ret={entry['mean_return']:>8} succ={entry['success']:.0%} "
                  f"vio/step={entry['vio_per_step']:.2f} req={n_requests} "
                  f"C04={n_c04_pass_requests}/{n_c04_rejected_requests} "
                  f"assign={n_new_assignments}+{n_handoffs} "
                  f"bid_cf={n_cf_bid_changed_winner}/{n_cf_bid_events} "
                  f"bid_info={entry['cf_bid_informative_fraction']:.2f} "
                  f"bid_grad={entry['bid_gradient_norm']:.2e} "
                  f"bid_delta={entry['bid_parameter_delta_l2']:.2e} "
                  f"bid_kl={entry['bid_post_update_kl']:.2e} "
                  f"stable={stable_progress_ticks} qp={entry['qp_share']:.2f} "
                  f"kl={entry['approx_kl']:.3f}", flush=True)
            _save_checkpoint(pol, OUT / f"curr_k{CURR_K}.pt")
            latest_archive = _archive_checkpoint(
                run_record, OUT / f"curr_k{CURR_K}.pt",
                "curriculum_iteration_checkpoint",
                f"curr_k{CURR_K}_iter_{it:06d}.pt")
            add_output(run_record, OUT / f"curr_k{CURR_K}.pt",
                       "curriculum_latest_checkpoint")
            entry["checkpoint"] = artifact_record(
                OUT / f"curr_k{CURR_K}.pt", "curriculum_iteration_checkpoint")
            entry["checkpoint"]["run_id"] = run_record["run_id"]
            entry["checkpoint"]["archive"] = latest_archive
            should_validate = (
                (it + 1) % VALIDATION_EVERY == 0
                or it == ITERS0 + N_ITERS - 1)
            if should_validate:
                paired = evaluate_paired(
                    pol, shield, VALIDATION_SEEDS, k=CURR_K)
                paired["run_id"] = run_record["run_id"]
                paired["source_manifest_sha256"] = run_record[
                    "effective_source"]["manifest_sha256"]
                current_checkpoint = OUT / f"curr_k{CURR_K}.pt"
                paired["checkpoint"] = artifact_record(
                    current_checkpoint, "curriculum_validation_checkpoint")
                candidate_rank = _validation_rank(paired)
                is_best = best_rank is None or candidate_rank > best_rank
                log[ck][-1]["validation"] = paired
                if is_best:
                    _save_checkpoint(pol, best_path)
                    best_archive = _archive_checkpoint(
                        run_record, best_path, "curriculum_best_checkpoint",
                        f"curr_k{CURR_K}_best_iter_{it:06d}.pt")
                    add_output(run_record, best_path,
                               "curriculum_best_checkpoint")
                    paired["best_checkpoint"] = artifact_record(
                        best_path, "curriculum_best_checkpoint")
                    paired["best_checkpoint"]["archive"] = best_archive
                    best_rank = candidate_rank
                    log[best_key] = {
                        "iter": it,
                        "checkpoint": best_path.name,
                        "checkpoint_sha256": paired["best_checkpoint"]["sha256"],
                        "rank": list(candidate_rank),
                        "validation": paired,
                    }
                rl_metrics, heur_metrics = paired["rl"], paired["heur"]
                print(
                    f"  [paired validation K={CURR_K}] "
                    f"RL={rl_metrics['success_rate']:.0%}/"
                    f"{rl_metrics['mean_return']:.0f}/"
                    f"{rl_metrics['violations_per_step']:.4f} vs HEUR="
                    f"{heur_metrics['success_rate']:.0%}/"
                    f"{heur_metrics['mean_return']:.0f}/"
                    f"{heur_metrics['violations_per_step']:.4f}; "
                    f"best={'updated' if is_best else 'kept'} "
                    f"({best_path.name})",
                    flush=True)
            _persist_run_record(log, run_record)
        if best_path.exists():
            pol.load(str(best_path))
        best_record = log.get(best_key)
        if not isinstance(best_record, dict) or not isinstance(
                best_record.get("validation"), dict):
            raise RuntimeError(
                f"No paired validation checkpoint available: {best_path}")
        paired = best_record["validation"]
        rl_metrics, heur_metrics = paired["rl"], paired["heur"]
        checks = {
            "success_at_least_heur": (
                rl_metrics["success_rate"] >= heur_metrics["success_rate"]),
            "mean_return_at_least_heur": (
                rl_metrics["mean_return"] >= heur_metrics["mean_return"]),
            "violations_below_heur": (
                rl_metrics["violations_per_step"]
                < heur_metrics["violations_per_step"]),
        }
        paired["acceptance"] = {**checks, "passed": all(checks.values())}
        log[f"{ck}_validation"] = paired
        if best_path.exists():
            add_output(run_record, best_path, "curriculum_best_checkpoint")
            best_archive = next(
                (archive for archive in reversed(
                    run_record.get("checkpoint_archives", []))
                 if archive.get("role") == "curriculum_best_checkpoint_archive"),
                None)
            if best_archive is not None:
                paired["best_checkpoint"]["archive"] = best_archive
        print(
            f"[best paired validation K={CURR_K} iter={best_record['iter']}] RL success="
            f"{rl_metrics['success_rate']:.0%} vs HEUR="
            f"{heur_metrics['success_rate']:.0%}; return="
            f"{rl_metrics['mean_return']:.1f} vs HEUR "
            f"{heur_metrics['mean_return']:.1f}; "
            f"violations/step={rl_metrics['violations_per_step']:.4f} vs HEUR="
            f"{heur_metrics['violations_per_step']:.4f}; "
            f"acceptance={'PASS' if paired['acceptance']['passed'] else 'FAIL'}",
            flush=True)
        _persist_run_record(log, run_record)
        return

    # ---------------- Phase 3: final eval ----------------
    if PHASE in ("all", "eval"):
        final_checkpoint = OUT / "pilot_final.pt"
        _load_policy_checkpoint(
            pol, final_checkpoint, run_record,
            "final_evaluation_checkpoint")
        _persist_run_record(_load_log(), run_record)
        env = _make_env()
        pol.env = env
        final_eval = evaluate(pol, shield, EVAL_SEEDS)
        log = _load_log()
        log["final_eval"] = final_eval
        log["final_eval_run"] = {
            "run_id": run_record["run_id"],
            "checkpoint": artifact_record(
                final_checkpoint, "final_evaluation_checkpoint"),
            "task_config_sha256": run_record["task_config_sha256"],
            "source_manifest_sha256": run_record[
                "effective_source"]["manifest_sha256"],
        }
        succ = sum(r["terminal"] == "success" for r in final_eval) / len(final_eval)
        mean_ret = sum(r["return"] for r in final_eval) / len(final_eval)
        print(f"[FINAL eval] {succ:.0%} success, mean return {mean_ret:.0f}", flush=True)
        _persist_run_record(log, run_record)

    print(f"wall {round((time.time() - t_start) / 60, 1)} min -> {OUT}")


if __name__ == "__main__":
    main()
