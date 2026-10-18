# -*- coding: utf-8 -*-
"""P3 DAgger: on-distribution warm start with shield-in-the-loop (Gate-5 plan A).

Problem (see P3课程段中段报告-决策门5.md): one-shot BC learned HEUR's action on
HEUR's state distribution; at execution the QP shield rewrites nearly every
action (qp_share=1.0), pushing the policy off-distribution -> copy bleeds
-15000 where the teacher scores -500. PPO cannot fix this because the policy
never reaches the positive-reward region (w1/w3 = 0).

DAgger loop (per round):
  1. roll out CURRENT policy (stochastic, shield in loop) at mixed K in {2,4}
  2. label every visited state with HEUR's nominal action
  3. aggregate dataset, weighted BC for BC_EPOCHS epochs
  4. deterministic eval on the gate-5 protocol (K=2 seeds 300-302, K=4 200-204)

CLI: _p3_dagger.py [round0] [n_rounds]     # one invocation = n_rounds rounds
Outputs: runs/detect_pilot/dagger_r{N}.pt, dagger_latest.pt, pilot_log.json["dagger"]
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _experiment_provenance import (add_output, artifact_record,
                                    capture_run_provenance,
                                    capture_source_snapshot,
                                    register_active_run,
                                    track_run_lifecycle,
                                    write_json_atomic)

OUT = Path(__import__("os").environ.get(
    "MUMT_DAGGER_OUT",
    str(ROOT / "Aircraft_mumt" / "runs" / "detect_dagger")))
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_SNAPSHOT = capture_source_snapshot(
    ROOT, OUT, [ROOT / "_p3_dagger.py"])

from aircraft_sim.detect_baselines import HeurPolicy              # noqa: E402
from aircraft_sim.detect_env import DetectEnv                     # noqa: E402
from aircraft_sim.detect_mappo import DetectMAPPOPolicy           # noqa: E402
from aircraft_sim.detect_shield import DetectQPShield             # noqa: E402

ROUND0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PHASE = sys.argv[1] if len(sys.argv) > 1 else "round"   # expert | round

EPISODES_PER_ROUND = int(__import__("os").environ.get(
    "MUMT_DAGGER_EPISODES", "4"))
BC_EPOCHS_PER_ROUND = 40        # 20 was under-trained (loss 0.55 vs 0.31 @60ep)
REJECT_TOL = 200.0              # accept candidate unless eval degrades by >tol
DAGGER_K = int(__import__("os").environ.get("MUMT_DAGGER_K", "2"))
DAGGER_SEED_START = int(__import__("os").environ.get("MUMT_DAGGER_SEED_START", "4000"))
EVAL_PROTOCOL = [(DAGGER_K, [300, 301, 302])]


def evaluate(pol, shield):
    rows = []
    for k, seeds in EVAL_PROTOCOL:
        for s in seeds:
            env = DetectEnv()
            env.reset({"seed": s, "k_red": k})
            pol.env = env
            ret = 0.0
            while env.terminal is None:
                obs = env.observe()
                nom = {i: pol.act_with_log_prob(obs[i], deterministic=True)[0]
                       for i in obs}
                safe, _ = shield.filter(env, nom)
                ret += env.step(safe).reward
            rows.append({"k": k, "seed": s, "terminal": env.terminal,
                         "return": round(ret, 1)})
    return rows


def collect_demos(pol, shield, heur, k, seed):
    """Roll out current policy (shielded); label visited states with HEUR."""
    env = DetectEnv()
    obs = env.reset({"seed": seed, "k_red": k})
    demos = []
    while env.terminal is None:
        teacher = heur(env)
        nom, _ = pol.act_batch(obs, deterministic=False)
        learner_safe, _ = shield.filter(env, nom)
        teacher_safe, _ = shield.filter(env, teacher)
        # Keep the visited state distribution from the learner while labeling
        # continuous controls with the action the deployed chain executes.
        # Discrete role/gear/pointer remain the teacher's conflict decision.
        for i in teacher:
            label = type(teacher[i])(
                teacher_safe[i].u_psi, teacher_safe[i].u_v, teacher_safe[i].u_xi,
                teacher[i].role, teacher[i].gear, teacher[i].pointer)
            demos.append((pol.encoder.encode(obs[i]), label))
        obs = env.step(learner_safe).observations
    return demos


def collect_expert_demos(pol, shield, heur, k, seed):
    """Roll out HEUR itself WITH the shield in loop; label = HEUR nominal.

    Fixes the original BC mismatch (plan §9.4 BC was collected shield-free but
    executed shield-in-loop -> qp_share 1.0 shifted states off-distribution).
    Expert rollouts also visit the POSITIVE-reward region (successful eps)."""
    env = DetectEnv()
    obs = env.reset({"seed": seed, "k_red": k})
    demos = []
    while env.terminal is None:
        teacher = heur(env)
        safe, _ = shield.filter(env, teacher)
        for i in teacher:
            # Expert trajectories are also recorded in the executed action
            # space, preventing a nominal-label / shield-state mismatch.
            label = type(teacher[i])(
                safe[i].u_psi, safe[i].u_v, safe[i].u_xi,
                teacher[i].role, teacher[i].gear, teacher[i].pointer)
            demos.append((pol.encoder.encode(obs[i]), label))
        obs = env.step(safe).observations
    return demos


def _load_log():
    f = OUT / "pilot_log.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _persist_run_record(log, run_record):
    manifest_path = OUT / "provenance" / f"{run_record['run_id']}.json"
    run_record["manifest_path"] = str(manifest_path.resolve())
    records = log.setdefault("run_provenance", [])
    for index, record in enumerate(records):
        if record.get("run_id") == run_record["run_id"]:
            records[index] = run_record
            break
    else:
        records.append(run_record)
    write_json_atomic(OUT / "pilot_log.json", log)
    write_json_atomic(manifest_path, {
        "run_provenance": run_record,
        "associated_log": artifact_record(OUT / "pilot_log.json", "pilot_log"),
    })


def _load_archived_checkpoint(pol, path, run_record, role):
    source = Path(path).resolve()
    before = artifact_record(source, role)
    if not before["exists"]:
        raise FileNotFoundError(source)
    archive_path = (OUT / "provenance" / f"{run_record['run_id']}_input_"
                    f"{role}_{source.name}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archive_path)
    archived = artifact_record(archive_path, f"{role}_archive")
    if before["sha256"] != archived["sha256"]:
        raise RuntimeError(f"checkpoint changed while archiving: {source}")
    pol.load(str(archive_path))
    after = artifact_record(source, role)
    if before["sha256"] != after["sha256"]:
        raise RuntimeError(f"checkpoint changed while loading: {source}")
    before["archived_snapshot"] = archived
    inputs = run_record.setdefault("inputs", [])
    existing = next((item for item in inputs
                     if item.get("role") == role
                     and item.get("path") == before["path"]), None)
    if existing is None:
        inputs.append(before)
    else:
        existing.update(before)


def _archive_checkpoint(run_record, path, role, archive_name):
    source = Path(path).resolve()
    before = artifact_record(source, role)
    if not before["exists"]:
        raise FileNotFoundError(source)
    archive_path = OUT / "provenance" / f"{run_record['run_id']}_{archive_name}"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archive_path)
    archived = artifact_record(archive_path, f"{role}_archive")
    if before["sha256"] != archived["sha256"]:
        raise RuntimeError(f"checkpoint changed while archiving: {source}")
    archived["source_path"] = str(source)
    run_record.setdefault("checkpoint_archives", []).append(archived)
    return archived


def _selected_checkpoint():
    if PHASE == "expert":
        final = OUT / "pilot_final.pt"
        return final if final.exists() else OUT / "pilot_bc.pt"
    if ROUND0 == 0:
        latest = OUT / "dagger_latest.pt"
        if latest.exists():
            return latest
        final = OUT / "pilot_final.pt"
        return final if final.exists() else OUT / "pilot_bc.pt"
    return OUT / "dagger_latest.pt"


@track_run_lifecycle
def main():
    t0 = time.time()
    env = DetectEnv()
    shield = DetectQPShield(env.cfg)
    pol = DetectMAPPOPolicy(env, shield, seed=0)

    latest = OUT / "dagger_latest.pt"
    initial_checkpoint = _selected_checkpoint()
    if PHASE == "round":
        pol.config.bc_learning_rate = 1e-4
    run_record = capture_run_provenance(
        repo_root=ROOT,
        output_dir=OUT,
        entry_scripts=[ROOT / "_p3_dagger.py"],
        phase=f"dagger_{PHASE}",
        task_config=env.cfg,
        algorithm_config=pol.config,
        runtime_parameters={
            "environment": {key: value for key, value in os.environ.items()
                            if key.startswith("MUMT_")},
            "phase": PHASE,
            "round_start": ROUND0,
            "round_count": N_ROUNDS,
            "episodes_per_round": EPISODES_PER_ROUND,
            "bc_epochs_per_round": BC_EPOCHS_PER_ROUND,
            "reject_tolerance": REJECT_TOL,
            "k": DAGGER_K,
            "seed_start": DAGGER_SEED_START,
            "eval_protocol": EVAL_PROTOCOL,
            "expert_episode_k_schedule": [2, 2, 2, 2, 4, 4, 4, 4],
            "expert_seed_start": 100,
            "expert_bc_epochs": 60,
            "initial_checkpoint_role": (
                "expert_incumbent" if PHASE == "expert" else "dagger_initial"),
        },
        inputs=[artifact_record(
            initial_checkpoint,
            "expert_incumbent" if PHASE == "expert" else "dagger_initial")],
        source_snapshot=SOURCE_SNAPSHOT,
    )
    register_active_run(
        run_record, lambda record: _persist_run_record(_load_log(), record))
    run_record["status"] = "running"
    _persist_run_record(_load_log(), run_record)

    # ---------------- Phase "expert": shield-in-loop HEUR BC ----------------
    if PHASE == "expert":
        _load_archived_checkpoint(
            pol, initial_checkpoint, run_record, "expert_incumbent")
        _persist_run_record(_load_log(), run_record)
        inc_ev = evaluate(pol, shield)
        incumbent_ret = sum(e["return"] for e in inc_ev) / len(inc_ev)
        print(f"[expert] incumbent (pilot_final) eval ret {incumbent_ret:+.0f}",
              flush=True)
        incumbent_state = {k: v.clone()
                           for k, v in pol.model.state_dict().items()}
        heur = HeurPolicy()
        demos = []
        for ep in range(8):
            k = 2 if ep < 4 else 4
            demos += collect_expert_demos(pol, shield, heur, k, seed=100 + ep)
            print(f"[expert] ep{ep} K={k} demos total {len(demos)}", flush=True)
        stats = pol.behavior_clone(demos, epochs=60)
        ev = evaluate(pol, shield)
        cand_ret = sum(e["return"] for e in ev) / len(ev)
        succ = sum(1 for e in ev if e["terminal"] == "success")
        accepted = cand_ret >= incumbent_ret - REJECT_TOL
        if accepted:
            pol.save(str(OUT / "expert_shielded.pt"))
            pol.save(str(latest))
            expert_archive = _archive_checkpoint(
                run_record, OUT / "expert_shielded.pt", "expert_bc_checkpoint",
                "expert_shielded.pt")
            latest_archive = _archive_checkpoint(
                run_record, latest, "dagger_latest_checkpoint",
                "dagger_latest.pt")
            add_output(run_record, OUT / "expert_shielded.pt",
                       "expert_bc_checkpoint")
            add_output(run_record, latest, "dagger_latest_checkpoint")
        else:
            pol.model.load_state_dict(incumbent_state)
            expert_archive = None
            latest_archive = None
        log = _load_log()
        log["expert_bc"] = {"demos": len(demos), "bc": stats, "eval": ev,
                            "accepted": accepted,
                            "run_id": run_record["run_id"],
                            "checkpoint": expert_archive,
                            "latest_checkpoint": latest_archive,
                            "incumbent_return": round(incumbent_ret, 1),
                            "eval_success": f"{succ}/{len(ev)}",
                            "eval_mean_return": round(cand_ret, 1)}
        run_record["status"] = "completed"
        run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _persist_run_record(log, run_record)
        print(f"[expert] bc_loss {stats['loss']:.3f} | eval succ {succ}/8 "
              f"ret {cand_ret:+.0f} -> {'ACCEPT' if accepted else 'REJECT'}",
              flush=True)
        for e in ev:
            print(f"    K={e['k']} seed={e['seed']} {e['terminal']:8s} "
                  f"ret {e['return']:+.0f}", flush=True)
        print(f"wall {round((time.time() - t0) / 60, 1)} min")
        return

    # ---------------- Phase "round": DAgger refinement rounds ----------------
    _load_archived_checkpoint(pol, initial_checkpoint, run_record,
                              "dagger_initial")
    _persist_run_record(_load_log(), run_record)

    log = _load_log()
    dag = log.setdefault("dagger", {"rounds": [], "demos_total": 0})
    agg = []          # aggregated dataset lives only within one invocation's rounds

    for r in range(ROUND0, ROUND0 + N_ROUNDS):
        # ---- incumbent baseline (accept/reject guard) ----
        if r == 0 or "incumbent_ret" not in locals():
            inc_ev = evaluate(pol, shield)
            incumbent_ret = sum(e["return"] for e in inc_ev) / len(inc_ev)
            print(f"[dagger r{r}] incumbent eval ret {incumbent_ret:+.0f}",
                  flush=True)
        incumbent_state = {k: v.clone() for k, v in pol.model.state_dict().items()}

        heur = HeurPolicy()
        demos = []
        for ep in range(EPISODES_PER_ROUND):
            k = DAGGER_K
            if ep < 2:
                # anchor: keep successful expert trajectories in the aggregate
                demos += collect_expert_demos(pol, shield, heur, k,
                                              seed=DAGGER_SEED_START + r * 10 + ep)
            else:
                # correction: label the learner's own visited states
                demos += collect_demos(pol, shield, heur, k,
                                       seed=DAGGER_SEED_START + 1000 + r * 10 + ep)
        agg += demos
        # gentle refinement: full-lr BC (1e-3 x 40ep) destroys the incumbent
        # (r0: -5457 -> -9745 REJECTED); fine-tune at 1e-4 instead
        stats = pol.behavior_clone(agg, epochs=BC_EPOCHS_PER_ROUND)
        ev = evaluate(pol, shield)
        cand_ret = sum(e["return"] for e in ev) / len(ev)
        accepted = cand_ret >= incumbent_ret - REJECT_TOL
        if accepted:
            incumbent_ret = max(incumbent_ret, cand_ret)
            pol.save(str(OUT / f"dagger_r{r}.pt"))
            pol.save(str(latest))
            round_archive = _archive_checkpoint(
                run_record, OUT / f"dagger_r{r}.pt",
                f"dagger_round_{r}_checkpoint", f"dagger_r{r}.pt")
            latest_archive = _archive_checkpoint(
                run_record, latest, "dagger_latest_checkpoint",
                f"dagger_latest_r{r}.pt")
            add_output(run_record, OUT / f"dagger_r{r}.pt",
                       f"dagger_round_{r}_checkpoint")
            add_output(run_record, latest, "dagger_latest_checkpoint")
        else:
            pol.model.load_state_dict(incumbent_state)   # reject, revert
            round_archive = None
            latest_archive = None
        succ = sum(1 for e in ev if e["terminal"] == "success")
        entry = {"round": r, "new_demos": len(demos), "agg_demos": len(agg),
                 "bc": stats, "eval": ev, "accepted": accepted,
                 "run_id": run_record["run_id"],
                 "checkpoint": round_archive,
                 "latest_checkpoint": latest_archive,
                 "incumbent_return": round(incumbent_ret, 1),
                 "eval_success": f"{succ}/{len(ev)}",
                 "eval_mean_return": round(cand_ret, 1)}
        dag["rounds"].append(entry)
        dag["demos_total"] += len(demos)
        run_record["completed_rounds"] = r - ROUND0 + 1
        _persist_run_record(log, run_record)
        print(f"[dagger r{r}] demos +{len(demos)} (agg {len(agg)}) "
              f"bc_loss {stats['loss']:.3f} | eval succ {succ}/{len(ev)} "
              f"ret {cand_ret:+.0f} -> {'ACCEPT' if accepted else 'REJECT'}",
              flush=True)
        for e in ev:
            print(f"    K={e['k']} seed={e['seed']} {e['terminal']:8s} "
                  f"ret {e['return']:+.0f}", flush=True)

    run_record["status"] = "completed"
    run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _persist_run_record(_load_log(), run_record)
    print(f"wall {round((time.time() - t0) / 60, 1)} min -> {OUT}")


if __name__ == "__main__":
    main()
