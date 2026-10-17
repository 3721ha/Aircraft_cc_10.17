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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from aircraft_sim.detect_baselines import HeurPolicy              # noqa: E402
from aircraft_sim.detect_env import DetectEnv                     # noqa: E402
from aircraft_sim.detect_mappo import DetectMAPPOPolicy           # noqa: E402
from aircraft_sim.detect_shield import DetectQPShield             # noqa: E402

OUT = Path(os.environ.get(
    "MUMT_PILOT_OUT",
    str(ROOT / "Aircraft_mumt" / "runs" / "detect_pilot")))
OUT.mkdir(parents=True, exist_ok=True)

# CLI: _p3_pilot.py [phase] ; phase = bc | pilot <iter0> <n_iters> | eval
#      | curr <k> <iter0> <n_iters>  (curriculum: train at K=k, eval each 10)
PHASE = sys.argv[1] if len(sys.argv) > 1 else "all"
ITERS0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
CURR_K = int(sys.argv[2]) if PHASE == "curr" and len(sys.argv) > 2 else 2
if PHASE == "curr":
    ITERS0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    N_ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 50

BC_EPISODES = 8          # K=4, seeds 100+
BC_EPOCHS = 60
BC_K = int(os.environ.get("MUMT_BC_K", "4"))
BC_SHIELD_LABELS = os.environ.get("MUMT_BC_SHIELD_LABELS", "1") != "0"
BC_SEED_START = int(os.environ.get("MUMT_BC_SEED_START", "100"))
CURRICULUM_START = int(os.environ.get("MUMT_W8_START", "0"))
CURRICULUM_SPAN = int(os.environ.get("MUMT_W8_SPAN", "100"))
# STABLE_WINDOW curriculum: start at 10 s, restore to 30 s by iter 50
SW_START = 10                   # initial STABLE_WINDOW override (training only)
SW_FINAL = 30                   # standard window (matches eval)
SW_RAMP_START = 0               # iter at which ramp begins
SW_RAMP_SPAN = int(os.environ.get("MUMT_SW_SPAN", "50"))
EVAL_SEEDS = [200, 201, 202, 203, 204]
EVAL_K = int(os.environ.get("MUMT_EVAL_K", "4"))
PPO_K = int(os.environ.get("MUMT_PPO_K", "4"))
EPISODES_PER_ITER = int(os.environ.get("MUMT_EPISODES_PER_ITER", "4"))


def evaluate(pol, shield, seeds, k=EVAL_K):
    rows = []
    for s in seeds:
        env = DetectEnv()
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


def _load_log():
    f = OUT / "pilot_log.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"iters": []}


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
    print(f"[PPO] lr={pol.config.learning_rate:g} epochs={pol.config.epochs} "
          f"shield_aware={pol.config.shield_aware_continuous} "
          f"cf_request_credit={pol.config.cf_request_credit:g}", flush=True)


def main():
    t_start = time.time()
    env = DetectEnv()
    shield = DetectQPShield(env.cfg)
    pol = DetectMAPPOPolicy(env, shield, seed=0)
    _configure_policy(pol)

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
        bc_stats = pol.behavior_clone(demos, epochs=BC_EPOCHS)
        pol.save(str(OUT / "pilot_bc.pt"))
        bc_eval = evaluate(pol, shield, EVAL_SEEDS)
        print(f"[BC] {bc_stats} | eval: "
              f"{sum(r['terminal'] == 'success' for r in bc_eval)}/{len(bc_eval)} success, "
              f"mean return {sum(r['return'] for r in bc_eval) / len(bc_eval):.0f}",
              flush=True)
        log = _load_log()
        log["bc"] = {"stats": bc_stats, "eval": bc_eval}
        (OUT / "pilot_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
        if PHASE == "bc":
            return

    # ---------------- Phase 2: MAPPO pilot ----------------
    if PHASE in ("all", "pilot"):
        pol.load(str(OUT / "pilot_bc.pt"))
        env = DetectEnv()
        pol.env = env
        log = _load_log()
        for it in range(ITERS0, ITERS0 + N_ITERS):
            rollout, ep_rets, ep_term, ep_steps, ep_vio = [], [], [], [], []
            n_qp = n_fb = 0
            for ep in range(EPISODES_PER_ITER):
                data = pol.collect(k_red=PPO_K, seed=1000 + it * 10 + ep)
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
            pol.save(str(OUT / "pilot_iter.pt"))
            (OUT / "pilot_log.json").write_text(
                json.dumps(log, ensure_ascii=False, indent=1))
        (OUT / "pilot_final.pt").rename(OUT / "pilot_final_prev.pt") \
            if (OUT / "pilot_final.pt").exists() else None
        (OUT / "pilot_iter.pt").rename(OUT / "pilot_final.pt")

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
            for cand in ("pilot_bc.pt", "dagger_latest.pt", "pilot_final.pt"):
                candidate = OUT / cand
                if candidate.exists():
                    pol.load(str(candidate))
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
            pol.load(str(ckpt))
            print(f"[curr K={CURR_K}] resume from {ckpt.name}", flush=True)
        env = DetectEnv()
        pol.env = env
        log = _load_log()
        ck = f"curr_k{CURR_K}"
        log.setdefault(ck, [])
        for it in range(ITERS0, ITERS0 + N_ITERS):
            # gate-6 penalty curriculum: cheap rules early, full bite by it 130
            w8_scale = 0.2 + 0.8 * min(1.0, max(0.0,
                                                (it - CURRICULUM_START)
                                                / CURRICULUM_SPAN))
            # STABLE_WINDOW curriculum: start at SW_START, ramp to SW_FINAL
            sw_frac = min(1.0, max(0.0, (it - SW_RAMP_START) / SW_RAMP_SPAN))
            stable_window = int(round(SW_START + (SW_FINAL - SW_START) * sw_frac))
            rollout, ep_rets, ep_term, ep_steps, ep_vio = [], [], [], [], []
            n_qp = n_fb = 0
            for ep in range(EPISODES_PER_ITER):
                data = pol.collect(k_red=CURR_K, seed=2000 + it * 10 + ep,
                                   w8_scale=w8_scale,
                                   stable_window=stable_window)
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
            n_interventions = sum(
                int(item.get("n_interventions", 0)) for item in rollout)
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
                "mean_return": round(sum(ep_rets) / len(ep_rets), 1),
                "success": sum(t == "success" for t in ep_term) / len(ep_term),
                "vio_per_step": round(sum(ep_vio) / max(1, sum(ep_steps)), 3),
                "qp_share": round(n_qp / max(1, n_qp + n_fb), 3),
                "approx_kl": round(stats["approx_kl"], 4),
                "w8_scale": round(w8_scale, 2),
                "stable_window": stable_window,
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
            }
            log[ck].append(entry)
            print(f"[{ck} {it:02d}] ret={entry['mean_return']:>8} succ={entry['success']:.0%} "
                  f"vio/step={entry['vio_per_step']:.2f} qp={entry['qp_share']:.2f} "
                  f"kl={entry['approx_kl']:.3f}", flush=True)
            pol.save(str(OUT / f"curr_k{CURR_K}.pt"))
            if (it + 1) % 10 == 0:
                ev = evaluate(pol, shield, [300, 301, 302], k=CURR_K)
                esucc = sum(r["terminal"] == "success" for r in ev) / len(ev)
                eret = sum(r["return"] for r in ev) / len(ev)
                log[ck][-1]["eval"] = {"success": esucc, "mean_return": round(eret, 1)}
                print(f"  [eval K={CURR_K}] {esucc:.0%} success, ret {eret:.0f}", flush=True)
            (OUT / "pilot_log.json").write_text(
                json.dumps(log, ensure_ascii=False, indent=1))
        return

    # ---------------- Phase 3: final eval ----------------
    if PHASE in ("all", "eval"):
        pol.load(str(OUT / "pilot_final.pt"))
        env = DetectEnv()
        pol.env = env
        final_eval = evaluate(pol, shield, EVAL_SEEDS)
        log = _load_log()
        log["final_eval"] = final_eval
        succ = sum(r["terminal"] == "success" for r in final_eval) / len(final_eval)
        mean_ret = sum(r["return"] for r in final_eval) / len(final_eval)
        print(f"[FINAL eval] {succ:.0%} success, mean return {mean_ret:.0f}", flush=True)
        (OUT / "pilot_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))

    print(f"wall {round((time.time() - t_start) / 60, 1)} min -> {OUT}")


if __name__ == "__main__":
    main()
