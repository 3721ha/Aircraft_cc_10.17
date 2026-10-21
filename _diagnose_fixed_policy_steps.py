"""Step-level, read-only diagnosis for the frozen K4/W8/A10 RL policies.

The script deliberately runs only the saved initial and update-8 policies.  It
does not train, select a checkpoint, inject requests, or call HEUR.  Diagnostic
fields are collected after policy actions are chosen and therefore never enter
the observation or action path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _experiment_provenance import (
    add_output, artifact_record, canonical_sha256, capture_run_provenance,
    capture_source_snapshot, changed_source_files, register_active_run,
    track_run_lifecycle, write_json_atomic,
)
from _diagnose_sensor_profile import (
    build_configs, evaluation_policy_config, model_fingerprint,
)
from _evaluate_sensor_adaptation import verify_source_snapshot

DEFAULT_SOURCE = ROOT / "runs/model_optimization_v1/sensor_joint_adapt_k4_w8a10_v1"
EXPECTED_LOG = "9018f87365c90d9ee7cdc627d83480a28b6c069cad7ae29c8222bbac543c7475"
EXPECTED_PROVENANCE = "e7f4ced938b4fcaeeb6f998cf1d93b4dbb62d42a1462f516c0b58ba57a7989c5"
EXPECTED_INITIAL = "ef86e5734d2a6466694371a0931386d7f1937355f279eed28f1560dcae27499b"
EXPECTED_UPDATE8 = "b30c53060f0374b204ca95fd6797bbf9a7ec5b1fb00d6d628e2afe5998f0e801"
RESERVED_TEST = set(range(208000, 208030))
PREVIOUSLY_USED = {204900, *range(205000, 205030), *range(206000, 206096),
                   *range(207000, 207012), *range(209900, 209904), 209990, 209991,
                   *range(210000, 210030), *range(219900, 219911),
                   *range(250000, 250006)}
ENTRY_SOURCES = [Path(__file__), ROOT / "_diagnose_sensor_profile.py",
                 ROOT / "_evaluate_sensor_adaptation.py"]


def checkpoint_fingerprint(data: dict) -> str:
    digest = hashlib.sha256()
    for label in ("model", "xi_anchor_model", "policy_anchor_model"):
        state = data.get(label)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"checkpoint lacks complete {label} state")
        for name, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{label}/{name}/{tensor.dtype}/{tuple(tensor.shape)}".encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _track_record(env, i, track):
    """Record local track geometry without changing the environment."""
    ex, ey = track.extrapolate(env.t)
    w = env.wingmen[i]
    gate = env._gate_diagnostics(i, ex, ey)
    distance = math.hypot(ex - w.x, ey - w.y)
    sigma = _finite(track.sigma)
    d_eff = None if sigma is None else distance - env.cfg.K_SIGMA * sigma
    return {
        "target_id": int(track.target_id), "state": str(track.state),
        "age_s": _finite(track.age(env.t)), "sigma_km": sigma,
        "estimate_xy_km": [_finite(ex), _finite(ey)],
        "owned_by": env.responsibility.get(track.target_id),
        "owned_by_local_wingman": env.responsibility.get(track.target_id) == i,
        "d_eff_km": _finite(d_eff), "local_estimate_c04_geometry": gate,
    }


def _phi_components(policy):
    """Read the existing training potentials, plus independently exposed parts."""
    env, cfg = policy.env, policy.config
    sim = env.cfg
    phi_geo = phi_align = phi_fresh = phi_prog = 0.0
    if cfg.shaping_kappa > 0:
        for i, w in enumerate(env.wingmen):
            for tr in w.tracks.slots:
                if tr is None or env.responsibility.get(tr.target_id) != i:
                    continue
                phi_fresh += min(tr.age(env.t), cfg.shaping_age_cap) / cfg.shaping_age_cap
    for i, w in enumerate(env.wingmen):
        for tr in w.tracks.slots:
            if tr is None or env.responsibility.get(tr.target_id) != i:
                continue
            ex, ey = tr.extrapolate(env.t)
            d = math.hypot(ex - w.x, ey - w.y)
            if cfg.geo_shaping_kappa > 0:
                support = sim.C04_RANGE * cfg.geo_shaping_range_fraction
                phi_geo += max(0.0, 1.0 - d / support)
            if sim.zone_of(ex) == "tracking" and cfg.stable_shaping_kappa > 0:
                bearing = math.atan2(ey - w.y, ex - w.x)
                boresight = w.psi + math.radians(w.radar.axis_deg)
                off = abs(math.atan2(math.sin(bearing - boresight),
                                     math.cos(bearing - boresight)))
                if cfg.stable_alignment_geometry == "c04_usable_sector":
                    half_width_deg = max(0.0, min(
                        w.radar.gear,
                        sim.SECTOR_AXIS_MAX_DEG - abs(w.radar.axis_deg)))
                else:
                    half_width_deg = max(
                        1.0, w.radar.gear - abs(w.radar.axis_deg))
                if half_width_deg > 0.0:
                    half = math.radians(half_width_deg)
                    phi_align += max(0.0, 1.0 - off / half)
    if cfg.stable_shaping_kappa > 0:
        phi_prog = sum(float(p) / sim.STABLE_WINDOW
                       for p in env.stable_progress.values())
    track_potential = float(policy._track_potential())
    stable_potential = float(policy._stable_potential())
    expected_track_potential = (-cfg.shaping_kappa * phi_fresh
                                + cfg.geo_shaping_kappa * phi_geo)
    expected_stable_potential = cfg.stable_shaping_kappa * (phi_prog + phi_align)
    if not math.isclose(track_potential, expected_track_potential,
                        rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("independent track-potential decomposition differs from policy")
    if not math.isclose(stable_potential, expected_stable_potential,
                        rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("independent stable-potential decomposition differs from policy")
    return {
        "phi_fresh_unscaled": phi_fresh,
        "phi_geo_unscaled": phi_geo,
        "phi_align_unscaled": phi_align,
        "phi_progress_unscaled": phi_prog,
        "phi_geo": cfg.geo_shaping_kappa * phi_geo,
        "phi_align": cfg.stable_shaping_kappa * phi_align,
        "track_potential": track_potential,
        "stable_potential": stable_potential,
    }


def _radar_snapshot(env):
    return {
        str(w.idx): {
            "role": w.role.value, "role_age_s": _finite(w.role_age),
            "gear_deg": _finite(w.radar.gear), "axis_deg": _finite(w.radar.axis_deg),
            "phase": int(w.radar.phase), "frame_progress_s": _finite(w.radar.frame_progress),
            "frame_period_s": _finite(w.radar.frame_period()),
            "switch_ready": bool(w.radar.switch_ready()),
            "last_switch_step": int(w.radar.last_switch_step),
        } for w in env.wingmen
    }


def _action_snapshot(actions):
    out = {}
    for i, a in actions.items():
        out[str(i)] = {
            "u_psi": _finite(a.u_psi), "u_v": _finite(a.u_v), "u_xi": _finite(a.u_xi),
            "role": a.role.value, "gear_deg": _finite(a.gear), "pointer": int(a.pointer),
            "arbitration_bid": _finite(a.arbitration_bid),
            "rl_direct_arbitration": bool(a.rl_direct_arbitration),
        }
    return out


def run_episode(cfg, checkpoint, saved_config, seed):
    from aircraft_sim.detect_mappo import DetectMAPPOPolicy
    from aircraft_sim.detect_shield import DetectQPShield
    from aircraft_sim.detect_sensor_diagnostics import SensorDiagnosticEnv

    env = SensorDiagnosticEnv(cfg)
    shield = DetectQPShield(cfg)
    policy = DetectMAPPOPolicy(env, shield, evaluation_policy_config(saved_config), seed=0)
    policy.load(str(checkpoint))
    policy.model.eval(); policy.xi_anchor_model.eval(); policy.policy_anchor_model.eval()
    before_fp = model_fingerprint(policy)
    env.reset({"seed": seed, "k_red": cfg.K_RED})
    steps, total_return = [], 0.0
    dwell_events = []
    from aircraft_sim.radar import beam_offsets, wrap_pi
    for wingman in env.wingmen:
        radar = wingman.radar
        original_scan = radar.scan_dwell

        def diagnostic_scan(heading_rad, own_xy, targets, rng, *, _radar=radar,
                            _original=original_scan, _wingman=wingman):
            phase, gear, axis = _radar.phase, _radar.gear, _radar.axis_deg
            offsets = beam_offsets(gear, _radar.cfg.FOV_HALF_DEG)
            bore = heading_rad + math.radians(axis + offsets[phase])
            eligible = 0
            for tx, ty in targets.values():
                dx, dy = tx - own_xy[0], ty - own_xy[1]
                distance = math.hypot(dx, dy)
                bearing = math.atan2(dy, dx)
                theta = abs(wrap_pi(bearing - heading_rad))
                r_eff = _radar.cfg.R_MAX * math.sqrt(max(0.0, math.cos(theta)))
                if (distance <= r_eff
                        and abs(wrap_pi(bearing - bore)) <= math.radians(_radar.cfg.FOV_HALF_DEG)):
                    eligible += 1
            hits, used_bore = _original(heading_rad, own_xy, targets, rng)
            dwell_events.append({"wingman": _wingman.idx, "gear_deg": gear,
                                 "phase": phase, "eligible_targets": eligible,
                                 "hit_targets": len(hits), "bore_rad": _finite(used_bore)})
            return hits, used_bore

        radar.scan_dwell = diagnostic_scan
    while env.terminal is None:
        pre_radar = _radar_snapshot(env)
        pre_tracks = {str(i): [_track_record(env, i, tr) for tr in w.tracks.slots if tr is not None]
                      for i, w in enumerate(env.wingmen)}
        pre_phi = _phi_components(policy)
        observations = env.observe()
        actions, _ = policy.act_batch(observations, deterministic=True)
        if not all(a.rl_direct_arbitration for a in actions.values()):
            raise RuntimeError("RL direct arbitration flag lost")
        requested = _action_snapshot(actions)
        ready_before = {str(i): bool(w.radar.switch_ready()) for i, w in enumerate(env.wingmen)}
        safe_actions, corrections = shield.filter(env, actions)
        local_request_geometry = env._request_pre_gate_diagnostics(safe_actions)
        dwell_offset = len(dwell_events)
        result = env.step(safe_actions)
        if any(event.get("arbitration_mode") != "rl_direct"
               for event in result.info.get("arbitration", [])):
            raise RuntimeError("frozen RL evaluation used non-RL arbitration")
        total_return += float(result.reward)
        post_radar = _radar_snapshot(env)
        post_tracks = {str(i): [_track_record(env, i, tr) for tr in w.tracks.slots if tr is not None]
                       for i, w in enumerate(env.wingmen)}
        outcomes = result.info.get("request_outcomes", [])
        gate_outcomes = [{"target_id": o.get("target_id"), "requester": o.get("requester"),
                          "reason": o.get("reason"), "c04_gate_ok": o.get("c04_gate_ok"),
                          "geometry_from_referee_truth": o.get("c04_geometry")}
                         for o in outcomes]
        local_request_rows = [
            {"requester": int(i), "target_id": int(tid),
             "geometry_from_local_track_estimate": geometry}
            for (i, tid), geometry in sorted(local_request_geometry.items())]
        switch = {}
        for i, w in enumerate(env.wingmen):
            req = actions[i]
            safe_req = safe_actions[i]
            switch[str(i)] = {
                "requested_gear_deg": _finite(req.gear),
                "executed_gear_request_deg": _finite(safe_req.gear),
                "gear_before_deg": pre_radar[str(i)]["gear_deg"],
                "gear_after_deg": _finite(w.radar.gear),
                "switch_attempt": bool(safe_req.gear != pre_radar[str(i)]["gear_deg"]),
                "switch_ready_before": ready_before[str(i)],
                "switch_denied": bool(w.gear_switch_denied),
                "shield_intervened": bool(any(getattr(c, "uid", None) == i
                                               for c in corrections)),
            }
        dwell_summary = defaultdict(lambda: {"dwells": 0, "eligible_targets": 0,
                                              "hit_targets": 0})
        for event in dwell_events[dwell_offset:]:
            key = (str(event["wingman"]), str(event["gear_deg"]), str(event["phase"]))
            item = dwell_summary[key]
            item["dwells"] += 1
            item["eligible_targets"] += event["eligible_targets"]
            item["hit_targets"] += event["hit_targets"]
        steps.append({
            "step": int(env.step_count), "time_s": _finite(env.t),
            "reward": float(result.reward), "terminal": env.terminal,
            "radar_before": pre_radar, "radar_after": post_radar,
            "actions": requested, "safe_actions": _action_snapshot(safe_actions),
            "switch": switch, "tracks_before": pre_tracks, "tracks_after": post_tracks,
            "radar_dwell_by_wing_gear_phase": {
                "|".join(key): value for key, value in sorted(dwell_summary.items())},
            "request_estimates_from_local_tracks": local_request_rows,
            "referee_request_gate_outcomes": gate_outcomes,
            "referee_c04_breach_pairs": [list(x) for x in env.c04_breach],
            "local_estimate_diagnostics": {
                "tracks_before": pre_tracks, "tracks_after": post_tracks,
                "request_c04_geometry": local_request_rows,
                "potential_before": pre_phi, "potential_after": _phi_components(policy),
            },
            "referee_diagnostics": {
                "terminal": env.terminal,
                "violations": list(result.info["report"].violations),
                "request_gate_outcomes": gate_outcomes,
                "c04_breach_pairs": [list(x) for x in env.c04_breach],
            },
            "arbitration": result.info.get("arbitration", []),
            "milestone_hits": result.info.get("milestone_hits", []),
        })
    after_fp = model_fingerprint(policy)
    if before_fp != after_fp:
        raise RuntimeError("frozen evaluation mutated policy weights")
    summary = summarize_episode(steps)
    return {
        "seed": seed, "terminal": env.terminal, "return": total_return,
        "steps": len(steps), "violations": sum(
            len(s["referee_diagnostics"]["violations"]) for s in steps),
        "violations_per_step": sum(
            len(s["referee_diagnostics"]["violations"]) for s in steps) / max(1, len(steps)),
        "loaded_tensor_sha256": after_fp, "weights_unchanged": True,
        "task_config_sha256": canonical_sha256(asdict(cfg)),
        "summary": summary, "steps_detail": steps,
    }


def summarize_episode(steps):
    n = max(1, len(steps))
    by_wing = {}
    for i in range(3):
        key = str(i)
        switches = [step["switch"][key] for step in steps]
        radar = [step["radar_before"][key] for step in steps]
        abs_axis = [abs(item["axis_deg"]) for item in radar]
        by_wing[key] = {
            "switch_attempts": sum(item["switch_attempt"] for item in switches),
            "switch_denials": sum(item["switch_denied"] for item in switches),
            "switch_attempts_while_ready": sum(
                item["switch_attempt"] and item["switch_ready_before"] for item in switches),
            "switch_ready_steps": sum(item["switch_ready_before"] for item in switches),
            "abs_axis_deg_sum": sum(abs_axis),
            "axis_observations": len(abs_axis),
            "axis_at_limit_steps": sum(value >= 60.0 - 1e-6 for value in abs_axis),
            "gear_action_counts": dict(Counter(str(step["actions"][key]["gear_deg"])
                                                for step in steps)),
            "gear_state_counts": dict(Counter(str(item["gear_deg"]) for item in radar)),
            "radar_phase_state_counts": dict(Counter(str(item["phase"]) for item in radar)),
        }
    request_reasons = Counter()
    c04_failure_modes = Counter()
    c04_pass = c04_fail = 0
    d_eff_bins = Counter()
    local_gate_modes = Counter()
    dwell = defaultdict(lambda: {"dwells": 0, "eligible_targets": 0, "hit_targets": 0})
    phi_geo_active = phi_align_active = 0
    for step in steps:
        for outcome in step["referee_request_gate_outcomes"]:
            reason = outcome.get("reason") or "unknown"
            request_reasons[reason] += 1
            if outcome.get("c04_gate_ok") is True:
                c04_pass += 1
            elif outcome.get("c04_gate_ok") is False:
                c04_fail += 1
            geometry = outcome.get("geometry_from_referee_truth") or {}
            if geometry.get("failure_mode"):
                c04_failure_modes[geometry["failure_mode"]] += 1
        for local_wing, tracks in step["local_estimate_diagnostics"]["tracks_before"].items():
            for track in tracks:
                geometry = track.get("local_estimate_c04_geometry") or {}
                if geometry.get("failure_mode"):
                    local_gate_modes[geometry["failure_mode"]] += 1
                locally_owned = track.get("owned_by_local_wingman")
                if locally_owned is None:
                    locally_owned = track.get("owned_by") == int(local_wing)
                if (not locally_owned
                        or track.get("d_eff_km") is None):
                    continue
                d = track["d_eff_km"]
                d_eff_bins["below_MAR"] += int(d < 40.0)
                d_eff_bins["MAR_to_MOR"] += int(40.0 <= d < 60.0)
                d_eff_bins["at_or_above_MOR"] += int(d >= 60.0)
        before = step["local_estimate_diagnostics"]["potential_before"]
        phi_geo_active += int(abs(before["phi_geo"]) > 1e-12)
        phi_align_active += int(abs(before["phi_align"]) > 1e-12)
        for composite, counts in step["radar_dwell_by_wing_gear_phase"].items():
            for name in counts:
                dwell[composite][name] += counts[name]
        for local_request in step["request_estimates_from_local_tracks"]:
            if not local_request["geometry_from_local_track_estimate"]:
                continue
            mode = local_request["geometry_from_local_track_estimate"]["failure_mode"]
            local_gate_modes[f"request_{mode}"] += 1
    return {
        "switches_by_wingman": by_wing,
        "request_outcome_counts": dict(request_reasons),
        "c04_gate_passes": c04_pass, "c04_gate_failures": c04_fail,
        "c04_failure_modes": dict(c04_failure_modes),
        "local_track_c04_geometry_modes": dict(local_gate_modes),
        "owned_local_track_d_eff_bins_km": dict(d_eff_bins),
        "phi_geo_nonzero_steps": phi_geo_active,
        "phi_geo_nonzero_fraction": phi_geo_active / n,
        "phi_align_nonzero_steps": phi_align_active,
        "phi_align_nonzero_fraction": phi_align_active / n,
        "radar_dwell_by_wing_gear_phase": dict(dwell),
    }


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed-start", type=int, default=250006)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--plan-only", action="store_true")
    return ap


def summarize_models(rows):
    result = {}
    for label in ("initial_rl", "best_rl"):
        group = [row for row in rows if row["model"] == label]
        if not group:
            continue
        total_steps = sum(row["steps"] for row in group)
        total_violations = sum(row["violations"] for row in group)
        result[label] = {
            "episodes": len(group),
            "successes": sum(row["terminal"] == "success" for row in group),
            "success_rate": sum(row["terminal"] == "success" for row in group) / len(group),
            "mean_return": sum(row["return"] for row in group) / len(group),
            "total_steps": total_steps, "total_violations": total_violations,
            "pooled_violations_per_step": total_violations / max(1, total_steps),
            "episodes_by_terminal": dict(Counter(row["terminal"] for row in group)),
            "mean_phi_geo_nonzero_fraction": sum(
                row["summary"]["phi_geo_nonzero_fraction"] for row in group) / len(group),
            "mean_phi_align_nonzero_fraction": sum(
                row["summary"]["phi_align_nonzero_fraction"] for row in group) / len(group),
            "switch_attempts": sum(
                wing["switch_attempts"] for row in group
                for wing in row["summary"]["switches_by_wingman"].values()),
            "switch_denials": sum(
                wing["switch_denials"] for row in group
                for wing in row["summary"]["switches_by_wingman"].values()),
            "switches_by_wingman": {
                str(i): {
                    "attempts": sum(row["summary"]["switches_by_wingman"][str(i)]["switch_attempts"]
                                     for row in group),
                    "denials": sum(row["summary"]["switches_by_wingman"][str(i)]["switch_denials"]
                                   for row in group),
                    "mean_abs_axis_deg": sum(
                        row["summary"]["switches_by_wingman"][str(i)]["abs_axis_deg_sum"]
                        for row in group) / max(1, sum(
                            row["summary"]["switches_by_wingman"][str(i)]["axis_observations"]
                            for row in group)),
                    "axis_at_limit_steps": sum(
                        row["summary"]["switches_by_wingman"][str(i)]["axis_at_limit_steps"]
                        for row in group),
                } for i in range(3)},
            "gear_state_counts": _merge_wing_counts(group, "gear_state_counts"),
            "gear_action_counts": _merge_wing_counts(group, "gear_action_counts"),
            "radar_phase_state_counts": _merge_wing_counts(
                group, "radar_phase_state_counts"),
            "owned_local_track_d_eff_bins_km": _merge_summary_counts(
                group, "owned_local_track_d_eff_bins_km"),
            "local_track_c04_geometry_modes": _merge_summary_counts(
                group, "local_track_c04_geometry_modes"),
            "referee_request_c04_gate": {
                "passes": sum(row["summary"]["c04_gate_passes"] for row in group),
                "failures": sum(row["summary"]["c04_gate_failures"] for row in group),
                "failure_modes": _merge_summary_counts(group, "c04_failure_modes"),
            },
            "radar_dwell_by_wing_gear_phase": _merge_dwell(group),
        }
    return result


def _merge_summary_counts(group, key):
    merged = Counter()
    for row in group:
        merged.update(row["summary"].get(key, {}))
    return dict(merged)


def _merge_wing_counts(group, key):
    merged = defaultdict(Counter)
    for row in group:
        for wing, values in row["summary"]["switches_by_wingman"].items():
            merged[wing].update(values.get(key, {}))
    return {wing: dict(counts) for wing, counts in merged.items()}


def _merge_dwell(group):
    merged = defaultdict(Counter)
    for row in group:
        for key, counts in row["summary"]["radar_dwell_by_wing_gear_phase"].items():
            merged[key].update(counts)
    return {key: dict(counts) for key, counts in merged.items()}


def _verify(path, sha, role):
    rec = artifact_record(path, role)
    if not rec["exists"] or rec["sha256"] != sha:
        raise ValueError(f"{role} missing or hash mismatch: {path}")
    return rec


def preflight(args):
    source = args.source_run.resolve()
    if args.output.resolve().exists():
        raise ValueError("output already exists")
    if args.seed_start < 0 or args.episodes < 1 or args.torch_threads < 1:
        raise ValueError("seed/episodes/threads invalid")
    log_rec = _verify(source / "adaptation_log.json", EXPECTED_LOG, "source_log")
    prov_rec = _verify(source / "adaptation.provenance.json", EXPECTED_PROVENANCE, "source_provenance")
    initial = _verify(source / "initial.pt", EXPECTED_INITIAL, "initial_checkpoint")
    update8 = _verify(source / "checkpoints/update_0008.pt", EXPECTED_UPDATE8, "update8_checkpoint")
    log = json.loads((source / "adaptation_log.json").read_text(encoding="utf-8"))
    prov = json.loads((source / "adaptation.provenance.json").read_text(encoding="utf-8"))
    if (log.get("status") != "completed_short_adaptation"
            or prov.get("status") != "completed"
            or not prov.get("source_integrity", {}).get("unchanged")
            or prov.get("reserved_test_run")
            or log.get("best", {}).get("update") != 8
            or prov.get("selected_best_update") != 8):
        raise ValueError("source does not pin update 8")
    source_archive = verify_source_snapshot(prov)
    declared_outputs = {item["role"]: item["sha256"] for item in prov.get("outputs", [])}
    if (declared_outputs.get("adaptation_log") != EXPECTED_LOG
            or declared_outputs.get("effective_initial_checkpoint") != EXPECTED_INITIAL
            or declared_outputs.get("selected_best_checkpoint") != EXPECTED_UPDATE8):
        raise ValueError("source provenance outputs do not pin audited artifacts")
    if log["best"]["selected_checkpoint"]["sha256"] != EXPECTED_UPDATE8:
        raise ValueError("source selected checkpoint is not fixed update 8")
    cfg = build_configs(4, 8, 10.0)["deterministic_beam"]
    if canonical_sha256(asdict(cfg)) != log["task_config_sha256"]:
        raise ValueError("task config differs from audited run")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate diagnostic seeds")
    if set(seeds) & (RESERVED_TEST | PREVIOUSLY_USED):
        raise ValueError("diagnostic seed overlaps consumed or reserved seed pool")
    return {
        "source_run": str(source),
        "inputs": [log_rec, prov_rec, initial, update8, source_archive],
        "checkpoints": {"initial_rl": initial, "best_rl": update8},
        "effective_policy_config": log["policy_config"],
        "task_config": asdict(cfg), "task_config_sha256": log["task_config_sha256"],
        "seeds": seeds,
        "models": ["initial_rl", "best_rl"], "training": False,
        "checkpoint_selection": False, "final_test": False,
        "conflict_injection": False, "heuristic": False,
    }


@track_run_lifecycle
def main():
    args = parser().parse_args()
    try:
        plan = preflight(args)
    except (ValueError, OSError, KeyError) as exc:
        parser().error(str(exc))
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2, default=str)); return
    output = args.output.resolve(); output.mkdir(parents=True)
    source_snapshot = capture_source_snapshot(ROOT, output, ENTRY_SOURCES)
    record = capture_run_provenance(
        ROOT, output, ENTRY_SOURCES, "fixed_policy_step_diagnostic",
        plan["task_config"], algorithm_config={"models": plan["models"]},
        runtime_parameters={"arguments": vars(args), "protocol": plan},
        inputs=plan["inputs"], source_snapshot=source_snapshot)
    prov_path = output / "diagnostic.provenance.json"
    result_path = output / "step_diagnostic.json"
    progress_path = output / "episode_progress.json"
    register_active_run(record, lambda value: write_json_atomic(prov_path, value))
    write_json_atomic(prov_path, record)
    # Archive the exact source records and checkpoints. Episodes run only from
    # immutable checkpoint copies; the originals are rehashed between episodes.
    archived_inputs = {}
    for frozen in plan["inputs"]:
        original = Path(frozen["path"])
        target = output / "provenance" / f"{frozen['role']}_{original.name}"
        shutil.copy2(original, target)
        archived_item = artifact_record(target, f"archived_{frozen['role']}")
        if archived_item["sha256"] != frozen["sha256"]:
            raise RuntimeError(f"input changed while archiving: {original}")
        record.setdefault("inputs", []).append(archived_item)
        archived_inputs[frozen["role"]] = target
    archived = {
        "initial_rl": archived_inputs["initial_checkpoint"],
        "best_rl": archived_inputs["update8_checkpoint"],
    }
    write_json_atomic(prov_path, record)
    import torch
    torch.set_num_threads(args.torch_threads)
    cfg = build_configs(4, 8, 10.0)["deterministic_beam"]
    rows = []
    for label, item in plan["checkpoints"].items():
        checkpoint = archived[label]
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved = data["config"]
        if saved != plan["effective_policy_config"]:
            raise RuntimeError(f"{label} policy configuration differs from audited run")
        evaluation_policy_config(saved)
        expected_fingerprint = checkpoint_fingerprint(data)
        record.setdefault("effective_evaluation_policy_configs", {})[label] = saved
        record.setdefault("expected_loaded_tensor_sha256", {})[label] = expected_fingerprint
        write_json_atomic(prov_path, record)
        for seed in plan["seeds"]:
            if changed_source_files(source_snapshot):
                raise RuntimeError("source changed during diagnosis")
            for frozen in plan["inputs"]:
                current = artifact_record(frozen["path"], frozen["role"])
                if current["sha256"] != frozen["sha256"]:
                    raise RuntimeError(f"diagnostic input changed: {frozen['path']}")
            archived_current = artifact_record(checkpoint, f"archived_{label}")
            if archived_current["sha256"] != plan["checkpoints"][label]["sha256"]:
                raise RuntimeError(f"archived checkpoint changed: {checkpoint}")
            row = run_episode(cfg, checkpoint, saved, seed)
            if row["loaded_tensor_sha256"] != expected_fingerprint:
                raise RuntimeError(f"{label} loaded tensors differ from the fixed checkpoint")
            if not row["weights_unchanged"]:
                raise RuntimeError(f"{label} weights changed during diagnosis")
            row["model"] = label
            rows.append(row)
            write_json_atomic(progress_path, {"status": "incomplete", "episodes": rows})
            print(f"{label} seed={seed}: {row['terminal']} return={row['return']:.1f}", flush=True)
    serial_plan = json.loads(json.dumps(plan, default=str))
    result = {"status": "completed_step_diagnostic", "protocol": serial_plan,
              "model_summaries": summarize_models(rows),
              "episodes": rows, "provenance_sidecar": str(prov_path),
              "limits": ["read-only frozen policies", "no HEUR labels or injected requests"],
              "training_run": False, "checkpoint_selection_run": False,
              "final_test_run": False, "conflict_injection": False,
              "effective_policy_config": plan["effective_policy_config"]}
    if changed_source_files(source_snapshot):
        raise RuntimeError("source changed before diagnostic completion")
    for frozen in plan["inputs"]:
        current = artifact_record(frozen["path"], frozen["role"])
        if current["sha256"] != frozen["sha256"]:
            raise RuntimeError(f"diagnostic input changed: {frozen['path']}")
    write_json_atomic(result_path, result); progress_path.unlink()
    add_output(record, result_path, "step_diagnostic")
    record["inputs_unchanged"] = True; record["training_run"] = False
    print(f"Completed: {result_path}", flush=True)


if __name__ == "__main__":
    main()
