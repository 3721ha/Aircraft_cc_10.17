"""Recompute this fixed follow-up audit; writes only integrity_audit.json."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import _evaluate_sensor_adaptation as runner


def shaping_witnesses(source_log):
    """Isolate actual potential methods; these are synthetic states, not episodes."""
    from aircraft_sim.detect_env import DetectEnv
    from aircraft_sim.detect_config import ZONE_TRACKING
    from aircraft_sim.detect_mappo import DetectMAPPOPolicy

    cfg = runner.build_configs(4, 8, 10.0)["deterministic_beam"]
    saved = source_log["policy_config"]
    config = SimpleNamespace(shaping_kappa=0.0, geo_shaping_kappa=saved["geo_shaping_kappa"],
                             stable_shaping_kappa=saved["stable_shaping_kappa"])
    track = SimpleNamespace(target_id=0)
    wing = SimpleNamespace(x=0.0, y=60.0, psi=0.0, tracks=SimpleNamespace(slots=[track]),
                           radar=SimpleNamespace(axis_deg=30.0, gear=45.0))
    env = SimpleNamespace(cfg=cfg, t=0.0, wingmen=[wing], responsibility={0: 0}, stable_progress={})
    policy = SimpleNamespace(config=config, env=env)
    cases = []
    for distance in (35.0, 45.0, 55.0, 60.0, 79.0):
        track.extrapolate = lambda _t, distance=distance: (distance, 60.0)
        value = DetectMAPPOPolicy._track_potential(policy)
        effective = distance - cfg.K_SIGMA * cfg.SIGMA_REF
        assert value == max(0.0, 1 - distance / (cfg.C04_RANGE * .5)) * saved["geo_shaping_kappa"]
        cases.append({"distance_km": distance, "sigma_km": cfg.SIGMA_REF,
                      "d_eff_km": effective, "mar_estimate_feasible": effective >= cfg.MAR,
                      "isolated_phi_geo": value})
    wing.x = 50.0
    angle = math.radians(50.0)
    tx, ty = wing.x + 30 * math.cos(angle), 60 + 30 * math.sin(angle)
    track.extrapolate = lambda _t: (tx, ty)
    gate = DetectEnv._gate_diagnostics(env, 0, tx, ty)
    alignment = DetectMAPPOPolicy._stable_potential(policy)
    assert cfg.zone_of(tx) == ZONE_TRACKING and gate["failure_mode"] == "pass" and alignment == 0
    reference_angle = math.radians(40.0)
    track.extrapolate = lambda _t: (wing.x + 30 * math.cos(reference_angle),
                                   60 + 30 * math.sin(reference_angle))
    reference_alignment = DetectMAPPOPolicy._stable_potential(policy)
    assert math.isclose(reference_alignment, saved["stable_shaping_kappa"] / 3, abs_tol=1e-12)
    return {
        "kind": "synthetic method-level witnesses; no task episodes or learning performed",
        "saved_training_values": {key: saved[key] for key in (
            "geo_shaping_kappa", "shaping_kappa", "stable_shaping_kappa", "reward_scale")},
        "distance_cases": cases,
        "alignment_case": {"gear_deg": 45.0, "axis_deg": 30.0, "bearing_deg": 50.0,
                           "gate": gate, "isolated_phi_alignment": alignment,
                           "shaping_half_width_deg": 15.0,
                           "target_zone": cfg.zone_of(tx),
                           "positive_control_off_axis_deg": 10.0,
                           "positive_control_phi_alignment": reference_alignment},
        "interpretation": "The geometric potential is zero in the nominal safe sensing band; alignment shaping and C04 use different sector widths. These are training-objective mismatches, not evidence that changing them will solve the task.",
    }


def main():
    import torch

    directory = Path(__file__).resolve().parent
    result_path = directory / "followup_result.json"
    sidecar_path = directory / "followup.provenance.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    check("completed_lifecycle", report["status"] == "completed_independent_development_followup"
          and sidecar["status"] == "completed")
    check("source_unchanged", sidecar["source_integrity"]["unchanged"]
          and not runner.changed_source_files(sidecar["effective_source"]))
    check("all_twelve_inputs_intact", len(sidecar["inputs"]) == 12 and all(
        runner.artifact_record(item["path"], item["role"])["sha256"] == item["sha256"]
        and Path(item["path"]).stat().st_size == item["size_bytes"] for item in sidecar["inputs"]))
    check("output_hashes_intact", len(sidecar["outputs"]) == 1 and all(
        runner.artifact_record(item["path"], item["role"])["sha256"] == item["sha256"]
        for item in sidecar["outputs"]))
    source = sidecar["effective_source"]
    check("source_zip_hash", runner.artifact_record(ROOT / source["snapshot_path"], "zip")["sha256"]
          == source["snapshot_sha256"])
    check("source_manifest_hash", runner.canonical_sha256(source["files"]) == source["manifest_sha256"])
    with zipfile.ZipFile(ROOT / source["snapshot_path"]) as archive:
        check("source_members_intact", sorted(archive.namelist()) == sorted(
            item["path"] for item in source["files"]) and all(
            hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
            and len(archive.read(item["path"])) == item["size_bytes"] for item in source["files"]))
    inputs = {item["role"]: item for item in sidecar["inputs"]}
    original = json.loads(Path(inputs["archived_source_provenance"]["path"]).read_text(encoding="utf-8"))
    runner.verify_source_snapshot(original)
    check("original_training_source_intact", True)
    for role, expected in (("source_log", runner.EXPECTED_LOG),
                           ("source_provenance", runner.EXPECTED_PROVENANCE),
                           ("initial_rl", runner.EXPECTED_INITIAL),
                           ("best_rl", runner.EXPECTED_BEST),
                           ("selected_update8", runner.EXPECTED_BEST)):
        check(role + "_fixed_hash", inputs[role]["sha256"] == expected)
    source_log = json.loads(Path(inputs["archived_source_log"]["path"]).read_text(encoding="utf-8"))
    check("update8_fixed_before_followup", report["protocol"]["best_update_fixed"] == 8
          and source_log["best"]["update"] == 8 and max(source_log["validations"],
          key=lambda row: runner.rank(row["summary"]))["update"] == 8)
    seeds = list(range(210000, 210030))
    check("thirty_declared_new_seeds", report["protocol"]["seeds"] == seeds)
    check("ninety_unique_complete_episodes", len(report["episodes"]) == 90 and {
        (row["model"], row["seed"]) for row in report["episodes"]} == {
        (label, seed) for label in runner.LABELS for seed in seeds})
    consumed = set(source_log["protocol"]["training_seeds"]) | set(source_log["protocol"]["validation_seeds"])
    check("no_training_selection_test_seed_overlap", not set(seeds) & (
        consumed | runner.RESERVED_TEST | runner.PREVIOUSLY_USED))
    check("no_training_selection_or_final_test", not any(sidecar[key] for key in (
        "training_run", "checkpoint_selection_run", "reserved_test_run")) and not any(
        report["protocol"][key] for key in ("training", "checkpoint_selection", "reserved_test_run", "conflict_injection")))
    check("reserved_test_still_excluded", report["protocol"]["reserved_test_seeds_not_run"] == sorted(runner.RESERVED_TEST))
    check("full_task_config", runner.canonical_sha256(report["task_config"]) ==
          report["task_config_sha256"] == source_log["task_config_sha256"] ==
          runner.canonical_sha256(asdict(runner.build_configs(4, 8, 10.0)["deterministic_beam"])))
    for label in runner.LABELS[:2]:
        data = torch.load(inputs["archived_" + label]["path"], map_location="cpu", weights_only=False)
        fingerprint = runner.checkpoint_fingerprint(data)
        check(label + "_saved_config", data["config"] == source_log["policy_config"])
        check(label + "_runtime_tensors", all(
            row["loaded_tensor_sha256"] == fingerprint == sidecar["expected_loaded_tensor_sha256"][label]
            and row["weights_unchanged"] for row in report["episodes"] if row["model"] == label))
        check(label + "_runtime_config", all(
            row["effective_policy_config"] == asdict(runner.evaluation_policy_config(data["config"]))
            for row in report["episodes"] if row["model"] == label))
    check("episode_tasks_match", all(row["effective_task_config_sha256"] == report["task_config_sha256"]
          and row["sensor_profile"] == "deterministic_beam" and row["k"] == 4 for row in report["episodes"]))
    check("deterministic_beam_hits", all(all(
        row["sensor_diagnostics"]["radar"][scope]["misses"] == 0 and
        row["sensor_diagnostics"]["radar"][scope]["hits"] == row["sensor_diagnostics"]["radar"][scope]["opportunities"]
        for scope in ("all_targets", "unfinished_targets")) for row in report["episodes"]))
    check("all_rl_conflicts_direct", all(not set(row["arbitration_modes"]) - {"rl_direct"}
          and sum(row["arbitration_modes"].values()) == row["multi_candidate_events"]
          for row in report["episodes"] if row["model"] != "heur"))
    check("all_episode_count_rates_consistent", all(
        row["steps"] > 0 and sum(row["violations_by_family"].values()) == row["violations"]
        and row["violations_per_step"] == row["violations"] / row["steps"] for row in report["episodes"]))
    check("all_summaries_and_paired_statistics_recomputed", all(report[key] == value
          for key, value in runner.make_report(report["episodes"], seeds, 10000).items()))
    check("incomplete_progress_absent", not (directory / "episode_progress.json").exists())
    diagnostics = {}
    for label, summary in report["summaries"].items():
        rows = [row for row in report["episodes"] if row["model"] == label]
        diagnostics[label] = {
            "all_target_hits_per_step": summary["radar_target_dwell_counts"]["all_targets"]["hits"] / summary["total_steps"],
            "unfinished_target_hits_per_step": summary["radar_target_dwell_counts"]["unfinished_targets"]["hits"] / summary["total_steps"],
            "active_dwell_fraction": sum(row["sensor_diagnostics"]["radar"]["active_dwell_calls"] for row in rows) /
                                     sum(row["sensor_diagnostics"]["radar"]["dwell_calls"] for row in rows),
            "violations_by_family_per_step": {key: value / summary["total_steps"]
                                              for key, value in summary["violations_by_family"].items()},
        }
    audit = {
        "status": "passed", "checks_passed": len(checks), "checks": checks,
        "audit_script": runner.artifact_record(__file__, "audit_script"),
        "result": runner.artifact_record(result_path, "audited_result"),
        "provenance": runner.artifact_record(sidecar_path, "audited_provenance"),
        "diagnostic_rates": diagnostics,
        "training_objective_witnesses": shaping_witnesses(source_log),
        "decision": "Both predeclared joint descriptive screens failed. Do not promote update8 or extend this joint adaptation on these results.",
        "limits": ["This follow-up is development evidence, not the reserved final test.",
                   "Success exact paired p=0.06543 versus initial is not significant at 5%; bootstrap and exact intervals must not be selectively reported.",
                   "Coverage and ownership diagnostics are associations; they do not establish a causal failure mechanism or the impossibility of RL."]}
    runner.write_json_atomic(directory / "integrity_audit.json", audit)
    print(json.dumps({"status": audit["status"], "checks_passed": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
