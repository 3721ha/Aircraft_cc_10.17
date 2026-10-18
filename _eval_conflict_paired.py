"""Paired HEUR/RL evaluation for learned conflict arbitration.

The deployed RL chain is actor -> continuous QP shield -> environment.  HEUR
is instantiated only as a reference controller for the paired comparison;
neither controller calls a fixed responsibility score.  In the environment,
feasible competing requests are resolved by the actor's learned bid.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))
from _experiment_provenance import (add_output, artifact_record,
                                    capture_run_provenance,
                                    capture_source_snapshot,
                                    register_active_run,
                                    track_run_lifecycle,
                                    write_json_atomic)

_startup_parser = argparse.ArgumentParser(add_help=False)
_startup_parser.add_argument("--output", type=Path)
_startup_args, _ = _startup_parser.parse_known_args()
_startup_output_dir = (_startup_args.output.resolve().parent
                       if _startup_args.output else
                       ROOT / "runs" / "standalone_evaluations")
_startup_output_conflict = False
if _startup_args.output:
    _startup_result = _startup_args.output.resolve()
    _startup_sidecar = _startup_result.with_suffix(
        _startup_result.suffix + ".provenance.json")
    _startup_output_conflict = (
        _startup_result.exists() or _startup_sidecar.exists())
SOURCE_SNAPSHOT = (
    None if ("--help" in sys.argv or "-h" in sys.argv
            or _startup_output_conflict) else
    capture_source_snapshot(
        ROOT, _startup_output_dir,
        [ROOT / "_eval_conflict_paired.py"]))

from aircraft_sim.detect_baselines import HeurPolicy
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_encoder import DetectObservationEncoder
from aircraft_sim.detect_mappo import (DetectMAPPOConfig, DetectMAPPOPolicy,
                                       N_CONFLICT_FEATURES)
from aircraft_sim.detect_shield import DetectQPShield
from aircraft_sim.detect_config import DetectSimConfig


def _winner(scores: dict, owner: int | None, hysteresis: float) -> int:
    best = max(scores, key=lambda i: (scores[i], i == owner, -i))
    if (owner in scores and best != owner
            and scores[best] <= scores[owner] + hysteresis):
        return owner
    return best


def _snapshot_checkpoint(checkpoint_path: Path, provenance_dir: Path,
                         run_id: str) -> tuple[Path, dict]:
    """Copy the exact evaluated weights into the evaluation's run archive."""
    before = artifact_record(checkpoint_path, "policy_checkpoint")
    if not before["exists"]:
        raise FileNotFoundError(checkpoint_path)
    snapshot_path = (provenance_dir / "provenance" /
                     f"{run_id}_input_policy_checkpoint_{checkpoint_path.name}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        raise FileExistsError(snapshot_path)
    shutil.copy2(checkpoint_path, snapshot_path)
    snapshot = artifact_record(snapshot_path, "policy_checkpoint_snapshot")
    after = artifact_record(checkpoint_path, "policy_checkpoint_after_snapshot")
    if before["sha256"] != snapshot["sha256"] or before["sha256"] != after["sha256"]:
        raise RuntimeError(
            f"checkpoint changed while archiving: {checkpoint_path}")
    snapshot["source_path"] = str(checkpoint_path)
    before["archived_snapshot"] = snapshot
    return snapshot_path.resolve(), before


def _evaluation_output_paths(output_path: Path | None,
                             provenance_dir: Path,
                             run_id: str,
                             mode: str) -> tuple[Path, Path]:
    if output_path is None:
        output_path = provenance_dir / f"{run_id}_{mode}.json"
    provenance_path = output_path.with_suffix(
        output_path.suffix + ".provenance.json")
    existing = [path for path in (output_path, provenance_path)
                if path.exists()]
    if existing:
        raise FileExistsError(
            "Evaluation output already exists: "
            + ", ".join(str(path) for path in existing))
    return output_path, provenance_path


def _preflight_output_path(output_path: Path | None) -> None:
    """Reject an occupied explicit result path before archiving run inputs."""
    if output_path is None:
        return
    provenance_path = output_path.with_suffix(
        output_path.suffix + ".provenance.json")
    existing = [path for path in (output_path, provenance_path)
                if path.exists()]
    if existing:
        raise FileExistsError(
            "Evaluation output already exists: "
            + ", ".join(str(path) for path in existing))


def run(controller, checkpoint: str | None, k_red: int, seed: int,
        cfg: DetectSimConfig, conflict_probe: bool = False,
        direct_rl_arbitration: bool = False) -> dict:
    env = DetectEnv(cfg)
    shield = DetectQPShield(env.cfg)
    policy = None
    if checkpoint:
        policy = DetectMAPPOPolicy(env, shield, seed=0)
        policy.config.direct_rl_arbitration = direct_rl_arbitration
        policy.load(checkpoint)
        policy._conflict_rng = random.Random(int(seed) + 0xC0F11C7)
        policy._conflict_probe_rng = random.Random(int(seed) ^ 0x5A17)
    scenario = {"seed": seed, "k_red": k_red}
    env.reset(scenario)
    total = 0.0
    steps = 0
    violations = 0
    conflicts = 0
    bid_changed_winner = 0
    probe_exposure_steps = 0
    probe_injected_pairs = 0
    probe_exposed_agents = 0
    probe_conflicts = 0
    probe_bid_changed_winner = 0
    probe_event_details = []
    bid_abs_sum = 0.0
    bid_candidate_count = 0
    request_outcomes: Counter[str] = Counter()
    incumbent_overrides = 0
    intervention_count = 0
    correction_total = 0.0
    family_counts: dict[str, int] = {}
    pair_names = [
        "S04-M02", "S05-M02", "C01-M02", "C03-M02", "C02-M01",
        "M01-M02", "S02-M02", "S03-M02", "I02-M02", "I03-M02",
        "M04-S04", "C01-C03", "M05-E03",
    ]
    pair_active_steps = [0] * len(pair_names)
    pair_preference_sum = [0.0] * len(pair_names)
    pair_preference_count = [0] * len(pair_names)
    encoder = policy.encoder if policy is not None else DetectObservationEncoder(env.cfg)
    while env.terminal is None:
        observations = env.observe()
        encoded = {i: encoder.encode(observations[i]) for i in observations}
        for vector in encoded.values():
            for j, pressure in enumerate(vector[-N_CONFLICT_FEATURES:]):
                if pressure > 0.1:
                    pair_active_steps[j] += 1
        if policy is None:
            nominal = controller(env)
            exposure_mask = [False] * env.cfg.N_WINGMEN
        else:
            nominal, _ = policy.act_batch(observations, deterministic=True)
            exposure_mask = [False] * env.cfg.N_WINGMEN
            if conflict_probe:
                policy.config.conflict_exposure_prob = 1.0
                nominal, exposure_mask, injected = \
                    policy._expose_shared_target_conflict(
                        nominal, deterministic=False,
                        probe_rng=policy._conflict_probe_rng)
                probe_injected_pairs += injected
                if injected:
                    probe_exposure_steps += 1
                    probe_exposed_agents += sum(exposure_mask)
                    ids = [i for i, exposed in enumerate(exposure_mask)
                           if exposed]
                    obs_t = torch.tensor(
                        np.asarray([encoded[i] for i in ids]),
                        dtype=torch.float32)
                    with torch.no_grad():
                        # Arbitration consumes the slot-specific bid vector.
                        # Updating only ``arbitration_bid`` leaves the actual
                        # env input unchanged whenever arbitration_bids is
                        # present, which would make this probe measure the
                        # pre-probe deterministic action instead.
                        bid_loc = policy.model.action_parameters(obs_t)[-1]
                        bids = torch.tanh(bid_loc)
                    for row, i in enumerate(ids):
                        slot = max(0, min(7, int(nominal[i].pointer) - 1))
                        bid_vector = list(nominal[i].arbitration_bids or
                                          (0.0,) * 8)
                        bid_vector[slot] = float(bids[row, slot])
                        nominal[i] = replace(
                            nominal[i],
                            arbitration_bid=float(bid_vector[slot]),
                            arbitration_bids=tuple(bid_vector))
        safe, interventions = shield.filter(env, nominal)
        intervention_count += len(interventions)
        correction_total += sum(float(item.magnitude) for item in interventions)
        result = env.step(safe)
        steps += 1
        total += result.reward
        violations += len(result.info["report"].violations)
        for outcome in result.info.get("request_outcomes", []):
            request_outcomes[outcome.get("reason", "unknown")] += 1
        for key in result.info["report"].violations:
            family = key.split(":", 1)[0]
            family_counts[family] = family_counts.get(family, 0) + 1
        events = result.info.get("arbitration", [])
        conflicts += len(events)
        for event in events:
            candidates = [int(i) for i in event.get("candidates", [])]
            probe_event = sum(
                bool(exposure_mask[i]) for i in candidates) >= 2
            if probe_event:
                probe_conflicts += 1
            bids = event.get("bids", {})
            scores = event.get("scores", {})
            adjusted = event.get("adjusted_scores", {})
            owner = event.get("previous_owner")
            hysteresis = float(event.get("hysteresis", 0.0))
            if scores and adjusted:
                raw_winner = _winner(scores, owner, hysteresis)
                direct_event = event.get("arbitration_mode") == "rl_direct"
                adjusted_winner = (int(event["winner"]) if direct_event else
                                   _winner(adjusted, owner, hysteresis))
                changed = int(raw_winner != adjusted_winner)
                bid_changed_winner += changed
                if probe_event:
                    probe_bid_changed_winner += changed
                    probe_event_details.append({
                        "seed": int(seed),
                        "target_id": int(event.get("tid", -1)),
                        "candidates": candidates,
                        "raw_winner": raw_winner,
                        "adjusted_winner": adjusted_winner,
                        "scores": {str(i): float(scores[i])
                                   for i in candidates},
                        "bids": {str(i): float(bids.get(
                            i, bids.get(str(i), 0.0))) for i in candidates},
                        "adjusted_scores": {str(i): float(adjusted.get(
                            i, adjusted.get(str(i), 0.0)))
                                            for i in candidates},
                        "arbitration_mode": event.get("arbitration_mode"),
                        "previous_owner": owner,
                        "hysteresis": hysteresis,
                    })
                bid_abs_sum += sum(abs(float(v)) for v in bids.values())
                bid_candidate_count += len(bids)
        incumbent_overrides += sum(
            int(e["previous_owner"] is not None
                and e["winner"] != e["previous_owner"])
            for e in events)
    return {
        "controller": controller.__class__.__name__ if policy is None else "RL",
        "seed": seed, "k": k_red, "terminal": env.terminal,
        "stable_window": env.cfg.STABLE_WINDOW,
        "return": round(total, 3), "steps": steps,
        "violations": violations,
        "conflicts": conflicts,
        "bid_changed_winner": bid_changed_winner,
        "rl_direct_arbitration": direct_rl_arbitration,
        "conflict_probe": {
            "enabled": bool(conflict_probe),
            "exposure_steps": probe_exposure_steps,
            "injected_request_pairs": probe_injected_pairs,
            "exposed_agents": probe_exposed_agents,
            "multi_candidate_events": probe_conflicts,
            "bid_changed_winner": probe_bid_changed_winner,
            "bid_winner_change_rate": round(
                probe_bid_changed_winner / max(1, probe_conflicts), 5),
        },
        "mean_abs_bid_on_conflicts": round(
            bid_abs_sum / max(1, bid_candidate_count), 5),
        "conflict_probe_events": probe_event_details,
        "request_outcomes": dict(request_outcomes),
        "incumbent_overrides": incumbent_overrides,
        "shield_interventions": intervention_count,
        "mean_correction": round(correction_total / max(1, intervention_count), 4),
        "violations_by_family": family_counts,
        "active_steps_by_conflict_pair": dict(zip(pair_names, pair_active_steps)),
        "mean_rl_preference_when_active": {},
    }


@track_run_lifecycle
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seeds", default="300,301,302")
    ap.add_argument("--stable-window", type=int)
    ap.add_argument("--age-std-track", type=float)
    ap.add_argument("--c04-range", type=float)
    ap.add_argument("--arb-bid-weight", type=float)
    ap.add_argument("--arb-hysteresis", type=float)
    ap.add_argument(
        "--conflict-probe", action="store_true",
        help="force paired legal local-track requests for a separate diagnostic")
    ap.add_argument(
        "--direct-rl-arbitration", action="store_true",
        help="let the RL policy directly select among C04-feasible candidates")
    ap.add_argument(
        "--verify-heur-repro", action="store_true",
        help="run HEUR twice per seed and require exact metric agreement")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not args.checkpoint and not args.verify_heur_repro:
        ap.error("--checkpoint is required unless --verify-heur-repro is set")
    output_path = args.output.resolve() if args.output else None
    provenance_dir = (output_path.parent if output_path else
                      ROOT / "runs" / "standalone_evaluations")
    cfg = replace(
        DetectSimConfig(),
        **{key: value for key, value in {
            "STABLE_WINDOW": args.stable_window,
            "AGE_STD_TRACK": args.age_std_track,
            "C04_RANGE": args.c04_range,
            "ARB_BID_WEIGHT": args.arb_bid_weight,
            "ARB_HYSTERESIS": args.arb_hysteresis,
        }.items() if value is not None},
    )
    if args.verify_heur_repro and args.conflict_probe:
        ap.error("--verify-heur-repro cannot be combined with --conflict-probe")
    _preflight_output_path(output_path)
    mode = ("heur_reproducibility_check" if args.verify_heur_repro else
            "forced_local_track_conflict_probe" if args.conflict_probe else
            "natural_paired_eval")
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    checkpoint_config = None
    if checkpoint_path:
        if not checkpoint_path.is_file():
            ap.error(f"checkpoint does not exist: {checkpoint_path}")
        checkpoint_data = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint_data.get("config")
    source_entry = [ROOT / "_eval_conflict_paired.py"]
    run_record = capture_run_provenance(
        repo_root=ROOT,
        output_dir=provenance_dir,
        entry_scripts=source_entry,
        phase=mode,
        task_config=cfg,
        algorithm_config={
            "checkpoint_saved_config": checkpoint_config,
            "evaluation_policy_config": asdict(DetectMAPPOConfig(
                direct_rl_arbitration=args.direct_rl_arbitration)),
        } if args.checkpoint else None,
        runtime_parameters={
            "k": args.k,
            "seeds": seeds,
            "mode": mode,
            "conflict_probe": args.conflict_probe,
            "direct_rl_arbitration": args.direct_rl_arbitration,
            "effective_overrides": {
                "conflict_exposure_prob": 1.0 if args.conflict_probe else None,
            },
            "verify_heur_repro": args.verify_heur_repro,
            "environment": {key: value for key, value in os.environ.items()
                            if key.startswith("MUMT_")},
            "arguments": vars(args),
        },
        inputs=([artifact_record(checkpoint_path, "policy_checkpoint")]
                if checkpoint_path else []),
        source_snapshot=SOURCE_SNAPSHOT,
    )
    evaluation_checkpoint_path = checkpoint_path
    output_path, provenance_path = _evaluation_output_paths(
        output_path, provenance_dir, run_record["run_id"], mode)
    run_record["result_path"] = str(output_path)
    run_record["provenance_path"] = str(provenance_path)
    run_record["status"] = "running"
    write_json_atomic(provenance_path, run_record)
    register_active_run(run_record, lambda record: write_json_atomic(
        provenance_path, record))
    if checkpoint_path:
        evaluation_checkpoint_path, input_record = _snapshot_checkpoint(
            checkpoint_path, provenance_dir, run_record["run_id"])
        run_record["inputs"][0] = input_record
        write_json_atomic(provenance_path, run_record)

    heur = HeurPolicy()
    rows = []
    mismatches = []
    if args.verify_heur_repro:
        first = [run(heur, None, args.k, seed, cfg) for seed in seeds]
        second = [run(heur, None, args.k, seed, cfg) for seed in seeds]
        comparable = ("terminal", "return", "steps", "violations")
        mismatches = [
            {"seed": seed,
             "first": {key: first_row[key] for key in comparable},
             "second": {key: second_row[key] for key in comparable}}
            for seed, first_row, second_row in zip(seeds, first, second)
            if any(first_row[key] != second_row[key] for key in comparable)
        ]
        rows = [{"seed": seed, "first": first_row, "second": second_row}
                for seed, first_row, second_row in zip(seeds, first, second)]
    elif args.conflict_probe:
        for seed in seeds:
            rows.append(run(None, str(evaluation_checkpoint_path), args.k, seed, cfg,
                            conflict_probe=True,
                            direct_rl_arbitration=args.direct_rl_arbitration))
    else:
        for seed in seeds:
            rows.append({"heur": run(heur, None, args.k, seed, cfg),
                         "rl": run(None, str(evaluation_checkpoint_path), args.k,
                                   seed, cfg,
                                   direct_rl_arbitration=args.direct_rl_arbitration)})
    if checkpoint_path:
        before_hash = run_record["inputs"][0]["sha256"]
        after_hash = artifact_record(
            checkpoint_path, "policy_checkpoint_after_evaluation")["sha256"]
        if after_hash != before_hash:
            raise RuntimeError(
                "Policy checkpoint changed while it was being evaluated; "
                "the results were not written.")
        snapshot_after_hash = artifact_record(
            evaluation_checkpoint_path,
            "policy_checkpoint_snapshot_after_evaluation")["sha256"]
        if snapshot_after_hash != run_record["inputs"][0][
                "archived_snapshot"]["sha256"]:
            raise RuntimeError(
                "Archived policy checkpoint changed during evaluation; "
                "the results were not written.")
        run_record["checkpoint_unchanged_during_evaluation"] = True
        run_record["evaluation_checkpoint"] = artifact_record(
            evaluation_checkpoint_path, "evaluated_policy_checkpoint")
    payload = {"checkpoint": (str(checkpoint_path)
                               if checkpoint_path else None), "k": args.k,
               "mode": mode,
               "stable_window": cfg.STABLE_WINDOW,
               "age_std_track": cfg.AGE_STD_TRACK,
               "arb_bid_weight": cfg.ARB_BID_WEIGHT,
               "arb_hysteresis": cfg.ARB_HYSTERESIS,
               "direct_rl_arbitration": args.direct_rl_arbitration,
               "task_config": run_record["task_config"],
               "task_config_sha256": run_record["task_config_sha256"],
               "provenance": {"run_id": run_record["run_id"],
                              "sidecar": str(provenance_path)}}
    if args.verify_heur_repro:
        payload["runs"] = rows
        payload["exact_match"] = not mismatches
        payload["mismatches"] = mismatches
    else:
        payload["probe_runs" if args.conflict_probe else "paired"] = rows
    text = json.dumps(payload, indent=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    add_output(run_record, output_path, "evaluation_result")
    run_record["status"] = "completed"
    run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(provenance_path, run_record)
    print(text)
    if args.verify_heur_repro and mismatches:
        raise SystemExit(f"HEUR reproducibility check failed: {len(mismatches)} mismatches")


if __name__ == "__main__":
    main()
