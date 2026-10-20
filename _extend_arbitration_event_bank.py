"""Continue held-out collection after a coverage stop, preserving existing cases.

No learning or performance-based seed selection occurs in this tool. A fixed
new seed block is collected in full; the original event bank stays untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import torch

from _audit_arbitration_learning import _json_ready, case_coverage, collect_cases
from _curriculum_guard import file_sha256
from _experiment_provenance import (
    canonical_sha256, capture_source_snapshot, changed_source_files, write_json_atomic,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))


def load_stopped_bank(source_run: Path, extra_seeds: list[int]):
    report_path = source_run / "learning_audit.json"
    bank_path = source_run / "event_bank.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    content = {key: value for key, value in report.items()
               if key != "report_content_sha256"}
    if canonical_sha256(content) != report.get("report_content_sha256"):
        raise ValueError("Original stopped report content hash mismatch")
    if (report.get("status") != "insufficient_independent_event_coverage"
            or report.get("updates") != []):
        raise ValueError("Extension requires a coverage stop before any training")
    if any(source_run.glob("*.pt")):
        raise ValueError("Stopped source run unexpectedly contains model checkpoints")
    if file_sha256(bank_path) != report["event_bank_sha256"]:
        raise ValueError("Original event bank SHA256 mismatch")
    if report.get("source_changed_during_run"):
        raise ValueError("Original source changed during collection")
    if changed_source_files(bank["collection_source_snapshot"]):
        raise ValueError("Original collector/simulator sources changed; cannot mix utilities")
    snapshot = bank["collection_source_snapshot"]
    if file_sha256(ROOT / snapshot["snapshot_path"]) != snapshot["snapshot_sha256"]:
        raise ValueError("Original collection source archive SHA256 mismatch")
    arguments = report["arguments"]
    checkpoint = (ROOT / arguments["checkpoint"]).resolve()
    digest = file_sha256(checkpoint)
    if digest != bank["source_checkpoint_sha256"] or digest != report["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint SHA256 mismatch")
    original_seeds = bank["train_seeds"] + bank["heldout_seeds"]
    if len(set(original_seeds)) != len(original_seeds):
        raise ValueError("Original train and held-out episode seeds overlap")
    if (not extra_seeds or min(extra_seeds) < 0
            or len(set(extra_seeds)) != len(extra_seeds)
            or set(extra_seeds) & set(original_seeds)):
        raise ValueError("New held-out seeds must be unique and disjoint from both original splits")
    if (bank["train_seeds"] != [int(value) for value in arguments["train_seeds"].split(",")]
            or bank["heldout_seeds"] != [int(value) for value in arguments["heldout_seeds"].split(",")]):
        raise ValueError("Original collection seed record mismatch")
    expected_selection = {"max_events_per_episode": arguments["max_events_per_episode"],
                          "min_event_gap": arguments["min_event_gap"]}
    if bank["selection_policy"] != expected_selection:
        raise ValueError("Original temporal selection policy mismatch")
    for split in ("train", "heldout"):
        if case_coverage(bank[split]) != report["coverage"][split]:
            raise ValueError("Original event coverage record mismatch")
        if any(case["episode_seed"] not in bank[split + "_seeds"] for case in bank[split]):
            raise ValueError("Event episode is outside its recorded split")
    if report["coverage"]["train"]["informative_episodes"] < arguments["min_train_event_episodes"]:
        raise ValueError("Training coverage is also insufficient; held-out-only extension is inapplicable")
    remaining = arguments["max_events_per_split"] - len(bank["heldout"])
    if remaining <= 0:
        raise ValueError("No capacity remains in the fixed held-out event budget")
    return report, bank, checkpoint, remaining


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-seeds", required=True)
    parser.add_argument("--collection-seed", type=int, default=20261023)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    source_run = (ROOT / args.source_run).resolve()
    output = (ROOT / args.output).resolve()
    if (source_run == output or source_run in output.parents or output in source_run.parents):
        raise ValueError("Output must be separate from the original collection")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Extension output must be new or empty")
    if not 0 <= args.collection_seed < 2**63:
        raise ValueError("Collection seed must be in [0, 2**63)")
    extra_seeds = [int(value) for value in args.heldout_seeds.split(",")]
    original, bank, checkpoint, remaining = load_stopped_bank(source_run, extra_seeds)
    plan = {
        "status": "preflight_passed", "source_run": str(source_run),
        "source_checkpoint_sha256": bank["source_checkpoint_sha256"],
        "original_coverage": original["coverage"],
        "extra_heldout_seeds": extra_seeds, "collection_seed": args.collection_seed,
        "training_collection_reused": True,
        "min_heldout_event_episodes": original["arguments"]["min_heldout_event_episodes"],
        "selection_policy": bank["selection_policy"], "remaining_event_budget": remaining,
        "seed_budget_policy": "collect_entire_fixed_block_without_performance_selection",
    }
    if args.preflight_only:
        print(json.dumps(plan, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    snapshot = capture_source_snapshot(ROOT, output, [Path(__file__),
        ROOT / "_audit_arbitration_learning.py", ROOT / "_arbitration_training.py"])
    from aircraft_sim.detect_config import DetectSimConfig
    from aircraft_sim.detect_env import DetectEnv
    from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
    from aircraft_sim.detect_shield import DetectQPShield
    from _arbitration_training import repair_dead_candidate_head_for_training

    settings = original["arguments"]
    torch.set_num_threads(1)
    env_cfg = replace(DetectSimConfig(), STABLE_WINDOW=8, AGE_STD_TRACK=10.0,
                      SECTOR_AXIS_MAX_DEG=60.0, C04_RANGE=80.0,
                      ARB_BID_WEIGHT=0.35, ARB_HYSTERESIS=0.2)
    policy_cfg = DetectMAPPOConfig(
        bid_only_ppo=True, direct_rl_arbitration=True,
        conflict_exposure_prob=settings["conflict_exposure"],
        learning_rate=settings["learning_rate"], epochs=settings["epochs"],
        cf_bid_utility_scale=settings["utility_scale"],
        cf_bid_horizon=settings["cf_bid_horizon"], entropy_coef=0.0, policy_anchor_coef=0.0)
    env = DetectEnv(env_cfg)
    policy = DetectMAPPOPolicy(env, DetectQPShield(env_cfg),
                               config=policy_cfg, seed=settings["training_seed"])
    policy.load(str(checkpoint))
    policy.config = policy_cfg
    repair = repair_dead_candidate_head_for_training(policy.model, seed=settings["repair_seed"])
    if _json_ready(asdict(env_cfg)) != bank["environment_config"]:
        raise ValueError("Environment config differs from original collection")
    utility = {name: getattr(policy_cfg, name) for name in (
        "gamma", "reward_scale", "shaping_kappa", "shaping_age_cap",
        "geo_shaping_kappa", "stable_shaping_kappa")}
    if policy_cfg.cf_bid_horizon != 1:
        utility.update(cf_bid_horizon=policy_cfg.cf_bid_horizon,
                       continuation="frozen_deterministic_rl_actor_and_independent_branch_shield")
    if utility != bank["utility_definition"]:
        raise ValueError("Counterfactual utility differs from original collection")
    model_before = {name: value.detach().clone() for name, value in policy.model.state_dict().items()}
    # A new, logged independent collection stream; no weights or labels change.
    torch.manual_seed(args.collection_seed)
    cases, episodes = collect_cases(policy, extra_seeds, bank["k"], remaining,
                                    settings["max_events_per_episode"], settings["min_event_gap"])
    if any(not torch.equal(value, policy.model.state_dict()[name])
           for name, value in model_before.items()):
        raise RuntimeError("Collection unexpectedly changed policy weights")
    changed = changed_source_files(snapshot)
    if changed or changed_source_files(bank["collection_source_snapshot"]):
        raise RuntimeError("Source changed during held-out extension")
    # Recheck original artifacts before persisting the merged bank.
    load_stopped_bank(source_run, extra_seeds)
    merged = deepcopy(bank)
    merged["heldout"].extend(cases)
    merged["heldout_seeds"].extend(extra_seeds)
    merged["collection_episodes"]["heldout"].extend(episodes)
    merged["collection_extensions"] = [{
        "split": "heldout", "seeds": extra_seeds, "collection_seed": args.collection_seed,
        "source_snapshot": snapshot, "repair": repair,
        "original_bank": str(source_run / "event_bank.json"),
        "original_bank_sha256": original["event_bank_sha256"],
        "original_case_count": len(bank["heldout"]), "added_case_count": len(cases),
    }]
    bank_path = output / "event_bank.json"
    write_json_atomic(bank_path, merged)
    coverage = {split: case_coverage(merged[split]) for split in ("train", "heldout")}
    report = {**plan, "status": "collection_completed_without_training",
              "coverage": coverage, "source_snapshot": snapshot,
              "source_changed_during_run": changed,
              "event_bank_sha256": file_sha256(bank_path), "updates": [],
              "training_cases_unchanged": merged["train"] == bank["train"],
              "original_heldout_cases_unchanged": merged["heldout"][:len(bank["heldout"])] == bank["heldout"],
              "original_report_sha256": file_sha256(source_run / "learning_audit.json")}
    report["report_content_sha256"] = canonical_sha256(report)
    write_json_atomic(output / "collection_report.json", report)
    print(json.dumps({"status": report["status"], "coverage": coverage,
                      "event_bank": str(bank_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
