"""Screen explicit detection-task simplifications before retraining.

The comparison freezes one RL checkpoint and evaluates it against HEUR on
paired seeds.  It changes only numeric task configuration fields; rules,
red-agent code, and the arbitration implementation remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_configs(k: int, stable_window: int, age_std_track: float):
    from aircraft_sim.detect_config import detect_config_profile

    common = {"K_RED": k, "STABLE_WINDOW": stable_window,
              "AGE_STD_TRACK": age_std_track}
    return {
        "baseline": detect_config_profile("baseline", **common),
        # Isolate the geometry change from simplified_conflict's many
        # simultaneous radar/tracking/rule overrides.  This widens the shared
        # observation and C04 feasible region while leaving those rules intact.
        "wide_observation_baseline": detect_config_profile(
            "baseline", **common, FOV_HALF_DEG=30.0,
            R_MAX=120.0, C04_RANGE=120.0),
        "simplified_tracking": detect_config_profile(
            "simplified_tracking", **common),
        "simplified_conflict": detect_config_profile(
            "simplified_conflict", **common),
        "simplified_wide": detect_config_profile(
            "simplified_conflict", **common, FOV_HALF_DEG=30.0,
            R_MAX=120.0, C04_RANGE=120.0),
        "k2_lane_center_40": detect_config_profile(
            "k2_lane_center_40", **common),
        "k2_lane_center_50": detect_config_profile(
            "k2_lane_center_50", **common),
        "k2_lane_center_50_threat_cone_sensor_simple": detect_config_profile(
            "k2_lane_center_50_threat_cone_sensor_simple", **common),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--expected-checkpoint-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--stable-window", type=int, default=8)
    ap.add_argument("--age-std-track", type=float, default=10.0)
    ap.add_argument("--seed-start", type=int, default=270000)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--profiles", nargs="+", choices=(
        "baseline", "wide_observation_baseline", "simplified_tracking",
        "simplified_conflict", "simplified_wide", "k2_lane_center_40",
        "k2_lane_center_50", "k2_lane_center_50_threat_cone_sensor_simple"),
        default=None, help="profiles to evaluate (default: all)")
    args = ap.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    actual = sha256(checkpoint)
    if actual != args.expected_checkpoint_sha256:
        raise SystemExit("checkpoint SHA256 differs from declared input")
    if args.k < 1 or args.k > 12 or args.episodes < 1:
        raise SystemExit("k must be in [1, 12] and episodes must be positive")

    import torch
    from _diagnose_sensor_profile import run_episode, summarize

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_config = checkpoint_data.get("config")
    if not isinstance(saved_config, dict):
        raise SystemExit("checkpoint has no saved policy config")
    configs = make_configs(args.k, args.stable_window, args.age_std_track)
    if args.profiles:
        configs = {name: configs[name] for name in args.profiles}
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    rows = []
    for profile, cfg in configs.items():
        for controller in ("rl", "heur"):
            for seed in seeds:
                row = run_episode(cfg, controller, seed, checkpoint, saved_config)
                row["task_profile"] = profile
                rows.append(row)
                print(f"{profile}/{controller} seed={seed}: "
                      f"{row['terminal']} return={row['return']:.1f} "
                      f"vio/step={row['violations_per_step']:.4f}", flush=True)
    summaries = {
        profile: {
            controller: summarize([r for r in rows
                                    if r["task_profile"] == profile
                                    and r["controller"] == controller])
            for controller in ("rl", "heur")
        }
        for profile in configs
    }
    report = {
        "status": "completed_frozen_profile_screen",
        "checkpoint": {"path": str(checkpoint), "sha256": actual},
        "seeds": seeds,
        "task_profiles": {name: asdict(cfg) for name, cfg in configs.items()},
        "summaries": summaries,
        "rows": rows,
        "rules_unchanged": True,
        "teacher_untouched": True,
        "training": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "profile_sweep.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output / 'profile_sweep.json'}")


if __name__ == "__main__":
    main()
