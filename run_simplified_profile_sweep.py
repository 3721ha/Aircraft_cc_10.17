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
        "simplified_tracking": detect_config_profile(
            "simplified_tracking", **common),
        "simplified_conflict": detect_config_profile(
            "simplified_conflict", **common),
        "simplified_wide": detect_config_profile(
            "simplified_conflict", **common, FOV_HALF_DEG=30.0,
            R_MAX=120.0, C04_RANGE=120.0),
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
