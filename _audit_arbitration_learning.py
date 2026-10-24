"""Audit bid PPO on frozen conflicts from real simulator trajectories.

The fixed cases retain the local observations, task actions, candidate slots
and finite-horizon environment-copy utilities that collection produced.
Only bids are resampled during this diagnostic. Training and held-out cases
come from disjoint episode seeds; no HEUR labels or truth fixtures are used.
This measures conditional arbitration learning, not episode success.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _event_values(event: dict, n_agents: int = 3
                  ) -> tuple[list[int], dict[int, int], np.ndarray]:
    candidates = sorted(int(agent) for agent in event["candidates"])
    slots = {int(agent): int(slot)
             for agent, slot in event["bid_slots"].items()}
    utilities = {int(agent): float(utility)
                 for agent, utility in event["utilities"].items()}
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("A frozen case needs distinct real conflict candidates")
    if any(not 0 <= agent < n_agents for agent in candidates):
        raise ValueError("Conflict candidate is outside the agent range")
    if any(agent not in slots or agent not in utilities for agent in candidates):
        raise ValueError("Candidate, slot and environment utility table mismatch")
    if any(not 0 <= slots[agent] < 8 for agent in candidates):
        raise ValueError("Invalid frozen candidate slot")
    values = np.asarray([utilities[agent] for agent in candidates], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite frozen environment utility")
    return candidates, slots, values


def _case_template(row: dict, n_agents: int) -> dict:
    # Every case is an independent one-step bid diagnostic.  Episode GAE and
    # task-head losses do not participate; update() still runs the real PPO.
    return _json_ready({
        **{key: deepcopy(row[key]) for key in (
            "obs", "team", "nominal", "safe", "old_logp", "bid_actions")},
        "reward": 0.0, "value": 0.0, "done": True, "f_role": 0.0,
        "gear_gate_credit": [0.0] * n_agents,
        "cf_deltas": [0.0] * n_agents,
        "bid_rank": [0.0] * n_agents,
        "bid_mask": [False] * n_agents,
        "action_exposure_mask": [True] * n_agents,
    })


def _event_exposure_source(row: dict, event: dict) -> str:
    """Mark a conflict exposed when one of its candidates had a forced request."""
    mask = row.get("action_exposure_mask", [False] * 3)
    candidates = [int(agent) for agent in event["candidates"]]
    if any(not 0 <= agent < len(mask) for agent in candidates):
        raise ValueError("Conflict candidate is outside the exposure mask")
    return ("exposed" if any(bool(mask[agent]) for agent in candidates)
            else "natural")


def _case_exposure_source(case: dict) -> str:
    template = case.get("template", {})
    mask = case.get("collection_exposure_mask")
    observations = template.get("obs", {})
    if (not isinstance(mask, list) or len(mask) != len(observations)
            or any(type(value) is not bool for value in mask)):
        raise ValueError("Frozen case has an invalid collection exposure mask")
    source = _event_exposure_source(
        {"action_exposure_mask": mask}, case["event"])
    if case.get("exposure_source") != source:
        raise ValueError("Frozen case exposure source disagrees with its mask")
    return source


def collect_cases(policy, seeds: list[int], k: int, max_cases: int,
                  max_per_episode: int | None = None,
                  min_event_gap: int = 0,
                  collection_seed: int | None = None) -> tuple[list, list]:
    """Collect untouched real rollouts; freeze only validated simulator events."""
    cases, episode_records = [], []
    collection_rng = (random.Random(collection_seed)
                      if collection_seed is not None else None)
    for seed in seeds:
        conflict_precheck = getattr(policy, "_may_have_arbitration_conflict", None)
        if callable(conflict_precheck):
            # This precheck can miss conflicts after freshness or C04 changes in a step.
            policy._may_have_arbitration_conflict = lambda _actions: True
        try:
            if collection_rng is None:
                rollout = policy.collect(k_red=k, seed=seed, deterministic=False,
                                         w8_scale=1.0)
                actor_rng_seed = None
            else:
                actor_rng_seed = collection_rng.getrandbits(63)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(actor_rng_seed)
                    rollout = policy.collect(k_red=k, seed=seed,
                                             deterministic=False, w8_scale=1.0)
        finally:
            if callable(conflict_precheck):
                policy._may_have_arbitration_conflict = conflict_precheck
        found, retained = 0, 0
        last_retained_step = None
        for step_index, row in enumerate(rollout):
            if "bid_cf_event_records" not in row:
                raise RuntimeError("Collector lacks frozen environment utility records")
            for event in row["bid_cf_event_records"]:
                _event_values(event, policy.env.cfg.N_WINGMEN)
                found += 1
                if (len(cases) < max_cases
                        and (max_per_episode is None or retained < max_per_episode)
                        and (last_retained_step is None
                             or step_index - last_retained_step >= min_event_gap)):
                    exposure_source = _event_exposure_source(row, event)
                    cases.append({
                        "episode_seed": seed, "step_index": step_index,
                        "exposure_source": exposure_source,
                        "collection_exposure_mask": [
                            bool(value) for value in row.get(
                                "action_exposure_mask", [False] *
                                policy.env.cfg.N_WINGMEN)],
                        "event": _json_ready(event),
                        "template": _case_template(row, policy.env.cfg.N_WINGMEN),
                    })
                    retained += 1
                    last_retained_step = step_index
        episode_records.append({"seed": seed, "actor_rng_seed": actor_rng_seed,
                                "steps": len(rollout),
                                "controllable_events": found,
                                "retained_events": retained})
        print(json.dumps({"collection_seed": seed, "steps": len(rollout),
                          "actor_rng_seed": actor_rng_seed,
                          "events": found, "retained": len(cases)}), flush=True)
    return cases, episode_records


def case_coverage(cases: list[dict]) -> dict:
    counts, informative_counts = {}, {}
    by_source, informative_by_source = {}, {}
    informative_episodes_by_source = {}
    for case in cases:
        seed = str(case["episode_seed"])
        source = case.get("exposure_source", "natural")
        counts[seed] = counts.get(seed, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1
        if np.ptp(_event_values(case["event"])[2]) > 1e-8:
            informative_counts[seed] = informative_counts.get(seed, 0) + 1
            informative_by_source[source] = informative_by_source.get(source, 0) + 1
            informative_episodes_by_source.setdefault(source, set()).add(seed)
    return {"events": len(cases),
            "informative_events": sum(informative_counts.values()),
            "event_episodes": len(counts),
            "informative_episodes": len(informative_counts),
            "events_by_source": by_source,
            "informative_events_by_source": informative_by_source,
            "informative_episodes_by_source": {
                source: len(seeds)
                for source, seeds in informative_episodes_by_source.items()},
            "events_by_episode": counts,
            "informative_events_by_episode": informative_counts}


def sample_training_cases(cases, count, sampler, episode_balanced=False,
                          source_balanced=False):
    if not cases:
        raise ValueError("Cannot sample an empty training event population")
    if source_balanced:
        groups = {}
        for case in cases:
            source = case.get("exposure_source", "natural")
            groups.setdefault(source, {}).setdefault(
                case["episode_seed"], []).append(case)
        sources = sorted(groups)
        selected = []
        for _ in range(count):
            source = sampler.choice(sources)
            episode = sampler.choice(sorted(groups[source]))
            selected.append(sampler.choice(groups[source][episode]))
        return selected
    if not episode_balanced:
        return [sampler.choice(cases) for _ in range(count)]
    groups = {}
    for case in cases:
        groups.setdefault(case["episode_seed"], []).append(case)
    episodes = sorted(groups)
    return [sampler.choice(groups[sampler.choice(episodes)]) for _ in range(count)]


def _environment_config(args):
    from aircraft_sim.detect_config import detect_config_profile

    if args.scenario_profile.startswith("k2_lane_center_") and args.k != 2:
        raise ValueError("K2 lane-center profiles require --k 2")
    return detect_config_profile(
        args.scenario_profile, K_RED=args.k, STABLE_WINDOW=8,
        AGE_STD_TRACK=10.0, RADAR_HIT_MODEL=args.radar_hit_model,
        SECTOR_AXIS_MAX_DEG=60.0, C04_RANGE=80.0,
        ARB_BID_WEIGHT=0.35, ARB_HYSTERESIS=0.2)


def _split_collection_seeds(seed: int) -> dict[str, int]:
    if not 0 <= seed < 2**63 - 1:
        raise ValueError("collection seed must be in [0, 2**63 - 2]")
    return {"train": seed, "heldout": seed + 1}


def _runtime_fingerprint() -> dict:
    return {"python": sys.version, "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads()}


def _parse_seed_spec(spec: str) -> list[int]:
    seeds = []
    for component in spec.split(","):
        component = component.strip()
        if not component:
            raise ValueError("Seed specification contains an empty item")
        if ":" in component:
            parts = component.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid seed range {component!r}")
            start, stop = (int(value) for value in parts)
            if stop < start:
                raise ValueError(f"Descending seed range {component!r}")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(component))
    return seeds


def _validate_event_bank(bank: dict, *, checkpoint_hash: str, args,
                         env_cfg, utility_definition: dict,
                         train_seeds: list[int], heldout_seeds: list[int],
                         collection_rng_seeds: dict,
                         selection_policy: dict, runtime: dict,
                         source_manifest_sha256: str) -> None:
    if bank.get("schema_version") != 3:
        raise ValueError("Frozen event bank uses an unsupported schema")
    expected = {
        "source_checkpoint_sha256": checkpoint_hash,
        "k": args.k,
        "environment_config": _json_ready(asdict(env_cfg)),
        "utility_definition": utility_definition,
        "scenario_profile": args.scenario_profile,
        "radar_hit_model": args.radar_hit_model,
        "train_exposure_probability": args.conflict_exposure,
        "heldout_exposure_probability": 0.0,
        "collection_rng_seeds": collection_rng_seeds,
        "runtime": runtime,
        "collection_source_manifest_sha256": source_manifest_sha256,
        "teacher_used": False,
        "rule_inventory_modified": False,
        "red_model_modified": False,
        "train_seeds": train_seeds,
        "heldout_seeds": heldout_seeds,
        "selection_policy": selection_policy,
    }
    for key, value in expected.items():
        if bank.get(key) != value:
            raise ValueError(f"Frozen event bank {key} mismatch")
    for split, allowed_seeds in (("train", set(train_seeds)),
                                 ("heldout", set(heldout_seeds))):
        for case in bank.get(split, []):
            if case.get("episode_seed") not in allowed_seeds:
                raise ValueError(f"Frozen {split} case has an unknown episode seed")
            source = _case_exposure_source(case)
            if split == "heldout" and source != "natural":
                raise ValueError("Held-out event bank must contain natural conflicts only")


def _obs_tensor(case: dict) -> torch.Tensor:
    obs = case["template"]["obs"]
    return torch.tensor(np.asarray([obs.get(str(agent), obs.get(agent))
                                   for agent in range(len(obs))]),
                        dtype=torch.float32)


def evaluate_cases(model, cases: list[dict], samples: int, noise_seed: int) -> dict:
    """Expected and deterministic regret with paired common random numbers."""
    if samples < 2:
        raise ValueError("Expected regret requires at least two Monte Carlo samples")
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    rows = []
    for case in cases:
        event = case["event"]
        candidates, slots, utilities = _event_values(event)
        with torch.no_grad():
            loc = model.action_parameters(_obs_tensor(case))[-1]
            means = torch.stack([loc[agent, slots[agent]] for agent in candidates])
            noises = torch.randn((samples, len(candidates)), generator=generator)
            # tanh is monotone, but use executed bounded bids for exact ties.
            bids = torch.tanh(means.unsqueeze(0)
                              + model.bid_log_std.exp() * noises)
            winners = bids.argmax(-1).numpy()
            deterministic = int(torch.tanh(means).argmax())
        best, spread = float(utilities.max()), float(np.ptp(utilities))
        regrets = best - utilities[winners]
        probabilities = {str(agent): float(np.mean(winners == index))
                         for index, agent in enumerate(candidates)}
        rows.append({
            "episode_seed": case["episode_seed"],
            "step_index": case["step_index"], "tid": event["tid"],
            "utility_spread": spread,
            "expected_regret": float(regrets.mean()),
            "expected_optimal_winner_rate": float(np.mean(regrets <= 1e-8)),
            "deterministic_regret": best - float(utilities[deterministic]),
            "deterministic_winner": candidates[deterministic],
            "winner_probabilities": probabilities,
        })
    informative = [row for row in rows if row["utility_spread"] > 1e-8]

    def aggregate(selected):
        return {name: float(np.mean([row[name] for row in selected]))
                if selected else None for name in (
                    "expected_regret", "expected_optimal_winner_rate",
                    "deterministic_regret")}

    by_episode = {}
    for row in informative:
        by_episode.setdefault(str(row["episode_seed"]), []).append(row)
    episode_rows = [aggregate(items) for items in by_episode.values()]
    return {"events": len(rows), "informative_events": len(informative),
            "monte_carlo_samples_per_event": samples,
            **aggregate(rows), "informative": aggregate(informative),
            "informative_episode_balanced": aggregate(episode_rows),
            "informative_by_episode": {seed: aggregate(items)
                                        for seed, items in by_episode.items()},
            "rows": rows}


def resample_case_row(policy, case: dict) -> dict:
    """Sample bid actions and assign each candidate its relative utility."""
    from aircraft_sim.detect_mappo import _squashed_log_prob

    row = deepcopy(case["template"])
    for key in ("obs", "nominal", "safe", "old_logp", "bid_actions"):
        row[key] = {int(agent): value for agent, value in row[key].items()}
    n_agents = len(row["obs"])
    candidates, slots, utilities = _event_values(case["event"])
    with torch.no_grad():
        loc = policy.model.action_parameters(_obs_tensor(case))[-1]
        distribution = Normal(loc, policy.model.bid_log_std.exp())
        sampled = torch.tanh(distribution.sample())
        bids = torch.tensor([row["bid_actions"][agent]
                             for agent in range(n_agents)], dtype=torch.float32)
        for agent in candidates:
            bids[agent, slots[agent]] = sampled[agent, slots[agent]]
        old_logps = _squashed_log_prob(distribution, bids)
    winner = max(candidates, key=lambda agent: (float(bids[agent, slots[agent]]),
                                               -agent))
    winner_index = candidates.index(winner)
    utility_mean = float(utilities.mean())
    candidate_advantages = {
        agent: float(utilities[index] - utility_mean)
        for index, agent in enumerate(candidates)
    }
    advantage = candidate_advantages[winner]
    informative = float(np.ptp(utilities)) > 1e-8
    masks = [[False] * 8 for _ in range(n_agents)]
    event_ids = [[-1] * 8 for _ in range(n_agents)]
    advantages = [[0.0] * 8 for _ in range(n_agents)]
    for agent in range(n_agents):
        row["bid_actions"][agent] = bids[agent].tolist()
        row["old_logp"][agent]["bid_slots"] = old_logps[agent].tolist()
        pointer = int(row["nominal"][agent][5])
        if pointer > 0:
            selected = pointer - 1
            row["nominal"][agent][6] = float(bids[agent, selected])
            row["safe"][agent][6] = float(bids[agent, selected])
            row["old_logp"][agent]["bid"] = float(old_logps[agent, selected])
        if agent in candidates:
            slot = slots[agent]
            masks[agent][slot] = informative
            event_ids[agent][slot] = 0
            advantages[agent][slot] = candidate_advantages[agent]
            row["bid_mask"][agent] = True
    row["bid_cf_adv_slots"] = advantages
    row["bid_cf_mask_slots"] = masks
    row["bid_event_mask_slots"] = [
        [event_id >= 0 for event_id in agent_ids] for agent_ids in event_ids]
    row["bid_event_ids_slots"] = event_ids
    row["bid_cf_event_records"] = [{
        **deepcopy(case["event"]), "winner": winner,
        "event_advantage": advantage,
        "candidate_advantages": candidate_advantages,
        "actual_regret": float(utilities.max() - utilities[winner_index]),
    }]
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--scenario-profile", choices=(
        "baseline", "k2_lane_center_40", "k2_lane_center_50",
        "k2_lane_center_50_threat_cone_15_40_60",
        "k2_lane_center_50_threat_cone_simple_red"),
        default="k2_lane_center_50")
    parser.add_argument("--radar-hit-model", choices=(
        "bernoulli", "deterministic_beam"), default="bernoulli")
    parser.add_argument("--train-seeds", default="910000:910199")
    parser.add_argument("--heldout-seeds", default="920000:920099")
    parser.add_argument("--max-events-per-split", type=int, default=2000)
    parser.add_argument("--event-bank", type=Path,
                        help="Reuse frozen real cases from this audit's event_bank.json")
    parser.add_argument("--updates", type=int, default=48)
    parser.add_argument("--events-per-update", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--utility-scale", type=float, default=0.1)
    parser.add_argument("--cf-bid-horizon", type=int, default=4,
                        help="Candidate rollout decisions; continuation is frozen deterministic RL.")
    parser.add_argument("--conflict-exposure", type=float, default=0.5,
                        help="Existing training-only request exposure; no fixture or truth tracks.")
    parser.add_argument("--max-events-per-episode", type=int, default=8)
    parser.add_argument("--min-event-gap", type=int, default=4)
    parser.add_argument("--episode-balanced",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-balanced",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Sample natural and exposed cases equally when both exist.")
    parser.add_argument("--min-train-event-episodes", type=int, default=20)
    parser.add_argument("--min-train-source-event-episodes", type=int, default=10)
    parser.add_argument("--min-heldout-event-episodes", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=4096)
    parser.add_argument("--training-seed", type=int, default=20261021)
    parser.add_argument("--collection-seed", type=int, default=20261023)
    parser.add_argument("--repair-seed", type=int, default=20261021)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from _curriculum_guard import file_sha256
    from _experiment_provenance import (canonical_sha256, capture_source_snapshot,
                                        changed_source_files, write_json_atomic)

    if not args.source_balanced or not args.episode_balanced:
        raise ValueError("This training protocol requires source- and episode-balanced sampling")
    checkpoint = (ROOT / args.checkpoint).resolve()
    checkpoint_hash = file_sha256(checkpoint)
    if args.expected_sha256 and checkpoint_hash != args.expected_sha256:
        raise ValueError("Source checkpoint SHA256 mismatch")
    if any(value < 1 for value in (args.k, args.max_events_per_split, args.updates,
                                   args.events_per_update, args.epochs)):
        raise ValueError("Audit budgets must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not np.isfinite(args.utility_scale) or args.utility_scale <= 0:
        raise ValueError("Utility scale must be finite and positive")
    if not 1 <= args.cf_bid_horizon <= 32:
        raise ValueError("Counterfactual horizon must be in [1, 32]")
    if not np.isfinite(args.conflict_exposure) or not 0 <= args.conflict_exposure <= 1:
        raise ValueError("Conflict exposure must be in [0, 1]")
    if (args.min_event_gap < 0
            or (args.max_events_per_episode is not None and args.max_events_per_episode < 1)
            or args.min_train_event_episodes < 1 or args.min_heldout_event_episodes < 1
            or args.min_train_source_event_episodes < 1):
        raise ValueError("Event spacing must be nonnegative and coverage budgets positive")
    if args.mc_samples < 2:
        raise ValueError("Expected regret requires at least two Monte Carlo samples")
    collection_rng_seeds = _split_collection_seeds(args.collection_seed)
    train_seeds = _parse_seed_spec(args.train_seeds)
    heldout_seeds = _parse_seed_spec(args.heldout_seeds)
    if (not train_seeds or not heldout_seeds or min(train_seeds + heldout_seeds) < 0
            or len(set(train_seeds)) != len(train_seeds)
            or len(set(heldout_seeds)) != len(heldout_seeds)
            or set(train_seeds) & set(heldout_seeds)):
        raise ValueError("Collection splits require distinct, disjoint episode seeds")
    if (len(train_seeds) < args.min_train_event_episodes
            or len(heldout_seeds) < args.min_heldout_event_episodes):
        raise ValueError("Requested coverage exceeds the number of episode seeds")
    output = (ROOT / args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Audit output must be new or empty")
    selection_policy = {"max_events_per_episode": args.max_events_per_episode,
                        "min_event_gap": args.min_event_gap}
    if args.preflight_only:
        env_cfg = _environment_config(args)
        print(json.dumps({"status": "preflight_passed", "source_checkpoint": str(checkpoint),
                          "source_checkpoint_sha256": checkpoint_hash,
                          "scenario_profile": args.scenario_profile,
                          "radar_hit_model": args.radar_hit_model,
                          "environment_config": _json_ready(asdict(env_cfg)),
                          "collection_rng_seeds": collection_rng_seeds,
                          "runtime": _runtime_fingerprint(),
                          "selection_policy": selection_policy,
                          "arguments": {key: str(value) if isinstance(value, Path) else value
                                        for key, value in vars(args).items()}}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    snapshot = capture_source_snapshot(ROOT, output, [
        Path(__file__), ROOT / "_arbitration_training.py",
        ROOT / "_curriculum_guard.py",
        ROOT / "runs" / "model_optimization_v1" /
        "arbitration_head_training_protocol_v1.md"])
    from aircraft_sim.detect_env import DetectEnv
    from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
    from aircraft_sim.detect_shield import DetectQPShield
    from _arbitration_training import repair_dead_candidate_head_for_training

    torch.set_num_threads(1)
    runtime = _runtime_fingerprint()
    env_cfg = _environment_config(args)
    policy_cfg = DetectMAPPOConfig(
        bid_only_ppo=True, direct_rl_arbitration=True,
        conflict_exposure_prob=args.conflict_exposure, learning_rate=args.learning_rate,
        epochs=args.epochs, cf_bid_utility_scale=args.utility_scale,
        cf_bid_horizon=args.cf_bid_horizon,
        entropy_coef=0.0, policy_anchor_coef=0.0)
    env = DetectEnv(env_cfg)
    policy = DetectMAPPOPolicy(env, DetectQPShield(env_cfg),
                               config=policy_cfg, seed=args.training_seed)
    policy.load(str(checkpoint))
    # Loading supplies weights only; explicitly retain this audit's settings.
    policy.config = policy_cfg
    repair = repair_dead_candidate_head_for_training(policy.model,
                                                     seed=args.repair_seed)
    utility_definition = {name: getattr(policy_cfg, name) for name in (
        "gamma", "reward_scale", "shaping_kappa", "shaping_age_cap",
        "geo_shaping_kappa", "stable_shaping_kappa")}
    utility_definition.update({
        "cf_bid_horizon": args.cf_bid_horizon,
        "continuation": (
            "frozen_deterministic_rl_actor_and_independent_branch_shield"
            if args.cf_bid_horizon > 1 else "none_after_candidate_step"),
    })
    if args.event_bank:
        bank_path = (ROOT / args.event_bank).resolve()
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        bank_input = {"path": str(bank_path), "sha256": file_sha256(bank_path)}
    else:
        train_policy_cfg = policy_cfg
        policy.config = train_policy_cfg
        train, train_episodes = collect_cases(policy, train_seeds, args.k,
                                              args.max_events_per_split,
                                              args.max_events_per_episode,
                                              args.min_event_gap,
                                              collection_rng_seeds["train"])
        policy.config = replace(train_policy_cfg, conflict_exposure_prob=0.0)
        try:
            heldout, heldout_episodes = collect_cases(
                policy, heldout_seeds, args.k, args.max_events_per_split,
                args.max_events_per_episode, args.min_event_gap,
                collection_rng_seeds["heldout"])
        finally:
            policy.config = train_policy_cfg
        bank = {"schema_version": 3,
                "source_checkpoint_sha256": checkpoint_hash,
                "k": args.k, "environment_config": _json_ready(asdict(env_cfg)),
                "scenario_profile": args.scenario_profile,
                "radar_hit_model": args.radar_hit_model,
                "train_exposure_probability": args.conflict_exposure,
                "heldout_exposure_probability": 0.0,
                "collection_rng_seeds": collection_rng_seeds,
                "runtime": runtime,
                "collection_source_manifest_sha256": snapshot["manifest_sha256"],
                "collection_source_snapshot": snapshot,
                "teacher_used": False,
                "rule_inventory_modified": False,
                "red_model_modified": False,
                "utility_definition": utility_definition,
                "selection_policy": selection_policy,
                "train_seeds": train_seeds, "heldout_seeds": heldout_seeds,
                "train": train, "heldout": heldout,
                "collection_episodes": {"train": train_episodes,
                                         "heldout": heldout_episodes}}
        bank_input = None
    _validate_event_bank(
        bank, checkpoint_hash=checkpoint_hash, args=args, env_cfg=env_cfg,
        utility_definition=utility_definition, train_seeds=train_seeds,
        heldout_seeds=heldout_seeds,
        collection_rng_seeds=collection_rng_seeds,
        selection_policy=selection_policy, runtime=runtime,
        source_manifest_sha256=snapshot["manifest_sha256"])
    train, heldout = bank["train"], bank["heldout"]
    bank_path = output / "event_bank.json"
    write_json_atomic(bank_path, bank)
    informative_train = [case for case in train
                         if np.ptp(_event_values(case["event"])[2]) > 1e-8]
    coverage = {split: case_coverage(cases)
                for split, cases in (("train", train), ("heldout", heldout))}
    insufficient_sources = {}
    if args.source_balanced and 0 < args.conflict_exposure < 1:
        for source in ("natural", "exposed"):
            count = coverage["train"]["informative_episodes_by_source"].get(source, 0)
            if count < args.min_train_source_event_episodes:
                insufficient_sources[source] = count
    insufficient_coverage = (
        coverage["train"]["informative_episodes"] < args.min_train_event_episodes
        or coverage["heldout"]["informative_episodes"] < args.min_heldout_event_episodes
        or bool(insufficient_sources))
    if insufficient_coverage:
        report = {
            "status": "insufficient_independent_event_coverage", "updates": [],
            "coverage": coverage, "source_checkpoint_sha256": checkpoint_hash,
            "insufficient_train_source_episodes": insufficient_sources,
            "scenario_profile": args.scenario_profile,
            "radar_hit_model": args.radar_hit_model,
            "collection_rng_seeds": collection_rng_seeds,
            "collection_source_manifest_sha256": snapshot["manifest_sha256"],
            "runtime": runtime,
            "teacher_used": False, "rule_inventory_modified": False,
            "red_model_modified": False,
            "source_snapshot": snapshot,
            "source_changed_during_run": changed_source_files(snapshot),
            "event_bank_sha256": file_sha256(bank_path), "event_bank_input": bank_input,
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        }
        report["report_content_sha256"] = canonical_sha256(_json_ready(report))
        write_json_atomic(output / "learning_audit.json", report)
        print(json.dumps({"status": report["status"], "coverage": coverage}), flush=True)
        return 2
    before = {split: evaluate_cases(policy.model, cases, args.mc_samples, 9017)
              for split, cases in (("train", train), ("heldout", heldout))}
    policy.save(str(output / "initial_repaired.pt"))
    torch.manual_seed(args.training_seed)
    sampler = random.Random(args.training_seed)
    updates = []
    for iteration in range(args.updates):
        # Cases remain fixed; each PPO call receives fresh actions from the
        # current actor, never stale demonstrations or utility-ranked bids.
        selected = sample_training_cases(informative_train, args.events_per_update,
                                         sampler, args.episode_balanced,
                                         args.source_balanced)
        rollout = [resample_case_row(policy, case) for case in selected]
        stats = policy.update(rollout)
        updates.append({"iteration": iteration, **stats})
        print(json.dumps({"update": iteration, "bid_grad_norm":
                          stats.get("bid_gradient_norm"), "policy_loss":
                          stats.get("policy_loss")}), flush=True)
    after = {split: evaluate_cases(policy.model, cases, args.mc_samples, 9017)
             for split, cases in (("train", train), ("heldout", heldout))}
    policy.save(str(output / "final.pt"))
    changed = changed_source_files(snapshot)
    report = {
        "status": "completed_diagnostic", "coverage": coverage,
        "diagnostic_scope": "conditional_frozen_continuation_arbitration_not_task_success",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_snapshot": snapshot, "source_changed_during_run": changed,
        "event_bank_sha256": file_sha256(bank_path),
        "event_bank_input": bank_input, "repair": repair,
        "environment_config": asdict(env_cfg), "policy_config": asdict(policy_cfg),
        "scenario_profile": args.scenario_profile,
        "radar_hit_model": args.radar_hit_model,
        "collection_rng_seeds": collection_rng_seeds,
        "collection_source_manifest_sha256": snapshot["manifest_sha256"],
        "runtime": runtime,
        "teacher_used": False, "rule_inventory_modified": False,
        "red_model_modified": False,
        "training_event_sources": coverage["train"]["events_by_source"],
        "training_informative_episodes_by_source": (
            coverage["train"]["informative_episodes_by_source"]),
        "heldout_event_sources": coverage["heldout"]["events_by_source"],
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "before": before, "after": after, "updates": updates,
        "final_checkpoint_sha256": file_sha256(output / "final.pt"),
        "fixed_updates_no_heldout_selection": True,
        "environment_utility_source": "collector_same_snapshot_forced_candidate_winner",
    }
    report["report_content_sha256"] = canonical_sha256(_json_ready(report))
    write_json_atomic(output / "learning_audit.json", report)
    print(json.dumps({"output": str(output), "before_heldout":
                      before["heldout"]["informative"], "after_heldout":
                      after["heldout"]["informative"],
                      "source_changed_during_run": changed}), flush=True)
    if changed:
        raise RuntimeError("Source changed during audit; inspect provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
