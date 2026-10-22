"""Small artificial fixtures test the audit math, never task performance."""
from copy import deepcopy
from pathlib import Path
import sys
import random

import numpy as np
import pytest
import torch
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _audit_arbitration_learning import (
    _case_exposure_source, _environment_config, _event_exposure_source, _event_values,
    _parse_seed_spec, _split_collection_seeds, _validate_event_bank, case_coverage,
    build_parser, collect_cases, evaluate_cases, resample_case_row,
    sample_training_cases, main)
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
from aircraft_sim.detect_shield import DetectQPShield


def _policy_and_case(utilities=(1.0, 2.0, 3.0)):
    env = DetectEnv()
    policy = DetectMAPPOPolicy(
        env, DetectQPShield(env.cfg), DetectMAPPOConfig(
            bid_only_ppo=True, direct_rl_arbitration=True, entropy_coef=0.0),
        seed=111)
    observations = env.reset({"seed": 11, "k_red": 2})
    actions, logps = policy.act_batch(observations, deterministic=True)
    obs = {str(agent): policy.encoder.encode(observation).tolist()
           for agent, observation in observations.items()}
    n = env.cfg.N_WINGMEN
    row = {
        "obs": obs, "team": np.concatenate(list(obs.values())).tolist(),
        "nominal": {str(i): policy.encode_action(actions[i]) for i in range(n)},
        "safe": {str(i): policy.encode_action(actions[i]) for i in range(n)},
        "old_logp": {str(i): logps[i] for i in range(n)},
        "bid_actions": {str(i): list(actions[i].arbitration_bids) for i in range(n)},
        "reward": 0.0, "value": 0.0, "done": True, "f_role": 0.0,
        "gear_gate_credit": [0.0] * n, "cf_deltas": [0.0] * n,
        "bid_rank": [0.0] * n, "bid_mask": [False] * n,
        "action_exposure_mask": [True] * n,
    }
    event = {"tid": 99, "candidates": [2, 0, 1],
             "bid_slots": {"0": 4, "1": 2, "2": 6},
             "utilities": {str(i): utility for i, utility in enumerate(utilities)},
             "winner": 0}
    return policy, {"episode_seed": 11, "step_index": 7,
                    "event": event, "template": row}


def test_event_table_validation_rejects_missing_or_nonfinite_utilities():
    _, case = _policy_and_case()
    event = deepcopy(case["event"])
    del event["utilities"]["1"]
    with pytest.raises(ValueError, match="mismatch"):
        _event_values(event)
    event = deepcopy(case["event"])
    event["utilities"]["1"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        _event_values(event)
    event = deepcopy(case["event"])
    event["candidates"] = [-1, 1]
    with pytest.raises(ValueError, match="agent range"):
        _event_values(event)


def test_exposure_source_uses_only_the_conflict_candidates():
    _, case = _policy_and_case()
    case["event"]["candidates"] = [0, 1]
    row = {"action_exposure_mask": [False, False, True]}
    assert _event_exposure_source(row, case["event"]) == "natural"
    row["action_exposure_mask"] = [False, True, False]
    assert _event_exposure_source(row, case["event"]) == "exposed"
    case["event"]["candidates"] = [-1, 1]
    with pytest.raises(ValueError, match="exposure mask"):
        _event_exposure_source(row, case["event"])


def test_resampling_credits_actual_event_winner_and_preserves_task_actions(monkeypatch):
    policy, case = _policy_and_case()
    original = deepcopy(case)

    def sample(_distribution, sample_shape=torch.Size()):
        values = torch.zeros_like(_distribution.loc)
        values[0, 4] = 0.9
        values[1, 2] = -0.1
        values[2, 6] = 0.1
        return values

    monkeypatch.setattr(Normal, "sample", sample)
    row = resample_case_row(policy, case)
    assert row["bid_cf_event_records"][0]["winner"] == 0
    assert row["bid_cf_event_records"][0]["event_advantage"] == -1.0
    for agent, slot in ((0, 4), (1, 2), (2, 6)):
        assert row["bid_cf_adv_slots"][agent][slot] == -1.0
        assert sum(row["bid_cf_mask_slots"][agent]) == 1
        assert row["bid_event_ids_slots"][agent][slot] == 0
        assert row["nominal"][agent][:6] == original["template"]["nominal"][str(agent)][:6]
        assert row["safe"][agent][:6] == original["template"]["safe"][str(agent)][:6]
    assert original == case


def test_middle_utility_winner_keeps_zero_advantage_in_ppo_event_population(monkeypatch):
    policy, case = _policy_and_case()

    def sample(_distribution, sample_shape=torch.Size()):
        values = torch.zeros_like(_distribution.loc)
        values[1, 2] = 0.5
        return values

    monkeypatch.setattr(Normal, "sample", sample)
    row = resample_case_row(policy, case)
    assert row["bid_cf_event_records"][0]["winner"] == 1
    assert row["bid_cf_event_records"][0]["event_advantage"] == 0.0
    assert sum(sum(masks) for masks in row["bid_cf_mask_slots"]) == 3
    stats = policy.update([row, deepcopy(row)])
    assert stats["bid_active_slot_count"] == 6
    assert np.isfinite(stats["loss"])
    assert stats["bid_gradient_norm"] == 0.0


def test_fixed_evaluation_uses_correct_slots_and_deployment_tie_break():
    policy, case = _policy_and_case()

    class FixedModel:
        bid_log_std = torch.tensor(-20.0)

        def action_parameters(self, obs):
            loc = torch.zeros((obs.shape[0], 8))
            loc[2, 6] = 1.0
            return (None, None, None, None, loc)

    result = evaluate_cases(FixedModel(), [case], 64, 131)
    assert result["expected_regret"] == 0.0
    assert result["expected_optimal_winner_rate"] == 1.0
    assert result["rows"][0]["deterministic_winner"] == 2
    assert result == evaluate_cases(FixedModel(), [case], 64, 131)

    class TiedModel(FixedModel):
        def action_parameters(self, obs):
            return (None, None, None, None, torch.zeros((obs.shape[0], 8)))

    assert evaluate_cases(TiedModel(), [case], 64, 131)["rows"][0]["deterministic_winner"] == 0


def test_collection_requires_real_collector_utility_records():
    class NoUtilities:
        def collect(self, **kwargs):
            return [{"obs": {}}]

    with pytest.raises(RuntimeError, match="utility records"):
        collect_cases(NoUtilities(), [100000], 4, 40)


def test_temporal_selection_caps_each_episode_without_ranking_utilities():
    policy, case = _policy_and_case()
    row = deepcopy(case["template"])
    row["bid_cf_event_records"] = [deepcopy(case["event"])]
    policy.collect = lambda **kwargs: [deepcopy(row) for _ in range(12)]
    cases, episodes = collect_cases(policy, [10, 20], 4, 20,
                                    max_per_episode=2, min_event_gap=4)
    assert [(item["episode_seed"], item["step_index"]) for item in cases] == [
        (10, 0), (10, 4), (20, 0), (20, 4)]
    assert all(item["controllable_events"] == 12 for item in episodes)
    assert all(item["retained_events"] == 2 for item in episodes)


def test_collection_uses_reproducible_split_rng_and_restores_precheck():
    policy, case = _policy_and_case()
    base_row = deepcopy(case["template"])
    base_row["action_exposure_mask"] = [False, False, False]

    def collect(**kwargs):
        assert policy._may_have_arbitration_conflict({}) is True
        row = deepcopy(base_row)
        row["obs"]["0"] = [float(torch.rand(()))]
        row["bid_cf_event_records"] = [deepcopy(case["event"])]
        return [row]

    policy.collect = collect
    torch.manual_seed(99)
    expected_next = torch.rand(1)
    torch.manual_seed(99)
    first, first_episodes = collect_cases(
        policy, [12], 2, 10, collection_seed=321)
    actual_next = torch.rand(1)
    second, second_episodes = collect_cases(
        policy, [12], 2, 10, collection_seed=321)
    other, _ = collect_cases(policy, [12], 2, 10, collection_seed=322)

    assert first[0]["exposure_source"] == "natural"
    assert first[0]["template"]["obs"]["0"] == second[0]["template"]["obs"]["0"]
    assert first[0]["template"]["obs"]["0"] != other[0]["template"]["obs"]["0"]
    assert first_episodes[0]["actor_rng_seed"] == second_episodes[0]["actor_rng_seed"]
    assert torch.equal(expected_next, actual_next)
    assert policy._may_have_arbitration_conflict({}) is False


def test_environment_profile_and_split_collection_seeds_are_explicit():
    from types import SimpleNamespace

    args = SimpleNamespace(scenario_profile="k2_lane_center_50", k=2,
                           radar_hit_model="bernoulli")
    config = _environment_config(args)
    assert config.K_RED == 2
    assert config.RED_LANE_K2_CENTER_KM == 50.0
    assert config.STABLE_WINDOW == 8
    assert config.AGE_STD_TRACK == 10.0
    assert config.RADAR_HIT_MODEL == "bernoulli"
    assert _split_collection_seeds(123) == {"train": 123, "heldout": 124}
    assert _parse_seed_spec("10:12, 20") == [10, 11, 12, 20]
    with pytest.raises(ValueError, match="Descending"):
        _parse_seed_spec("12:10")
    args.k = 3
    with pytest.raises(ValueError, match="require --k 2"):
        _environment_config(args)


def test_training_defaults_match_the_locked_data_and_update_protocol():
    args = build_parser().parse_args(["--checkpoint", "input.pt", "--output", "out"])
    assert args.train_seeds == "910000:910199"
    assert args.heldout_seeds == "920000:920099"
    assert args.cf_bid_horizon == 4
    assert args.updates == 48
    assert args.learning_rate == pytest.approx(1e-5)
    assert args.source_balanced is True
    assert args.episode_balanced is True
    assert args.min_train_event_episodes == 20
    assert args.min_train_source_event_episodes == 10
    assert args.min_heldout_event_episodes == 10


def test_source_balancing_weights_sources_and_episodes_equally():
    cases = ([{"episode_seed": 1, "exposure_source": "natural"}] * 1000
             + [{"episode_seed": 2, "exposure_source": "natural"}]
             + [{"episode_seed": 3, "exposure_source": "exposed"}] * 1000
             + [{"episode_seed": 4, "exposure_source": "exposed"}])
    selected = sample_training_cases(cases, 4000, random.Random(38),
                                     source_balanced=True)
    source_fraction = sum(case["exposure_source"] == "natural"
                          for case in selected) / len(selected)
    natural_fraction = sum(case["episode_seed"] == 2 for case in selected) / sum(
        case["exposure_source"] == "natural" for case in selected)
    exposed_fraction = sum(case["episode_seed"] == 4 for case in selected) / sum(
        case["exposure_source"] == "exposed" for case in selected)
    assert 0.47 < source_fraction < 0.53
    assert 0.47 < natural_fraction < 0.53
    assert 0.47 < exposed_fraction < 0.53


def test_episode_sampling_prevents_one_long_trajectory_dominating_updates():
    cases = ([{"episode_seed": 1}] * 1000 + [{"episode_seed": 2}])
    selected = sample_training_cases(cases, 4000, random.Random(12), True)
    fraction = sum(case["episode_seed"] == 2 for case in selected) / len(selected)
    assert 0.47 < fraction < 0.53
    legacy = sample_training_cases(cases, 4000, random.Random(12), False)
    assert sum(case["episode_seed"] == 2 for case in legacy) < 20


def test_coverage_counts_independent_informative_episodes():
    _, case = _policy_and_case()
    constant = deepcopy(case)
    constant["episode_seed"] = 12
    constant["event"]["utilities"] = {str(i): 1.0 for i in range(3)}
    coverage = case_coverage([deepcopy(case) for _ in range(100)] + [constant])
    assert coverage["events"] == 101
    assert coverage["event_episodes"] == 2
    assert coverage["informative_episodes"] == 1
    assert coverage["informative_events"] == 100
    assert coverage["informative_episodes_by_source"] == {"natural": 1}


def test_evaluation_reports_episode_weighted_regret_for_correlated_events():
    _, easy = _policy_and_case()
    hard = deepcopy(easy)
    hard["episode_seed"] = 12
    hard["event"]["utilities"] = {"0": 3.0, "1": 2.0, "2": 1.0}

    class FixedModel:
        bid_log_std = torch.tensor(-20.0)

        def action_parameters(self, obs):
            loc = torch.zeros((obs.shape[0], 8))
            loc[2, 6] = 1.0
            return (None, None, None, None, loc)

    result = evaluate_cases(FixedModel(), [deepcopy(easy)] * 4 + [hard], 64, 13)
    assert result["informative"]["expected_regret"] == pytest.approx(0.4)
    assert result["informative_episode_balanced"]["expected_regret"] == pytest.approx(1.0)
    assert len(result["informative_by_episode"]) == 2


def test_insufficient_coverage_writes_failure_report_without_updating(tmp_path, monkeypatch):
    import json
    import _audit_arbitration_learning as audit
    policy, _ = _policy_and_case()
    checkpoint = tmp_path / "policy.pt"
    policy.save(str(checkpoint))
    output = tmp_path / "insufficient"
    observed_exposures = []

    def collect(policy, *args, **kwargs):
        observed_exposures.append(policy.config.conflict_exposure_prob)
        return [], []

    monkeypatch.setattr(audit, "collect_cases", collect)
    assert main(["--checkpoint", str(checkpoint), "--output", str(output),
                 "--train-seeds", "1", "--heldout-seeds", "2",
                 "--min-train-event-episodes", "1",
                 "--min-heldout-event-episodes", "1"]) == 2
    report = json.loads((output / "learning_audit.json").read_text())
    assert report["status"] == "insufficient_independent_event_coverage"
    assert report["updates"] == []
    assert report["source_changed_during_run"] == []
    assert observed_exposures == [0.5, 0.0]
    assert report["teacher_used"] is False
    assert not (output / "final.pt").exists()


def test_preflight_does_not_create_output_or_collect(tmp_path, monkeypatch):
    import _audit_arbitration_learning as audit
    policy, _ = _policy_and_case()
    checkpoint = tmp_path / "policy.pt"
    policy.save(str(checkpoint))
    output = tmp_path / "preflight"
    monkeypatch.setattr(audit, "collect_cases", lambda *args: pytest.fail("collected"))
    assert main(["--checkpoint", str(checkpoint), "--output", str(output),
                 "--train-seeds", "1", "--heldout-seeds", "2",
                 "--min-train-event-episodes", "1",
                 "--min-heldout-event-episodes", "1",
                 "--preflight-only"]) == 0
    assert not output.exists()


def test_frozen_bank_rejects_exposed_or_mismatched_heldout_events():
    from types import SimpleNamespace
    from dataclasses import asdict
    from aircraft_sim.detect_config import detect_config_profile

    args = SimpleNamespace(k=2, scenario_profile="k2_lane_center_50",
                           radar_hit_model="bernoulli", conflict_exposure=0.5)
    env_cfg = detect_config_profile("k2_lane_center_50", K_RED=2,
                                    STABLE_WINDOW=8, AGE_STD_TRACK=10.0,
                                    RADAR_HIT_MODEL="bernoulli", C04_RANGE=80.0,
                                    SECTOR_AXIS_MAX_DEG=60.0,
                                    ARB_BID_WEIGHT=0.35, ARB_HYSTERESIS=0.2)
    utility = {"cf_bid_horizon": 4}
    seeds = {"train": [10], "heldout": [20]}
    selection = {"max_events_per_episode": 2, "min_event_gap": 1}
    bank = {"schema_version": 3, "source_checkpoint_sha256": "sha",
            "k": 2, "environment_config": _json_config(asdict(env_cfg)),
            "utility_definition": utility,
            "scenario_profile": args.scenario_profile,
            "radar_hit_model": args.radar_hit_model,
            "train_exposure_probability": 0.5,
            "heldout_exposure_probability": 0.0,
            "collection_rng_seeds": {"train": 31, "heldout": 32},
            "runtime": {"python": "3", "platform": "test", "torch": "2",
                        "numpy": "2", "torch_num_threads": 1},
            "train_seeds": [10], "heldout_seeds": [20],
            "selection_policy": selection,
            "collection_source_manifest_sha256": "manifest",
            "teacher_used": False, "rule_inventory_modified": False,
            "red_model_modified": False,
            "train": [], "heldout": [{
                "episode_seed": 20, "exposure_source": "natural",
                "collection_exposure_mask": [False, False, False],
                "template": {"obs": {"0": [], "1": [], "2": []}},
                "event": {"candidates": [0, 1, 2]}}]}
    kwargs = dict(checkpoint_hash="sha", args=args, env_cfg=env_cfg,
                  utility_definition=utility, train_seeds=seeds["train"],
                  heldout_seeds=seeds["heldout"],
                  collection_rng_seeds={"train": 31, "heldout": 32},
                  selection_policy=selection, runtime=bank["runtime"],
                  source_manifest_sha256="manifest")
    _validate_event_bank(bank, **kwargs)
    bank["heldout"][0]["collection_exposure_mask"] = [True, False, False]
    with pytest.raises(ValueError, match="disagrees with its mask"):
        _case_exposure_source(bank["heldout"][0])
    bank["heldout"][0]["exposure_source"] = "exposed"
    with pytest.raises(ValueError, match="natural conflicts only"):
        _validate_event_bank(bank, **kwargs)
    bank["collection_source_manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="manifest"):
        _validate_event_bank(bank, **kwargs)


def _json_config(value):
    import json
    return json.loads(json.dumps(value))
