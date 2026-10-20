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
    _event_values, case_coverage, collect_cases, evaluate_cases,
    resample_case_row, sample_training_cases, main)
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
    monkeypatch.setattr(audit, "collect_cases", lambda *args: ([], []))
    assert main(["--checkpoint", str(checkpoint), "--output", str(output),
                 "--train-seeds", "1", "--heldout-seeds", "2"]) == 2
    report = json.loads((output / "learning_audit.json").read_text())
    assert report["status"] == "insufficient_independent_event_coverage"
    assert report["updates"] == []
    assert report["source_changed_during_run"] == []
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
                 "--preflight-only"]) == 0
    assert not output.exists()
