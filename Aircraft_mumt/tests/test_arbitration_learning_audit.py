"""Small artificial fixtures test the audit math, never task performance."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _audit_arbitration_learning import (
    _event_values, collect_cases, evaluate_cases, resample_case_row)
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
