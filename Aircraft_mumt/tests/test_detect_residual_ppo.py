"""Task residual refinement preserves the inherited controller and radar axis."""
from copy import deepcopy
from dataclasses import replace
import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
from aircraft_sim.detect_shield import DetectQPShield


def _policy(seed=813, **kwargs):
    cfg = replace(DetectSimConfig(), horizon=6)
    env = DetectEnv(cfg)
    env.reset({"seed": seed, "k_red": 2})
    config = DetectMAPPOConfig(hidden_size=16, epochs=2,
                              task_residual_only_ppo=True,
                              target_kl=1.0, bid_target_kl=1.0,
                              policy_anchor_coef=0.5, **kwargs)
    return DetectMAPPOPolicy(env, DetectQPShield(cfg), config=config, seed=seed)


def _data(policy, *, with_bid_event):
    data = policy.collect(k_red=2, seed=813)
    assert len(data) >= 2
    # This real rollout already activates conflict proxies. Leave observations
    # and the recorded behavior likelihoods intact so task PPO stays on-policy.
    assert any(torch.count_nonzero(torch.as_tensor(encoded[156:169])).item()
               for item in data for encoded in item["obs"].values())
    for t, item in enumerate(data):
        item["reward"] = float((t % 3) - 1)
        item["f_role"] = 0.2 * (1 if t % 2 else -1)
        item["cf_deltas"] = [0.1, -0.1, 0.0]
        item["action_exposure_mask"] = [False] * 3
        item["bid_cf_adv_slots"] = [[0.0] * 8 for _ in range(3)]
        item["bid_cf_mask_slots"] = [[False] * 8 for _ in range(3)]
        item["bid_event_mask_slots"] = [[False] * 8 for _ in range(3)]
        item["bid_event_ids_slots"] = [[-1] * 8 for _ in range(3)]
    if with_bid_event:
        item = data[0]
        for i in (0, 1):
            item["bid_cf_adv_slots"][i][0] = 0.1
            item["bid_cf_mask_slots"][i][0] = True
            item["bid_event_mask_slots"][i][0] = True
            item["bid_event_ids_slots"][i][0] = 0
    return data


def _is_bid(name):
    return (name.startswith(("bid_head.", "bid_candidate_head.",
                             "bid_conflict_residual."))
            or name == "bid_log_std")


def _seed_old_adam_state(policy):
    for parameter in policy.model.parameters():
        policy.optimizer.state[parameter] = {
            "step": torch.tensor(4.0),
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }


def test_residual_and_bid_learn_while_base_and_xi_stay_exact_with_old_adam():
    policy = _policy()
    # Both the inherited task-head momentum and a protected nonzero xi row
    # must survive an update exactly, including the otherwise-clamped std.
    with torch.no_grad():
        policy.model.log_std.fill_(0.1)
        policy.model.conflict_residual.weight[2].fill_(0.125)
    data = _data(policy, with_bid_event=True)
    _seed_old_adam_state(policy)
    before = deepcopy(policy.model.state_dict())
    anchor_before = deepcopy(policy.policy_anchor_model.state_dict())
    stats = policy.update(data)
    after = policy.model.state_dict()
    trainable = {name for name, parameter in policy.model.named_parameters()
                 if parameter.requires_grad}
    expected = {name for name in before
                if _is_bid(name) or name.startswith("critic.")
                or name == "conflict_residual.weight"}
    assert trainable == expected
    for name in before:
        if name not in expected:
            assert torch.equal(before[name], after[name]), name
    assert torch.equal(before["conflict_residual.weight"][2],
                       after["conflict_residual.weight"][2])
    non_xi = torch.tensor([row for row in range(20) if row != 2])
    assert not torch.equal(before["conflict_residual.weight"][non_xi],
                           after["conflict_residual.weight"][non_xi])
    assert any(not torch.equal(before[name], after[name]) for name in before if _is_bid(name))
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("critic."))
    assert stats["bid_gradient_norm"] > 0.0
    assert stats["bid_parameter_delta_l2"] > 0.0
    assert stats["task_residual_gradient_norm"] > 0.0
    assert stats["critic_gradient_norm"] > 0.0
    assert stats["task_residual_parameter_delta_l2"] > 0.0
    assert stats["task_residual_xi_row_unchanged"] is True
    assert all(math.isfinite(value) for value in stats.values() if isinstance(value, float))
    assert all(torch.equal(value, policy.policy_anchor_model.state_dict()[name])
               for name, value in anchor_before.items())


def test_task_residual_and_critic_update_without_any_arbitration_event():
    policy = _policy(seed=814)
    data = _data(policy, with_bid_event=False)
    before = deepcopy(policy.model.state_dict())
    stats = policy.update(data)
    after = policy.model.state_dict()
    assert stats["bid_active_slot_count"] == 0
    assert stats["bid_gradient_norm"] == 0.0
    assert stats["bid_parameter_delta_l2"] == 0.0
    assert not torch.equal(before["conflict_residual.weight"], after["conflict_residual.weight"])
    assert torch.equal(before["conflict_residual.weight"][2], after["conflict_residual.weight"][2])
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("critic."))
    for name in before:
        if name.startswith(("actor.", "cont_head.", "role_head.",
                            "gear_head.", "pointer_head.")) or name == "log_std":
            assert torch.equal(before[name], after[name]), name


def test_residual_mode_clips_sparse_bid_separately_from_task_and_critic():
    policy = _policy(seed=815)
    data = _data(policy, with_bid_event=True)
    expected_bid = {id(parameter) for name, parameter in policy.model.named_parameters() if _is_bid(name)}
    expected_task = {id(policy.model.conflict_residual.weight)}
    expected_critic = {id(parameter) for parameter in policy.model.critic.parameters()}
    recorded = []
    original_clip = nn.utils.clip_grad_norm_
    def record_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        recorded.append(({id(parameter) for parameter in parameters}, max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)
    with patch("aircraft_sim.detect_mappo.nn.utils.clip_grad_norm_", side_effect=record_clip):
        policy.update(data)
    assert len(recorded) >= 3
    assert len(recorded) % 3 == 0
    assert all(recorded[index] == (expected_bid, 0.5) for index in range(0, len(recorded), 3))
    assert all(recorded[index] == (expected_task, 0.5) for index in range(1, len(recorded), 3))
    assert all(recorded[index] == (expected_critic, 0.5) for index in range(2, len(recorded), 3))
    assert not expected_bid & expected_task
    assert not expected_critic & (expected_bid | expected_task)


def test_conflicting_modes_rejected_during_config_construction():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DetectMAPPOConfig(bid_only_ppo=True, task_residual_only_ppo=True)


def test_conflicting_modes_rejected_by_policy_constructor_after_config_mutation():
    cfg = replace(DetectSimConfig(), horizon=2)
    env = DetectEnv(cfg)
    config = DetectMAPPOConfig(task_residual_only_ppo=True)
    config.bid_only_ppo = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        DetectMAPPOPolicy(env, DetectQPShield(cfg), config=config)


def test_conflicting_modes_rejected_after_runtime_config_mutation_even_without_data():
    policy = _policy()
    policy.config.bid_only_ppo = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        policy.update([])
