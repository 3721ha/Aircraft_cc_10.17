"""The optional task-policy anchor uses the deployed local action law."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
from aircraft_sim.detect_shield import DetectQPShield


def _policy(seed=170, **kwargs):
    cfg = replace(DetectSimConfig(), horizon=6)
    env = DetectEnv(cfg)
    env.reset({"seed": seed, "k_red": 4})
    policy = DetectMAPPOPolicy(
        env, DetectQPShield(cfg),
        config=DetectMAPPOConfig(hidden_size=16, epochs=1, **kwargs),
        seed=seed)
    return policy


def _obs(policy, count=3):
    obs = torch.zeros((count, policy.encoder.dimension))
    # Two fresh tentative populated slots, the remaining six invalid.
    for slot in (0, 1):
        obs[:, 17 + slot * 14 + 8] = 1.0
        obs[:, 17 + slot * 14 + 11] = 1.0
    return obs


def _state(module):
    return {key: value.detach().clone()
            for key, value in module.state_dict().items()}


def _assert_state(expected, module):
    assert expected.keys() == module.state_dict().keys()
    for key, value in module.state_dict().items():
        assert torch.equal(expected[key], value), key


@pytest.mark.parametrize("coefficient", [-1.0, float("inf"), float("nan")])
def test_invalid_anchor_coefficient_is_rejected(coefficient):
    with pytest.raises(ValueError, match="policy_anchor_coef"):
        _policy(policy_anchor_coef=coefficient)


def test_initial_kl_zero_reference_frozen_and_refresh_independent_of_xi():
    policy = _policy(policy_anchor_coef=1.0)
    obs = _obs(policy)
    assert policy._policy_anchor_kl(obs).item() == pytest.approx(0.0, abs=1e-7)
    assert not policy.policy_anchor_model.training
    assert all(not parameter.requires_grad
               for parameter in policy.policy_anchor_model.parameters())
    xi_state = _state(policy.xi_anchor_model)
    with torch.no_grad():
        policy.model.cont_head.bias.add_(0.3)
    assert policy._policy_anchor_kl(obs).item() > 0.0
    policy.refresh_policy_anchor()
    assert policy._policy_anchor_kl(obs).item() == pytest.approx(0.0, abs=1e-7)
    _assert_state(xi_state, policy.xi_anchor_model)


@pytest.mark.parametrize("head,index", [
    ("cont_head", 0), ("cont_head", 1), ("cont_head", 2),
    ("role_head", 1), ("gear_head", 0), ("gear_head", 4),
    ("pointer_head", 1), ("log_std", 0), ("log_std", 1),
    ("log_std", 2),
])
def test_each_task_head_drift_has_positive_kl_and_restoring_gradient(head, index):
    policy = _policy(policy_anchor_coef=1.0)
    parameter = (policy.model.log_std if head == "log_std"
                 else getattr(policy.model, head).bias)
    with torch.no_grad():
        parameter[index].add_(0.2)
    loss = policy._policy_anchor_kl(_obs(policy))
    assert loss.item() > 0.0
    loss.backward()
    # A descent step must oppose the introduced positive displacement.
    assert parameter.grad[index].item() > 0.0
    assert all(parameter.grad is None
               for parameter in policy.policy_anchor_model.parameters())


@pytest.mark.parametrize("role,head,index", [
    (0, "gear_head", 0), (1, "gear_head", 4),
    (0, "pointer_head", 1), (1, "pointer_head", 8),
])
def test_masked_logits_have_no_anchor_cost_or_gradient(role, head, index):
    policy = _policy(policy_anchor_coef=1.0)
    with torch.no_grad():
        policy.model.role_head.weight.zero_()
        policy.model.role_head.bias.fill_(-1e9)
        policy.model.role_head.bias[role] = 0.0
    policy.refresh_policy_anchor()
    parameter = getattr(policy.model, head).bias
    with torch.no_grad():
        parameter[index].add_(0.8)
    loss = policy._policy_anchor_kl(_obs(policy))
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert parameter.grad[index].item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("reference_search,current_search", [(0.8, 0.3), (0.2, 0.7)])
def test_conditional_kl_equals_enumerated_joint_with_reference_role_weights(
        reference_search, current_search):
    policy = _policy(policy_anchor_coef=1.0)
    with torch.no_grad():
        for parameter in policy.model.parameters():
            parameter.zero_()
        policy.model.role_head.bias.copy_(torch.log(torch.tensor(
            [reference_search, 1.0 - reference_search])))
    policy.refresh_policy_anchor()
    with torch.no_grad():
        policy.model.role_head.bias.copy_(torch.log(torch.tensor(
            [current_search, 1.0 - current_search])))
        policy.model.gear_head.bias[0] = 0.7
        policy.model.gear_head.bias[4] = -0.4
        policy.model.pointer_head.bias[1] = 0.6
    expected = torch.tensor(0.0)
    # Independently enumerate the joint legal support, including both roles.
    for role in (0, 1):
        legal_gears = [4, 5] if role == 0 else [0, 1, 2, 3]
        legal_pointers = [0] if role == 0 else [0, 1, 2]
        ref_role = reference_search if role == 0 else 1 - reference_search
        cur_role = current_search if role == 0 else 1 - current_search
        cur_gear = policy.model.gear_head.bias[legal_gears].softmax(-1)
        cur_ptr = policy.model.pointer_head.bias[legal_pointers].softmax(-1)
        for gear_index in range(len(legal_gears)):
            for pointer_index in range(len(legal_pointers)):
                reference_prob = ref_role / len(legal_gears) / len(legal_pointers)
                current_prob = cur_role * cur_gear[gear_index] * cur_ptr[pointer_index]
                expected = expected + reference_prob * (
                    torch.log(torch.tensor(reference_prob)) - current_prob.log())
    assert policy._policy_anchor_kl(_obs(policy)).item() == pytest.approx(
        expected.item(), abs=2e-7)


def test_bid_and_critic_have_no_direct_anchor_gradient():
    policy = _policy(policy_anchor_coef=1.0)
    with torch.no_grad():
        policy.model.cont_head.bias[0].add_(0.2)
        policy.model.bid_head.bias.add_(2.0)
        policy.model.bid_log_std.add_(0.3)
    loss = policy._policy_anchor_kl(_obs(policy))
    loss.backward()
    for name, parameter in policy.model.named_parameters():
        if (name.startswith(("bid_head.", "bid_candidate_head.",
                             "bid_conflict_residual.", "critic."))
                or name == "bid_log_std"):
            assert parameter.grad is None, name
    assert policy.model.bid_conflict_residual.weight.grad is None


def test_old_checkpoint_defaults_reference_to_loaded_model(tmp_path):
    source = _policy(seed=171)
    with torch.no_grad():
        source.model.cont_head.bias.add_(0.4)
    path = tmp_path / "legacy.pt"
    torch.save({"model": source.model.state_dict()}, path)
    target = _policy(seed=172, policy_anchor_coef=0.5)
    target.load(str(path))
    _assert_state(_state(source.model), target.model)
    _assert_state(_state(source.model), target.policy_anchor_model)
    assert target._policy_anchor_kl(_obs(target)).item() == pytest.approx(0.0, abs=1e-7)
    assert target.config.policy_anchor_coef == 0.5


def test_saved_reference_preserves_original_start_across_load(tmp_path):
    source = _policy(policy_anchor_coef=0.5)
    reference = _state(source.policy_anchor_model)
    with torch.no_grad():
        source.model.role_head.bias[0].add_(0.5)
    expected_kl = source._policy_anchor_kl(_obs(source)).item()
    path = tmp_path / "refined.pt"
    source.save(str(path))
    target = _policy(seed=172, policy_anchor_coef=0.25)
    target.load(str(path))
    _assert_state(reference, target.policy_anchor_model)
    _assert_state(_state(source.model), target.model)
    assert target._policy_anchor_kl(_obs(target)).item() == pytest.approx(expected_kl)
    assert not target.policy_anchor_model.training
    assert all(not parameter.requires_grad
               for parameter in target.policy_anchor_model.parameters())


def test_disabled_anchor_skips_all_reference_forward_and_reports_zero():
    policy = _policy(policy_anchor_coef=0.0)
    rollout = policy.collect(k_red=4, seed=173)
    with patch.object(policy.policy_anchor_model, "action_parameters",
                      side_effect=AssertionError("disabled reference forward")):
        stats = policy.update(rollout)
    assert stats["policy_anchor_kl"] == 0.0
    assert stats["post_update_policy_anchor_kl"] == 0.0


def test_rollout_records_environment_and_shaping_rewards_without_changing_total():
    policy = _policy()
    recorded_rewards = []
    original_step = policy.env.step

    def record_step(actions):
        result = original_step(actions)
        recorded_rewards.append(float(result.reward))
        return result

    with patch.object(policy.env, "step", side_effect=record_step):
        rollout = policy.collect(k_red=4, seed=176)
    assert [row["environment_reward"] for row in rollout] == recorded_rewards
    for row in rollout:
        assert row["reward"] == pytest.approx(
            row["environment_reward"] * policy.config.reward_scale
            + row["shaping_reward"])


def test_enabled_update_reports_cumulative_drift_without_moving_reference():
    policy = _policy(policy_anchor_coef=0.5)
    rollout = policy.collect(k_red=4, seed=174)
    reference = _state(policy.policy_anchor_model)
    # Make current != reference before the update so loss monitoring is tested.
    with torch.no_grad():
        policy.model.cont_head.bias[0].add_(0.1)
    stats = policy.update(rollout)
    _assert_state(reference, policy.policy_anchor_model)
    assert stats["policy_anchor_kl"] > 0.0
    assert stats["post_update_policy_anchor_kl"] > 0.0
    obs = torch.as_tensor([item["obs"][agent]
                           for item in rollout
                           for agent in range(policy.env.cfg.N_WINGMEN)])
    assert stats["post_update_policy_anchor_kl"] == pytest.approx(
        policy._policy_anchor_kl(obs).item())


def test_monitor_only_reports_real_drift_without_enabling_regularization():
    control = _policy(policy_anchor_coef=0.0, policy_anchor_monitor=False)
    monitored = _policy(policy_anchor_coef=0.0, policy_anchor_monitor=True)
    rollout = control.collect(k_red=4, seed=178)
    reference = _state(monitored.policy_anchor_model)
    with torch.no_grad():
        control.model.cont_head.bias[0].add_(0.2)
        monitored.model.cont_head.bias[0].add_(0.2)
    control_stats = control.update(rollout)
    monitored_stats = monitored.update(rollout)
    _assert_state(_state(control.model), monitored.model)
    _assert_state(reference, monitored.policy_anchor_model)
    assert control_stats["policy_anchor_kl"] == 0.0
    assert monitored_stats["policy_anchor_kl"] == 0.0
    assert control_stats["post_update_policy_anchor_kl"] == 0.0
    assert monitored_stats["post_update_policy_anchor_kl"] > 0.0
    assert control_stats["loss"] == monitored_stats["loss"]
    assert all(parameter.grad is None
               for parameter in monitored.policy_anchor_model.parameters())


def test_anchor_term_in_actual_update_restores_policy_when_other_credit_is_zero():
    common = dict(value_coef=0.0, entropy_coef=0.0, intervention_coef=0.0,
                  xi_anchor_coef=0.0, role_adv_coef=0.0, cf_adv_coef=0.0,
                  cf_role_coef=0.0, cf_bid_coef=0.0)
    control = _policy(policy_anchor_coef=0.0, **common)
    anchored = _policy(policy_anchor_coef=1.0, **common)
    rollout = anchored.collect(k_red=4, seed=177)
    for row in rollout:
        row["reward"] = 0.0
        row["value"] = 0.0
    for policy in (control, anchored):
        with torch.no_grad():
            policy.model.cont_head.bias[0].add_(0.2)
    obs = torch.as_tensor([item["obs"][agent]
                           for item in rollout
                           for agent in range(anchored.env.cfg.N_WINGMEN)])
    control_before = _state(control.model)
    initial_kl = anchored._policy_anchor_kl(obs).item()
    control.update(rollout)
    anchored.update(rollout)
    _assert_state(control_before, control.model)
    assert anchored._policy_anchor_kl(obs).item() < initial_kl


def test_bid_only_update_does_not_optimize_task_anchor():
    policy = _policy(policy_anchor_coef=10.0, bid_only_ppo=True)
    rollout = policy.collect(k_red=4, seed=175)
    with torch.no_grad():
        policy.model.cont_head.bias[0].add_(0.2)
    before = _state(policy.model)
    stats = policy.update(rollout)
    assert stats["policy_anchor_kl"] > 0.0
    for key, value in policy.model.state_dict().items():
        if (key.startswith(("bid_head.", "bid_candidate_head.",
                            "bid_conflict_residual."))
                or key == "bid_log_std"):
            continue
        assert torch.equal(before[key], value), key
