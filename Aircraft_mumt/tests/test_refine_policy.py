"""Regression coverage for fresh-policy refinement from a selected checkpoint."""
from dataclasses import asdict, replace

import pytest
import torch

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
from aircraft_sim.detect_shield import DetectQPShield


def _policy(seed=0, **config_kwargs):
    # This local override keeps regression runs short; task configuration files
    # and the real environment implementation remain unchanged.
    cfg = replace(DetectSimConfig(), horizon=8)
    env = DetectEnv(cfg)
    env.reset({"seed": seed, "k_red": 2})
    config = DetectMAPPOConfig(hidden_size=32, **config_kwargs)
    policy = DetectMAPPOPolicy(
        env, DetectQPShield(cfg), config=config, seed=seed)
    return env, policy


def _state(module):
    return {key: value.detach().clone()
            for key, value in module.state_dict().items()}


def _assert_state_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        assert torch.equal(value, actual[key]), key


def test_checkpoint_roundtrip_preserves_all_policy_and_xi_anchor_tensors(tmp_path):
    _, source = _policy(seed=120)
    # Make the anchor distinguishable from the policy, including zero-initialized
    # residual/candidate tensors. Loading the policy into both must be caught.
    with torch.no_grad():
        for index, value in enumerate(source.model.state_dict().values(), 1):
            value.add_(index * 0.0001)
        for index, value in enumerate(
                source.xi_anchor_model.state_dict().values(), 1):
            value.sub_(index * 0.0002)
    expected_model = _state(source.model)
    expected_anchor = _state(source.xi_anchor_model)
    checkpoint = tmp_path / "best.pt"
    source.save(str(checkpoint))

    _, restored = _policy(seed=121)
    restored.load(str(checkpoint))

    _assert_state_equal(expected_model, restored.model.state_dict())
    _assert_state_equal(expected_anchor, restored.xi_anchor_model.state_dict())
    assert not restored.xi_anchor_model.training
    assert all(not parameter.requires_grad
               for parameter in restored.xi_anchor_model.parameters())


@pytest.mark.parametrize("bid_only,direct_rl", [(True, True), (False, False)])
def test_load_preserves_fresh_runtime_config_and_optimizer(
        tmp_path, bid_only, direct_rl):
    _, source = _policy(
        seed=122, learning_rate=3e-4, epochs=4,
        bid_only_ppo=not bid_only, direct_rl_arbitration=not direct_rl)
    checkpoint = tmp_path / "best.pt"
    source.save(str(checkpoint))
    _, restored = _policy(
        seed=123, learning_rate=1e-5, epochs=1,
        bid_only_ppo=bid_only, direct_rl_arbitration=direct_rl,
        conflict_exposure_prob=0.25, bid_target_kl=0.005)
    runtime_config = restored.config
    expected_config = asdict(runtime_config)
    optimizer = restored.optimizer
    assert not optimizer.state

    restored.load(str(checkpoint))

    # A refinement checkpoint supplies weights; the launcher owns runtime
    # settings and intentionally creates a fresh optimizer for its new stage.
    assert restored.config is runtime_config
    assert asdict(restored.config) == expected_config
    assert restored.config.bid_only_ppo is bid_only
    assert restored.config.direct_rl_arbitration is direct_rl
    assert restored.optimizer is optimizer
    assert all(group["lr"] == 1e-5 for group in optimizer.param_groups)
    assert not optimizer.state
    actions, _ = restored.act_batch(restored.env.observe(), deterministic=True)
    assert all(action.rl_direct_arbitration is direct_rl
               for action in actions.values())


def test_bid_only_update_preserves_all_non_bid_state_and_action_parameters(
        tmp_path):
    _, source = _policy(seed=124)
    checkpoint = tmp_path / "best.pt"
    source.save(str(checkpoint))
    env, policy = _policy(
        seed=125, learning_rate=1e-5, bid_only_ppo=True,
        direct_rl_arbitration=True, epochs=2)
    policy.load(str(checkpoint))
    assert not policy.optimizer.state
    rollout = policy.collect(k_red=2, seed=125)
    assert 0 < len(rollout) <= env.cfg.horizon
    n_agents = env.cfg.N_WINGMEN
    for step, item in enumerate(rollout):
        item["bid_cf_adv_slots"] = [[0.0] * 8 for _ in range(n_agents)]
        item["bid_cf_mask_slots"] = [[False] * 8 for _ in range(n_agents)]
        item["bid_event_mask_slots"] = [[False] * 8 for _ in range(n_agents)]
        item["bid_event_ids_slots"] = [[-1] * 8 for _ in range(n_agents)]
        for agent, encoded in item["obs"].items():
            # The residual starts at zero, so these proxy activations preserve
            # recorded old-policy likelihoods while exercising its bid row.
            encoded[-13:] = 0.0
            encoded[-13 + ((step + agent) % 13)] = 0.25
            start = agent * policy.encoder.dimension
            item["team"][start + policy.encoder.dimension - 13:
                         start + policy.encoder.dimension] = encoded[-13:]
    # Like the existing sparse-credit tests, inject one pivotal candidate so
    # correctness is independent of naturally occurring short-rollout ties.
    event = rollout[0]
    event["bid_cf_adv_slots"][0][0] = 1.0
    event["bid_cf_mask_slots"][0][0] = True
    for agent in (0, 1):
        event["bid_event_mask_slots"][agent][0] = True
        event["bid_event_ids_slots"][agent][0] = 0
    observations = torch.tensor([
        item["obs"][agent]
        for item in rollout for agent in range(n_agents)])
    before = _state(policy.model)
    anchor_before = _state(policy.xi_anchor_model)
    with torch.no_grad():
        parameters_before = tuple(
            value.clone() for value in policy.model.action_parameters(observations))

    stats = policy.update(rollout)

    after = policy.model.state_dict()
    for key, expected in before.items():
        if key.startswith(("bid_head.", "bid_candidate_head.")) or key == "bid_log_std":
            continue
        if key == "conflict_residual.weight":
            assert torch.equal(expected[:-1], after[key][:-1]), key
        else:
            assert torch.equal(expected, after[key]), key
    _assert_state_equal(anchor_before, policy.xi_anchor_model.state_dict())
    with torch.no_grad():
        parameters_after = policy.model.action_parameters(observations)
    for head, expected, actual in zip(
            ("continuous", "role", "gear", "pointer"),
            parameters_before[:4], parameters_after[:4]):
        assert torch.equal(expected, actual), head
    assert stats["bid_active_slot_count"] == 1
    assert stats["bid_gradient_norm"] > 0.0
    assert stats["bid_parameter_delta_l2"] > 0.0
    assert any(not torch.equal(before[key], after[key]) for key in before
               if key.startswith(("bid_head.", "bid_candidate_head.")))
    assert not torch.equal(parameters_before[-1], parameters_after[-1])
