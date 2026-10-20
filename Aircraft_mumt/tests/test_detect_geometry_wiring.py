"""Deployment, checkpoint, and real PPO checks for the local geometry adapter."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import (
    DetectActorCritic, DetectMAPPOConfig, DetectMAPPOPolicy,
)
from aircraft_sim.detect_shield import DetectQPShield


def _policy(seed=926, *, train_geometry=True):
    cfg = replace(DetectSimConfig(), horizon=6)
    env = DetectEnv(cfg)
    env.reset({"seed": seed, "k_red": 2})
    config = DetectMAPPOConfig(
        hidden_size=16, epochs=2, task_residual_only_ppo=True,
        train_geometry_adapter=train_geometry,
        target_kl=1.0, bid_target_kl=1.0, policy_anchor_coef=0.5,
    )
    return DetectMAPPOPolicy(env, DetectQPShield(cfg), config=config, seed=seed)


def _observations():
    generator = torch.Generator().manual_seed(384)
    obs = torch.randn(7, 169, generator=generator)
    obs[:, -13:] = torch.rand(7, 13, generator=generator)
    obs[:, -1] = 1.0
    return obs


def _activate(adapter):
    with torch.no_grad():
        adapter.output_layer.weight.fill_(0.01)
        adapter.output_layer.bias.copy_(torch.linspace(-0.3, 0.4, 20))


def _model_states(policy):
    return {
        "model": deepcopy(policy.model.state_dict()),
        "xi_anchor_model": deepcopy(policy.xi_anchor_model.state_dict()),
        "policy_anchor_model": deepcopy(policy.policy_anchor_model.state_dict()),
    }


def _task_rollout(policy):
    rollout = policy.collect(k_red=2, seed=926)
    assert len(rollout) >= 2
    assert any(any(value != 0 for value in encoded[-13:])
               for row in rollout for encoded in row["obs"].values())
    for step, row in enumerate(rollout):
        # Keep the actually sampled actions and their behavior likelihoods.
        # A nonconstant task return makes this a task-policy update without
        # introducing any expert action or arbitration-winner label.
        row["reward"] = float((step % 3) - 1)
        row["f_role"] = 0.2 * (1 if step % 2 else -1)
        row["cf_deltas"] = [0.1, -0.1, 0.0]
        row["action_exposure_mask"] = [False] * 3
        row["bid_cf_adv_slots"] = [[0.0] * 8 for _ in range(3)]
        row["bid_cf_mask_slots"] = [[False] * 8 for _ in range(3)]
        row["bid_event_mask_slots"] = [[False] * 8 for _ in range(3)]
        row["bid_event_ids_slots"] = [[-1] * 8 for _ in range(3)]
    return rollout


def _old_adam_momentum(policy):
    for parameter in policy.model.parameters():
        policy.optimizer.state[parameter] = {
            "step": torch.tensor(4.0),
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }


def test_adapter_reaches_all_task_heads_but_never_axis_or_bid():
    model = DetectActorCritic(hidden=16)
    obs = _observations()
    before = tuple(value.detach().clone() for value in model.action_parameters(obs))
    _activate(model.task_geometry_adapter)
    residual = model.task_geometry_adapter(obs)
    after = model.action_parameters(obs)
    assert torch.count_nonzero(residual[:, 2]) == 0
    for index, start, stop in ((0, 0, 3), (1, 3, 5), (2, 5, 11), (3, 11, 20)):
        assert torch.equal(after[index], before[index] + residual[:, start:stop])
        assert not torch.equal(after[index], before[index])
    assert torch.equal(after[0][:, 2], before[0][:, 2])
    assert torch.equal(after[4], before[4])


def test_legacy_policy_and_both_anchors_gain_zero_output_adapter(tmp_path):
    source = _policy(seed=911)
    obs = _observations()
    # Distinct references ensure loading cannot accidentally replace either
    # serialized anchor with the current policy.
    with torch.no_grad():
        source.model.conflict_residual.weight.fill_(0.13)
        source.xi_anchor_model.role_head.bias.add_(0.4)
        source.policy_anchor_model.gear_head.bias.sub_(0.3)
    components = {
        "model": source.model,
        "xi_anchor_model": source.xi_anchor_model,
        "policy_anchor_model": source.policy_anchor_model,
    }
    expected = {name: tuple(value.detach().clone()
                            for value in model.action_parameters(obs))
                for name, model in components.items()}
    checkpoint = _model_states(source)
    for state in checkpoint.values():
        for name in list(state):
            if name.startswith("task_geometry_adapter."):
                del state[name]
    path = tmp_path / "before_geometry_adapter.pt"
    torch.save(checkpoint, path)
    restored = _policy(seed=912)
    for model in (restored.model, restored.xi_anchor_model, restored.policy_anchor_model):
        _activate(model.task_geometry_adapter)
    rng_before = torch.get_rng_state().clone()
    restored.load(str(path))
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, model in (("model", restored.model),
                        ("xi_anchor_model", restored.xi_anchor_model),
                        ("policy_anchor_model", restored.policy_anchor_model)):
        actual = model.action_parameters(obs)
        assert all(torch.equal(left, right)
                   for left, right in zip(expected[name], actual)), name
        assert torch.count_nonzero(model.task_geometry_adapter.input_layer.weight) > 0
        assert torch.count_nonzero(model.task_geometry_adapter.output_layer.weight) == 0
        assert torch.count_nonzero(model.task_geometry_adapter.output_layer.bias) == 0
    assert restored.checkpoint_migration_record["geometry_adapter"] == {
        "policy": "zero_output_legacy_addition",
        "xi-anchor": "zero_output_legacy_addition",
        "policy-anchor": "zero_output_legacy_addition",
    }


def test_active_adapter_and_distinct_references_survive_save_load(tmp_path):
    source = _policy(seed=913)
    for component in (source.model, source.xi_anchor_model, source.policy_anchor_model):
        _activate(component.task_geometry_adapter)
    with torch.no_grad():
        source.policy_anchor_model.task_geometry_adapter.output_layer.bias.add_(0.17)
    obs = _observations()
    path = tmp_path / "learned_geometry.pt"
    source.save(str(path))
    restored = _policy(seed=914, train_geometry=False)
    restored.load(str(path))
    assert restored.checkpoint_migration_record["geometry_adapter"] == {
        "policy": "restored", "xi-anchor": "restored", "policy-anchor": "restored",
    }
    for name in ("model", "xi_anchor_model", "policy_anchor_model"):
        original = getattr(source, name)
        actual = getattr(restored, name)
        assert all(torch.equal(value, actual.state_dict()[key])
                   for key, value in original.state_dict().items())
        assert all(torch.equal(left, right)
                   for left, right in zip(original.action_parameters(obs),
                                          actual.action_parameters(obs)))


@pytest.mark.parametrize("component", ["model", "xi_anchor_model", "policy_anchor_model"])
def test_partially_saved_adapter_rejected_in_each_component(tmp_path, component):
    policy = _policy()
    checkpoint = _model_states(policy)
    del checkpoint[component]["task_geometry_adapter.input_layer.bias"]
    path = tmp_path / "partial_adapter.pt"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="incomplete .* geometry adapter"):
        policy.load(str(path))


def test_enabled_adapter_learns_in_real_ppo_with_base_and_axis_preserved():
    policy = _policy(train_geometry=True)
    rollout = _task_rollout(policy)
    _old_adam_momentum(policy)
    before = _model_states(policy)
    obs = _observations()
    axis_before = policy.model.action_parameters(obs)[0][:, 2].detach().clone()
    stats = policy.update(rollout)
    after = policy.model.state_dict()
    trainable = {name for name, parameter in policy.model.named_parameters()
                 if parameter.requires_grad}
    geometry_names = {name for name, _ in policy.model.named_parameters()
                      if name.startswith("task_geometry_adapter.")}
    assert geometry_names <= trainable
    assert stats["geometry_adapter_gradient_norm"] > 0
    assert stats["geometry_adapter_parameter_delta_l2"] > 0
    assert stats["geometry_adapter_mean_abs_output"] > 0
    assert 0 < stats["geometry_adapter_max_abs_output"] <= 1
    assert stats["task_residual_xi_row_unchanged"] is True
    assert torch.equal(policy.model.action_parameters(obs)[0][:, 2], axis_before)
    for name, value in before["model"].items():
        if name.startswith(("actor.", "cont_head.", "role_head.", "gear_head.",
                            "pointer_head.")) or name == "log_std":
            assert torch.equal(value, after[name]), name
    for component in ("xi_anchor_model", "policy_anchor_model"):
        assert all(torch.equal(value, getattr(policy, component).state_dict()[name])
                   for name, value in before[component].items()), component


def test_disabled_active_adapter_frozen_even_with_old_adam_momentum():
    policy = _policy(train_geometry=False)
    _activate(policy.model.task_geometry_adapter)
    rollout = _task_rollout(policy)
    _old_adam_momentum(policy)
    before = deepcopy(policy.model.task_geometry_adapter.state_dict())
    obs = _observations()
    output_before = policy.model.task_geometry_adapter(obs).detach().clone()
    stats = policy.update(rollout)
    assert stats["geometry_adapter_gradient_norm"] == 0
    assert stats["geometry_adapter_parameter_delta_l2"] == 0
    assert all(not parameter.requires_grad
               for parameter in policy.model.task_geometry_adapter.parameters())
    assert all(torch.equal(value, policy.model.task_geometry_adapter.state_dict()[name])
               for name, value in before.items())
    assert torch.equal(output_before, policy.model.task_geometry_adapter(obs))


def test_geometry_training_scope_rejected_at_construction_and_after_mutation():
    with pytest.raises(ValueError, match="requires task_residual_only_ppo"):
        DetectMAPPOConfig(train_geometry_adapter=True)
    policy = _policy()
    policy.config.task_residual_only_ppo = False
    with pytest.raises(ValueError, match="requires task_residual_only_ppo"):
        policy.update([])
