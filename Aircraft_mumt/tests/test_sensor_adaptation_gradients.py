"""Radar credit survives a flight-only projection only by explicit opt-in."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
from aircraft_sim.detect_shield import DetectQPShield


def _policy(**overrides):
    cfg = replace(DetectSimConfig(), horizon=8, K_RED=4,
                  STABLE_WINDOW=8, AGE_STD_TRACK=10,
                  RADAR_HIT_MODEL="deterministic_beam")
    config = dict(hidden_size=16, epochs=1, target_kl=1.0,
                  bid_target_kl=1.0, xi_ppo_coef=1.0, xi_anchor_coef=0.0,
                  policy_anchor_coef=0.0, entropy_coef=0.0,
                  value_coef=0.0, intervention_coef=0.0,
                  direct_rl_arbitration=True, conflict_exposure_prob=0.0)
    config.update(overrides)
    env = DetectEnv(cfg)
    return DetectMAPPOPolicy(env, DetectQPShield(cfg),
                            DetectMAPPOConfig(**config), seed=275)


def _flight_corrected_rollout(policy, *, exposed=False):
    data = policy.collect(k_red=4, seed=276)
    assert len(data) == policy.env.cfg.horizon
    for step, item in enumerate(data):
        # A controlled return signal and recorded flight projection isolate
        # the sample mask. No action, winner or radar answer is supplied.
        item["reward"] = float(step % 3 - 1)
        item["value"] = 0.0
        item["action_exposure_mask"] = [exposed] * 3
        for i in range(3):
            safe = list(item["safe"][i])
            nominal = item["nominal"][i]
            safe[0] = nominal[0] + (0.25 if nominal[0] <= 0.5 else -0.25)
            safe[1] = nominal[1]
            safe[2] = nominal[2]
            item["safe"][i] = safe
        assert not any(item["bid_cf_mask"])
    return data


@pytest.mark.parametrize("mask_axis", [True, False])
def test_radar_axis_opt_in_retains_return_credit_but_flight_stays_masked(mask_axis):
    policy = _policy(mask_intervened_radar_axis=mask_axis)
    data = _flight_corrected_rollout(policy)
    before = deepcopy(policy.model.cont_head.state_dict())
    stats = policy.update(data)
    after = policy.model.cont_head.state_dict()
    assert stats["flight_ppo_active_fraction"] == 0.0
    assert stats["radar_axis_ppo_active_fraction"] == (0.0 if mask_axis else 1.0)
    for name in ("weight", "bias"):
        assert torch.equal(before[name][:2], after[name][:2]), name
        assert torch.count_nonzero(getattr(policy.model.cont_head, name).grad[:2]) == 0
    if mask_axis:
        assert all(torch.equal(before[name], after[name]) for name in before)
        assert torch.count_nonzero(policy.model.cont_head.weight.grad[2]) == 0
    else:
        assert not torch.equal(before["weight"][2], after["weight"][2])
        assert policy.model.cont_head.weight.grad[2].norm() > 0


def test_default_radar_mask_is_exactly_the_explicit_legacy_setting():
    default = _policy()
    explicit = _policy(mask_intervened_radar_axis=True)
    assert default.config.mask_intervened_radar_axis is True
    data = _flight_corrected_rollout(default)
    left = default.update(deepcopy(data))
    right = explicit.update(deepcopy(data))
    assert left == right
    for name, value in default.model.state_dict().items():
        assert torch.equal(value, explicit.model.state_dict()[name]), name


def test_exposure_still_masks_radar_axis_even_with_projection_mask_disabled():
    policy = _policy(mask_intervened_radar_axis=False)
    data = _flight_corrected_rollout(policy, exposed=True)
    before = deepcopy(policy.model.state_dict())
    stats = policy.update(data)
    assert stats["radar_axis_ppo_active_fraction"] == 0.0
    assert stats["flight_ppo_active_fraction"] == 0.0
    assert stats["bid_active_slot_count"] == 0
    for name, value in policy.model.state_dict().items():
        assert torch.equal(before[name], value), name


def test_disabling_all_flight_projection_masking_preserves_legacy_axis_behavior():
    policy = _policy(mask_intervened_continuous=False,
                     mask_intervened_radar_axis=True)
    stats = policy.update(_flight_corrected_rollout(policy))
    assert stats["flight_ppo_active_fraction"] == 1.0
    assert stats["radar_axis_ppo_active_fraction"] == 1.0
    assert policy.model.cont_head.weight.grad[:2].norm() > 0
    assert policy.model.cont_head.weight.grad[2].norm() > 0


def test_zero_radar_ppo_coefficient_reports_no_active_radar_ppo_samples():
    policy = _policy(mask_intervened_radar_axis=False, xi_ppo_coef=0.0)
    stats = policy.update(_flight_corrected_rollout(policy))
    assert stats["radar_axis_ppo_active_fraction"] == 0.0
    assert torch.count_nonzero(policy.model.cont_head.weight.grad[2]) == 0


def test_joint_environment_return_update_moves_task_and_axis_keeps_references_fixed():
    policy = _policy(mask_intervened_radar_axis=False, value_coef=0.5,
                     intervention_coef=0.2, entropy_coef=0.01,
                     policy_anchor_coef=0.5, policy_anchor_monitor=True)
    original_reset = policy.env.reset

    def reset_with_local_contacts(scenario=None):
        original_reset(scenario)
        # A small contact-state fixture exercises the legal pointer support.
        # Contacts follow the normal own-radar track ingestion API; there is
        # no expert action or forced responsibility/arbitration label.
        for i, wingman in enumerate(policy.env.wingmen):
            wingman.x = 100.0
            target = policy.env.reds.agents[i]
            hit = [(target.uid, target.x, target.y)]
            wingman.tracks.update_with_hits(hit, -0.5, 1.0)
            wingman.tracks.update_with_hits(hit, 0.0, 1.0)
        return policy.env.observe()

    with patch.object(policy.env, "reset", side_effect=reset_with_local_contacts):
        data = policy.collect(k_red=4, seed=277)
    assert not any(any(item["action_exposure_mask"]) for item in data)
    assert sum(item["n_forced_overlap_pairs"] for item in data) == 0
    assert all(item["nominal"][i][2] == item["safe"][i][2]
               for item in data for i in range(3))
    before = deepcopy(policy.model.state_dict())
    xi_reference = deepcopy(policy.xi_anchor_model.state_dict())
    task_reference = deepcopy(policy.policy_anchor_model.state_dict())
    stats = policy.update(data)
    after = policy.model.state_dict()
    for prefix in ("actor.", "cont_head.", "role_head.", "gear_head.",
                   "pointer_head.", "critic.", "conflict_residual."):
        assert any(not torch.equal(before[name], after[name])
                   for name in before if name.startswith(prefix)), prefix
    assert not torch.equal(before["cont_head.weight"][2], after["cont_head.weight"][2])
    for name in before:
        if name.startswith("task_geometry_adapter."):
            assert torch.equal(before[name], after[name]), name
    for module, reference in ((policy.xi_anchor_model, xi_reference),
                              (policy.policy_anchor_model, task_reference)):
        assert all(torch.equal(value, module.state_dict()[name])
                   for name, value in reference.items())
        assert all(not parameter.requires_grad for parameter in module.parameters())
    # No natural informative event means no claim of new arbitration learning.
    if stats["bid_active_slot_count"] == 0:
        assert stats["bid_gradient_norm"] == 0.0
        assert stats["bid_parameter_delta_l2"] == 0.0
    assert stats["radar_axis_ppo_active_fraction"] == 1.0
