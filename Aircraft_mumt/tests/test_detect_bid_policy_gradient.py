"""Regression checks for environment-only joint arbitration policy gradients."""

import math

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import (
    DetectMAPPOConfig, DetectMAPPOPolicy, _event_winner_advantage,
    _grouped_masked_sum_mean, _squashed_log_prob,
)
from aircraft_sim.detect_shield import DetectQPShield


@pytest.mark.parametrize(
    "utilities, coefficient",
    [([3.0, 1.0], 1.0 / math.sqrt(math.pi)),
     ([3.0, 1.0, -1.0], 3.0 / (4.0 * math.sqrt(math.pi)))],
)
def test_joint_winner_gradient_matches_gaussian_arbitration(utilities, coefficient):
    """The score gradient matches the analytic derivative of E[U(argmax bids)]."""
    n = len(utilities)
    unit = torch.quasirandom.SobolEngine(
        n, scramble=True, seed=12).draw_base2(16).double().clamp(1e-9, 1 - 1e-9)
    bids = math.sqrt(2.0) * torch.erfinv(2.0 * unit - 1.0)
    utility = torch.tensor(utilities, dtype=torch.float64)
    winners = bids.argmax(-1)
    baseline = utility.mean()
    groups = torch.arange(len(bids)).unsqueeze(1).expand_as(bids)
    mask = torch.ones_like(bids, dtype=torch.bool)
    expected = (utility - baseline) * coefficient

    for scale in (1.0, -1.0, 3.0):
        loc = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        advantage = scale * (utility[winners] - baseline)
        logp = Normal(loc, torch.ones_like(loc)).log_prob(bids)
        objective = _grouped_masked_sum_mean(
            logp * advantage.unsqueeze(1), mask, groups)
        gradient, = torch.autograd.grad(objective, loc)
        assert torch.allclose(gradient, expected * scale, atol=0.004 * abs(scale))
        for winner in range(n):
            assert _event_winner_advantage(
                dict(enumerate(utilities)), winner) == pytest.approx(
                    utilities[winner] - float(baseline))

    # The previous candidate-specific constant is independent of its sampled
    # bid, so its expected score gradient is zero instead of the task gradient.
    old_gradient = ((utility - baseline).unsqueeze(0) * bids).mean(0)
    assert old_gradient.abs().max() < 0.001


@pytest.mark.parametrize("utility_sign", [1.0, -1.0])
def test_real_bid_only_update_moves_utility_preference_in_correct_direction(utility_sign):
    """Exercise update(), not only a stand-alone score-function surrogate."""
    env = DetectEnv()
    env.reset({"seed": 81, "k_red": 2})
    policy = DetectMAPPOPolicy(
        env, DetectQPShield(env.cfg), seed=81,
        config=DetectMAPPOConfig(
            bid_only_ppo=True, epochs=1, entropy_coef=0.0,
            learning_rate=1e-3, cf_bid_utility_scale=0.1))
    n = env.cfg.N_WINGMEN
    assert n == 3
    # Distinct observed conflict features give the shared bid head a lawful
    # state-dependent way to distinguish candidates. All initial means are 0.
    obs = np.zeros((n, policy.encoder.dimension), dtype=np.float32)
    obs[:, -13:-10] = np.eye(n, dtype=np.float32)
    with torch.no_grad():
        for module in (policy.model.bid_head, policy.model.bid_candidate_head,
                       policy.model.bid_conflict_residual):
            for parameter in module.parameters():
                parameter.zero_()
        initial = policy.model.action_parameters(torch.from_numpy(obs))[-1][:, 0]
    unit = torch.quasirandom.SobolEngine(
        n, scramble=True, seed=81).draw_base2(10).clamp(1e-7, 1 - 1e-7)
    latent = math.sqrt(2.0) * torch.erfinv(2.0 * unit - 1.0)
    std = policy.model.bid_log_std.detach().exp()
    sampled = torch.tanh(std * latent)
    utility = utility_sign * torch.tensor([3.0, 1.0, -1.0])
    rollout = []
    for sample in sampled:
        advantage = float(utility[sample.argmax()] - utility.mean())
        bids = torch.zeros((n, 8))
        bids[:, 0] = sample
        logp = _squashed_log_prob(
            Normal(torch.zeros_like(bids), std), bids)
        old_logp = {i: {
            "uv": 0.0, "xi": 0.0, "role": 0.0, "gear": 0.0, "ptr": 0.0,
            "bid": float(logp[i, 0]), "bid_slots": logp[i].tolist(),
            "combined": float(logp[i, 0]),
        } for i in range(n)}
        nominal = {i: [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, float(sample[i])]
                   for i in range(n)}
        rollout.append({
            "obs": {i: obs[i] for i in range(n)},
            "team": obs.reshape(-1), "nominal": nominal, "safe": nominal,
            "bid_actions": {i: bids[i].tolist() for i in range(n)},
            "old_logp": old_logp, "reward": 0.0, "value": 0.0, "done": True,
            "f_role": 0.0, "gear_gate_credit": [0.0] * n,
            "cf_deltas": [0.0] * n, "bid_rank": [0.0] * n,
            "bid_mask": [True] * n,
            "bid_cf_adv_slots": [[advantage] + [0.0] * 7 for _ in range(n)],
            "bid_cf_mask_slots": [[True] + [False] * 7 for _ in range(n)],
            "bid_event_mask_slots": [[True] + [False] * 7 for _ in range(n)],
            "bid_event_ids_slots": [[0] + [-1] * 7 for _ in range(n)],
            "action_exposure_mask": [False] * n,
        })
    policy.update(rollout)
    with torch.no_grad():
        final = policy.model.action_parameters(torch.from_numpy(obs))[-1][:, 0]
    preferred, inferior = (0, 2) if utility_sign > 0 else (2, 0)
    assert float(final[preferred] - final[inferior]) > (
        float(initial[preferred] - initial[inferior]) + 1e-4)
