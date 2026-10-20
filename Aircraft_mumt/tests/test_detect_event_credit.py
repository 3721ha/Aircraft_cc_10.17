"""Expected direction checks for the actual continuous bid-policy objective."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from aircraft_sim.detect_mappo import (
    DetectMAPPOPolicy, DetectMAPPOConfig, _event_winner_advantage,
    _grouped_masked_sum_mean, _squashed_log_prob,
)
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_models import DetectAction, DetectRole
from aircraft_sim.detect_shield import DetectQPShield


def _quasi_normal(n_candidates, n_samples=32768):
    u = torch.quasirandom.SobolEngine(
        n_candidates, scramble=True, seed=53).draw(n_samples).double()
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


@pytest.mark.parametrize('utilities', [(3., 1.), (3., 1., -1.)])
def test_realized_winner_gradient_matches_expected_utility(utilities):
    z = _quasi_normal(len(utilities))
    utility = torch.tensor(utilities, dtype=torch.float64)
    winner = z.argmax(-1)
    advantages = utility[winner] - utility.mean()
    mu = torch.zeros(len(utilities), dtype=torch.float64, requires_grad=True)
    logp = _squashed_log_prob(Normal(mu, torch.ones_like(mu)), z.tanh())
    old_logp = logp.detach()
    # At the on-policy parameters the PPO ratio's derivative equals the
    # joint-action score function. Sum candidates, then average events.
    objective = ((logp - old_logp).exp() * advantages[:, None]).sum(-1).mean()
    objective.backward()
    # For two/three independent unit Normals at equal means, the pairwise
    # boundary integral is 1/(2 sqrt(pi)) or 1/(4 sqrt(pi)), respectively.
    expected = (utility - utility.mean()) / (2.0 * math.sqrt(math.pi))
    expected *= 2.0 if len(utilities) == 2 else 1.5
    assert torch.allclose(mu.grad, expected, atol=0.008, rtol=0.01)
    assert mu.grad[0] > 0.0 and mu.grad[-1] < 0.0

    # The old candidate-independent credit cancels in expectation.
    wrong_mu = torch.zeros_like(mu, requires_grad=True)
    wrong_logp = _squashed_log_prob(
        Normal(wrong_mu, torch.ones_like(wrong_mu)), z.tanh())
    ((wrong_logp - wrong_logp.detach()).exp()
     * (utility - utility.mean())[None]).sum(-1).mean().backward()
    assert float(wrong_mu.grad.abs().max()) < 0.003


def test_event_advantage_reverses_and_scales_with_utility():
    assert np.isclose(_event_winner_advantage({0: .3, 1: .1}, 0), .1)
    assert np.isclose(_event_winner_advantage({0: .1, 1: .3}, 0), -.1)
    assert np.isclose(_event_winner_advantage({0: 3., 1: 1.}, 0), 1.)


def test_joint_event_reduction_sums_agents_and_averages_events():
    x = torch.tensor([1., 2., 3., 4., 5., 123.], requires_grad=True)
    active = torch.tensor([True, True, True, True, True, False])
    groups = torch.tensor([0, 0, 1, 1, 1, -1])
    result = _grouped_masked_sum_mean(x, active, groups)
    assert result == 7.5
    result.backward()
    assert torch.equal(x.grad, torch.tensor([.5, .5, .5, .5, .5, 0.]))


def test_middle_utility_winner_keeps_zero_advantage_event_mask():
    env = DetectEnv()
    env.reset({'seed': 91, 'k_red': 3})
    policy = DetectMAPPOPolicy(env, DetectQPShield(env.cfg),
        config=DetectMAPPOConfig(), seed=91)
    policy._potential_in_env = lambda *_: 0.0

    class Replay:
        def step(self, actions):
            winner = max(actions, key=lambda i: actions[i].arbitration_bids[0])
            return SimpleNamespace(reward=(3., 2., 1.)[winner], done=False,
                info={'arbitration': [{'tid': 9, 'winner': winner}]})

    actions = {i: DetectAction(role=DetectRole.TRACK,
        arbitration_bids=((.5 if i == 1 else -.1),) + (0.,) * 7)
        for i in range(3)}
    advantage = [[0.] * 8 for _ in range(3)]
    mask = [[False] * 8 for _ in range(3)]
    stats = policy._counterfactual_bid_credit(Replay(), actions,
        [{'tid': 9, 'candidates': [0, 1, 2], 'winner': 1,
          'arbitration_mode': 'rl_direct', 'bid_slots': {0: 0, 1: 0, 2: 0}}],
        0., 0., advantage, mask)
    assert stats['informative_events'] == 1
    assert all(row[0] for row in mask)
    assert all(abs(row[0]) < 1e-8 for row in advantage)
