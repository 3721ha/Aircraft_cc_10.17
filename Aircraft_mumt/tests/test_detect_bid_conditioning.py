import tempfile
from pathlib import Path

import torch

from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import (DetectActorCritic, DetectMAPPOPolicy,
                                       _squashed_log_prob)
from aircraft_sim.detect_models import DetectAction, DetectRole
from aircraft_sim.detect_shield import DetectQPShield
from torch.distributions import Normal


def test_bid_residual_depends_on_selected_local_track():
    model = DetectActorCritic(hidden=16)
    obs = torch.zeros((1, 169))
    obs[0, 17] = 1.0
    obs[0, 31] = -1.0
    pointer = torch.tensor([1])

    base_bid = model.action_parameters(obs)[-1]
    initial_bid = model.action_parameters(obs, pointer)[-1]
    assert torch.equal(initial_bid, base_bid[:, :1])

    with torch.no_grad():
        first = model.bid_candidate_head[0]
        last = model.bid_candidate_head[-1]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, model.actor[-2].out_features] = 1.0
        last.weight.zero_()
        last.bias.zero_()
        last.weight[0, 0] = 1.0

    pointers = torch.tensor([1, 2])
    bids = model.action_parameters(obs.expand(2, -1), pointers)[-1]
    hidden = model.actor(obs.expand(2, -1)).detach()
    base_selected = model.bid_head(hidden.detach())
    deltas = bids - base_selected
    assert deltas[0, 0] > 0
    assert deltas[1, 0] < 0

    zero_pointer = model.action_parameters(obs, torch.tensor([0]))[-1]
    assert torch.equal(zero_pointer, base_bid[:, :1])


def test_bid_candidate_head_receives_policy_gradient():
    model = DetectActorCritic(hidden=16)
    obs = torch.zeros((2, 169))
    obs[0, 17] = 0.5
    obs[1, 17] = -0.5
    bids = model.action_parameters(obs, torch.tensor([1, 1]))[-1]
    (bids[0].sum() - bids[1].sum()).backward()
    assert model.bid_candidate_head[-1].weight.grad is not None
    assert model.bid_candidate_head[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.actor.parameters())


def test_exposed_bid_resample_refreshes_old_policy_likelihood():
    env = DetectEnv()
    policy = DetectMAPPOPolicy(env, DetectQPShield(env.cfg), seed=19)
    encoded = {0: torch.zeros(169).numpy()}
    encoded[0][17] = 1.0
    actions = {0: DetectAction(pointer=1)}
    old_logp = {0: {"bid": -4.0, "combined": -5.0}}

    policy.condition_exposed_bids(encoded, actions, old_logp, [True])

    obs = torch.tensor(encoded[0]).unsqueeze(0)
    pointer = torch.tensor([1])
    with torch.no_grad():
        bid_loc = policy.model.action_parameters(obs, pointer)[-1]
        bid_dist = Normal(bid_loc, policy.model.bid_log_std.exp())
        expected = _squashed_log_prob(
            bid_dist,
            torch.tensor([[actions[0].arbitration_bid]])).sum(-1)
    assert abs(old_logp[0]["bid"] - float(expected)) < 1e-5
    assert abs(old_logp[0]["combined"] - (-1.0 + float(expected))) < 1e-5


def test_legacy_checkpoint_preserves_initialized_candidate_head():
    env = DetectEnv()
    policy = DetectMAPPOPolicy(env, DetectQPShield(env.cfg), seed=17)
    model_candidate = {
        key: value.detach().clone()
        for key, value in policy.model.state_dict().items()
        if key.startswith("bid_candidate_head.")
    }
    anchor_candidate = {
        key: value.detach().clone()
        for key, value in policy.xi_anchor_model.state_dict().items()
        if key.startswith("bid_candidate_head.")
    }
    legacy_state = {
        key: value for key, value in policy.model.state_dict().items()
        if not key.startswith("bid_candidate_head.")
        and key != "conflict_residual.weight"
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "legacy.pt"
        torch.save({"model": legacy_state,
                    "xi_anchor_model": legacy_state}, path)
        with torch.no_grad():
            for parameter in policy.xi_anchor_model.bid_candidate_head.parameters():
                parameter.fill_(3.0)
        anchor_candidate = {
            key: value.detach().clone()
            for key, value in policy.xi_anchor_model.state_dict().items()
            if key.startswith("bid_candidate_head.")
        }
        policy.load(str(path))

    for key, expected in model_candidate.items():
        assert torch.equal(policy.model.state_dict()[key], expected)
    for key, expected in anchor_candidate.items():
        assert torch.equal(policy.xi_anchor_model.state_dict()[key], expected)
    assert policy.model.bid_candidate_head[0].weight.abs().sum() > 0
    assert torch.count_nonzero(policy.model.conflict_residual.weight) == 0
    assert torch.count_nonzero(policy.xi_anchor_model.conflict_residual.weight) == 0

    obs = torch.zeros((2, 169))
    obs[0, 17] = 0.5
    obs[1, 17] = -0.5
    bids = policy.model.action_parameters(obs, torch.tensor([1, 1]))[-1]
    (bids[0].sum() - bids[1].sum()).backward()
    grad = policy.model.bid_candidate_head[-1].weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0

    head_optimizer = torch.optim.SGD(
        policy.model.bid_candidate_head.parameters(), lr=0.1)
    head_optimizer.step()
    head_optimizer.zero_grad()
    bids = policy.model.action_parameters(obs, torch.tensor([1, 1]))[-1]
    (bids[0].sum() - bids[1].sum()).backward()
    feature_grad = policy.model.bid_candidate_head[0].weight.grad
    assert feature_grad is not None
    assert feature_grad.abs().sum() > 0


def test_behavior_clone_does_not_train_conflict_arbitration_head():
    env = DetectEnv()
    policy = DetectMAPPOPolicy(env, DetectQPShield(env.cfg), seed=21)
    observation = env.observe()[0]
    encoded = policy.encoder.encode(observation)
    before = [parameter.detach().clone()
              for parameter in policy.model.bid_candidate_head.parameters()]

    policy.behavior_clone(
        [(encoded, DetectAction(role=DetectRole.TRACK, pointer=1))],
        epochs=1, batch_size=1)

    for before_param, after_param in zip(
            before, policy.model.bid_candidate_head.parameters()):
        assert torch.equal(before_param, after_param)
        assert after_param.requires_grad
