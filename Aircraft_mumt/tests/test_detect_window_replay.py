"""Conditional multi-step RL replay: delayed outcomes and isolation."""
from copy import deepcopy
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
from aircraft_sim.detect_models import DetectAction, DetectRole


def _actions():
    return {
        0: DetectAction(role=DetectRole.TRACK, pointer=1,
                        arbitration_bids=(0.3, 0.2) + (0.0,) * 6),
        1: DetectAction(role=DetectRole.TRACK, pointer=1,
                        arbitration_bids=(-0.1, 0.4) + (0.0,) * 6),
    }


class ReplayEnv:
    def __init__(self, rewards=((3.0, -5.0), (1.0, 4.0)),
                 terminal_at=None, pool=(0, 1), consume_global_rng=False):
        self.rewards = rewards
        self.terminal_at = terminal_at
        self.pool = pool
        self.consume_global_rng = consume_global_rng
        self.t = 0
        self.winner = None
        self.rng = random.Random(491)
        self.draws = []
        self.stable_progress = {9: 2}
        self.judged = {9: False}
        self.track_phi = 0.7
        self.stable_phi = 0.2

    def observe(self):
        return {"winner": self.winner, "t": self.t,
                "private_draw": self.draws[-1]}

    def step(self, actions):
        if self.consume_global_rng:
            random.random()
            np.random.random()
            torch.rand(1)
        self.draws.append(self.rng.random())
        if not self.t:
            self.winner = max(actions,
                              key=lambda i: actions[i].arbitration_bids[0])
        else:
            # A continuation built from the live/main env cannot pass.
            assert actions[0].u_psi == float(self.winner)
        offset = min(self.t, len(self.rewards[self.winner]) - 1)
        reward = self.rewards[self.winner][offset]
        self.t += 1
        self.stable_progress[9] += 1
        self.judged[9] = self.t >= 2 and self.winner == 1
        return SimpleNamespace(
            reward=reward, done=self.t == self.terminal_at,
            info={"arbitration": [{"tid": 9, "winner": self.winner,
                                    "candidates": list(self.pool)}]})


class ReplayShield:
    def __init__(self):
        self.calls = 0

    def filter(self, env, actions):
        assert env.t >= 1
        self.calls += 1
        assert self.calls == env.t
        return actions, []


def _policy(horizon=1, **kwargs):
    # Isolate replay semantics from actor initialization and unrelated rules.
    policy = object.__new__(DetectMAPPOPolicy)
    policy.config = DetectMAPPOConfig(cf_bid_horizon=horizon, **kwargs)
    policy.env = SimpleNamespace(marker="main-env")
    policy.shield = ReplayShield()
    policy._potential_in_env = lambda env, kind: (
        env.track_phi if kind == "track" else env.stable_phi)

    def actor(observations, deterministic=False):
        assert deterministic is True
        assert observations["t"] >= 1
        assert policy.env.marker == "main-env"
        return {0: DetectAction(u_psi=float(observations["winner"]))}, {}

    policy.act_batch = actor
    return policy


def _credit(policy, env):
    adv = [[0.0] * 8 for _ in range(2)]
    masks = [[False] * 8 for _ in range(2)]
    result = policy._counterfactual_bid_credit(
        env, _actions(),
        [{"tid": 9, "winner": 0, "candidates": [0, 1],
          "arbitration_mode": "rl_direct", "bid_slots": {0: 0, 1: 0}}],
        0.0, 0.0, adv, masks)
    return result, adv, masks


def test_horizon_one_matches_existing_one_step_utility():
    policy = _policy(horizon=1, gamma=0.8, reward_scale=0.1)
    utility, matching, diag = policy._rollout_counterfactual_bid_branch(
        ReplayEnv(), _actions(), 9, 0.4, 0.3)
    expected = 3.0 * 0.1 + 0.8 * 0.7 - 0.4 + 0.8 * 0.2 - 0.3
    assert utility == pytest.approx(expected)
    assert matching["winner"] == 0
    assert diag["steps"] == diag["horizon"] == 1
    assert diag["discounted_env_return"] == 3.0


def test_delayed_outcome_reverses_one_step_preference():
    short = _policy(horizon=1, gamma=0.9)
    long = _policy(horizon=2, gamma=0.9)
    short_result, short_adv, _ = _credit(short, ReplayEnv())
    long_result, long_adv, long_masks = _credit(long, ReplayEnv())
    assert short_adv[0][0] > 0.0
    assert long_adv[0][0] < 0.0
    assert long_adv[1][0] == long_adv[0][0]
    assert long_masks[0][0] and long_masks[1][0]
    first = short_result["event_records"][0]
    event = long_result["event_records"][0]
    assert first["utilities"][0] > first["utilities"][1]
    assert event["utilities"][0] < event["utilities"][1]
    assert event["horizon"] == 2
    assert event["branch_results"][1]["judged_after"] is True
    assert event["branch_results"][0]["stable_progress_after"] == 4


@pytest.mark.parametrize("terminal_at,expected_steps", [(None, 3), (1, 1), (2, 2)])
def test_discounted_rewards_telescoping_potentials_and_terminal(
        terminal_at, expected_steps):
    policy = _policy(horizon=3, gamma=0.5, reward_scale=0.2)
    utility, _, diag = policy._rollout_counterfactual_bid_branch(
        ReplayEnv(rewards=((2, 4, 8), (2, 4, 8)),
                  terminal_at=terminal_at), _actions(), 9, 0.3, 0.4)
    discounted = sum((0.5 ** i) * r
                     for i, r in enumerate((2, 4, 8)[:expected_steps]))
    final_phi = 0.9 if terminal_at is None else 0.0
    expected = 0.2 * discounted + (0.5 ** expected_steps) * final_phi - 0.7
    assert utility == pytest.approx(expected)
    assert diag["discounted_env_return"] == discounted
    assert diag["steps"] == expected_steps
    assert diag["terminal"] is (terminal_at is not None)
    if terminal_at is not None:
        assert diag["track_potential_after"] == 0.0
        assert diag["stable_potential_after"] == 0.0


def test_branch_env_shield_actions_and_global_rng_are_isolated():
    policy = _policy(horizon=3)
    replay = ReplayEnv(consume_global_rng=True)
    actions = _actions()
    original_env = pickle.dumps(replay)
    original_actions = deepcopy(actions)
    main_env = policy.env
    python_before = random.getstate()
    numpy_before = pickle.dumps(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    first = policy._rollout_counterfactual_bid_branch(
        replay, actions, 9, 0.0, 0.0)
    second = policy._rollout_counterfactual_bid_branch(
        replay, actions, 9, 0.0, 0.0)
    assert first == second
    assert pickle.dumps(replay) == original_env
    assert actions == original_actions
    assert policy.env is main_env
    assert policy.shield.calls == 0
    assert random.getstate() == python_before
    assert pickle.dumps(np.random.get_state()) == numpy_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_rng_and_env_are_restored_after_replay_exception():
    policy = _policy(horizon=2)
    replay = ReplayEnv(consume_global_rng=True)
    main_env = policy.env
    python_before = random.getstate()
    numpy_before = pickle.dumps(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()

    def fail(_observations, deterministic):
        raise RuntimeError("continuation failed")

    policy.act_batch = fail
    with pytest.raises(RuntimeError, match="continuation failed"):
        policy._rollout_counterfactual_bid_branch(replay, _actions(), 9, 0.0, 0.0)
    assert policy.env is main_env
    assert random.getstate() == python_before
    assert pickle.dumps(np.random.get_state()) == numpy_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert replay.t == 0


def test_changed_candidate_pool_is_uncontrollable():
    result, adv, masks = _credit(_policy(horizon=2), ReplayEnv(pool=(0, 1, 2)))
    assert result["events"] == 0
    assert result["uncontrollable_events"] == 1
    assert not any(any(row) for row in masks)
    assert not any(any(row) for row in adv)


@pytest.mark.parametrize("horizon", [True, False, 0, -1, 33, 1.0, 8.5, "8", None])
def test_invalid_horizon_is_rejected_on_construction_and_runtime(horizon):
    with pytest.raises(ValueError, match="cf_bid_horizon"):
        DetectMAPPOConfig(cf_bid_horizon=horizon)
    policy = _policy()
    policy.config.cf_bid_horizon = horizon
    with pytest.raises(ValueError, match="cf_bid_horizon"):
        policy.update([])
    with pytest.raises(ValueError, match="cf_bid_horizon"):
        _credit(policy, ReplayEnv())
    with pytest.raises(ValueError, match="cf_bid_horizon"):
        policy._rollout_counterfactual_bid_branch(ReplayEnv(), _actions(), 9, 0, 0)


@pytest.mark.parametrize("horizon", [1, 8, 32])
def test_horizon_boundaries_are_valid(horizon):
    assert DetectMAPPOConfig(cf_bid_horizon=horizon).cf_bid_horizon == horizon
