"""Multi-step replay isolation with the real simulator and QP shield."""
from copy import deepcopy
from dataclasses import replace
import pickle
import random

import numpy as np
import torch

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy
from aircraft_sim.detect_models import DetectAction, DetectRole
from aircraft_sim.detect_shield import DetectQPShield


def _real_contested_state():
    """Find shared sensor tracks through ordinary dynamics, without truth edits.

    These are test requests, not a training protocol or a winner label. The
    target and pointer slots come from the actual onboard sensor track files.
    """
    cfg = replace(DetectSimConfig(), STABLE_WINDOW=8, AGE_STD_TRACK=10.0)
    env = DetectEnv(cfg)
    env.reset({"seed": 180100, "k_red": 4})
    shield = DetectQPShield(cfg)
    search = {i: DetectAction(u_v=0.5) for i in range(env.n)}
    for _ in range(400):
        visible = {}
        for i, wingman in enumerate(env.wingmen):
            for slot, track in enumerate(wingman.tracks.slots):
                if (track is not None and track.state in ("confirmed", "tentative")
                        and track.age(env.t) <= 8.0):
                    visible.setdefault(track.target_id, []).append((i, slot))
        for tid, requesters in sorted(visible.items()):
            if len(requesters) < 2:
                continue
            nominal = deepcopy(search)
            for i, slot in requesters:
                bids = [0.0] * 8
                bids[slot] = 0.7 if i == requesters[0][0] else -0.2
                nominal[i] = DetectAction(
                    u_v=0.5, role=DetectRole.TRACK, gear=45.0,
                    pointer=slot + 1, arbitration_bids=tuple(bids),
                    rl_direct_arbitration=True)
            actions, _ = shield.filter(env, nominal)
            preview = deepcopy(env).step(deepcopy(actions))
            event = next((item for item in preview.info["arbitration"]
                          if item["tid"] == tid
                          and len(item["candidates"]) >= 2), None)
            if event is not None:
                return env, shield, actions, tid, event
        actions, _ = shield.filter(env, deepcopy(search))
        result = env.step(actions)
        assert not result.done, "test requires a shared live sensor track"
    raise AssertionError("no genuine contested request found in 400 decisions")


def test_h8_real_environment_sensor_red_rng_shield_and_actor_are_isolated(monkeypatch):
    env, shield, actions, tid, actual_event = _real_contested_state()
    policy = DetectMAPPOPolicy(
        env, shield, DetectMAPPOConfig(hidden_size=16, cf_bid_horizon=8),
        seed=180101)
    before_env = pickle.dumps(env)
    before_shield = pickle.dumps(shield)
    before_actions = deepcopy(actions)
    before_parameters = {name: value.clone()
                         for name, value in policy.model.state_dict().items()}
    before_private_rng = (policy._conflict_rng.getstate(),
                          policy._conflict_probe_rng.getstate())
    before_python = random.getstate()
    before_numpy = pickle.dumps(np.random.get_state())
    before_torch = torch.random.get_rng_state().clone()
    before_cuda = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_initialized() else None)
    traces = []
    continuation_shields = []
    continuation_times = []
    original_step = DetectEnv.step
    original_filter = DetectQPShield.filter
    original_actor = policy.act_batch

    def observe_real_step(branch, branch_actions):
        assert branch is not env
        assert branch.rng is not env.rng
        assert branch.reds.rng is not env.reds.rng
        sensor_rng = branch.rng.getstate()
        result = original_step(branch, branch_actions)
        traces.append((branch.t, sensor_rng != branch.rng.getstate(),
                       pickle.dumps(branch.reds)))
        return result

    def observe_real_filter(branch_shield, branch, nominal):
        assert branch_shield is not shield
        assert branch is not env
        continuation_shields.append(branch_shield)
        return original_filter(branch_shield, branch, nominal)

    def observe_real_actor(observations, deterministic=False):
        assert deterministic is True
        assert policy.env is env
        timestamp = observations[0]["mission"]["t"]
        assert timestamp > env.t
        continuation_times.append(timestamp)
        return original_actor(observations, deterministic=deterministic)

    monkeypatch.setattr(DetectEnv, "step", observe_real_step)
    monkeypatch.setattr(DetectQPShield, "filter", observe_real_filter)
    monkeypatch.setattr(policy, "act_batch", observe_real_actor)
    phi = policy._potential_in_env(env, "track")
    phi_stable = policy._potential_in_env(env, "stable")
    first = policy._rollout_counterfactual_bid_branch(
        env, actions, tid, phi, phi_stable)
    second = policy._rollout_counterfactual_bid_branch(
        env, actions, tid, phi, phi_stable)

    assert first == second
    assert first[1] == actual_event
    assert first[2]["steps"] == first[2]["horizon"] == 8
    assert first[2]["terminal"] is False
    assert len(traces) == 16
    # The actual sensor RNG advances inside each branch; this is not merely a
    # copy of a simulator state in which no random sensor operation occurred.
    assert any(sensor_changed for _, sensor_changed, _ in traces)
    assert traces[:8] == traces[8:]
    assert traces[0][2] != traces[-1][2]  # real red-agent dynamics progressed
    assert continuation_times == [env.t + i for i in range(1, 8)] * 2
    assert len(continuation_shields) == 14
    assert len({id(value) for value in continuation_shields[:7]}) == 1
    assert len({id(value) for value in continuation_shields[7:]}) == 1
    assert continuation_shields[0] is not continuation_shields[7]
    assert pickle.dumps(env) == before_env
    assert pickle.dumps(shield) == before_shield
    assert actions == before_actions
    assert policy.env is env
    assert policy.shield is shield
    assert before_private_rng == (policy._conflict_rng.getstate(),
                                  policy._conflict_probe_rng.getstate())
    assert all(torch.equal(value, before_parameters[name])
               for name, value in policy.model.state_dict().items())
    assert random.getstate() == before_python
    assert pickle.dumps(np.random.get_state()) == before_numpy
    assert torch.equal(torch.random.get_rng_state(), before_torch)
    if before_cuda is not None:
        assert all(torch.equal(before, after) for before, after in
                   zip(before_cuda, torch.cuda.get_rng_state_all()))
