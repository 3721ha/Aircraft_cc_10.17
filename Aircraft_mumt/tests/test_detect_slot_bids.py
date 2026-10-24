from types import SimpleNamespace

import numpy as np
import torch

from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_mappo import DetectMAPPOPolicy, DetectMAPPOConfig
from aircraft_sim.detect_models import DetectAction, DetectRole
from aircraft_sim.detect_shield import DetectQPShield


def _policy(seed=0, **config_kwargs):
    env = DetectEnv()
    observations = env.reset({"seed": seed, "k_red": 2})
    policy = DetectMAPPOPolicy(
        env, DetectQPShield(env.cfg),
        config=DetectMAPPOConfig(**config_kwargs), seed=seed)
    return env, observations, policy


def test_candidate_features_produce_slot_specific_bid_means():
    _, _, policy = _policy(seed=31)
    obs = torch.zeros((1, policy.encoder.dimension))
    obs[0, 17 + 2 * 14] = 1.0
    with torch.no_grad():
        first = policy.model.bid_candidate_head[0]
        last = policy.model.bid_candidate_head[-1]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 128] = 1.0
        last.weight.zero_()
        last.bias.zero_()
        last.weight[0, 0] = 1.0
        slot_locs = policy.model.action_parameters(obs)[-1]

    assert slot_locs.shape == (1, 8)
    assert torch.allclose(
        slot_locs[0, 2] - slot_locs[0, 0],
        torch.tanh(torch.tensor(1.0)))
    assert torch.allclose(
        slot_locs[0, [0, 1, 3, 4, 5, 6, 7]],
        slot_locs[0, 0].expand(7))


def test_counterfactual_bid_credit_uses_only_event_bid_slots():
    _, _, policy = _policy(seed=32)
    baseline = {
        0: DetectAction(role=DetectRole.TRACK,
                        arbitration_bids=(0.1, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0)),
        1: DetectAction(role=DetectRole.TRACK,
                        arbitration_bids=(0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0)),
    }

    class ReplayEnv:
        def __init__(self, calls=None):
            self.calls = [] if calls is None else calls

        def __deepcopy__(self, memo):
            return ReplayEnv(self.calls)

        def step(self, actions):
            self.calls.append({
                i: tuple(actions[i].arbitration_bids)
                for i in actions})
            winner = (0 if actions[0].arbitration_bids[2] > 0.0 else 1)
            return SimpleNamespace(
                reward=2.0 if winner == 0 else 1.0,
                done=False,
                info={"arbitration": [{"tid": 9, "winner": winner}]})

    replay = ReplayEnv()
    policy._potential_in_env = lambda _env, _kind: 0.0
    advantages = [[0.0] * 8 for _ in range(3)]
    masks = [[False] * 8 for _ in range(3)]
    stats = policy._counterfactual_bid_credit(
        replay, baseline,
        [{"tid": 9, "candidates": [0, 1], "winner": 0,
          "arbitration_mode": "rl_direct",
          "bid_slots": {0: 2, 1: 5}}],
        phi_before=0.0,
        phi_stable_before=0.0, advantages=advantages, masks=masks)

    assert stats["events"] == 1
    assert stats["informative_events"] == 1
    assert stats["candidate_branches"] == 2
    assert stats["actual_regrets"] == [0.0]
    assert masks[0][2]
    assert not any(masks[0][slot] for slot in range(8) if slot != 2)
    assert masks[1][5]
    assert not any(masks[1][slot] for slot in range(8) if slot != 5)
    assert advantages[1][5] == -advantages[0][2]
    assert advantages[0][2] > 0.0
    assert len(replay.calls) == 2
    assert replay.calls[0][0][2] == 1.0
    assert replay.calls[0][1][5] == -1.0
    assert replay.calls[0][0][0] == baseline[0].arbitration_bids[0]
    assert replay.calls[0][1][0] == baseline[1].arbitration_bids[0]
    assert replay.calls[1][0][2] == -1.0
    assert replay.calls[1][1][5] == 1.0
    assert replay.calls[1][0][0] == baseline[0].arbitration_bids[0]
    assert replay.calls[1][1][0] == baseline[1].arbitration_bids[0]


def test_ppo_trains_and_reports_only_active_bid_slot():
    env, observations, policy = _policy(seed=33, bid_only_ppo=True)
    actions, old_logp = policy.act_batch(observations)
    encoded = {i: policy.encoder.encode(observations[i]) for i in observations}
    team = np.concatenate([encoded[i] for i in range(env.cfg.N_WINGMEN)])
    active_agent, active_slot = 0, 4
    cf_adv = [[0.0] * 8 for _ in range(env.cfg.N_WINGMEN)]
    cf_mask = [[False] * 8 for _ in range(env.cfg.N_WINGMEN)]
    candidate_mask = [[False] * 8 for _ in range(env.cfg.N_WINGMEN)]
    cf_adv[active_agent][active_slot] = 1.0
    cf_mask[active_agent][active_slot] = True
    candidate_mask[active_agent][active_slot] = True
    candidate_mask[1][active_slot] = True
    item = {
        "obs": encoded,
        "team": team.astype(np.float32),
        "nominal": {i: policy.encode_action(actions[i]) for i in actions},
        "safe": {i: policy.encode_action(actions[i]) for i in actions},
        "bid_actions": {
            i: actions[i].arbitration_bids for i in actions},
        "old_logp": old_logp,
        "reward": 0.0,
        "value": 0.0,
        "done": True,
        "f_role": 0.0,
        "gear_gate_credit": [0.0] * env.cfg.N_WINGMEN,
        "cf_deltas": [0.0] * env.cfg.N_WINGMEN,
        "bid_rank": [0.0] * env.cfg.N_WINGMEN,
        "bid_mask": [False] * env.cfg.N_WINGMEN,
        "bid_cf_adv_slots": cf_adv,
        "bid_cf_mask_slots": cf_mask,
        "bid_event_mask_slots": candidate_mask,
        "action_exposure_mask": [False] * env.cfg.N_WINGMEN,
    }
    before = [p.detach().clone() for p in (
        list(policy.model.bid_head.parameters())
        + list(policy.model.bid_candidate_head.parameters()))]

    stats = policy.update([item])

    after = (list(policy.model.bid_head.parameters())
             + list(policy.model.bid_candidate_head.parameters()))
    assert stats["bid_active_slot_count"] == 1
    assert stats["cf_bid_informative_fraction"] == 0.5
    assert stats["bid_active_slot_count_by_index"] == [0, 0, 0, 0, 1, 0, 0, 0]
    assert stats["bid_gradient_norm"] > 0.0
    assert stats["bid_parameter_delta_l2"] > 0.0
    assert stats["bid_post_update_kl"] >= 0.0
    assert any(not torch.equal(old, new.detach())
               for old, new in zip(before, after))
