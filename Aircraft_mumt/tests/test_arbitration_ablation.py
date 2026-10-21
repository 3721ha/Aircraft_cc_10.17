from dataclasses import fields
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _diagnose_arbitration_ablation import apply_arbitration_treatment
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_models import DetectAction, DetectRole


def test_fixed_score_treatment_changes_only_arbitration_fields():
    actions = {
        0: DetectAction(0.2, -0.3, 0.4, DetectRole.TRACK, 15.0, 2,
                        -0.8, (-0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), True),
        1: DetectAction(-0.1, 0.5, -0.2, DetectRole.SEARCH, 60.0, 0,
                        0.9, (0.9, 0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7), True),
    }

    fixed = apply_arbitration_treatment(actions, direct_rl=False)
    arbitration_fields = {"arbitration_bid", "arbitration_bids",
                          "rl_direct_arbitration"}
    for agent_id, action in actions.items():
        before = {item.name: getattr(action, item.name) for item in fields(action)}
        after = {item.name: getattr(fixed[agent_id], item.name)
                 for item in fields(fixed[agent_id])}
        assert {key: value for key, value in before.items()
                if key not in arbitration_fields} == {
                    key: value for key, value in after.items()
                    if key not in arbitration_fields}
        assert fixed[agent_id].arbitration_bid == 0.0
        assert fixed[agent_id].arbitration_bids == (0.0,) * 8
        assert not fixed[agent_id].rl_direct_arbitration


def _resolve(actions):
    env = DetectEnv()
    env.reset({"seed": 29, "k_red": 2})
    tid = env.reds.agents[0].uid
    env.responsibility[tid] = 0
    env.wingmen[0].owned.add(tid)
    env.wingmen[1].requests = [tid]
    for agent_id in (0, 1):
        action = actions[agent_id]
        wingman = env.wingmen[agent_id]
        wingman.arbitration_bid = action.arbitration_bid
        wingman.arbitration_bids = action.arbitration_bids
        wingman.rl_direct_arbitration = action.rl_direct_arbitration
    env._gate_ok = lambda _i, _tx, _ty: True
    env._score = lambda _i, _target, _tx, _ty: 0.0
    _, conflicts = env._arbitrate()
    return env, tid, conflicts


def test_fixed_score_ignores_bids_while_direct_rl_uses_them():
    actions = {
        0: DetectAction(arbitration_bid=-1.0, rl_direct_arbitration=True),
        1: DetectAction(arbitration_bid=1.0, rl_direct_arbitration=True),
    }

    direct_env, direct_tid, direct_events = _resolve(
        apply_arbitration_treatment(actions, direct_rl=True))
    fixed_env, fixed_tid, fixed_events = _resolve(
        apply_arbitration_treatment(actions, direct_rl=False))

    assert direct_env.responsibility[direct_tid] == 1
    assert fixed_env.responsibility[fixed_tid] == 0
    assert direct_events[0]["arbitration_mode"] == "rl_direct"
    assert direct_events[0]["bids"] == {0: -1.0, 1: 1.0}
    assert fixed_events[0]["arbitration_mode"] == "fixed_score_plus_bid"
    assert fixed_events[0]["bids"] == {0: 0.0, 1: 0.0}
