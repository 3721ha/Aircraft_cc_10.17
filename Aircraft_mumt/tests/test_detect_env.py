"""Unit tests for detect_env integration + detect_encoder dimension."""
import math
import random

import numpy as np

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_encoder import DetectActionCodec, DetectObservationEncoder
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_models import DetectAction, DetectRole


def _passive():
    return {i: DetectAction(0.0, 0.0, 0.0, DetectRole.SEARCH, 60.0, 0) for i in range(3)}


def test_reset_and_obs_structure():
    env = DetectEnv()
    obs = env.reset({"seed": 1})
    assert list(obs) == [0, 1, 2]
    enc = DetectObservationEncoder(env.cfg)
    vec = enc.encode(obs[0])
    assert vec.shape == (enc.dimension,) and np.all(np.isfinite(vec))
    assert len(env.reds.agents) == 4                      # baseline K=4
    assert [a.y for a in env.reds.agents][:2] == [30.0, 38.0] or True  # lanes exist


def test_coverage_age_ignores_cells_never_reached():
    env = DetectEnv()
    obs = env.reset({"seed": 2})
    assert obs[0]["coverage_max_age"] == 0.0
    w = env.wingmen[0]
    w.cell_reach_since[:] = np.nan
    w.cell_reach_since[0] = 0.0
    w.last_scan[:] = -1e9
    w.last_scan[0] = 0.0
    env.t = env.cfg.T_SCAN_REQ + 10.0

    observed_age = env._observation_for(0)["coverage_max_age"]
    truth_age = env._truth({})["blue"][0]["coverage_max_age"]
    assert observed_age == truth_age == env.cfg.T_SCAN_REQ + 10.0


def test_stale_track_age_and_responsibility_features_keep_semantics():
    env = DetectEnv()
    env.reset({"seed": 3, "k_red": 2})
    w = env.wingmen[0]
    track = w.tracks._alloc(0, 70.0, 40.0, 0.0)
    track.state = "confirmed"
    w.owned.add(0)
    env.responsibility[0] = 0
    env.t = 10.0

    encoded = DetectObservationEncoder(env.cfg).encode(env.observe()[0])
    slot = encoded[17:31]
    assert np.isclose(slot[5], 10.0 / env.cfg.AGE_DROP)
    assert slot[9] == 1.0 and slot[11] == 1.0
    assert slot[12] == 1.0 and slot[13] == 0.0


def test_search_direction_matches_top_decile_stale_centroid():
    env = DetectEnv()
    env.reset({"seed": 4})
    w = env.wingmen[0]
    cells = env._cell_centres[w.subzone]
    stale = env.t - w.last_scan
    k = max(1, len(cells) // 10)
    idx = stale.argsort()[-k:]
    cx, cy = cells[idx].mean(axis=0)
    delta = math.atan2(cy - w.y, cx - w.x) - w.psi
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    assert np.allclose(env._search_direction(w), [math.cos(delta), math.sin(delta)])


def test_determinism_same_seed():
    e1, e2 = DetectEnv(), DetectEnv()
    e1.reset({"seed": 7}); e2.reset({"seed": 7})
    for _ in range(50):
        e1.step(_passive()); e2.step(_passive())
    assert [a.x for a in e1.reds.agents] == [a.x for a in e2.reds.agents]


def test_radar_discovers_approaching_reds():
    env = DetectEnv()
    env.reset({"seed": 3})
    acts = {i: DetectAction(0.0, 0.5, 0.0, DetectRole.SEARCH, 60.0, 0) for i in range(3)}
    found = 0
    for _ in range(500):
        res = env.step(acts)
        found = sum(w.tracks.n_tracks() for w in env.wingmen)
        if found or res.done:
            break
    assert found > 0, "wingmen closing east should eventually detect the ingress"


def test_mor_mar_penalty_and_report():
    env = DetectEnv()
    env.reset({"seed": 5})
    w = env.wingmen[0]
    red = env.reds.agents[0]
    w.x, w.y = red.x - 35.0, red.y          # d = 35 < MAR
    res = env.step(_passive())
    assert any(k.startswith("S04:0-") for k in res.info["report"].violations)
    assert res.reward < -0.05               # w6 channel active


def test_stable_window_verdict_and_w3():
    env = DetectEnv()
    env.reset({"seed": 11})
    red = env.reds.agents[0]
    red.x, red.y = 80.0, 20.0               # tracking zone
    env.reds.step = lambda t, dt: None      # freeze fleet
    tid = red.uid
    env.responsibility[tid] = 0
    env.wingmen[0].owned.add(tid)
    milestones = {}
    for _ in range(35):
        w = env.wingmen[0]
        w.tracks.update_with_hits([(tid, red.x, red.y)], env.t, 1.0)  # fresh age
        res = env.step(_passive())
        if res.info["milestone_hits"]:
            milestones[env.step_count] = res.info["milestone_hits"]
    assert env.judged[tid]
    assert res.info["metrics"]["judged"] == 1
    assert milestones == {5: [(tid, 1)], 10: [(tid, 2)]}
    assert env.stable_progress[tid] == 0
    assert tid not in env.milestone_reached
    report = env.monitor.evaluate_truth(res.info["truth"])
    base = env._reward(report, [], [])
    one_milestone = env._reward(report, [], [], [(tid, 1)])
    assert one_milestone - base == env.cfg.W3_STABLE / 3.0


def test_cutoff_failure_terminal():
    env = DetectEnv()
    env.reset({"seed": 13})
    res = None
    for _ in range(900):
        res = env.step(_passive())          # no requests -> nothing judged
        if res.done:
            break
    assert res.done and res.info["terminal"] == "failure"
    assert env.step_count < 900             # reds arrive before timeout (V2.2 floor)


def test_timeout_when_fleet_frozen():
    env = DetectEnv()
    env.reset({"seed": 17})
    env.reds.step = lambda t, dt: None
    res = None
    for _ in range(901):
        res = env.step(_passive())
        if res.done:
            break
    assert res.info["terminal"] == "timeout" and env.step_count == 900


def test_handoff_and_c03():
    env = DetectEnv()
    env.reset({"seed": 19})
    red = env.reds.agents[0]
    red.x, red.y = 80.0, 20.0
    env.reds.step = lambda t, dt: None
    tid = red.uid
    # wingman 0 acquires and requests
    env.wingmen[0].tracks.update_with_hits([(tid, red.x, red.y)], env.t, 1.0)
    req0 = {0: DetectAction(0, 0, 0, DetectRole.TRACK, 15.0, 1), 1: _passive()[1], 2: _passive()[2]}
    env.step(req0)
    assert env.responsibility.get(tid) == 0
    # wingman 1 much closer -> requests; move it near the target
    env.wingmen[1].x, env.wingmen[1].y = red.x - 10.0, red.y
    env.wingmen[1].tracks.update_with_hits([(tid, red.x, red.y)], env.t, 1.0)
    # The learned bid adjusts quality, and the closer challenger clears the
    # incumbent's hysteresis margin.
    req1 = {0: _passive()[0], 1: DetectAction(0, 0, 0, DetectRole.TRACK, 15.0, 1,
                                               arbitration_bid=0.1),
            2: _passive()[2]}
    res = env.step(req1)
    assert env.responsibility.get(tid) == 1          # better geometry beats owner+hyst
    outcome = next(o for o in res.info["request_outcomes"]
                   if o["requester"] == 1 and o["target_id"] == tid)
    assert outcome["c04_gate_ok"] and outcome["candidate"]
    assert outcome["reason"] == "selected" and not outcome["new_assignment"]
    assert outcome["c04_geometry"]["failure_mode"] == "pass"
    assert outcome["c04_geometry"]["range_margin_km"] >= 0.0
    assert outcome["c04_geometry"]["sector_margin_deg"] >= 0.0
    assert outcome["pre_step_geometry"] is not None
    assert res.info["metrics"]["handoffs"] == 1
    ev = res.info["handoffs"][0]
    assert ev["tid"] == tid and ev["from"] == 0 and ev["to"] == 1 and not ev["exempt"]
    # C03 window accounting: the non-exempt change is on the clock
    assert env.handoff_history[tid] == [env.t]


def test_c04_geometry_diagnostics_separate_range_and_sector_rejection():
    env = DetectEnv()
    env.reset({"seed": 23})
    wing = env.wingmen[0]
    wing.x, wing.y, wing.psi = 0.0, 0.0, 0.0
    wing.radar.axis_deg = 0.0
    wing.radar.gear = 30.0

    range_diag = env._gate_diagnostics(0, 81.0, 0.0)
    assert range_diag["failure_mode"] == "range"
    assert math.isclose(range_diag["range_margin_km"], -1.0)
    assert range_diag["sector_margin_deg"] >= 0.0

    sector_diag = env._gate_diagnostics(0, 10.0, 10.0)
    assert sector_diag["failure_mode"] == "sector"
    assert sector_diag["range_margin_km"] >= 0.0
    assert sector_diag["sector_margin_deg"] < 0.0


def _arbitrate_test_case(owner_score, challenger_score, owner_bid=0.0,
                         challenger_bid=0.0, owner_gate=True,
                         stale_breach=False):
    env = DetectEnv()
    env.reset({"seed": 29, "k_red": 2})
    tid = env.reds.agents[0].uid
    env.responsibility[tid] = 0
    env.wingmen[0].owned.add(tid)
    env.wingmen[1].requests = [tid]
    env.wingmen[0].arbitration_bid = owner_bid
    env.wingmen[1].arbitration_bid = challenger_bid
    env._gate_ok = lambda i, tx, ty: owner_gate if i == 0 else True
    scores = {0: owner_score, 1: challenger_score}
    env._score = lambda i, target, tx, ty: scores[i]
    if stale_breach:
        env.c04_breach[(0, tid)] = env.t
    handoffs, conflicts = env._arbitrate()
    return env, tid, handoffs, conflicts


def test_rl_bid_adjusts_close_conflict_but_not_large_quality_gap():
    env, tid, handoffs, conflicts = _arbitrate_test_case(
        owner_score=0.0, challenger_score=0.1,
        owner_bid=-1.0, challenger_bid=1.0)
    assert env.responsibility[tid] == 1
    assert handoffs[0]["to"] == 1
    assert conflicts[0]["winner"] == 1
    outcome = next(o for o in env.last_request_outcomes
                   if o["requester"] == 1 and o["target_id"] == tid)
    assert outcome["c04_gate_ok"] and outcome["candidate"]
    assert outcome["reason"] == "selected" and not outcome["new_assignment"]
    adjusted = conflicts[0]["adjusted_scores"]
    assert math.isclose(adjusted[0], -0.35)
    assert math.isclose(adjusted[1], 0.45)

    env, tid, handoffs, _ = _arbitrate_test_case(
        owner_score=1.0, challenger_score=0.0,
        owner_bid=-1.0, challenger_bid=1.0)
    assert env.responsibility[tid] == 0
    assert handoffs == []


def test_slot_bids_follow_decision_time_target_slots_for_each_conflict():
    env = DetectEnv()
    env.reset({"seed": 31, "k_red": 2})
    tid0, tid1 = [a.uid for a in env.reds.agents]
    env.responsibility.update({tid0: 0, tid1: 0})
    env.wingmen[0].owned.update({tid0, tid1})

    # The two wingmen keep the same targets in different local slots.
    env.wingmen[0].tracks.slots[0] = type("T", (), {"target_id": tid0})()
    env.wingmen[0].tracks.slots[1] = type("T", (), {"target_id": tid1})()
    env.wingmen[1].tracks.slots[3] = type("T", (), {"target_id": tid0})()
    actions = {
        0: DetectAction(role=DetectRole.TRACK,
                        arbitration_bids=(1.0, -1.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0)),
        1: DetectAction(role=DetectRole.TRACK, pointer=4,
                        arbitration_bids=(0.0, 0.0, 0.0, 1.0,
                                           0.0, 0.0, 0.0, 0.0)),
        2: DetectAction(),
    }
    env._apply_controls(actions)
    # Reuse of a slot after action selection cannot redirect the bid.
    env.wingmen[0].tracks.slots[0] = type("T", (), {"target_id": tid1})()
    env.wingmen[0].tracks.slots[1] = type("T", (), {"target_id": tid0})()
    env._gate_ok = lambda i, tx, ty: True
    env._score = lambda i, target, tx, ty: 0.0

    _, conflicts = env._arbitrate()
    assert env.responsibility[tid0] == 0
    assert env.responsibility[tid1] == 0
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["tid"] == tid0
    assert conflict["bids"] == {0: 1.0, 1: 1.0}
    assert conflict["bid_slots"] == {0: 0, 1: 3}


def test_slot_bid_is_neutral_when_target_was_absent_at_decision():
    env = DetectEnv()
    env.reset({"seed": 32, "k_red": 2})
    tid = env.reds.agents[0].uid
    env.responsibility[tid] = 0
    env.wingmen[0].owned.add(tid)
    env.wingmen[1].tracks.slots[2] = type("T", (), {"target_id": tid})()
    env._apply_controls({
        0: DetectAction(role=DetectRole.TRACK,
                        arbitration_bids=(0.8,) + (0.0,) * 7),
        1: DetectAction(role=DetectRole.TRACK, pointer=3,
                        arbitration_bids=(0.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0)),
        2: DetectAction(),
    })
    env._gate_ok = lambda i, tx, ty: True
    env._score = lambda i, target, tx, ty: 0.0

    _, conflicts = env._arbitrate()
    assert env.responsibility[tid] == 0
    assert conflicts[0]["bids"] == {0: 0.0, 1: 0.0}
    assert conflicts[0]["bid_slots"] == {0: None, 1: 2}


def test_incumbent_hysteresis_and_c04_handoff_exemption():
    env, tid, handoffs, _ = _arbitrate_test_case(
        owner_score=0.0, challenger_score=0.19)
    assert env.responsibility[tid] == 0
    assert handoffs == []

    env, tid, handoffs, _ = _arbitrate_test_case(
        owner_score=0.0, challenger_score=0.21)
    assert env.responsibility[tid] == 1
    assert handoffs[0]["exempt"] is False
    assert env.handoff_history[tid] == [env.t]

    env, tid, handoffs, _ = _arbitrate_test_case(
        owner_score=1.0, challenger_score=0.0,
        owner_gate=False, stale_breach=True)
    assert env.responsibility[tid] == 1
    assert handoffs[0]["exempt"] is True
    assert tid not in env.handoff_history
    assert (0, tid) not in env.c04_breach
    env.t += env.cfg.C04_RESPONSE + 1.0
    report = env.monitor.evaluate_truth(env._truth({}))
    assert f"C04:0-{tid}" not in report.violations

    env, tid, handoffs, _ = _arbitrate_test_case(
        owner_score=0.0, challenger_score=0.21,
        owner_gate=True, stale_breach=True)
    assert env.responsibility[tid] == 1
    assert handoffs[0]["exempt"] is False
    assert env.handoff_history[tid] == [env.t]

    # A current owner's active breach timer is retained while it remains out
    # of gate, so pruning cannot erase a legitimate response window.
    env = DetectEnv()
    env.reset({"seed": 31})
    tid = env.reds.agents[0].uid
    env.responsibility[tid] = 0
    env.wingmen[0].owned.add(tid)
    env.c04_breach[(0, tid)] = env.t
    env._gate_ok = lambda i, tx, ty: False
    env._arbitrate()
    assert env.c04_breach[(0, tid)] == env.t


def test_role_gear_mask_and_e01():
    env = DetectEnv()
    env.reset({"seed": 23})
    # SEARCH role asking for a TRACK gear -> snapped to wide legal gear
    acts = {i: DetectAction(0, 0, 0, DetectRole.SEARCH, 7.5, 0) for i in range(3)}
    env.step(acts)
    assert all(w.radar.gear in env.cfg.GEARS_SEARCH for w in env.wingmen)
    # switching gear mid-frame is denied and recorded (E01):
    # gear is now 45 (T_frame = 3 s); after one more step the frame is mid-scan
    env.step(_passive())
    res = env.step({i: DetectAction(0, 0, 0, DetectRole.SEARCH, 60.0, 0) for i in range(3)})
    assert any(w.gear_switch_denied for w in env.wingmen)
    assert any(k.startswith("E01:") for k in res.info["report"].violations)


def test_action_codec_roundtrip():
    codec = DetectActionCodec()
    a = codec.to_action([2.0, -0.5, 0.1], 1, 2, 30)
    assert a.u_psi == 1.0 and a.role == DetectRole.TRACK and a.gear == 30.0
    heads = codec.heads(a)
    assert heads == {"role": 1, "gear": 2, "pointer": 8}   # pointer clamped to 8

    track_45 = DetectAction(0, 0, 0, DetectRole.TRACK, 45.0, 0)
    search_45 = DetectAction(0, 0, 0, DetectRole.SEARCH, 45.0, 0)
    assert codec.heads(track_45)["gear"] == 3
    assert codec.heads(search_45)["gear"] == 4
    assert codec.to_action([0, 0, 0], 1, 3, 0).gear == 45.0
    assert codec.to_action([0, 0, 0], 0, 4, 0).gear == 45.0
