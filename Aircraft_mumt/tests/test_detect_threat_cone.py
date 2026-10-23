import math

from aircraft_sim.detect_config import detect_config_profile
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_rules import DetectRuleMonitor
from aircraft_sim.detect_threat import ata_degrees, truth_in_threat_cone


def _blue(x=50.0, y=60.0):
    return {
        "x": x, "y": y, "psi": 0.0, "v": 0.2,
        "role": "SEARCH", "role_age": 10.0, "role_age_prev": 10.0,
        "role_changed": False, "gear": 60.0, "axis_deg": 0.0,
        "subzone": 1, "link_degraded": False, "uplink_loss": 0.0,
        "coverage_max_age": 0.0, "cover_area_rate": 8.0, "tracks": {},
        "pointer_target": None, "requests": [], "handoff_attempted": False,
        "gear_switch_denied": False, "owned": [],
    }


def _truth(red):
    return {
        "t": 1.0, "step": 1,
        "blue": [_blue(), _blue(y=20.0), _blue(y=100.0), _blue(x=10.0)],
        "red": [red], "responsibility": {}, "handoff_history": {},
        "c04_breach": {}, "m03_stall": {}, "raw_actions": {},
    }


def test_ata_cone_boundaries_and_unknown_heading_are_conservative():
    assert math.isclose(ata_degrees(math.pi, 80.0, 60.0, 50.0, 60.0), 0.0)
    assert truth_in_threat_cone(60.0, math.pi, 80.0, 60.0, 50.0, 60.0)
    assert truth_in_threat_cone(60.0, math.pi - math.radians(60.0),
                                80.0, 60.0, 50.0, 60.0)
    assert not truth_in_threat_cone(60.0, 0.0, 80.0, 60.0, 50.0, 60.0)
    assert truth_in_threat_cone(60.0, None, 80.0, 60.0, 50.0, 60.0)


def test_direction_gated_rules_keep_s04_s05_mutually_exclusive():
    cfg = detect_config_profile("k2_lane_center_50_threat_cone_15_40_60")
    monitor = DetectRuleMonitor(cfg)

    incoming = monitor.evaluate_truth(_truth({
        "uid": 0, "x": 80.0, "y": 60.0, "heading": math.pi,
        "judged": False,
    }))
    assert "S05:0-0" in incoming.violations
    assert "S04:0-0" not in incoming.violations

    red_line = monitor.evaluate_truth(_truth({
        "uid": 0, "x": 60.0, "y": 60.0, "heading": math.pi,
        "judged": False,
    }))
    assert "S04:0-0" in red_line.violations
    assert "S05:0-0" not in red_line.violations

    departing = monitor.evaluate_truth(_truth({
        "uid": 0, "x": 80.0, "y": 60.0, "heading": 0.0,
        "judged": False,
    }))
    assert not any(key.startswith(("S04:0-0", "S05:0-0"))
                   for key in departing.violations)


def test_cone_profile_reward_only_charges_incoming_pairs():
    cfg = detect_config_profile("k2_lane_center_50_threat_cone_15_40_60",
                                K_RED=2)
    env = DetectEnv(cfg)
    env.reset({"seed": 123})
    red = env.reds.agents[0]
    wing = env.wingmen[0]
    wing.x, wing.y, wing.v = 50.0, 60.0, 0.0
    red.x, red.y, red.heading = 80.0, 60.0, math.pi
    report_in = env.monitor.evaluate_truth(env._truth({}))
    reward_in = env._reward(report_in, [], [])

    red.heading = 0.0
    report_out = env.monitor.evaluate_truth(env._truth({}))
    reward_out = env._reward(report_out, [], [])

    expected = cfg.W5_MOR_GRAD * (cfg.MOR - 30.0) / cfg.MOR_GRADIENT_SPAN
    assert math.isclose(reward_out - reward_in, expected, rel_tol=1e-9)
