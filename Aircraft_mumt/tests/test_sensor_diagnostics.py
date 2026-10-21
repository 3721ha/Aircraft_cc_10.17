"""Sensor diagnostics must observe transitions without changing them."""
from dataclasses import replace
import math
import random

import numpy as np
import pytest

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_encoder import DetectObservationEncoder
from aircraft_sim.detect_models import DetectAction, DetectRole
from aircraft_sim.detect_sensor_diagnostics import (
    DiagnosticRadarState, SensorDiagnosticEnv)


def _config(profile="bernoulli"):
    return replace(DetectSimConfig(), RADAR_HIT_MODEL=profile,
                   STABLE_WINDOW=8, AGE_STD_TRACK=10.0)


def _passive():
    return {i: DetectAction(role=DetectRole.SEARCH, gear=60.0)
            for i in range(3)}


@pytest.mark.parametrize("profile", ["bernoulli", "deterministic_beam"])
def test_diagnostics_preserve_observations_transitions_and_rng(profile):
    base = DetectEnv(_config(profile))
    audited = SensorDiagnosticEnv(_config(profile))
    for env in (base, audited):
        env.reset({"seed": 290001, "k_red": 2})
        # Place a moving target within sensor reach to exercise actual draws.
        env.reds.agents[0].x, env.reds.agents[0].y = 85.0, 20.0
        env.reds.agents[1].x, env.reds.agents[1].y = 90.0, 100.0
    encoder = DetectObservationEncoder(base.cfg)
    for _ in range(35):
        left, right = base.step(_passive()), audited.step(_passive())
        assert left.reward == right.reward
        assert left.done == right.done
        assert left.info == right.info
        for i in left.observations:
            np.testing.assert_array_equal(
                encoder.encode(left.observations[i]),
                encoder.encode(right.observations[i]))
            np.testing.assert_equal(
                {key: value for key, value in left.observations[i].items()
                 if key != "self"},
                {key: value for key, value in right.observations[i].items()
                 if key != "self"})
            normal_radar = right.observations[i]["self"].radar
            assert normal_radar.__dict__ == base.wingmen[i].radar.__dict__
            assert not isinstance(normal_radar, DiagnosticRadarState)
        assert base.rng.getstate() == audited.rng.getstate()
        assert [w.tracks.slots for w in base.wingmen] == [
            w.tracks.slots for w in audited.wingmen]
        for old, new in zip(base.wingmen, audited.wingmen):
            for key, value in old.radar.__dict__.items():
                assert new.radar.__dict__[key] == value
    radar = audited.sensor_diagnostics()["radar"]
    assert radar["dwell_calls"] == 35 * 2 * 3
    assert radar["all_targets"]["opportunities"] > 0
    assert radar["all_targets"]["opportunities"] == (
        radar["all_targets"]["hits"] + radar["all_targets"]["misses"])
    if profile == "deterministic_beam":
        assert radar["all_targets"]["misses"] == 0


@pytest.mark.parametrize("profile, expected_hits", [
    ("bernoulli", 0), ("deterministic_beam", 1)])
def test_counters_measure_current_beam_and_attenuated_range(profile, expected_hits):
    radar = DiagnosticRadarState(_config(profile), gear=7.5)
    radar.unfinished_target_ids = frozenset({0})
    targets = {
        0: (60.0, 0.0),
        1: (60.0 * math.cos(math.radians(30)),
            60.0 * math.sin(math.radians(30))),
        2: (81.0, 0.0),
    }
    rng = random.Random(0)  # First draw 0.844... misses Pd=0.6.
    control_rng = random.Random(0)
    control_rng.random()
    hits, _ = radar.scan_dwell(0.0, (0.0, 0.0), targets, rng)
    assert len(hits) == expected_hits
    assert rng.getstate() == control_rng.getstate()
    assert radar.diagnostic_counts["all_opportunities"] == 1
    assert radar.diagnostic_counts["all_hits"] == expected_hits
    assert radar.diagnostic_counts["unfinished_opportunities"] == 1
    assert radar.diagnostic_counts["unfinished_misses"] == 1 - expected_hits
    radar.on = False
    before = rng.getstate()
    assert radar.scan_dwell(0.0, (0.0, 0.0), targets, rng)[0] == []
    assert rng.getstate() == before
    assert radar.diagnostic_counts["dwell_calls"] == 2
    assert radar.diagnostic_counts["active_dwell_calls"] == 1
    assert radar.diagnostic_counts["all_opportunities"] == 1


def test_a_deterministic_hit_does_not_complete_unassigned_target():
    env = SensorDiagnosticEnv(_config("deterministic_beam"))
    env.reset({"seed": 290002, "k_red": 1})
    target = env.reds.agents[0]
    target.x, target.y = 80.0, 20.0
    env.reds.step = lambda t, dt: None
    env.wingmen[0].radar.gear = 7.5
    for _ in range(env.cfg.STABLE_WINDOW + 1):
        env.step(_passive())
    assert env.sensor_diagnostics()["radar"]["unfinished_targets"]["hits"] > 0
    assert not env.judged[target.uid]
    assert target.uid not in env.responsibility
    assert env.stable_progress.get(target.uid, 0) == 0
    assert env.terminal != "success"


@pytest.mark.parametrize("cause", [
    "outside_tracking_zone", "missing_owner_track", "stale_owner_track"])
def test_verdict_counts_actual_progress_resets_with_exclusive_causes(cause):
    env = SensorDiagnosticEnv(_config())
    env.reset({"seed": 290003, "k_red": 1})
    target = env.reds.agents[0]
    tid = target.uid
    target.x, target.y = 80.0, 20.0
    env.responsibility[tid] = 0
    env.wingmen[0].owned.add(tid)
    env.t = 20.0
    if cause != "missing_owner_track":
        # At the exact age threshold, strict age < A fails.
        stamp = env.t - env.cfg.AGE_STD_TRACK
        env.wingmen[0].tracks.update_with_hits(
            [(tid, target.x, target.y)], stamp, 1.0)
    if cause == "outside_tracking_zone":
        target.x = env.cfg.X_TRACK + 1.0
    env.stable_progress[tid] = 3
    assert env._update_verdicts()[0] == []
    assert env.stable_progress[tid] == 0
    verdict = env.sensor_diagnostics()["verdict"]
    assert verdict["stable_progress_resets"] == 1
    assert verdict["reset_causes"][cause] == 1
    assert sum(verdict["reset_causes"].values()) == 1
    assert sum(verdict["owner_gate_failure_causes"].values()) == 1
    # A failed gate with zero progress is not another reset.
    env._update_verdicts()
    assert env.sensor_diagnostics()["verdict"]["stable_progress_resets"] == 1


def test_completion_clear_is_not_a_reset_and_reset_clears_diagnostics():
    env = SensorDiagnosticEnv(_config())
    env.reset({"seed": 290004, "k_red": 1})
    target = env.reds.agents[0]
    target.x, target.y = 80.0, 20.0
    tid = target.uid
    env.responsibility[tid] = 0
    env.wingmen[0].tracks.update_with_hits(
        [(tid, target.x, target.y)], env.t, 1.0)
    env.stable_progress[tid] = env.cfg.STABLE_WINDOW - 1
    newly_judged, _ = env._update_verdicts()
    assert newly_judged == [tid]
    assert env.stable_progress[tid] == 0
    verdict = env.sensor_diagnostics()["verdict"]
    assert verdict["stable_progress_resets"] == 0
    assert verdict["newly_judged_targets"] == 1
    assert verdict["max_stable_progress_by_target"][str(tid)] == 8
    env.reset({"seed": 290004, "k_red": 1})
    counters = env.sensor_diagnostics()
    assert counters["verdict"]["newly_judged_targets"] == 0
    assert counters["verdict"]["max_stable_progress_by_target"] == {}
    assert counters["radar"]["dwell_calls"] == 0
