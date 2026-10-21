"""Single-factor ideal-hit profile: preserve geometry, timing and RNG draws."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
import math
import random

import pytest

from aircraft_sim.detect_config import DetectSimConfig, RADAR_HIT_MODELS
from aircraft_sim.radar import RadarState


def _target(distance, bearing_deg):
    bearing = math.radians(bearing_deg)
    return distance * math.cos(bearing), distance * math.sin(bearing)


def _ideal(**kwargs):
    return DetectSimConfig(RADAR_HIT_MODEL="deterministic_beam", **kwargs)


class CountingRng:
    def __init__(self, seed=37):
        self.rng = random.Random(seed)
        self.draws = 0

    def random(self):
        self.draws += 1
        return self.rng.random()


def test_profile_is_explicit_frozen_serializable_and_validated():
    baseline = DetectSimConfig()
    assert baseline.RADAR_HIT_MODEL == "bernoulli"
    ideal = replace(baseline, RADAR_HIT_MODEL="deterministic_beam")
    old_dict, ideal_dict = asdict(baseline), asdict(ideal)
    assert {key for key in old_dict if old_dict[key] != ideal_dict[key]} == {"RADAR_HIT_MODEL"}
    assert json.loads(json.dumps(ideal_dict))["RADAR_HIT_MODEL"] == "deterministic_beam"
    assert RADAR_HIT_MODELS == ("bernoulli", "deterministic_beam")
    with pytest.raises(FrozenInstanceError):
        ideal.RADAR_HIT_MODEL = "bernoulli"
    for invalid in ("", "deterministic", "BERNOULLI", None):
        with pytest.raises(ValueError, match="RADAR_HIT_MODEL"):
            replace(baseline, RADAR_HIT_MODEL=invalid)


def test_ideal_hits_actual_beam_not_whole_reachable_sector():
    radar = RadarState(_ideal(), gear=60.0)
    # The first beam is centred at -52.5 degrees; +50 is reachable but not swept.
    targets = {1: _target(30.0, -52.5), 2: _target(30.0, 50.0)}
    rng = CountingRng()
    hits, boresight = radar.scan_dwell(0.0, (0.0, 0.0), targets, rng)
    assert hits == [1]
    assert math.degrees(boresight) == pytest.approx(-52.5)
    assert rng.draws == 1
    for _ in range(6):
        assert radar.scan_dwell(0.0, (0.0, 0.0), targets, rng)[0] == []
    assert radar.scan_dwell(0.0, (0.0, 0.0), targets, rng)[0] == [2]
    assert rng.draws == 2


def test_ideal_keeps_attenuated_range_and_raw_pd_helper():
    cfg = _ideal()
    radar = RadarState(cfg, gear=7.5, axis_deg=45.0)
    effective_range = cfg.R_MAX * math.sqrt(math.cos(math.radians(45.0)))
    targets = {1: _target(effective_range - 0.01, 45.0),
               2: _target(effective_range + 0.01, 45.0),
               3: _target(cfg.R_MAX + 0.01, 45.0)}
    rng = CountingRng()
    assert radar.scan_dwell(0.0, (0.0, 0.0), targets, rng)[0] == [1]
    assert rng.draws == 1
    assert radar.pd(effective_range - 0.01, math.radians(45.0)) == 0.3
    assert radar.pd(effective_range + 0.01, math.radians(45.0)) == 0.0


@pytest.mark.parametrize("profile", RADAR_HIT_MODELS)
def test_radar_off_does_not_scan_or_consume_rng(profile):
    radar = RadarState(DetectSimConfig(RADAR_HIT_MODEL=profile), gear=60.0, on=False)
    rng = CountingRng()
    assert radar.scan_step(0.0, (0.0, 0.0), {1: _target(20.0, -52.5)}, rng) == []
    assert (radar.phase, radar.frame_progress, rng.draws) == (0, 0.0, 0)


@pytest.mark.parametrize("gear,first_bearing,n", [
    (7.5, 0.0, 1), (15.0, -7.5, 2), (30.0, -22.5, 4),
    (45.0, -37.5, 6), (60.0, -52.5, 8),
])
def test_ideal_preserves_each_gear_revisit_and_switch_timing(gear, first_bearing, n):
    radar = RadarState(_ideal(), gear=gear)
    radar.last_switch_step = 0
    rng = CountingRng()
    targets = {1: _target(20.0, first_bearing)}
    hit_slots = []
    for slot in range(2 * n):
        hits, _ = radar.scan_dwell(0.0, (0.0, 0.0), targets, rng)
        if hits:
            hit_slots.append(slot)
        # A completed frame boundary, rather than each hit, permits switching.
        assert radar.switch_ready() == ((slot + 1) % n == 0)
    assert hit_slots == [0, n]
    assert (radar.phase, radar.frame_progress) == (0, 2 * radar.frame_period())
    assert rng.draws == 2
    new_gear = 15.0 if gear != 15.0 else 30.0
    assert radar.switch_gear(new_gear, step=10)
    assert (radar.phase, radar.frame_progress) == (0, 0.0)


def test_scan_step_uses_same_two_dwells_and_retains_duplicate_hits():
    targets = {1: _target(70.0, 3.0)}
    batched = RadarState(_ideal(), gear=7.5)
    manual = RadarState(_ideal(), gear=7.5)
    rng_batch, rng_manual = CountingRng(), CountingRng()
    actual = batched.scan_step(0.0, (0.0, 0.0), targets, rng_batch)
    expected = []
    for _ in range(2):
        expected.extend(manual.scan_dwell(0.0, (0.0, 0.0), targets, rng_manual)[0])
    assert actual == expected == [1, 1]
    assert (batched.phase, batched.frame_progress) == (manual.phase, manual.frame_progress)
    assert rng_batch.rng.getstate() == rng_manual.rng.getstate()


def test_profiles_keep_rng_aligned_for_identical_geometry_even_when_pd_zero():
    base_cfg = DetectSimConfig(PD_TIERS=((1.0, 0.0),))
    ordinary = RadarState(base_cfg, gear=30.0)
    ideal = RadarState(replace(base_cfg, RADAR_HIT_MODEL="deterministic_beam"), gear=30.0)
    targets = {1: _target(30.0, -22.5), 2: _target(70.0, 22.5),
               3: _target(90.0, -22.5), 4: _target(10.0, 90.0)}
    rng_base, rng_ideal = CountingRng(), CountingRng()
    ideal_hits = []
    for _ in range(8):
        assert ordinary.scan_dwell(0.0, (0.0, 0.0), targets, rng_base)[0] == []
        ideal_hits.extend(ideal.scan_dwell(0.0, (0.0, 0.0), targets, rng_ideal)[0])
        assert rng_base.rng.getstate() == rng_ideal.rng.getstate()
        assert ordinary.phase == ideal.phase
        assert ordinary.frame_progress == ideal.frame_progress
    assert ideal_hits == [1, 2, 1, 2]
    assert rng_base.draws == rng_ideal.draws == 4


def test_bernoulli_default_matches_prechange_fixed_trajectory_and_rng():
    # Captured from the unmodified radar before adding profiles. This guards
    # both legacy hit output and the RNG stream used by subsequent link draws.
    radar = RadarState(DetectSimConfig(), gear=30.0, axis_deg=5.0)
    explicit = RadarState(DetectSimConfig(RADAR_HIT_MODEL="bernoulli"), gear=30.0, axis_deg=5.0)
    rng, rng_explicit = random.Random(271828), random.Random(271828)
    targets = {tid: _target(distance, angle) for tid, distance, angle in (
        (1, 30.0, -12.0), (2, 50.0, 10.0), (3, 70.0, 27.0), (4, 81.0, 5.0),
    )}
    actual = []
    for step in range(12):
        heading = math.radians(step * 2.0)
        hits = radar.scan_step(heading, (0.0, 0.0), targets, rng)
        assert hits == explicit.scan_step(heading, (0.0, 0.0), targets, rng_explicit)
        actual.append(hits)
    assert actual == [[1], [], [1], [], [1], [3], [1, 2], [3], [], [], [], []]
    assert (radar.phase, radar.frame_progress) == (0, 12.0)
    assert rng.getstate() == rng_explicit.getstate()
    assert rng.random() == 0.08972017422622869
