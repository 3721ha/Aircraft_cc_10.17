from types import SimpleNamespace

import pytest

from aircraft_sim.detect_mappo import DetectMAPPOConfig, DetectMAPPOPolicy


class Track:
    target_id = 7

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def extrapolate(self, _time):
        return self.x, self.y


def policy_for(track, *, geo_fraction=0.5, alignment="legacy", axis=30.0,
               gear=45.0):
    config = DetectMAPPOConfig(
        shaping_kappa=0.0, geo_shaping_kappa=0.15,
        geo_shaping_range_fraction=geo_fraction,
        stable_shaping_kappa=1.0,
        stable_alignment_geometry=alignment)
    wing = SimpleNamespace(
        x=0.0, y=0.0, psi=0.0,
        radar=SimpleNamespace(axis_deg=axis, gear=gear),
        tracks=SimpleNamespace(slots=[track]))
    sim = SimpleNamespace(
        C04_RANGE=80.0, STABLE_WINDOW=8, SECTOR_AXIS_MAX_DEG=60.0,
        zone_of=lambda _x: "tracking")
    env = SimpleNamespace(
        cfg=sim, t=0.0, wingmen=[wing], responsibility={track.target_id: 0},
        stable_progress={})
    return SimpleNamespace(config=config, env=env)


def test_geo_shaping_support_defaults_to_legacy_and_can_reach_c04_gate():
    track = Track(60.0, 0.0)
    legacy = policy_for(track, geo_fraction=0.5)
    c04_aligned = policy_for(track, geo_fraction=1.0)

    assert DetectMAPPOPolicy._track_potential(legacy) == pytest.approx(0.0)
    assert DetectMAPPOPolicy._track_potential(c04_aligned) == pytest.approx(0.0375)


def test_alignment_shaping_can_use_the_c04_usable_sector_width():
    track = Track(10.0 * 0.6427876097, 10.0 * 0.7660444431)
    legacy = policy_for(track, alignment="legacy")
    c04_aligned = policy_for(track, alignment="c04_usable_sector")

    assert DetectMAPPOPolicy._stable_potential(legacy) == pytest.approx(0.0)
    assert DetectMAPPOPolicy._stable_potential(c04_aligned) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("fraction", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_geo_shaping_support_is_rejected(fraction):
    with pytest.raises(ValueError, match="geo_shaping_range_fraction"):
        DetectMAPPOConfig(geo_shaping_range_fraction=fraction)


def test_unknown_alignment_geometry_is_rejected():
    with pytest.raises(ValueError, match="stable_alignment_geometry"):
        DetectMAPPOConfig(stable_alignment_geometry="other")
