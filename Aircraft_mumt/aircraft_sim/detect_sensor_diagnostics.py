"""Read-only sensor and verdict instrumentation for frozen-policy diagnostics.

The deployed environment remains :class:`DetectEnv`.  This subclass adds
episode counters without changing sensor draws, observations, rewards, rules,
or verdicts.  Its truth-based diagnostics are never supplied to a policy.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields, replace
import math
from typing import Dict, FrozenSet, Optional, Tuple

from .detect_config import ZONE_TRACKING
from .detect_env import DetectEnv
from .radar import RadarState, wrap_pi


_GATE_CAUSES = (
    "outside_tracking_zone", "missing_owner_track", "stale_owner_track")


@dataclass
class DiagnosticRadarState(RadarState):
    """Count target-dwell opportunities around one unmodified radar call."""

    diagnostic_counts: Counter = field(default_factory=Counter, repr=False)
    unfinished_target_ids: FrozenSet[int] = field(
        default_factory=frozenset, repr=False)

    def scan_dwell(self, heading_rad: float, own_xy: Tuple[float, float],
                   targets: Dict[int, Tuple[float, float]], rng):
        was_on = self.on
        hits, bore = super().scan_dwell(heading_rad, own_xy, targets, rng)
        self.diagnostic_counts["dwell_calls"] += 1
        if not was_on:
            return hits, bore
        self.diagnostic_counts["active_dwell_calls"] += 1
        hit_ids = set(hits)
        half = math.radians(self.cfg.FOV_HALF_DEG)
        for tid, (tx, ty) in targets.items():
            dx, dy = tx - own_xy[0], ty - own_xy[1]
            distance = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)
            theta = abs(wrap_pi(bearing - heading_rad))
            effective_range = self.cfg.R_MAX * math.sqrt(
                max(0.0, math.cos(theta)))
            if (distance > effective_range
                    or abs(wrap_pi(bearing - bore)) > half):
                continue
            result = "hits" if tid in hit_ids else "misses"
            self.diagnostic_counts["all_opportunities"] += 1
            self.diagnostic_counts[f"all_{result}"] += 1
            if tid in self.unfinished_target_ids:
                self.diagnostic_counts["unfinished_opportunities"] += 1
                self.diagnostic_counts[f"unfinished_{result}"] += 1
        return hits, bore


class SensorDiagnosticEnv(DetectEnv):
    """Expose episode diagnostics while preserving DetectEnv transitions."""

    def reset(self, scenario: Optional[dict] = None):
        super().reset(scenario)
        self._sensor_verdict_counts = Counter()
        self._sensor_max_stable_progress = {}
        for wingman in self.wingmen:
            original = wingman.radar
            radar = DiagnosticRadarState(cfg=original.cfg)
            # Preserve every existing radar state, including future additions;
            # only the diagnostic subclass's extra counters are new.
            radar.__dict__.update(original.__dict__)
            wingman.radar = radar
        return self.observe()

    def _observation_for(self, i: int) -> dict:
        observation = super()._observation_for(i)
        wingman = observation["self"]
        if isinstance(wingman.radar, DiagnosticRadarState):
            # The base structured observation contains a BlueWingman object.
            # Present its normal local radar state without truth-based audit
            # fields, even though the encoder would ignore those fields.
            radar = RadarState(**{
                item.name: getattr(wingman.radar, item.name)
                for item in fields(RadarState)})
            observation["self"] = replace(wingman, radar=radar)
        return observation

    def _sense(self) -> None:
        unfinished = frozenset(
            tid for tid, judged in self.judged.items() if not judged)
        for wingman in self.wingmen:
            wingman.radar.unfinished_target_ids = unfinished
        super()._sense()

    def _update_verdicts(self):
        """Observe gates at the exact time the base verdict evaluates them.

        Failure causes are exclusive, ordered as the base gate expression:
        zone, missing track, then stale track.  Completion also clears progress
        in the base environment and is deliberately excluded from resets.
        """
        counts = self._sensor_verdict_counts
        red_by_uid = {target.uid: target for target in self.reds.agents}
        unfinished = {
            tid for tid in red_by_uid if not self.judged.get(tid, False)}
        counts["unfinished_target_steps"] += len(unfinished)
        for tid in unfinished:
            target = red_by_uid[tid]
            if (tid not in self.responsibility
                    and self.cfg.zone_of(target.x) == ZONE_TRACKING):
                counts["unowned_tracking_target_steps"] += 1
        facts = []
        for tid, owner in self.responsibility.items():
            if tid not in unfinished:
                continue
            target = red_by_uid[tid]
            track = self.wingmen[owner].tracks.track_of(tid)
            cause = None
            if self.cfg.zone_of(target.x) != ZONE_TRACKING:
                cause = "outside_tracking_zone"
            elif track is None:
                cause = "missing_owner_track"
            elif track.age(self.t) >= self.cfg.AGE_STD_TRACK:
                cause = "stale_owner_track"
            previous = self.stable_progress.get(tid, 0)
            counts["owned_unfinished_target_steps"] += 1
            if cause is not None:
                counts["owner_gate_failed_target_steps"] += 1
                counts[f"gate_{cause}"] += 1
            facts.append((tid, previous, cause))

        newly_judged, milestones = super()._update_verdicts()
        counts["newly_judged_targets"] += len(newly_judged)
        completed = set(newly_judged)
        for tid, previous, cause in facts:
            current = self.stable_progress.get(tid, 0)
            attained = (previous + 1 if tid in completed else current)
            self._sensor_max_stable_progress[tid] = max(
                self._sensor_max_stable_progress.get(tid, 0), previous,
                attained)
            if (previous > 0 and cause is not None and current == 0
                    and tid not in completed):
                counts["stable_progress_resets"] += 1
                counts[f"reset_{cause}"] += 1
        return newly_judged, milestones

    def sensor_diagnostics(self) -> dict:
        """Return a detached JSON-compatible snapshot of this episode."""
        totals = Counter()
        per_wingman = {}
        for wingman in self.wingmen:
            counts = wingman.radar.diagnostic_counts
            totals.update(counts)
            per_wingman[str(wingman.idx)] = self._radar_diagnostics(counts)
        verdict = self._sensor_verdict_counts
        return {
            "units": {
                "radar": "target-dwell; each platform counts separately",
                "verdict": "target-verdict-update; once per decision step",
                "reset_causes": "exclusive: zone, missing track, then age",
            },
            "radar": {
                **self._radar_diagnostics(totals),
                "per_wingman": per_wingman,
            },
            "verdict": {
                **{key: int(verdict[key]) for key in (
                    "unfinished_target_steps", "owned_unfinished_target_steps",
                    "unowned_tracking_target_steps",
                    "owner_gate_failed_target_steps", "stable_progress_resets",
                    "newly_judged_targets")},
                "owner_gate_failure_causes": {
                    key: int(verdict[f"gate_{key}"]) for key in _GATE_CAUSES},
                "reset_causes": {
                    key: int(verdict[f"reset_{key}"]) for key in _GATE_CAUSES},
                "max_stable_progress_by_target": {
                    str(tid): int(progress) for tid, progress
                    in self._sensor_max_stable_progress.items()},
            },
        }

    @staticmethod
    def _radar_diagnostics(counts: Counter) -> dict:
        return {
            "dwell_calls": int(counts["dwell_calls"]),
            "active_dwell_calls": int(counts["active_dwell_calls"]),
            "all_targets": {
                key: int(counts[f"all_{key}"])
                for key in ("opportunities", "hits", "misses")},
            "unfinished_targets": {
                key: int(counts[f"unfinished_{key}"])
                for key in ("opportunities", "hits", "misses")},
        }
