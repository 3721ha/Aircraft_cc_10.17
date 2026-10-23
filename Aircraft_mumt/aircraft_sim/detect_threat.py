"""Shared threat-angle calculations for referee, observation features and shield."""
from __future__ import annotations

import math


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ata_degrees(red_heading: float | None, red_x: float, red_y: float,
                blue_x: float, blue_y: float) -> float | None:
    """Absolute angle from red velocity direction to the red-to-blue bearing.

    A missing heading leaves ATA undefined; callers handle that conservatively
    as inside the cone when a cone gate is enabled.
    """
    if red_heading is None:
        return None
    bearing = math.atan2(blue_y - red_y, blue_x - red_x)
    return abs(math.degrees(wrap_pi(bearing - red_heading)))


def truth_in_threat_cone(half_angle_deg: float | None, red_heading: float | None,
                         red_x: float, red_y: float,
                         blue_x: float, blue_y: float) -> bool:
    """Return whether a true red-to-blue ATA passes the configured cone gate."""
    if half_angle_deg is None:
        return True
    angle = ata_degrees(red_heading, red_x, red_y, blue_x, blue_y)
    # Unknown true heading is treated as threatening rather than an escape.
    return angle is None or angle <= half_angle_deg + 1e-9


def estimate_in_threat_cone(half_angle_deg: float | None,
                            red_vx: float, red_vy: float,
                            red_x: float, red_y: float,
                            blue_x: float, blue_y: float) -> bool:
    """Track-file version of the cone gate; unknown/zero velocity is unsafe."""
    if half_angle_deg is None:
        return True
    if math.hypot(red_vx, red_vy) <= 1e-6:
        return True
    heading = math.atan2(red_vy, red_vx)
    angle = ata_degrees(heading, red_x, red_y, blue_x, blue_y)
    return angle is None or angle <= half_angle_deg + 1e-9
