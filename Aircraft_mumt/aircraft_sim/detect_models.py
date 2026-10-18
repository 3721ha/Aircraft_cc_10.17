"""Action / role data model for the MUMT cooperative-detection scenario (V2.4).

Separate from legacy ``models.py`` (which stays untouched): the new scenario
uses a hybrid action space -- 3 continuous controls + 3 factorised discrete
heads (plan §7.2).  Units follow ``detect_config``: km, s, deg for gears,
rad for headings.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


DETECT_BID_SLOTS = 8


class DetectRole(str, Enum):
    SEARCH = "SEARCH"    # 搜索巡逻: wide gears {45, 60}
    TRACK = "TRACK"      # 责任精跟: gears {7.5, 15, 30, 45}


@dataclass
class DetectAction:
    """One wingman decision (plan §7.2).

    Continuous heads are raw policy outputs; ``clipped`` enforces the S01
    envelope (|u| <= 1).  Discrete heads:
      role    -- SEARCH / TRACK
      gear    -- sector half-angle in deg, one of the five gears
      pointer -- 0 = no target, 1..TRACK_SLOTS = own track-slot index;
                 the responsibility-request carrier for TRACK (plan §7.2).
    """

    u_psi: float = 0.0
    u_v: float = 0.0
    u_xi: float = 0.0
    role: DetectRole = DetectRole.SEARCH
    gear: float = 60.0
    pointer: int = 0
    arbitration_bid: float = 0.0
    arbitration_bids: Optional[Tuple[float, ...]] = None
    rl_direct_arbitration: bool = False

    def clipped(self) -> "DetectAction":
        c = lambda v: max(-1.0, min(1.0, float(v)))
        bids = self.arbitration_bids
        if bids is not None:
            if len(bids) != DETECT_BID_SLOTS:
                raise ValueError(
                    f"arbitration_bids must contain {DETECT_BID_SLOTS} values")
            bids = tuple(c(v) for v in bids)
        return DetectAction(c(self.u_psi), c(self.u_v), c(self.u_xi),
                            self.role, self.gear, int(self.pointer),
                            c(self.arbitration_bid), bids,
                            bool(self.rl_direct_arbitration))
