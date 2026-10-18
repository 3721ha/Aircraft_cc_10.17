"""Observation encoder (169-dim, plan V2.4 §7.1 + CR-MAPPO §B) + action codec (§7.2).

Encoding discipline (V1.23, plan §7.1 preamble): circular quantities as
sin/cos, geometry relative to own ship, zero absolute coordinates.

Block layout (total 169 = 17 + 8*14 + 2*6 + 4 + 6 + 3 + 2 + 13):

    [0:17)    own state: boundary d/40 + inward-normal sin/cos (3),
              sin psi / cos psi (2), V/(0.95 Ma) (1), gear one-hot (5),
              xi/60 (1), role one-hot (2), role_age/T_dwell (1),
              radar on (1), radiate duty window (1, constant 1.0 in V2.4)
    [17:129)  own track cluster, 8 slots x 14: the 12 track features plus
              own-responsibility and teammate-responsibility flags
    [129:141) teammates x2: dx/R, dy/R, sin psi, cos psi, V/(0.95 Ma),
              has-pointer flag
    [141:145) manned: dx/R, dy/R, V/(0.95 Ma), link state
    [145:151) assignment table: owned/8, subzone one-hot (3),
              coverage margin (T_scan_req - t_since)/T_scan_req, table age/2
    [151:154) area posture: detection-zone est/16, tracking-zone est/16,
              judged/16
    [154:156) SEARCH target direction in local forward/left axes
    [156:169) conflict-regime features: 13 soft proxies for the 13 rule-conflict
              pairs (DETECT_CONFLICT_PAIRS).  Each ∈ [0,1]; computed from local
              observations only (own track file, own state, teammate kinematics,
              mission table) — no truth data.  See _conflict_features().
"""
import math
from typing import Dict, List

import numpy as np

from .detect_config import A_MACH_KMS, DetectSimConfig
from .detect_models import DETECT_BID_SLOTS, DetectAction, DetectRole

GEAR_ORDER = (7.5, 15.0, 30.0, 45.0, 60.0)
# Policy slots duplicate 45 degrees so each role has a contiguous legal range.
POLICY_GEAR_ORDER = (7.5, 15.0, 30.0, 45.0, 45.0, 60.0)
V_NORM = 0.95 * A_MACH_KMS
AGE_DOWNWEIGHT = 0.5          # age > 8 s feature scaling (plan §7.1, V1.23)


class DetectObservationEncoder:
    dimension = 169

    def __init__(self, cfg: DetectSimConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _conflict_features(self, obs: dict) -> List[float]:
        """13 soft conflict-regime proxies for DETECT_CONFLICT_PAIRS (CR-MAPPO §B).

        Strictly local-observable: own track file, own state, teammate
        kinematics, mission table.  No truth data — red positions are never
        accessed.  Each feature ∈ [0, 1]; higher = conflict more acute.

        Index → pair (matches DETECT_CONFLICT_PAIRS order in detect_rules):
          0  S04-M02  MAR approach pressure (inflated estimate)
          1  S05-M02  fraction of owned tracks in MOR band
          2  C01-M02  ownership saturation
          3  C03-M02  recent-handoff gating (low role_age = C03 binding)
          4  C02-M01  coverage-update budget conflict
          5  M01-M02  dual-zone demand
          6  S02-M02  teammate proximity pressure
          7  S03-M02  boundary proximity pressure
          8  I02-M02  link degradation active
          9  I03-M02  stale owned track pressure
         10  M04-S04  endgame MAR-press fraction
         11  C01-C03  over-owned AND recently switched (product)
         12  M05-E03  forced ownership vs anti-loiter (product)
        """
        cfg = self.cfg
        w = obs["self"]             # BlueWingman (local state)
        ms = obs["mission"]
        cov_age = obs["coverage_max_age"]
        t_now = ms["t"]

        n_owned = len(w.owned)
        n_det   = float(ms["n_detect_est"])
        n_trk   = float(ms["n_track_est"])
        n_jud   = float(ms["n_judged"])

        # owned-track geometry from own track file (local, not truth)
        min_deff       = math.inf
        max_owned_age  = 0.0
        n_mor_band     = 0
        for tr in w.tracks.slots:
            if tr is None or tr.target_id not in w.owned:
                continue
            ex, ey    = tr.extrapolate(t_now)
            d_est     = math.hypot(ex - w.x, ey - w.y)
            d_eff     = d_est - cfg.K_SIGMA * tr.sigma
            min_deff  = min(min_deff, d_eff)
            age_n     = min(1.0, tr.age(t_now) / cfg.AGE_DROP)
            max_owned_age = max(max_owned_age, age_n)
            if cfg.MAR <= d_est < cfg.MOR:
                n_mor_band += 1

        # cover-area rate from e03 sliding window (own sensor log)
        e_area = sum(a for a, _ in w.e03_window)
        e_dist = sum(d for _, d in w.e03_window)
        cover_rate = e_area / e_dist if e_dist > 1e-6 else cfg.E03_AREA_PER_KM

        # nearest teammate range (from link-delivered kinematics)
        min_tm = min(
            (math.hypot(tm["dx"], tm["dy"]) for tm in obs["teammates"].values()),
            default=cfg.R_MAX)

        # boundary distance (same calc as encode block-0)
        d_bound = min(w.x, cfg.L_X - w.x, w.y, cfg.L_Y - w.y)

        # f0  S04-M02: MAR approach pressure (d_eff < MAR → conflict acute)
        f0 = 0.0 if math.isinf(min_deff) else max(0.0, min(1.0, 1.0 - min_deff / cfg.MAR))

        # f1  S05-M02: fraction of owned tracks estimated in MOR band
        f1 = min(1.0, n_mor_band / max(1, n_owned)) if n_owned > 0 else 0.0

        # f2  C01-M02: ownership saturation
        f2 = min(1.0, n_owned / cfg.TRACK_SLOTS)

        # f3  C03-M02: handoff-gating pressure (role_age low = recently switched)
        f3 = max(0.0, 1.0 - min(1.0, w.role_age / max(1.0, cfg.HANDOFF_MIN_INTERVAL)))

        # f4  C02-M01: coverage-update budget conflict (high track load, low margin)
        cov_margin = max(0.0, min(1.0, (cfg.T_SCAN_REQ - cov_age) / cfg.T_SCAN_REQ))
        f4 = min(1.0, n_trk / 8.0) * (1.0 - cov_margin)

        # f5  M01-M02: dual-zone demand (both zones busy)
        f5 = min(1.0, (n_det + n_trk) / 8.0)

        # f6  S02-M02: teammate proximity (separation constraint binding)
        f6 = max(0.0, min(1.0, 1.0 - min_tm / (3.0 * cfg.D_SAFE)))

        # f7  S03-M02: boundary proximity (30 km comfortable margin)
        f7 = max(0.0, min(1.0, 1.0 - d_bound / 30.0))

        # f8  I02-M02: link degradation (honest pointer report blocked)
        f8 = 1.0 if w.link_degraded else 0.0

        # f9  I03-M02: stale owned track (update obligation unmet)
        f9 = max_owned_age if n_owned > 0 else 0.0

        # f10 M04-S04: endgame — judged fraction (approaching terminal)
        f10 = min(1.0, n_jud / max(1.0, n_det + n_trk + n_jud))

        # f11 C01-C03: misassignment fix gated (over-owned AND recently switched)
        over = min(1.0, max(0.0, n_owned - 1.0) / max(1, cfg.TRACK_SLOTS - 1))
        f11 = over * f3

        # f12 M05-E03: forced ownership vs anti-loiter (both pressures active)
        ownership_p = min(1.0, n_owned / max(1, cfg.TRACK_SLOTS))
        loiter_p    = max(0.0, 1.0 - min(1.0, cover_rate / max(1e-9, cfg.E03_AREA_PER_KM)))
        f12 = ownership_p * loiter_p

        return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

    # ------------------------------------------------------------------
    def encode(self, obs: dict) -> np.ndarray:
        cfg = self.cfg
        w = obs["self"]                    # BlueWingman
        out: List[float] = []

        # -- own state (17) --
        d_bound = min(w.x, cfg.L_X - w.x, w.y, cfg.L_Y - w.y)
        # inward normal of the nearest boundary (absolute angle)
        candidates = [(w.x, math.pi), (cfg.L_X - w.x, 0.0),
                      (w.y, math.pi / 2), (cfg.L_Y - w.y, -math.pi / 2)]
        _, n_ang = min(candidates, key=lambda c: c[0])
        rel = n_ang - w.psi
        out += [min(1.0, d_bound / 40.0), math.sin(rel), math.cos(rel),
                math.sin(w.psi), math.cos(w.psi),
                min(1.0, w.v / V_NORM)]
        out += [1.0 if abs(w.radar.gear - g) < 1e-6 else 0.0 for g in GEAR_ORDER]
        out += [w.radar.axis_deg / cfg.SECTOR_AXIS_MAX_DEG]
        out += [1.0 if w.role == DetectRole.SEARCH else 0.0,
                1.0 if w.role == DetectRole.TRACK else 0.0]
        out += [min(1.0, w.role_age / cfg.T_DWELL), 1.0, 1.0]

        # -- own track cluster (8 x 14): track state + responsibility --
        feats = obs["tracks"]
        responsibility = obs["mission"]["responsibility"]
        for slot, track in zip(feats, w.tracks.slots):
            if slot[11] > 0 and slot[5] * cfg.AGE_DROP > cfg.AGE_DEGRADE:
                adjusted = list(slot)
                adjusted[:5] = [v * AGE_DOWNWEIGHT for v in adjusted[:5]]
                out += adjusted
            else:
                out += slot
            owner = responsibility.get(track.target_id) if track is not None else None
            out += [1.0 if owner == w.idx else 0.0,
                    1.0 if owner is not None and owner != w.idx else 0.0]

        # -- teammates (2 x 6) --
        for k in sorted(obs["teammates"]):
            tm = obs["teammates"][k]
            out += [tm["dx"] / cfg.R_MAX, tm["dy"] / cfg.R_MAX,
                    math.sin(tm["psi"]), math.cos(tm["psi"]),
                    min(1.0, tm["v"] / V_NORM), tm["has_pointer"]]

        # -- manned (4) --
        mn = obs["manned"]
        out += [mn["dx"] / cfg.R_MAX, mn["dy"] / cfg.R_MAX,
                min(1.0, mn["v"] / V_NORM), mn["link"]]

        # -- assignment table (6) --
        cov_age = obs["coverage_max_age"]
        out += [len(w.owned) / cfg.TRACK_SLOTS]
        out += [1.0 if w.subzone == s else 0.0 for s in range(cfg.N_SUBZONES)]
        out += [max(-2.0, (cfg.T_SCAN_REQ - cov_age) / cfg.T_SCAN_REQ),
                obs["mission"]["table_age"] / 2.0]

        # -- area posture (3) --
        ms = obs["mission"]
        out += [ms["n_detect_est"] / 16.0, ms["n_track_est"] / 16.0,
                ms["n_judged"] / 16.0]

        # -- SEARCH teacher target direction in local forward/left axes (2) --
        out += obs["search_direction"]

        # -- conflict-regime features (13, CR-MAPPO §B) --
        out += self._conflict_features(obs)

        vec = np.asarray(out, dtype=np.float32)
        assert vec.shape == (self.dimension,), f"obs dim {vec.shape} != {self.dimension}"
        return vec


class DetectActionCodec:
    """Hybrid action codec: continuous 3 + discrete (role 2, gear 6, pointer 9).

    P3 consumes this from the policy heads; the env applies interface masking
    (role-gear, link-degraded requests) regardless.
    """

    CONT_DIM = 3
    N_ROLE = 2
    N_GEAR = 6
    N_POINTER = 9
    N_BID_SLOTS = DETECT_BID_SLOTS

    def to_action(self, cont, role_idx: int, gear_idx: int, pointer_idx: int,
                  arbitration_bid: float = 0.0,
                  arbitration_bids=None) -> DetectAction:
        u = [max(-1.0, min(1.0, float(c))) for c in cont[:3]]
        role = DetectRole.SEARCH if int(role_idx) == 0 else DetectRole.TRACK
        gear = POLICY_GEAR_ORDER[int(gear_idx) % self.N_GEAR]
        pointer = max(0, min(self.N_POINTER - 1, int(pointer_idx)))
        bid = max(-1.0, min(1.0, float(arbitration_bid)))
        bids = None
        if arbitration_bids is not None:
            if len(arbitration_bids) != self.N_BID_SLOTS:
                raise ValueError(
                    f"arbitration_bids must contain {self.N_BID_SLOTS} values")
            bids = tuple(max(-1.0, min(1.0, float(value)))
                         for value in arbitration_bids)
        return DetectAction(u[0], u[1], u[2], role, gear, pointer, bid, bids)

    def heads(self, action: DetectAction) -> Dict[str, int]:
        role_idx = 0 if action.role == DetectRole.SEARCH else 1
        legal = (4, 5) if role_idx == 0 else (0, 1, 2, 3)
        gear_idx = min(legal, key=lambda idx: abs(
            POLICY_GEAR_ORDER[idx] - action.gear))
        return {"role": role_idx,
                "gear": gear_idx,
                "pointer": max(0, min(self.N_POINTER - 1, action.pointer))}
