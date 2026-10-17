"""MAPPO trainer for the MUMT detection scenario (hybrid action space).

Hybrid heads (plan §7.2): 3 continuous (u_psi, u_v, u_xi) + role (2) + gear
(5) + pointer (9).  One shared actor for all 3 wingmen; critic consumes the
concatenated team observation (centralised training, decentralised
execution).  Gear logits are masked by the sampled role at both collection
and update time (SEARCH -> {45, 60}, TRACK -> {7.5, 15, 30}, plan §6.1
role-gear mask), so PPO ratios are computed on the masked distribution the
env actually sees.

Intervention-consistent learning (一行不动, safety-chain component 6): PPO
ratios use the *nominal* actions; on steps where the shield corrected an
agent, an auxiliary imitation loss pulls the nominal heads toward the safe
action (coef = intervention_coef), so the policy learns to stop provoking
interventions instead of relying on them.

Role-Decomposed Advantage with Conflict-Conditioned shaping (RDAC-MAPPO):
The mixed discrete-continuous action space has three semantically distinct
head families: (1) continuous flight control (u_psi, u_v, u_xi), (2) role +
gear selection, and (3) pointer-based responsibility requests.  A single
global advantage signal cannot simultaneously bootstrap all three when the
positive reward chain requires all of them to fire in sequence.  RDAC-MAPPO
maintains one role-shaping potential Phi_role and one counterfactual pointer
advantage, both computed from local observations only (no truth data), and
blends them into the per-head PPO loss via two scalar coefficients.  All
shaping potentials are potential-based (Ng 1999) so they provably preserve
the optimal policy: eval paths never inject them.
"""
from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .detect_config import DetectSimConfig, ZONE_TRACKING
from .detect_encoder import DetectObservationEncoder, DetectActionCodec, GEAR_ORDER
from .detect_models import DetectAction, DetectRole

ACT_DIM = 6        # [u_psi, u_v, u_xi, role, gear, pointer]


def _gear_masked_logits(gear_logits: torch.Tensor, role_idx: torch.Tensor,
                        allow_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Role-gear mask (plan §6.1).  ``allow_idx`` (teacher/safe target gear) is
    unioned into the legal set so a lagging radar switch (teacher gear outside
    the role's legal set) never puts -inf on the target logit."""
    mask = torch.full_like(gear_logits, -1e9)
    track = role_idx == 1
    mask[~track, 3:] = 0.0      # single-expression indexing: in-place
    mask[track, :3] = 0.0       # (mask[~track][:, 3:] writes to a copy!)
    if allow_idx is not None:
        mask[torch.arange(gear_logits.shape[0]), allow_idx.long().clamp(0, 4)] = 0.0
    return gear_logits + mask


class DetectActorCritic(nn.Module):
    def __init__(self, observation_dim: int = 151, n_agents: int = 3,
                 hidden: int = 128, initial_log_std: float = -1.2):
        super().__init__()
        self.n_agents = n_agents
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.cont_head = nn.Linear(hidden, 3)
        self.role_head = nn.Linear(hidden, 2)
        self.gear_head = nn.Linear(hidden, 5)
        self.pointer_head = nn.Linear(hidden, 9)
        self.log_std = nn.Parameter(torch.full((3,), initial_log_std))
        self.critic = nn.Sequential(
            nn.Linear(observation_dim * n_agents, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def actor_distribution(self, obs: torch.Tensor, role_idx: torch.Tensor | None = None):
        h = self.actor(obs)
        cont = Normal(self.cont_head(h), self.log_std.exp())
        role = Categorical(logits=self.role_head(h))
        gear_logits = self.gear_head(h)
        if role_idx is not None:
            # SEARCH(0) -> gears {45,60} = idx {3,4}; TRACK(1) -> idx {0,1,2}
            mask = torch.full_like(gear_logits, -1e9)
            track = role_idx == 1
            mask[~track, 3:] = 0.0
            mask[track, :3] = 0.0
            gear_logits = gear_logits + mask
        gear = Categorical(logits=gear_logits)
        pointer = Categorical(logits=self.pointer_head(h))
        return cont, role, gear, pointer

    def value(self, team_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(team_obs).squeeze(-1)


@dataclass
class DetectMAPPOConfig:
    hidden_size: int = 128
    clip_ratio: float = 0.2
    learning_rate: float = 1e-4        # gate-5 conservative (3e-4 -> 1e-4: K=2 segment degraded policy)
    bc_learning_rate: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01         # gate-6 exploration bump (0.001 -> 0.01: loitering plateau, KL pinned ~0.002)
    intervention_coef: float = 0.2
    epochs: int = 2                    # gate-5 conservative (4 -> 2)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 0.1
    initial_log_std: float = -1.2
    target_kl: float = 0.01            # tightened from 0.02: prevents policy degradation at high-progress states
    # -- gate-6 potential-based shaping (TRAINING ONLY; eval untouched) ------
    # F = gamma*Phi(s') - Phi(s), Phi = -kappa * sum over owned tracks of
    # min(age, cap)/cap.  Bridges the 30 s credit chain to w1/w3 (Ng 1999:
    # potential-based F provably preserves the optimal policy).
    shaping_kappa: float = 0.05
    shaping_age_cap: float = 8.0
    # -- RDAC-MAPPO: role shaping potential (TRAINING ONLY) ------------------
    # Phi_role(s) = kappa_r * N_fresh_owned_in_tz, where N_fresh_owned_in_tz
    # counts agents i whose track file contains ≥1 confirmed track in the
    # tracking zone with age < AGE_STD_TRACK AND i owns it.  This is a local
    # potential that rewards the team for being in the "TRACK is useful" state;
    # the role/gear advantage is F_role = gamma*Phi_role(s') - Phi_role(s).
    # kappa_r calibrated so one fresh owned tracking-zone target ≈ one W1 tick.
    role_shaping_kappa: float = 0.3    # ~0.02 W1_TICK / reward_scale = 0.2; 0.3 leaves margin
    # Blend coefficient for role/gear advantage: adv_role = adv_t + role_adv_coef * F_role_norm
    role_adv_coef: float = 1.5
    # -- RDAC-MAPPO: geometric proximity shaping for continuous heads -----------
    # Phi_geo(s) = kappa_geo * sum_i sum_{j in owned_i} max(0, 1 - d_ij/C04_RANGE)
    # Added to _track_potential so the continuous heads learn to stay close to
    # owned targets, reducing C04 breaches.  TRAINING ONLY (potential-based).
    geo_shaping_kappa: float = 0.15
    # -- RDAC-MAPPO: stable-window shaping (TRAINING ONLY) -------------------
    # Phi_stable(s) = kappa_stable * sum_{tid} stable_progress[tid] / STABLE_WINDOW
    # Converts the all-or-nothing W3 reward into a dense per-step signal.
    # F_stable = gamma*Phi_stable(s') - Phi_stable(s) is +ve when progress
    # advances, -ve when C04 breach causes a reset.  Calibration: kappa_stable
    # chosen so a full 30-step window yields ~W3_STABLE * reward_scale = 1.0.
    stable_shaping_kappa: float = 1.0
    # -- RDAC-MAPPO: counterfactual pointer advantage (§A) -------------------
    # pointer head advantage is replaced by Q(a) - V_cf(pointer=0).
    # cf_adv_coef blends CF and global advantage: 1.0 = pure CF, 0.0 = original.
    cf_adv_coef: float = 1.0
    # immediate shaped reward for a pointer request that wins new responsibility;
    # units match shaped reward scale (reward_scale already applied to cf_ptr_adv).
    cf_ownership_bonus: float = 0.5


class DetectMAPPOPolicy:
    def __init__(self, env, shield, config: DetectMAPPOConfig | None = None,
                 seed: int = 0):
        torch.manual_seed(seed)
        self.env = env
        self.shield = shield
        self.config = config or DetectMAPPOConfig()
        self.encoder = DetectObservationEncoder(env.cfg)
        self.codec = DetectActionCodec()
        self.model = DetectActorCritic(
            self.encoder.dimension, env.cfg.N_WINGMEN,
            hidden=self.config.hidden_size,
            initial_log_std=self.config.initial_log_std)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate)

    # ------------------------------------------------------------------
    @staticmethod
    def encode_action(a: DetectAction) -> List[float]:
        heads = DetectActionCodec().heads(a)
        return [a.u_psi, a.u_v, a.u_xi,
                float(heads["role"]), float(heads["gear"]), float(heads["pointer"])]

    def _log_prob_entropy(self, obs: torch.Tensor, act: torch.Tensor):
        """Combined log-prob and entropy (used by BC and eval paths only)."""
        role_idx = act[:, 3].long().clamp(0, 1)
        cont, role, gear, pointer = self.model.actor_distribution(obs, role_idx)
        cont_a = act[:, :3].clamp(-1.0, 1.0)
        logp = (cont.log_prob(cont_a).sum(-1)
                + role.log_prob(role_idx)
                + gear.log_prob(act[:, 4].long().clamp(0, 4))
                + pointer.log_prob(act[:, 5].long().clamp(0, 8)))
        entropy = (cont.entropy().sum(-1) + role.entropy()
                   + gear.entropy() + pointer.entropy())
        return logp, entropy

    def act_with_log_prob(self, observation: dict, deterministic: bool = False):
        actions, logps = self.act_batch({0: observation}, deterministic)
        return actions[0], logps[0]["combined"]

    def act_batch(self, observations: Dict[int, dict], deterministic: bool = False):
        """One batched forward for all agents (collection hot path)."""
        ids = sorted(observations)
        obs_t = torch.tensor(
            np.asarray([self.encoder.encode(observations[i]) for i in ids]),
            dtype=torch.float32)
        with torch.no_grad():
            h = self.model.actor(obs_t)
            cont_loc = self.model.cont_head(h)
            cont_dist = Normal(cont_loc, self.model.log_std.exp())
            role_logits = self.model.role_head(h)
            pointer_logits = self.model.pointer_head(h)
            if deterministic:
                cont = cont_loc.clamp(-1.0, 1.0)
                role_idx = role_logits.argmax(-1)
                gear_logits = _gear_masked_logits(self.model.gear_head(h), role_idx)
                gear_idx = gear_logits.argmax(-1)
                pointer_idx = pointer_logits.argmax(-1)
            else:
                cont = cont_dist.sample().clamp(-1.0, 1.0)
                role_idx = Categorical(logits=role_logits).sample()
                gear_logits = _gear_masked_logits(self.model.gear_head(h), role_idx)
                gear_idx = Categorical(logits=gear_logits).sample()
                pointer_idx = Categorical(logits=pointer_logits).sample()
            # Per-head log-probs stored separately so update() can compute
            # per-head PPO ratios without cross-head contamination (RDAC-MAPPO).
            cont_logp_v  = cont_dist.log_prob(cont).sum(-1)
            role_logp_v  = Categorical(logits=role_logits).log_prob(role_idx)
            gear_logp_v  = Categorical(logits=gear_logits).log_prob(gear_idx)
            ptr_logp_v   = Categorical(logits=pointer_logits).log_prob(pointer_idx)
            logp = cont_logp_v + role_logp_v + gear_logp_v + ptr_logp_v
        actions, logps = {}, {}
        for r, i in enumerate(ids):
            actions[i] = self.codec.to_action(cont[r].tolist(),
                                              int(role_idx[r]), int(gear_idx[r]),
                                              int(pointer_idx[r]))
            logps[i] = {
                "combined": float(logp[r]),
                "cont":     float(cont_logp_v[r]),
                "role":     float(role_logp_v[r]),
                "gear":     float(gear_logp_v[r]),
                "ptr":      float(ptr_logp_v[r]),
            }
        return actions, logps

    # ------------------------------------------------------------------
    def collect(self, k_red: int = 4, seed: int = 0,
                deterministic: bool = False,
                w8_scale: float | None = None,
                stable_window: int | None = None) -> List[dict]:
        cfg: DetectSimConfig = self.env.cfg
        scenario = {"seed": seed, "k_red": k_red}
        if w8_scale is not None:            # gate-6 penalty curriculum (train only)
            scenario["w8_scale"] = w8_scale
        if stable_window is not None:       # STABLE_WINDOW curriculum (train only)
            scenario["stable_window"] = stable_window
        obs = self.env.reset(scenario)
        data: List[dict] = []
        while True:
            encoded = {i: self.encoder.encode(obs[i]) for i in obs}
            nominal, old_logp = self.act_batch(obs, deterministic)
            safe, interventions = self.shield.filter(self.env, nominal)
            phi_before = self._track_potential()
            phi_role_before = self._role_potential()
            phi_stable_before = self._stable_potential()
            # RDAC-MAPPO: snapshot arbitration state before env.step mutates it
            cf_snap = self._cf_arb_snapshot()
            res = self.env.step(safe)
            # RDAC-MAPPO: counterfactual pointer advantage (post-step, local only)
            cf_deltas = self._cf_pointer_advantage(cf_snap, nominal)
            team = np.concatenate([
                encoded.get(i, np.zeros(self.encoder.dimension, dtype=np.float32))
                for i in range(cfg.N_WINGMEN)]).astype(np.float32)
            with torch.no_grad():
                value = float(self.model.value(
                    torch.tensor([team], dtype=torch.float32)).item())
            # potential-based shaping: terminal states grounded at Phi = 0
            phi_after = 0.0 if res.done else self._track_potential()
            phi_role_after = 0.0 if res.done else self._role_potential()
            phi_stable_after = 0.0 if res.done else self._stable_potential()
            shaped = float(res.reward) * self.config.reward_scale \
                + self.config.gamma * phi_after - phi_before \
                + self.config.gamma * phi_stable_after - phi_stable_before
            # role shaping increment F_role = gamma*Phi_role(s') - Phi_role(s)
            f_role = self.config.gamma * phi_role_after - phi_role_before
            data.append({
                "obs": encoded, "team": team,
                "nominal": {i: self.encode_action(nominal[i]) for i in nominal},
                "safe": {i: self.encode_action(safe[i]) for i in safe},
                # old_logp: per-head dict for RDAC per-head PPO ratios
                "old_logp": old_logp,
                "value": value,
                "reward": shaped,
                "done": res.done,
                "intervened": {it.uid: it.kind for it in interventions},
                "n_interventions": len(interventions),
                "n_qp": sum(1 for it in interventions if it.kind == "continuous_qp"),
                "n_fallback": sum(1 for it in interventions if it.kind == "reference_fallback"),
                "violations": len(res.info["report"].violations),
                "terminal": res.info["terminal"],
                "cf_deltas": cf_deltas,   # RDAC: per-agent CF pointer advantage
                "f_role": f_role,         # RDAC: role shaping increment (team-level)
            })
            obs = res.observations
            if res.done:
                break
        return data

    # ------------------------------------------------------------------
    def _cf_arb_snapshot(self) -> dict:
        """Cheap snapshot of the arbitration-relevant env state (pre-step).

        Captures only what _cf_pointer_advantage needs: current ownership
        table and per-wingman track files.  Uses only local state already
        available on the Python object — no truth positions accessed.
        """
        env = self.env
        return {
            "responsibility": dict(env.responsibility),
            "owned": [frozenset(w.owned) for w in env.wingmen],
            "t": env.t,
        }

    def _cf_pointer_advantage(
        self,
        snap_before: dict,
        actual_actions: dict,
    ) -> List[float]:
        """Contrastive counterfactual pointer advantage (CR-MAPPO §A, v3).

        For each agent i, find the BEST ownership opportunity in their track
        file — a slot where choosing role=TRACK + pointer=slot+1 would let i
        WIN responsibility it does not currently hold.  Then apply a
        contrastive sign based on whether the agent actually requested:

            delta_i = +opp   if agent chose pointer > 0  (seized opportunity)
            delta_i = -opp   if agent chose pointer = 0  (missed opportunity)
            delta_i =  0     if no winnable target exists

        This breaks the bootstrap deadlock without reinforcing wrong
        behaviour: SEARCH+pointer=0 at a step with a winnable target gets a
        negative advantage, pushing the policy toward TRACK+request.

        Uses only local track file state — no truth positions.  Pool scoring
        mirrors _arbitrate: _gate_ok + _score with ARB_HYSTERESIS.
        """
        cfg = self.env.cfg
        n = cfg.N_WINGMEN
        deltas = [0.0] * n

        for i in range(n):
            w = self.env.wingmen[i]
            best_opp = 0.0  # best achievable W1+bonus from a winnable unowned target

            for tr in w.tracks.slots:
                if tr is None:
                    continue
                tid = tr.target_id

                # Already owns it — no marginal gain
                if self.env.responsibility.get(tid) == i:
                    continue

                # Capability gate
                ex, ey = tr.extrapolate(self.env.t)
                if not self.env._gate_ok(i, ex, ey):
                    continue

                my_score = self.env._score(i, tid, ex, ey)
                pre_owner = snap_before["responsibility"].get(tid)

                # Would i beat the pre-step owner (plus hysteresis)?
                if pre_owner is not None and pre_owner != i:
                    their_score = self.env._score(pre_owner, tid, ex, ey)
                    if my_score <= their_score + cfg.ARB_HYSTERESIS:
                        continue

                # i would win this target — estimate value
                in_tz = cfg.zone_of(ex) == ZONE_TRACKING
                fresh = tr.age(self.env.t) < cfg.AGE_STD_TRACK and in_tz
                w1_gain = cfg.W1_TRACK_TICK if fresh else 0.0
                opp = (w1_gain + self.config.cf_ownership_bonus) * self.config.reward_scale
                best_opp = max(best_opp, opp)

            # Contrastive sign: +opp if seized, -opp if missed
            if best_opp > 0.0:
                a = actual_actions.get(i)
                if a is not None and a.pointer > 0:
                    deltas[i] = +best_opp
                else:
                    deltas[i] = -best_opp

        return deltas

    # ------------------------------------------------------------------
    def _track_potential(self) -> float:
        """Gate-6 shaping potential: -kappa * freshness deficit of OWNED tracks.

        Team-level (matches the team reward). Only targets this wingman owns
        (responsibility == i) count, mirroring w1's own-responsibility scope.

        RDAC extension: also rewards geometric coverage of owned targets via
        Phi_geo(s) = kappa_geo * sum_i sum_{j in owned_i} max(0, 1 - d_ij/C04_RANGE)
        where d_ij is the extrapolated distance from wingman i to owned target j.
        This is a positive potential that peaks at d=0 and decays linearly to zero
        at the C04 capability range, guiding the continuous heads to stay close
        to owned targets without using truth positions (distances are from own
        track file, same as the shield)."""
        cfg_m = self.config
        env = self.env
        cfg = env.cfg

        # Original freshness deficit term
        deficit = 0.0
        if cfg_m.shaping_kappa > 0.0:
            for i, w in enumerate(env.wingmen):
                for tr in w.tracks.slots:
                    if tr is None:
                        continue
                    if env.responsibility.get(tr.target_id) != i:
                        continue
                    deficit += min(tr.age(env.t), cfg_m.shaping_age_cap) / cfg_m.shaping_age_cap
        phi_fresh = -cfg_m.shaping_kappa * deficit

        # Geometric coverage term: reward proximity to owned targets
        phi_geo = 0.0
        if cfg_m.geo_shaping_kappa > 0.0:
            for i, w in enumerate(env.wingmen):
                for tr in w.tracks.slots:
                    if tr is None:
                        continue
                    if env.responsibility.get(tr.target_id) != i:
                        continue
                    ex, ey = tr.extrapolate(env.t)
                    d = math.hypot(ex - w.x, ey - w.y)
                    # Target d = 50% of C04_RANGE so gradient is active well before breach
                    coverage = max(0.0, 1.0 - d / (cfg.C04_RANGE * 0.5))
                    phi_geo += cfg_m.geo_shaping_kappa * coverage

        return phi_fresh + phi_geo

    def _role_potential(self) -> float:
        """RDAC-MAPPO role shaping potential (TRAINING ONLY, §RDAC-role).

        Phi_role(s) = kappa_r * N_fresh_tz, where N_fresh_tz counts (agent, track)
        pairs where agent i owns the track, the extrapolated position is in the
        tracking zone, and track age < AGE_STD_TRACK.

        F_role = gamma * Phi_role(s') - Phi_role(s) is added to the role/gear
        head's advantage signal so that TRACK actions receive positive gradient
        immediately when the team state transitions toward "owning fresh tracks
        in tracking zone", without waiting for the 30 s W3 stable window.

        Uses only local track file state — no truth positions.  Terminal states
        are grounded at Phi_role = 0 (same convention as _track_potential)."""
        cfg = self.config
        if cfg.role_shaping_kappa <= 0.0:
            return 0.0
        env = self.env
        n_fresh = 0.0
        for i, w in enumerate(env.wingmen):
            for tr in w.tracks.slots:
                if tr is None:
                    continue
                if env.responsibility.get(tr.target_id) != i:
                    continue
                ex, _ = tr.extrapolate(env.t)
                if env.cfg.zone_of(ex) == ZONE_TRACKING and tr.age(env.t) < env.cfg.AGE_STD_TRACK:
                    n_fresh += 1.0
        return cfg.role_shaping_kappa * n_fresh

    def _stable_potential(self) -> float:
        """RDAC-MAPPO stable-window shaping potential (TRAINING ONLY, §RDAC-stable).

        Two components:
        1. Progress bonus: Phi_prog = kappa * sum(stable_progress[tid]/STABLE_WINDOW)
           Dense per-step signal for accumulating W3 credit.

        2. Sector-alignment bonus: Phi_align = kappa * sum_{i,j in owned_i} align_ij
           where align_ij = max(0, 1 - |bearing_ij - (psi_i + xi_i)| / half_width)
           using track-file bearing, not truth bearing.  Guides u_xi to keep
           owned targets inside the radar sector, preventing C04 breach from
           sector misalignment even when the aircraft is geometrically close.

        Both are potential-based (Ng 1999), eval-grounded at 0.  No truth."""
        cfg_m = self.config
        if cfg_m.stable_shaping_kappa <= 0.0:
            return 0.0
        env = self.env
        cfg = env.cfg

        # Component 1: progress bonus
        phi_prog = 0.0
        for prog in env.stable_progress.values():
            phi_prog += prog / float(cfg.STABLE_WINDOW)

        # Component 2: sector alignment bonus for ALL owned targets in tracking zone.
        # Fires regardless of stable_progress so u_xi gets signal from the start
        # of ownership, not only after 30 s have accumulated.
        phi_align = 0.0
        for i, w in enumerate(env.wingmen):
            for tr in w.tracks.slots:
                if tr is None:
                    continue
                if env.responsibility.get(tr.target_id) != i:
                    continue
                ex, ey = tr.extrapolate(env.t)
                if env.cfg.zone_of(ex) != ZONE_TRACKING:
                    continue
                bearing = math.atan2(ey - w.y, ex - w.x)
                boresight = w.psi + math.radians(w.radar.axis_deg)
                off = abs(math.atan2(math.sin(bearing - boresight),
                                     math.cos(bearing - boresight)))
                half_w = math.radians(max(1.0, w.radar.gear -
                                          abs(w.radar.axis_deg)))
                phi_align += max(0.0, 1.0 - off / half_w)

        return cfg_m.stable_shaping_kappa * (phi_prog + phi_align)

    # ------------------------------------------------------------------
    def update(self, rollout: Sequence[dict]) -> dict:
        """RDAC-MAPPO update: three decomposed advantage channels.

        adv_cont  → continuous heads (global GAE)
        adv_role  → role + gear heads (GAE + role shaping F_role)
        adv_ptr   → pointer head (GAE + contrastive CF advantage)

        All three are normalised independently to unit variance before use.
        This prevents the role/pointer signals from being swamped by the
        global advantage when the latter is dominated by large penalties."""
        if not rollout:
            return {"loss": 0.0}
        cfg = self.config
        n = self.env.cfg.N_WINGMEN
        obs, team, acts, safes, old_logp = [], [], [], [], []
        rewards, values, dones = [], [], []
        for item in rollout:
            for i in range(n):
                obs.append(item["obs"][i])
                acts.append(item["nominal"][i])
                safes.append(item["safe"][i])
                old_logp.append(item["old_logp"][i])  # dict with per-head logps
            team.append(item["team"])
            rewards.append(item["reward"])
            values.append(item["value"])
            dones.append(item["done"])
        obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32)
        team_t = torch.tensor(np.asarray(team), dtype=torch.float32)
        act_t = torch.tensor(acts, dtype=torch.float32)
        safe_t = torch.tensor(safes, dtype=torch.float32)
        # Per-head old log-probs for per-head PPO ratios (RDAC-MAPPO)
        old_logp_cont  = torch.tensor([item["old_logp"][i]["cont"]     for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_role  = torch.tensor([item["old_logp"][i]["role"]     for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_gear  = torch.tensor([item["old_logp"][i]["gear"]     for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_ptr   = torch.tensor([item["old_logp"][i]["ptr"]      for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_comb  = torch.tensor([item["old_logp"][i]["combined"] for item in rollout for i in range(n)], dtype=torch.float32)

        # GAE over episode-boundary-aware concatenated rollout
        adv = [0.0] * len(rollout)
        gae = 0.0
        next_value = 0.0
        for t in range(len(rollout) - 1, -1, -1):
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + cfg.gamma * next_value * nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
            next_value = values[t]
        returns = torch.tensor([adv[t] + values[t] for t in range(len(rollout))],
                               dtype=torch.float32)

        # --- Channel 1: global advantage (continuous heads) ---------------
        adv_t = torch.tensor(adv, dtype=torch.float32).repeat_interleave(n)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # --- Channel 2: role advantage (role + gear heads) ----------------
        # F_role = gamma*Phi_role(s') - Phi_role(s), team-level; broadcast to
        # all n agents at the same step (same role shaping increment for all).
        f_role_flat = torch.tensor(
            [item["f_role"] for item in rollout for _ in range(n)],
            dtype=torch.float32)
        if f_role_flat.std() > 1e-8:
            f_role_norm = (f_role_flat - f_role_flat.mean()) / (f_role_flat.std() + 1e-8)
        else:
            f_role_norm = f_role_flat
        adv_role_t = adv_t + cfg.role_adv_coef * f_role_norm

        # --- Channel 3: pointer advantage (pointer head) ------------------
        # Contrastive CF delta: +opp if seized opportunity, -opp if missed.
        cf_delta_flat = torch.tensor(
            [item["cf_deltas"][i] for item in rollout for i in range(n)],
            dtype=torch.float32)
        if cf_delta_flat.std() > 1e-8:
            cf_norm = (cf_delta_flat - cf_delta_flat.mean()) / (cf_delta_flat.std() + 1e-8)
        else:
            cf_norm = cf_delta_flat
        adv_ptr_t = adv_t + cfg.cf_adv_coef * cf_norm

        intervention_mask = (act_t - safe_t).abs().sum(-1) > 1e-8
        stats = {}
        approx_kl = torch.zeros(())
        for _ in range(cfg.epochs):
            # --- per-head forward pass ------------------------------------
            role_idx = act_t[:, 3].long().clamp(0, 1)
            h = self.model.actor(obs_t)
            cont_dist = Normal(self.model.cont_head(h), self.model.log_std.exp())
            role_dist = Categorical(logits=self.model.role_head(h))
            gear_logits = _gear_masked_logits(self.model.gear_head(h), role_idx)
            gear_dist = Categorical(logits=gear_logits)
            ptr_dist = Categorical(logits=self.model.pointer_head(h))

            cont_logp = cont_dist.log_prob(act_t[:, :3].clamp(-1.0, 1.0)).sum(-1)
            role_logp = role_dist.log_prob(role_idx)
            gear_logp = gear_dist.log_prob(act_t[:, 4].long().clamp(0, 4))
            ptr_logp  = ptr_dist.log_prob(act_t[:, 5].long().clamp(0, 8))
            entropy = (cont_dist.entropy().sum(-1) + role_dist.entropy()
                       + gear_dist.entropy() + ptr_dist.entropy())

            def _ppo_head_loss(logp_head, old_logp_head, adv_head):
                log_ratio = (logp_head - old_logp_head).clamp(-10.0, 10.0)
                ratio = log_ratio.exp()
                clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                return -torch.min(ratio * adv_head, clipped * adv_head).mean()

            # Approximate KL uses the combined old logp as reference
            combined_logp = cont_logp + role_logp + gear_logp + ptr_logp
            log_ratio_full = (combined_logp - old_logp_comb).clamp(-10.0, 10.0)
            ratio_full = log_ratio_full.exp()
            approx_kl = ((ratio_full - 1.0) - log_ratio_full).mean()

            # Per-head PPO losses with per-head old logps (RDAC-MAPPO)
            loss_cont = _ppo_head_loss(cont_logp, old_logp_cont, adv_t)
            loss_role = _ppo_head_loss(role_logp, old_logp_role, adv_role_t)
            loss_gear = _ppo_head_loss(gear_logp, old_logp_gear, adv_role_t)
            loss_ptr  = _ppo_head_loss(ptr_logp,  old_logp_ptr,  adv_ptr_t)
            policy_loss = loss_cont + loss_role + loss_gear + loss_ptr

            # Value loss (relative scale — see pilot calibration note)
            value_loss_raw = ((self.model.value(team_t) - returns) ** 2).mean()
            value_loss = value_loss_raw / (returns.var().detach() + 1e-4)

            # Intervention imitation loss
            if intervention_mask.any():
                m_obs = obs_t[intervention_mask]
                m_safe = safe_t[intervention_mask]
                h_m = self.model.actor(m_obs)
                s_role = m_safe[:, 3].long().clamp(0, 1)
                cont_loss = ((self.model.cont_head(h_m) - m_safe[:, :3].clamp(-1, 1)) ** 2).mean()
                role_loss_il = nn.functional.cross_entropy(self.model.role_head(h_m), s_role)
                gear_loss_il = nn.functional.cross_entropy(
                    _gear_masked_logits(self.model.gear_head(h_m), s_role,
                                        allow_idx=m_safe[:, 4]),
                    m_safe[:, 4].long().clamp(0, 4))
                ptr_loss_il = nn.functional.cross_entropy(
                    self.model.pointer_head(h_m), m_safe[:, 5].long().clamp(0, 8))
                intervention_loss = cont_loss + role_loss_il + gear_loss_il + ptr_loss_il
            else:
                intervention_loss = torch.zeros(())

            loss = (policy_loss + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy.mean()
                    + cfg.intervention_coef * intervention_loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            with torch.no_grad():
                self.model.log_std.clamp_(-2.5, 0.0)
            if float(approx_kl.detach()) > cfg.target_kl:
                break
        stats.update({
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "intervention_loss": float(intervention_loss.detach()),
            "intervention_fraction": float(intervention_mask.float().mean()),
            "approx_kl": float(approx_kl.detach()),
            "mean_return_scaled": float(returns.mean()),
            "rollout_steps": len(rollout),
            # RDAC diagnostics
            "f_role_mean": float(f_role_flat.mean()),
            "cf_delta_mean": float(cf_delta_flat.mean()),
            "cf_nonzero_frac": float((cf_delta_flat != 0).float().mean()),
        })
        return stats

    # ------------------------------------------------------------------
    def behavior_clone(self, demos: Sequence[Tuple[np.ndarray, DetectAction]],
                       epochs: int = 60, batch_size: int = 4096,
                       focus_weight: float = 5.0) -> dict:
        """BC warm start on HEUR demonstrations (plan §9.4, 50-100 epochs).

        ``focus_weight`` up-weights TRACK+pointer steps, which are ~24% of
        teacher steps but critical for the whole task loop.  Both conditions
        share the same weight: role=TRACK (without pointer) is the prerequisite
        for any request to go through, so it needs the same emphasis."""
        obs = torch.tensor(np.asarray([d[0] for d in demos]), dtype=torch.float32)
        act = torch.tensor([self.encode_action(d[1]) for d in demos],
                           dtype=torch.float32)
        # Up-weight any step where role=TRACK (index 3 == 1) OR pointer > 0
        is_track = act[:, 3] > 0.5          # role == TRACK
        is_ptr   = act[:, 5] > 0             # pointer head requesting
        samp_w = torch.where(is_track | is_ptr,
                             torch.tensor(focus_weight), torch.tensor(1.0))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.bc_learning_rate)
        n = obs.shape[0]
        last = {}
        for epoch in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch_size):
                b = perm[s:s + batch_size]
                ob, ac, sw = obs[b], act[b], samp_w[b]
                role_idx = ac[:, 3].long().clamp(0, 1)
                h = self.model.actor(ob)
                cont_loss = (((self.model.cont_head(h) - ac[:, :3].clamp(-1, 1)) ** 2)
                             .mean(-1) * sw).mean()
                ce = nn.functional.cross_entropy
                role_loss = (ce(self.model.role_head(h), role_idx,
                                reduction="none") * sw).mean()
                gear_loss = (ce(_gear_masked_logits(self.model.gear_head(h), role_idx,
                                                    allow_idx=ac[:, 4]),
                                ac[:, 4].long().clamp(0, 4),
                                reduction="none") * sw).mean()
                pointer_loss = (ce(self.model.pointer_head(h),
                                   ac[:, 5].long().clamp(0, 8),
                                   reduction="none") * sw).mean()
                loss = cont_loss + role_loss + gear_loss + pointer_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
            last = {"epoch": epoch + 1, "loss": float(loss.detach()),
                    "cont": float(cont_loss.detach()),
                    "role": float(role_loss.detach()),
                    "gear": float(gear_loss.detach()),
                    "pointer": float(pointer_loss.detach())}
        # freeze the std closer to the demonstrated (deterministic) behaviour
        # (pilot calibration: cap -0.7 -> -1.4; std 0.5 swamped tracking loops)
        with torch.no_grad():
            self.model.log_std.clamp_(-2.5, -1.4)
        return {"samples": n, **last}

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(),
                    "config": self.config.__dict__}, path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location="cpu")["model"])
