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
from copy import deepcopy
from dataclasses import dataclass, replace
import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal, kl_divergence

from .detect_config import DetectSimConfig, ZONE_TRACKING
from .detect_encoder import DetectObservationEncoder, DetectActionCodec, GEAR_ORDER
from .detect_models import DetectAction, DetectRole
from .detect_task_adapter import ObservationalTaskAdapter
from .track import Track

ACT_DIM = 7        # [u_psi, u_v, u_xi, role, gear, pointer, arbitration_bid]
SQUASH_EPS = 1e-5
N_CONFLICT_FEATURES = 13
CONFLICT_TASK_RESIDUAL_DIM = 3 + 2 + 6 + 9
BID_CONFLICT_RESIDUAL_DIM = 1
TRACK_BLOCK_START = 17
TRACK_FEATURES_PER_SLOT = 14
TRACK_CANDIDATE_FEATURES = TRACK_FEATURES_PER_SLOT


def _atanh_action(action: torch.Tensor) -> torch.Tensor:
    """Inverse of the bounded continuous action transform."""
    a = action.clamp(-1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)
    return 0.5 * (torch.log1p(a) - torch.log1p(-a))


def _squashed_log_prob(dist: Normal, action: torch.Tensor) -> torch.Tensor:
    """Log-probability after mapping a Normal through ``tanh``.

    The environment action is bounded by construction, so the likelihood and
    the executed value stay consistent even when the actor mean is large.
    """
    a = action.clamp(-1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)
    latent = _atanh_action(a)
    correction = torch.log(1.0 - a.square() + SQUASH_EPS)
    return dist.log_prob(latent) - correction


def _bounded_log_prob(dist: Normal, action: torch.Tensor) -> torch.Tensor:
    """Censored-Normal likelihood for demonstrations at +/-1 boundaries.

    Expert controllers legitimately emit envelope limits.  Treating an exact
    boundary as ``atanh(1)`` makes a squashed-Normal BC objective singular;
    censored likelihood assigns it the probability mass beyond that limit.
    Interior actions retain the ordinary Normal density.
    """
    a = action.clamp(-1.0, 1.0)
    scale = dist.scale.clamp_min(1e-6)
    z_hi = (dist.loc - 1.0) / scale
    z_lo = (-1.0 - dist.loc) / scale
    log_hi = torch.special.log_ndtr(z_hi)
    log_lo = torch.special.log_ndtr(z_lo)
    interior = (a.abs() < 1.0 - SQUASH_EPS)
    density = _squashed_log_prob(dist, a)
    out = torch.where(a >= 1.0 - SQUASH_EPS, log_hi,
                      torch.where(a <= -1.0 + SQUASH_EPS, log_lo, density))
    return out


def _gear_masked_logits(gear_logits: torch.Tensor, role_idx: torch.Tensor,
                        allow_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Role-gear mask (plan §6.1).  ``allow_idx`` (teacher/safe target gear) is
    unioned into the legal set so a lagging radar switch (teacher gear outside
    the role's legal set) never puts -inf on the target logit."""
    mask = torch.full_like(gear_logits, -1e9)
    track = role_idx == 1
    mask[~track, 4:] = 0.0      # single-expression indexing: in-place
    mask[track, :4] = 0.0       # (mask[~track][:, 4:] writes to a copy!)
    if allow_idx is not None:
        mask[torch.arange(gear_logits.shape[0]), allow_idx.long().clamp(0, 5)] = 0.0
    return gear_logits + mask


def _encoded_track_mask(obs: torch.Tensor) -> torch.Tensor:
    """Return local track-slot validity from the encoded observation."""
    if obs.ndim != 2 or obs.shape[1] < 129:
        return torch.ones((obs.shape[0], 8), dtype=torch.bool,
                          device=obs.device)
    # Track block starts at 17; slot validity is feature 11 of each 14-vector.
    return obs[:, 17 + 11:129:14] > 1e-6


def _encoded_request_mask(obs: torch.Tensor) -> torch.Tensor:
    """Return track slots eligible for a rule-valid responsibility request."""
    if obs.ndim != 2 or obs.shape[1] < 129:
        return torch.ones((obs.shape[0], 8), dtype=torch.bool,
                          device=obs.device)
    slots = obs[:, 17:129].reshape(obs.shape[0], 8, 14)
    populated = slots[:, :, 11] > 1e-6
    # State one-hot order is detected, tentative, confirmed, coast.
    confirmed_or_tentative = (slots[:, :, 8] > 1e-6) | (slots[:, :, 9] > 1e-6)
    age_ok = slots[:, :, 5] <= (8.0 / 13.0 + 1e-6)
    return populated & confirmed_or_tentative & age_ok


def _pointer_masked_logits(pointer_logits: torch.Tensor,
                            role_idx: torch.Tensor,
                            valid_slots: torch.Tensor | None = None,
                            allow_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Mask pointer choices that the environment necessarily maps to no-op.

    Pointer 0 is always legal. SEARCH cannot issue a responsibility request;
    TRACK may point only to a populated local slot. ``allow_idx`` keeps a
    recorded BC/legacy action finite if its slot disappeared later.
    """
    bsz = pointer_logits.shape[0]
    if valid_slots is None:
        valid_slots = torch.ones((bsz, 8), dtype=torch.bool,
                                 device=pointer_logits.device)
    valid_slots = valid_slots.to(dtype=torch.bool, device=pointer_logits.device)
    track = role_idx == 1
    allowed = torch.zeros((bsz, 9), dtype=torch.bool,
                          device=pointer_logits.device)
    allowed[:, 0] = True
    allowed[:, 1:] = valid_slots & track.unsqueeze(1)
    if allow_idx is not None:
        rows = torch.arange(bsz, device=pointer_logits.device)
        idx = allow_idx.long().clamp(0, 8)
        slot_idx = (idx - 1).clamp(0, 7)
        slot_ok = valid_slots.gather(1, slot_idx.unsqueeze(1)).squeeze(1)
        legal = (idx == 0) | (track & slot_ok)
        allowed[rows, idx] |= legal
    mask = torch.where(allowed, torch.zeros_like(pointer_logits),
                       torch.full_like(pointer_logits, -1e9))
    return pointer_logits + mask


def _legal_pointer_targets(obs: torch.Tensor,
                           role_idx: torch.Tensor,
                           pointer_idx: torch.Tensor) -> torch.Tensor:
    """Project recorded pointer labels onto the current legal action set."""
    idx = pointer_idx.long().clamp(0, 8)
    valid_slots = _encoded_request_mask(obs)
    slot_idx = (idx - 1).clamp(0, 7)
    slot_ok = valid_slots.gather(1, slot_idx.unsqueeze(1)).squeeze(1)
    legal = (idx == 0) | ((role_idx == 1) & slot_ok)
    return torch.where(legal, idx, torch.zeros_like(idx))


def _grouped_masked_mean(values: torch.Tensor, mask: torch.Tensor,
                         groups: torch.Tensor) -> torch.Tensor:
    """Average selected samples within events, then give each event equal weight."""
    active = mask.to(dtype=torch.bool) & (groups >= 0)
    if not active.any():
        return values.sum() * 0.0
    selected_groups = groups[active]
    _, inverse = torch.unique(selected_groups, sorted=True,
                              return_inverse=True)
    selected_values = values[active]
    n_groups = int(inverse.max().item()) + 1
    totals = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    counts = torch.zeros_like(totals)
    totals.scatter_add_(0, inverse, selected_values)
    counts.scatter_add_(0, inverse, torch.ones_like(selected_values))
    return (totals / counts.clamp_min(1.0)).mean()


def _grouped_masked_sum_mean(values: torch.Tensor, mask: torch.Tensor,
                             groups: torch.Tensor) -> torch.Tensor:
    """Sum participating agents' scores per joint event, then average events."""
    active = mask.to(dtype=torch.bool) & (groups >= 0)
    if not active.any():
        return values.sum() * 0.0
    _, inverse = torch.unique(groups[active], sorted=True,
                              return_inverse=True)
    totals = values.new_zeros(int(inverse.max().item()) + 1)
    totals.scatter_add_(0, inverse, values[active])
    return totals.mean()


def _event_winner_advantage(utilities: Dict[int, float], winner: int) -> float:
    """Joint bid-action outcome relative to an action-independent event baseline."""
    return float(utilities[winner]) - float(np.mean(list(utilities.values())))


class DetectActorCritic(nn.Module):
    def __init__(self, observation_dim: int = 169, n_agents: int = 3,
                 hidden: int = 128, initial_log_std: float = -1.2):
        super().__init__()
        self.n_agents = n_agents
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.cont_head = nn.Linear(hidden, 3)
        self.role_head = nn.Linear(hidden, 2)
        self.gear_head = nn.Linear(hidden, 6)
        self.pointer_head = nn.Linear(hidden, 9)
        self.bid_head = nn.Linear(hidden, 1)
        self.bid_candidate_head = nn.Sequential(
            nn.Linear(hidden + TRACK_CANDIDATE_FEATURES, hidden // 2),
            nn.Tanh(), nn.Linear(hidden // 2, 1))
        # Keep legacy checkpoints' bid behavior exactly until PPO learns a
        # candidate-specific correction from arbitration outcomes.
        nn.init.zeros_(self.bid_candidate_head[-1].weight)
        nn.init.zeros_(self.bid_candidate_head[-1].bias)
        # Each local rule-pair proxy can adjust every action family.  Zero
        # initialization preserves the BC/base policy until PPO finds useful
        # conflict-conditioned residuals.
        self.conflict_residual = nn.Linear(
            N_CONFLICT_FEATURES, CONFLICT_TASK_RESIDUAL_DIM, bias=False)
        nn.init.zeros_(self.conflict_residual.weight)
        self.bid_conflict_residual = nn.Linear(
            N_CONFLICT_FEATURES, BID_CONFLICT_RESIDUAL_DIM, bias=False)
        nn.init.zeros_(self.bid_conflict_residual.weight)
        self.log_std = nn.Parameter(torch.full((3,), initial_log_std))
        self.bid_log_std = nn.Parameter(torch.tensor(-1.5))
        self.critic = nn.Sequential(
            nn.Linear(observation_dim * n_agents, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        # Added after all legacy layers; its private RNG and zero output
        # preserve both inherited behavior and legacy initialization streams.
        self.task_geometry_adapter = ObservationalTaskAdapter()

    def action_parameters(self, obs: torch.Tensor,
                          pointer_idx: torch.Tensor | None = None):
        """Return policy parameters, optionally conditioning bid on its target."""
        h = self.actor(obs)
        if obs.shape[-1] >= N_CONFLICT_FEATURES:
            conflict_features = obs[..., -N_CONFLICT_FEATURES:]
        else:
            conflict_features = obs.new_zeros((*obs.shape[:-1],
                                                N_CONFLICT_FEATURES))
        residual = (self.conflict_residual(conflict_features)
                    + self.task_geometry_adapter(obs))
        bid_residual = self.bid_conflict_residual(conflict_features)
        i = 0
        cont_loc = self.cont_head(h) + residual[..., i:i + 3]
        i += 3
        role_logits = self.role_head(h) + residual[..., i:i + 2]
        i += 2
        gear_logits = self.gear_head(h) + residual[..., i:i + 6]
        i += 6
        pointer_logits = self.pointer_head(h) + residual[..., i:i + 9]
        i += 9
        # Each bid is conditioned on the track in the corresponding local
        # slot.  Detaching the shared actor keeps sparse arbitration credit in
        # the bid branch; the candidate head still learns from slot features.
        bid_base = self.bid_head(h.detach()) + bid_residual
        slots = obs[..., TRACK_BLOCK_START:
                    TRACK_BLOCK_START + 8 * TRACK_FEATURES_PER_SLOT]
        slots = slots.reshape(*obs.shape[:-1], 8, TRACK_FEATURES_PER_SLOT)
        expanded_h = h.detach().unsqueeze(-2).expand(
            *h.shape[:-1], 8, h.shape[-1])
        candidate_delta = self.bid_candidate_head(
            torch.cat((expanded_h, slots), dim=-1)).squeeze(-1)
        bid_slot_loc = bid_base + candidate_delta
        if pointer_idx is None:
            bid_loc = bid_slot_loc
        else:
            # Preserve the legacy scalar API for BC/checkpoint diagnostics.
            # PPO collection and updates use the full eight-slot tensor.
            pointer_idx = pointer_idx.long().clamp(0, 8)
            slot_idx = (pointer_idx - 1).clamp(0, 7)
            selected = bid_slot_loc.gather(
                -1, slot_idx.unsqueeze(-1))
            bid_loc = torch.where(
                (pointer_idx > 0).unsqueeze(-1), selected, bid_base)
        return cont_loc, role_logits, gear_logits, pointer_logits, bid_loc

    def actor_distribution(self, obs: torch.Tensor,
                           role_idx: torch.Tensor | None = None,
                           pointer_idx: torch.Tensor | None = None):
        cont_loc, role_logits, gear_logits, pointer_logits, bid_loc = \
            self.action_parameters(obs, pointer_idx)
        cont = Normal(cont_loc, self.log_std.exp())
        role = Categorical(logits=role_logits)
        if role_idx is not None:
            # SEARCH(0) -> {45,60} = idx {4,5}; TRACK(1) -> {7.5,15,30,45} = idx {0..3}
            mask = torch.full_like(gear_logits, -1e9)
            track = role_idx == 1
            mask[~track, 4:] = 0.0
            mask[track, :4] = 0.0
            gear_logits = gear_logits + mask
        gear = Categorical(logits=gear_logits)
        if role_idx is not None:
            pointer_logits = _pointer_masked_logits(
                pointer_logits, role_idx, _encoded_request_mask(obs), pointer_idx)
        pointer = Categorical(logits=pointer_logits)
        bid = Normal(bid_loc, self.bid_log_std.exp())
        return cont, role, gear, pointer, bid

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
    # The QP projection is many-to-one, so pi(safe_action) is not a valid
    # likelihood for the executed push-forward distribution.  Keep nominal
    # PPO credit and mask intervened continuous samples; their safe controls
    # are learned through the intervention imitation term below.
    shield_aware_continuous: bool = False
    mask_intervened_continuous: bool = True
    # Legacy updates mask every continuous head when flight controls are
    # projected. Opt-in sensor adaptation may retain u_xi credit because the
    # shield changes only u_psi/u_v, leaving the sampled radar action intact.
    mask_intervened_radar_axis: bool = True
    epochs: int = 2                    # gate-5 conservative (4 -> 2)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 0.1
    initial_log_std: float = -1.2
    target_kl: float = 0.01            # tightened from 0.02: prevents policy degradation at high-progress states
    bid_target_kl: float = 0.01        # separate KL budget for sparse arbitration updates
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
    # Historical half-range support by default; 1.0 aligns the shaping support
    # with the C04 distance gate for the geometry ablation.
    geo_shaping_range_fraction: float = 0.5
    # -- RDAC-MAPPO: stable-window shaping (TRAINING ONLY) -------------------
    # Phi_stable(s) = kappa_stable * sum_{tid} stable_progress[tid] / STABLE_WINDOW
    # Converts the all-or-nothing W3 reward into a dense per-step signal.
    # F_stable = gamma*Phi_stable(s') - Phi_stable(s) is +ve when progress
    # advances, -ve when C04 breach causes a reset.  Calibration: kappa_stable
    # chosen so a full 30-step window yields ~W3_STABLE * reward_scale = 1.0.
    stable_shaping_kappa: float = 1.0
    # Keep the historical alignment width unless a training ablation opts in.
    stable_alignment_geometry: str = "legacy"
    # -- RDAC-MAPPO: counterfactual pointer advantage (§A) -------------------
    # pointer head advantage is replaced by Q(a) - V_cf(pointer=0).
    # cf_adv_coef blends CF and global advantage: 1.0 = pure CF, 0.0 = original.
    cf_adv_coef: float = 1.0
    # immediate shaped reward for a pointer request that wins new responsibility;
    # units match shaped reward scale (reward_scale already applied to cf_ptr_adv).
    cf_ownership_bonus: float = 0.5
    # Dense part of the conflict-conditioned credit: a legal TRACK request
    # receives this fraction even when arbitration awards the target to a
    # different requester.  The full ownership bonus remains reserved for a
    # request that actually wins, so the signal teaches both participation and
    # conflict resolution without putting a rule policy in the deployment path.
    cf_request_credit: float = 0.25
    # Keep the global return signal as a weak tie-breaker for the request
    # head.  Outside a legal request opportunity the head receives no global
    # gradient, so a bad continuous trajectory cannot teach it to suppress
    # all responsibility requests.
    cf_global_mix: float = 0.25
    cf_role_coef: float = 0.5
    # Joint bid credit uses the actual sampled winner's environment-copy
    # utility minus the event's action-independent mean candidate utility.
    # A fixed positive scale preserves utility differences across events.
    cf_bid_coef: float = 1.0
    cf_bid_utility_scale: float = 0.1
    # Local forced-winner evaluation horizon. Continuations use this actor's
    # deterministic actions, independently in each copied environment.
    cf_bid_horizon: int = 1
    # Legacy checkpoint configuration fields; the winner-credit loss does
    # not use a neutral-bid margin objective.
    cf_bid_margin_coef: float = 0.25
    cf_bid_margin_temp: float = 0.25
    # Targeted arbitration fine-tuning can isolate sparse bid credit from the
    # shared flight/task heads. The deployed actor still emits the full action.
    bid_only_ppo: bool = False
    # Adapt only conflict-conditioned task residuals and arbitration bids;
    # the inherited task controller remains a fixed deployment backbone.
    task_residual_only_ppo: bool = False
    train_geometry_adapter: bool = False
    direct_rl_arbitration: bool = False
    # Disabled pending a reliable post-action C04 counterfactual.  The current
    # pre-step local estimate cannot account exactly for the applied axis trim,
    # denied gear switches, and target motion before arbitration.
    gear_gate_coef: float = 0.0
    # Probability of a training-only shared-target request intervention when
    # two agents independently hold a fresh, C04-feasible local track. The
    # actor still supplies both roles/gears/bids; only the pointer is replaced
    # and that pointer sample is excluded from PPO/entropy gradients.
    conflict_exposure_prob: float = 0.0
    # Training-only conflict probe. A current real target is copied into two
    # empty local track slots for one step and removed immediately afterward.
    # It cannot change the deployed trajectory, red motion, or evaluation.
    conflict_probe_training: bool = False
    # Protect the radar-axis trim from noisy global return gradients.  The
    # reference policy is the loaded BC/checkpoint policy; zero direct PPO
    # weight keeps u_xi anchored until an explicit experiment enables it.
    xi_ppo_coef: float = 0.0
    xi_anchor_coef: float = 1.0
    # Optional task-policy trust region around the explicitly selected start.
    # This reference is an RL checkpoint, independent of the u_xi BC anchor;
    # it excludes arbitration bids and uses only encoded local observations.
    policy_anchor_coef: float = 0.0
    # Read-only cumulative drift diagnostics for unregularized control arms.
    policy_anchor_monitor: bool = False

    def validate_training_mode(self) -> None:
        self.validate_counterfactual_horizon()
        if (not math.isfinite(self.geo_shaping_range_fraction)
                or self.geo_shaping_range_fraction <= 0.0):
            raise ValueError("geo_shaping_range_fraction must be positive and finite")
        if self.stable_alignment_geometry not in {"legacy", "c04_usable_sector"}:
            raise ValueError("stable_alignment_geometry must be legacy or c04_usable_sector")
        if self.bid_only_ppo and self.task_residual_only_ppo:
            raise ValueError("bid_only_ppo and task_residual_only_ppo are mutually exclusive")
        if self.train_geometry_adapter and not self.task_residual_only_ppo:
            raise ValueError("geometry-adapter training requires task_residual_only_ppo")

    def validate_counterfactual_horizon(self) -> None:
        if type(self.cf_bid_horizon) is not int or not 1 <= self.cf_bid_horizon <= 32:
            raise ValueError("cf_bid_horizon must be an integer in [1, 32]")

    def __post_init__(self) -> None:
        self.validate_training_mode()


class DetectMAPPOPolicy:
    def __init__(self, env, shield, config: DetectMAPPOConfig | None = None,
                 seed: int = 0):
        torch.manual_seed(seed)
        self.env = env
        self.shield = shield
        self.config = config or DetectMAPPOConfig()
        self.config.validate_training_mode()
        self._conflict_rng = random.Random(seed)
        self._conflict_probe_rng = random.Random(seed ^ 0x5A17)
        self.encoder = DetectObservationEncoder(env.cfg)
        self.codec = DetectActionCodec()
        self.model = DetectActorCritic(
            self.encoder.dimension, env.cfg.N_WINGMEN,
            hidden=self.config.hidden_size,
            initial_log_std=self.config.initial_log_std)
        self.xi_anchor_model = deepcopy(self.model).eval()
        for parameter in self.xi_anchor_model.parameters():
            parameter.requires_grad_(False)
        if (not math.isfinite(self.config.policy_anchor_coef)
                or self.config.policy_anchor_coef < 0.0):
            raise ValueError("policy_anchor_coef must be finite and nonnegative")
        self.policy_anchor_model = deepcopy(self.model).eval()
        for parameter in self.policy_anchor_model.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate)

    def refresh_policy_anchor(self) -> None:
        """Freeze the current RL task policy as a new refinement reference.

        Call explicitly after loading a chosen starting checkpoint. Loading a
        resumed run instead preserves its saved reference across updates.
        """
        self.policy_anchor_model.load_state_dict(self.model.state_dict())
        self.policy_anchor_model.eval()
        for parameter in self.policy_anchor_model.parameters():
            parameter.requires_grad_(False)

    def _policy_anchor_kl(self, obs: torch.Tensor,
                          current_parameters=None) -> torch.Tensor:
        """Exact KL(reference task action || current task action), excluding bid.

        The three tanh-Normal controls share a bijective transform, so their
        KL equals the latent Normal KL. Gear and pointer are conditioned on
        role: sum both conditional KLs using the reference role probabilities,
        rather than the sampled role or a detached current-role weighting.
        """
        if current_parameters is None:
            current_parameters = self.model.action_parameters(obs)
        current_loc, current_role, current_gear, current_pointer = \
            current_parameters[:4]
        with torch.no_grad():
            self.policy_anchor_model.eval()
            reference = self.policy_anchor_model.action_parameters(obs)
            reference_loc, reference_role, reference_gear, reference_pointer = \
                reference[:4]
            # Match actor_distribution/act_batch exactly; update() clamps the
            # stored log_std after stepping, not inside either distribution.
            reference_cont = Normal(
                reference_loc, self.policy_anchor_model.log_std.exp())
            reference_role_dist = Categorical(logits=reference_role)
        current_cont = Normal(current_loc, self.model.log_std.exp())
        current_role_dist = Categorical(logits=current_role)
        per_sample = (kl_divergence(reference_cont, current_cont).sum(-1)
                      + kl_divergence(reference_role_dist, current_role_dist))
        valid_slots = _encoded_request_mask(obs)
        for role in (0, 1):
            role_idx = torch.full((obs.shape[0],), role, dtype=torch.long,
                                  device=obs.device)
            reference_gear_dist = Categorical(logits=_gear_masked_logits(
                reference_gear, role_idx))
            current_gear_dist = Categorical(logits=_gear_masked_logits(
                current_gear, role_idx))
            reference_pointer_dist = Categorical(logits=_pointer_masked_logits(
                reference_pointer, role_idx, valid_slots))
            current_pointer_dist = Categorical(logits=_pointer_masked_logits(
                current_pointer, role_idx, valid_slots))
            conditional_kl = (
                kl_divergence(reference_gear_dist, current_gear_dist)
                + kl_divergence(reference_pointer_dist, current_pointer_dist))
            per_sample = (per_sample
                          + reference_role_dist.probs[:, role] * conditional_kl)
        return per_sample.mean()

    def _refresh_xi_anchor(self) -> None:
        self.xi_anchor_model.load_state_dict(self.model.state_dict())
        self.xi_anchor_model.eval()
        for parameter in self.xi_anchor_model.parameters():
            parameter.requires_grad_(False)

    def _xi_anchor_kl(self, obs: torch.Tensor,
                      current_loc: torch.Tensor | None = None) -> torch.Tensor:
        """KL(reference u_xi || current u_xi) on a batch of local observations."""
        if current_loc is None:
            current_loc = self.model.action_parameters(obs)[0]
        with torch.no_grad():
            anchor_loc = self.xi_anchor_model.action_parameters(obs)[0][:, 2]
            anchor_log_std = self.xi_anchor_model.log_std[2]
        current_loc_xi = current_loc[:, 2]
        current_log_std = self.model.log_std[2]
        anchor_var = torch.exp(2.0 * anchor_log_std)
        current_var = torch.exp(2.0 * current_log_std)
        return (current_log_std - anchor_log_std
                + (anchor_var + (anchor_loc - current_loc_xi).square())
                / (2.0 * current_var) - 0.5).mean()

    # ------------------------------------------------------------------
    @staticmethod
    def encode_action(a: DetectAction) -> List[float]:
        heads = DetectActionCodec().heads(a)
        return [a.u_psi, a.u_v, a.u_xi,
                float(heads["role"]), float(heads["gear"]),
                float(heads["pointer"]), float(a.arbitration_bid)]

    def _log_prob_entropy(self, obs: torch.Tensor, act: torch.Tensor):
        """Combined log-prob and entropy (used by BC and eval paths only)."""
        role_idx = act[:, 3].long().clamp(0, 1)
        pointer_idx = _legal_pointer_targets(
            obs, role_idx, act[:, 5].long().clamp(0, 8))
        cont, role, gear, pointer, bid = self.model.actor_distribution(
            obs, role_idx, pointer_idx)
        cont_logp = _bounded_log_prob(cont, act[:, :3]).sum(-1)
        bid_logp = _bounded_log_prob(bid, act[:, 6:7]).sum(-1)
        logp = (cont_logp
                + role.log_prob(role_idx)
                + gear.log_prob(act[:, 4].long().clamp(0, 5))
                + pointer.log_prob(pointer_idx)
                + bid_logp)
        # Exact entropy of a squashed Normal has no closed form; the base
        # entropy is a stable exploration proxy for the PPO/BC diagnostic.
        entropy = (cont.entropy().sum(-1) + role.entropy()
                   + gear.entropy() + pointer.entropy()
                   + bid.entropy().sum(-1))
        return logp, entropy

    def condition_exposed_bids(
        self, encoded: Dict[int, np.ndarray],
        actions: Dict[int, DetectAction], old_logp: Dict[int, dict],
        exposure_mask: Sequence[bool],
    ) -> Dict[int, DetectAction]:
        """Resample training-exposed bids under their forced pointer labels."""
        ids = [i for i, exposed in enumerate(exposure_mask)
               if exposed and i in actions and i in encoded]
        if not ids:
            return actions
        obs_t = torch.tensor(np.asarray([encoded[i] for i in ids]),
                             dtype=torch.float32)
        pointer_idx = torch.tensor([actions[i].pointer for i in ids],
                                   dtype=torch.long)
        with torch.no_grad():
            slot_loc = self.model.action_parameters(obs_t)[-1]
            slot_dist = Normal(slot_loc, self.model.bid_log_std.exp())
            slot_bid = torch.tanh(slot_dist.sample())
            slot_logp = _squashed_log_prob(slot_dist, slot_bid)
            slot_idx = (pointer_idx - 1).clamp(0, 7)
            legacy_loc = self.model.action_parameters(obs_t, pointer_idx)[-1]
            legacy_dist = Normal(legacy_loc, self.model.bid_log_std.exp())
            legacy_bid = torch.tanh(legacy_dist.sample()).squeeze(-1)
            selected_slot_bid = slot_bid.gather(
                1, slot_idx.unsqueeze(1)).squeeze(1)
            selected_slot_logp = slot_logp.gather(
                1, slot_idx.unsqueeze(1)).squeeze(1)
            legacy_bid = torch.where(pointer_idx > 0,
                                     selected_slot_bid, legacy_bid)
            legacy_logp = torch.where(
                pointer_idx > 0, selected_slot_logp,
                _squashed_log_prob(
                    legacy_dist, legacy_bid.unsqueeze(-1)).sum(-1))
        for row, i in enumerate(ids):
            old_bid_logp = old_logp[i]["bid"]
            actions[i] = replace(
                actions[i], arbitration_bid=float(legacy_bid[row]),
                arbitration_bids=tuple(float(v) for v in slot_bid[row]))
            old_logp[i]["bid"] = float(legacy_logp[row])
            old_logp[i]["bid_slots"] = [
                float(v) for v in slot_logp[row]]
            old_logp[i]["combined"] += float(legacy_logp[row]) - old_bid_logp
        return actions

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
            (cont_loc, role_logits, base_gear_logits,
             pointer_logits, _) = self.model.action_parameters(obs_t)
            cont_dist = Normal(cont_loc, self.model.log_std.exp())
            if deterministic:
                cont = torch.tanh(cont_loc)
                role_idx = role_logits.argmax(-1)
                gear_logits = _gear_masked_logits(base_gear_logits, role_idx)
                gear_idx = gear_logits.argmax(-1)
            else:
                cont = torch.tanh(cont_dist.sample())
                role_idx = Categorical(logits=role_logits).sample()
                gear_logits = _gear_masked_logits(base_gear_logits, role_idx)
                gear_idx = Categorical(logits=gear_logits).sample()
            pointer_logits = _pointer_masked_logits(
                pointer_logits, role_idx, _encoded_request_mask(obs_t))
            pointer_dist = Categorical(logits=pointer_logits)
            pointer_idx = (pointer_logits.argmax(-1) if deterministic
                           else pointer_dist.sample())
            bid_loc = self.model.action_parameters(obs_t)[-1]
            bid_dist = Normal(bid_loc, self.model.bid_log_std.exp())
            bid_slots = (torch.tanh(bid_dist.loc) if deterministic
                         else torch.tanh(bid_dist.sample()))
            legacy_bid_loc = self.model.action_parameters(
                obs_t, pointer_idx)[-1]
            legacy_bid_dist = Normal(
                legacy_bid_loc, self.model.bid_log_std.exp())
            legacy_bid = (torch.tanh(legacy_bid_dist.loc).squeeze(-1)
                          if deterministic else
                          torch.tanh(legacy_bid_dist.sample()).squeeze(-1))
            # Per-head log-probs stored separately so update() can compute
            # per-head PPO ratios without cross-head contamination (RDAC-MAPPO).
            cont_logp_v  = _squashed_log_prob(cont_dist, cont).sum(-1)
            cont_dim_logp = _squashed_log_prob(cont_dist, cont)
            role_logp_v  = Categorical(logits=role_logits).log_prob(role_idx)
            gear_logp_v  = Categorical(logits=gear_logits).log_prob(gear_idx)
            ptr_logp_v   = pointer_dist.log_prob(pointer_idx)
            bid_slot_logp = _squashed_log_prob(bid_dist, bid_slots)
            selected_slot = (pointer_idx - 1).clamp(0, 7)
            selected_bid = bid_slots.gather(
                1, selected_slot.unsqueeze(1)).squeeze(1)
            selected_bid_logp = bid_slot_logp.gather(
                1, selected_slot.unsqueeze(1)).squeeze(1)
            legacy_bid = torch.where(pointer_idx > 0, selected_bid,
                                     legacy_bid)
            bid_logp_v = torch.where(
                pointer_idx > 0, selected_bid_logp,
                _squashed_log_prob(
                    legacy_bid_dist, legacy_bid.unsqueeze(-1)).sum(-1))
            logp = (cont_logp_v + role_logp_v + gear_logp_v + ptr_logp_v
                    + bid_logp_v)
        actions, logps = {}, {}
        for r, i in enumerate(ids):
            actions[i] = self.codec.to_action(cont[r].tolist(),
                                              int(role_idx[r]), int(gear_idx[r]),
                                              int(pointer_idx[r]),
                                              float(legacy_bid[r]),
                                              tuple(float(v) for v in bid_slots[r]),
                                              self.config.direct_rl_arbitration)
            logps[i] = {
                "combined": float(logp[r]),
                "cont":     float(cont_logp_v[r]),
                "uv":       float(cont_dim_logp[r, :2].sum()),
                "xi":       float(cont_dim_logp[r, 2]),
                "role":     float(role_logp_v[r]),
                "gear":     float(gear_logp_v[r]),
                "ptr":      float(ptr_logp_v[r]),
                "bid":      float(bid_logp_v[r]),
                "bid_slots": [float(v) for v in bid_slot_logp[r]],
            }
        return actions, logps

    def _continuous_log_probs(self, encoded: Dict[int, np.ndarray],
                              actions: Dict[int, DetectAction]) -> Dict[int, float]:
        """Evaluate old-policy log-probs for the controls actually executed.

        The safety shield can project only ``u_psi``/``u_v``.  Keeping this
        likelihood beside the nominal per-head likelihood lets PPO assign the
        environment reward to the executed continuous control without changing
        role, gear, or pointer credit.
        """
        ids = sorted(encoded)
        if not ids:
            return {}
        obs_t = torch.tensor(np.asarray([encoded[i] for i in ids]),
                             dtype=torch.float32)
        act_t = torch.tensor(
            [[actions[i].u_psi, actions[i].u_v, actions[i].u_xi]
             for i in ids], dtype=torch.float32)
        with torch.no_grad():
            cont_loc = self.model.action_parameters(obs_t)[0]
            dist = Normal(cont_loc, self.model.log_std.exp())
            logp = _squashed_log_prob(dist, act_t).sum(-1)
        return {i: float(logp[j]) for j, i in enumerate(ids)}

    def _install_training_conflict_probe(self) -> dict | None:
        """Install one reversible shared-track conflict for PPO collection.

        The target position is copied from the current environment into two
        empty local track slots.  This is a training fixture only: it does
        not alter red/blue kinematics and is removed after ``env.step``.
        Restricting the fixture to empty slots avoids overwriting a genuine
        local track or masking a legitimate sensing update.
        """
        env, cfg = self.env, self.env.cfg
        red_pos = {a.uid: (a.x, a.y) for a in env.reds.agents
                   if not env.judged.get(a.uid, False)}
        pairs = []
        for tid, (tx, ty) in sorted(red_pos.items()):
            for left in range(cfg.N_WINGMEN):
                for right in range(left + 1, cfg.N_WINGMEN):
                    if not env._gate_ok(left, tx, ty) or not env._gate_ok(
                            right, tx, ty):
                        continue
                    left_tracks = env.wingmen[left].tracks
                    right_tracks = env.wingmen[right].tracks
                    if (tid in left_tracks._by_target
                            or tid in right_tracks._by_target):
                        continue
                    left_slot = next(
                        (j for j, tr in enumerate(left_tracks.slots)
                         if tr is None), None)
                    right_slot = next(
                        (j for j, tr in enumerate(right_tracks.slots)
                         if tr is None), None)
                    if left_slot is not None and right_slot is not None:
                        pairs.append((tid, left, right, left_slot,
                                      right_slot, tx, ty))
        if not pairs:
            return None
        tid, left, right, left_slot, right_slot, tx, ty = pairs[
            self._conflict_probe_rng.randrange(len(pairs))]
        touched = {}
        for i, slot in ((left, left_slot), (right, right_slot)):
            tracks = env.wingmen[i].tracks
            touched[i] = (list(tracks.slots), dict(tracks._by_target))
            tracks.slots[slot] = Track(
                slot=slot, target_id=tid, x=tx, y=ty, state="confirmed",
                sigma=1.0, last_hit_t=env.t, first_hit_t=env.t,
                hits_in_window=[env.t])
            tracks._by_target[tid] = slot
        return {
            "target_id": tid,
            "pair": (left, right),
            "tracks": touched,
            "responsibility": env.responsibility.get(tid),
            "handoff_history": list(env.handoff_history.get(tid, [])),
            "last_handoff": env.last_handoff_t.get(tid),
            "tz_track_since": env.tz_track_since.get(tid),
            "stable_progress": env.stable_progress.get(tid),
            "milestone_reached": env.milestone_reached.get(tid),
            "judged": env.judged.get(tid, False),
        }

    def _restore_training_conflict_probe(self, state: dict | None) -> None:
        """Remove a probe while preserving unrelated environment progress."""
        if state is None:
            return
        env = self.env
        tid = int(state["target_id"])
        for i, (slots, by_target) in state["tracks"].items():
            env.wingmen[i].tracks.slots = slots
            env.wingmen[i].tracks._by_target = by_target
        original_owner = state["responsibility"]
        if original_owner is None:
            env.responsibility.pop(tid, None)
            for wingman in env.wingmen:
                wingman.owned.discard(tid)
        else:
            env.responsibility[tid] = original_owner
            for i, wingman in enumerate(env.wingmen):
                if i == original_owner:
                    wingman.owned.add(tid)
                else:
                    wingman.owned.discard(tid)
        if state["handoff_history"]:
            env.handoff_history[tid] = state["handoff_history"]
        else:
            env.handoff_history.pop(tid, None)
        if state["last_handoff"] is None:
            env.last_handoff_t.pop(tid, None)
        else:
            env.last_handoff_t[tid] = state["last_handoff"]
        for name in ("tz_track_since", "stable_progress", "milestone_reached"):
            value = state[name]
            mapping = getattr(env, name)
            if value is None:
                mapping.pop(tid, None)
            else:
                mapping[tid] = value
        env.judged[tid] = state["judged"]

    # ------------------------------------------------------------------
    def collect(self, k_red: int | None = None, seed: int = 0,
                deterministic: bool = False,
                w8_scale: float | None = None) -> List[dict]:
        cfg: DetectSimConfig = self.env.cfg
        scenario = {"seed": seed}
        if k_red is not None:
            scenario["k_red"] = int(k_red)
        if w8_scale is not None:            # gate-6 penalty curriculum (train only)
            scenario["w8_scale"] = w8_scale
        # Keep exposure sampling reproducible without consuming the task RNG
        # stream used for radar hits, red maneuvers, or link loss.
        self._conflict_rng = random.Random(int(seed) + 0xC0F11C7)
        obs = self.env.reset(scenario)
        data: List[dict] = []
        while True:
            probe_state = None
            if self.config.conflict_probe_training and not deterministic:
                probe_state = self._install_training_conflict_probe()
                if probe_state is not None:
                    # The actor must see the temporary track in its local
                    # observation before choosing the exposed pointer.
                    obs = self.env.observe()
            encoded = {i: self.encoder.encode(obs[i]) for i in obs}
            sampled, old_logp = self.act_batch(obs, deterministic)
            behavior, exposure_mask, exposure_pairs = \
                self._expose_shared_target_conflict(sampled, deterministic)
            behavior = self.condition_exposed_bids(
                encoded, behavior, old_logp, exposure_mask)
            req_stats = self._request_stats(behavior)
            safe, interventions = self.shield.filter(self.env, behavior)
            gear_gate_credit = self._gear_gate_credit(safe)
            safe_logp_cont = self._continuous_log_probs(encoded, safe)
            phi_before = self._track_potential()
            phi_role_before = self._role_potential()
            phi_stable_before = self._stable_potential()
            # RDAC-MAPPO: snapshot arbitration state before env.step mutates it
            cf_snap = self._cf_arb_snapshot()
            arbitration_replay_env = (
                deepcopy(self.env) if self._may_have_arbitration_conflict(safe)
                else None)
            try:
                res = self.env.step(safe)
            finally:
                # Restore only the temporary target state.  Blue/red motion
                # and all unrelated task progress remain in the environment.
                self._restore_training_conflict_probe(probe_state)
            request_outcomes = res.info.get("request_outcomes", [])
            # Credit requests from the environment's actual gate and arbitration result.
            cf_deltas = self._cf_pointer_advantage(
                cf_snap, sampled, request_outcomes)
            # The intervention selected these requests outside the pointer
            # policy, so they must not create pointer-head credit either.
            for i, exposed in enumerate(exposure_mask):
                if exposed:
                    cf_deltas[i] = 0.0
            bid_rank = [0.0] * cfg.N_WINGMEN
            bid_events = [0] * cfg.N_WINGMEN
            bid_event_mask_slots = [
                [False] * 8 for _ in range(cfg.N_WINGMEN)]
            bid_event_ids_slots = [
                [-1] * 8 for _ in range(cfg.N_WINGMEN)]
            bid_cf_adv_slots = [
                [0.0] * 8 for _ in range(cfg.N_WINGMEN)]
            bid_cf_mask_slots = [
                [False] * 8 for _ in range(cfg.N_WINGMEN)]
            arbitration = res.info.get("arbitration", [])
            exposed_conflicts = 0
            bid_changed_winner = 0
            for conflict_index, conflict in enumerate(arbitration):
                candidates = [int(i) for i in conflict.get("candidates", [])]
                if len(candidates) < 2:
                    continue
                if sum(bool(exposure_mask[i]) for i in candidates) >= 2:
                    exposed_conflicts += 1
                score = conflict.get("scores", {})
                bids = conflict.get("bids", {})
                owner = conflict.get("previous_owner")
                hysteresis = float(conflict.get("hysteresis", 0.0))
                no_bid = max(candidates, key=lambda i: (
                    float(score.get(i, score.get(str(i), 0.0))), i == owner, -i))
                owner_score = float(score.get(owner, score.get(str(owner), 0.0))) \
                    if owner is not None else 0.0
                if (owner in candidates and no_bid != owner
                        and float(score.get(no_bid, score.get(str(no_bid), 0.0)))
                        <= owner_score + hysteresis):
                    no_bid = owner
                if int(conflict["winner"]) != no_bid:
                    bid_changed_winner += 1
                winner = int(conflict["winner"])
                loser_credit = -1.0 / max(1, len(candidates) - 1)
                for i in candidates:
                    if not 0 <= i < cfg.N_WINGMEN:
                        continue
                    bid_events[i] += 1
                    bid_rank[i] += 1.0 if i == winner else loser_credit
                    slot_map = conflict.get("bid_slots", {})
                    slot = slot_map.get(i, slot_map.get(str(i)))
                    if slot is not None and 0 <= int(slot) < 8:
                        bid_event_mask_slots[i][int(slot)] = True
                        bid_event_ids_slots[i][int(slot)] = conflict_index
            for i, count in enumerate(bid_events):
                if count:
                    bid_rank[i] /= count
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
            cf_bid_stats = self._counterfactual_bid_credit(
                arbitration_replay_env, safe, arbitration,
                phi_before, phi_stable_before, bid_cf_adv_slots,
                bid_cf_mask_slots)
            bid_cf_adv = [
                (float(np.mean([value for value, active in zip(values, mask)
                                if active]))
                 if any(mask) else 0.0)
                for values, mask in zip(bid_cf_adv_slots, bid_cf_mask_slots)]
            bid_cf_mask = [any(mask) for mask in bid_cf_mask_slots]
            # role shaping increment F_role = gamma*Phi_role(s') - Phi_role(s)
            f_role = self.config.gamma * phi_role_after - phi_role_before
            data.append({
                "obs": encoded, "team": team,
                # Store the action sent through the shield. On exposed rows,
                # standard action-head losses are masked; the sampled RL bid
                # remains active for the real arbitration event.
                "nominal": {i: self.encode_action(behavior[i]) for i in behavior},
                "safe": {i: self.encode_action(safe[i]) for i in safe},
                # old_logp: per-head dict for RDAC per-head PPO ratios
                "old_logp": old_logp,
                # Continuous old-policy likelihood of the action that the
                # environment will actually execute after shielding.
                "safe_logp_cont": safe_logp_cont,
                "value": value,
                "reward": shaped,
                "environment_reward": float(res.reward),
                "shaping_reward": (shaped
                                   - float(res.reward) * self.config.reward_scale),
                "done": res.done,
                "intervened": {it.uid: it.kind for it in interventions},
                "n_interventions": len(interventions),
                "n_qp": sum(1 for it in interventions if it.kind == "continuous_qp"),
                "n_fallback": sum(1 for it in interventions if it.kind != "continuous_qp"),
                "correction_magnitudes": [float(it.magnitude) for it in interventions],
                "n_track_actions": req_stats["n_track"],
                "n_nonzero_pointer": req_stats["n_nonzero_pointer"],
                "n_valid_requests": req_stats["n_valid_request"],
                "n_requests": len(request_outcomes),
                "n_c04_pass_requests": sum(
                    bool(outcome["c04_gate_ok"])
                    for outcome in request_outcomes),
                "n_c04_rejected_requests": sum(
                    outcome.get("reason") == "c04_rejected"
                    for outcome in request_outcomes),
                "n_request_winners": sum(
                    outcome.get("reason") == "selected"
                    for outcome in request_outcomes),
                "n_new_assignments": sum(
                    bool(outcome["new_assignment"])
                    for outcome in request_outcomes),
                "n_multi_candidate_events": len(arbitration),
                "n_handoffs": len(res.info.get("handoffs", [])),
                "owned_count": len(self.env.responsibility),
                "stable_progress_total": sum(
                    int(progress) for progress in self.env.stable_progress.values()),
                "stable_targets_in_progress": sum(
                    int(progress > 0)
                    for progress in self.env.stable_progress.values()),
                "max_stable_progress": max(self.env.stable_progress.values(), default=0),
                "violations": len(res.info["report"].violations),
                "terminal": res.info["terminal"],
                "cf_deltas": cf_deltas,   # RDAC: per-agent CF pointer advantage
                "gear_gate_credit": gear_gate_credit,
                "bid_rank": bid_rank,
                "bid_mask": [count > 0 for count in bid_events],
                "bid_event_mask_slots": bid_event_mask_slots,
                "bid_event_ids_slots": bid_event_ids_slots,
                "bid_actions": {
                    i: tuple(float(v) for v in
                             (behavior[i].arbitration_bids or (0.0,) * 8))
                    for i in behavior},
                "bid_cf_adv": bid_cf_adv,
                "bid_cf_mask": bid_cf_mask,
                "bid_cf_adv_slots": bid_cf_adv_slots,
                "bid_cf_mask_slots": bid_cf_mask_slots,
                "n_cf_bid_events": cf_bid_stats["events"],
                "n_cf_bid_informative_events": (
                    cf_bid_stats["informative_events"]),
                "n_cf_bid_candidate_branches": (
                    cf_bid_stats["candidate_branches"]),
                "n_cf_bid_uncontrollable_events": (
                    cf_bid_stats["uncontrollable_events"]),
                "bid_cf_utility_spreads": cf_bid_stats["utility_spreads"],
                "bid_cf_actual_regrets": cf_bid_stats["actual_regrets"],
                "bid_cf_event_records": cf_bid_stats["event_records"],
                "n_cf_bid_active_slots": sum(
                    any(mask) for mask in bid_cf_mask_slots),
                "n_bid_events": len(arbitration),
                "action_exposure_mask": exposure_mask,
                "n_forced_overlap_pairs": exposure_pairs,
                "n_pointer_exposed": sum(exposure_mask),
                "n_exposed_multi_candidate_events": exposed_conflicts,
                "n_bid_changed_winner": bid_changed_winner,
                "f_role": f_role,         # RDAC: role shaping increment (team-level)
            })
            obs = (self.env.observe() if probe_state is not None
                   else res.observations)
            if res.done:
                break
        return data

    def _expose_shared_target_conflict(
        self, sampled: Dict[int, DetectAction], deterministic: bool = False,
        probe_rng: random.Random | None = None,
        preferred_pair: Tuple[int, int, int] | None = None,
    ) -> Tuple[Dict[int, DetectAction], List[bool], int]:
        """Training-only request overlap using independently held local tracks.

        This does not alter env state, geometry, rules, or arbitration. For a
        sampled shared target, it temporarily sets two agents to a legal TRACK
        request. Their role/gear/pointer samples are excluded from PPO; the
        independently sampled RL bids remain live in the actual environment
        arbitration.
        """
        cfg = self.env.cfg
        rng = probe_rng or self._conflict_rng
        probability = float(self.config.conflict_exposure_prob)
        probability = min(1.0, max(0.0, probability))
        actions = dict(sampled)
        exposed = [False] * cfg.N_WINGMEN
        if (probability <= 0.0 or deterministic
                or rng.random() >= probability):
            return actions, exposed, 0

        # Candidates are derived solely from each agent's own track file.
        # An incumbent may participate in a challenger event; targets owned
        # by the third, uninvolved agent are excluded.
        # Use the gear that can actually be active after this request. Exposed
        # samples hold the current axis (u_xi=0), preserving the measured gate
        # geometry through the control update.
        local = {}
        for i, wingman in enumerate(self.env.wingmen):
            if i not in actions or wingman.link_degraded:
                continue
            tracks = {}
            for slot_idx, track in enumerate(wingman.tracks.slots):
                if (track is None or track.state == "detected"
                        or track.age(self.env.t) > cfg.AGE_DEGRADE
                        or self.env.judged.get(track.target_id, False)):
                    continue
                tx, ty = track.extrapolate(self.env.t)
                geometry = self.env._gate_diagnostics(i, tx, ty)
                switch_available = (wingman.radar.last_switch_step < 0
                                    or wingman.radar.switch_ready())
                effective_gear = (
                    max(cfg.GEARS_TRACK) if switch_available
                    else wingman.radar.gear)
                usable_half_width = min(
                    effective_gear,
                    cfg.SECTOR_AXIS_MAX_DEG - abs(wingman.radar.axis_deg))
                sector_margin = (usable_half_width
                                 - geometry["bearing_offset_deg"])
                if (geometry["range_margin_km"] >= 2.0
                        and sector_margin >= 0.0
                        and track.age(self.env.t) <= cfg.AGE_DEGRADE
                        and track.state != "detected"
                        and not self.env.judged.get(track.target_id, False)):
                    tracks[track.target_id] = slot_idx + 1
            local[i] = tracks

        pairs = []
        ids = sorted(local)
        for left_pos, left in enumerate(ids):
            for right in ids[left_pos + 1:]:
                shared = sorted(set(local[left]).intersection(local[right]))
                for tid in shared:
                    owner = self.env.responsibility.get(tid)
                    if owner is not None and owner not in (left, right):
                        continue
                    pairs.append((left, right, tid))
        if preferred_pair is not None:
            pairs = [pair for pair in pairs if pair == preferred_pair]
        if not pairs:
            return actions, exposed, 0

        left, right, tid = pairs[rng.randrange(len(pairs))]
        for i in (left, right):
            actions[i] = replace(
                actions[i], u_xi=0.0, role=DetectRole.TRACK,
                gear=max(cfg.GEARS_TRACK), pointer=local[i][tid])
        exposed[left] = exposed[right] = True
        return actions, exposed, 1

    def _may_have_arbitration_conflict(
        self, actions: Dict[int, DetectAction],
    ) -> bool:
        """Cheap local precheck before copying env for counterfactual replay."""
        cfg = self.env.cfg
        requesters_by_target: Dict[int, set] = {}
        for i, action in actions.items():
            if action.role != DetectRole.TRACK or action.pointer <= 0:
                continue
            wingman = self.env.wingmen[i]
            if wingman.link_degraded:
                continue
            slot_idx = action.pointer - 1
            if not 0 <= slot_idx < len(wingman.tracks.slots):
                continue
            track = wingman.tracks.slots[slot_idx]
            if (track is None or track.state == "detected"
                    or track.age(self.env.t) > cfg.AGE_DEGRADE
                    or self.env.judged.get(track.target_id, False)):
                continue
            requesters_by_target.setdefault(track.target_id, set()).add(i)
        for tid, requesters in requesters_by_target.items():
            owner = self.env.responsibility.get(tid)
            if owner is not None and owner not in requesters:
                return True
            if len(requesters | ({owner} if owner is not None else set())) > 1:
                return True
        return False

    def _rollout_counterfactual_bid_branch(
        self, env_before, branch_actions: Dict[int, DetectAction],
        target_id: int, phi_before: float, phi_stable_before: float,
        post_first_step_cleanup: Callable[[object, dict], None] | None = None,
    ) -> tuple:
        """Evaluate one forced first-step winner using a local RL continuation.

        The first actions already passed through the live safety shield. Later
        actions come only from this actor and a copied shield, using observations
        of this branch. This is a finite-horizon conditional auxiliary target,
        not a full-episode value or a deployment planner. Each branch starts
        from a fresh environment copy, including its private simulator RNG.
        Global RNG streams are restored even if a replay raises.
        """
        self.config.validate_counterfactual_horizon()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.random.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_initialized() else None)
        try:
            branch = deepcopy(env_before)
            shield = (deepcopy(self.shield)
                      if self.config.cf_bid_horizon > 1 else None)
            actions = deepcopy(branch_actions)
            discounted_return = 0.0
            discount = 1.0
            matching = None
            steps = 0
            terminal = False
            for offset in range(self.config.cf_bid_horizon):
                if offset:
                    nominal, _ = self.act_batch(
                        branch.observe(), deterministic=True)
                    actions, _ = shield.filter(branch, nominal)
                step = branch.step(actions)
                discounted_return += discount * float(step.reward)
                discount *= self.config.gamma
                steps += 1
                terminal = bool(step.done)
                if not offset:
                    matching = next((deepcopy(item) for item in
                                     step.info.get("arbitration", [])
                                     if item.get("tid") == target_id), None)
                    if matching is None:
                        return None, None, {}
                    if post_first_step_cleanup is not None:
                        post_first_step_cleanup(branch, matching)
                if terminal:
                    break
            track_after = (0.0 if terminal else
                           self._potential_in_env(branch, "track"))
            stable_after = (0.0 if terminal else
                            self._potential_in_env(branch, "stable"))
            utility = (
                discounted_return * self.config.reward_scale
                + discount * track_after - phi_before
                + discount * stable_after - phi_stable_before)
            initial_progress = getattr(env_before, "stable_progress", {})
            final_progress = getattr(branch, "stable_progress", {})
            initial_judged = getattr(env_before, "judged", {})
            final_judged = getattr(branch, "judged", {})
            diagnostics = {
                "horizon": self.config.cf_bid_horizon,
                "steps": steps,
                "terminal": terminal,
                "discounted_env_return": float(discounted_return),
                "stable_progress_before": int(
                    initial_progress.get(target_id, 0)),
                "stable_progress_after": int(
                    final_progress.get(target_id, 0)),
                "judged_before": bool(initial_judged.get(target_id, False)),
                "judged_after": bool(final_judged.get(target_id, False)),
                "track_potential_after": float(track_after),
                "stable_potential_after": float(stable_after),
            }
            return float(utility), matching, diagnostics
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def _counterfactual_bid_credit(
        self, env_before, actions: Dict[int, DetectAction],
        arbitration: Sequence[dict],
        phi_before: float, phi_stable_before: float,
        advantages: List[List[float]], masks: List[List[bool]],
        post_first_step_cleanup: Callable[[object, dict], None] | None = None,
    ) -> dict:
        """Assign candidate-specific relative utility to each bid slot.

        Each forced-winner branch is an action-independent potential outcome
        for one candidate.  Centering those utilities by their event mean
        gives the bid for a better branch positive credit and the bids for
        worse branches negative credit.  A single sampled-winner advantage
        copied to every candidate cannot teach relative bid ordering because
        all candidates receive the same gradient sign.
        """
        self.config.validate_counterfactual_horizon()
        stats = {
            "events": 0,
            "informative_events": 0,
            "candidate_branches": 0,
            "uncontrollable_events": 0,
            "utility_spreads": [],
            "actual_regrets": [],
            "event_records": [],
        }
        if env_before is None or not arbitration:
            return stats

        for event in arbitration:
            candidates = [int(i) for i in event.get("candidates", [])
                          if int(i) in actions]
            if (len(candidates) < 2
                    or event.get("arbitration_mode") != "rl_direct"):
                continue
            slot_map = event.get("bid_slots", {})
            slots = {}
            controllable = True
            for agent_idx in candidates:
                slot_value = slot_map.get(
                    agent_idx, slot_map.get(str(agent_idx)))
                action = actions[agent_idx]
                if (slot_value is None or action.arbitration_bids is None):
                    controllable = False
                    break
                slot_idx = int(slot_value)
                if not 0 <= slot_idx < len(action.arbitration_bids):
                    controllable = False
                    break
                if hasattr(env_before, "wingmen"):
                    wingmen = env_before.wingmen
                    if (not 0 <= agent_idx < len(wingmen)
                            or slot_idx >= len(wingmen[agent_idx].tracks.slots)):
                        controllable = False
                        break
                    track = wingmen[agent_idx].tracks.slots[slot_idx]
                    if (track is None
                            or track.target_id != event.get("tid")):
                        controllable = False
                        break
                slots[agent_idx] = slot_idx
            if not controllable:
                stats["uncontrollable_events"] += 1
                continue

            utilities = {}
            branch_results = {}
            for forced_winner in candidates:
                branch_actions = dict(actions)
                for agent_idx in candidates:
                    action = actions[agent_idx]
                    slot_idx = slots[agent_idx]
                    forced_bid = 1.0 if agent_idx == forced_winner else -1.0
                    bids = list(action.arbitration_bids)
                    bids[slot_idx] = forced_bid
                    branch_actions[agent_idx] = replace(
                        action, arbitration_bid=forced_bid,
                        arbitration_bids=tuple(bids))
                utility, matching, diagnostics = \
                    self._rollout_counterfactual_bid_branch(
                        env_before, branch_actions, event.get("tid"),
                        phi_before, phi_stable_before,
                        post_first_step_cleanup=post_first_step_cleanup)
                stats["candidate_branches"] += 1
                if (utility is None or matching is None
                        or int(matching.get("winner", -1)) != forced_winner
                        or ("candidates" in matching and
                            sorted(int(i) for i in matching["candidates"])
                            != sorted(candidates))):
                    controllable = False
                    break
                utilities[forced_winner] = utility
                branch_results[forced_winner] = diagnostics
            if not controllable or len(utilities) != len(candidates):
                stats["uncontrollable_events"] += 1
                continue

            actual_winner = int(event.get("winner", -1))
            if actual_winner not in utilities:
                stats["uncontrollable_events"] += 1
                continue
            utility_values = list(utilities.values())
            utility_spread = max(utility_values) - min(utility_values)
            advantage = _event_winner_advantage(utilities, actual_winner)
            candidate_advantages = {
                candidate: float(utilities[candidate]
                                 - np.mean(utility_values))
                for candidate in candidates
            }
            actual_regret = max(utility_values) - utilities[actual_winner]
            stats["events"] += 1
            stats["utility_spreads"].append(float(utility_spread))
            stats["actual_regrets"].append(float(actual_regret))
            stats["event_records"].append({
                "tid": event.get("tid"),
                "candidates": candidates,
                "bid_slots": slots,
                "winner": actual_winner,
                "utilities": utilities,
                "candidate_advantages": candidate_advantages,
                "utility_spread": float(utility_spread),
                "actual_regret": float(actual_regret),
                "event_advantage": advantage,
                "horizon": self.config.cf_bid_horizon,
                "branch_results": branch_results,
            })
            if utility_spread <= 1e-8:
                continue
            for agent_idx, slot_idx in slots.items():
                advantages[agent_idx][slot_idx] += candidate_advantages[agent_idx]
                # Keep a middle-utility candidate's zero advantage in the
                # event population. Dropping that event would condition the
                # loss denominator on the sampled bids.
                masks[agent_idx][slot_idx] = True
            stats["informative_events"] += 1
        return stats

    def _potential_in_env(self, env, kind: str) -> float:
        """Evaluate existing training potentials against a replayed env."""
        previous = self.env
        try:
            self.env = env
            if kind == "track":
                return self._track_potential()
            if kind == "stable":
                return self._stable_potential()
            raise ValueError(kind)
        finally:
            self.env = previous

    def _request_stats(self, actions: Dict[int, DetectAction]) -> dict:
        """Measure action support/credit before the environment applies it."""
        n_track = n_nonzero = n_valid = 0
        for i, a in actions.items():
            if a.role == DetectRole.TRACK:
                n_track += 1
            if a.pointer > 0:
                n_nonzero += 1
            w = self.env.wingmen[i]
            valid = (a.role == DetectRole.TRACK and a.pointer > 0
                     and 0 <= a.pointer - 1 < len(w.tracks.slots)
                     and w.tracks.slots[a.pointer - 1] is not None
                     and not w.link_degraded)
            if valid:
                tr = w.tracks.slots[a.pointer - 1]
                # I02/I03 are method-visible request validity conditions:
                # do not call a single-hit or stale pointer a valid learning
                # opportunity merely because the slot is populated.
                valid = (tr.state != "detected"
                         and tr.age(self.env.t) <= self.env.cfg.AGE_DEGRADE)
            n_valid += int(valid)
        return {"n_track": n_track, "n_nonzero_pointer": n_nonzero,
                "n_valid_request": n_valid}

    def _gear_gate_credit(self, actions: Dict[int, DetectAction]) -> List[float]:
        """Return local counterfactual credit for a narrow TRACK gear.

        The signal is active only when the same track request is outside the
        selected sector but inside the widest legal TRACK sector.  A sampled
        narrow gear is then a negative action outcome, so PPO shifts its mass
        toward the available wide TRACK gear without consulting HEUR or truth.
        """
        cfg = self.env.cfg
        widest = max(cfg.GEARS_TRACK)
        credit = [0.0] * cfg.N_WINGMEN
        for i, action in actions.items():
            if action.role != DetectRole.TRACK or action.pointer <= 0:
                continue
            wingman = self.env.wingmen[i]
            slot_idx = action.pointer - 1
            if slot_idx >= len(wingman.tracks.slots):
                continue
            track = wingman.tracks.slots[slot_idx]
            if track is None or wingman.link_degraded:
                continue
            tx, ty = track.extrapolate(self.env.t)
            geometry = self.env._gate_diagnostics(i, tx, ty)
            if geometry["range_margin_km"] < 0.0:
                continue
            axis_room = cfg.SECTOR_AXIS_MAX_DEG - abs(wingman.radar.axis_deg)
            widest_usable = min(widest, axis_room)
            selected_usable = min(float(action.gear), axis_room)
            widest_passes = geometry["bearing_offset_deg"] <= widest_usable
            selected_passes = geometry["bearing_offset_deg"] <= selected_usable
            if widest_passes and not selected_passes:
                credit[i] = -1.0
        return credit

    # ------------------------------------------------------------------
    def _cf_arb_snapshot(self) -> dict:
        """Cheap snapshot of the arbitration-relevant env state (pre-step).

        Captures only what _cf_pointer_advantage needs: current ownership
        table and per-wingman track files.  Uses only local state already
        available on the Python object — no truth positions accessed.
        """
        env = self.env
        # Preserve the pre-action local track files.  Sensing/arbitration may
        # mutate slots during env.step; reading live slots afterward can give a
        # pointer credit for a different target.
        tracks = []
        wing_states = []
        for w in env.wingmen:
            wing_states.append({
                "x": w.x, "y": w.y, "psi": w.psi,
                "gear": w.radar.gear, "axis_deg": w.radar.axis_deg,
            })
            slots = []
            for tr in w.tracks.slots:
                slots.append(None if tr is None else {
                    "target_id": tr.target_id, "x": tr.x, "y": tr.y,
                    "vx": tr.vx, "vy": tr.vy, "sigma": tr.sigma,
                    "last_hit_t": tr.last_hit_t, "state": tr.state,
                })
            tracks.append(slots)
        return {
            "responsibility": dict(env.responsibility),
            "owned": [frozenset(w.owned) for w in env.wingmen],
            "tracks": tracks,
            "wing_states": wing_states,
            "link_degraded": [bool(w.link_degraded) for w in env.wingmen],
            "t": env.t,
        }

    def _cf_pointer_advantage_legacy(
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

    def _cf_pointer_advantage(
        self, snap_before: dict, actual_actions: dict,
        request_outcomes: List[dict] | None = None) -> List[float]:
        """Assign pointer credit only to requests that can take effect.

        A non-zero pointer is merely an intention.  The environment may reject
        it because the role is SEARCH, the slot is empty, the link is degraded,
        or arbitration gives the target to another wingman. Actual C04 and
        arbitration outcomes determine request credit; the pre-step local track
        file is used only to identify a missed unowned opportunity. C04
        rejections and missing targets are neutral, legal arbitration losers
        receive participation credit, and invalid pointer values are neutral.
        """
        cfg = self.env.cfg
        n = cfg.N_WINGMEN
        deltas = [0.0] * n
        post_resp = self.env.responsibility
        tracks_by_agent = snap_before.get("tracks", [[] for _ in range(n)])
        wing_states = snap_before.get("wing_states", [])
        pre_t = float(snap_before["t"])
        outcomes_by_request = {
            (int(outcome["requester"]), int(outcome["target_id"])): outcome
            for outcome in (request_outcomes or [])
        }

        def local_gate(i: int, ex: float, ey: float) -> bool:
            if i >= len(wing_states):
                return False
            st = wing_states[i]
            d = math.hypot(ex - st["x"], ey - st["y"])
            if d > cfg.C04_RANGE:
                return False
            bearing = math.atan2(ey - st["y"], ex - st["x"])
            delta = bearing - (st["psi"] + math.radians(st["axis_deg"]))
            off = math.degrees(abs(math.atan2(math.sin(delta), math.cos(delta))))
            usable = min(st["gear"], cfg.SECTOR_AXIS_MAX_DEG - abs(st["axis_deg"]))
            return off <= usable

        for i in range(n):
            slots = tracks_by_agent[i] if i < len(tracks_by_agent) else []
            candidates = {}
            unowned_opps = []
            for slot_idx, slot in enumerate(slots):
                if slot is None:
                    continue
                tid = slot["target_id"]
                # A populated slot is not automatically a legitimate request
                # basis.  Rewarding these cases would reinforce I02/I03
                # violations even when arbitration happens to accept them.
                if slot.get("state") == "detected":
                    continue
                if snap_before["responsibility"].get(tid) == i:
                    continue
                dt = pre_t - float(slot["last_hit_t"])
                if dt > cfg.AGE_DEGRADE:
                    continue
                ex = slot["x"] + slot["vx"] * dt
                ey = slot["y"] + slot["vy"] * dt
                if not local_gate(i, ex, ey):
                    continue
                in_tz = cfg.zone_of(ex) == ZONE_TRACKING
                fresh = dt < cfg.AGE_STD_TRACK and in_tz
                w1_gain = cfg.W1_TRACK_TICK if fresh else 0.0
                opp = (w1_gain + self.config.cf_ownership_bonus) * self.config.reward_scale
                candidates[slot_idx] = (tid, opp)
                if snap_before["responsibility"].get(tid) is None:
                    unowned_opps.append(opp)

            if not candidates:
                continue
            best_opp = max(unowned_opps, default=0.0)
            a = actual_actions.get(i)
            if a is None:
                continue

            selected = None
            if (a.role == DetectRole.TRACK and a.pointer > 0
                    and not snap_before["link_degraded"][i]):
                selected = candidates.get(a.pointer - 1)
            if selected is not None:
                tid, opp = selected
                pre_owner = snap_before["responsibility"].get(tid)
                outcome = outcomes_by_request.get((i, tid))
                if outcome is None or outcome.get("reason") in (
                        "target_missing", "c04_rejected"):
                    continue
                if not outcome.get("c04_gate_ok", False):
                    continue
                if outcome.get("reason") == "lost_arbitration":
                    deltas[i] = opp * self.config.cf_request_credit
                elif (outcome.get("reason") == "selected"
                      and outcome.get("winner") == i
                      and pre_owner != i and post_resp.get(tid) == i):
                    deltas[i] = opp
            elif (a.role == DetectRole.TRACK and a.pointer == 0
                  and not snap_before["link_degraded"][i]
                  and best_opp > 0.0):
                # Missing a feasible request is negative, but keep its scale
                # symmetric with the partial participation credit so the
                # sparse ownership wins do not destabilise the pointer head.
                deltas[i] = -best_opp * self.config.cf_request_credit

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
                    support = cfg.C04_RANGE * cfg_m.geo_shaping_range_fraction
                    coverage = max(0.0, 1.0 - d / support)
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
                if cfg_m.stable_alignment_geometry == "c04_usable_sector":
                    half_width_deg = max(0.0, min(
                        w.radar.gear,
                        cfg.SECTOR_AXIS_MAX_DEG - abs(w.radar.axis_deg)))
                else:
                    half_width_deg = max(
                        1.0, w.radar.gear - abs(w.radar.axis_deg))
                if half_width_deg > 0.0:
                    half_w = math.radians(half_width_deg)
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
        cfg = self.config
        cfg.validate_training_mode()
        if not rollout:
            return {"loss": 0.0}
        n = self.env.cfg.N_WINGMEN
        obs, team, acts, safes, bid_actions, old_logp = [], [], [], [], [], []
        rewards, values, dones = [], [], []
        for item in rollout:
            for i in range(n):
                obs.append(item["obs"][i])
                acts.append(item["nominal"][i])
                safes.append(item["safe"][i])
                item_bid_actions = item.get("bid_actions", {})
                bid_actions.append(item_bid_actions.get(
                    i, (float(item["nominal"][i][6]),) * 8))
                old_logp.append(item["old_logp"][i])  # dict with per-head logps
            team.append(item["team"])
            rewards.append(item["reward"])
            values.append(item["value"])
            dones.append(item["done"])
        obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32)
        team_t = torch.tensor(np.asarray(team), dtype=torch.float32)
        act_t = torch.tensor(acts, dtype=torch.float32)
        safe_t = torch.tensor(safes, dtype=torch.float32)
        bid_actions_t = torch.tensor(bid_actions, dtype=torch.float32)
        if bid_actions_t.shape != (len(acts), 8):
            raise ValueError(
                f"bid_actions must have shape {(len(acts), 8)}, "
                f"got {tuple(bid_actions_t.shape)}")
        # Per-head old log-probs for per-head PPO ratios (RDAC-MAPPO)
        old_logp_uv    = torch.tensor([item["old_logp"][i]["uv"]      for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_xi    = torch.tensor([item["old_logp"][i]["xi"]      for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_role  = torch.tensor([item["old_logp"][i]["role"]     for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_gear  = torch.tensor([item["old_logp"][i]["gear"]     for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_ptr   = torch.tensor([item["old_logp"][i]["ptr"]      for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_bid   = torch.tensor([item["old_logp"][i]["bid"]      for item in rollout for i in range(n)], dtype=torch.float32)
        old_logp_bid_slots = torch.tensor([
            item["old_logp"][i].get(
                "bid_slots", [item["old_logp"][i]["bid"]] * 8)
            for item in rollout for i in range(n)], dtype=torch.float32)
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

        # Channel 2b: local counterfactual TRACK gear credit.  The credit is
        # sparse by design: it only fires when a wider legal beam would have
        # cleared the C04 sector gate for the selected request.
        gear_credit_flat = torch.tensor(
            [item["gear_gate_credit"][i]
             for item in rollout for i in range(n)], dtype=torch.float32)
        gear_active = gear_credit_flat.abs() > 1e-8
        # Keep the signed action outcome instead of centering it: all active
        # examples are intentionally "narrow beam was wrong" outcomes.
        gear_signal = torch.where(
            gear_active, gear_credit_flat, torch.zeros_like(gear_credit_flat))
        adv_gear_t = adv_role_t + cfg.gear_gate_coef * gear_signal

        # --- Channel 3: pointer advantage (pointer head) ------------------
        # Contrastive CF delta: +opp if seized opportunity, -opp if missed.
        cf_delta_flat = torch.tensor(
            [item["cf_deltas"][i] for item in rollout for i in range(n)],
            dtype=torch.float32)
        if cf_delta_flat.std() > 1e-8:
            cf_norm = (cf_delta_flat - cf_delta_flat.mean()) / (cf_delta_flat.std() + 1e-8)
        else:
            cf_norm = cf_delta_flat
        cf_active = cf_delta_flat.abs() > 1e-8
        # Conflict-conditioned credit is only meaningful where a legal local
        # request was available.  This avoids turning the overwhelming number
        # of no-contact steps into an implicit ``pointer=0`` label.
        cf_signal = torch.where(cf_active, cf_norm, torch.zeros_like(cf_norm))
        adv_ptr_t = cfg.cf_global_mix * adv_t + cfg.cf_adv_coef * cf_signal
        adv_role_t = adv_role_t + cfg.cf_role_coef * cf_signal

        # Channel 4: joint bid-action credit from forced-winner replays.
        bid_rank_t = torch.tensor(
            [item["bid_rank"][i] for item in rollout for i in range(n)],
            dtype=torch.float32)
        bid_mask_t = torch.tensor(
            [item["bid_mask"][i] for item in rollout for i in range(n)],
            dtype=torch.bool)
        # A counterfactual return difference is already an action advantage.
        # It must not be multiplied by the shared team GAE, which can reverse
        # the causal task-utility direction. Legacy rank-only rollouts are
        # intentionally inactive here rather than teaching a tautological
        # winner label.
        cf_bid_rows, cf_bid_mask_rows = [], []
        bid_candidate_rows = []
        bid_event_group_rows = []
        next_bid_event_group = 0
        for item in rollout:
            explicit_adv = item.get("bid_cf_adv_slots")
            explicit_mask = item.get("bid_cf_mask_slots")
            candidate_mask = item.get("bid_event_mask_slots")
            event_ids = item.get("bid_event_ids_slots")
            if explicit_adv is not None and explicit_mask is not None:
                step_adv = [list(row) for row in explicit_adv]
                step_mask = [list(row) for row in explicit_mask]
            else:
                # A scalar legacy counterfactual cannot be assigned to a local
                # target without the environment event's bid_slots mapping.
                step_adv = [[0.0] * 8 for _ in range(n)]
                step_mask = [[False] * 8 for _ in range(n)]
            if candidate_mask is None:
                candidate_mask = [[False] * 8 for _ in range(n)]
            if event_ids is None:
                event_ids = [[-1] * 8 for _ in range(n)]
                for agent in range(n):
                    for slot in range(8):
                        if (candidate_mask[agent][slot]
                                or step_mask[agent][slot]):
                            event_ids[agent][slot] = next_bid_event_group
                            next_bid_event_group += 1
            else:
                step_event_ids = [
                    int(event_id)
                    for row in event_ids for event_id in row
                    if int(event_id) >= 0]
                offset = next_bid_event_group
                event_ids = [[
                    offset + int(event_id) if int(event_id) >= 0 else -1
                    for event_id in row] for row in event_ids]
                if step_event_ids:
                    next_bid_event_group += max(step_event_ids) + 1
            bid_candidate_rows.extend(
                [list(row) for row in candidate_mask])
            bid_event_group_rows.extend(
                [list(row) for row in event_ids])
            cf_bid_rows.extend(step_adv)
            cf_bid_mask_rows.extend(step_mask)
        cf_bid_t = torch.tensor(cf_bid_rows, dtype=torch.float32)
        bid_active_t = torch.tensor(cf_bid_mask_rows, dtype=torch.bool)
        bid_candidate_t = torch.tensor(bid_candidate_rows, dtype=torch.bool)
        bid_event_group_t = torch.tensor(
            bid_event_group_rows, dtype=torch.long)
        missing_event_ids = torch.nonzero(
            bid_active_t & (bid_event_group_t < 0), as_tuple=False)
        if missing_event_ids.numel():
            bid_event_group_t = bid_event_group_t.clone()
            first_missing_group = int(bid_event_group_t.max().item()) + 1
            for offset, (row, slot) in enumerate(missing_event_ids.tolist()):
                bid_event_group_t[row, slot] = first_missing_group + offset
        action_exposure_t = torch.tensor(
            [item.get("action_exposure_mask",
                      item.get("pointer_exposure_mask", [False] * n))[i]
             for item in rollout for i in range(n)], dtype=torch.bool)
        # A fixed utility scale preserves the relative importance of events.
        # A scale estimated from sampled winner advantages depends on actions
        # and can cancel/reweight their policy gradient, especially in sparse
        # batches. Never normalize these credits by the sampled batch RMS.
        if (not math.isfinite(cfg.cf_bid_utility_scale)
                or cfg.cf_bid_utility_scale <= 0.0):
            raise ValueError("cf_bid_utility_scale must be finite and positive")
        cf_bid_norm = cf_bid_t / cfg.cf_bid_utility_scale
        adv_bid_t = torch.where(
            bid_active_t, cf_bid_norm, torch.zeros_like(cf_bid_norm))

        # The shield owns only the continuous flight controls.  Keep the
        # intervention target head-local: copying safe role/gear/pointer labels
        # would inject artificial discrete supervision even though those heads
        # were never changed by the safety layer.
        cont_delta = (act_t[:, :2] - safe_t[:, :2]).abs()
        intervention_mask = cont_delta.sum(-1) > 1e-8
        stats = {}
        approx_kl = torch.zeros(())
        bid_approx_kl = torch.zeros(())
        bid_grad_norm = torch.zeros(())
        task_residual_grad_norm = torch.zeros(())
        critic_grad_norm = torch.zeros(())
        bid_parameter_prefixes = (
            "bid_head.", "bid_candidate_head.", "bid_conflict_residual.")
        for name, parameter in self.model.named_parameters():
            is_bid_parameter = (
                name.startswith(bid_parameter_prefixes)
                or name == "bid_log_std")
            if cfg.task_residual_only_ppo:
                trainable = (is_bid_parameter
                             or name == "conflict_residual.weight"
                             or name.startswith("critic."))
            else:
                trainable = is_bid_parameter or not cfg.bid_only_ppo
            if name.startswith("task_geometry_adapter."):
                trainable = cfg.train_geometry_adapter
            parameter.requires_grad_(trainable)
        bid_parameters = (
            list(self.model.bid_head.parameters())
            + list(self.model.bid_candidate_head.parameters())
            + list(self.model.bid_conflict_residual.parameters())
            + [self.model.bid_log_std])
        task_residual_parameters = list(self.model.conflict_residual.parameters())
        geometry_parameters = list(self.model.task_geometry_adapter.parameters())
        if cfg.train_geometry_adapter:
            task_residual_parameters += geometry_parameters
        critic_parameters = list(self.model.critic.parameters())
        bid_parameter_before = [p.detach().clone() for p in bid_parameters]
        geometry_before = [p.detach().clone() for p in geometry_parameters]
        geometry_grad_norm = torch.zeros(())
        task_residual_before = (self.model.conflict_residual.weight.detach().clone()
                                if cfg.task_residual_only_ppo else None)
        for _ in range(cfg.epochs):
            # --- per-head forward pass ------------------------------------
            role_idx = act_t[:, 3].long().clamp(0, 1)
            (cont_loc, role_logits, base_gear_logits,
             base_pointer_logits, bid_loc) = self.model.action_parameters(
                 obs_t, act_t[:, 5].long().clamp(0, 8))
            bid_slot_loc = self.model.action_parameters(obs_t)[-1]
            cont_dist = Normal(cont_loc, self.model.log_std.exp())
            role_dist = Categorical(logits=role_logits)
            gear_logits = _gear_masked_logits(base_gear_logits, role_idx)
            gear_dist = Categorical(logits=gear_logits)
            pointer_target = _legal_pointer_targets(
                obs_t, role_idx, act_t[:, 5].long().clamp(0, 8))
            ptr_logits = _pointer_masked_logits(
                base_pointer_logits, role_idx,
                _encoded_request_mask(obs_t), pointer_target)
            ptr_dist = Categorical(logits=ptr_logits)
            bid_slot_dist = Normal(
                bid_slot_loc, self.model.bid_log_std.exp())

            # Nominal actions have the actual behavior likelihood.  A safe
            # action is retained for imitation only; using its density in a
            # PPO ratio would ignore the shield's many-to-one projection.
            cont_nominal_logp = _squashed_log_prob(
                cont_dist, act_t[:, :3]).sum(-1)
            cont_dim_logp = _squashed_log_prob(cont_dist, act_t[:, :3])
            cont_uv_logp = cont_dim_logp[:, :2].sum(-1)
            cont_xi_logp = cont_dim_logp[:, 2]
            role_logp = role_dist.log_prob(role_idx)
            gear_logp = gear_dist.log_prob(act_t[:, 4].long().clamp(0, 5))
            ptr_logp  = ptr_dist.log_prob(pointer_target)
            bid_slot_logp = _squashed_log_prob(
                bid_slot_dist, bid_actions_t)
            cont_entropy = cont_dist.entropy()
            regular_entropy = (
                cont_entropy[:, :2].sum(-1)
                + cfg.xi_ppo_coef * cont_entropy[:, 2]
                + role_dist.entropy() + gear_dist.entropy()
                + ptr_dist.entropy())
            bid_entropy = bid_slot_dist.entropy().expand_as(
                bid_active_t.float())
            bid_entropy_mean = _grouped_masked_mean(
                bid_entropy, bid_active_t, bid_event_group_t)

            def _ppo_head_loss(logp_head, old_logp_head, adv_head,
                               sample_weight=None, sample_groups=None):
                log_ratio = (logp_head - old_logp_head).clamp(-10.0, 10.0)
                ratio = log_ratio.exp()
                clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                per_sample = -torch.min(ratio * adv_head,
                                        clipped * adv_head)
                if sample_weight is None:
                    return per_sample.mean()
                if sample_groups is not None:
                    return _grouped_masked_sum_mean(
                        per_sample, sample_weight > 0, sample_groups)
                return (per_sample * sample_weight).sum() \
                    / sample_weight.sum().clamp_min(1.0)

            # Keep the task-policy KL separate from the sparse slot-wise bid
            # KL. The combined behavior logp includes the selected scalar bid.
            combined_logp = (cont_nominal_logp + role_logp + gear_logp
                             + ptr_logp)
            old_logp_task = old_logp_comb - old_logp_bid
            log_ratio_full = (combined_logp - old_logp_task).clamp(-10.0, 10.0)
            ratio_full = log_ratio_full.exp()
            kl_rows = ~action_exposure_t
            approx_kl = (((ratio_full - 1.0) - log_ratio_full)[kl_rows].mean()
                         if kl_rows.any() else torch.zeros(()))
            bid_log_ratio = (
                bid_slot_logp - old_logp_bid_slots).clamp(-10.0, 10.0)
            bid_ratio = bid_log_ratio.exp()
            bid_approx_kl = (
                ((bid_ratio - 1.0) - bid_log_ratio)[bid_active_t].mean()
                if bid_active_t.any() else torch.zeros(()))

            # Per-head PPO losses with per-head old logps (RDAC-MAPPO)
            cont_weight = (~action_exposure_t).float()
            if cfg.mask_intervened_continuous:
                cont_weight = cont_weight * (~intervention_mask).float()
            xi_weight = (~action_exposure_t).float()
            if (cfg.mask_intervened_continuous
                    and cfg.mask_intervened_radar_axis):
                xi_weight = xi_weight * (~intervention_mask).float()
            loss_cont_uv = _ppo_head_loss(cont_uv_logp, old_logp_uv, adv_t,
                                          cont_weight)
            loss_cont_xi = _ppo_head_loss(cont_xi_logp, old_logp_xi, adv_t,
                                          xi_weight)
            xi_anchor_loss = self._xi_anchor_kl(obs_t, cont_loc)
            policy_anchor_kl = (
                self._policy_anchor_kl(
                    obs_t, (cont_loc, role_logits, base_gear_logits,
                            base_pointer_logits))
                if cfg.policy_anchor_coef > 0.0 else obs_t.new_zeros(()))
            loss_role = _ppo_head_loss(
                role_logp, old_logp_role, adv_role_t,
                (~action_exposure_t).float())
            loss_gear = _ppo_head_loss(
                gear_logp, old_logp_gear, adv_gear_t,
                (~action_exposure_t).float())
            loss_ptr  = _ppo_head_loss(
                ptr_logp, old_logp_ptr, adv_ptr_t,
                (~action_exposure_t).float())
            loss_bid  = cfg.cf_bid_coef * _ppo_head_loss(
                bid_slot_logp, old_logp_bid_slots, adv_bid_t,
                bid_active_t.float(), bid_event_group_t)
            policy_loss = (loss_cont_uv + cfg.xi_ppo_coef * loss_cont_xi
                           + cfg.xi_anchor_coef * xi_anchor_loss
                           + cfg.policy_anchor_coef * policy_anchor_kl
                           + loss_role + loss_gear + loss_ptr + loss_bid)
            if cfg.bid_only_ppo:
                policy_loss = loss_bid
                entropy_bonus = bid_entropy_mean
            else:
                task_entropy = (
                    regular_entropy * (~action_exposure_t).float()).mean()
                entropy_bonus = task_entropy + bid_entropy_mean

            # Value loss (relative scale — see pilot calibration note)
            # Bid-only updates use causal counterfactual utilities directly;
            # the critic contributes no actor signal and must not dilute the
            # sparse arbitration gradient through the shared global clip.
            if cfg.bid_only_ppo:
                value_loss = torch.zeros((), device=returns.device)
            else:
                value_loss_raw = ((self.model.value(team_t) - returns) ** 2).mean()
                value_loss = value_loss_raw / (
                    returns.var(unbiased=False).detach() + 1e-4)

            # Intervention imitation loss: continuous heads only, weighted by
            # the size of the actual safety correction.  A tiny numerical
            # projection should not count as strongly as a large breakaway.
            imitation_mask = intervention_mask & ~action_exposure_t
            if imitation_mask.any():
                m_obs = obs_t[imitation_mask]
                m_safe = safe_t[imitation_mask]
                m_delta = cont_delta[imitation_mask]
                pred_uv = torch.tanh(
                    self.model.action_parameters(m_obs)[0][:, :2])
                target_uv = m_safe[:, :2].clamp(-1, 1)
                per_dim = (pred_uv - target_uv) ** 2
                # Weight by correction magnitude, detached so the policy
                # cannot reduce its loss by changing the weight itself.
                weights = (1.0 + m_delta.detach().mean(-1) / 0.25).clamp(1.0, 5.0)
                intervention_loss = (per_dim.mean(-1) * weights).sum() \
                    / weights.sum().clamp_min(1e-6)
            else:
                intervention_loss = torch.zeros(())
            if cfg.bid_only_ppo:
                intervention_loss = torch.zeros(())

            loss = (policy_loss + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_bonus
                    + cfg.intervention_coef * intervention_loss)
            self.optimizer.zero_grad()
            loss.backward()
            bid_grad_terms = [
                parameter.grad.detach().square().sum()
                for parameter in (list(self.model.bid_head.parameters())
                                  + list(self.model.bid_candidate_head.parameters())
                                  + list(self.model.bid_conflict_residual.parameters())
                                  + [self.model.bid_log_std])
                if parameter.grad is not None]
            bid_grad_norm = torch.sqrt(
                torch.stack(bid_grad_terms).sum()
                if bid_grad_terms else torch.zeros(()))
            if cfg.task_residual_only_ppo:
                # u_xi is residual row 2. Mask its gradient before clipping,
                # then restore its value after Adam so inherited momentum or
                # weight decay cannot move this protected radar-axis control.
                xi_residual = self.model.conflict_residual.weight
                xi_residual_before = xi_residual[2].detach().clone()
                if xi_residual.grad is not None:
                    xi_residual.grad[2].zero_()
                geometry_grad_norm = torch.sqrt(sum(
                    (p.grad.detach().square().sum() for p in geometry_parameters
                     if p.grad is not None), torch.zeros(())))
                nn.utils.clip_grad_norm_(bid_parameters, 0.5)
                task_residual_grad_norm = nn.utils.clip_grad_norm_(
                    task_residual_parameters, 0.5)
                critic_grad_norm = nn.utils.clip_grad_norm_(critic_parameters, 0.5)
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            with torch.no_grad():
                if cfg.task_residual_only_ppo:
                    xi_residual[2].copy_(xi_residual_before)
                if not cfg.bid_only_ppo and not cfg.task_residual_only_ppo:
                    self.model.log_std.clamp_(-2.5, 0.0)
                self.model.bid_log_std.clamp_(-2.5, 0.0)
            if (float(approx_kl.detach()) > cfg.target_kl
                    or float(bid_approx_kl.detach()) > cfg.bid_target_kl):
                break
        bid_parameter_delta = torch.sqrt(sum(
            (before - after.detach()).square().sum()
            for before, after in zip(
                bid_parameter_before,
                list(self.model.bid_head.parameters())
                + list(self.model.bid_candidate_head.parameters())
                + list(self.model.bid_conflict_residual.parameters())
                + [self.model.bid_log_std])))
        with torch.no_grad():
            post_bid_loc = self.model.action_parameters(obs_t)[-1]
            post_bid_dist = Normal(
                post_bid_loc, self.model.bid_log_std.exp())
            post_bid_logp = _squashed_log_prob(
                post_bid_dist, bid_actions_t)
            post_bid_log_ratio = (
                post_bid_logp - old_logp_bid_slots).clamp(-10.0, 10.0)
            post_bid_ratio = post_bid_log_ratio.exp()
            post_bid_kl = (
                ((post_bid_ratio - 1.0) - post_bid_log_ratio)[bid_active_t].mean()
                if bid_active_t.any() else torch.zeros(()))
            post_policy_anchor_kl = (
                self._policy_anchor_kl(obs_t)
                if (cfg.policy_anchor_coef > 0.0 or cfg.policy_anchor_monitor)
                else obs_t.new_zeros(()))
        active_bid_actions = bid_actions_t[bid_active_t]
        stats.update({
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "xi_ppo_loss": float(loss_cont_xi.detach()),
            "xi_anchor_loss": float(xi_anchor_loss.detach()),
            "policy_anchor_kl": float(policy_anchor_kl.detach()),
            "post_update_policy_anchor_kl": float(post_policy_anchor_kl.detach()),
            "value_loss": float(value_loss.detach()),
            "intervention_loss": float(intervention_loss.detach()),
            "intervention_fraction": float(intervention_mask.float().mean()),
            "continuous_intervention_fraction": float(intervention_mask.float().mean()),
            "masked_continuous_fraction": float(intervention_mask.float().mean()),
            "flight_ppo_active_fraction": (float(cont_weight.mean())
                                           if not cfg.bid_only_ppo else 0.0),
            "radar_axis_ppo_active_fraction": (float(xi_weight.mean())
                                               if (not cfg.bid_only_ppo
                                                   and cfg.xi_ppo_coef > 0.0)
                                               else 0.0),
            "approx_kl": float(approx_kl.detach()),
            "bid_approx_kl": float(bid_approx_kl.detach()),
            "bid_post_update_kl": float(post_bid_kl.detach()),
            "bid_active_slot_count": int(bid_active_t.sum()),
            "bid_active_slot_count_by_index": [
                int(v) for v in bid_active_t.sum(0).tolist()],
            "cf_bid_informative_fraction": float(
                (bid_active_t & bid_candidate_t).sum().float()
                / bid_candidate_t.sum()) if bid_candidate_t.any() else 0.0,
            "bid_gradient_norm": float(bid_grad_norm.detach()),
            "bid_parameter_delta_l2": float(bid_parameter_delta.detach()),
            "task_residual_gradient_norm": float(task_residual_grad_norm.detach()),
            "critic_gradient_norm": float(critic_grad_norm.detach()),
            "task_residual_parameter_delta_l2": (float(torch.linalg.vector_norm(
                self.model.conflict_residual.weight.detach() - task_residual_before))
                if task_residual_before is not None else 0.0),
            "task_residual_xi_row_unchanged": (bool(torch.equal(
                self.model.conflict_residual.weight[2].detach(), task_residual_before[2]))
                if task_residual_before is not None else True),
            "geometry_adapter_gradient_norm": float(geometry_grad_norm),
            "geometry_adapter_parameter_delta_l2": float(torch.sqrt(sum(
                ((before - after.detach()).square().sum()
                 for before, after in zip(geometry_before, geometry_parameters)),
                torch.zeros(())))),
            "geometry_adapter_mean_abs_output": float(
                self.model.task_geometry_adapter(obs_t).detach().abs().mean()),
            "geometry_adapter_max_abs_output": float(
                self.model.task_geometry_adapter(obs_t).detach().abs().max()),
            "geometry_pressure_gate_nonzero_fraction": float(
                (obs_t[:, -13:].clamp(0, 1).amax(-1) > 0).float().mean()),
            "mean_return_scaled": float(returns.mean()),
            "rollout_steps": len(rollout),
            # RDAC diagnostics
            "f_role_mean": float(f_role_flat.mean()),
            "cf_delta_mean": float(cf_delta_flat.mean()),
            "cf_nonzero_frac": float((cf_delta_flat != 0).float().mean()),
            "gear_gate_active_frac": float(gear_active.float().mean()),
            "gear_gate_credit_mean": float(gear_credit_flat.mean()),
            "bid_event_frac": float(bid_mask_t.float().mean()),
            "cf_bid_event_frac": float(bid_active_t.float().mean()),
            "cf_bid_adv_mean": float(cf_bid_t[bid_active_t].mean())
            if bid_active_t.any() else 0.0,
            "bid_action_mean": float(active_bid_actions.mean())
            if active_bid_actions.numel() else 0.0,
            "bid_action_std": float(active_bid_actions.std())
            if active_bid_actions.numel() > 1 else 0.0,
            "pointer_exposure_agent_frac": float(
                action_exposure_t.float().mean()),
            "pointer_exposure_steps": int(
                action_exposure_t.reshape(-1, n).any(-1).sum()),
        })
        return stats

    # ------------------------------------------------------------------
    def behavior_clone(self, demos: Sequence[Tuple[np.ndarray, DetectAction]],
                       epochs: int = 60, batch_size: int = 4096,
                       focus_weight: float = 5.0,
                       continuous_objective: str = "likelihood") -> dict:
        """BC warm start on HEUR demonstrations (plan §9.4, 50-100 epochs).

        ``focus_weight`` up-weights TRACK+pointer steps, which are ~24% of
        teacher steps but critical for the whole task loop.  Both conditions
        share the same weight: role=TRACK (without pointer) is the prerequisite
        for any request to go through, so it needs the same emphasis."""
        if not demos:
            return {"samples": 0, "epoch": 0, "loss": 0.0,
                    "cont": 0.0, "role": 0.0, "gear": 0.0,
                    "pointer": 0.0, "bid": 0.0}
        if continuous_objective not in ("likelihood", "latent_regression"):
            raise ValueError(
                "continuous_objective must be 'likelihood' or 'latent_regression'")
        # The demonstrations teach flight/task execution.  Conflict arbitration
        # remains an RL capability and is trained only by PPO rollouts.
        self.model.conflict_residual.weight.requires_grad_(False)
        self.model.bid_conflict_residual.weight.requires_grad_(False)
        for parameter in self.model.bid_candidate_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.task_geometry_adapter.parameters():
            parameter.requires_grad_(False)
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
                (cont_loc, role_logits, base_gear_logits,
                 base_pointer_logits, bid_loc) = self.model.action_parameters(
                     ob, ac[:, 5].long().clamp(0, 8))
                # Match the bounded policy likelihood used by PPO/eval.  A
                # raw MSE on tanh(loc) gives a different objective near the
                # action limits and can make the BC likelihood worse after a
                # few epochs.
                bid_dist = Normal(bid_loc, self.model.bid_log_std.exp())
                if continuous_objective == "latent_regression":
                    # Deterministic evaluation emits tanh(loc). Regress its
                    # inverse transform so saturated expert controls retain
                    # a useful gradient toward the same bounded action.
                    target = ac[:, :3].clamp(-0.98, 0.98)
                    latent_target = _atanh_action(target)
                    cont_error = nn.functional.smooth_l1_loss(
                        cont_loc, latent_target, reduction="none").sum(-1)
                    cont_loss = (cont_error * sw).mean()
                else:
                    cont_dist = Normal(cont_loc, self.model.log_std.exp())
                    cont_nll = -_bounded_log_prob(
                        cont_dist, ac[:, :3]).sum(-1)
                    cont_loss = (cont_nll * sw).mean()
                # HEUR emits the neutral default bid (0).  BC establishes
                # that neutral prior; conflict-specific bid preferences are
                # learned later from PPO arbitration credit, never copied
                # from the teacher's tie-breaking behavior.
                bid_nll = -_bounded_log_prob(
                    bid_dist, ac[:, 6:7]).sum(-1)
                bid_loss = (bid_nll * sw).mean()
                ce = nn.functional.cross_entropy
                role_loss = (ce(role_logits, role_idx,
                                reduction="none") * sw).mean()
                gear_loss = (ce(_gear_masked_logits(base_gear_logits, role_idx,
                                                    allow_idx=ac[:, 4]),
                                ac[:, 4].long().clamp(0, 5),
                                reduction="none") * sw).mean()
                # Pointer is conditional on the demonstrated role and on
                # populated local slots.  Training it as an unconditional
                # 9-way classifier teaches SEARCH samples to compete with
                # request samples and creates invalid pointer gradients.
                pointer_target = _legal_pointer_targets(
                    ob, role_idx, ac[:, 5].long().clamp(0, 8))
                pointer_logits = _pointer_masked_logits(
                    base_pointer_logits, role_idx,
                    _encoded_request_mask(ob), pointer_target)
                pointer_loss = (ce(pointer_logits,
                                   pointer_target,
                                   reduction="none") * sw).mean()
                loss = (cont_loss + role_loss + gear_loss + pointer_loss
                        + bid_loss)
                opt.zero_grad()
                loss.backward()
                opt.step()
            last = {"epoch": epoch + 1, "loss": float(loss.detach()),
                    "cont": float(cont_loss.detach()),
                    "role": float(role_loss.detach()),
                    "gear": float(gear_loss.detach()),
                    "pointer": float(pointer_loss.detach()),
                    "bid": float(bid_loss.detach())}
        # freeze the std closer to the demonstrated (deterministic) behaviour
        # (pilot calibration: cap -0.7 -> -1.4; std 0.5 swamped tracking loops)
        with torch.no_grad():
            self.model.log_std.clamp_(-2.5, -1.4)
        for parameter in self.model.bid_candidate_head.parameters():
            parameter.requires_grad_(True)
        self._refresh_xi_anchor()
        return {"samples": n, **last}

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(),
                    "xi_anchor_model": self.xi_anchor_model.state_dict(),
                    "policy_anchor_model": self.policy_anchor_model.state_dict(),
                    "config": self.config.__dict__}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.checkpoint_migration_record = {
            "source": str(path), "geometry_adapter": {},
            "adapter_initialization_seed": 20261022,
            "legacy_output_preservation": "final geometry weights and bias are zero",
        }
        allowed_legacy_missing = {
            "conflict_residual.weight", "bid_conflict_residual.weight"}
        legacy_observation_dim = self.encoder.dimension - 18

        def expand_legacy_input(state, expected, key, chunked=False):
            old = state.get(key)
            target = expected[key]
            if (old is None or old.ndim != 2 or target.ndim != 2
                    or old.shape[0] != target.shape[0]):
                return state
            if chunked:
                if old.shape[1] != legacy_observation_dim * self.env.cfg.N_WINGMEN:
                    return state
                expanded = torch.zeros_like(target)
                for agent in range(self.env.cfg.N_WINGMEN):
                    source_start = agent * legacy_observation_dim
                    target_start = agent * self.encoder.dimension
                    expanded[:, target_start:target_start + legacy_observation_dim] = \
                        old[:, source_start:source_start + legacy_observation_dim]
            else:
                if old.shape[1] != legacy_observation_dim:
                    return state
                expanded = torch.zeros_like(target)
                expanded[:, :legacy_observation_dim] = old
            compatible = dict(state)
            compatible[key] = expanded
            return compatible

        def compatible_state(state, expected, component):
            state = dict(state)
            geometry_keys = {key for key in expected
                             if key.startswith("task_geometry_adapter.")}
            present_geometry = geometry_keys & set(state)
            if present_geometry and present_geometry != geometry_keys:
                raise RuntimeError(f"incomplete {component} geometry adapter checkpoint")
            self.checkpoint_migration_record["geometry_adapter"][component] = (
                "restored" if present_geometry else "zero_output_legacy_addition")
            state = expand_legacy_input(
                state, expected, "actor.0.weight")
            state = expand_legacy_input(
                state, expected, "critic.0.weight", chunked=True)
            legacy_residual = state.get("conflict_residual.weight")
            bid_residual_key = "bid_conflict_residual.weight"
            task_residual_shape = expected["conflict_residual.weight"].shape
            bid_residual_shape = expected[bid_residual_key].shape
            if (legacy_residual is not None
                    and bid_residual_key not in state
                    and legacy_residual.shape == (
                        task_residual_shape[0] + bid_residual_shape[0],
                        task_residual_shape[1])):
                state["conflict_residual.weight"] = legacy_residual[
                    :task_residual_shape[0]]
                state[bid_residual_key] = legacy_residual[
                    task_residual_shape[0]:]
            missing = set(expected) - set(state)
            unexpected = set(state) - set(expected)
            mismatched = {
                key: (tuple(state[key].shape), tuple(expected[key].shape))
                for key in set(state) & set(expected)
                if state[key].shape != expected[key].shape
            }
            allowed_missing = allowed_legacy_missing | {
                key for key in expected
                if key.startswith("bid_candidate_head.")} | geometry_keys
            unsupported_missing = missing - allowed_missing
            if unsupported_missing or unexpected or mismatched:
                raise RuntimeError(
                    f"incompatible DetectMAPPO {component} checkpoint: "
                    f"missing={sorted(unsupported_missing)}, "
                    f"unexpected={sorted(unexpected)}, "
                    f"shape_mismatches={mismatched}"
                )

            compatible = dict(state)
            fresh_geometry = (ObservationalTaskAdapter().state_dict()
                              if not present_geometry else {})
            for key in missing:
                # Residuals are zero-effect legacy additions. The candidate
                # head must retain its fresh initialization so PPO can learn it.
                if key in geometry_keys:
                    compatible[key] = fresh_geometry[
                        key.removeprefix("task_geometry_adapter.")].clone()
                else:
                    compatible[key] = (
                        torch.zeros_like(expected[key])
                        if key in allowed_legacy_missing else expected[key])
            return compatible

        compatible = compatible_state(
            checkpoint["model"], self.model.state_dict(), "policy")
        self.model.load_state_dict(compatible, strict=True)
        anchor_state = checkpoint.get("xi_anchor_model")
        if anchor_state is None:
            self._refresh_xi_anchor()
        else:
            anchor_compatible = compatible_state(
                anchor_state, self.xi_anchor_model.state_dict(), "xi-anchor")
            self.xi_anchor_model.load_state_dict(anchor_compatible, strict=True)
            self.xi_anchor_model.eval()
            for parameter in self.xi_anchor_model.parameters():
                parameter.requires_grad_(False)
        policy_anchor_state = checkpoint.get("policy_anchor_model")
        if policy_anchor_state is None:
            self.refresh_policy_anchor()
        else:
            policy_anchor_compatible = compatible_state(
                policy_anchor_state, self.policy_anchor_model.state_dict(),
                "policy-anchor")
            self.policy_anchor_model.load_state_dict(
                policy_anchor_compatible, strict=True)
            self.policy_anchor_model.eval()
            for parameter in self.policy_anchor_model.parameters():
                parameter.requires_grad_(False)
