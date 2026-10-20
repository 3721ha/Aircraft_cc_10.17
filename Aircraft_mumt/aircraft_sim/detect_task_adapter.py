"""Bounded task residuals from the existing local observation encoding.

The last 13 observation entries are soft pressure proxies, not ground-truth
rule-conflict indicators.  This module neither reads environment state nor
provides expert labels.  Its zero output initialization preserves the policy
to which its residual is added.
"""
import math

import torch
from torch import nn


class ObservationalTaskAdapter(nn.Module):
    """Map local 169-D observations to 20 bounded task residuals.

    Output index 2 controls the radar axis in the policy's task parameter
    layout and is always masked to zero, independently of learned weights or
    optimizer momentum.  ``scale`` is fixed and saved in the state dict.
    Hidden parameters use ordinary linear-layer initialization with a private
    CPU generator; construction leaves the global Torch RNG unchanged.  The
    final layer starts at zero so its weights learn before the hidden layer.
    """

    OBSERVATION_DIM = 169
    OUTPUT_DIM = 20
    PRESSURE_DIM = 13
    HIDDEN_DIM = 64
    PROTECTED_OUTPUT_INDEX = 2

    def __init__(self, scale: float = 1.0, seed: int = 20261022):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("task adapter scale must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("task adapter seed must be an integer in [0, 2**63)")
        with torch.random.fork_rng(devices=[]):
            self.input_layer = nn.Linear(self.OBSERVATION_DIM, self.HIDDEN_DIM)
            self.output_layer = nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            nn.init.kaiming_uniform_(self.input_layer.weight, a=math.sqrt(5),
                                    generator=generator)
            bound = 1.0 / math.sqrt(self.OBSERVATION_DIM)
            nn.init.uniform_(self.input_layer.bias, -bound, bound, generator=generator)
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)
        self.register_buffer("output_scale", torch.tensor(float(scale)))
        output_mask = torch.ones(self.OUTPUT_DIM)
        output_mask[self.PROTECTED_OUTPUT_INDEX] = 0.0
        # A fixed architectural mask is reconstructed, never restored from a
        # checkpoint that could accidentally turn protected outputs back on.
        self.register_buffer("output_mask", output_mask, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "output_scale"
        if key in state_dict:
            incoming_scale = state_dict[key]
            if (not isinstance(incoming_scale, torch.Tensor)
                    or incoming_scale.shape != self.output_scale.shape
                    or not torch.equal(incoming_scale.detach().to(
                                           dtype=self.output_scale.dtype, device="cpu"),
                                       self.output_scale.detach().cpu())):
                error_msgs.append(f"{key}: checkpoint task adapter scale must match "
                                  "the fixed constructor scale")
                # A failed load must not overwrite the configured deployment
                # bound before PyTorch raises its accumulated load errors.
                state_dict[key] = self.output_scale.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        if encoded_obs.ndim < 1 or encoded_obs.shape[-1] != self.OBSERVATION_DIM:
            raise ValueError("task adapter expects observations with last dimension 169")
        if not torch.is_floating_point(encoded_obs):
            raise TypeError("task adapter observations must be floating point")
        pressure = encoded_obs[..., -self.PRESSURE_DIM:].clamp(0.0, 1.0)
        gate = pressure.amax(dim=-1, keepdim=True)
        hidden = torch.tanh(self.input_layer(encoded_obs))
        residual = torch.tanh(self.output_layer(hidden))
        return residual * gate * self.output_scale * self.output_mask
