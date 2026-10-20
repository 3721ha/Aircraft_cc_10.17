"""Explicit training migrations for inherited arbitration parameters.

Evaluation and ordinary checkpoint loading must never call these migrations.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn


def _tensor_record(tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    metadata = {"shape": list(value.shape), "dtype": str(value.dtype)}
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return {**metadata, "sha256": digest.hexdigest(),
            "nonzero_count": int(torch.count_nonzero(value).item())}


def candidate_repair_request(environ, *, phase: str, iteration: int) -> dict:
    """Validate opt-in before any run files are written or weights are changed."""
    flag = environ.get("MUMT_REPAIR_DEAD_CANDIDATE_HEAD", "0")
    if flag not in {"0", "1"}:
        raise ValueError("MUMT_REPAIR_DEAD_CANDIDATE_HEAD must be 0 or 1")
    enabled = flag == "1"
    if enabled and (phase != "curr" or iteration != 0):
        raise ValueError(
            "Candidate-head repair is allowed only at NEW curriculum training "
            "startup (curr iter0=0). Disable MUMT_REPAIR_DEAD_CANDIDATE_HEAD "
            "for resume or evaluation; repaired checkpoints retain their weights.")
    seed = int(environ.get("MUMT_CANDIDATE_REPAIR_SEED", "20261021"))
    if not 0 <= seed < 2**63:
        raise ValueError("MUMT_CANDIDATE_REPAIR_SEED must be in [0, 2**63)")
    return {"enabled": enabled, "seed": seed}


def repair_dead_candidate_head_for_training(model: nn.Module, *, seed: int) -> dict:
    """Revive exactly-zero inherited features without changing policy outputs.

    A zero output weight guarantees an unchanged candidate correction for every
    finite input. Only the preceding Linear weight and bias may be initialized;
    all other tensors, including a learned output bias, are left untouched.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("candidate repair seed must be an integer in [0, 2**63)")
    branch = model.bid_candidate_head
    if (not isinstance(branch, nn.Sequential) or len(branch) != 3
            or not isinstance(branch[0], nn.Linear)
            or not isinstance(branch[1], nn.Tanh)
            or not isinstance(branch[2], nn.Linear)
            or branch[0].bias is None or branch[2].bias is None
            or branch[0].out_features != branch[2].in_features):
        raise ValueError("Unsupported candidate head; expected Linear/Tanh/Linear")
    hidden, output = branch[0], branch[2]
    tracked = {
        "bid_candidate_head.0.weight": hidden.weight,
        "bid_candidate_head.0.bias": hidden.bias,
        "bid_candidate_head.2.weight": output.weight,
        "bid_candidate_head.2.bias": output.bias,
    }
    before = {name: _tensor_record(value) for name, value in tracked.items()}
    dead = all(before[name]["nonzero_count"] == 0 for name in (
        "bid_candidate_head.0.weight", "bid_candidate_head.0.bias",
        "bid_candidate_head.2.weight"))
    record = {
        "method": "zero_output_preserving_hidden_linear_reset_v1",
        "scope": "explicit initial curriculum training startup only",
        "seed": seed,
        "dead_detection": "input weight, input bias, output weight are exactly zero",
        "detected_dead": dead,
        "status": "repaired" if dead else "skipped_healthy",
        "before": before,
        "global_torch_rng_modified": False,
    }
    if dead:
        # Reproduce nn.Linear.reset_parameters with a separate CPU generator.
        # No manual_seed/fork_rng calls touch process-wide Torch RNG streams.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        weight = torch.empty(hidden.weight.shape, dtype=hidden.weight.dtype,
                             device="cpu")
        bias = torch.empty(hidden.bias.shape, dtype=hidden.bias.dtype, device="cpu")
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
        bound = 1 / math.sqrt(hidden.in_features)
        nn.init.uniform_(bias, -bound, bound, generator=generator)
        with torch.no_grad():
            hidden.weight.copy_(weight)
            hidden.bias.copy_(bias)
    after = {name: _tensor_record(value) for name, value in tracked.items()}
    record["after"] = after
    record["changed_tensors"] = [name for name in before
                                 if before[name]["sha256"] != after[name]["sha256"]]
    allowed = {"bid_candidate_head.0.weight", "bid_candidate_head.0.bias"}
    if set(record["changed_tensors"]) - allowed:
        raise RuntimeError("Candidate repair changed a protected tensor")
    record["output_preservation"] = {
        "verified": bool(dead),
        "guarantee": (
            "Output weight remains exactly zero and output bias is unchanged; "
            "the candidate head output equals its existing output bias for all "
            "finite inputs. Other model tensors are not modified."
            if dead else "Healthy branch left unchanged."),
        "output_weight_unchanged": before["bid_candidate_head.2.weight"] ==
                                   after["bid_candidate_head.2.weight"],
        "output_bias_unchanged": before["bid_candidate_head.2.bias"] ==
                                 after["bid_candidate_head.2.bias"],
    }
    return record
