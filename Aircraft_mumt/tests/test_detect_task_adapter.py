"""Local-only residual initialization, protection, and learnability checks."""
from copy import deepcopy
import io

import pytest
import torch

from aircraft_sim.detect_task_adapter import ObservationalTaskAdapter


def _observations(*shape):
    generator = torch.Generator().manual_seed(483)
    obs = torch.randn(*shape, 169, generator=generator)
    obs[..., -13:] = torch.rand(*shape, 13, generator=generator)
    return obs


def _activate_output(adapter):
    with torch.no_grad():
        adapter.output_layer.weight.fill_(0.03)
        adapter.output_layer.bias.fill_(0.2)


def test_zero_initialization_preserves_policy_parameters_exactly():
    adapter = ObservationalTaskAdapter()
    obs = _observations(9)
    base = torch.linspace(-2.0, 2.0, 180).reshape(9, 20)
    assert torch.count_nonzero(adapter.input_layer.weight) > 0
    assert torch.count_nonzero(adapter.output_layer.weight) == 0
    assert torch.count_nonzero(adapter.output_layer.bias) == 0
    residual = adapter(obs)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(base + residual, base)


def test_construction_uses_reproducible_private_rng():
    before = torch.get_rng_state().clone()
    first = ObservationalTaskAdapter(seed=34)
    assert torch.equal(before, torch.get_rng_state())
    second = ObservationalTaskAdapter(seed=34)
    different = ObservationalTaskAdapter(seed=35)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(first.input_layer.weight, second.input_layer.weight)
    assert torch.equal(first.input_layer.bias, second.input_layer.bias)
    assert not torch.equal(first.input_layer.weight, different.input_layer.weight)


def test_soft_pressure_gate_clamps_and_uses_maximum_local_proxy():
    adapter = ObservationalTaskAdapter(scale=0.4)
    with torch.no_grad():
        adapter.output_layer.bias.fill_(0.8)
    obs = torch.zeros(4, 169)
    obs[0, -13:] = -2.0
    obs[1, -1] = 0.25
    obs[2, -13:] = 1.0
    obs[3, -7] = 3.0
    result = adapter(obs)
    expected = 0.4 * torch.tanh(torch.tensor(0.8))
    assert torch.equal(result[0], torch.zeros(20))
    assert result[1, 0].item() == pytest.approx(float(expected / 4))
    assert result[2, 0].item() == pytest.approx(float(expected))
    assert torch.equal(result[2], result[3])
    assert torch.count_nonzero(result[:, 2]) == 0


def test_axis_stays_zero_with_nonzero_weights_and_old_optimizer_momentum():
    adapter = ObservationalTaskAdapter()
    _activate_output(adapter)
    with torch.no_grad():
        adapter.output_layer.weight[2].fill_(3.0)
        adapter.output_layer.bias[2] = 2.0
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1, momentum=0.9)
    for parameter in adapter.parameters():
        optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)
    obs = _observations(8)
    protected_before = adapter.output_layer.weight[2].detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        result = adapter(obs)
        assert torch.equal(result[:, 2], torch.zeros(8))
        result.sum().backward()
        assert torch.count_nonzero(adapter.output_layer.weight.grad[2]) == 0
        assert adapter.output_layer.bias.grad[2] == 0.0
        optimizer.step()
    assert not torch.equal(protected_before, adapter.output_layer.weight[2])
    assert torch.count_nonzero(adapter(obs)[:, 2]) == 0


def test_adapter_can_distinguish_local_geometry_with_identical_pressure():
    adapter = ObservationalTaskAdapter()
    with torch.no_grad():
        adapter.input_layer.weight.zero_()
        adapter.input_layer.bias.zero_()
        adapter.input_layer.weight[0, 17] = 1.0
        adapter.output_layer.weight[0, 0] = 1.0
    obs = torch.zeros(2, 169)
    obs[:, -1] = 0.5
    obs[1, 17] = 0.75
    assert torch.equal(obs[0, -13:], obs[1, -13:])
    values = adapter(obs)
    assert values[0, 0] == 0.0
    assert values[1, 0] > values[0, 0]


def test_first_update_learns_final_layer_and_second_can_learn_hidden_geometry():
    adapter = ObservationalTaskAdapter()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    obs = _observations(5)
    obs[:, -1] = 1.0
    initial_hidden = adapter.input_layer.weight.detach().clone()
    (-adapter(obs)[:, 0].mean()).backward()
    assert torch.count_nonzero(adapter.output_layer.weight.grad[0]) > 0
    assert torch.count_nonzero(adapter.input_layer.weight.grad) == 0
    optimizer.step()
    assert torch.equal(adapter.input_layer.weight, initial_hidden)
    optimizer.zero_grad()
    (-adapter(obs)[:, 0].mean()).backward()
    assert torch.count_nonzero(adapter.input_layer.weight.grad) > 0
    assert torch.count_nonzero(adapter.input_layer.bias.grad) > 0
    optimizer.step()
    assert not torch.equal(adapter.input_layer.weight, initial_hidden)


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
def test_shape_dtype_and_finite_bounded_outputs(shape):
    adapter = ObservationalTaskAdapter(scale=0.7).double()
    _activate_output(adapter)
    obs = _observations(*shape).double()
    result = adapter(obs)
    assert result.shape == shape + (20,)
    assert result.dtype == torch.float64
    assert torch.isfinite(result).all()
    assert (result.abs() <= 0.7).all()
    assert (result[..., 2] == 0).all()


@pytest.mark.parametrize("shape", [(), (168,), (2, 170), (2, 0)])
def test_wrong_observation_shape_rejected(shape):
    with pytest.raises(ValueError, match="last dimension 169"):
        ObservationalTaskAdapter()(torch.zeros(shape))


def test_integer_observation_rejected():
    with pytest.raises(TypeError, match="floating point"):
        ObservationalTaskAdapter()(torch.zeros(169, dtype=torch.long))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        ObservationalTaskAdapter(scale=scale)


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "3"])
def test_invalid_private_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        ObservationalTaskAdapter(seed=seed)


def test_state_dict_round_trip_preserves_output_scale_and_axis_protection():
    source = ObservationalTaskAdapter(scale=0.35)
    _activate_output(source)
    obs = _observations(5)
    state = deepcopy(source.state_dict())
    assert "output_scale" in state
    assert "output_mask" not in state
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored = ObservationalTaskAdapter(scale=0.35)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(source(obs), restored(obs))
    assert torch.equal(source.output_scale, restored.output_scale)
    assert torch.count_nonzero(restored(obs)[:, 2]) == 0


@pytest.mark.parametrize("bad_scale", [torch.tensor(0.3), torch.tensor(float("nan")),
                                      torch.tensor([1.0])])
def test_checkpoint_cannot_change_fixed_scale(bad_scale):
    adapter = ObservationalTaskAdapter(scale=1.0)
    state = deepcopy(adapter.state_dict())
    state["output_scale"] = bad_scale
    with pytest.raises(RuntimeError, match="must match the fixed constructor scale"):
        adapter.load_state_dict(state)
    assert adapter.output_scale == 1.0
