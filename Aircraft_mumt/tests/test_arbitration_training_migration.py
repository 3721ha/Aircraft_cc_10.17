"""Prove inherited zero features revive only through explicit training startup."""
import ast
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from _arbitration_training import (candidate_repair_request,
                                   repair_dead_candidate_head_for_training)
from _experiment_provenance import artifact_record, add_output
from aircraft_sim.detect_mappo import DetectActorCritic


def _dead_model():
    model = DetectActorCritic()
    with torch.no_grad():
        model.bid_candidate_head[0].weight.zero_()
        model.bid_candidate_head[0].bias.zero_()
        model.bid_candidate_head[2].weight.zero_()
        model.bid_candidate_head[2].bias.fill_(0.125)
    return model


def test_explicit_repair_preserves_full_actor_outputs_and_all_protected_tensors():
    model = _dead_model()
    obs = torch.randn(11, 169)
    before_outputs = tuple(value.detach().clone() for value in model.action_parameters(obs))
    state = deepcopy(model.state_dict())
    record = repair_dead_candidate_head_for_training(model, seed=20261021)
    after_outputs = model.action_parameters(obs)
    assert all(torch.equal(left, right) for left, right in zip(before_outputs, after_outputs))
    changed = {key for key in state if not torch.equal(state[key], model.state_dict()[key])}
    assert changed == {"bid_candidate_head.0.weight", "bid_candidate_head.0.bias"}
    assert set(record["changed_tensors"]) == changed
    assert record["status"] == "repaired"
    assert record["output_preservation"]["verified"]
    assert record["output_preservation"]["output_bias_unchanged"]
    for key in changed:
        assert record["before"][key]["nonzero_count"] == 0
        assert record["after"][key]["nonzero_count"] > 0
        assert record["after"][key]["sha256"] != record["before"][key]["sha256"]


def test_repair_restores_gradient_path_across_candidate_features():
    model = _dead_model()
    branch = model.bid_candidate_head
    features = torch.zeros(2, branch[0].in_features)
    features[1, -1] = 1.0
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
    # A ranking objective cannot move an inherited all-zero two-layer branch.
    values = branch(features).flatten()
    (values[0] - values[1]).backward()
    assert torch.count_nonzero(branch[0].weight.grad) == 0
    assert torch.count_nonzero(branch[2].weight.grad) == 0
    optimizer.zero_grad()
    repair_dead_candidate_head_for_training(model, seed=53)
    # First update learns output weights; the next can also adapt hidden inputs.
    values = branch(features).flatten()
    (values[0] - values[1]).backward()
    assert torch.count_nonzero(branch[2].weight.grad) > 0
    optimizer.step()
    optimizer.zero_grad()
    values = branch(features).flatten()
    assert values[1] > values[0]
    (values[0] - values[1]).backward()
    assert torch.count_nonzero(branch[0].weight.grad) > 0


def test_repair_uses_local_rng_and_is_reproducible():
    first = _dead_model()
    second = deepcopy(first)
    different = deepcopy(first)
    rng_before = torch.get_rng_state().clone()
    one = repair_dead_candidate_head_for_training(first, seed=71)
    assert torch.equal(rng_before, torch.get_rng_state())
    two = repair_dead_candidate_head_for_training(second, seed=71)
    repair_dead_candidate_head_for_training(different, seed=72)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert one == two
    assert torch.equal(first.bid_candidate_head[0].weight,
                       second.bid_candidate_head[0].weight)
    assert not torch.equal(first.bid_candidate_head[0].weight,
                           different.bid_candidate_head[0].weight)


def test_healthy_or_already_repaired_heads_are_not_reset():
    model = _dead_model()
    repair_dead_candidate_head_for_training(model, seed=71)
    state = deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    record = repair_dead_candidate_head_for_training(model, seed=72)
    assert record["status"] == "skipped_healthy"
    assert record["changed_tensors"] == []
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert torch.equal(rng_before, torch.get_rng_state())


@pytest.mark.parametrize("phase,iteration", [("eval", 0), ("pilot", 0), ("bc", 0), ("curr", 2)])
def test_opt_in_is_rejected_outside_new_curriculum_training(phase, iteration):
    with pytest.raises(ValueError, match="NEW curriculum training"):
        candidate_repair_request({"MUMT_REPAIR_DEAD_CANDIDATE_HEAD": "1"},
                                 phase=phase, iteration=iteration)


def test_disabled_repair_is_valid_for_resume_and_evaluation():
    assert not candidate_repair_request({}, phase="curr", iteration=3)["enabled"]
    assert not candidate_repair_request({}, phase="eval", iteration=0)["enabled"]
    assert candidate_repair_request({"MUMT_REPAIR_DEAD_CANDIDATE_HEAD": "1",
                                     "MUMT_CANDIDATE_REPAIR_SEED": "87"},
                                    phase="curr", iteration=0) == {"enabled": True, "seed": 87}


@pytest.mark.parametrize("flag,seed", [("yes", "1"), ("", "1"), ("1", "-1"),
                                       ("1", str(2**63)), ("1", "invalid")])
def test_invalid_repair_request_fails_preflight(flag, seed):
    with pytest.raises(ValueError):
        candidate_repair_request({"MUMT_REPAIR_DEAD_CANDIDATE_HEAD": flag,
                                  "MUMT_CANDIDATE_REPAIR_SEED": seed},
                                 phase="curr", iteration=0)


def _runtime_helpers(output, enabled):
    source = ROOT / "_p3_pilot.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_initialize_candidate_training_branch",
             "_initialize_curriculum_anchor", "_apply_bid_std_override"}
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in names]
    namespace = {"CANDIDATE_REPAIR": {"enabled": enabled, "seed": 93}, "PHASE": "curr",
                 "OUT": output, "artifact_record": artifact_record,
                 "add_output": add_output, "math": math, "os": os,
                 "repair_dead_candidate_head_for_training": repair_dead_candidate_head_for_training}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _runtime_initializer(output, enabled):
    return _runtime_helpers(output, enabled)["_initialize_candidate_training_branch"]


def test_runtime_repair_archives_repaired_start_and_source_hash(tmp_path):
    model = _dead_model()
    def save(path):
        torch.save({"model": model.state_dict()}, path)
    policy = SimpleNamespace(model=model, save=save)
    record = {"run_id": "repair-test", "outputs": []}
    start = {"sha256": "source-sha256", "path": "source-best.pt"}
    _runtime_initializer(tmp_path, True)(policy, record, start, 0)
    repair = record["candidate_head_training_repair"]
    assert repair["source_checkpoint_sha256"] == "source-sha256"
    saved = repair["repaired_checkpoint"]
    assert saved == repair["effective_checkpoint"]
    assert saved["sha256"] == artifact_record(saved["path"], saved["role"])["sha256"]
    assert record["outputs"][-1]["sha256"] == saved["sha256"]
    state = torch.load(saved["path"], weights_only=True)["model"]
    assert torch.equal(state["bid_candidate_head.0.weight"], model.bid_candidate_head[0].weight)
    assert torch.count_nonzero(state["bid_candidate_head.2.weight"]) == 0


def test_healthy_head_archives_bid_std_override_and_refreshed_anchor(
        tmp_path, monkeypatch):
    model = DetectActorCritic()
    policy = SimpleNamespace(
        model=model, policy_anchor_model=deepcopy(model),
        config=SimpleNamespace(policy_anchor_coef=0.0, policy_anchor_monitor=False))
    def refresh_anchor():
        policy.policy_anchor_model = deepcopy(policy.model)
    def save(path):
        torch.save({"model": model.state_dict(),
                    "policy_anchor_model": policy.policy_anchor_model.state_dict()}, path)
    policy.refresh_policy_anchor = refresh_anchor
    policy.save = save
    original = deepcopy(model.state_dict())
    monkeypatch.setenv("MUMT_BID_LOG_STD", "-0.7")
    helpers = _runtime_helpers(tmp_path, True)
    record = {"run_id": "healthy-effective-test", "outputs": []}
    start = {"sha256": "healthy-source-sha", "path": "healthy-best.pt"}
    helpers["_apply_bid_std_override"](policy)
    helpers["_initialize_curriculum_anchor"](policy, record, start, 0)
    helpers["_initialize_candidate_training_branch"](policy, record, start, 0)
    repair = record["candidate_head_training_repair"]
    assert repair["status"] == "skipped_healthy"
    assert repair["changed_tensors"] == []
    assert repair["source_checkpoint_sha256"] == "healthy-source-sha"
    assert repair["source_checkpoint_path"] == "healthy-best.pt"
    assert "repaired_checkpoint" not in repair
    saved = repair["effective_checkpoint"]
    assert saved["sha256"] == artifact_record(saved["path"], saved["role"])["sha256"]
    checkpoint = torch.load(saved["path"], weights_only=True)
    assert checkpoint["model"]["bid_log_std"].item() == pytest.approx(-0.7)
    assert not torch.equal(original["bid_log_std"], checkpoint["model"]["bid_log_std"])
    assert all(torch.equal(value, checkpoint["policy_anchor_model"][key])
               for key, value in checkpoint["model"].items())
    assert record["outputs"] == [saved]


def test_new_curriculum_archives_effective_start_with_repair_disabled(tmp_path):
    model = _dead_model()
    original = deepcopy(model.state_dict())
    def save(path):
        torch.save({"model": model.state_dict()}, path)
    policy = SimpleNamespace(model=model, save=save)
    record = {"run_id": "disabled-effective-test", "outputs": []}
    start = {"sha256": "disabled-source-sha", "path": "unchanged-best.pt"}
    _runtime_initializer(tmp_path, False)(policy, record, start, 0)
    repair = record["candidate_head_training_repair"]
    assert repair["status"] == "not_requested"
    assert repair["source_checkpoint_sha256"] == "disabled-source-sha"
    assert repair["source_checkpoint_path"] == "unchanged-best.pt"
    assert repair["resume"] is False
    assert "repaired_checkpoint" not in repair
    saved = repair["effective_checkpoint"]
    assert saved["sha256"] == artifact_record(saved["path"], saved["role"])["sha256"]
    state = torch.load(saved["path"], weights_only=True)["model"]
    assert all(torch.equal(value, state[key]) for key, value in original.items())
    assert record["outputs"] == [saved]


def test_resume_never_repairs_hidden_weights(tmp_path):
    model = _dead_model()
    policy = SimpleNamespace(model=model)
    start = {"sha256": "resume-sha", "path": "latest.pt"}
    record = {}
    _runtime_initializer(tmp_path, False)(policy, record, start, 7)
    assert record["candidate_head_training_repair"]["status"] == "not_requested"
    assert record["candidate_head_training_repair"]["resume"] is True
    assert "effective_checkpoint" not in record["candidate_head_training_repair"]
    assert torch.count_nonzero(model.bid_candidate_head[0].weight) == 0
    with pytest.raises(ValueError, match="resume/evaluation"):
        _runtime_initializer(tmp_path, True)(policy, {}, start, 7)
    assert torch.count_nonzero(model.bid_candidate_head[0].weight) == 0
    assert list(tmp_path.iterdir()) == []


def test_curriculum_repairs_after_anchor_to_archive_effective_start():
    tree = ast.parse((ROOT / "_p3_pilot.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    curriculum = next(node for node in main.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "PHASE == 'curr'")
    calls = {node.value.func.id: index for index, node in enumerate(curriculum.body)
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name)}
    assert calls["_load_policy_checkpoint"] < calls["_initialize_candidate_training_branch"]
    assert calls["_load_policy_checkpoint"] < calls["_initialize_curriculum_anchor"]
    assert calls["_load_policy_checkpoint"] < calls["_apply_bid_std_override"]
    assert calls["_apply_bid_std_override"] < calls["_initialize_curriculum_anchor"]
    assert calls["_initialize_curriculum_anchor"] < calls["_initialize_candidate_training_branch"]
    baseline_assignment = next(
        node for node in ast.walk(curriculum)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript)
                and ast.unparse(target) == "baseline['checkpoint']"
                for target in node.targets))
    assert ast.unparse(baseline_assignment.value).startswith(
        "run_record['candidate_head_training_repair'].get('effective_checkpoint',")
    # The migration's source is frozen with the run, as is the entry script.
    assert "_arbitration_training.py" in ast.unparse(tree)
