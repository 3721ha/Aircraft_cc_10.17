"""Guard the runtime wiring between checkpoint loading and frozen references."""
import ast
from copy import deepcopy
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _functions(*names):
    tree = ast.parse((ROOT / "_p3_pilot.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = {"os": os, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 str(ROOT / "_p3_pilot.py"), "exec"), namespace)
    return [namespace[name] for name in names]


class ReferencePolicy:
    def __init__(self, current, reference):
        self.current = deepcopy(current)
        self.reference = deepcopy(reference)
        self.refresh_count = 0
        self.config = SimpleNamespace(policy_anchor_coef=1.0,
                                      policy_anchor_monitor=True)

    def refresh_policy_anchor(self):
        self.reference = deepcopy(self.current)
        self.refresh_count += 1


def test_new_refinement_reanchors_to_the_loaded_policy():
    initialize, = _functions("_initialize_curriculum_anchor")
    policy = ReferencePolicy({"actor": [2.0, 3.0]}, {"actor": [-9.0, -9.0]})
    record = {}
    initialize(policy, record, {"sha256": "selected-best-sha"}, 0)
    assert policy.reference == policy.current
    assert policy.reference is not policy.current
    assert policy.refresh_count == 1
    assert record["policy_anchor"]["initial_checkpoint_sha256"] == "selected-best-sha"
    assert record["policy_anchor"]["monitor_enabled"] is True


def test_refining_a_checkpoint_with_an_existing_reference_replaces_that_reference():
    initialize, = _functions("_initialize_curriculum_anchor")
    # Loading a checkpoint restores both its current actor and its older
    # reference. A separate refinement must use the selected current actor.
    policy = ReferencePolicy({"actor": [4.0]}, {"actor": [1.0]})
    inherited = deepcopy(policy.reference)
    initialize(policy, {}, {"sha256": "new-best-sha"}, 0)
    assert policy.reference == {"actor": [4.0]}
    assert policy.reference != inherited


def test_resuming_a_run_keeps_the_original_serialized_reference():
    initialize, = _functions("_initialize_curriculum_anchor")
    policy = ReferencePolicy({"actor": [4.0]}, {"actor": [1.0]})
    reference = policy.reference
    record = {}
    initialize(policy, record, {"sha256": "latest-sha"}, 8)
    assert policy.reference is reference
    assert policy.reference == {"actor": [1.0]}
    assert policy.refresh_count == 0
    assert record["policy_anchor"]["reference"] == "reference restored from checkpoint"
    assert record["policy_anchor"]["initial_checkpoint_sha256"] is None


def test_curriculum_initialization_runs_after_load_and_before_baseline():
    tree = ast.parse((ROOT / "_p3_pilot.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    curriculum = next(node for node in main.body
                      if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "PHASE == 'curr'")
    direct_calls = {
        node.value.func.id: index
        for index, node in enumerate(curriculum.body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    }
    initialize_index = direct_calls["_initialize_curriculum_anchor"]
    assert direct_calls["_load_policy_checkpoint"] < initialize_index
    baseline_index = next(index for index, node in enumerate(curriculum.body)
                          if any(isinstance(child, ast.Call)
                                 and isinstance(child.func, ast.Name)
                                 and child.func.id == "evaluate_paired"
                                 for child in ast.walk(node)))
    assert initialize_index < baseline_index
    pilot = next(node for node in main.body
                 if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "PHASE in ('all', 'pilot')")
    assert not any(isinstance(node, ast.Name) and node.id == "CURR_START"
                   for node in ast.walk(pilot))


@pytest.fixture
def configured_policy(monkeypatch):
    for key in tuple(os.environ):
        if key.upper().startswith("MUMT_"):
            monkeypatch.delenv(key)
    return SimpleNamespace(
        config=SimpleNamespace(
            learning_rate=1e-4, epochs=2, shield_aware_continuous=False,
            cf_request_credit=0.25, conflict_exposure_prob=1.0,
            bid_only_ppo=False, direct_rl_arbitration=True,
            policy_anchor_coef=0.0, policy_anchor_monitor=False),
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
    )


@pytest.mark.parametrize("coefficient,monitor", [("0", "1"), ("1", "1"), ("0.5", "0")])
def test_runtime_controls_apply_without_disabling_zero_coefficient_monitoring(
        monkeypatch, configured_policy, coefficient, monitor):
    configure, _ = _functions("_configure_policy", "_apply_bid_std_override")
    for key, value in {
        "MUMT_POLICY_ANCHOR_COEF": coefficient,
        "MUMT_POLICY_ANCHOR_MONITOR": monitor,
        "MUMT_CONFLICT_EXPOSURE_PROB": "0",
        "MUMT_PPO_LR": "0.00001",
        "MUMT_PPO_EPOCHS": "2",
    }.items():
        monkeypatch.setenv(key, value)
    configure(configured_policy)
    assert configured_policy.config.policy_anchor_coef == float(coefficient)
    assert configured_policy.config.policy_anchor_monitor is (monitor == "1")
    assert configured_policy.config.conflict_exposure_prob == 0.0
    assert configured_policy.config.learning_rate == 1e-5
    assert configured_policy.optimizer.param_groups[0]["lr"] == 1e-5


def test_runtime_defaults_preserve_the_unregularized_training_path(configured_policy):
    configure, _ = _functions("_configure_policy", "_apply_bid_std_override")
    configure(configured_policy)
    assert configured_policy.config.policy_anchor_coef == 0.0
    assert configured_policy.config.policy_anchor_monitor is False


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1", "", "invalid"])
def test_runtime_rejects_invalid_policy_anchor_coefficient(
        monkeypatch, configured_policy, value):
    configure, _ = _functions("_configure_policy", "_apply_bid_std_override")
    monkeypatch.setenv("MUMT_POLICY_ANCHOR_COEF", value)
    with pytest.raises(ValueError):
        configure(configured_policy)


@pytest.mark.parametrize("value", ["", "true", "2", "-1", " 1"])
def test_runtime_rejects_ambiguous_monitor_values(monkeypatch, configured_policy, value):
    configure, _ = _functions("_configure_policy", "_apply_bid_std_override")
    monkeypatch.setenv("MUMT_POLICY_ANCHOR_MONITOR", value)
    with pytest.raises(ValueError, match="MUMT_POLICY_ANCHOR_MONITOR"):
        configure(configured_policy)
