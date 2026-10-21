"""Adaptation protocol regression tests using synthetic weights and short fakes."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import _train_sensor_adaptation as runner
from aircraft_sim.detect_mappo import DetectMAPPOConfig


def arguments(tmp_path, *extra):
    checkpoint = tmp_path / "source_best.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"synthetic preflight checkpoint")
    return runner.parser().parse_args([
        "--output", str(tmp_path / "output"),
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        *extra,
    ])


def test_plan_only_does_not_import_runtime_or_create_output(tmp_path):
    args = arguments(tmp_path)
    script = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('aircraft_sim'):
            raise AssertionError('plan imported runtime: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
sys.path.insert(0, sys.argv.pop(1))
import _train_sensor_adaptation as runner
runner.main()
"""
    result = subprocess.run([
        sys.executable, "-B", "-c", script, str(ROOT),
        "--output", str(args.output), "--checkpoint", str(args.checkpoint),
        "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
        "--plan-only",
    ], cwd=tmp_path, text=True, capture_output=True, check=True, timeout=30)
    plan = json.loads(result.stdout)
    assert result.stderr == ""
    assert not args.output.exists()
    assert list(tmp_path.iterdir()) == [args.checkpoint]
    assert plan["task"] == {
        "K_RED": 4, "STABLE_WINDOW": 8, "AGE_STD_TRACK": 10.0,
        "RADAR_HIT_MODEL": "deterministic_beam", "W8_SCALE": 1.0,
    }
    assert plan["training_seeds"] == list(range(206000, 206096))
    assert plan["validation_seeds"] == list(range(207000, 207012))
    assert plan["reserved_test_seeds_not_run"] == list(range(208000, 208030))
    assert plan["no_final_test"] and plan["no_training_teacher"]
    assert plan["no_conflict_injection"]


def test_preflight_rejects_existing_output_without_touching_it(tmp_path):
    args = arguments(tmp_path)
    args.output.mkdir()
    marker = args.output / "keep.txt"
    marker.write_bytes(b"keep")
    with pytest.raises(ValueError, match="output already exists"):
        runner.preflight(args)
    assert list(args.output.iterdir()) == [marker]
    assert marker.read_bytes() == b"keep"


@pytest.mark.parametrize("change", ["wrong", "missing", "invalid_hash"])
def test_preflight_checkpoint_gate_leaves_no_output(tmp_path, change):
    args = arguments(tmp_path)
    if change == "wrong":
        args.expected_checkpoint_sha256 = "0" * 64
    elif change == "missing":
        args.checkpoint = tmp_path / "missing.pt"
    else:
        args.expected_checkpoint_sha256 = "G" * 64
    with pytest.raises(ValueError, match="SHA256"):
        runner.preflight(args)
    assert not args.output.exists()


@pytest.mark.parametrize("name,value", [
    ("iterations", 0), ("episodes_per_iter", 0), ("validation_every", 0),
    ("validation_episodes", 0), ("epochs", 0), ("torch_threads", 0),
    ("train_seed_start", -1), ("validation_seed_start", -1),
    ("optimizer_seed", -1), ("learning_rate", 0),
    ("learning_rate", float("nan")), ("learning_rate", float("inf")),
])
def test_preflight_invalid_numeric_arguments_leave_no_output(tmp_path, name, value):
    args = arguments(tmp_path)
    setattr(args, name, value)
    with pytest.raises(ValueError):
        runner.preflight(args)
    assert not args.output.exists()


@pytest.mark.parametrize("role", ["train_seed_start", "validation_seed_start"])
@pytest.mark.parametrize("seed", [162000, 162029, 205000, 205029, 208000, 208029])
def test_consumed_and_reserved_seeds_rejected_for_both_roles(tmp_path, role, seed):
    args = arguments(tmp_path)
    setattr(args, role, seed)
    with pytest.raises(ValueError, match="reserved test or consumed"):
        runner.preflight(args)
    assert not args.output.exists()


def test_seed_overlap_is_checked_over_whole_training_range(tmp_path):
    args = arguments(tmp_path, "--train-seed-start", "206999")
    with pytest.raises(ValueError, match="training and validation seeds overlap"):
        runner.preflight(args)
    args = arguments(tmp_path, "--train-seed-start", "204999")
    with pytest.raises(ValueError, match="reserved test or consumed"):
        runner.preflight(args)


@pytest.mark.parametrize("mode", ["bid_only_ppo", "task_residual_only_ppo"])
def test_joint_config_clears_freeze_modes_without_mutating_saved_config(mode):
    saved = asdict(DetectMAPPOConfig(
        **{mode: True}, direct_rl_arbitration=True,
        entropy_coef=0.012345, role_adv_coef=1.234,
        conflict_exposure_prob=0.7, xi_ppo_coef=0.0,
        xi_anchor_coef=1.0, policy_anchor_coef=9.0))
    before = deepcopy(saved)
    config = runner.adaptation_config(saved, learning_rate=2e-5, epochs=3)
    expected = dict(saved, learning_rate=2e-5, epochs=3,
                    bid_only_ppo=False, task_residual_only_ppo=False,
                    train_geometry_adapter=False, direct_rl_arbitration=True,
                    conflict_exposure_prob=0.0, xi_ppo_coef=1.0,
                    xi_anchor_coef=0.0, policy_anchor_coef=0.5,
                    policy_anchor_monitor=True,
                    mask_intervened_continuous=True,
                    mask_intervened_radar_axis=False, cf_bid_horizon=1)
    assert asdict(config) == expected
    assert saved == before
    config.validate_training_mode()
    # Historical policies retain masking unless this runner explicitly opts out.
    assert DetectMAPPOConfig().mask_intervened_radar_axis is True


def test_selection_is_success_then_raw_return_then_lower_pooled_violations():
    base = {"successes": 2, "mean_return": -100.0, "violations_per_step": 4.0}
    assert runner.selection_rank(dict(base)) == runner.selection_rank(base)
    assert not runner.selection_rank(dict(base)) > runner.selection_rank(base)
    assert runner.selection_rank(dict(base, successes=3, mean_return=-99999)) > runner.selection_rank(base)
    assert runner.selection_rank(dict(base, mean_return=-99, violations_per_step=999)) > runner.selection_rank(base)
    assert runner.selection_rank(dict(base, violations_per_step=3.9)) > runner.selection_rank(base)
    assert runner.selection_rank(dict(base, successes=1, mean_return=99999)) < runner.selection_rank(base)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_restores_training_torch_rng_even_on_exception(monkeypatch, fail):
    calls = []
    def fake_episode(cfg, controller, seed, checkpoint, saved):
        calls.append((controller, seed))
        torch.manual_seed(0)
        torch.rand(11)
        if fail:
            raise RuntimeError("synthetic evaluation failure")
        return {"seed": seed}
    monkeypatch.setattr(runner, "run_episode", fake_episode)
    torch.manual_seed(9123)
    before = torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
            runner.evaluate_checkpoint(None, Path("unused"), {}, [207000, 207001])
    else:
        assert runner.evaluate_checkpoint(None, Path("unused"), {}, [207000, 207001]) == [
            {"seed": 207000}, {"seed": 207001}]
        assert calls == [("rl", 207000), ("rl", 207001)]
    assert torch.equal(torch.get_rng_state(), before)


def test_parameter_changes_separates_radar_flight_bid_and_fixed_modules():
    before = {
        "actor.weight": torch.zeros(1),
        "cont_head.weight": torch.zeros(3, 1),
        "cont_head.bias": torch.zeros(3), "log_std": torch.zeros(3),
        "role_head.weight": torch.zeros(1), "gear_head.weight": torch.zeros(1),
        "pointer_head.weight": torch.zeros(1), "bid_head.weight": torch.zeros(1),
        "bid_candidate_head.weight": torch.zeros(1),
        "bid_conflict_residual.weight": torch.zeros(1), "bid_log_std": torch.zeros(1),
        "conflict_residual.weight": torch.zeros(3, 1),
        "critic.weight": torch.zeros(1), "task_geometry_adapter.weight": torch.zeros(1),
    }
    after = {k: v.clone() for k, v in before.items()}
    after["actor.weight"] += 1
    after["cont_head.weight"][:, 0] = torch.tensor([3.0, 4.0, 12.0])
    after["conflict_residual.weight"][2, 0] = 5.0
    for name in ("bid_head.weight", "bid_candidate_head.weight",
                 "bid_conflict_residual.weight", "bid_log_std"):
        after[name] += 2
    delta = runner.parameter_changes(before, after)
    assert delta["actor"] == 1.0
    assert delta["flight_controls"] == 5.0
    assert delta["radar_axis"] == 13.0
    assert delta["task_conflict_residual"] == 5.0
    assert delta["bid"] == 4.0
    assert all(delta[k] == 0 for k in (
        "role", "gear", "pointer", "critic", "geometry_adapter"))
    assert all(not value.any() for value in before.values())


def test_digest_and_copied_state_detect_fixed_module_change():
    module = torch.nn.Linear(2, 3)
    before = runner.tensor_state(module)
    fingerprint = runner.module_digest(module)
    assert fingerprint == runner.module_digest(deepcopy(module))
    with torch.no_grad():
        module.weight[0, 0] += 1
    assert fingerprint != runner.module_digest(module)
    assert not torch.equal(before["weight"], module.weight)


@pytest.mark.parametrize("scores,selected", [
    ([(2, 10.0, 1.0), (1, 100.0, 0.0), (2, 10.0, 1.0)], 0),
    ([(0, 10.0, 1.0), (1, 5.0, 1.0), (0, 100.0, 0.0)], 1),
])
def test_short_fake_main_selects_best_not_latest_and_never_runs_test(
        tmp_path, monkeypatch, scores, selected):
    """Exercise save/load/selection wiring with one fake step per episode."""
    import aircraft_sim.detect_mappo as mappo
    import aircraft_sim.detect_sensor_diagnostics as diagnostics
    import aircraft_sim.detect_shield as shield

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(1, 1)
            self.cont_head = torch.nn.Linear(1, 3)
            self.task_geometry_adapter = torch.nn.Linear(1, 1)

    class Env:
        def __init__(self, cfg):
            self.cfg = cfg
            self.judged = {0: True}
        def sensor_diagnostics(self):
            return {"fake": True}

    training_seeds, evaluation_calls = [], []
    class Policy:
        def __init__(self, env, _shield, config, seed):
            self.env, self.config = env, config
            self.model = Model()
            self.xi_anchor_model = deepcopy(self.model)
            self.policy_anchor_model = deepcopy(self.model)
            self.checkpoint_migration_record = {"synthetic": True}
        def load(self, path):
            data = torch.load(path, weights_only=False)
            for model in (self.model, self.xi_anchor_model, self.policy_anchor_model):
                model.load_state_dict(data["model"])
        def refresh_policy_anchor(self):
            self.policy_anchor_model.load_state_dict(self.model.state_dict())
        def save(self, path):
            torch.save({"model": self.model.state_dict(), "config": asdict(self.config)}, path)
        def collect(self, *, k_red, seed):
            assert k_red == 4
            training_seeds.append(seed)
            return [{"done": True, "terminal": "success", "environment_reward": 1.0,
                     "reward": 0.1, "violations": 0, "n_multi_candidate_events": 0,
                     "n_cf_bid_informative_events": 0, "n_cf_bid_candidate_branches": 0,
                     "bid_cf_actual_regrets": [], "n_requests": 1,
                     "n_c04_rejected_requests": 0}]
        def update(self, rollout):
            assert len(rollout) == 1
            with torch.no_grad():
                self.model.actor.weight.add_(0.01)
                self.model.cont_head.weight[2].add_(0.01)
            return {"loss": 0.0}

    checkpoint = tmp_path / "source_best.pt"
    saved = asdict(DetectMAPPOConfig(direct_rl_arbitration=True,
                                   task_residual_only_ppo=True))
    torch.save({"model": Model().state_dict(), "config": saved}, checkpoint)
    args = arguments(tmp_path, "--iterations", "2", "--episodes-per-iter", "1",
                     "--validation-every", "1", "--validation-episodes", "2")
    def snapshot(_root, output, _scripts):
        (output / "provenance").mkdir(parents=True)
        return {"files": []}
    def fake_episode(cfg, controller, seed, path, config):
        evaluation_calls.append((controller, seed, Path(path).name))
        assert cfg.RADAR_HIT_MODEL == "deterministic_beam"
        assert seed in (207000, 207001)
        torch.manual_seed(0)
        if controller == "heur":
            successes, reward, violations = 2, 50.0, 1.0
        else:
            update = 0 if Path(path).name == "initial.pt" else int(Path(path).stem.split("_")[-1])
            successes, reward, violations = scores[update]
        return {"terminal": "success" if seed - 207000 < successes else "cutoff",
                "return": reward, "violations_per_step": violations}
    def summarize(rows):
        successes = sum(row["terminal"] == "success" for row in rows)
        return {"successes": successes, "success_rate": successes / len(rows),
                "mean_return": sum(row["return"] for row in rows) / len(rows),
                "violations_per_step": sum(row["violations_per_step"] for row in rows) / len(rows)}
    monkeypatch.setattr(mappo, "DetectMAPPOPolicy", Policy)
    monkeypatch.setattr(diagnostics, "SensorDiagnosticEnv", Env)
    monkeypatch.setattr(shield, "DetectQPShield", lambda cfg: None)
    monkeypatch.setattr(runner, "capture_source_snapshot", snapshot)
    monkeypatch.setattr(runner, "capture_run_provenance", lambda *a, **k: {
        "effective_source": {"files": []}, "outputs": []})
    monkeypatch.setattr(runner, "run_episode", fake_episode)
    monkeypatch.setattr(runner, "summarize", summarize)
    monkeypatch.setattr(runner, "paired_difference", lambda left, right: {})
    monkeypatch.setattr(sys, "argv", ["adaptation", "--output", str(args.output),
        "--checkpoint", str(checkpoint), "--expected-checkpoint-sha256",
        args.expected_checkpoint_sha256, "--iterations", "2", "--episodes-per-iter", "1",
        "--validation-every", "1", "--validation-episodes", "2"])
    runner.main()
    log = json.loads((args.output / "adaptation_log.json").read_text())
    provenance = json.loads((args.output / "adaptation.provenance.json").read_text())
    assert log["status"] == "completed_short_adaptation"
    assert log["best"]["update"] == selected
    assert log["latest"]["update"] == 2
    assert log["best"]["checkpoint"]["sha256"] != log["latest"]["checkpoint"]["sha256"]
    assert hashlib.sha256((args.output / "best.pt").read_bytes()).hexdigest() == log["best"]["checkpoint"]["sha256"]
    assert log["validations"][2]["selected_as_best"] is False
    assert training_seeds == [206000, 206001]
    assert len(evaluation_calls) == 8
    assert sum(controller == "heur" for controller, _, _ in evaluation_calls) == 2
    assert provenance["status"] == "completed"
    assert provenance["reserved_test_run"] is False
    assert provenance["frozen_modules_unchanged"] is True
    assert log["training_episodes_completed"] == 2
    assert all(item["frozen_modules_unchanged"] for item in log["training"])
    assert checkpoint.read_bytes() == (args.output / "provenance/source_best.pt").read_bytes()
