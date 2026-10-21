"""Frozen sensor protocol arithmetic and preflight; no task performance claims."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import _diagnose_sensor_profile as runner
from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_mappo import DetectMAPPOConfig


def _row(seed=205000, terminal="success", steps=10, violations=10, reward=20.0,
         profile="bernoulli", controller="rl"):
    return {
        "sensor_profile": profile, "controller": controller, "seed": seed,
        "terminal": terminal, "steps": steps, "violations": violations,
        "violations_per_step": violations / steps, "return": reward,
        "judged_targets": 4 if terminal == "success" else 1,
        "violations_by_family": {"C04": violations},
        "active_agent_steps_by_conflict_pair": {"S04-M02": 3},
        "multi_candidate_events": 2,
        "winners_different_from_fixed_score": 1,
        "loaded_tensor_sha256": "a" * 64,
        "sensor_diagnostics": {
            "radar": {
                "all_targets": {"opportunities": 20, "hits": 12, "misses": 8},
                "unfinished_targets": {"opportunities": 10, "hits": 7, "misses": 3},
            },
            "verdict": {
                "stable_progress_resets": 2,
                "reset_causes": {"stale_owner_track": 2},
                "owner_gate_failure_causes": {"stale_owner_track": 5},
                "unowned_tracking_target_steps": 6,
            },
        },
    }


def test_import_does_not_import_simulator_or_torch_or_write_run_artifacts(tmp_path):
    # A fresh process catches eager imports masked by pytest's loaded modules.
    script = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('aircraft_sim'):
            raise AssertionError('runtime import at module startup: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
sys.path.insert(0, sys.argv[1])
import _diagnose_sensor_profile
print('imported')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(ROOT)], cwd=tmp_path,
        capture_output=True, text=True, check=True, timeout=30)
    assert result.stdout.strip() == "imported"
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_configs_change_only_hit_model_and_keep_explicit_task_parameters():
    configs = runner.build_configs(4, 8, 10.0)
    baseline = DetectSimConfig()
    for profile, cfg in configs.items():
        assert cfg.K_RED == 4 and cfg.STABLE_WINDOW == 8 and cfg.AGE_STD_TRACK == 10.0
        expected = asdict(baseline)
        expected.update(K_RED=4, STABLE_WINDOW=8, AGE_STD_TRACK=10.0,
                        RADAR_HIT_MODEL=profile)
        assert asdict(cfg) == expected
    left, right = [asdict(configs[profile]) for profile in runner.PROFILES]
    assert [key for key in left if left[key] != right[key]] == ["RADAR_HIT_MODEL"]


@pytest.mark.parametrize("k,window,age", [
    (0, 8, 10.0), (13, 8, 10.0), (4, 0, 10.0), (4, 8, 0.0),
    (4, 8, float("nan")), (4, 8, float("inf")),
])
def test_configs_reject_invalid_task_parameters(k, window, age):
    with pytest.raises(ValueError):
        runner.build_configs(k, window, age)


def test_saved_policy_config_is_preserved_except_training_exposure():
    saved = asdict(DetectMAPPOConfig(
        direct_rl_arbitration=True, conflict_exposure_prob=1.0,
        bid_only_ppo=True, entropy_coef=0.00123))
    before = deepcopy(saved)
    config = runner.evaluation_policy_config(saved)
    expected = dict(saved, conflict_exposure_prob=0.0)
    assert asdict(config) == expected
    assert config.direct_rl_arbitration is True
    assert saved == before


@pytest.mark.parametrize("saved,match", [
    (None, "saved policy configuration"),
    ({"direct_rl_arbitration": False}, "direct RL arbitration"),
    ({"direct_rl_arbitration": True, "unknown_future_field": 1}, "unsupported"),
])
def test_policy_config_refuses_missing_unsupported_or_fixed_score_policy(saved, match):
    with pytest.raises(ValueError, match=match):
        runner.evaluation_policy_config(saved)


def test_summary_uses_weighted_violation_rate_and_separate_radar_scopes():
    rows = [_row(), _row(seed=205001, terminal="cutoff", steps=90,
                        violations=0, reward=-40.0)]
    result = runner.summarize(rows)
    assert result["success_rate"] == 0.5
    assert result["mean_return"] == -10.0
    assert result["violations_per_step"] == 0.1
    assert result["mean_episode_violations_per_step"] == 0.5
    assert result["radar_target_dwell_counts"]["all_targets"] == {
        "opportunities": 40, "hits": 24, "misses": 16,
        "hit_fraction_per_opportunity": 0.6,
    }
    assert result["radar_target_dwell_counts"]["unfinished_targets"] == {
        "opportunities": 20, "hits": 14, "misses": 6,
        "hit_fraction_per_opportunity": 0.7,
    }
    assert result["stable_progress_resets"] == 4
    assert result["stable_reset_causes"] == {"stale_owner_track": 4}
    assert result["owner_gate_failure_target_step_causes"] == {"stale_owner_track": 10}
    assert result["unowned_tracking_target_steps"] == 12
    assert result["conflict_pairs_observed"] == 1
    assert len(result["active_agent_steps_by_conflict_pair"]) == 13
    assert result["success_rate_wilson_95"][0] < 0.5 < result["success_rate_wilson_95"][1]
    assert rows[0]["sensor_diagnostics"]["radar"]["all_targets"]["opportunities"] == 20


def test_summary_zero_opportunities_reports_missing_fraction():
    row = _row()
    for scope in ("all_targets", "unfinished_targets"):
        row["sensor_diagnostics"]["radar"][scope] = {
            "opportunities": 0, "hits": 0, "misses": 0}
    result = runner.summarize([row])
    assert result["radar_target_dwell_counts"]["all_targets"]["hit_fraction_per_opportunity"] is None
    with pytest.raises(ValueError, match="zero episodes"):
        runner.summarize([])


def test_pairing_uses_unique_matching_seeds_and_direction_right_minus_left():
    left = [_row(seed=1, terminal="success", reward=10.0),
            _row(seed=2, terminal="cutoff", reward=-30.0)]
    right = [_row(seed=2, terminal="success", reward=20.0),
             _row(seed=1, terminal="cutoff", reward=-10.0)]
    paired = runner.paired_difference(left, right)
    assert paired["direction"] == "right_minus_left"
    assert paired["mean_return_delta"] == 15.0
    assert paired["success_rate_delta"] == 0.0
    assert paired["right_only_successes"] == paired["left_only_successes"] == 1
    for invalid in ([right[0], right[0]], [dict(right[0], seed=3), right[1]]):
        with pytest.raises(ValueError, match="identical unique seeds"):
            runner.paired_difference(left, invalid)


def test_report_refuses_condition_dependent_effective_rl_weights():
    rows = [_row(profile=profile, controller=controller)
            for profile in runner.PROFILES for controller in runner.CONTROLLERS]
    report = runner.make_report(rows)
    assert report["same_loaded_rl_tensors_all_conditions"] is True
    assert report["loaded_tensor_sha256"] == "a" * 64
    rows[2]["loaded_tensor_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="different effective RL weights"):
        runner.make_report(rows)


def test_fingerprint_covers_model_both_anchors_and_tensor_metadata():
    policy = SimpleNamespace(**{
        name: torch.nn.Linear(2, 2)
        for name in ("model", "xi_anchor_model", "policy_anchor_model")})
    before = runner.model_fingerprint(policy)
    assert before == runner.model_fingerprint(deepcopy(policy))
    for name in ("model", "xi_anchor_model", "policy_anchor_model"):
        modified = deepcopy(policy)
        with torch.no_grad():
            getattr(modified, name).weight[0, 0].add_(1.0)
        assert runner.model_fingerprint(modified) != before
    double = deepcopy(policy)
    double.model.double()
    assert runner.model_fingerprint(double) != before


def test_existing_output_is_rejected_before_any_snapshot(tmp_path, monkeypatch, capsys):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_bytes(b"retained")
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--output", str(output)])
    monkeypatch.setattr(runner, "capture_source_snapshot",
                        lambda *_args: pytest.fail("created snapshot before preflight"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "output already exists" in capsys.readouterr().err
    assert list(output.iterdir()) == [marker]
    assert marker.read_bytes() == b"retained"


def test_wrong_checkpoint_hash_is_rejected_without_output_artifacts(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "input.pt"
    checkpoint.write_bytes(b"synthetic checkpoint preflight only")
    output = tmp_path / "must_not_exist"
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    wrong = ("0" if actual[0] != "0" else "1") + actual[1:]
    monkeypatch.setattr(sys, "argv", [
        "diagnostic", "--checkpoint", str(checkpoint), "--output", str(output),
        "--expected-checkpoint-sha256", wrong,
    ])
    monkeypatch.setattr(runner, "capture_source_snapshot",
                        lambda *_args: pytest.fail("created snapshot before checksum gate"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "SHA256 differs" in capsys.readouterr().err
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.parametrize("extra", [
    ["--episodes", "0"], ["--seed-start", "-1"], ["--torch-threads", "0"],
])
def test_invalid_basic_arguments_leave_no_output(tmp_path, monkeypatch, extra):
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--output", str(output), *extra])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not output.exists()
