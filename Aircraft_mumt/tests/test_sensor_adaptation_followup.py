"""Fixed-checkpoint follow-up gates and lifecycle without running task episodes."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import _evaluate_sensor_adaptation as runner
from aircraft_sim.detect_mappo import DetectMAPPOConfig
from _diagnose_sensor_profile import model_fingerprint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row(seed=210000, *, model="initial_rl", terminal="cutoff", reward=-20.0,
        steps=10, violations=10):
    return {
        "seed": seed, "model": model, "terminal": terminal, "return": reward,
        "steps": steps, "violations": violations,
        "violations_per_step": violations / steps,
        "judged_targets": 4 if terminal == "success" else 1,
        "violations_by_family": {"C04": violations},
        "active_agent_steps_by_conflict_pair": {"S04-M02": 2},
        "multi_candidate_events": 2, "winners_different_from_fixed_score": 1,
        "loaded_tensor_sha256": ("a" if model == "initial_rl" else "b") * 64,
        "weights_unchanged": True, "arbitration_modes": {"rl_direct": 2},
        "sensor_diagnostics": {
            "radar": {scope: {"opportunities": 8, "hits": 8, "misses": 0}
                      for scope in ("all_targets", "unfinished_targets")},
            "verdict": {"stable_progress_resets": 1,
                        "reset_causes": {"stale_owner_track": 1},
                        "owner_gate_failure_causes": {"stale_owner_track": 3},
                        "unowned_tracking_target_steps": 4},
        },
    }


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    task = asdict(runner.build_configs(4, 8, 10.0)["deterministic_beam"])
    policy = asdict(DetectMAPPOConfig(direct_rl_arbitration=True,
                                    conflict_exposure_prob=0.0))
    for name, value in (("initial.pt", 1.0), ("best.pt", 2.0)):
        tensors = {"weight": torch.tensor([value])}
        torch.save({"config": policy, "model": tensors,
                    "xi_anchor_model": deepcopy(tensors),
                    "policy_anchor_model": deepcopy(tensors)}, source / name)
    (source / "checkpoints/update_0008.pt").write_bytes((source / "best.pt").read_bytes())
    initial = runner.artifact_record(source / "initial.pt", "initial_rl")
    best = runner.artifact_record(source / "best.pt", "best_rl")
    initial_summary = runner.summarize([row(207000), row(207001)])
    best_rows = [row(seed, model="best_rl", terminal="success", reward=-10,
                     violations=5) for seed in (207000, 207001)]
    best_summary = runner.summarize(best_rows)
    log = {
        "status": "completed_short_adaptation",
        "task_config": task, "task_config_sha256": runner.canonical_sha256(task),
        "policy_config": policy,
        "initial_checkpoint": initial,
        "best": {"update": 8, "checkpoint": best,
                 "selected_checkpoint": best, "validation_summary": best_summary},
        "protocol": {"training_seeds": [206000, 206001],
                     "validation_seeds": [207000, 207001]},
        "training": [{"episodes": [{"seed": 206000}, {"seed": 206001}]}],
        "validations": [
            {"update": 0, "checkpoint": initial, "summary": initial_summary,
             "episodes": [row(207000), row(207001)]},
            {"update": 8, "checkpoint": best, "summary": best_summary,
             "episodes": best_rows},
        ],
    }
    snapshot = runner.capture_source_snapshot(ROOT, source, [ROOT / "_diagnose_sensor_profile.py"])
    provenance = {
        "status": "completed", "selected_best_update": 8,
        "source_integrity": {"unchanged": True, "changed_files": []},
        "reserved_test_run": False, "effective_source": snapshot,
        "task_config": task, "task_config_sha256": runner.canonical_sha256(task),
        "algorithm_config": {"effective": policy},
    }

    def persist():
        (source / "adaptation_log.json").write_text(json.dumps(log), encoding="utf-8")
        provenance["outputs"] = [
            runner.artifact_record(source / "adaptation_log.json", "adaptation_log"),
            runner.artifact_record(source / "initial.pt", "effective_initial_checkpoint"),
            runner.artifact_record(source / "best.pt", "selected_best_checkpoint"),
        ]
        (source / "adaptation.provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        monkeypatch.setattr(runner, "EXPECTED_LOG", digest(source / "adaptation_log.json"))
        monkeypatch.setattr(runner, "EXPECTED_PROVENANCE", digest(source / "adaptation.provenance.json"))
        monkeypatch.setattr(runner, "EXPECTED_INITIAL", initial["sha256"])
        monkeypatch.setattr(runner, "EXPECTED_BEST", best["sha256"])

    persist()
    args = runner.parser().parse_args(["--source-run", str(source),
        "--output", str(tmp_path / "output"), "--episodes", "2",
        "--bootstrap-samples", "100"])
    return SimpleNamespace(path=source, log=log, provenance=provenance,
                           persist=persist, args=args)


def test_preflight_pins_existing_best_and_preserves_reserved_test(source_run):
    plan = runner.preflight(source_run.args)
    assert plan["protocol"]["seeds"] == [210000, 210001]
    assert plan["protocol"]["best_update_fixed"] == 8
    assert plan["protocol"]["training"] is False
    assert plan["protocol"]["checkpoint_selection"] is False
    assert plan["protocol"]["reserved_test_seeds_not_run"] == list(range(208000, 208030))
    assert not source_run.args.output.exists()


@pytest.mark.parametrize("seed", [162000, 204900, 205029, 206000, 207001,
                                 208000, 208029, 209900, 209990])
def test_preflight_rejects_used_or_reserved_seeds(source_run, seed):
    source_run.args.seed_start = seed
    with pytest.raises(ValueError, match="overlap"):
        runner.preflight(source_run.args)
    assert not source_run.args.output.exists()


def test_seed_overlap_in_later_episode_is_rejected(source_run):
    source_run.args.seed_start = 207999
    with pytest.raises(ValueError, match="overlap"):
        runner.preflight(source_run.args)


@pytest.mark.parametrize("name,value", [("episodes", 0), ("seed_start", -1),
                                       ("torch_threads", 0), ("bootstrap_samples", 99)])
def test_invalid_controls_fail_before_output(source_run, name, value):
    setattr(source_run.args, name, value)
    with pytest.raises(ValueError):
        runner.preflight(source_run.args)
    assert not source_run.args.output.exists()


@pytest.mark.parametrize("name", ["adaptation_log.json", "adaptation.provenance.json",
                                  "initial.pt", "best.pt", "checkpoints/update_0008.pt"])
def test_any_pinned_input_tamper_is_rejected(source_run, name):
    path = source_run.path / name
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        runner.preflight(source_run.args)
    assert not source_run.args.output.exists()


@pytest.mark.parametrize("kind", ["status", "selected_update", "rank", "full_penalty",
                                  "window", "task_hash", "checkpoint_declaration"])
def test_semantic_gates_survive_valid_fixture_hashes(source_run, kind):
    log = source_run.log
    if kind == "status":
        log["status"] = "running"
    elif kind == "selected_update":
        log["best"]["update"] = 12
    elif kind == "rank":
        log["validations"][0]["summary"]["successes"] = 3
    elif kind == "full_penalty":
        log["task_config"]["W8_SCALE"] = .5
    elif kind == "window":
        log["task_config"]["STABLE_WINDOW"] = 9
    elif kind == "task_hash":
        log["task_config"]["R_MAX"] = 99999
    else:
        log["best"]["selected_checkpoint"]["sha256"] = "0" * 64
    source_run.persist()
    with pytest.raises(ValueError):
        runner.preflight(source_run.args)
    assert not source_run.args.output.exists()


def test_existing_output_is_never_touched(source_run):
    source_run.args.output.mkdir()
    marker = source_run.args.output / "keep"
    marker.write_bytes(b"keep")
    with pytest.raises(ValueError, match="output already exists"):
        runner.preflight(source_run.args)
    assert list(source_run.args.output.iterdir()) == [marker]
    assert marker.read_bytes() == b"keep"


@pytest.mark.parametrize("kind", ["manifest", "zip_hash", "payload", "extra_member", "current_source"])
def test_training_snapshot_rejects_manifest_archive_or_current_source_drift(
        source_run, monkeypatch, kind):
    snapshot = source_run.provenance["effective_source"]
    archive_path = Path(snapshot["snapshot_path"])
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    if kind == "manifest":
        snapshot["manifest_sha256"] = "0" * 64
    elif kind == "zip_hash":
        archive_path.write_bytes(archive_path.read_bytes() + b"changed")
    elif kind == "current_source":
        monkeypatch.setattr(runner, "changed_source_files", lambda _source: ["radar.py"])
    else:
        with zipfile.ZipFile(archive_path) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        if kind == "payload":
            contents[next(iter(contents))] += b"changed"
        else:
            contents["unrecorded.py"] = b"unexpected"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, content in contents.items():
                archive.writestr(name, content)
        snapshot["snapshot_sha256"] = digest(archive_path)
    source_run.persist()
    with pytest.raises(ValueError, match="snapshot|training run"):
        runner.preflight(source_run.args)
    assert not source_run.args.output.exists()


def test_saved_fingerprint_matches_runtime_and_covers_both_anchors():
    policy = SimpleNamespace(**{name: torch.nn.Linear(2, 3)
        for name in ("model", "xi_anchor_model", "policy_anchor_model")})
    data = {name: getattr(policy, name).state_dict()
            for name in ("model", "xi_anchor_model", "policy_anchor_model")}
    expected = model_fingerprint(policy)
    assert runner.checkpoint_fingerprint(data) == expected
    for label in data:
        changed = deepcopy(data)
        changed[label]["weight"][0, 0] += 1
        assert runner.checkpoint_fingerprint(changed) != expected
        with pytest.raises(ValueError, match=label):
            runner.checkpoint_fingerprint({key: value for key, value in data.items() if key != label})
    changed = deepcopy(data)
    changed["model"]["weight"] = changed["model"]["weight"].double()
    assert runner.checkpoint_fingerprint(changed) != expected


@pytest.mark.parametrize("kind", ["source", "input"])
def test_runtime_source_or_input_drift_stops_evaluation(tmp_path, monkeypatch, kind):
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"frozen")
    artifact = runner.artifact_record(checkpoint, "fixed_model")
    monkeypatch.setattr(runner, "changed_source_files", lambda _source: [])
    runner.require_unchanged({"files": []}, [artifact])
    if kind == "source":
        monkeypatch.setattr(runner, "changed_source_files", lambda _source: ["radar.py"])
    else:
        checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match=f"evaluation {kind}.*changed"):
        runner.require_unchanged({"files": []}, [artifact])


def test_plan_only_imports_no_runtime_and_writes_nothing(source_run):
    before = {str(path.relative_to(source_run.path)): digest(path)
              for path in source_run.path.rglob("*") if path.is_file()}
    script = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('aircraft_sim'):
            raise AssertionError('plan imported runtime: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
sys.path.insert(0, sys.argv.pop(1))
import _evaluate_sensor_adaptation as runner
runner.EXPECTED_LOG, runner.EXPECTED_PROVENANCE, runner.EXPECTED_INITIAL, runner.EXPECTED_BEST = sys.argv[1:5]
del sys.argv[1:5]
runner.main()
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT),
        runner.EXPECTED_LOG, runner.EXPECTED_PROVENANCE,
        runner.EXPECTED_INITIAL, runner.EXPECTED_BEST,
        "--source-run", str(source_run.path), "--output", str(source_run.args.output),
        "--episodes", "2", "--plan-only"], cwd=source_run.path.parent,
        text=True, capture_output=True, check=True, timeout=30)
    plan = json.loads(result.stdout)
    assert result.stderr == ""
    assert plan["seeds"] == [210000, 210001]
    assert not source_run.args.output.exists()
    assert before == {str(path.relative_to(source_run.path)): digest(path)
                      for path in source_run.path.rglob("*") if path.is_file()}


def test_paired_statistics_use_seeds_exact_success_test_and_deterministic_bootstrap():
    left = [row(seed) for seed in range(4)]
    right = [row(seed, model="best_rl", terminal="success", reward=-15,
                 violations=0) for seed in reversed(range(4))]
    result = runner.paired_statistics(left, right, samples=200)
    assert result["right_only_successes"] == 4
    assert result["left_only_successes"] == 0
    assert result["exact_paired_success_two_sided_p"] == .125
    assert result["mean_return_delta"] == 5
    assert result["violations_per_step_delta"] == -1
    assert result["mean_episode_violations_per_step_delta"] == -1
    assert result["per_seed_differences"] == [
        {"seed": seed, "return_delta": 5, "violations_per_step_delta": -1,
         "success_delta": 1} for seed in range(4)]
    assert result["success_flip_seeds"] == {
        "right_only_successes": [0, 1, 2, 3], "left_only_successes": []}
    intervals = result["paired_percentile_bootstrap_95"]
    assert intervals["mean_return_delta"] == [5.0, 5.0]
    assert intervals["success_rate_delta"] == [1.0, 1.0]
    assert intervals["violations_per_step_delta"] == [-1.0, -1.0]
    assert intervals["mean_episode_violations_per_step_delta"] == [-1.0, -1.0]
    assert result == runner.paired_statistics(left, right, samples=200)
    unchanged = runner.paired_statistics(left, deepcopy(left), samples=100)
    assert unchanged["exact_paired_success_two_sided_p"] == 1.0
    for invalid in ([right[0]] * 4, [dict(right[0], seed=9), *right[1:]]):
        with pytest.raises(ValueError, match="identical unique seeds"):
            runner.paired_statistics(left, invalid, samples=100)
    with pytest.raises(ValueError, match="positive"):
        runner.paired_statistics(left, right, samples=0)


def test_pooled_and_episode_weighted_rate_changes_are_distinct():
    left = [row(1, steps=1, violations=10), row(2, steps=99, violations=0)]
    right = [row(1, steps=99, violations=99), row(2, steps=1, violations=1)]
    result = runner.paired_statistics(left, right, samples=100)
    assert result["violations_per_step_delta"] == pytest.approx(.9)
    assert result["mean_episode_violations_per_step_delta"] == -4


def test_report_requires_complete_frozen_conditions_and_descriptive_screen():
    rows = [row(seed, model=label,
                terminal="success" if label != "initial_rl" else "cutoff",
                reward=-10 if label == "best_rl" else -20,
                violations=0 if label == "best_rl" else 10)
            for label in runner.LABELS for seed in (1, 2)]
    report = runner.make_report(rows, [1, 2], 100)
    assert report["descriptive_screen_against_initial"]["passed"]
    assert report["descriptive_screen_against_heur"]["passed"]
    assert "no automatic" in report["descriptive_screen_against_heur"]["scope"]
    for invalid in (rows[:-1], [*rows[:-1], rows[0]]):
        with pytest.raises(ValueError, match="exactly one"):
            runner.make_report(invalid, [1, 2], 100)
    for field, value in (("loaded_tensor_sha256", "c" * 64), ("weights_unchanged", False)):
        invalid = deepcopy(rows)
        invalid[0][field] = value
        with pytest.raises(ValueError, match="changing policy"):
            runner.make_report(invalid, [1, 2], 100)


@pytest.mark.parametrize("failure", [None, "episode_failure", "wrong_tensor", "wrong_policy",
                                     "random_miss", "fixed_score", "weight_mutation", "task_drift"])
def test_fake_main_archives_fixed_inputs_no_training_and_finalizes_lifecycle(
        source_run, monkeypatch, failure):
    source = source_run
    before = {str(path.relative_to(source.path)): digest(path)
              for path in source.path.rglob("*") if path.is_file()}
    calls = []

    def snapshot(_root, output, _scripts):
        (output / "provenance").mkdir(parents=True)
        return {"files": []}

    def episode(cfg, controller, seed, checkpoint, config):
        label = "heur" if controller == "heur" else (
            "initial_rl" if "initial_rl" in str(checkpoint) else "best_rl")
        calls.append((label, seed, str(checkpoint)))
        assert seed in (210000, 210001)
        assert "provenance" in Path(checkpoint).parts
        assert config == source.log["policy_config"]
        assert asdict(cfg) == source.log["task_config"]
        if failure == "episode_failure" and len(calls) == 2:
            raise RuntimeError("synthetic follow-up failure")
        result = row(seed, model=label,
                     terminal="success" if label == "best_rl" else "cutoff")
        result["effective_task_config_sha256"] = source.log["task_config_sha256"]
        result["effective_policy_config"] = asdict(runner.evaluation_policy_config(config))
        if hasattr(runner, "checkpoint_fingerprint"):
            result["loaded_tensor_sha256"] = runner.checkpoint_fingerprint(
                torch.load(checkpoint, weights_only=False))
        if len(calls) == 2:
            if failure == "wrong_tensor":
                result["loaded_tensor_sha256"] = "0" * 64
            elif failure == "wrong_policy":
                result["effective_policy_config"]["direct_rl_arbitration"] = False
            elif failure == "random_miss":
                result["sensor_diagnostics"]["radar"]["unfinished_targets"]["misses"] = 1
            elif failure == "fixed_score":
                result["arbitration_modes"] = {"fixed_score": 1}
            elif failure == "weight_mutation":
                result["weights_unchanged"] = False
            elif failure == "task_drift":
                result["effective_task_config_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(runner, "capture_source_snapshot", snapshot)
    monkeypatch.setattr(runner, "capture_run_provenance", lambda *a, **k: {
        "effective_source": {"files": []}, "inputs": list(k["inputs"]), "outputs": []})
    monkeypatch.setattr(runner, "run_episode", episode)
    from aircraft_sim.detect_mappo import DetectMAPPOPolicy
    for method in ("collect", "update", "save"):
        monkeypatch.setattr(DetectMAPPOPolicy, method,
                            lambda *_a, **_k: pytest.fail("follow-up attempted training or model writing"))
    monkeypatch.setattr(sys, "argv", ["follow-up", "--source-run", str(source.path),
        "--output", str(source.args.output), "--episodes", "2", "--bootstrap-samples", "100"])
    if failure:
        with pytest.raises(RuntimeError):
            runner.main()
    else:
        runner.main()
    provenance = json.loads((source.args.output / "followup.provenance.json").read_text())
    assert provenance["status"] == ("failed" if failure else "completed")
    assert provenance["source_integrity"]["unchanged"] is True
    assert before == {str(path.relative_to(source.path)): digest(path)
                      for path in source.path.rglob("*") if path.is_file()}
    for item in provenance["inputs"]:
        assert digest(item["path"]) == item["sha256"]
    if failure:
        assert not (source.args.output / "followup_result.json").exists()
        assert (source.args.output / "episode_progress.json").exists()
        assert provenance["failure"]["type"] == "RuntimeError"
    else:
        report = json.loads((source.args.output / "followup_result.json").read_text())
        assert len(calls) == len(report["episodes"]) == 6
        assert {(label, seed) for label, seed, _ in calls} == {
            (label, seed) for label in runner.LABELS for seed in (210000, 210001)}
        assert not (source.args.output / "episode_progress.json").exists()
        assert report["protocol"]["training"] is False
        assert report["protocol"]["checkpoint_selection"] is False
        assert provenance["training_run"] is False
        assert provenance["checkpoint_selection_run"] is False
        assert provenance["reserved_test_run"] is False
