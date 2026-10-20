import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import _run_best_refine as launcher
import _run_k4_diagnostics as suite


@pytest.fixture
def diagnostic_inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "curr_k4_best.pt"
    checkpoint.write_bytes(b"fixed K4 best policy")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    log = {
        "experiment": {
            "task_numeric_config": launcher.TASK_CONFIG.copy(),
            "ppo_config": {"direct_rl_arbitration": True},
        },
        "curr_k4_best": {
            "iter": 24,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": digest,
            "validation": {
                "k": 4,
                "task_config": {key.upper(): value for key, value in launcher.TASK_CONFIG.items()},
            },
        },
    }
    (source / "pilot_log.json").write_text(json.dumps(log), encoding="utf-8")
    code = tmp_path / "tracked_training_code.py"
    code.write_text("unchanged training code", encoding="utf-8")
    monkeypatch.setattr(suite, "_code_hashes", lambda: {
        "_run_k4_diagnostics.py": suite.file_sha256(code),
    })
    return source, tmp_path / "suite_output", code


def suite_args(inputs, *extra):
    source, output, _ = inputs
    return suite.build_parser().parse_args([
        "--source-run", str(source), "--output-root", str(output), *extra,
    ])


def test_paired_cases_change_only_the_intended_controls(diagnostic_inputs):
    plan = suite.prepare_suite(suite_args(diagnostic_inputs))
    jobs = {job["name"]: job for job in plan["jobs"]}
    expected = {
        "control": (1.0, 0.2, 0.0),
        "natural_requests": (0.0, 0.2, 0.0),
        "full_penalty": (1.0, 1.0, 0.0),
        "natural_full_penalty": (0.0, 1.0, 0.0),
        "natural_full_penalty_anchor": (0.0, 1.0, 1.0),
    }
    assert list(jobs) == list(expected)
    source = jobs["control"]["source"]
    common_env = jobs["control"]["training_environment"]
    varying_env_keys = {
        "MUMT_PILOT_OUT", "MUMT_CONFLICT_EXPOSURE_PROB", "MUMT_W8_FIXED",
        "MUMT_POLICY_ANCHOR_COEF",
    }
    for name, controls in expected.items():
        job = jobs[name]
        assert job["source"] == source
        assert job["training_controls"] == dict(zip(
            ("conflict_exposure", "penalty_scale", "policy_anchor_coef"), controls))
        assert job["command"] == jobs["control"]["command"]
        assert job["task_numeric_config"] == launcher.TASK_CONFIG
        env = job["training_environment"]
        assert env["MUMT_TRAIN_SEED_START"] == "30000"
        assert env["MUMT_VALIDATION_SEEDS"] == ",".join(map(str, range(230, 240)))
        assert env["MUMT_CURR_PATIENCE"] == "0"
        assert env["MUMT_EPISODES_PER_ITER"] == "8"
        assert env["MUMT_VALIDATION_EVERY"] == "2"
        assert job["command"][-4:] == ["curr", "4", "0", "8"]
        assert {k: v for k, v in env.items() if k not in varying_env_keys} == {
            k: v for k, v in common_env.items() if k not in varying_env_keys}
    assert plan["budget"] == {
        "iterations": 8, "episodes_per_iter": 8, "validation_every": 2,
        "training_seed_start": 30000, "validation_seeds": list(range(230, 240)),
        "early_stopping_patience": 0,
    }


@pytest.mark.parametrize("failure", ["nonempty_output", "duplicate_cases"])
def test_old_output_and_duplicate_cases_are_rejected(diagnostic_inputs, failure):
    source, output, _ = diagnostic_inputs
    extra = []
    if failure == "nonempty_output":
        output.mkdir()
        (output / "keep.txt").write_text("existing experiment", encoding="utf-8")
    else:
        extra = ["--cases", "control", "control"]
    with pytest.raises(ValueError):
        suite.prepare_suite(suite_args(diagnostic_inputs, *extra))
    assert (source / "curr_k4_best.pt").read_bytes() == b"fixed K4 best policy"
    if failure == "nonempty_output":
        assert (output / "keep.txt").read_text(encoding="utf-8") == "existing experiment"


def test_preflight_is_read_only_and_public_plan_does_not_leak_environment(
        diagnostic_inputs, monkeypatch, capsys):
    source, output, _ = diagnostic_inputs
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    monkeypatch.setenv("DIAGNOSTIC_TEST_SECRET", "must_not_appear_in_public_plan")
    monkeypatch.setattr(suite, "run_suite", lambda plan: pytest.fail("Preflight ran training"))
    result = suite.main([
        "--source-run", str(source), "--output-root", str(output), "--preflight-only",
    ])
    printed = capsys.readouterr().out
    public = json.loads(printed)
    assert result == 0
    assert not output.exists()
    assert "must_not_appear_in_public_plan" not in printed
    assert all("environment" not in job for job in public["jobs"])
    assert all("training_environment" in job for job in public["jobs"])
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_subprocess_failure_is_saved_and_stops_following_cases(diagnostic_inputs, monkeypatch):
    plan = suite.prepare_suite(suite_args(diagnostic_inputs))
    calls = []

    def failed_training(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("expected simulated process failure\n")
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(suite.subprocess, "run", failed_training)
    monkeypatch.setattr(suite, "verify_completed_run", lambda *a: pytest.fail("Verified failed run"))
    assert suite.run_suite(plan) == 17
    assert len(calls) == 1
    manifest = json.loads((Path(plan["output_root"]) / "diagnostic_suite.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert len(manifest["results"]) == 1
    result = manifest["results"][0]
    assert result["status"] == "failed"
    assert result["returncode"] == 17
    assert "exited 17" in result["error"]
    assert Path(result["stdout"]).read_text(encoding="utf-8") == "expected simulated process failure\n"
    assert all("environment" not in job for job in manifest["plan"]["jobs"])


def test_different_initial_evaluation_across_cases_fails_and_is_saved(diagnostic_inputs, monkeypatch):
    plan = suite.prepare_suite(suite_args(diagnostic_inputs))
    calls = []

    def successful_training(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", successful_training)
    monkeypatch.setattr(suite, "verify_completed_run", lambda job, budget: {
        "baseline_by_seed_sha256": job["name"],
    })
    assert suite.run_suite(plan) == 1
    assert len(calls) == 2
    manifest = json.loads((Path(plan["output_root"]) / "diagnostic_suite.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert [r["status"] for r in manifest["results"]] == ["completed", "failed"]
    assert "different initial evaluation results" in manifest["results"][-1]["error"]


@pytest.mark.parametrize("changed_input", ["checkpoint", "source_log", "training_code"])
def test_input_or_code_mutation_rejected_between_cases(diagnostic_inputs, changed_input):
    source, _, code = diagnostic_inputs
    plan = suite.prepare_suite(suite_args(diagnostic_inputs))
    job = plan["jobs"][0]
    suite._check_inputs(plan, job)
    changed = {
        "checkpoint": source / "curr_k4_best.pt",
        "source_log": source / "pilot_log.json",
        "training_code": code,
    }[changed_input]
    changed.write_bytes(changed.read_bytes() + b" changed during suite")
    with pytest.raises(ValueError, match="changed"):
        suite._check_inputs(plan, job)


@pytest.fixture
def completed_natural_run(diagnostic_inputs):
    plan = suite.prepare_suite(suite_args(
        diagnostic_inputs, "--cases", "natural_requests", "--iterations", "2",
        "--episodes-per-iter", "2", "--validation-every", "1",
        "--validation-seeds", "230,231"))
    job, budget = plan["jobs"][0], plan["budget"]
    output = Path(job["output"])
    output.mkdir(parents=True)
    (output / "curr_k4_best.pt").write_bytes(b"selected iteration zero weights")
    (output / "curr_k4.pt").write_bytes(b"latest iteration one weights")

    def validation(iteration, success, reward, violations):
        return {
            "iter": iteration, "k": 4, "seeds": [230, 231],
            "task_config_sha256": "fixed task configuration",
            "rl": {"success_rate": success, "mean_return": reward,
                   "violations_per_step": violations},
            "heur": {"success_rate": 0.8, "mean_return": -4032.2,
                     "violations_per_step": 7.6631},
            "natural_arbitration": {"forced_probe_enabled": False},
            "by_seed": [{"seed": 230, "rl": {"success": True}},
                        {"seed": 231, "rl": {"success": False}}],
        }

    baseline = validation(-1, 0.7, -4891.3, 6.8201)
    candidate = validation(0, 0.8, -4300.0, 6.5)
    last = validation(1, 0.6, -6000.0, 7.2)
    rows = [{
        "iter": iteration,
        "episode_seeds": [30000 + iteration * 2, 30001 + iteration * 2],
        "w8_scale": 0.2, "post_update_policy_anchor_kl": 0.01 * (iteration + 1),
        "forced_overlap_pairs": 0, "exposed_agent_steps": 0,
        "environment_steps": 12, "agent_steps": 24,
        "validation": value,
    } for iteration, value in enumerate((candidate, last))]
    rows[-1]["checkpoint"] = {"sha256": suite.file_sha256(output / "curr_k4.pt")}
    log = {
        "curr_k4": rows,
        "experiment": {
            "validation_seeds": [230, 231], "fixed_w8_scale": 0.2,
            "training_seed_start": 30000, "early_stopping_patience": 0,
            "ppo_config": {
                "conflict_exposure_prob": 0.0, "policy_anchor_coef": 0.0,
                "direct_rl_arbitration": True, "bid_only_ppo": False,
            },
        },
        "run_provenance": [{
            "status": "completed", "source_integrity": {"unchanged": True},
            "completed_iterations": 2, "early_stopping": {"triggered": False},
            "curriculum_start": {"sha256": job["source"]["checkpoint_sha256"]},
        }],
        "curr_k4_baseline": baseline,
        "curr_k4_best": {
            "iter": 0, "rank": [0.8, -4300.0, -6.5],
            "checkpoint_sha256": suite.file_sha256(output / "curr_k4_best.pt"),
        },
        "curr_k4_last_validation": last,
    }
    (output / "pilot_log.json").write_text(json.dumps(log), encoding="utf-8")
    return job, budget, log, output


def test_completed_run_verification_uses_actual_artifacts_and_selected_validation(completed_natural_run):
    job, budget, log, _ = completed_natural_run
    result = suite.verify_completed_run(job, budget)
    assert result["best_iter"] == 0
    assert result["best"]["rl"] == log["curr_k4"][0]["validation"]["rl"]
    assert result["last"]["rl"] == log["curr_k4_last_validation"]["rl"]
    assert result["baseline_by_seed_sha256"] == suite.canonical_sha256(log["curr_k4_baseline"]["by_seed"])
    assert result["total_environment_steps"] == 24
    assert result["total_agent_steps"] == 48
    assert result["total_exposed_agent_steps"] == 0


@pytest.mark.parametrize("corruption", [
    "input_sha", "best_iter", "best_rank", "best_bytes", "latest_bytes",
    "episode_seeds", "w8_scale", "forced_overlap_pairs", "exposed_agent_steps",
    "source_integrity", "incomplete_budget", "early_stopped", "completed_count",
    "validation_seeds", "forced_probe", "task_hash", "actual_controls", "nonfinite_kl",
])
def test_completed_run_verification_rejects_inconsistent_logs_or_weights(completed_natural_run, corruption):
    job, budget, log, output = completed_natural_run
    # Establish that each mutated case starts from a complete, valid run.
    suite.verify_completed_run(job, budget)
    record = log["run_provenance"][-1]
    row = log["curr_k4"][0]
    if corruption == "input_sha":
        record["curriculum_start"]["sha256"] = "0" * 64
    elif corruption == "best_iter":
        log["curr_k4_best"]["iter"] = 1
    elif corruption == "best_rank":
        log["curr_k4_best"]["rank"] = [1.0, -4300.0, -6.5]
    elif corruption == "best_bytes":
        (output / "curr_k4_best.pt").write_bytes(b"different best weights")
    elif corruption == "latest_bytes":
        (output / "curr_k4.pt").write_bytes(b"different latest weights")
    elif corruption == "episode_seeds":
        row["episode_seeds"][0] += 1
    elif corruption == "w8_scale":
        row["w8_scale"] = 1.0
    elif corruption in ("forced_overlap_pairs", "exposed_agent_steps"):
        row[corruption] = 1
    elif corruption == "source_integrity":
        record["source_integrity"]["unchanged"] = False
    elif corruption == "incomplete_budget":
        log["curr_k4"].pop()
    elif corruption == "early_stopped":
        record["early_stopping"]["triggered"] = True
    elif corruption == "completed_count":
        record["completed_iterations"] = 1
    elif corruption == "validation_seeds":
        row["validation"]["seeds"] = [240, 241]
    elif corruption == "forced_probe":
        row["validation"]["natural_arbitration"]["forced_probe_enabled"] = True
    elif corruption == "task_hash":
        row["validation"]["task_config_sha256"] = "changed task configuration"
    elif corruption == "actual_controls":
        log["experiment"]["ppo_config"]["policy_anchor_coef"] = 1.0
    elif corruption == "nonfinite_kl":
        row["post_update_policy_anchor_kl"] = float("nan")
    (output / "pilot_log.json").write_text(json.dumps(log), encoding="utf-8")
    with pytest.raises(ValueError):
        suite.verify_completed_run(job, budget)
