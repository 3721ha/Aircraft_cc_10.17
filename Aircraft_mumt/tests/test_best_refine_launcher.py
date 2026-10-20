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


@pytest.fixture
def source_run(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "curr_k3_best.pt"
    checkpoint.write_bytes(b"verified best checkpoint, distinct from latest")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    log = {
        "experiment": {
            "task_numeric_config": launcher.TASK_CONFIG.copy(),
            "ppo_config": {"direct_rl_arbitration": True},
        },
        "curr_k3_best": {
            "iter": 4,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": digest,
            "validation": {
                "k": 3,
                "task_config": {key.upper(): value for key, value in launcher.TASK_CONFIG.items()},
                "best_checkpoint": {"path": str(checkpoint), "sha256": digest},
            },
        },
    }
    (source / "pilot_log.json").write_text(json.dumps(log), encoding="utf-8")
    return source


def parse(source, output, *extra):
    return launcher.build_parser().parse_args([
        "--k", "3", "--source-run", str(source), "--output", str(output), *extra])


def rewrite_log(source, change):
    path = source / "pilot_log.json"
    log = json.loads(path.read_text(encoding="utf-8"))
    change(log)
    path.write_text(json.dumps(log), encoding="utf-8")


def test_environment_isolated_and_exact_best_selected(source_run, tmp_path):
    output = tmp_path / "new_run"
    launch = launcher.prepare_launch(parse(source_run, output), {
        "PATH": "some-path",
        "MUMT_CURR_INIT": "latest.pt",
        "MUMT_BID_LOG_STD": "-3.0",
        "MUMT_BID_ONLY_PPO": "1",
        "MUMT_UNKNOWN_STALE": "bad",
        "MUMT_CONFLICT_EXPOSURE_PROB": "0.7",
        "MUMT_W8_FIXED": "1",
        "MUMT_POLICY_ANCHOR_COEF": "999",
        "mumt_w8_span": "1",
    })
    env = launch["environment"]
    assert env["PATH"] == "some-path"
    assert env["MUMT_CURR_INIT"] == str(source_run / "curr_k3_best.pt")
    assert env["MUMT_CURR_EXPECTED_SHA256"] == hashlib.sha256((source_run / "curr_k3_best.pt").read_bytes()).hexdigest()
    assert env["MUMT_CURR_START_MODE"] == "best"
    assert launch["mode"] == "bid-only"
    assert env["MUMT_BID_ONLY_PPO"] == "1"
    assert env["MUMT_TRAIN_SEED_START"] == "10000"
    assert env["MUMT_SECTOR_AXIS_MAX_DEG"] == "60"
    assert env["MUMT_CONFLICT_EXPOSURE_PROB"] == "1.0"
    assert env["MUMT_W8_FIXED"] == "0.2"
    assert env["MUMT_POLICY_ANCHOR_COEF"] == "0.0"
    assert launch["training_controls"] == {
        "conflict_exposure": 1.0, "penalty_scale": 0.2, "policy_anchor_coef": 0.0,
    }
    assert launch["source"]["source_log_sha256"] == hashlib.sha256(
        (source_run / "pilot_log.json").read_bytes()).hexdigest()
    assert "MUMT_BID_LOG_STD" not in env
    assert "MUMT_UNKNOWN_STALE" not in env
    assert "mumt_w8_span" not in env
    assert launch["command"][-4:] == ["curr", "3", "0", "20"]
    assert not output.exists()


@pytest.mark.parametrize("exposure,penalty,anchor", [
    (0.0, 1.0, 0.0), (1.0, 0.0, 2.5), (0.4, 0.7, 0.15),
])
def test_explicit_training_controls_replace_inherited_values(source_run, tmp_path, exposure, penalty, anchor):
    args = parse(source_run, tmp_path / "new_run", "--conflict-exposure", str(exposure),
                 "--penalty-scale", str(penalty), "--policy-anchor-coef", str(anchor))
    launch = launcher.prepare_launch(args, {
        "MUMT_CONFLICT_EXPOSURE_PROB": "1",
        "MUMT_W8_FIXED": "0.2",
        "MUMT_POLICY_ANCHOR_COEF": "999",
        "mumt_policy_anchor_coef": "888",
    })
    assert launch["training_controls"] == {
        "conflict_exposure": exposure, "penalty_scale": penalty, "policy_anchor_coef": anchor,
    }
    assert launch["environment"]["MUMT_CONFLICT_EXPOSURE_PROB"] == str(exposure)
    assert launch["environment"]["MUMT_W8_FIXED"] == str(penalty)
    assert launch["environment"]["MUMT_POLICY_ANCHOR_COEF"] == str(anchor)
    assert "mumt_policy_anchor_coef" not in launch["environment"]
    assert launch["task_numeric_config"] == launcher.TASK_CONFIG


@pytest.mark.parametrize("option,value", [
    (option, value)
    for option in ("--conflict-exposure", "--penalty-scale")
    for value in ("-0.01", "1.01", "nan", "inf", "-inf")
] + [("--policy-anchor-coef", value) for value in ("-0.01", "nan", "inf", "-inf")])
def test_invalid_training_controls_rejected_by_cli(source_run, tmp_path, option, value):
    output = tmp_path / "never_created"
    with pytest.raises(SystemExit) as exc:
        parse(source_run, output, f"{option}={value}")
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("failure", ["missing_log", "missing_best", "missing_checkpoint", "hash_mismatch", "latest_recorded", "artifact_hash_mismatch"])
def test_missing_or_mismatched_best_fails(source_run, tmp_path, failure):
    if failure == "missing_log":
        (source_run / "pilot_log.json").unlink()
    elif failure == "missing_best":
        rewrite_log(source_run, lambda log: log.pop("curr_k3_best"))
    elif failure == "missing_checkpoint":
        (source_run / "curr_k3_best.pt").unlink()
    elif failure == "hash_mismatch":
        (source_run / "curr_k3_best.pt").write_bytes(b"overwritten checkpoint")
    elif failure == "latest_recorded":
        rewrite_log(source_run, lambda log: log["curr_k3_best"].update(checkpoint="curr_k3.pt"))
    else:
        rewrite_log(source_run, lambda log: log["curr_k3_best"]["validation"]["best_checkpoint"].update(sha256="0" * 64))
    with pytest.raises(ValueError):
        launcher.prepare_launch(parse(source_run, tmp_path / "new_run"))


def test_existing_nonempty_output_and_source_overlap_fail(source_run, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("existing experiment", encoding="utf-8")
    for target in (output, source_run, source_run / "nested", tmp_path):
        with pytest.raises(ValueError):
            launcher.prepare_launch(parse(source_run, target))
    assert (output / "keep.txt").read_text(encoding="utf-8") == "existing experiment"


@pytest.mark.parametrize("key", list(launcher.TASK_CONFIG))
def test_numeric_config_mismatch_fails(source_run, tmp_path, key):
    rewrite_log(source_run, lambda log: log["curr_k3_best"]["validation"]["task_config"].update({key.upper(): 999}))
    with pytest.raises(ValueError, match="mismatch"):
        launcher.prepare_launch(parse(source_run, tmp_path / "new_run"))


def test_non_direct_rl_source_rejected(source_run, tmp_path):
    rewrite_log(source_run, lambda log: log["experiment"]["ppo_config"].update(direct_rl_arbitration=False))
    with pytest.raises(ValueError, match="direct_rl_arbitration"):
        launcher.prepare_launch(parse(source_run, tmp_path / "new_run"))


def test_initial_best_iter_minus_one_is_valid(source_run, tmp_path):
    rewrite_log(source_run, lambda log: log["curr_k3_best"].update(iter=-1))
    launch = launcher.prepare_launch(parse(source_run, tmp_path / "new_run"))
    assert launch["source"]["source_best_iter"] == -1


def test_training_validation_overlap_rejected(source_run, tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        launcher.prepare_launch(parse(source_run, tmp_path / "new_run", "--training-seed-start", "200"))


def test_preflight_is_read_only(source_run, tmp_path, monkeypatch, capsys):
    def never_spawn(*args, **kwargs):
        raise AssertionError("preflight must not start training")

    monkeypatch.setattr(launcher.subprocess, "run", never_spawn)
    output = tmp_path / "preflight_output"
    original_source = {path.name: path.read_bytes() for path in source_run.iterdir()}
    result = launcher.main(["--k", "3", "--source-run", str(source_run),
                            "--output", str(output), "--preflight-only",
                            "--conflict-exposure", "0", "--penalty-scale", "1",
                            "--policy-anchor-coef", "0.25"])
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert not output.exists()
    assert report["source"]["source_best_iter"] == 4
    assert "environment" not in report
    assert report["training_environment"]["MUMT_CURR_START_MODE"] == "best"
    assert report["training_controls"] == {
        "conflict_exposure": 0.0, "penalty_scale": 1.0, "policy_anchor_coef": 0.25,
    }
    assert {path.name: path.read_bytes() for path in source_run.iterdir()} == original_source


def test_execution_uses_root_and_propagates_child_failure(source_run, tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert not (tmp_path / "output").exists()
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = launcher.main(["--k", "3", "--source-run", str(source_run),
                            "--output", str(tmp_path / "output"), "--mode", "bid-only"])
    assert result == 17
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == sys.executable
    assert options["cwd"] == ROOT
    assert options["env"]["MUMT_BID_ONLY_PPO"] == "1"
    assert options["check"] is False
