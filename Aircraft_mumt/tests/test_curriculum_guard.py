"""Regressions for explicit curriculum inputs and isolated validation."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _curriculum_guard import (  # noqa: E402
    file_sha256, resolve_curriculum_start, validate_best_record,
)


def _checkpoint(path, payload=b"designated weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_log(output, log):
    (output / "pilot_log.json").write_text(json.dumps(log), encoding="utf-8")


def _logged_resume(tmp_path):
    output = tmp_path / "resume"
    checkpoint = _checkpoint(output / "curr_k3.pt")
    log = {"curr_k3": [{"iter": 5, "checkpoint": {
        "sha256": file_sha256(checkpoint)}}]}
    _write_log(output, log)
    return output, checkpoint, log


def test_explicit_missing_initial_checkpoint_never_falls_back(tmp_path):
    output = tmp_path / "fresh"
    for name in ("pilot_bc.pt", "dagger_latest.pt", "pilot_final.pt"):
        _checkpoint(output / name)
    with pytest.raises(FileNotFoundError, match="Chosen curriculum checkpoint"):
        resolve_curriculum_start(tmp_path, output, 3, 0, 2, initial="missing.pt")


def test_relative_explicit_initial_checkpoint_is_resolved_from_root(tmp_path):
    initial = _checkpoint(tmp_path / "models" / "warm.pt")
    result = resolve_curriculum_start(
        tmp_path, tmp_path / "new", 3, 0, 2, initial="models/warm.pt")
    assert result == {"mode": "fresh", "path": str(initial.resolve()),
                      "sha256": file_sha256(initial), "iter_start": 0,
                      "weights_only": True}


def test_explicit_initial_checkpoint_cannot_be_ignored_by_resume(tmp_path):
    output, checkpoint, _ = _logged_resume(tmp_path)
    with pytest.raises(ValueError, match="Resume reads latest weights only"):
        resolve_curriculum_start(tmp_path, output, 3, 6, 2, initial=str(checkpoint))


@pytest.mark.parametrize("k,iteration,iterations", [(0, 0, 2), (3, -1, 2), (3, 0, 0)])
def test_invalid_iteration_or_count_fails(tmp_path, k, iteration, iterations):
    with pytest.raises(ValueError, match="curr requires"):
        resolve_curriculum_start(tmp_path, tmp_path / "new", k, iteration, iterations)


@pytest.mark.parametrize("iteration,mode", [(0, "resume"), (5, "fresh"), (5, "best")])
def test_incompatible_start_mode_and_iteration_fail(tmp_path, iteration, mode):
    with pytest.raises(ValueError, match="Resume"):
        resolve_curriculum_start(tmp_path, tmp_path / "new", 3, iteration, 2, mode=mode)


def test_unknown_start_mode_fails(tmp_path):
    with pytest.raises(ValueError, match="MUMT_CURR_START_MODE"):
        resolve_curriculum_start(tmp_path, tmp_path / "new", 3, 0, 2, mode="latest")


def test_best_requires_explicit_initial_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="MUMT_CURR_INIT"):
        resolve_curriculum_start(tmp_path, tmp_path / "new", 3, 0, 2, mode="best")


@pytest.mark.parametrize("name", ["curr_k4_best.pt", "curr_k3.pt", "pilot_bc.pt"])
def test_best_requires_matching_best_filename(tmp_path, name):
    initial = _checkpoint(tmp_path / "models" / name)
    with pytest.raises(ValueError, match="matching curr_kK_best.pt"):
        resolve_curriculum_start(
            tmp_path, tmp_path / "new", 3, 0, 2, initial=str(initial),
            mode="best", expected_sha256=file_sha256(initial))


def test_best_requires_expected_digest(tmp_path):
    initial = _checkpoint(tmp_path / "models" / "curr_k3_best.pt")
    with pytest.raises(ValueError, match="MUMT_CURR_EXPECTED_SHA256"):
        resolve_curriculum_start(
            tmp_path, tmp_path / "new", 3, 0, 2, initial=str(initial), mode="best")


def test_best_rejects_wrong_digest(tmp_path):
    initial = _checkpoint(tmp_path / "models" / "curr_k3_best.pt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_curriculum_start(
            tmp_path, tmp_path / "new", 3, 0, 2, initial=str(initial),
            mode="best", expected_sha256="0" * 64)


def test_best_returns_verified_exact_input(tmp_path):
    initial = _checkpoint(tmp_path / "models" / "curr_k3_best.pt")
    digest = file_sha256(initial)
    result = resolve_curriculum_start(
        tmp_path, tmp_path / "new", 3, 0, 2, initial=str(initial),
        mode="best", expected_sha256=digest.upper())
    assert result == {"mode": "best", "path": str(initial.resolve()),
                      "sha256": digest, "iter_start": 0, "weights_only": True}


@pytest.mark.parametrize("occupied", ["pilot_log.json", "curr_k3.pt", "curr_k4_best.pt"])
def test_fresh_output_cannot_overwrite_curriculum_run(tmp_path, occupied):
    output = tmp_path / "occupied"
    _checkpoint(output / occupied)
    initial = _checkpoint(tmp_path / "source.pt")
    with pytest.raises(FileExistsError, match="NEW output directory"):
        resolve_curriculum_start(tmp_path, output, 3, 0, 2, initial=str(initial))


def test_resume_requires_existing_log(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_curriculum_start(tmp_path, tmp_path / "missing", 3, 6, 2)


def test_resume_iteration_must_follow_last_logged_iteration(tmp_path):
    output, _, _ = _logged_resume(tmp_path)
    with pytest.raises(ValueError, match="last logged iteration"):
        resolve_curriculum_start(tmp_path, output, 3, 7, 2)


@pytest.mark.parametrize("rows", [[], [{"iter": 4}], [{"iter": "5"}], [{"iter": None}]])
def test_resume_missing_or_invalid_logged_iteration_fails(tmp_path, rows):
    output, _, log = _logged_resume(tmp_path)
    log["curr_k3"] = rows
    _write_log(output, log)
    with pytest.raises(ValueError, match="last logged iteration"):
        resolve_curriculum_start(tmp_path, output, 3, 6, 2)


def test_resume_requires_logged_checkpoint_digest(tmp_path):
    output, _, log = _logged_resume(tmp_path)
    log["curr_k3"][-1]["checkpoint"] = {}
    _write_log(output, log)
    with pytest.raises(ValueError, match="logged latest checkpoint SHA256"):
        resolve_curriculum_start(tmp_path, output, 3, 6, 2)


def test_resume_missing_logged_iteration_has_clear_failure(tmp_path):
    output, _, log = _logged_resume(tmp_path)
    del log["curr_k3"][-1]["iter"]
    _write_log(output, log)
    with pytest.raises(ValueError, match="last logged iteration"):
        resolve_curriculum_start(tmp_path, output, 3, 6, 2)


def test_resume_user_digest_cannot_replace_missing_logged_digest(tmp_path):
    output, checkpoint, log = _logged_resume(tmp_path)
    log["curr_k3"][-1]["checkpoint"] = {}
    _write_log(output, log)
    with pytest.raises(ValueError, match="logged latest checkpoint SHA256"):
        resolve_curriculum_start(
            tmp_path, output, 3, 6, 2, expected_sha256=file_sha256(checkpoint))


def test_resume_user_digest_cannot_bypass_logged_checkpoint_history(tmp_path):
    output, checkpoint, _ = _logged_resume(tmp_path)
    checkpoint.write_bytes(b"replacement-not-associated-with-iteration-five")
    with pytest.raises(ValueError, match="SHA256"):
        resolve_curriculum_start(
            tmp_path, output, 3, 6, 2, expected_sha256=file_sha256(checkpoint))


def test_resume_rejects_checkpoint_changed_since_logged_iteration(tmp_path):
    output, checkpoint, _ = _logged_resume(tmp_path)
    checkpoint.write_bytes(b"replacement checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_curriculum_start(tmp_path, output, 3, 6, 2)


def test_resume_returns_logged_exact_latest_checkpoint(tmp_path):
    output, checkpoint, _ = _logged_resume(tmp_path)
    result = resolve_curriculum_start(tmp_path, output, 3, 6, 2)
    assert result["mode"] == "resume"
    assert result["iter_start"] == 6
    assert Path(result["path"]) == checkpoint.resolve()
    assert result["sha256"] == file_sha256(checkpoint)


def _best_record(tmp_path):
    checkpoint = _checkpoint(tmp_path / "curr_k3_best.pt")
    validation = {"k": 3, "seeds": [230, 231],
                  "task_config_sha256": "task-fingerprint",
                  "rl": {"success_rate": 0.9, "mean_return": -3163.4,
                         "violations_per_step": 4.7069}}
    return checkpoint, {"curr_k3_best": {
        "checkpoint_sha256": file_sha256(checkpoint), "validation": validation}}


def test_validate_best_record_returns_success_return_negative_violation_rank(tmp_path):
    checkpoint, log = _best_record(tmp_path)
    assert validate_best_record(
        log, "curr_k3_best", checkpoint, 3, [230, 231], "task-fingerprint"
    ) == (0.9, -3163.4, -4.7069)


@pytest.mark.parametrize("field,value", [("k", 4), ("seeds", [231, 230]),
                                         ("task_config_sha256", "different-task")])
def test_validate_best_record_rejects_different_contract(tmp_path, field, value):
    checkpoint, log = _best_record(tmp_path)
    log["curr_k3_best"]["validation"][field] = value
    with pytest.raises(ValueError, match="different K, validation seeds or task config"):
        validate_best_record(
            log, "curr_k3_best", checkpoint, 3, [230, 231], "task-fingerprint")


def test_validate_best_record_rejects_changed_weights(tmp_path):
    checkpoint, log = _best_record(tmp_path)
    checkpoint.write_bytes(b"unlogged replacement")
    with pytest.raises(ValueError, match="SHA256 does not match"):
        validate_best_record(
            log, "curr_k3_best", checkpoint, 3, [230, 231], "task-fingerprint")


@pytest.mark.parametrize("missing", ["checkpoint", "record"])
def test_validate_best_record_requires_weights_and_record(tmp_path, missing):
    checkpoint, log = _best_record(tmp_path)
    if missing == "checkpoint":
        checkpoint.unlink()
    else:
        log.clear()
    with pytest.raises(ValueError, match="both the best checkpoint and its log record"):
        validate_best_record(
            log, "curr_k3_best", checkpoint, 3, [230, 231], "task-fingerprint")


def _pilot_function(name, namespace):
    """Compile only the requested function, avoiding import-time run artifacts."""
    tree = ast.parse((ROOT / "_p3_pilot.py").read_text(encoding="utf-8"))
    function = deepcopy(next(node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name))
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(ROOT / "_p3_pilot.py"), "exec"), namespace)
    return namespace[name]


class _EvaluationEnv:
    def __init__(self, failure=None, events=None):
        self.failure, self.terminal = failure, None
        self.scenario, self.actions = None, None
        self.events = events

    def reset(self, scenario):
        self.scenario = scenario

    def observe(self):
        return {0: "zero", 1: "one"}

    def step(self, actions):
        self.actions = actions
        if self.failure == "step":
            raise RuntimeError("injected evaluation failure")
        self.terminal = "success"
        info = {"report": SimpleNamespace(violations=["one-violation"])}
        if self.events is not None:
            info["arbitration"] = self.events
        return SimpleNamespace(reward=2.25, info=info)


def _evaluate_fixture(failure=None, events=None):
    original = SimpleNamespace(cfg=SimpleNamespace(W8_SCALE=0.2))
    environments, calls = [], []
    policy = SimpleNamespace(env=original)

    def make_env():
        env = _EvaluationEnv(failure, events)
        environments.append(env)
        return env

    def act_with_log_prob(observation, deterministic=False):
        assert policy.env is environments[-1]
        assert deterministic is True
        calls.append(observation)
        if failure == "actor":
            raise RuntimeError("injected evaluation failure")
        return ("action-" + observation, -0.25)

    def shield_filter(env, nominal):
        assert env is environments[-1]
        return nominal, []

    policy.act_with_log_prob = act_with_log_prob
    shield = SimpleNamespace(filter=shield_filter)
    evaluate = _pilot_function("evaluate", {"_make_env": make_env, "EVAL_K": 4})
    return evaluate, policy, shield, original, environments, calls


def test_evaluate_restores_training_env_and_uses_deterministic_actor_api():
    evaluate, policy, shield, original, environments, calls = _evaluate_fixture()
    result = evaluate(policy, shield, [240, 241], k=3)
    assert policy.env is original
    assert original.cfg.W8_SCALE == 0.2
    assert [env.scenario for env in environments] == [
        {"seed": 240, "k_red": 3}, {"seed": 241, "k_red": 3}]
    assert calls == ["zero", "one"] * 2
    assert [env.actions for env in environments] == [{0: "action-zero", 1: "action-one"}] * 2
    assert result == [{"seed": seed, "terminal": "success", "return": 2.2,
                      "steps": 1, "violations": 1, "multi_candidate_events": 0,
                      "winner_changes_vs_fixed_score": 0} for seed in [240, 241]]


@pytest.mark.parametrize("failure", ["actor", "step"])
def test_evaluate_restores_training_env_after_failure(failure):
    evaluate, policy, shield, original, _, _ = _evaluate_fixture(failure)
    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        evaluate(policy, shield, [240], k=3)
    assert policy.env is original
    assert original.cfg.W8_SCALE == 0.2


def test_evaluate_counts_natural_arbitration_with_real_event_schema():
    # Runtime events use integer agent keys. Cover an RL override, unchanged
    # winner, incumbent hysteresis, incumbent tie preference and index tie break.
    cases = [
        ({0: 0.8, 1: 0.2}, None, 0.2, 1),
        ({0: 0.2, 1: 0.8}, None, 0.2, 1),
        ({0: 0.6, 1: 0.5}, 1, 0.2, 0),
        ({0: 0.5, 1: 0.5}, 1, 0.2, 1),
        ({0: 0.5, 1: 0.5}, None, 0.2, 0),
    ]
    events = [{
        "tid": target, "candidates": [0, 1], "winner": winner,
        "previous_owner": owner, "scores": scores,
        "adjusted_scores": {0: 0.9, 1: 0.3},
        "bids": {0: 0.8 if winner == 0 else -0.2,
                 1: 0.8 if winner == 1 else -0.2},
        "bid_slots": {0: 0, 1: 1}, "bid_weight": 0.35,
        "hysteresis": hysteresis, "arbitration_mode": "rl_direct",
    } for target, (scores, owner, hysteresis, winner) in enumerate(cases)]
    # These defensive entries must not be mistaken for multi-candidate events.
    events.extend([{"candidates": [0], "winner": 0}, {"candidates": []}])
    evaluate, policy, shield, original, _, _ = _evaluate_fixture(events=events)
    rows = evaluate(policy, shield, [240, 241], k=3)
    assert policy.env is original
    assert [row["multi_candidate_events"] for row in rows] == [5, 5]
    assert [row["winner_changes_vs_fixed_score"] for row in rows] == [2, 2]
    assert [row["return"] for row in rows] == [2.2, 2.2]
    assert [row["violations"] for row in rows] == [1, 1]


def test_evaluate_paired_aggregates_natural_events_without_probe():
    rl_rows = [
        {"seed": 240, "terminal": "success", "return": -10.0,
         "steps": 2, "violations": 1, "multi_candidate_events": 5,
         "winner_changes_vs_fixed_score": 2},
        {"seed": 241, "terminal": "failed", "return": -30.0,
         "steps": 4, "violations": 2, "multi_candidate_events": 3,
         "winner_changes_vs_fixed_score": 1},
    ]
    heur_rows = [
        {"seed": 240, "terminal": "success", "return": -5.0,
         "steps": 3, "violations": 3},
        {"seed": 241, "terminal": "success", "return": -15.0,
         "steps": 3, "violations": 3},
    ]
    policy, shield = object(), object()
    calls = []

    def evaluate(pol, safety, seeds, k):
        assert pol is policy and safety is shield
        calls.append(("rl", list(seeds), k))
        return rl_rows

    def evaluate_heuristic(safety, seeds, k):
        assert safety is shield
        calls.append(("heur", list(seeds), k))
        return heur_rows

    namespace = {
        "evaluate": evaluate, "evaluate_heuristic": evaluate_heuristic,
        "_env_config": lambda: SimpleNamespace(STABLE_WINDOW=8, GEARS_TRACK=(7.5, 15.0)),
        "canonical_sha256": lambda config: "fixed-task-hash",
    }
    _pilot_function("_eval_summary", namespace)
    paired = _pilot_function("evaluate_paired", namespace)
    result = paired(policy, shield, [240, 241], k=3)
    assert calls == [("rl", [240, 241], 3), ("heur", [240, 241], 3)]
    assert result == {
        "k": 3, "seeds": [240, 241], "stable_window": 8,
        "task_config": {"STABLE_WINDOW": 8, "GEARS_TRACK": [7.5, 15.0]},
        "task_config_sha256": "fixed-task-hash",
        "rl": {"success_rate": 0.5, "mean_return": -20.0, "violations_per_step": 0.5},
        "heur": {"success_rate": 1.0, "mean_return": -10.0, "violations_per_step": 1.0},
        "natural_arbitration": {
            "rl_multi_candidate_events": 8,
            "rl_winner_changes_vs_fixed_score": 3,
            "forced_probe_enabled": False,
        },
        "by_seed": [{"seed": seed, "rl": rl, "heur": heur}
                    for seed, rl, heur in zip([240, 241], rl_rows, heur_rows)],
    }


def _candidate_fixture(tmp_path):
    calls = []
    policy = SimpleNamespace(weights=b"initial-best")
    best_path = tmp_path / "curr_k3_best.pt"
    log = {}
    run_record = {"run_id": "test-run", "effective_source": {
        "manifest_sha256": "source-fingerprint"}}

    def save_checkpoint(pol, path):
        calls.append(("save", Path(path)))
        path.write_bytes(pol.weights)

    def archive_checkpoint(record, path, role, name):
        calls.append(("archive", name))
        return {"sha256": file_sha256(path), "role": role, "name": name}

    def artifact_record(path, role):
        return {"sha256": file_sha256(path), "role": role}

    def add_output(record, path, role):
        calls.append(("output", role))

    namespace = {"math": math, "_save_checkpoint": save_checkpoint,
                 "_archive_checkpoint": archive_checkpoint,
                 "artifact_record": artifact_record, "add_output": add_output}
    _pilot_function("_validation_rank", namespace)
    consider = _pilot_function("_consider_curriculum_candidate", namespace)
    return consider, policy, best_path, log, run_record, calls


def _paired(success=0.9, reward=-3163.4, violations=4.7069):
    return {"rl": {"success_rate": success, "mean_return": reward,
                   "violations_per_step": violations},
            "heur": {"success_rate": 0.8, "mean_return": -3342.5,
                     "violations_per_step": 5.8651}}


def test_initial_baseline_is_saved_before_any_ppo_candidate(tmp_path):
    consider, policy, path, log, record, calls = _candidate_fixture(tmp_path)
    baseline = _paired()
    rank, improved = consider(policy, log, record, baseline, -1, path,
                             "curr_k3_best", None, "initial_checkpoint")
    assert improved is True
    assert rank == (0.9, -3163.4, -4.7069)
    assert path.read_bytes() == b"initial-best"
    assert [event[0] for event in calls] == ["save", "archive", "output"]
    best = log["curr_k3_best"]
    assert best["iter"] == -1
    assert best["stage"] == "initial_checkpoint"
    assert best["checkpoint_sha256"] == file_sha256(path)
    assert best["validation"] is baseline
    assert baseline["run_id"] == "test-run"
    assert baseline["source_manifest_sha256"] == "source-fingerprint"


@pytest.mark.parametrize("candidate", [
    _paired(), _paired(success=0.8, reward=1000.0, violations=0.0),
    _paired(reward=-4000.0, violations=0.0), _paired(violations=5.0),
])
def test_ties_and_worse_candidates_never_replace_initial_best(tmp_path, candidate):
    consider, policy, path, log, record, calls = _candidate_fixture(tmp_path)
    rank, _ = consider(policy, log, record, _paired(), -1, path,
                       "curr_k3_best", None, "initial_checkpoint")
    original_record = deepcopy(log["curr_k3_best"])
    original_bytes = path.read_bytes()
    calls.clear()
    policy.weights = b"degraded-or-tied-candidate"
    new_rank, improved = consider(policy, log, record, candidate, 0, path,
                                 "curr_k3_best", rank, "ppo_update")
    assert improved is False
    assert new_rank == rank
    assert calls == []
    assert path.read_bytes() == original_bytes
    assert log["curr_k3_best"] == original_record


@pytest.mark.parametrize("candidate", [
    _paired(success=1.0), _paired(reward=-3000.0), _paired(violations=4.5),
])
def test_strictly_better_candidate_replaces_best_with_matching_weights(tmp_path, candidate):
    consider, policy, path, log, record, calls = _candidate_fixture(tmp_path)
    rank, _ = consider(policy, log, record, _paired(), -1, path,
                       "curr_k3_best", None, "initial_checkpoint")
    calls.clear()
    policy.weights = b"improved-ppo-candidate"
    new_rank, improved = consider(policy, log, record, candidate, 4, path,
                                 "curr_k3_best", rank, "ppo_update")
    assert improved is True
    assert new_rank > rank
    assert [event[0] for event in calls] == ["save", "archive", "output"]
    assert path.read_bytes() == policy.weights
    assert log["curr_k3_best"]["iter"] == 4
    assert log["curr_k3_best"]["validation"] is candidate
    assert log["curr_k3_best"]["checkpoint_sha256"] == file_sha256(path)


@pytest.mark.parametrize("side", ["rl", "heur"])
@pytest.mark.parametrize("field", ["success_rate", "mean_return", "violations_per_step"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_validation_metrics_fail_before_saving(tmp_path, side, field, value):
    consider, policy, path, log, record, calls = _candidate_fixture(tmp_path)
    candidate = _paired()
    candidate[side][field] = value
    with pytest.raises(ValueError, match="Nonfinite paired validation metrics"):
        consider(policy, log, record, candidate, -1, path,
                 "curr_k3_best", None, "initial_checkpoint")
    assert calls == []
    assert not path.exists()
    assert log == {}


def test_atomic_checkpoint_preserves_existing_file_after_failed_save(tmp_path):
    path = _checkpoint(tmp_path / "best.pt", b"original-best")
    written_paths = []

    def failed_save(temporary):
        temporary = Path(temporary)
        written_paths.append(temporary)
        temporary.write_bytes(b"incomplete-new-weights")
        raise RuntimeError("interrupted-save")

    save = _pilot_function("_save_checkpoint", {"Path": Path})
    with pytest.raises(RuntimeError, match="interrupted-save"):
        save(SimpleNamespace(save=failed_save), path)
    assert path.read_bytes() == b"original-best"
    assert written_paths == [tmp_path / "best.tmp.pt"]


def test_atomic_checkpoint_replaces_only_after_completed_save(tmp_path):
    path = _checkpoint(tmp_path / "best.pt", b"original-best")

    def completed_save(temporary):
        assert path.read_bytes() == b"original-best"
        assert Path(temporary) != path
        Path(temporary).write_bytes(b"complete-new-weights")

    save = _pilot_function("_save_checkpoint", {"Path": Path})
    save(SimpleNamespace(save=completed_save), path)
    assert path.read_bytes() == b"complete-new-weights"
    assert not (tmp_path / "best.tmp.pt").exists()
