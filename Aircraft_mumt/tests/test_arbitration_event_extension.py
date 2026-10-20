"""Test collection continuation and provenance without simulator performance claims."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _extend_arbitration_event_bank as extension
from _audit_arbitration_learning import _json_ready, case_coverage
from _curriculum_guard import file_sha256
from _experiment_provenance import (
    canonical_sha256, capture_source_snapshot, write_json_atomic,
)
from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_mappo import DetectMAPPOConfig


def _case(seed):
    return {"episode_seed": seed, "step_index": 0,
            "event": {"candidates": [0, 1], "bid_slots": {"0": 0, "1": 1},
                      "utilities": {"0": 0.0, "1": 1.0}}}


def _persist(source, report, bank):
    write_json_atomic(source / "event_bank.json", bank)
    report["event_bank_sha256"] = file_sha256(source / "event_bank.json")
    report.pop("report_content_sha256", None)
    report["report_content_sha256"] = canonical_sha256(report)
    write_json_atomic(source / "learning_audit.json", report)


@pytest.fixture
def stopped_bank(tmp_path):
    source = tmp_path / "stopped"
    source.mkdir()
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"unchanged source checkpoint")
    snapshot = capture_source_snapshot(extension.ROOT, source, [
        extension.ROOT / "_audit_arbitration_learning.py",
        extension.ROOT / "_arbitration_training.py"])
    env_cfg = replace(DetectSimConfig(), STABLE_WINDOW=8, AGE_STD_TRACK=10.0,
                      SECTOR_AXIS_MAX_DEG=60.0, C04_RANGE=80.0,
                      ARB_BID_WEIGHT=0.35, ARB_HYSTERESIS=0.2)
    config = DetectMAPPOConfig(cf_bid_horizon=8)
    utility = {name: getattr(config, name) for name in (
        "gamma", "reward_scale", "shaping_kappa", "shaping_age_cap",
        "geo_shaping_kappa", "stable_shaping_kappa")}
    utility.update(cf_bid_horizon=8,
                   continuation="frozen_deterministic_rl_actor_and_independent_branch_shield")
    settings = {"checkpoint": str(checkpoint), "train_seeds": "1,2",
                "heldout_seeds": "3,4", "max_events_per_episode": 3,
                "min_event_gap": 8, "max_events_per_split": 40,
                "min_train_event_episodes": 1, "min_heldout_event_episodes": 2,
                "conflict_exposure": 1.0, "learning_rate": 0.0001, "epochs": 2,
                "utility_scale": 0.1, "cf_bid_horizon": 8,
                "training_seed": 20261021, "repair_seed": 20261021}
    bank = {"schema_version": 1, "source_checkpoint_sha256": file_sha256(checkpoint),
            "k": 4, "environment_config": _json_ready(asdict(env_cfg)),
            "collection_source_snapshot": snapshot, "utility_definition": utility,
            "selection_policy": {key: settings[key] for key in (
                "max_events_per_episode", "min_event_gap")},
            "train_seeds": [1, 2], "heldout_seeds": [3, 4],
            "train": [_case(1)], "heldout": [_case(3)],
            "collection_episodes": {"train": [{"seed": 1}, {"seed": 2}],
                                    "heldout": [{"seed": 3}, {"seed": 4}]}}
    report = {"status": "insufficient_independent_event_coverage", "updates": [],
              "source_checkpoint_sha256": file_sha256(checkpoint),
              "source_changed_during_run": [], "arguments": settings,
              "coverage": {split: case_coverage(bank[split])
                           for split in ("train", "heldout")}}
    _persist(source, report, bank)
    return source, report, bank, checkpoint


def test_valid_stopped_bank_retains_fixed_split_budget(stopped_bank):
    source, original, bank, checkpoint = stopped_bank
    report, loaded, source_checkpoint, remaining = extension.load_stopped_bank(source, [5, 6])
    assert report == original
    assert loaded == bank
    assert source_checkpoint == checkpoint
    assert remaining == 39


@pytest.mark.parametrize("seeds", [[], [-1], [5, 5], [1], [3], [2, 6]])
def test_new_seeds_must_be_distinct_from_all_original_episodes(stopped_bank, seeds):
    with pytest.raises(ValueError, match="unique and disjoint"):
        extension.load_stopped_bank(stopped_bank[0], seeds)


@pytest.mark.parametrize("change, message", [
    (lambda report, bank: report.update(updates=[{"iteration": 0}]), "before any training"),
    (lambda report, bank: report.update(status="completed_diagnostic"), "before any training"),
    (lambda report, bank: report["arguments"].update(min_train_event_episodes=2),
     "Training coverage is also insufficient"),
    (lambda report, bank: report["arguments"].update(max_events_per_split=1),
     "No capacity remains"),
    (lambda report, bank: bank.update(train_seeds=[1, 3]), "seeds overlap"),
    (lambda report, bank: report["arguments"].update(heldout_seeds="3,9"), "seed record mismatch"),
    (lambda report, bank: bank["selection_policy"].update(min_event_gap=0),
     "temporal selection policy mismatch"),
    (lambda report, bank: report["coverage"]["heldout"].update(events=50),
     "coverage record mismatch"),
    (lambda report, bank: report.update(source_changed_during_run=["detect_env.py"]),
     "source changed during collection"),
])
def test_incompatible_stopped_runs_are_rejected(stopped_bank, change, message):
    source, report, bank, _ = stopped_bank
    change(report, bank)
    _persist(source, report, bank)
    with pytest.raises(ValueError, match=message):
        extension.load_stopped_bank(source, [5, 6])


def test_report_and_bank_tampering_are_detected_independently(stopped_bank):
    source, report, bank, _ = stopped_bank
    changed = deepcopy(report)
    changed["arguments"]["repair_seed"] += 1
    write_json_atomic(source / "learning_audit.json", changed)
    with pytest.raises(ValueError, match="report content hash mismatch"):
        extension.load_stopped_bank(source, [5])
    write_json_atomic(source / "learning_audit.json", report)
    changed_bank = deepcopy(bank)
    changed_bank["heldout"][0]["event"]["utilities"]["0"] = 20.0
    write_json_atomic(source / "event_bank.json", changed_bank)
    with pytest.raises(ValueError, match="event bank SHA256 mismatch"):
        extension.load_stopped_bank(source, [5])


def test_changed_source_or_checkpoint_cannot_be_mixed(stopped_bank, monkeypatch):
    source, _, _, checkpoint = stopped_bank
    with monkeypatch.context() as patch:
        patch.setattr(extension, "changed_source_files", lambda snapshot: ["changed.py"])
        with pytest.raises(ValueError, match="sources changed"):
            extension.load_stopped_bank(source, [5])
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        extension.load_stopped_bank(source, [5])


def test_trained_checkpoint_in_source_run_rejects_extension(stopped_bank):
    source = stopped_bank[0]
    (source / "final.pt").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="contains model checkpoints"):
        extension.load_stopped_bank(source, [5])


def test_preflight_collects_nothing_and_preserves_all_original_bytes(stopped_bank, monkeypatch):
    source = stopped_bank[0]
    output = source.parent / "preflight"
    files_before = {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    monkeypatch.setattr(extension, "collect_cases", lambda *args: pytest.fail("collected"))
    assert extension.main(["--source-run", str(source), "--output", str(output),
                           "--heldout-seeds", "5,6", "--preflight-only"]) == 0
    assert not output.exists()
    assert files_before == {str(path): path.read_bytes()
                            for path in source.rglob("*") if path.is_file()}


class _CollectionOnlyPolicy:
    def __init__(self, env, shield, config, seed):
        self.env, self.config = env, config
        self.model = torch.nn.Linear(1, 1)

    def load(self, path):
        assert Path(path).read_bytes() == b"unchanged source checkpoint"

    def update(self, *args, **kwargs):
        pytest.fail("Collection continuation must never train")


def _stub_policy(monkeypatch):
    import aircraft_sim.detect_mappo as mappo
    import _arbitration_training as training
    monkeypatch.setattr(mappo, "DetectMAPPOPolicy", _CollectionOnlyPolicy)
    monkeypatch.setattr(training, "repair_dead_candidate_head_for_training",
                        lambda *args, **kwargs: {"status": "skipped_healthy"})


def test_extension_reuses_original_cases_and_collects_entire_fixed_block(stopped_bank, monkeypatch):
    source, _, bank, _ = stopped_bank
    output = source.parent / "extended"
    original_bytes = {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    _stub_policy(monkeypatch)
    calls = []

    def collect(policy, seeds, k, budget, per_episode, gap):
        calls.append((seeds, k, budget, per_episode, gap))
        # Later seeds are collected even if the first fills the coverage gate.
        return [_case(5), _case(7)], [{"seed": seed} for seed in seeds]

    monkeypatch.setattr(extension, "collect_cases", collect)
    assert extension.main(["--source-run", str(source), "--output", str(output),
                           "--heldout-seeds", "5,6,7", "--collection-seed", "17"]) == 0
    merged = json.loads((output / "event_bank.json").read_text())
    report = json.loads((output / "collection_report.json").read_text())
    assert calls == [([5, 6, 7], 4, 39, 3, 8)]
    assert merged["train"] == bank["train"]
    assert merged["heldout"] == bank["heldout"] + [_case(5), _case(7)]
    assert merged["train_seeds"] == bank["train_seeds"]
    assert merged["heldout_seeds"] == [3, 4, 5, 6, 7]
    assert merged["collection_episodes"]["heldout"] == bank["collection_episodes"]["heldout"] + [
        {"seed": 5}, {"seed": 6}, {"seed": 7}]
    assert report["updates"] == []
    assert report["training_cases_unchanged"] is True
    assert report["original_heldout_cases_unchanged"] is True
    assert report["source_changed_during_run"] == []
    assert report["event_bank_sha256"] == file_sha256(output / "event_bank.json")
    content = {key: value for key, value in report.items() if key != "report_content_sha256"}
    assert report["report_content_sha256"] == canonical_sha256(content)
    assert not list(output.glob("*.pt"))
    assert original_bytes == {str(path): path.read_bytes()
                              for path in source.rglob("*") if path.is_file()}


def test_weight_mutation_during_collection_is_rejected(stopped_bank, monkeypatch):
    source = stopped_bank[0]
    output = source.parent / "invalid"
    _stub_policy(monkeypatch)

    def collect(policy, *args):
        with torch.no_grad():
            policy.model.weight.add_(1)
        return [], []

    monkeypatch.setattr(extension, "collect_cases", collect)
    with pytest.raises(RuntimeError, match="changed policy weights"):
        extension.main(["--source-run", str(source), "--output", str(output),
                        "--heldout-seeds", "5,6"])
    assert not (output / "event_bank.json").exists()


@pytest.mark.parametrize("kind", ["same", "child", "parent", "occupied"])
def test_extension_does_not_overwrite_or_nest_original_run(stopped_bank, kind):
    source = stopped_bank[0]
    if kind == "same":
        output = source
    elif kind == "child":
        output = source / "child"
    elif kind == "parent":
        output = source.parent
    else:
        output = source.parent / "occupied"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="separate|new or empty"):
        extension.main(["--source-run", str(source), "--output", str(output),
                        "--heldout-seeds", "5,6", "--preflight-only"])
