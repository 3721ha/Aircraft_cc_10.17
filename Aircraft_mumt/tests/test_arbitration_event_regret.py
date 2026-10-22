from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _diagnose_arbitration_event_regret import (
    _assert_reference_match, _normalize_horizons, _summarize_horizons,
    _counterfactual_unavailable_counts, _task_config,
    _verify_profile_sweep_reference,
    _verify_reference,
    compare_winners,
)


def _event():
    return {
        "tid": 7, "candidates": [0, 1], "winner": 1,
        "previous_owner": 0, "scores": {0: 1.0, 1: 0.7},
        "hysteresis": 0.2,
    }


def test_same_snapshot_comparison_uses_fixed_score_winner_and_event_utilities():
    result = compare_winners(_event(), {
        "candidates": [0, 1], "utilities": {0: 1.25, 1: 2.0},
        "branch_results": {"0": {"terminal": False}, "1": {"terminal": False}},
    })

    assert result["fixed_score_winner"] == 0
    assert result["rl_winner"] == 1
    assert result["winners_differ"] is True
    assert result["rl_utility_minus_fixed_score"] == pytest.approx(0.75)
    assert result["rl_regret"] == 0.0
    assert result["fixed_score_regret"] == pytest.approx(0.75)


def test_horizon_summaries_keep_utility_scales_separate():
    rows = [{"event_regrets": [{
        "seed": 11,
        "counterfactual_by_horizon": {
            "1": {"counterfactual_available": True,
                  "winners_differ": True, "utility_spread": 2.0,
                  "rl_utility_minus_fixed_score": -1.0},
            "4": {"counterfactual_available": True,
                  "winners_differ": True, "utility_spread": 3.0,
                  "rl_utility_minus_fixed_score": 0.5},
            "8": {"counterfactual_available": False},
        },
    }]}]

    summaries = _summarize_horizons(rows, [1, 4, 8])

    assert summaries["1"]["counterfactual_events"] == 1
    assert summaries["1"]["fixed_score_better_changed_events"] == 1
    assert summaries["4"]["rl_better_changed_events"] == 1
    assert summaries["8"]["counterfactual_events"] == 0


def test_counterfactual_horizons_are_deduplicated_and_bounded():
    assert _normalize_horizons([1, 4, 4, 8]) == [1, 4, 8]
    for invalid in ([], [0], [33], [True]):
        with pytest.raises(ValueError, match="integers in \\[1, 32\\]"):
            _normalize_horizons(invalid)


def test_uncontrollable_event_count_deduplicates_across_horizons():
    rows = [
        {"step": 4, "tid": 2, "candidates": (0, 1),
         "counterfactual_by_horizon": {
             "1": {"counterfactual_available": False,
                   "unavailable_reason": "branch_uncontrollable"},
             "4": {"counterfactual_available": False,
                   "unavailable_reason": "branch_uncontrollable"},
             "8": {"counterfactual_available": True}}},
        {"step": 7, "tid": 1, "candidates": (1, 2),
         "counterfactual_by_horizon": {
             "1": {"counterfactual_available": True},
             "4": {"counterfactual_available": False,
                   "unavailable_reason": "preconflict_snapshot_unavailable"},
             "8": {"counterfactual_available": True}}},
    ]
    unavailable, snapshot_missing, branch_failed, unique = (
        _counterfactual_unavailable_counts(rows, [1, 4, 8]))
    assert unavailable == {1: 1, 4: 2, 8: 0}
    assert snapshot_missing == {1: 0, 4: 1, 8: 0}
    assert branch_failed == {1: 1, 4: 1, 8: 0}
    assert unique == {(4, 2, (0, 1)), (7, 1, (1, 2))}


def test_reference_replay_requires_exact_seed_block_and_metrics():
    reference = {
        "status": "completed_frozen_arbitration_ablation",
        "seeds": [10], "loaded_tensor_sha256": "weights",
        "episodes": [{"seed": 10, "arbitration": "rl_direct",
                      "terminal": "success", "return": -10.0,
                      "steps": 20, "multi_candidate_events": 3,
                      "winners_different_from_fixed_score": 1,
                      "weights_unchanged": True,
                      "loaded_tensor_sha256": "weights",
                      "effective_policy_config": {
                          "direct_rl_arbitration": True,
                      }}],
    }
    direct = _verify_reference(reference, [10], "weights")
    _assert_reference_match({"seed": 10, "terminal": "success", "return": -10.0,
                             "steps": 20, "multi_candidate_events": 3,
                             "winners_different_from_fixed_score": 1,
                             "loaded_tensor_sha256": "weights"}, direct[10])
    with pytest.raises(ValueError, match="exactly match"):
        _verify_reference(reference, [11])

    changed = deepcopy(reference["episodes"][0])
    changed["return"] = -9.0
    with pytest.raises(RuntimeError, match="metrics differ for return"):
        _assert_reference_match({"seed": 10, "terminal": "success", "return": -10.0,
                                 "steps": 20, "multi_candidate_events": 3,
                                 "winners_different_from_fixed_score": 1}, changed)


def test_k2_lane50_event_audit_config_matches_profile_sweep_config():
    from aircraft_sim.detect_config import detect_config_profile

    actual = _task_config("bernoulli", k=2, stable_window=8,
                          age_std_track=10.0,
                          scenario_profile="k2_lane_center_50")
    expected = detect_config_profile(
        "k2_lane_center_50", K_RED=2, STABLE_WINDOW=8,
        AGE_STD_TRACK=10.0, RADAR_HIT_MODEL="bernoulli")
    task = asdict(actual)
    assert task == {**asdict(expected), "W8_SCALE": 1.0}


def test_profile_sweep_reference_requires_exact_checkpoint_profile_and_seeds():
    cfg = _task_config("bernoulli", k=2, stable_window=8,
                       age_std_track=10.0,
                       scenario_profile="k2_lane_center_50")
    expected_config = asdict(cfg)
    reference = {
        "status": "completed_frozen_profile_screen",
        "checkpoint": {"sha256": "checkpoint-sha"},
        "seeds": [500000],
        "task_profiles": {"k2_lane_center_50": expected_config},
        "training": False,
        "teacher_untouched": True,
        "rules_unchanged": True,
        "rows": [{
            "task_profile": "k2_lane_center_50", "controller": "rl",
            "seed": 500000, "terminal": "success", "return": -50.0,
            "steps": 20, "multi_candidate_events": 1,
            "winners_different_from_fixed_score": 1,
            "loaded_tensor_sha256": "tensor-sha", "weights_unchanged": True,
            "effective_policy_config": {"direct_rl_arbitration": True},
        }],
    }

    direct = _verify_profile_sweep_reference(
        reference, [500000], "checkpoint-sha", "k2_lane_center_50",
        expected_config)
    assert direct[500000]["loaded_tensor_sha256"] == "tensor-sha"

    with pytest.raises(ValueError, match="covered"):
        _verify_profile_sweep_reference(
            reference, [500001], "checkpoint-sha", "k2_lane_center_50",
            expected_config)
    missing_rule_attestation = deepcopy(reference)
    missing_rule_attestation["rules_unchanged"] = False
    with pytest.raises(ValueError, match="without teacher use"):
        _verify_profile_sweep_reference(
            missing_rule_attestation, [500000], "checkpoint-sha",
            "k2_lane_center_50", expected_config)
    with pytest.raises(ValueError, match="different checkpoint"):
        _verify_profile_sweep_reference(
            reference, [500000], "other-sha", "k2_lane_center_50",
            expected_config)
    with pytest.raises(ValueError, match="task configuration"):
        _verify_profile_sweep_reference(
            reference, [500000], "checkpoint-sha", "baseline",
            expected_config)
    with pytest.raises(ValueError, match="inconsistent policy fingerprints"):
        malformed = deepcopy(reference)
        malformed["rows"][0]["loaded_tensor_sha256"] = None
        _verify_profile_sweep_reference(
            malformed, [500000], "checkpoint-sha", "k2_lane_center_50",
            expected_config)


def test_replay_reference_match_checks_loaded_policy_fingerprint():
    reference = {"seed": 12, "terminal": "success", "return": -3.0,
                 "steps": 10, "multi_candidate_events": 2,
                 "winners_different_from_fixed_score": 1,
                 "loaded_tensor_sha256": "expected"}
    matching = {key: value for key, value in reference.items()}
    _assert_reference_match(matching, reference)
    matching["loaded_tensor_sha256"] = "different"
    with pytest.raises(RuntimeError, match="different policy tensors"):
        _assert_reference_match(matching, reference)


def test_replay_reference_match_checks_episode_violation_metrics():
    reference = {"seed": 13, "terminal": "success", "return": -3.0,
                 "steps": 10, "violations": 4, "violations_per_step": 0.4,
                 "judged_targets": 2, "shield_interventions": 1,
                 "violations_by_family": {"C04": 4},
                 "multi_candidate_events": 2,
                 "winners_different_from_fixed_score": 1}
    matching = {**reference}
    _assert_reference_match(matching, reference)
    matching["violations_by_family"] = {"C04": 3, "S05": 1}
    with pytest.raises(RuntimeError, match="violations_by_family"):
        _assert_reference_match(matching, reference)
