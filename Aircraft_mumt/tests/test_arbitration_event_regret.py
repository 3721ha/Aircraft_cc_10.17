from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from _diagnose_arbitration_event_regret import (
    _assert_reference_match, _normalize_horizons, _summarize_horizons,
    _verify_reference, compare_winners,
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
                             "winners_different_from_fixed_score": 1}, direct[10])
    with pytest.raises(ValueError, match="exactly match"):
        _verify_reference(reference, [11])

    changed = deepcopy(reference["episodes"][0])
    changed["return"] = -9.0
    with pytest.raises(RuntimeError, match="return differs"):
        _assert_reference_match({"seed": 10, "terminal": "success", "return": -10.0,
                                 "steps": 20, "multi_candidate_events": 3,
                                 "winners_different_from_fixed_score": 1}, changed)
