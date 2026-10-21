from _diagnose_fixed_policy_steps import _merge_wing_counts, summarize_episode


def test_merge_wing_counts_reads_nested_per_aircraft_summary():
    rows = [{
        "summary": {
            "switches_by_wingman": {
                "0": {"gear_state_counts": {"45.0": 3}},
                "1": {"gear_state_counts": {"30.0": 2}},
                "2": {"gear_state_counts": {"15.0": 1}},
            }
        }
    }]

    assert _merge_wing_counts(rows, "gear_state_counts") == {
        "0": {"45.0": 3}, "1": {"30.0": 2}, "2": {"15.0": 1}}


def test_owned_distance_bins_exclude_tracks_held_by_other_aircraft():
    track = {
        "owned_by": 1, "d_eff_km": 50.0,
        "local_estimate_c04_geometry": {"failure_mode": None},
    }
    steps = [{
        "switch": {str(i): {"switch_attempt": False, "switch_denied": False,
                            "switch_ready_before": False} for i in range(3)},
        "radar_before": {str(i): {"axis_deg": 0.0, "gear_deg": 30.0, "phase": 0}
                         for i in range(3)},
        "actions": {str(i): {"gear_deg": 30.0} for i in range(3)},
        "referee_request_gate_outcomes": [],
        "local_estimate_diagnostics": {
            "tracks_before": {"0": [track], "1": [track], "2": []},
            "potential_before": {"phi_geo": 0.0, "phi_align": 0.0},
        },
        "radar_dwell_by_wing_gear_phase": {},
        "request_estimates_from_local_tracks": [],
    }]

    summary = summarize_episode(steps)

    assert summary["owned_local_track_d_eff_bins_km"]["MAR_to_MOR"] == 1
    assert summary["owned_local_track_d_eff_bins_km"]["below_MAR"] == 0
    assert summary["owned_local_track_d_eff_bins_km"]["at_or_above_MOR"] == 0
    assert summary["switches_by_wingman"]["0"]["radar_phase_state_counts"] == {"0": 1}
