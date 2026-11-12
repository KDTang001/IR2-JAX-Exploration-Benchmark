"""Independent hand-computed metric oracles."""

import numpy as np
import pytest

from ir2_benchmark.evaluation.metrics import (
    EpisodeOutcome,
    MetricError,
    PlanSample,
    allocation_concentration,
    communication_totals,
    coverage_auc,
    cross_agent_overlap,
    denormalize_positions,
    grouped_plan_errors,
    normalize_positions,
    physical_coverage,
    plan_errors,
    summarize_episode,
)


def test_auc_is_round_mean_when_full_trajectory_recorded() -> None:
    # Given / When: four round-end coverage values.
    auc = coverage_auc([0.25, 0.5, 0.75, 1.0], horizon=4)
    # Then: trapezoidal integration would give the wrong value here.
    assert auc == 0.625  # noqa: PLR2004


def test_auc_holds_actual_termination_value_when_success_early() -> None:
    # Given / When: round two succeeds at 99%, not 100%.
    auc = coverage_auc([0.25, 0.99], horizon=4, terminated=True)
    # Then: the last actual coverage is held until the horizon.
    assert auc == pytest.approx(0.805)


def test_timeout_has_no_success_time_when_threshold_never_reached() -> None:
    # Given / When: incomplete exploration reaches its time limit.
    result = summarize_episode([0.25, 0.5], horizon=2, n_agents=2, outcome=EpisodeOutcome.TIMEOUT)
    # Then: failure is retained and has no invented threshold crossing time.
    assert not result.success
    assert result.success_round is None
    assert result.success_event_offset is None
    assert result.timeout
    assert result.final_coverage == 0.5  # noqa: PLR2004


def test_success_reports_event_offset_when_crossing_within_round() -> None:
    # Given / When: third event reaches success, freezing the second round.
    result = summarize_episode(
        [0.5, 0.99],
        horizon=4,
        n_agents=2,
        outcome=EpisodeOutcome.SUCCESS,
        event_coverage=[0.25, 0.5, 0.99],
    )
    # Then: round indexing is one-based; the event offset inside a round is zero-based.
    assert result.success_round == 2  # noqa: PLR2004
    assert result.success_event_offset == 0
    assert result.auc == pytest.approx(0.8675)


def test_physical_union_and_overlap_ignore_received_maps_when_maps_are_merged() -> None:
    # Given: personal observations {a,b} and {b,c}; d is unreachable ground truth.
    personal = np.array([[[1, 1, 0, 0]], [[0, 1, 1, 0]]], dtype=bool)
    component = np.array([[1, 1, 1, 0]], dtype=bool)
    received_maps = np.zeros((2, 1, 4), dtype=np.uint8)
    # When: calculating from physical footprints rather than received map contents.
    coverage = physical_coverage(personal, component)
    overlap = cross_agent_overlap(personal, component)
    # Then: unreachable or received-only cells cannot add coverage.
    assert coverage == 1.0
    assert overlap == 0.25  # noqa: PLR2004
    assert np.all(received_maps == 0)


def test_overlap_is_missing_when_no_free_cell_physically_observed() -> None:
    # Given / When: no robot has observed a free component cell.
    overlap = cross_agent_overlap(np.zeros((2, 2, 2), dtype=bool), np.ones((2, 2), dtype=bool))
    # Then: no arbitrary zero/one fills the undefined denominator.
    assert overlap is None


def test_allocation_concentration_uses_first_observer_counts_when_uneven() -> None:
    # Given / When: shares are 0.75 and 0.25.
    value = allocation_concentration([3, 1])
    # Then: concentration is squared shares, not overlap or variance.
    assert value == 0.625  # noqa: PLR2004


def test_communication_counts_broadcast_receivers_when_payload_exceeds_emission() -> None:
    # Given / When: one PLAN is broadcast to three recipients.
    result = communication_totals([45], [29], np.array([[1, 1, 1]], dtype=bool))
    # Then: only one charged packet, but three delivered payload copies.
    assert result.emitted_bytes == 45  # noqa: PLR2004
    assert result.delivered_payload_bytes == 87  # noqa: PLR2004
    assert result.delivered_messages == 3  # noqa: PLR2004


def test_ade_fde_omit_padded_positions_when_mask_sparse() -> None:
    # Given: valid displacements 5 and 10, with irrelevant huge padding.
    predicted = np.array([[3, 4], [999, 999], [6, 8], [999, 999]], dtype=float)
    target = np.zeros((4, 2))
    # When: applying an explicit validity mask.
    result = plan_errors(predicted, target, np.array([1, 0, 1, 0], dtype=bool))
    # Then: final error uses the last valid timestep, not the array end.
    assert result.ade == 7.5  # noqa: PLR2004
    assert result.fde == 10.0  # noqa: PLR2004
    assert result.valid_points == 2  # noqa: PLR2004


def test_plan_error_is_missing_when_valid_mask_empty() -> None:
    # Given / When: no path entries are valid.
    result = plan_errors(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros(3, dtype=bool))
    # Then: missing labels never create a perfect prediction.
    assert result.ade is None
    assert result.fde is None


def test_xy_coordinate_normalization_roundtrips_when_rectangular() -> None:
    # Given: valid_shape is height,width, coordinates are x,y.
    positions = np.array([[0, 0], [9, 3], [4.5, 1.5]])
    # When: normalize by the last valid x and y coordinate.
    normalized = normalize_positions(positions, (4, 10))
    # Then: axes are not swapped and restoration preserves positions.
    np.testing.assert_array_equal(normalized, [[0, 0], [1, 1], [0.5, 0.5]])
    np.testing.assert_allclose(denormalize_positions(normalized, (4, 10)), positions)


def test_grouped_errors_preserve_cold_start_and_phase_when_messages_missing() -> None:
    # Given: cold-start and received-message samples at different ages/phases.
    samples = (
        PlanSample(1.0, 2.0, None, "normal", "intention"),
        PlanSample(3.0, 4.0, 2, "fault", "intention"),
        PlanSample(5.0, 6.0, 2, "recovery", "actual_position"),
    )
    # When: stratifying quality without discarding no-message samples.
    groups = grouped_plan_errors(samples)
    # Then: prediction target, message age, and phase remain independent strata.
    assert len(groups) == 3  # noqa: PLR2004
    assert next(group for group in groups if group.cold_start).count == 1
    assert {group.phase for group in groups} == {"normal", "fault", "recovery"}


def test_short_timeout_trajectory_is_rejected_when_horizon_not_reached() -> None:
    # Given / When / Then: a runtime interruption is not silently padded into a valid timeout.
    with pytest.raises(MetricError, match="unfinished_trajectory"):
        coverage_auc([0.25], horizon=4, terminated=False)


def test_runtime_failure_is_distinct_from_timeout_when_model_raises() -> None:
    # Given / When: a run stops early due to a model error.
    result = summarize_episode([0.25], horizon=4, n_agents=2, outcome=EpisodeOutcome.MODEL_ERROR)
    # Then: no successful or normal-timeout episode metric is fabricated.
    assert result.outcome == EpisodeOutcome.MODEL_ERROR
    assert not result.timeout
    assert not result.success
    assert result.auc is None
    assert result.success_round is None


def test_nll_calibration_and_goal_error_use_valid_coordinates_when_padding_nan() -> None:
    # Given: one valid zero-error point and NaN values in padding.
    predicted = np.array([[0.0, 0.0], [np.nan, np.nan]])
    target = np.zeros((2, 2))
    sigma = np.array([[1.0, 1.0], [np.nan, np.nan]])
    # When: compute optional probabilistic and target metrics.
    result = plan_errors(
        predicted,
        target,
        np.array([True, False]),
        stddev=sigma,
        predicted_goal=np.array([3.0, 4.0]),
        target_goal=np.array([0.0, 0.0]),
    )
    # Then: two-dimensional unit Gaussian NLL is log(2pi); padding never enters it.
    assert result.nll == pytest.approx(np.log(2 * np.pi))
    assert result.coverage_one_sigma == 1.0
    assert result.goal_error == 5.0  # noqa: PLR2004
