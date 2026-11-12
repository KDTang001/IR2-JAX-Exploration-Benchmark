"""Actual paired trajectories have independently computable fault and recovery gains."""

from pathlib import Path

import numpy as np

from ir2_benchmark.evaluation.recovery import paired_gains, write_recovery
from ir2_benchmark.evaluation.statistics import ReportRow


def test_fault_and_recovery_windows_compare_actual_normal_gains() -> None:
    # Given: independently realized Normal and Burst coverage curves, not copied positions.
    normal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    burst = np.array([0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6])
    # When: comparing rounds2..3 and the following4 recovery rounds.
    result = paired_gains(normal, burst, 2, 2, recovery_length=4)
    # Then: both windows retain a0.1 coverage deficit relative to their paired Normal run.
    np.testing.assert_allclose(result.fault_minus_normal_fault_window, -0.1)
    np.testing.assert_allclose(result.fault_minus_normal_recovery_window, -0.1)


def test_no_recovery_window_remains_missing_instead_of_zero_gain() -> None:
    # Given: an episode ending exactly at the end of the Burst window.
    normal, burst = np.array([0.1, 0.2, 0.4]), np.array([0.1, 0.15, 0.2])
    # When: requesting a recovery comparison beyond the recorded horizon.
    result = paired_gains(normal, burst, 1, 2)
    # Then: missing recovery observations are not presented as successful recovery.
    assert result.fault_minus_normal_recovery_window is None
    assert result.normal_recovery_gain is None


def test_missing_normal_counterpart_is_reported_explicitly(tmp_path: Path) -> None:
    # Given: an observed Burst result with no matching Normal task/trace.
    burst = ReportRow("unpaired", "A", 0, "ir2", "rooms", "map", fault_mode=2)
    # When: the recovery table is generated from the available workload.
    missing = write_recovery((burst,), None, tmp_path)
    # Then: no numeric recovery benefit is invented for the missing counterpart.
    assert missing == ("unpaired",)
    assert "missing_normal_or_trace" in (tmp_path / "recovery.csv").read_text(encoding="utf-8")
