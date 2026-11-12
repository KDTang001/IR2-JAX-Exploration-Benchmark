"""Pair actual Normal/Burst trajectories by frozen map/spawn/fault/config identity."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ir2_benchmark.config import FAULT_BURST
from ir2_benchmark.evaluation.scenarios import identity, read_task_manifest

if TYPE_CHECKING:
    from ir2_benchmark.evaluation.report import ReportRow
    from ir2_benchmark.evaluation.scenarios import Scenario


@dataclass(frozen=True, slots=True)
class WindowDifference:
    """Fault minus Normal gains in fractions of physically covered free cells."""

    normal_fault_gain: float | None
    burst_fault_gain: float | None
    fault_minus_normal_fault_window: float | None
    normal_recovery_gain: float | None
    burst_recovery_gain: float | None
    fault_minus_normal_recovery_window: float | None


def paired_gains(  # noqa: PLR0913
    normal: np.ndarray,
    burst: np.ndarray,
    start: int,
    duration: int,
    *,
    recovery_length: int = 64,
    initial_normal: float = 0,
    initial_burst: float = 0,
) -> WindowDifference:
    """Compare independent actual trajectories, holding terminal coverage already stored in each."""
    horizon = min(len(normal), len(burst))

    def gain(values: np.ndarray, begin: int, end: int, initial: float) -> float | None:
        if begin >= horizon or end <= begin:
            return None
        before = initial if begin == 0 else values[begin - 1]
        return float(values[min(end, horizon) - 1] - before)

    end = start + duration
    normal_fault = gain(normal, start, end, initial_normal)
    burst_fault = gain(burst, start, end, initial_burst)
    normal_recovery = gain(normal, end, end + recovery_length, initial_normal)
    burst_recovery = gain(burst, end, end + recovery_length, initial_burst)
    return WindowDifference(
        normal_fault,
        burst_fault,
        burst_fault - normal_fault
        if normal_fault is not None and burst_fault is not None
        else None,
        normal_recovery,
        burst_recovery,
        burst_recovery - normal_recovery
        if normal_recovery is not None and burst_recovery is not None
        else None,
    )


def _pair_key(row: ReportRow, scenario: Scenario) -> str:
    """Remove only the external fault condition; preserve all other paired semantics."""
    config = asdict(scenario.config)
    for field in ("fault_mode", "dropout_probability", "burst_start", "burst_duration"):
        del config[field]
    return identity(
        [
            row.method_id,
            row.training_seed,
            scenario.map_hash,
            scenario.map_scenario_hash,
            scenario.spawn_seed,
            scenario.fault_seed,
            config,
            scenario.budget_mode,
            scenario.requested_team_bytes,
            scenario.unallocated_bytes,
        ]
    )


def write_recovery(
    rows: tuple[ReportRow, ...], manifest: Path | None, output: Path
) -> tuple[str, ...]:
    """Export paired gains or a named missing counterpart, never an assumed Normal baseline."""
    scenes = (
        {task.result_id: task.scenario for task in read_task_manifest(manifest)} if manifest else {}
    )
    normal = {
        _pair_key(row, scenes[row.result_id]): row
        for row in rows
        if row.fault_mode == 0 and row.result_id in scenes and row.trace_file
    }
    missing = []
    fields = (
        "result_id",
        "normal_result_id",
        "method_id",
        "training_seed",
        "split",
        "map_hash",
        "n_agents",
        "budget",
        "budget_mode",
        "requested_team_bytes",
        "unallocated_bytes",
        "burst_start",
        "burst_duration",
        "pair_status",
        *WindowDifference.__dataclass_fields__,
    )
    with (output / "recovery.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.fault_mode != FAULT_BURST:
                continue
            scene = scenes.get(row.result_id)
            counterpart = normal.get(_pair_key(row, scene)) if scene is not None else None
            record = {
                "result_id": row.result_id,
                "normal_result_id": counterpart.result_id if counterpart else "",
                "method_id": row.method_id,
                "training_seed": row.training_seed,
                "split": row.split,
                "map_hash": row.map_hash,
                "n_agents": row.n_agents,
                "budget": row.budget,
                "budget_mode": row.budget_mode,
                "requested_team_bytes": row.requested_team_bytes,
                "unallocated_bytes": row.unallocated_bytes,
                "burst_start": scene.burst_start if scene else None,
                "burst_duration": row.burst_duration,
            }
            if (
                counterpart is None
                or not Path(counterpart.trace_file).is_file()
                or not Path(row.trace_file).is_file()
            ):
                record["pair_status"] = "missing_normal_or_trace"
                missing.append(row.result_id)
            else:
                with np.load(counterpart.trace_file) as first, np.load(row.trace_file) as second:
                    comparison = paired_gains(
                        first["coverage"][row.n_agents - 1 :: row.n_agents],
                        second["coverage"][row.n_agents - 1 :: row.n_agents],
                        scene.burst_start,
                        scene.burst_duration,
                        initial_normal=float(first["initial_coverage"]),
                        initial_burst=float(second["initial_coverage"]),
                    )
                record.update(asdict(comparison))
                record["pair_status"] = "paired_actual_trajectories"
            writer.writerow(record)
    return tuple(missing)
