"""Export only decision-time intention diagnostics actually recorded by the actor adapter."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ir2_benchmark.evaluation.metrics import PlanSample, grouped_plan_errors

if TYPE_CHECKING:
    from ir2_benchmark.evaluation.report import ReportRow


def _present(value: np.floating) -> float | None:
    """Preserve NaN-masked unavailable estimates as missing metric values."""
    return float(value) if np.isfinite(value) else None


def write_plan_quality(rows: tuple[ReportRow, ...], output: Path) -> tuple[str, ...]:
    """Keep age/phase/cold-start and per-result identity, without retrospective inference."""
    missing = []
    records = 0
    with (output / "plan_quality.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for row in rows:
            if row.route in ("heuristic", "baseline") or row.outcome not in ("success", "timeout"):
                continue
            if not row.plan_diagnostics or not Path(row.trace_file).is_file():
                missing.append(row.result_id)
                continue
            with np.load(row.trace_file) as trace:
                samples = []
                for event, peer in np.argwhere(trace["plan_has_target"]):
                    age = int(trace["plan_message_age"][event, peer])
                    cold = bool(trace["plan_cold_start"][event, peer])
                    phase = ("normal", "fault", "recovery")[int(trace["plan_phase"][event])]
                    samples.append(
                        PlanSample(
                            _present(trace["plan_ade"][event, peer]),
                            _present(trace["plan_fde"][event, peer]),
                            None if cold or age < 0 else age,
                            phase,
                            "intention",
                            _present(trace["plan_goal_error"][event, peer]),
                            _present(trace["plan_nll"][event, peer]),
                            _present(trace["plan_coverage_one_sigma"][event, peer]),
                        )
                    )
            for group in grouped_plan_errors(samples):
                record = {
                    "result_id": row.result_id,
                    "method_id": row.method_id,
                    "training_seed": row.training_seed,
                    "source": row.source,
                    "family": row.family,
                    "map_hash": row.map_hash,
                    **asdict(group),
                }
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                records += 1
    (output / "plan_quality_status.json").write_text(
        json.dumps(
            {
                "group_rows": records,
                "missing_result_ids": missing,
                "prediction_target": "event_before_latest_generated_intention",
                "ade_fde_units": "map_cells",
                "nll_units": "normalized_coordinates",
                "zoh_uncertainty": "unavailable",
                "future_actual_positions": "not_used_as_intention_labels",
                "aggregation": "age/phase/cold-start per result; map and training_seed retained",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(missing)
