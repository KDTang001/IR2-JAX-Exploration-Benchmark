"""Actual cost/cold-start tables and independently stratified condition statistics."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from ir2_benchmark.evaluation.figures import Facet, curve_facet
from ir2_benchmark.evaluation.scope import budget_axis_value
from ir2_benchmark.evaluation.statistics import MethodSummary, ReportRow, summarize_method


def write_cost_tables(rows: tuple[ReportRow, ...], output: Path) -> None:
    """Report actual byte utilization and time spent without received information."""
    identity_fields = (
        "result_id",
        "method_id",
        "training_seed",
        "split",
        "n_agents",
        "budget",
        "budget_mode",
        "requested_team_bytes",
        "unallocated_bytes",
        "auxiliary_kind",
        "fault_mode",
        "drop_probability",
        "burst_duration",
        "outcome",
    )
    cost_fields = (
        *identity_fields,
        "initial_team_bytes",
        "emitted_bytes",
        "delivered_payload_bytes",
        "budget_utilization",
    )
    cold_fields = (*identity_fields, "cold_start_events", "active_events", "cold_start_fraction")
    with (
        (output / "costs.csv").open("w", newline="", encoding="utf-8") as cost_stream,
        (output / "cold_start.csv").open("w", newline="", encoding="utf-8") as cold_stream,
    ):
        costs = csv.DictWriter(cost_stream, fieldnames=cost_fields)
        cold = csv.DictWriter(cold_stream, fieldnames=cold_fields)
        costs.writeheader()
        cold.writeheader()
        for row in rows:
            raw = asdict(row)
            identifiers = {field: raw[field] for field in identity_fields}
            budget = row.budget * row.n_agents
            costs.writerow(
                identifiers
                | {
                    "initial_team_bytes": budget,
                    "emitted_bytes": row.emitted_bytes,
                    "delivered_payload_bytes": row.delivered_payload_bytes,
                    "budget_utilization": row.emitted_bytes / budget if budget else None,
                }
            )
            cold.writerow(
                identifiers
                | {
                    "cold_start_events": row.cold_start_events,
                    "active_events": row.active_events,
                    "cold_start_fraction": row.cold_start_events / row.active_events
                    if row.active_events
                    else None,
                }
            )


def write_condition_statistics(
    rows: tuple[ReportRow, ...], output: Path, expected_seeds: tuple[int, ...], repetitions: int
) -> None:
    """Keep ID/OOD, N, condition and budget separate for primary scores and intervals."""
    grouped: dict[tuple[Facet, int, str], list[ReportRow]] = defaultdict(list)
    for row in rows:
        grouped[curve_facet(row), budget_axis_value(row), row.method_id].append(row)
    fields = (
        "split",
        "n_agents",
        "fault_mode",
        "drop_probability",
        "burst_duration",
        "budget",
        "budget_mode",
        "auxiliary_kind",
        "budget_axis_bytes",
        "requested_team_bytes",
        "unallocated_bytes",
        *MethodSummary.__dataclass_fields__,
    )
    with (output / "condition_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (facet, budget, method), entries in sorted(grouped.items()):
            split, agents, mode, probability, duration, budget_mode, auxiliary_kind = facet
            seeds = (-1,) if all(row.route == "heuristic" for row in entries) else expected_seeds
            summary = summarize_method(method, tuple(entries), seeds, repetitions=repetitions)
            writer.writerow(
                {
                    "split": split,
                    "n_agents": agents,
                    "fault_mode": mode,
                    "drop_probability": probability,
                    "burst_duration": duration,
                    "budget": entries[0].budget,
                    "budget_mode": budget_mode,
                    "auxiliary_kind": auxiliary_kind,
                    "budget_axis_bytes": budget,
                    "requested_team_bytes": entries[0].requested_team_bytes,
                    "unallocated_bytes": entries[0].unallocated_bytes,
                    **asdict(summary),
                }
            )
