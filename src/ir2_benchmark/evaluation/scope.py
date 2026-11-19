"""Core and declared auxiliary scopes remain distinct in completeness and statistics."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, NamedTuple

from ir2_benchmark.config import FAULT_BURST

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ir2_benchmark.evaluation.budgets import BudgetMode
    from ir2_benchmark.evaluation.scenarios import EvaluationTask
    from ir2_benchmark.evaluation.statistics import ReportRow


class Condition(NamedTuple):
    """One declared method repetition and physical condition, independent of maps or seeds."""

    method: str
    training_seed: int
    agents: int
    budget_mode: BudgetMode
    budget: int | None
    fault: int
    probability: float
    duration: int


def declared_method_fields(declaration: dict) -> dict[str, tuple[tuple[str, str], ...]]:
    """Keep only explicitly declared policy fields alongside the existing condition axes."""
    return {
        entry["method_id"]: tuple(
            (field, entry[field])
            for field in ("route", "communication", "estimator", "decision_rule")
            if field in entry
        )
        for entry in declaration.get("methods", [])
    }


def normalized_fault(mode: int, probability: float, duration: int) -> tuple[int, float, int]:
    """Keep irrelevant Normal/Dropout burst durations out of condition identity."""
    return mode, probability if mode == 1 else 0, duration if mode == FAULT_BURST else 64


def condition_for_row(row: ReportRow) -> Condition:
    """Use requested team totals and an explicit unlimited marker for report condition keys."""
    return Condition(
        row.method_id,
        row.training_seed,
        row.n_agents,
        row.budget_mode,
        None if row.budget_mode == "unlimited" else budget_axis_value(row),
        *normalized_fault(row.fault_mode, row.drop_probability, row.burst_duration),
    )


def condition_for_task(task: EvaluationTask) -> Condition:
    """Read actual task metadata without constructing an expected task matrix."""
    scene = task.scenario
    amount = scene.requested_team_bytes if scene.budget_mode == "team_total" else scene.budget
    return Condition(
        task.method.method_id,
        task.method.training_seed,
        scene.n_agents,
        scene.budget_mode,
        None if scene.budget_mode == "unlimited" else amount,
        *normalized_fault(scene.fault_mode, scene.drop_probability, scene.burst_duration),
    )


def core_conditions(declaration: dict) -> Iterator[Condition]:
    """Enumerate only the compact condition axes, never map-by-condition task objects."""
    axes = (
        declaration.get("methods", []),
        declaration.get("training_seeds", []),
        declaration.get("n_agents", []),
        declaration.get("budgets", []),
        declaration.get("conditions", []),
    )
    for method, seed, agents, budget, fault in product(*axes):
        yield Condition(
            method["method_id"], seed, agents, "per_robot", budget, *normalized_fault(*fault)
        )


def auxiliary_conditions(declaration: dict) -> Iterator[tuple[str, Condition]]:
    """Keep the already-declared diagnostic blocks distinct from the core Cartesian axes."""
    for block in declaration.get("auxiliary_matrix", []):
        mode = block.get("budget_mode", "per_robot")
        axes = (
            block["method_ids"],
            block.get("training_seeds", declaration.get("training_seeds", [])),
            block.get("n_agents", declaration.get("n_agents", [])),
            [None] if mode == "unlimited" else block["budgets"],
            block["conditions"],
        )
        for method, seed, agents, budget, fault in product(*axes):
            yield (
                block["category"],
                Condition(method, seed, agents, mode, budget, *normalized_fault(*fault)),
            )


def missing_conditions(rows: tuple[ReportRow, ...], declaration: dict) -> tuple[str, ...]:
    """Retain the legacy missing-condition strings, using the shared declared axes."""
    present = {condition_for_row(row) for row in rows}
    return tuple(
        ":".join(
            str(value)
            for value in (
                key.method,
                key.training_seed,
                key.agents,
                key.budget,
                key.fault,
                key.probability,
                key.duration,
            )
        )
        for key in core_conditions(declaration)
        if key not in present
    )


def missing_auxiliary_conditions(rows: tuple[ReportRow, ...], declaration: dict) -> tuple[str, ...]:
    """Report undefined scopes and missing declared diagnostic conditions separately."""
    defined = {block["category"] for block in declaration.get("auxiliary_matrix", [])}
    present = {condition_for_row(row) for row in rows}
    undeclared = tuple(
        f"undeclared_scope:{kind}"
        for kind in declaration.get("auxiliary", [])
        if kind not in defined
    )
    return undeclared + tuple(
        ":".join(str(value) for value in (category, *key))
        for category, key in auxiliary_conditions(declaration)
        if key not in present
    )


def row_auxiliary_kind(row: ReportRow) -> str:
    """Recognize legacy method names too, after the original result checksum was verified."""
    if row.route == "baseline":
        return ""
    if row.auxiliary_kind:
        return row.auxiliary_kind
    if row.route == "heuristic":
        return "heuristic"
    if row.method_id.endswith("_none"):
        return "no_neighbor_plan"
    if row.method_id.startswith("B_real_updates_"):
        return "B_real_updates"
    return ""


def is_auxiliary(row: ReportRow) -> bool:
    """Budget diagnostics and auxiliary algorithms never enter core rankings."""
    return row.budget_mode != "per_robot" or bool(row_auxiliary_kind(row))


def budget_axis_value(row: ReportRow) -> int:
    """Team curves use requested totals, retaining equal balances and remainder in cost tables."""
    if row.budget_mode == "team_total":
        if row.requested_team_bytes is None:
            from ir2_benchmark.config import ConfigError  # noqa: PLC0415

            raise ConfigError(field="requested_team_bytes", detail="required for team_total result")
        return row.requested_team_bytes
    return row.budget


def invalid_auxiliary_quotas(rows: tuple[ReportRow, ...], declaration: dict) -> tuple[str, ...]:
    """Only learned auxiliary methods need the same actual real-event training quota."""
    expected = declaration.get("real_training_events", 2000000)
    return tuple(
        sorted(
            {
                f"{row.method_id}:{row.training_seed}:{row.real_training_events}"
                for row in rows
                if is_auxiliary(row)
                and row.route not in ("heuristic", "baseline")
                and row.real_training_events != expected
            }
        )
    )


def has_missing_diagnostics(
    rows: tuple[ReportRow, ...], plan: tuple[str, ...], recovery: tuple[str, ...]
) -> bool:
    """Check missing measurements independently for core and auxiliary rows."""
    missing = set(plan) | set(recovery)
    return any(row.result_id in missing for row in rows)


def write_auxiliary_views(
    rows: tuple[ReportRow, ...], output: Path, expected_seeds: tuple[int, ...], repetitions: int
) -> None:
    """Keep all diagnostics out of core rankings; zero reuses measured core endpoint rows."""
    from ir2_benchmark.evaluation.figures import plot_facets  # noqa: PLC0415
    from ir2_benchmark.evaluation.tables import write_condition_statistics  # noqa: PLC0415

    for name, selected in (
        ("auxiliary", tuple(row for row in rows if is_auxiliary(row))),
        (
            "zero_communication",
            tuple(row for row in rows if row.budget_mode == "per_robot" and row.budget == 0),
        ),
    ):
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        write_condition_statistics(selected, destination, expected_seeds, repetitions)
        plot_facets(selected, destination)


def invalid_training_quotas(rows: tuple[ReportRow, ...], declaration: dict) -> tuple[str, ...]:
    """Require the actual full-run core quota; selected weights may come from an earlier event."""
    expected = declaration.get("real_training_events", 2000000)
    core = {method["method_id"] for method in declaration.get("methods", [])}
    return tuple(
        sorted(
            {
                f"{row.method_id}:{row.training_seed}:{row.real_training_events}"
                for row in rows
                if row.method_id in core and row.real_training_events != expected
            }
        )
    )
