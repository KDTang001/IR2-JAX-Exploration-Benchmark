"""Primary coverage/budget figures keep split, N and external fault conditions separate."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from ir2_benchmark.config import FAULT_BURST
from ir2_benchmark.evaluation.budgets import BudgetMode
from ir2_benchmark.evaluation.scope import budget_axis_value, row_auxiliary_kind
from ir2_benchmark.evaluation.statistics import family_balanced_mean

if TYPE_CHECKING:
    from ir2_benchmark.evaluation.statistics import ReportRow

Facet: TypeAlias = tuple[str, int, int, float, int, BudgetMode, str]


def curve_facet(row: ReportRow) -> Facet:
    """Normalize irrelevant fault parameters while preserving every distinct physical condition."""
    return (
        row.split,
        row.n_agents,
        row.fault_mode,
        row.drop_probability if row.fault_mode == 1 else 0,
        row.burst_duration if row.fault_mode == FAULT_BURST else 0,
        row.budget_mode,
        row_auxiliary_kind(row),
    )


def _curve(rows: tuple[ReportRow, ...]) -> np.ndarray | None:
    """Apply the same map/family/training-seed weights to actual round-end trajectories."""
    traces: dict[tuple[int, str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if row.trace_file and Path(row.trace_file).is_file():
            with np.load(row.trace_file) as trace:
                curve = trace["coverage"][row.n_agents - 1 :: row.n_agents]
            traces[row.training_seed, f"{row.source}:{row.family}", row.map_hash].append(curve)
    if not traces:
        return None
    horizon = min(len(curve) for curves in traces.values() for curve in curves)
    families: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    for (seed, family, _), curves in traces.items():
        families[seed, family].append(np.mean([curve[:horizon] for curve in curves], axis=0))
    seeds: dict[int, list[np.ndarray]] = defaultdict(list)
    for (seed, _), curves in families.items():
        seeds[seed].append(np.mean(curves, axis=0))
    return np.mean([np.mean(curves, axis=0) for curves in seeds.values()], axis=0)


def plot_facets(rows: tuple[ReportRow, ...], output: Path) -> None:
    """Export each condition separately; coverage-over-time also keeps byte budgets separate."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    groups: dict[Facet, list[ReportRow]] = defaultdict(list)
    for row in rows:
        groups[curve_facet(row)].append(row)
    for facet, entries in sorted(groups.items()):
        split, agents, mode, probability, duration, budget_mode, auxiliary_kind = facet
        label = (
            f"{split}_N{agents}_fault{mode}_p{probability:g}_duration{duration}_"
            f"{budget_mode}_{auxiliary_kind or 'core'}"
        ).replace(".", "p")
        budgets = sorted({budget_axis_value(row) for row in entries})
        methods = sorted({row.method_id for row in entries})
        figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for method in methods:
            values = [
                family_balanced_mean(
                    tuple(
                        row
                        for row in entries
                        if row.method_id == method and budget_axis_value(row) == budget
                    )
                )
                for budget in budgets
            ]
            axis.plot(budgets, values, marker="o", label=method)
        axis.set(
            xlabel="Requested team bytes"
            if budget_mode == "team_total"
            else "Initial bytes per robot",
            ylabel="Coverage AUC",
            ylim=(0, 1),
            title=label,
        )
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
        figure.savefig(output / f"budget_{label}.png", dpi=180)
        plt.close(figure)
        for budget in budgets:
            figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
            for method in methods:
                values = _curve(
                    tuple(
                        row
                        for row in entries
                        if row.method_id == method and budget_axis_value(row) == budget
                    )
                )
                if values is not None:
                    axis.plot(np.arange(1, len(values) + 1), values, label=method)
            axis.set(
                xlabel="Team round",
                ylabel="Physical coverage",
                ylim=(0, 1),
                title=f"{label}: {budget} bytes",
            )
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize=7)
            figure.savefig(output / f"coverage_{label}_budget{budget}.png", dpi=180)
            plt.close(figure)
