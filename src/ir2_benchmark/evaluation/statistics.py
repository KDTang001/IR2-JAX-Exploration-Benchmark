"""Equal-map/equal-family estimates and training-seed/map hierarchical bootstrap."""

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from ir2_benchmark.evaluation.budgets import BudgetMode


@dataclass(frozen=True, slots=True)
class ReportRow:
    """One actual episode; missing scores stay absent for execution errors."""

    result_id: str
    method_id: str
    training_seed: int
    source: str
    family: str
    map_hash: str
    outcome: str = "timeout"
    auc: float | None = None
    final_coverage: float | None = None
    success_round: int | None = None
    n_agents: int = 2
    budget: int = 8192
    fault_mode: int = 0
    drop_probability: float = 0
    burst_duration: int = 64
    emitted_bytes: int = 0
    delivered_payload_bytes: int = 0
    cold_start_events: int = 0
    recovery_coverage_gain: float | None = None
    limited: bool = False
    trace_file: str = ""
    error: str | None = None
    route: str = ""
    plan_diagnostics: bool = False
    split: str = "unknown"
    map_scenario_hash: str = ""
    decision_rule: str = "unspecified"
    real_training_events: int = 0
    selected_at_events: int | None = None
    active_events: int = 0
    budget_mode: BudgetMode = "per_robot"
    requested_team_bytes: int | None = None
    unallocated_bytes: int = 0
    auxiliary_kind: str = ""
    spent_bytes: int | None = None
    overlap: float | None = None
    allocation_concentration: float | None = None


@dataclass(frozen=True, slots=True)
class MethodSummary:
    """Mean and interval use training repetitions, not episode pseudo-replication."""

    method_id: str
    auc: float | None
    ci_low: float | None
    ci_high: float | None
    success_rate: float | None
    training_seeds: tuple[int, ...]
    missing_training_seeds: tuple[int, ...]
    episodes: int
    errors: int


def family_balanced_mean(rows: tuple[ReportRow, ...]) -> float | None:
    """Average scenarios within maps, maps within families, families within training seeds."""
    maps: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.auc is not None:
            maps[row.training_seed, f"{row.source}:{row.family}", row.map_hash].append(row.auc)
    families: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (seed, family, _), scores in maps.items():
        families[seed, family].append(float(np.mean(scores)))
    seeds: dict[int, list[float]] = defaultdict(list)
    for (seed, _), scores in families.items():
        seeds[seed].append(float(np.mean(scores)))
    return float(np.mean([np.mean(scores) for scores in seeds.values()])) if seeds else None


def hierarchical_interval(
    rows: tuple[ReportRow, ...], *, repetitions: int = 1000, seed: int = 0
) -> tuple[float | None, float | None]:
    """Resample seeds/maps; a single fixed-policy sentinel seed reduces to map bootstrap."""
    maps: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.auc is not None:
            maps[row.training_seed, f"{row.source}:{row.family}", row.map_hash].append(row.auc)
    groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (training_seed, family, _), scores in maps.items():
        groups[training_seed][family].append(float(np.mean(scores)))
    if not groups:
        return None, None
    rng = np.random.default_rng(seed)
    training_seeds = sorted(groups)
    samples = []
    for _ in range(repetitions):
        seed_scores = []
        for selected in rng.choice(training_seeds, len(training_seeds), replace=True):
            family_scores = [
                float(np.mean(rng.choice(scores, len(scores), replace=True)))
                for scores in groups[int(selected)].values()
            ]
            seed_scores.append(float(np.mean(family_scores)))
        samples.append(float(np.mean(seed_scores)))
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def summarize_method(
    method: str,
    rows: tuple[ReportRow, ...],
    expected_seeds: tuple[int, ...],
    *,
    repetitions: int = 1000,
) -> MethodSummary:
    """Keep failures explicit and require independent maps for fixed-policy uncertainty."""
    selected = tuple(row for row in rows if row.method_id == method)
    seeds = tuple(sorted({row.training_seed for row in selected}))
    fixed_policy_maps: dict[tuple[str, str], set[str]] = defaultdict(set)
    if all(row.route == "baseline" for row in selected):
        for row in selected:
            if row.auc is not None:
                fixed_policy_maps[row.source, row.family].add(row.map_hash)
    low, high = (
        (None, None)
        if any(len(maps) <= 1 for maps in fixed_policy_maps.values())
        else hierarchical_interval(selected, repetitions=repetitions)
    )
    errors = sum(row.outcome not in ("success", "timeout") for row in selected)
    success_rows = tuple(
        replace(row, auc=float(row.outcome == "success"))
        for row in selected
        if row.outcome in ("success", "timeout")
    )
    return MethodSummary(
        method,
        family_balanced_mean(selected),
        low,
        high,
        family_balanced_mean(success_rows),
        seeds,
        tuple(seed for seed in expected_seeds if seed not in seeds),
        len(selected),
        errors,
    )
