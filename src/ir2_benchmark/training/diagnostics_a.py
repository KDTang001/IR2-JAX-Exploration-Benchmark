"""Host summaries of exact actor-time intent diagnostics shared with the evaluator."""

from typing import TypedDict

import jax
import numpy as np
from numpy.typing import NDArray

from ir2_benchmark.training.collect_a import Collection


class PlanDiagnosticsSummary(TypedDict):
    """Cold-start and undefined path cases remain visible beside conditional errors."""

    plan_target_pairs: int
    plan_cold_start_pairs: int
    plan_scored_pairs: int
    plan_ade: float | None
    plan_fde: float | None
    plan_goal_error: float | None
    plan_nll: float | None
    plan_coverage_one_sigma: float | None
    message_age_mean: float | None


def _finite_mean(values: NDArray[np.float32]) -> float | None:
    """Preserve missing densities/paths instead of inventing zero error."""
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else None


def plan_diagnostics(collection: Collection) -> PlanDiagnosticsSummary:
    """Aggregate the exact saved actor estimates, without recomputing model predictions."""
    diagnostics = jax.tree.map(lambda value: np.asarray(value[0]), collection.plan_diagnostics)
    targets = diagnostics.has_target
    return {
        "plan_target_pairs": int(targets.sum()),
        "plan_cold_start_pairs": int((targets & diagnostics.cold_start).sum()),
        "plan_scored_pairs": int((targets & diagnostics.has_prediction).sum()),
        "plan_ade": _finite_mean(diagnostics.ade),
        "plan_fde": _finite_mean(diagnostics.fde),
        "plan_goal_error": _finite_mean(diagnostics.goal_error),
        "plan_nll": _finite_mean(diagnostics.nll),
        "plan_coverage_one_sigma": _finite_mean(diagnostics.coverage_one_sigma),
        "message_age_mean": _finite_mean(
            np.where(targets & (diagnostics.message_age >= 0), diagnostics.message_age, np.nan)
        ),
    }
