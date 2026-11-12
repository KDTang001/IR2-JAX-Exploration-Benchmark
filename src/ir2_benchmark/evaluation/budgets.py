"""Evaluation-only budget allocation and explicit auxiliary-method classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict, assert_never

from ir2_benchmark.config import ConfigError

if TYPE_CHECKING:
    from ir2_benchmark.evaluation.scenarios import Method, Scenario

BudgetMode = Literal["per_robot", "team_total", "unlimited"]
MAX_PACKET_BYTES = 256


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    """Equal starting balances retain the requested total and any unallocated remainder."""

    per_robot: int
    requested_team_bytes: int | None = None
    unallocated_bytes: int = 0


def allocate_budget(mode: BudgetMode, amount: int, agents: int, rounds: int) -> BudgetAllocation:
    """Bound unlimited sends by one at-most-256-byte packet per own action, for T actions."""
    if amount < 0:
        raise ConfigError(field="budget", detail="must be nonnegative")
    match mode:
        case "per_robot":
            return BudgetAllocation(amount)
        case "team_total":
            per_robot, remainder = divmod(amount, agents)
            return BudgetAllocation(per_robot, amount, remainder)
        case "unlimited":
            return BudgetAllocation(rounds * MAX_PACKET_BYTES)
        case unreachable:
            assert_never(unreachable)


def auxiliary_kind(method: Method) -> str:
    """Reuse actual policy metadata; zero is a view of the core budget endpoint."""
    if method.route == "heuristic":
        return "heuristic"
    if method.estimator == "none":
        return "no_neighbor_plan"
    if method.method_id.startswith("B_real_updates_"):
        return "B_real_updates"
    return ""


class BudgetRecordFields(TypedDict):
    """The same budget provenance is retained for successful and failed tasks."""

    budget_mode: BudgetMode
    requested_team_bytes: int | None
    unallocated_bytes: int
    auxiliary_kind: str


def result_budget_fields(scenario: Scenario, method: Method) -> BudgetRecordFields:
    """Carry scenario allocation and actual method identity into result records."""
    return BudgetRecordFields(
        budget_mode=scenario.budget_mode,
        requested_team_bytes=scenario.requested_team_bytes,
        unallocated_bytes=scenario.unallocated_bytes,
        auxiliary_kind=auxiliary_kind(method),
    )
