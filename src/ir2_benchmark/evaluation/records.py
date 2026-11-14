"""Checksummed unique result records and resumable artifact verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from ir2_benchmark.evaluation.budgets import result_budget_fields
from ir2_benchmark.evaluation.metrics import EpisodeMetrics, EpisodeOutcome
from ir2_benchmark.evaluation.scenarios import identity

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ir2_benchmark.evaluation.budgets import BudgetMode
    from ir2_benchmark.evaluation.scenarios import EvaluationTask, JsonValue


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One unique requested task, including execution failures and actual cost."""

    result_id: str
    scenario_id: str
    method_id: str
    route: str
    training_seed: int
    source: str
    family: str
    map_hash: str
    split: str
    n_agents: int
    budget: int
    fault_mode: int
    drop_probability: float
    burst_duration: int
    limited: bool
    outcome: str
    metrics: EpisodeMetrics | None
    emitted_bytes: int
    delivered_payload_bytes: int
    overlap: float | None
    allocation_concentration: float | None
    cold_start_events: int
    recovery_coverage_gain: float | None
    active_events: int
    checkpoint_hash: str
    configuration_hash: str
    trace_file: str
    trace_hash: str
    error: str | None
    plan_diagnostics: bool = False
    map_scenario_hash: str = ""
    decision_rule: str = "unspecified"
    real_training_events: int = 0
    selected_at_events: int | None = None
    budget_mode: BudgetMode = "per_robot"
    requested_team_bytes: int | None = None
    unallocated_bytes: int = 0
    auxiliary_kind: str = ""
    spent_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class RunnerSummary:
    """Saved tasks are unique; execution errors are retained rather than normal timeouts."""

    requested: int
    executed: int
    resumed: int
    failures: int


def file_hash(path: Path) -> str:
    """Hash a real artifact without loading a large checkpoint/trace into another buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_result(path: Path, result: EvaluationResult) -> None:
    """Atomically replace one result file with a checksum over its complete payload."""
    payload = asdict(result)
    envelope = {"result": payload, "checksum": identity(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _valid_completion(payload: Mapping[str, JsonValue], task: EvaluationTask) -> bool:
    """Require the writer's complete outcome metrics, including success at reset."""
    try:
        outcome = EpisodeOutcome(payload.get("outcome"))
    except (ValueError, TypeError):
        return False
    match outcome:
        case (
            EpisodeOutcome.MODEL_ERROR
            | EpisodeOutcome.MAP_ERROR
            | EpisodeOutcome.OOM
            | EpisodeOutcome.ENVIRONMENT_ERROR
        ):
            return (
                "metrics" in payload
                and payload["metrics"] is None
                and type(payload.get("error")) is str
            )
        case EpisodeOutcome.SUCCESS | EpisodeOutcome.TIMEOUT:
            pass
        case unreachable:
            assert_never(unreachable)
    try:
        metrics = EpisodeMetrics(**payload["metrics"])
    except (KeyError, TypeError):
        return False
    success = outcome == EpisodeOutcome.SUCCESS
    # Fraction bounds also reject NaN and infinities; booleans are not numeric scores.
    valid_scores = all(
        type(score) in (int, float) and 0 <= score <= 1
        for score in (metrics.auc, metrics.final_coverage)
    )
    if (
        payload.get("error") is not None
        or metrics.outcome != outcome
        or metrics.success is not success
        or metrics.timeout is not (not success)
        or type(metrics.observed_rounds) is not int
        or not 0 <= metrics.observed_rounds <= task.scenario.max_rounds
        or not valid_scores
        or metrics.auc > metrics.final_coverage + 1e-12
    ):
        return False
    if success:
        return (
            metrics.final_coverage >= task.scenario.config.coverage_target
            and type(metrics.success_round) is int
            and (
                (
                    metrics.observed_rounds == metrics.success_round == 0
                    and metrics.success_event_offset is None
                    and metrics.auc == metrics.final_coverage
                )
                or (
                    1 <= metrics.success_round <= metrics.observed_rounds
                    and (
                        metrics.success_event_offset is None
                        or (
                            type(metrics.success_event_offset) is int
                            and 0 <= metrics.success_event_offset < task.scenario.n_agents
                        )
                    )
                )
            )
        )
    return (
        metrics.final_coverage < task.scenario.config.coverage_target
        and metrics.observed_rounds == task.scenario.max_rounds
        and metrics.success_round is None
        and metrics.success_event_offset is None
    )


def completed_result(path: Path, task: EvaluationTask) -> bool:
    """Resume only an intact matching result and its checksum-verified real trace."""
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    match row:
        case {"result": dict() as payload, "checksum": str() as checksum}:
            pass
        case _:  # Untrusted JSON may have any shape.
            return False
    scenario, method = task.scenario, task.method
    expected = (
        {
            name: getattr(scenario, name)
            for name in (
                "scenario_id",
                "source",
                "family",
                "map_hash",
                "split",
                "n_agents",
                "budget",
                "fault_mode",
                "drop_probability",
                "burst_duration",
                "limited",
                "map_scenario_hash",
            )
        }
        | {
            name: getattr(method, name)
            for name in (
                "method_id",
                "route",
                "training_seed",
                "checkpoint_hash",
                "real_training_events",
                "selected_at_events",
            )
        }
        | {
            "result_id": task.result_id,
            "configuration_hash": scenario.config_hash,
            "decision_rule": "heuristic" if method.route == "heuristic" else method.decision_rule,
            **result_budget_fields(scenario, method),
        }
    )
    # Only appended result fields have historical defaults; missing identity is never inferred.
    defaults = {
        field.name: field.default
        for field in fields(EvaluationResult)
        if field.default is not MISSING
    }
    if (
        checksum != identity(payload)
        or any(payload.get(name, defaults.get(name)) != value for name, value in expected.items())
        or not _valid_completion(payload, task)
    ):
        return False
    trace = payload.get("trace_file", "")
    if type(trace) is not str:
        return False
    if payload.get("outcome") in (EpisodeOutcome.SUCCESS, EpisodeOutcome.TIMEOUT) and not trace:
        return False
    return not trace or (
        Path(trace).is_file() and file_hash(Path(trace)) == payload.get("trace_hash")
    )
