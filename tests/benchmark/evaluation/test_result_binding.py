"""Checksummed records cannot substitute task identity or invent completed metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import pytest

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.metrics import EpisodeMetrics, EpisodeOutcome, summarize_episode
from ir2_benchmark.evaluation.records import (
    completed_result,
    file_hash,
    save_result,
)
from ir2_benchmark.evaluation.runner import _error_result
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    identity,
    make_scenario,
    tasks,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ir2_benchmark.evaluation.records import EvaluationResult
    from ir2_benchmark.evaluation.scenarios import EvaluationTask, JsonValue


@dataclass(frozen=True, slots=True)
class ResultCase:
    """An actual saved result and trace for a task with nondefault provenance."""

    path: Path
    task: EvaluationTask
    result: EvaluationResult


@pytest.fixture
def result_case(tmp_path: Path) -> ResultCase:
    scenario = make_scenario(
        EvaluationMap("map.png", "map", "test", "ir2", "rooms", 64, "anchor"),
        HostConfig(
            n_agents=3,
            initial_budget=2730,
            max_rounds=4,
            fault_mode=1,
            dropout_probability=0.4,
            burst_duration=7,
        ),
        0,
        limited=True,
        budget_mode="team_total",
        requested_team_bytes=8192,
        unallocated_bytes=2,
    )
    method = Method("A", "A", "learned", "predict", 5, "", "weights", 2000000, "greedy", 1500000)
    task = next(tasks((scenario,), (method,)))
    trace = tmp_path / "trace.npz"
    trace.write_bytes(b"intact-real-trace-content")
    result = replace(
        _error_result(task, RuntimeError("fixture execution error")),
        outcome="timeout",
        metrics=summarize_episode(
            [0.1, 0.2, 0.3, 0.4], horizon=4, n_agents=3, outcome=EpisodeOutcome.TIMEOUT
        ),
        active_events=12,
        trace_file=str(trace),
        trace_hash=file_hash(trace),
        error=None,
    )
    path = tmp_path / "result.json"
    save_result(path, result)
    return ResultCase(path, task, result)


def _rewrite(case: ResultCase, payload: Mapping[str, JsonValue]) -> None:
    raw = dict(payload)
    case.path.write_text(json.dumps({"result": raw, "checksum": identity(raw)}), encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result_id", "other-result"),
        ("scenario_id", "other-scenario"),
        ("method_id", "B"),
        ("route", "B"),
        ("training_seed", 6),
        ("source", "procedural"),
        ("family", "corridor"),
        ("map_hash", "other-map"),
        ("split", "ood"),
        ("n_agents", 2),
        ("budget", 8192),
        ("fault_mode", 2),
        ("drop_probability", 0.2),
        ("burst_duration", 8),
        ("limited", False),
        ("map_scenario_hash", "other-anchor"),
        ("decision_rule", "unspecified"),
        ("real_training_events", 0),
        ("selected_at_events", None),
        ("budget_mode", "per_robot"),
        ("requested_team_bytes", None),
        ("unallocated_bytes", 0),
        ("auxiliary_kind", "heuristic"),
        ("checkpoint_hash", "other-weights"),
        ("configuration_hash", "other-config"),
    ],
)
def test_rejects_contradictory_identity_when_checksum_and_trace_are_intact(
    result_case: ResultCase,
    field: str,
    value: JsonValue,
) -> None:
    # Given: a valid writer-format result whose redundant identity was replaced.
    payload = asdict(result_case.result)
    payload[field] = value
    _rewrite(result_case, payload)
    # When / Then: a fresh checksum cannot make that record satisfy the requested task.
    assert not completed_result(result_case.path, result_case.task)


@pytest.mark.parametrize("field", ["method_id", "source", "limited", "map_scenario_hash"])
def test_rejects_missing_identity_when_task_has_no_matching_historical_default(
    result_case: ResultCase,
    field: str,
) -> None:
    # Given: required identity, or appended nondefault provenance, is absent.
    payload = asdict(result_case.result)
    payload.pop(field)
    _rewrite(result_case, payload)
    # When / Then: the missing field is never inferred from the requested task.
    assert not completed_result(result_case.path, result_case.task)


@pytest.mark.parametrize("metrics", [None, {}, {"auc": 0.25}])
def test_rejects_incomplete_metrics_when_normal_outcome_is_claimed(
    result_case: ResultCase,
    metrics: JsonValue,
) -> None:
    # Given: a normal result has no complete EpisodeMetrics despite an intact trace.
    payload = asdict(result_case.result)
    payload["metrics"] = metrics
    _rewrite(result_case, payload)
    # When / Then: the record cannot count as completed work.
    assert not completed_result(result_case.path, result_case.task)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("auc", None),
        ("auc", float("nan")),
        ("auc", float("inf")),
        ("auc", -0.1),
        ("auc", 1.1),
        ("auc", True),
        ("final_coverage", float("nan")),
        ("final_coverage", float("inf")),
        ("final_coverage", -0.1),
        ("final_coverage", 1.1),
        ("final_coverage", 0.1),
        ("final_coverage", 0.99),
        ("outcome", "success"),
        ("success", True),
        ("timeout", False),
        ("success_round", 1),
        ("success_event_offset", 0),
        ("observed_rounds", 0),
        ("observed_rounds", 3),
        ("observed_rounds", 5),
        ("observed_rounds", 4.0),
    ],
)
def test_rejects_invalid_metrics_when_normal_outcome_is_claimed(
    result_case: ResultCase,
    field: str,
    value: JsonValue,
) -> None:
    # Given: the writer's timeout metrics have one corrupt score, flag or time.
    payload = asdict(result_case.result)
    payload["metrics"][field] = value
    _rewrite(result_case, payload)
    # When / Then: recomputing the checksum does not certify an invalid episode.
    assert not completed_result(result_case.path, result_case.task)


@pytest.mark.parametrize(
    ("success_round", "offset", "coverage"),
    [
        (None, 1, 1.0),
        (0, 1, 1.0),
        (3, 1, 1.0),
        (2, 3, 1.0),
        (2, 1, 0.98),
    ],
)
def test_rejects_inconsistent_success_times_or_threshold(
    result_case: ResultCase,
    success_round: int | None,
    offset: int,
    coverage: float,
) -> None:
    # Given: a purported success contradicts its observed rounds, agent count or target.
    metrics = EpisodeMetrics(
        outcome=EpisodeOutcome.SUCCESS,
        success=True,
        timeout=False,
        auc=0.8,
        final_coverage=coverage,
        success_round=success_round,
        success_event_offset=offset,
        observed_rounds=2,
    )
    payload = asdict(replace(result_case.result, outcome="success", metrics=metrics))
    _rewrite(result_case, payload)
    # When / Then: a real trace hash cannot make inconsistent success metrics complete.
    assert not completed_result(result_case.path, result_case.task)


@pytest.mark.parametrize("outcome", list(EpisodeOutcome))
def test_accepts_valid_normal_and_explicit_error_results(
    result_case: ResultCase,
    outcome: EpisodeOutcome,
) -> None:
    # Given: each actual writer outcome has complete metrics or an explicit execution error.
    metrics = None
    error = "execution failed"
    if outcome in (EpisodeOutcome.SUCCESS, EpisodeOutcome.TIMEOUT):
        metrics = summarize_episode(
            [0.2, 1.0] if outcome == EpisodeOutcome.SUCCESS else [0.1] * 4,
            horizon=4,
            n_agents=3,
            outcome=outcome,
        )
        error = None
    save_result(
        result_case.path,
        replace(result_case.result, outcome=str(outcome), metrics=metrics, error=error),
    )
    # When / Then: completed results include genuine known failures without invented scores.
    assert completed_result(result_case.path, result_case.task)


def test_accepts_success_already_reached_at_reset(result_case: ResultCase) -> None:
    # Given: the writer's zero-event success includes its actual initial coverage.
    result = replace(
        result_case.result,
        outcome="success",
        active_events=0,
        metrics=EpisodeMetrics(
            outcome=EpisodeOutcome.SUCCESS,
            success=True,
            timeout=False,
            auc=1,
            final_coverage=1,
            success_round=0,
            success_event_offset=None,
            observed_rounds=0,
        ),
    )
    save_result(result_case.path, result)
    # When / Then: success before the first action remains a valid completed episode.
    assert completed_result(result_case.path, result_case.task)


def test_rejects_unknown_outcome_with_absent_metrics(result_case: ResultCase) -> None:
    # Given: a forged error category has the same shape as an execution failure.
    save_result(
        result_case.path,
        replace(result_case.result, outcome="unknown_error", metrics=None, error="failed"),
    )
    # When / Then: only the explicit execution error classes can complete without metrics.
    assert not completed_result(result_case.path, result_case.task)


def test_accepts_missing_appended_fields_only_when_task_uses_their_defaults(
    result_case: ResultCase,
) -> None:
    # Given: a historical record omits appended provenance that is default in its task.
    scenario = replace(
        result_case.task.scenario,
        map_scenario_hash="",
        budget_mode="per_robot",
        requested_team_bytes=None,
        unallocated_bytes=0,
    )
    method = replace(
        result_case.task.method,
        decision_rule="unspecified",
        real_training_events=0,
        selected_at_events=None,
    )
    task = replace(result_case.task, scenario=scenario, method=method)
    payload = asdict(result_case.result)
    for field in (
        "map_scenario_hash",
        "decision_rule",
        "real_training_events",
        "selected_at_events",
        "budget_mode",
        "requested_team_bytes",
        "unallocated_bytes",
        "auxiliary_kind",
    ):
        payload.pop(field)
    _rewrite(result_case, payload)
    # When / Then: historical defaults preserve the checksum without inventing task metadata.
    assert completed_result(result_case.path, task)


def test_accepts_writer_normalization_of_heuristic_decision_rule(result_case: ResultCase) -> None:
    # Given: heuristic result writers normalize even a manifest's unnormalized rule.
    method = replace(result_case.task.method, route="heuristic")
    task = replace(result_case.task, method=method)
    result = replace(
        result_case.result, route="heuristic", decision_rule="heuristic", auxiliary_kind="heuristic"
    )
    save_result(result_case.path, result)
    # When / Then: binding agrees with the actual writer's heuristic representation.
    assert completed_result(result_case.path, task)
