"""Dispatch the exact manifest, resume verified tasks, and preserve execution error classes."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, TypeAlias, assert_never

from ir2_benchmark.evaluation.batching import BoundedTaskQueue, ParameterCache, prepare_task
from ir2_benchmark.evaluation.budgets import result_budget_fields
from ir2_benchmark.evaluation.episode import execute_batch, load_lane, summarize_execution
from ir2_benchmark.evaluation.errors import ModelExecutionError, is_oom
from ir2_benchmark.evaluation.metrics import EpisodeOutcome
from ir2_benchmark.evaluation.records import (
    EvaluationResult,
    RunnerSummary,
    completed_result,
    save_result,
)
from ir2_benchmark.evaluation.scenarios import read_task_manifest
from ir2_benchmark.maps.prepare import MapError

if TYPE_CHECKING:
    from pathlib import Path

    from ir2_benchmark.evaluation.batching import PreparedTask
    from ir2_benchmark.evaluation.scenarios import EvaluationTask

ExecutionError: TypeAlias = (
    MapError
    | ModelExecutionError
    | MemoryError
    | RuntimeError
    | OSError
    | ValueError
    | KeyError
    | TypeError
)
_EXECUTION_ERRORS = (
    MapError,
    ModelExecutionError,
    MemoryError,
    RuntimeError,
    OSError,
    ValueError,
    KeyError,
    TypeError,
)


def _failed(task: EvaluationTask, outcome: EpisodeOutcome, error: str) -> EvaluationResult:
    """Retain failures without manufacturing coverage, success times or throughput."""
    scenario, method = task.scenario, task.method
    return EvaluationResult(
        task.result_id,
        scenario.scenario_id,
        method.method_id,
        method.route,
        method.training_seed,
        scenario.source,
        scenario.family,
        scenario.map_hash,
        scenario.split,
        scenario.n_agents,
        scenario.budget,
        scenario.fault_mode,
        scenario.drop_probability,
        scenario.burst_duration,
        scenario.limited,
        str(outcome),
        None,
        0,
        0,
        None,
        None,
        0,
        None,
        0,
        method.checkpoint_hash,
        scenario.config_hash,
        "",
        "",
        error,
        map_scenario_hash=scenario.map_scenario_hash,
        decision_rule="heuristic" if method.route == "heuristic" else method.decision_rule,
        real_training_events=method.real_training_events,
        selected_at_events=method.selected_at_events,
        **result_budget_fields(scenario, method),
    )


def _error_result(task: EvaluationTask, error: ExecutionError) -> EvaluationResult:
    """Preserve the existing lane error categories at preprocessing and compute boundaries."""
    match error:
        case MapError():
            outcome = EpisodeOutcome.MAP_ERROR
        case ModelExecutionError():
            outcome = EpisodeOutcome.MODEL_ERROR
        case MemoryError() | RuntimeError():
            outcome = EpisodeOutcome.OOM if is_oom(error) else EpisodeOutcome.ENVIRONMENT_ERROR
        case OSError() | ValueError() | KeyError() | TypeError():
            outcome = EpisodeOutcome.MODEL_ERROR
        case unreachable:
            assert_never(unreachable)
    return _failed(task, outcome, str(error))


def _run_group(
    group: tuple[PreparedTask, ...], parameters: ParameterCache, output: Path
) -> tuple[int, int]:
    """Preprocess lanes independently, then run exactly one actual batch without error retries."""
    results = []
    lanes = []
    for prepared in group:
        try:
            lanes.append(load_lane(prepared, parameters))
        except _EXECUTION_ERRORS as error:
            results.append(_error_result(prepared.task, error))
    if lanes:
        try:
            executions = execute_batch(tuple(lanes))
        except _EXECUTION_ERRORS as error:
            results.extend(_error_result(lane.prepared.task, error) for lane in lanes)
        else:
            for lane, execution in zip(lanes, executions, strict=True):
                task = lane.prepared.task
                try:
                    result = summarize_execution(
                        task, execution, output / "events" / f"{task.result_id}.npz"
                    )
                except _EXECUTION_ERRORS as error:
                    result = _error_result(task, error)
                results.append(result)
    for result in results:
        save_result(output / "results" / f"{result.result_id}.json", result)
    return len(results), sum(
        result.outcome not in (EpisodeOutcome.SUCCESS, EpisodeOutcome.TIMEOUT) for result in results
    )


def run_manifest(
    manifest: Path,
    output: Path,
    *,
    retry_errors: bool = False,
    batch_size: int = 1,
    max_pending_tasks: int = 128,
) -> RunnerSummary:
    """Stream compatible bounded batches, resuming the same verified per-task result IDs."""
    queue = BoundedTaskQueue(batch_size, max_pending_tasks)
    parameters = ParameterCache(batch_size)
    requested = executed = resumed = failures = 0
    for task in read_task_manifest(manifest):
        requested += 1
        result_path = output / "results" / f"{task.result_id}.json"
        if completed_result(result_path, task):
            outcome = json.loads(result_path.read_text(encoding="utf-8"))["result"]["outcome"]
            failed = outcome not in (EpisodeOutcome.SUCCESS, EpisodeOutcome.TIMEOUT)
            if not (retry_errors and failed):
                resumed += 1
                failures += failed
                continue
        try:
            prepared = prepare_task(task)
        except _EXECUTION_ERRORS as error:
            save_result(result_path, _error_result(task, error))
            executed += 1
            failures += 1
            continue
        for group in queue.append(prepared):
            count, failed_count = _run_group(group, parameters, output)
            executed += count
            failures += failed_count
    for group in queue.finish():
        count, failed_count = _run_group(group, parameters, output)
        executed += count
        failures += failed_count
    summary = RunnerSummary(requested, executed, resumed, failures)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runner_summary.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary
