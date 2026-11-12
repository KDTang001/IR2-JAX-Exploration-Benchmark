"""Execute actual paired episodes and summarize physical sensing, traffic and recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import numpy as np

from ir2_benchmark.config import FAULT_BURST, ConfigError
from ir2_benchmark.evaluation.batching import (
    AModel,
    BModel,
    ParameterCache,
    PreparedTask,
    get_heuristic_kernel,
    prepare_task,
)
from ir2_benchmark.evaluation.budgets import result_budget_fields
from ir2_benchmark.evaluation.errors import ModelExecutionError, is_oom
from ir2_benchmark.evaluation.metrics import (
    EpisodeMetrics,
    EpisodeOutcome,
    allocation_concentration,
    communication_totals,
    cross_agent_overlap,
    summarize_episode,
)
from ir2_benchmark.evaluation.records import EvaluationResult, file_hash
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash
from ir2_benchmark.maps.prepare import MapError, load_ir2_map, prepare_map, select_spawns

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array

    from ir2_benchmark.env.rollout import RolloutResult
    from ir2_benchmark.evaluation.batching import EvaluationKernel, ModelParameters
    from ir2_benchmark.evaluation.scenarios import EvaluationTask
    from ir2_benchmark.types import MapInput


@dataclass(frozen=True, slots=True)
class EpisodeExecution:
    """A real environment result and physical birth footprint for accounting."""

    rollout: RolloutResult
    initial_physical: np.ndarray
    component_area: int


@dataclass(frozen=True, slots=True)
class LoadedLane:
    """One working-window scenario with verified dynamic device inputs."""

    prepared: PreparedTask
    parameters: ModelParameters
    map_data: MapInput
    policy_key: Array
    spawns: Array
    fault_key: Array
    isolated_mask: Array
    component_area: int


def load_lane(prepared: PreparedTask, parameters: ParameterCache) -> LoadedLane:
    """Keep per-lane map/checkpoint failures separate before constructing a device batch."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from ir2_benchmark.env.reset import from_prepared_map  # noqa: PLC0415

    scenario, config = prepared.task.scenario, prepared.group.config
    data = load_ir2_map(scenario.map_path)
    if obstacle_hash(data.occupancy) != scenario.map_hash:
        raise MapError(scenario.map_path, "evaluation_manifest_occupancy_hash_mismatch")
    if scenario_hash(data.occupancy, data.anchor) != scenario.map_scenario_hash:
        raise MapError(scenario.map_path, "evaluation_manifest_anchor_hash_mismatch")
    padded = prepare_map(data, buckets=(scenario.bucket,))
    if padded.occupancy.shape != (scenario.bucket, scenario.bucket):
        raise MapError(scenario.map_path, "evaluation_manifest_bucket_mismatch")
    spawns = jnp.asarray(
        select_spawns(data, config.n_agents, scenario.spawn_seed, config.spawn_radius)
    )
    isolated = (
        jnp.zeros(config.n_slots, bool).at[jnp.asarray(scenario.isolation_ids, jnp.int32)].set(True)
    )
    return LoadedLane(
        prepared,
        parameters.load(prepared),
        from_prepared_map(padded),
        jax.random.PRNGKey(scenario.spawn_seed),
        spawns,
        jax.random.PRNGKey(scenario.fault_seed),
        isolated,
        int(padded.component.sum()),
    )


def evaluation_kernel(prepared: PreparedTask) -> EvaluationKernel:
    """Select a persistent route kernel; this function never creates a per-batch jit wrapper."""
    match prepared.model:
        case AModel(settings=settings):
            from ir2_benchmark.training.evaluate_a import get_evaluation_kernel  # noqa: PLC0415

            return get_evaluation_kernel(prepared.group.config, settings)
        case BModel(settings=settings):
            from ir2_benchmark.training.evaluate_b import get_evaluation_kernel  # noqa: PLC0415

            return get_evaluation_kernel(prepared.group.config, settings)
        case None:
            return get_heuristic_kernel(prepared.group.config)
        case unreachable:
            assert_never(unreachable)


def execute_batch(lanes: tuple[LoadedLane, ...]) -> tuple[EpisodeExecution, ...]:
    """Run one actual vmap batch; errors propagate for the entire batch without a serial retry."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if any(lane.prepared.group != lanes[0].prepared.group for lane in lanes):
        raise ConfigError(field="evaluation_batch", detail="mixed static inference groups")
    try:
        kernel = evaluation_kernel(lanes[0].prepared)
        values = jax.tree.map(
            lambda *parts: jnp.stack(parts),
            *(
                (
                    lane.parameters,
                    lane.map_data,
                    lane.policy_key,
                    lane.spawns,
                    lane.fault_key,
                    lane.isolated_mask,
                )
                for lane in lanes
            ),
        )
        rollouts, physical = kernel(*values)
        jax.block_until_ready((rollouts, physical))
    except RuntimeError as error:
        if is_oom(error) or lanes[0].prepared.model is None:
            raise
        detail = f"{type(error).__name__}:{error}"
        raise ModelExecutionError(detail) from error
    except (TypeError, ValueError, KeyError, OSError) as error:
        if lanes[0].prepared.model is None:
            raise
        detail = f"{type(error).__name__}:{error}"
        raise ModelExecutionError(detail) from error
    return tuple(
        EpisodeExecution(
            jax.tree.map(lambda value, index=index: value[index], rollouts),
            np.asarray(physical[index]),
            lane.component_area,
        )
        for index, lane in enumerate(lanes)
    )


def execute_episode(task: EvaluationTask) -> EpisodeExecution:
    """Preserve the single-task API through the same cached batch1 execution path."""
    return execute_batch((load_lane(prepare_task(task), ParameterCache(1)),))[0]


def summarize_execution(
    task: EvaluationTask, execution: EpisodeExecution, trace_path: Path
) -> EvaluationResult:
    """Compute shared metrics from actual physical footprints and paid event traces."""
    result, scenario, method = execution.rollout, task.scenario, task.method
    state, trace = result.final_state, result.trace
    from ir2_benchmark.types import validate_graph_state  # noqa: PLC0415

    validate_graph_state(state)
    mask = np.asarray(trace.active, bool)
    coverage = np.asarray(trace.coverage)[mask]
    rounds = list(coverage[scenario.n_agents - 1 :: scenario.n_agents])
    if len(coverage) % scenario.n_agents:
        rounds.append(coverage[-1])
    outcome = EpisodeOutcome.SUCCESS if bool(state.terminated) else EpisodeOutcome.TIMEOUT
    if len(coverage):
        metrics = summarize_episode(
            rounds,
            horizon=scenario.max_rounds,
            n_agents=scenario.n_agents,
            outcome=outcome,
            event_coverage=coverage,
            threshold=scenario.config.coverage_target,
        )
    else:
        initial_coverage = float(np.asarray(state.covered).sum() / execution.component_area)
        metrics = EpisodeMetrics(
            outcome,
            bool(state.terminated),
            bool(state.truncated),
            initial_coverage,
            initial_coverage,
            0,
            None,
            0,
        )
    spent = np.asarray(trace.spent)[mask]
    delivered = np.asarray(trace.delivered)[mask]
    totals = communication_totals(spent, np.where(spent > 0, spent - 16, 0), delivered)
    birth = execution.initial_physical & np.asarray(state.component)[None]
    counts = np.zeros(scenario.n_agents, np.int64)
    assigned = np.zeros_like(birth[0])
    for agent in range(scenario.n_agents):
        counts[agent] = np.count_nonzero(birth[agent] & ~assigned)
        assigned |= birth[agent]
    initial_coverage = assigned.sum() / execution.component_area
    changes = np.rint(np.diff(np.r_[initial_coverage, coverage]) * execution.component_area).astype(
        np.int64
    )
    seen = np.zeros(scenario.n_agents, bool)
    cold_start = 0
    for agent, gains, receivers in zip(
        np.asarray(trace.agent_id)[mask], changes, delivered, strict=True
    ):
        counts[agent] += gains
        cold_start += int(not seen[agent])
        seen |= receivers[: scenario.n_agents]
    recovery = None
    recovery_start = scenario.burst_start + scenario.burst_duration
    if scenario.fault_mode == FAULT_BURST and len(rounds) > recovery_start:
        recovery = float(
            rounds[min(len(rounds) - 1, recovery_start + 63)] - rounds[recovery_start - 1]
        )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in zip(trace._fields, trace, strict=True)}
    arrays["initial_coverage"] = np.asarray(initial_coverage)
    if result.plan_diagnostics is not None:
        diagnostics = result.plan_diagnostics
        arrays.update(
            {
                f"plan_{name}": np.asarray(value)
                for name, value in zip(diagnostics._fields, diagnostics, strict=True)
            }
        )
    np.savez_compressed(trace_path, **arrays)
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
        metrics,
        totals.emitted_bytes,
        totals.delivered_payload_bytes,
        cross_agent_overlap(
            np.asarray(state.physical_seen)[: scenario.n_agents], np.asarray(state.component)
        ),
        allocation_concentration(counts),
        cold_start,
        recovery,
        int(mask.sum()),
        method.checkpoint_hash,
        scenario.config_hash,
        str(trace_path),
        file_hash(trace_path),
        None,
        plan_diagnostics=result.plan_diagnostics is not None,
        map_scenario_hash=scenario.map_scenario_hash,
        decision_rule="heuristic" if method.route == "heuristic" else method.decision_rule,
        real_training_events=method.real_training_events,
        selected_at_events=method.selected_at_events,
        **result_budget_fields(scenario, method),
    )
