"""Real JAX evaluation execution and diagnostic validity, exercised on the training server."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ir2_benchmark.config import HostConfig  # noqa: E402
from ir2_benchmark.env.reset import reset  # noqa: E402
from ir2_benchmark.evaluation.diagnostics import measure_plan_estimate  # noqa: E402
from ir2_benchmark.evaluation.runner import run_manifest  # noqa: E402
from ir2_benchmark.evaluation.scenarios import (  # noqa: E402
    EvaluationMap,
    Method,
    make_scenario,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.evaluation.throughput import measure  # noqa: E402
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash  # noqa: E402
from ir2_benchmark.models.plan_predictor import PlanEstimate  # noqa: E402
from ir2_benchmark.types import MapInput  # noqa: E402


def test_interrupted_real_heuristic_suite_resumes_without_duplicate_results(tmp_path: Path) -> None:
    # Given: a real IR2-encoded tiny corridor and two independent paired scenarios.
    pixels = np.full((5, 12), 127, np.uint8)
    pixels[2, 1:11] = 195
    pixels[2, 1] = 208
    path = tmp_path / "corridor.png"
    Image.fromarray(pixels).save(path)
    world = (pixels <= 150).astype(np.uint8)  # noqa: PLR2004
    config = HostConfig(sensor_radius=1, max_distance=1, max_rounds=1, coverage_target=1)
    map_data = EvaluationMap(
        str(path), obstacle_hash(world), "test", "ir2", "corridor", 12, scenario_hash(world, (1, 2))
    )
    method = Method("heuristic", "heuristic", "fixed", "none", -1, "", "", 0)
    planned = tuple(tasks((make_scenario(map_data, config, seed) for seed in (0, 1)), (method,)))
    first_manifest, full_manifest = tmp_path / "first.jsonl", tmp_path / "full.jsonl"
    write_task_manifest(first_manifest, planned[:1])
    write_task_manifest(full_manifest, planned)
    output = tmp_path / "results"
    # When: a one-task prefix finishes before resuming the full saved workload.
    run_manifest(first_manifest, output)
    continued = run_manifest(full_manifest, output)
    # Then: the first actual episode is skipped, and only the missing episode runs.
    assert (continued.executed, continued.resumed, continued.failures) == (1, 1, 0)
    assert len(tuple((output / "results").glob("*.json"))) == 2  # noqa: PLR2004


def test_plan_telemetry_uses_current_intention_mask_and_reports_cold_start() -> None:
    # Given: a receiver prior with known normalized errors and one generated peer intention.
    config = HostConfig(sensor_radius=0, max_distance=1, plan_horizon=2)
    truth = jnp.zeros((5, 9), jnp.uint8)
    state = reset(
        MapInput(truth, truth == 0, jnp.array([5, 9]), jnp.array([1, 1])),
        config,
        jax.random.PRNGKey(2),
        jnp.array([[1, 1], [2, 1]]),
    )
    state = state._replace(
        event=jnp.int32(3),
        plans=state.plans.at[1].set(jnp.array([[3, 1], [4, 1]])),
        goals=state.goals.at[1].set(jnp.array([4, 1])),
        plan_lengths=state.plan_lengths.at[1].set(1),
        plan_generated_event=state.plan_generated_event.at[1].set(1),
    )
    positions = state.plans / jnp.array([8, 4])
    positions = positions.at[1, 0, 0].add(1 / 8).at[1, 1].set(jnp.array([99, 99]))
    estimate = PlanEstimate(
        state.goals / jnp.array([8, 4]),
        positions,
        jnp.ones((2, 3, 2)),
        jnp.zeros((2, 3)),
        jnp.full(2, -1),
        jnp.ones(2, bool),
    )
    # When: measuring the exact prior used at the actor cutoff, before any new transition.
    measured = jax.jit(
        lambda current: measure_plan_estimate(
            estimate, current, jnp.int32(0), config, event_valid=jnp.bool_(1), has_uncertainty=True
        )
    )(state)
    # Then: padding in the intended route is ignored, and cold-start remains explicit.
    np.testing.assert_allclose(measured.ade[1], 1)
    np.testing.assert_allclose(measured.fde[1], 1)
    assert bool(measured.cold_start[1])
    assert int(measured.message_age[1]) == -1
    assert int(measured.target_event[1]) == 1


def test_timing_counts_no_work_for_a_frozen_real_event() -> None:
    # Given: a real compiled environment already complete at birth.
    from ir2_benchmark.env.step import step_agent  # noqa: PLC0415
    from ir2_benchmark.types import make_action  # noqa: PLC0415

    config = HostConfig(sensor_radius=4, max_distance=1)
    truth = jnp.zeros((2, 2), jnp.uint8)
    state = reset(
        MapInput(truth, truth == 0, jnp.array([2, 2]), jnp.array([0, 0])),
        config,
        jax.random.PRNGKey(1),
        jnp.array([[0, 0], [1, 1]]),
    )
    function = jax.jit(lambda: step_agent(state, make_action(state.positions[0]), config))
    # When: synchronous timing counts only actually active real events.
    timing = measure(function, lambda result: int(result.info.active))
    # Then: frozen padding events cannot create a positive throughput number.
    assert timing.actual_units == 0
    assert timing.units_per_second is None
    assert len(timing.samples_seconds) == 10  # noqa: PLR2004
    assert timing.warmup_count == 3  # noqa: PLR2004
