"""Real batched inference, dynamic-value compilation reuse and legacy seed equivalence."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from flax.core import freeze
from jax import Array

from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.rollout import RolloutResult
from ir2_benchmark.training.checkpoint import save_checkpoint
from ir2_benchmark.training.collect_a import Parameters
from ir2_benchmark.training.evaluate_a import (
    EvaluationKernel,
    get_evaluation_kernel,
    inference_settings,
    run_evaluation,
)
from ir2_benchmark.training.train_a import TrainAConfig, configuration_hash, initialize_training
from ir2_benchmark.types import MapInput

KernelArgs = tuple[Parameters, MapInput, Array, Array, Array, Array]
KernelResult = tuple[RolloutResult, Array]


class KernelCase(NamedTuple):
    """Actual small-map state and parameters initialized by the production training initializer."""

    parameters: Parameters
    map_data: MapInput
    config: HostConfig
    settings: TrainAConfig
    spawns: Array


@pytest.fixture(scope="module")
def kernel_case() -> KernelCase:
    grid = jnp.ones((7, 7), jnp.uint8).at[1:6, 1:6].set(0)
    data = MapInput(grid, grid == 0, jnp.asarray([7, 7]), jnp.asarray([3, 3]))
    config = HostConfig(
        n_agents=2,
        sensor_radius=1,
        max_distance=1,
        plan_horizon=2,
        max_rounds=3,
        initial_budget=100,
        max_candidates=3,
        spawn_radius=1,
        fault_mode=1,
        dropout_probability=0.3,
    )
    settings = TrainAConfig(
        events=8,
        warmup_events=0,
        rollout_length=2,
        ppo_epochs=1,
        batch_size=1,
        hidden=8,
        predictor_replay_capacity=4,
        suite="micro",
    )
    state = initialize_training(data, config, settings, jax.random.PRNGKey(1))
    params = Parameters(
        freeze(state.actor.params), freeze(state.critic.params), freeze(state.predictor.params)
    )
    return KernelCase(params, data, config, settings, jnp.asarray([[3, 3], [4, 3]]))


def _arguments(case: KernelCase, batch_size: int) -> KernelArgs:
    """Every leaf receives a real batch axis; scenario randomness remains an array input."""
    parameters = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size, *value.shape)), case.parameters
    )
    data = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size, *value.shape)), case.map_data
    )
    return (
        parameters,
        data,
        jax.random.split(jax.random.PRNGKey(27), batch_size),
        jnp.broadcast_to(case.spawns, (batch_size, *case.spawns.shape)),
        jax.random.split(jax.random.PRNGKey(103), batch_size),
        jnp.zeros((batch_size, case.config.n_slots), bool),
    )


def _compilations(kernel: EvaluationKernel) -> int:
    """Use JAX 0.4.35's executable-cache counter as direct runtime compilation evidence."""
    return kernel._cache_size()  # noqa: SLF001


def _assert_same(actual: KernelResult, expected: KernelResult) -> None:
    """Compare cell distances in map-diagonal units; all discrete evidence stays exact.

    For s=max(valid_shape[::-1]-1,1), the reverse triangle inequality gives
    |D(mu_a)-D(mu_b)|/norm(s) <= max(abs(mu_a-mu_b)). Only goal_error/ADE/FDE
    use this dimensionless contract. Their cell-unit absolute floor therefore
    changes to norm(s)*1e-7; other floats retain rtol=1e-6 and atol=1e-7.
    """
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    np.testing.assert_array_equal(
        actual[0].final_state.valid_shape, expected[0].final_state.valid_shape
    )
    shapes = np.asarray(expected[0].final_state.valid_shape, dtype=np.float64)
    diagonal = np.linalg.norm(np.maximum(shapes[..., ::-1] - 1, 1), axis=-1)
    for (path, observed), (_, reference) in zip(
        jax.tree_util.tree_flatten_with_path(actual)[0],
        jax.tree_util.tree_flatten_with_path(expected)[0],
        strict=True,
    ):
        label = jax.tree_util.keystr(path)
        left, right = np.asarray(observed), np.asarray(reference)
        assert left.shape == right.shape, label
        assert left.dtype == right.dtype, label
        if not np.issubdtype(left.dtype, np.inexact):
            np.testing.assert_array_equal(left, right, err_msg=label)
            continue
        np.testing.assert_array_equal(np.isnan(left), np.isnan(right), err_msg=label)
        np.testing.assert_array_equal(np.isfinite(left), np.isfinite(right), err_msg=label)
        if label in (
            "[0].plan_diagnostics.goal_error",
            "[0].plan_diagnostics.ade",
            "[0].plan_diagnostics.fde",
        ):
            divisor = diagonal.reshape((diagonal.shape[0], *(1 for _ in left.shape[1:])))
            left, right = left.astype(np.float64) / divisor, right.astype(np.float64) / divisor
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True, err_msg=label)


def test_kernel_when_only_training_settings_and_dynamic_values_change_reuses_compilation(
    kernel_case: KernelCase,
) -> None:
    # Given: actual network parameters and one fixed inference/geometry signature.
    case = kernel_case
    kernel = get_evaluation_kernel(case.config, case.settings)
    other_training = replace(
        case.settings,
        events=32,
        warmup_events=4,
        rollout_length=4,
        batch_size=2,
        ppo_epochs=2,
        learning_rate=1e-4,
        predictor_replay_capacity=16,
        predictor_recent_fraction=0.75,
    )
    arguments = _arguments(case, 1)
    # When: execute twice with changed arrays while preserving all shapes/dtypes.
    jax.block_until_ready(kernel(*arguments))
    compiled = _compilations(kernel)
    params, data, _, spawns, _, _ = arguments
    grid = data.occupancy.at[0, 1, 1].set(1)
    changed = (
        jax.tree.map(lambda value: value * 1.01, params),
        data._replace(occupancy=grid, component=grid == 0),
        jax.random.PRNGKey(77)[None],
        spawns[:, ::-1],
        jax.random.PRNGKey(207)[None],
        jnp.ones((1, case.config.n_slots), bool),
    )
    jax.block_until_ready(kernel(*changed))
    # Then: optimizer/quota changes reuse the factory, and data changes reuse its executable.
    assert inference_settings(case.settings) == inference_settings(other_training)
    assert get_evaluation_kernel(case.config, other_training) is kernel
    assert _compilations(kernel) == compiled


def test_kernel_when_batch_has_two_real_scenarios_matches_two_single_runs(
    kernel_case: KernelCase,
) -> None:
    # Given: explicit independent policy/fault keys with the same physical bucket.
    kernel = get_evaluation_kernel(kernel_case.config, kernel_case.settings)
    arguments = _arguments(kernel_case, 2)
    # When: run the true environment in one batch and in two independently sliced batch-one calls.
    batched = kernel(*arguments)
    first = kernel(*jax.tree.map(lambda value: value[:1], arguments))
    second = kernel(*jax.tree.map(lambda value: value[1:2], arguments))
    expected = jax.tree.map(lambda left, right: jnp.concatenate((left, right)), first, second)
    # Then: every state, trace, diagnostic and initial sensing footprint agrees.
    _assert_same(batched, expected)
    # Negative controls reuse these outputs; no extra policy or environment execution.
    diagnostics = batched[0].plan_diagnostics
    assert diagnostics is not None
    point = tuple(np.argwhere(np.isfinite(np.asarray(diagnostics.goal_error)))[0])
    wrong_distance = diagnostics._replace(goal_error=diagnostics.goal_error.at[point].add(1))
    with pytest.raises(AssertionError, match=r"plan_diagnostics\.goal_error"):
        _assert_same((batched[0]._replace(plan_diagnostics=wrong_distance), batched[1]), expected)
    state = batched[0].final_state
    key_index = tuple(np.argwhere(np.asarray(state.key) > (1 << 24))[0])
    wrong_key = state.key.at[key_index].set(jnp.bitwise_xor(state.key[key_index], jnp.uint32(1)))
    with pytest.raises(AssertionError, match=r"final_state\.key"):
        _assert_same(
            (batched[0]._replace(final_state=state._replace(key=wrong_key)), batched[1]), expected
        )


def test_legacy_evaluation_when_spawns_are_implicit_keeps_seed_and_cached_kernel_semantics(
    kernel_case: KernelCase,
    tmp_path: Path,
) -> None:
    # Given: an actual initialized-model inference checkpoint and legacy implicit-spawn key.
    case = kernel_case
    checkpoint = tmp_path / "checkpoint"
    identity = configuration_hash(case.config, case.settings)
    models = {
        name: {"params": serialization.to_state_dict(params)}
        for name, params in zip(("actor", "critic", "predictor"), case.parameters, strict=True)
    }
    save_checkpoint(checkpoint, serialization.msgpack_serialize(models), identity)
    (tmp_path / "configuration.json").write_text(
        json.dumps(
            {
                "environment": asdict(case.config),
                "training": asdict(case.settings),
                "configuration_hash": identity,
            }
        ),
        encoding="utf-8",
    )
    key = jax.random.PRNGKey(29)
    initial = reset(case.map_data, case.config, key)
    kernel = get_evaluation_kernel(case.config, case.settings)
    expected, birth = kernel(
        jax.tree.map(lambda value: value[None], case.parameters),
        jax.tree.map(lambda value: value[None], case.map_data),
        key[None],
        initial.positions[: case.config.n_agents][None],
        key[None],
        initial.isolated_mask[None],
    )
    jax.block_until_ready(expected)
    compiled = _compilations(kernel)
    # When: the unchanged legacy signature delegates to batch one of the same kernel.
    actual = run_evaluation(checkpoint, case.map_data, case.config, case.settings, key)
    # Then: legacy birth randomness and rollout diagnostics retain the same executable.
    _assert_same(
        (jax.tree.map(lambda value: value[None], actual), initial.physical_seen[None]),
        (expected, birth),
    )
    assert _compilations(kernel) == compiled
