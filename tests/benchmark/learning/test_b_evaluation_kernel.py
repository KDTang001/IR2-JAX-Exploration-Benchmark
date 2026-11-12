"""B evaluation batching preserves independent real episodes and reuses compiled inference."""

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from flax.core import freeze
from jax import Array

from ir2_benchmark.agents.actor import Actor, initial_memory
from ir2_benchmark.communication.faults import KeyPurpose, fold_key
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.models.encoders import actor_features, selected_plan_features
from ir2_benchmark.models.plan_predictor import PlanPredictor, initial_plan_memory, plan_inputs
from ir2_benchmark.training.b_state import BTrainingConfig
from ir2_benchmark.training.checkpoint import save_checkpoint
from ir2_benchmark.training.evaluate_b import (
    Parameters,
    get_evaluation_kernel,
    load_parameters,
    run_evaluation,
)
from ir2_benchmark.types import MapInput


@dataclass(frozen=True, slots=True)
class EvaluationFixture:
    """Two independent real maps/checkpoints and their paired scenario inputs."""

    config: HostConfig
    settings: BTrainingConfig
    maps: tuple[MapInput, MapInput]
    checkpoints: tuple[Path, Path]
    parameters: tuple[Parameters, Parameters]
    keys: Array
    spawns: Array
    fault_keys: Array
    isolated: Array


@pytest.fixture(scope="module", params=[False, True])
def evaluation_fixture(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> EvaluationFixture:
    config = HostConfig(
        sensor_radius=1,
        max_distance=2,
        max_candidates=4,
        max_rounds=2,
        fault_mode=2,
        burst_start=0,
        burst_duration=1,
    )
    settings = BTrainingConfig(
        hidden=8, categories=2, classes=3, development=True, candidate_attention=request.param
    )
    first = jnp.zeros((9, 9), jnp.uint8)
    second = first.at[:, 4].set(1).at[4, 4].set(0)
    maps = tuple(
        MapInput(grid, grid == 0, jnp.array([9, 9]), jnp.array([3, 4])) for grid in (first, second)
    )
    spawns = jnp.array([[[2, 2], [6, 6]], [[2, 3], [6, 5]]], jnp.int32)
    initial = reset(maps[0], config, jax.random.PRNGKey(2), spawns=spawns[0])
    observation = observe(initial, initial.current_agent, config)
    actor = Actor(hidden=settings.hidden, candidate_attention=settings.candidate_attention)
    predictor = PlanPredictor(
        config.plan_horizon, settings.hidden, settings.categories, settings.classes
    )
    features = actor_features(
        observation,
        config.n_agents,
        initial_budget=config.initial_budget,
        max_rounds=config.max_rounds,
        patch_radius=1,
        candidate_attention=settings.candidate_attention,
    )
    plan = selected_plan_features(
        observation, observation.position, config.max_distance, config.plan_horizon
    )
    actor_params = actor.init(
        jax.random.PRNGKey(3), features, initial_memory(hidden=settings.hidden), plan
    )["params"]
    pair = initial_plan_memory(
        (config.n_slots,), settings.hidden, settings.categories, settings.classes
    )
    predictor_params = predictor.init(
        jax.random.PRNGKey(4), pair, plan_inputs(observation), jax.random.PRNGKey(5)
    )["params"]
    zeros = freeze(jax.tree_util.tree_map(jnp.zeros_like, actor_params))
    prediction = freeze(jax.tree_util.tree_map(jnp.zeros_like, predictor_params))
    parameters = tuple(
        Parameters(
            zeros.copy({"gate": zeros["gate"].copy({"bias": jnp.array([bias])})}), prediction
        )
        for bias in (100.0, -100.0)
    )
    configuration = {"environment": asdict(config), "training": asdict(settings)}
    if not settings.candidate_attention:
        del configuration["training"]["candidate_attention"]
    identity = sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    root = tmp_path_factory.mktemp("b-kernel")
    paths = []
    for index, params in enumerate(parameters):
        directory = root / str(index)
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(configuration), encoding="utf-8")
        path = directory / "checkpoint"
        save_checkpoint(
            path,
            serialization.to_bytes(
                {"actor": {"params": params.actor}, "predictor": {"params": params.predictor}}
            ),
            identity,
        )
        paths.append(path)
    return EvaluationFixture(
        config,
        settings,
        maps,
        tuple(paths),
        parameters,
        jax.random.split(jax.random.PRNGKey(7), 2),
        spawns,
        jax.random.split(jax.random.PRNGKey(8), 2),
        jnp.array([[False, False], [False, True]]),
    )


def test_batched_kernel_matches_independent_batch_one_episodes(
    evaluation_fixture: EvaluationFixture,
) -> None:
    # Given: two distinct checkpoints/maps and independent policy, spawn, and fault inputs.
    fixture = evaluation_fixture
    kernel = get_evaluation_kernel(fixture.config, fixture.settings)
    parameters = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *fixture.parameters)
    maps = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *fixture.maps)
    references = []
    births = []
    for index in range(2):
        references.append(
            run_evaluation(
                fixture.checkpoints[index],
                fixture.maps[index],
                fixture.config,
                fixture.settings,
                fixture.keys[index],
                spawns=fixture.spawns[index],
                fault_key=fixture.fault_keys[index],
                isolated_mask=fixture.isolated[index],
            )
        )
        births.append(
            reset(
                fixture.maps[index],
                fixture.config,
                fixture.keys[index],
                spawns=fixture.spawns[index],
            ).physical_seen
        )
    expected = (
        jax.tree_util.tree_map(lambda *values: jnp.stack(values), *references),
        jnp.stack(births),
    )
    # When: executing those same scenarios in one persistent batch-two kernel.
    actual = kernel(
        parameters, maps, fixture.keys, fixture.spawns, fixture.fault_keys, fixture.isolated
    )
    jax.block_until_ready(actual)
    # Then: all state, trace, diagnostic, and birth-footprint leaves match independent episodes.
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for observed, reference in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(observed, reference)


def test_same_shapes_reuse_kernel_when_policy_keys_and_parameters_change(
    evaluation_fixture: EvaluationFixture,
) -> None:
    # Given: a compiled shape and two weight sets differing only in a certain paid-send gate.
    fixture = evaluation_fixture
    kernel = get_evaluation_kernel(fixture.config, fixture.settings)
    positive = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value)), fixture.parameters[0]
    )
    negative = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value)), fixture.parameters[1]
    )
    maps = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), fixture.maps[0])
    first = kernel(
        positive, maps, fixture.keys, fixture.spawns, fixture.fault_keys, fixture.isolated
    )
    jax.block_until_ready(first)
    compilations = kernel._cache_size()  # noqa: SLF001
    candidates = jax.random.split(jax.random.PRNGKey(29), 32)
    choices = jax.vmap(
        lambda key: jax.random.categorical(
            jax.random.split(fold_key(key, KeyPurpose.POLICY), 3)[1], jnp.zeros(5)
        )
    )(candidates)
    changed = candidates[jnp.argmax(choices != choices[0])]
    # When: replacing dynamic weights and keys while preserving every input shape.
    second = kernel(
        negative,
        maps,
        jnp.stack((candidates[0], changed)),
        fixture.spawns,
        fixture.fault_keys,
        fixture.isolated,
    )
    jax.block_until_ready(second)
    # Then: the trace is reused, while dynamic weights and keys change their respective outputs.
    assert kernel._cache_size() == compilations  # noqa: SLF001
    assert int(first[0].trace.spent.sum()) > 0
    assert int(second[0].trace.spent.sum()) == 0
    assert not np.array_equal(
        second[0].trace.goals[0, 0, 0] - fixture.spawns[0, 0],
        second[0].trace.goals[1, 0, 0] - fixture.spawns[1, 0],
    )
    training_only = replace(
        fixture.settings, real_updates=True, logical_batch=2, learning_rate=1e-4
    )
    assert get_evaluation_kernel(fixture.config, training_only) is kernel


def test_parameter_loader_preserves_actual_checkpoint_weights(
    evaluation_fixture: EvaluationFixture,
) -> None:
    # Given: the saved low-send checkpoint and its declared training settings.
    fixture = evaluation_fixture
    # When: loading parameters through the public inference boundary.
    parameters, settings = load_parameters(fixture.checkpoints[1])
    # Then: both parameter trees and inference settings survive exactly.
    assert settings == fixture.settings
    for observed, expected in zip(
        jax.tree_util.tree_leaves(parameters),
        jax.tree_util.tree_leaves(fixture.parameters[1]),
        strict=True,
    ):
        np.testing.assert_array_equal(observed, expected)
