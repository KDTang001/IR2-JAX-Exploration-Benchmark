"""Measured route-B kernels receive dynamic arrays and start from actual replay cutoffs."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.scenarios import load_scenario
from ir2_benchmark.training.b_state import BTrainingConfig, _cognition
from ir2_benchmark.training.evaluate_b import BThroughputFunctions, benchmark_functions
from ir2_benchmark.training.replay import ReplayBuffer
from ir2_benchmark.training.train_b import train_b


@dataclass(frozen=True, slots=True)
class BenchmarkFixture:
    """One actual small training checkpoint with its host replay and benchmark callables."""

    replay: Path
    functions: BThroughputFunctions


@pytest.fixture(scope="module")
def benchmark_fixture(tmp_path_factory: pytest.TempPathFactory) -> BenchmarkFixture:
    root = tmp_path_factory.mktemp("b-benchmark")
    pixels = np.full((9, 9), 195, np.uint8)
    pixels[4, 4] = 208
    path = root / "map.png"
    Image.fromarray(pixels).save(path)
    config = HostConfig(sensor_radius=1, max_distance=2, max_candidates=4, max_rounds=2)
    settings = BTrainingConfig(
        hidden=8,
        categories=2,
        classes=3,
        sequence_length=3,
        burn_in=1,
        logical_batch=2,
        update_every=4,
        warmup_events=0,
        imagination_rounds=3,
        development=True,
    )
    output = root / "training"
    train_b(config, 17, 4, output, options=settings, map_path=path)
    functions = benchmark_functions(
        output / "checkpoint",
        load_scenario(config, 17, path),
        config,
        settings,
        jax.random.PRNGKey(23),
    )
    return BenchmarkFixture(output / "replay.npz", functions)


def test_imagination_origin_restores_actual_replay_and_burned_in_state(
    benchmark_fixture: BenchmarkFixture,
) -> None:
    # Given: a real trained checkpoint whose replay contains nonzero event cutoffs.
    functions = benchmark_fixture.functions
    metadata = functions.metadata
    initial, carry, *_ = functions.imagination.args
    frame = ReplayBuffer.load(benchmark_fixture.replay).reconstruct(
        metadata.episode_id, metadata.origin_index
    )
    # When: inspecting the exact dynamic origin bound outside the timed kernel.
    reference = _cognition(frame)
    # Then: all public map/inbox/budget fields and recurrent memories match the saved cutoff.
    for actual, expected in zip(
        jax.tree_util.tree_leaves(initial.cognition),
        jax.tree_util.tree_leaves(reference),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(carry.actor_memory, frame.field("all_actor_memory"))
    assert int(initial.cognition.event) >= 1
    assert np.any(np.asarray(initial.latent.deterministic) != 0)
    assert metadata.origin == "actual_replay_burn_in"
    assert metadata.world_logical_batch == 2  # noqa: PLR2004
    assert metadata.world_physical_microbatch == 1
    assert metadata.imagination_logical_origins == metadata.imagination_physical_batch == 1
    assert metadata.maximum_imagined_events == 6  # noqa: PLR2004


def test_timed_world_update_has_dynamic_inputs_and_performs_optimizer_step(
    benchmark_fixture: BenchmarkFixture,
) -> None:
    # Given: the production no-argument adapter binds inputs outside its compiled function.
    function = benchmark_fixture.functions.world_update
    world, batches, key = function.args
    boundary = jax.make_jaxpr(function.func)(*function.args)
    assert len(boundary.jaxpr.invars) == len(jax.tree_util.tree_leaves(function.args))
    assert boundary.jaxpr.invars
    # When: running the actual measured model optimizer and changing only dynamic weights.
    first = jax.block_until_ready(function())
    compilations = function.func._cache_size()  # noqa: SLF001
    changed = world.replace(params=jax.tree_util.tree_map(lambda value: value + 0.02, world.params))
    second = jax.block_until_ready(function.func(changed, batches, key))
    # Then: an optimizer step changes real parameters; same-shape weights reuse the executable.
    assert int(first.state.step) == int(world.step) + 1
    assert int(first.updates) == 1
    assert int(first.valid_training_events) > 0
    assert np.isfinite(float(first.loss))
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(world.params),
            jax.tree_util.tree_leaves(first.state.params),
            strict=True,
        )
    )
    assert not np.isclose(float(first.loss), float(second.loss))
    assert function.func._cache_size() == compilations  # noqa: SLF001


def test_timed_imagination_treats_runtime_origin_as_dynamic_input(
    benchmark_fixture: BenchmarkFixture,
) -> None:
    # Given: the same compiled modeled rollout accepts its full actual replay origin as input.
    function = benchmark_fixture.functions.imagination
    initial, *remaining = function.args
    boundary = jax.make_jaxpr(function.func)(*function.args)
    assert len(boundary.jaxpr.invars) == len(jax.tree_util.tree_leaves(function.args))
    first = jax.block_until_ready(function())
    compilations = function.func._cache_size()  # noqa: SLF001
    # When: the runtime origin is terminal, keeping every array shape and weight unchanged.
    stopped = initial._replace(cognition=initial.cognition._replace(terminated=jnp.bool_(1)))
    second = jax.block_until_ready(function.func(stopped, *remaining))
    # Then: frozen events are excluded from counted work without compiling another executable.
    assert int(first.valid_events) >= 1
    assert first.batch.valid.shape == (6,)
    assert int(first.valid_events) == int(first.batch.valid.sum())
    assert int(second.valid_events) == 0
    assert function.func._cache_size() == compilations  # noqa: SLF001
