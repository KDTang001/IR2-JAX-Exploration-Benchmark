"""Actual B physical lanes match independent events and freeze disabled quota tails."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.reset import reset
from ir2_benchmark.training.b_collect import make_real_collector, real_event
from ir2_benchmark.training.b_state import BTrainingConfig, _initialize
from ir2_benchmark.types import MapInput


def test_four_physical_lanes_match_four_independent_real_events_and_freeze_masked_lane() -> None:
    # Given: four independently spawned/fault-keyed real environments sharing one parameter set.
    config = HostConfig(sensor_radius=1, max_distance=2, max_candidates=4, max_rounds=4)
    options = BTrainingConfig(
        hidden=8, categories=2, classes=3, physical_env_count=4, development=True
    )
    grid = jnp.zeros((9, 9), jnp.uint8)
    data = MapInput(grid, grid == 0, jnp.array([9, 9]), jnp.array([4, 4]))
    keys = jax.random.split(jax.random.PRNGKey(19), 4)
    states = tuple(reset(data, config, key, episode=lane) for lane, key in enumerate(keys))
    actor, predictor, _, _, carry, _ = _initialize(config, options, states[0], keys[0])
    enabled = jnp.array([True, True, False, True])
    batched_states = jax.tree.map(lambda *values: jnp.stack(values), *states)
    batched_carry = jax.tree.map(lambda value: jnp.repeat(value[None], 4, axis=0), carry)
    references = tuple(
        real_event(
            state,
            carry,
            actor,
            predictor,
            key,
            jnp.bool_(0),
            enabled[lane],
            config=config,
            options=options,
        )
        for lane, (state, key) in enumerate(zip(states, keys, strict=True))
    )
    # When: one actual compiled vmapped event runs across all physical lanes.
    actual = make_real_collector(config, options)(
        batched_states, batched_carry, actor, predictor, keys, jnp.bool_(0), enabled
    )
    jax.block_until_ready(actual)
    expected = jax.tree.map(lambda *values: jnp.stack(values), *references)
    # Then: every simulated state/action/supervision leaf agrees with independent execution.
    for observed, reference in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
    ):
        np.testing.assert_allclose(observed, reference, rtol=1e-5, atol=1e-6)
    for before, after in zip(
        jax.tree_util.tree_leaves((batched_states, batched_carry)),
        jax.tree_util.tree_leaves((actual.result.state, actual.decision.carry)),
        strict=True,
    ):
        np.testing.assert_array_equal(before[2], after[2])
    np.testing.assert_array_equal(actual.result.info.active, enabled)
    assert not np.array_equal(states[0].key, states[1].key)
    assert actual.result.state.event.shape == (4,)


def test_physical_batch_one_has_same_event_as_original_single_lane_contract() -> None:
    # Given: the default physical batch is one and its actor/model dimensions are unchanged.
    config = HostConfig(sensor_radius=1, max_distance=2, max_candidates=4)
    options = BTrainingConfig(hidden=8, categories=2, classes=3, development=True)
    grid = jnp.zeros((9, 9), jnp.uint8)
    state = reset(
        MapInput(grid, grid == 0, jnp.array([9, 9]), jnp.array([4, 4])),
        config,
        jax.random.PRNGKey(5),
    )
    actor, predictor, _, _, carry, _ = _initialize(config, options, state, jax.random.PRNGKey(6))
    # When: P1 is compared with the exact same unbatched real event and warmup action rule.
    reference = real_event(
        state,
        carry,
        actor,
        predictor,
        jax.random.PRNGKey(7),
        jnp.bool_(1),
        jnp.bool_(1),
        config=config,
        options=options,
    )
    batched = jax.tree.map(lambda value: value[None], (state, carry, jax.random.PRNGKey(7)))
    result = make_real_collector(config, replace(options, physical_env_count=1))(
        batched[0], batched[1], actor, predictor, batched[2], jnp.bool_(1), jnp.ones(1, bool)
    )
    # Then: singleton batching preserves the actual state, actions and receiver-visible memories.
    for observed, expected in zip(
        jax.tree_util.tree_leaves(result), jax.tree_util.tree_leaves(reference), strict=True
    ):
        np.testing.assert_allclose(observed[0], expected, rtol=1e-5, atol=1e-6)
