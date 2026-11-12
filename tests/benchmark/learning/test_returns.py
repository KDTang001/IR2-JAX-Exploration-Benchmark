"""Independent hand arithmetic for ordered-event generalized advantages."""

import jax
import jax.numpy as jnp
import numpy as np

from ir2_benchmark.training.ppo import event_gae


def test_gae_when_two_step_rewards_are_asymmetric() -> None:
    # Given: G1=2 and G0=1+0.5*2=2, with actual final termination.
    rewards = jnp.asarray([1.0, 2.0])
    # When: reverse event GAE uses lambda=1 and zero values.
    result = jax.jit(event_gae)(
        rewards,
        jnp.zeros(3),
        jnp.asarray([False, True]),
        jnp.zeros(2, bool),
        jnp.ones(2, bool),
        0.5,
        1.0,
    )
    # Then: independent arithmetic defines both policy advantages.
    np.testing.assert_array_equal(result.advantages, [2.0, 2.0])


def test_gae_when_padding_follows_terminal_event() -> None:
    # Given: invalid padding has deliberately large rewards and value placeholders.
    # When: the real two-event prefix is evaluated with two padded events appended.
    result = event_gae(
        jnp.asarray([1.0, 2.0, 99.0, 88.0]),
        jnp.asarray([0.0, 0.0, 0.0, 77.0, 66.0]),
        jnp.asarray([False, True, True, True]),
        jnp.zeros(4, bool),
        jnp.asarray([True, True, False, False]),
        0.5,
        1.0,
    )
    # Then: done padding cannot change real returns or become a training sample.
    np.testing.assert_array_equal(result.advantages, [2.0, 2.0, 0.0, 0.0])


def test_gae_when_sampler_cutoff_is_not_task_time_limit() -> None:
    # Given: a one-event sampler cutoff retains next V=4, while task expiry has no future value.
    reward, values, valid = jnp.asarray([1.0]), jnp.asarray([0.0, 4.0]), jnp.ones(1, bool)
    # When: compute the sampler cutoff and finite-task-expiry targets independently.
    cutoff = event_gae(reward, values, jnp.zeros(1, bool), jnp.zeros(1, bool), valid, 0.5, 1.0)
    expiry = event_gae(reward, values, jnp.zeros(1, bool), jnp.ones(1, bool), valid, 0.5, 1.0)
    # Then: sampler bootstrap adds .5*4, whereas the known finite task ends at reward one.
    np.testing.assert_array_equal(cutoff.targets, [3.0])
    np.testing.assert_array_equal(expiry.targets, [1.0])
