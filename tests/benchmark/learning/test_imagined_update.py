"""Score-function training evidence requires no real environment step or simulator object."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from ir2_benchmark.agents.actor import Actor, IndexAction
from ir2_benchmark.models.encoders import ActorFeatures
from ir2_benchmark.models.world_model import ImaginedCritic
from ir2_benchmark.training.dreamer import (
    DreamerConfig,
    ImaginedBatch,
    lambda_returns,
    trajectory_returns,
    update_imagined,
)


def test_lambda_returns_match_hand_oracle_when_last_event_terminates() -> None:
    # Given: two rewards, one-step continuation, zero values, gamma=lambda=1.
    # When: backing up the two-event modeled trajectory.
    returns = lambda_returns(
        jnp.array([1.0, 2.0]),
        jnp.zeros(2),
        jnp.array([1.0, 0.0]),
        jnp.float32(99),
        gamma=1,
        lambda_=1,
    )
    # Then: termination blocks the unrelated bootstrap value.
    np.testing.assert_array_equal(returns, [3.0, 2.0])


@pytest.mark.parametrize("origins", [1, 2])
def test_actor_changes_from_imagined_batch_when_no_environment_object_exists(origins: int) -> None:
    # Given: local policy inputs and nonzero model rewards, with separate central features.
    steps = 4
    features = ActorFeatures(
        jnp.zeros((steps, 5, 5, 3)),
        jnp.ones((steps, 3, 7)),
        jnp.ones((steps, 3), bool),
        jnp.zeros((steps, 6)),
        jnp.zeros((steps, 2, 8)),
        jnp.ones((steps, 2), bool),
    )
    actor_model = Actor(hidden=16)
    critic_model = ImaginedCritic(hidden=16)
    key = jax.random.PRNGKey(73)
    memories = jnp.zeros((steps, 16))
    plans = jnp.ones((steps, 13))
    actor_variables = actor_model.init(key, features, memories, plans)
    central = jnp.arange(steps * 8, dtype=jnp.float32).reshape(steps, 8) / 10
    critic_variables = critic_model.init(key, central)
    actor = TrainState.create(
        apply_fn=actor_model.apply, params=actor_variables["params"], tx=optax.adam(1e-3)
    )
    critic = TrainState.create(
        apply_fn=critic_model.apply, params=critic_variables["params"], tx=optax.adam(1e-3)
    )
    batch = ImaginedBatch(
        features,
        memories,
        plans,
        IndexAction(jnp.array([0, 1, 2, 0]), jnp.ones(steps, bool), jnp.zeros(steps, jnp.int32)),
        central,
        jnp.zeros(8),
        jnp.array([1.0, 2.0, 3.0, 4.0]),
        jnp.array([1.0, 1.0, 1.0, 0.0]),
        jnp.ones(steps, bool),
    )
    if origins == 2:  # noqa: PLR2004
        second = batch._replace(rewards=batch.rewards + 10)
        batch = jax.tree_util.tree_map(
            lambda first, other: jnp.stack((first, other)), batch, second
        )
    # When: optimizing the imagined score-function objective alone.
    updated, _, statistics = jax.jit(update_imagined, static_argnames=("config",))(
        actor, critic, batch, DreamerConfig()
    )
    # Then: policy parameters changed with no real-env argument or real step to call.
    assert float(statistics.actor_parameter_delta) > 0
    assert int(updated.step) == 1
    assert int(statistics.valid_events) == steps * origins


def test_distinct_start_returns_do_not_cross_trajectories_when_cutoffs_nonterminal() -> None:
    # Given: two nonterminal two-step starts, each with its own zero bootstrap.
    rewards = jnp.array([[1.0, 2.0], [10.0, 20.0]])
    # When: backing up the starts as a batch rather than one concatenated timeline.
    result = trajectory_returns(
        rewards,
        jnp.zeros((2, 2)),
        jnp.ones((2, 2)),
        jnp.zeros(2),
        jnp.ones((2, 2), bool),
        DreamerConfig(gamma=1, lambda_=1),
    )
    # Then: trajectory one's return excludes trajectory two's rewards despite no terminal boundary.
    np.testing.assert_array_equal(result, [[3.0, 2.0], [30.0, 20.0]])


def test_padded_return_retains_cutoff_bootstrap_when_real_sequence_short() -> None:
    # Given: one nonterminal reward and two masked pad events, with cutoff value four.
    # When: computing a return using only the actual first event.
    result = lambda_returns(
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([2.0, 4.0, 4.0]),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.float32(4),
        gamma=1,
        lambda_=1,
        valid=jnp.array([True, False, False]),
    )
    # Then: padding neither erases the bootstrap nor invents stationary transitions.
    np.testing.assert_array_equal(result, [5.0, 0.0, 0.0])
