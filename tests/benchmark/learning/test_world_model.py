"""World-model dependence and supervision boundaries."""

import jax
import jax.numpy as jnp
import numpy as np

from ir2_benchmark.models.world_model import WorldModel, WorldObservation, initial_rssm
from ir2_benchmark.training.world_losses import WorldTarget, local_change_sample_mask, world_loss


def test_world_prediction_depends_on_action_when_history_and_noise_same() -> None:
    # Given: one initialized model, fixed history and shared latent randomness.
    model = WorldModel(2, 5, 2, deterministic=16, categories=2, classes=3)
    state = initial_rssm(deterministic=16, categories=2, classes=3)
    observation = WorldObservation(jnp.zeros((5, 5, 3)), jnp.zeros((2, 4)), jnp.zeros(2))
    action = jnp.zeros(8)
    key = jax.random.PRNGKey(11)
    variables = model.init(key, state, action, observation, key)
    # When: contrasting legal action encodings with the same history/key.
    first = model.apply(variables, state, action, key, method=model.imagine)
    second = model.apply(variables, state, action.at[2].set(1), key, method=model.imagine)
    # Then: action conditioning changes the predicted motion distribution.
    assert not np.allclose(first.movement_logits, second.movement_logits)


def test_world_loss_is_zero_when_all_replay_entries_padding() -> None:
    # Given: arbitrary decoder outputs and an entirely masked sequence.
    model = WorldModel(2, 3, 2, deterministic=8, categories=2, classes=3)
    state = initial_rssm(deterministic=8, categories=2, classes=3)
    observation = WorldObservation(jnp.zeros((3, 3, 3)), jnp.zeros((2, 4)), jnp.zeros(2))
    action = jnp.zeros(8)
    key = jax.random.PRNGKey(7)
    variables = model.init(key, state, action, observation, key)
    prediction = model.apply(variables, state, action, observation, key)
    target = WorldTarget(
        jnp.zeros((3, 3), jnp.int32),
        jnp.ones((3, 3), bool),
        jnp.int32(0),
        jnp.zeros(2, bool),
        jnp.ones(2, bool),
        jnp.float32(1),
        jnp.bool_(True),  # noqa: FBT003
        jnp.bool_(False),  # noqa: FBT003
    )
    # When: padding reaches all model objectives.
    loss = world_loss(prediction, target)
    # Then: no target is learned as a stationary transition and no denominator creates NaN.
    np.testing.assert_array_equal(np.asarray(loss), np.zeros(8))


def test_continuation_target_stays_one_when_episode_only_time_truncated() -> None:
    # Given: large positive continuation logit, with no true coverage termination.
    model = WorldModel(2, 3, 2, deterministic=8, categories=2, classes=3)
    state = initial_rssm(deterministic=8, categories=2, classes=3)
    observation = WorldObservation(jnp.zeros((3, 3, 3)), jnp.zeros((2, 4)), jnp.zeros(2))
    key = jax.random.PRNGKey(8)
    variables = model.init(key, state, jnp.zeros(8), observation, key)
    prediction = model.apply(variables, state, jnp.zeros(8), observation, key)
    prediction = prediction._replace(continuation_logit=jnp.float32(10))
    target = WorldTarget(
        jnp.zeros((3, 3), jnp.int32),
        jnp.ones((3, 3), bool),
        jnp.int32(0),
        jnp.zeros(2, bool),
        jnp.ones(2, bool),
        jnp.float32(0),
        jnp.bool_(False),  # noqa: FBT003
        jnp.bool_(True),  # noqa: FBT003
    )
    # When: training the continuation component.
    loss = world_loss(prediction, target)
    # Then: time-limit truncation is not a learned environment failure.
    assert float(loss.continuation) < 0.001  # noqa: PLR2004


def test_changed_local_states_are_selected_when_unchanged_sampling_disabled() -> None:
    # Given: two real changed cells and one changed padding cell in an edge-centered tile.
    before = jnp.full((5, 5), 2, jnp.uint8)
    after = before.at[2, 2].set(0).at[2, 4].set(1).at[0, 0].set(0)
    # When: selecting labels with zero unchanged sampling probability.
    mask = local_change_sample_mask(
        before, after, jnp.array([0, 0]), jnp.array([2, 3]), 2, jax.random.PRNGKey(9), 0
    )
    # Then: both changed valid cells are retained and padding never enters the loss.
    expected = np.zeros((5, 5), bool)
    expected[2, 2] = expected[2, 4] = True
    np.testing.assert_array_equal(mask, expected)


def test_unchanged_sample_mask_reproduces_when_event_key_fixed() -> None:
    # Given: unchanged labels, one actual map change, and one fixed event-specific sample stream.
    before = jnp.full((11, 11), 2, jnp.uint8)
    after = before.at[5, 5].set(0)
    key = jax.random.PRNGKey(73)
    # When: rebuilding the compact supervision mask twice from the same event cutoff.
    first = local_change_sample_mask(
        before, after, jnp.array([5, 5]), jnp.array([11, 11]), 5, key, 0.1
    )
    second = local_change_sample_mask(
        before, after, jnp.array([5, 5]), jnp.array([11, 11]), 5, key, 0.1
    )
    # Then: changed labels are certain, while unchanged labels are a reproducible strict subset.
    np.testing.assert_array_equal(first, second)
    assert bool(first[5, 5])
    assert 1 < int(first.sum()) < 81  # noqa: PLR2004
