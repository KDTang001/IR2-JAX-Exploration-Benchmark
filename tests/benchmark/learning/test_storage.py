"""Sampling tables are part of the optimized distribution and cannot be replaced."""

# Independent replay mixture/count expectations must remain literal oracle values.
# ruff: noqa: PLR2004

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from flax.training.train_state import TrainState

from ir2_benchmark.agents.actor import Actor, IndexAction
from ir2_benchmark.agents.critic import Critic, PrivilegedFeatures
from ir2_benchmark.models.encoders import ActorFeatures
from ir2_benchmark.models.plan_predictor import PlanInput, initial_plan_memory
from ir2_benchmark.training.collect_a import PlanTrainingBatch
from ir2_benchmark.training.plan_losses import PlanTarget
from ir2_benchmark.training.pools_a import _pack_booleans, _unpack_booleans
from ir2_benchmark.training.ppo import (
    Returns,
    bounded_diagnostics,
    event_gae,
    optimizer,
    ppo_diagnostics,
    ppo_update,
)
from ir2_benchmark.training.predictor_replay import empty_predictor_replay, sample_and_store
from ir2_benchmark.training.storage import BatchError, RolloutBatch, validate_batch


def _batch() -> RolloutBatch:
    candidates = jnp.asarray([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])
    mask = jnp.ones((2, 2), bool)
    features = ActorFeatures(
        jnp.zeros((2, 9, 9, 3)), candidates, mask, jnp.zeros((2, 6)), jnp.zeros((2, 2, 8)), mask
    )
    critic = PrivilegedFeatures(jnp.zeros((2, 9, 9, 3)), jnp.zeros((2, 2, 4)), jnp.zeros((2, 2)))
    return RolloutBatch(
        features,
        critic,
        jnp.zeros((2, 16)),
        jnp.zeros((2, 13)),
        IndexAction(jnp.asarray([0, 1]), jnp.zeros(2, bool), jnp.zeros(2, jnp.int32)),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([False, True]),
        jnp.zeros(2, bool),
        jnp.ones(2, bool),
        jnp.asarray([0, 1]),
        jnp.asarray([3, 3]),
        candidates,
        mask,
    )


def test_batch_when_update_reuses_exact_sampled_features() -> None:
    # Given: observations include their sampled candidates, masks and predictor version.
    batch = _batch()
    # When: the unchanged batch crosses the PPO update boundary.
    result = validate_batch(batch)
    # Then: the original frozen sampling contract is accepted.
    assert result is None


def test_batch_when_candidate_table_is_rebuilt() -> None:
    # Given: an old rollout was sampled with a different waypoint representation.
    batch = _batch()
    changed = batch._replace(
        actor_features=batch.actor_features._replace(
            candidates=batch.actor_features.candidates.at[0, 1, 0].set(7)
        )
    )
    # When / Then: a changed distribution table is rejected before compiled optimization.
    with pytest.raises(BatchError, match="candidates"):
        validate_batch(changed)


def test_batch_when_predictor_feature_versions_change_within_update() -> None:
    # Given: saved features accidentally mix two predictor parameter versions.
    batch = _batch()._replace(predictor_version=jnp.asarray([3, 4]))
    # When / Then: the sampling version contract refuses mixed features.
    with pytest.raises(BatchError, match="predictor_version"):
        validate_batch(batch)


def test_ppo_when_same_parameters_recompute_exact_conditional_action_probabilities() -> None:
    # Given: zero parameters imply uniform two-way motion, gate 1/2 and four message types.
    batch = _batch()
    actor, critic = Actor(hidden=16), Critic(hidden=16)
    actor_params = actor.init(
        jax.random.PRNGKey(1), batch.actor_features, batch.memories, batch.plan_features
    )["params"]
    actor_params = jax.tree.map(jnp.zeros_like, actor_params)
    critic_params = critic.init(jax.random.PRNGKey(2), batch.critic_features)["params"]
    critic_params = jax.tree.map(jnp.zeros_like, critic_params)
    batch = batch._replace(
        actions=batch.actions._replace(send=jnp.asarray([False, True])),
        old_log_probs=jnp.log(jnp.asarray([0.5 * 0.5, 0.5 * 0.5 * 0.25])),
    )
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optimizer())
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optimizer())
    returns = event_gae(
        jnp.asarray([1.0, 4.0]),
        jnp.zeros(3),
        batch.terminated,
        batch.truncated,
        batch.valid,
        0.5,
        1.0,
    )
    # When: actual PPO recomputes saved conditional likelihoods before its first gradient update.
    _, _, metrics = ppo_update(actor_state, critic_state, batch, returns, actor, critic)
    # Then: each independently calculated old action probability yields ratio one.
    np.testing.assert_allclose(metrics.ratio_mean, 1.0, atol=1e-6)


def _intent_rows(event: int) -> PlanTrainingBatch:
    """Use distinct event labels to identify which real compact buffer rows were sampled."""
    return PlanTrainingBatch(
        initial_plan_memory((4, 2), hidden=4, categories=2, classes=2),
        PlanInput(
            jnp.zeros((4, 2, 6)),
            jnp.zeros((4, 2, 20)),
            jnp.zeros((4, 2, 20), bool),
            jnp.zeros((4, 2), jnp.int32),
            jnp.full((4, 2), -1),
            jnp.full((4, 2), -1),
        ),
        PlanTarget(
            jnp.full((4, 2, 2), event / 10),
            jnp.zeros((4, 2, 5, 2)),
            jnp.zeros((4, 2), jnp.int32),
            jnp.ones((4, 2), bool),
            jnp.full((4, 2), event),
        ),
        jax.random.split(jax.random.PRNGKey(event), 4),
    )


def test_bounded_diagnostics_preserve_global_variance_and_mask_padding() -> None:
    base = _batch()
    batch = jax.tree.map(lambda value: value[jnp.asarray([0, 1, 0, 1, 0])], base)
    batch = batch._replace(valid=jnp.asarray([True, False, True, True, True]))
    actor, critic = Actor(hidden=16), Critic(hidden=16)
    actor_params = actor.init(
        jax.random.PRNGKey(1), batch.actor_features, batch.memories, batch.plan_features
    )["params"]
    critic_params = critic.init(jax.random.PRNGKey(2), batch.critic_features)["params"]
    a = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optimizer())
    c = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optimizer())
    returns = Returns(jnp.ones(5), jnp.asarray([2.0, 999.0, 8.0, -3.0, 1.0]))
    expected = ppo_diagnostics(a, c, batch, returns)
    diagnose = jax.jit(lambda a, c, b, r: ppo_diagnostics(a, c, b, r, moments=True))
    actual = bounded_diagnostics(a, c, batch, returns, 2, diagnose)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, rtol=2e-5, atol=2e-6)


def test_large_boolean_checkpoint_roundtrip_is_lossless_and_compact() -> None:
    mask = (np.arange(2**20 + 7) % 3 == 0).reshape(1, -1)
    original = {"nested": {"map": jnp.asarray(mask), "counter": jnp.arange(3)}, "scalar": 7}
    payload = serialization.msgpack_serialize(_pack_booleans(original))
    assert len(payload) < mask.nbytes // 4
    restored = _unpack_booleans(serialization.msgpack_restore(payload))
    np.testing.assert_array_equal(restored["nested"]["map"], mask)
    assert restored["nested"]["map"].dtype == bool
    np.testing.assert_array_equal(restored["nested"]["counter"], np.arange(3))
    assert restored["scalar"] == 7


def test_predictor_replay_when_later_batch_mixes_recent_and_actual_history() -> None:
    # Given: the ring contains the previous event-1 batch, while event-2 targets are new.
    old, recent = _intent_rows(1), _intent_rows(2)
    prototype = jax.tree.map(lambda value: value[0], old)
    replay, _ = sample_and_store(empty_predictor_replay(prototype, 8, jax.random.PRNGKey(7)), old)
    # When: a 1:1 mixture is drawn before inserting the new real rows.
    stored, mixed = sample_and_store(replay, recent, recent_fraction=0.5)
    # Then: half the labels are historical and half current, and both batches are retained.
    assert int(jnp.sum(mixed.target.generated_event[:, 0] == 1)) == 2
    assert int(jnp.sum(mixed.target.generated_event[:, 0] == 2)) == 2
    assert int(stored.size) == 8


def test_predictor_replay_when_restoring_cursor_data_and_random_key() -> None:
    # Given: a filled compact replay checkpoint includes its sampling RNG.
    old, recent = _intent_rows(1), _intent_rows(2)
    template = empty_predictor_replay(
        jax.tree.map(lambda value: value[0], old), 8, jax.random.PRNGKey(7)
    )
    saved, _ = sample_and_store(template, old)
    restored = serialization.from_bytes(template, serialization.to_bytes(saved))
    # When: both resumed and uninterrupted buffers receive the same next batch.
    expected = sample_and_store(saved, recent)
    actual = sample_and_store(restored, recent)
    # Then: history choices, current choices, data and next random state match exactly.
    for first, second in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(first, second)


def test_predictor_replay_when_capacity_wraps_replaces_oldest_compact_rows() -> None:
    # Given: two four-row batches exactly fill the eight-event bounded ring.
    first = _intent_rows(1)
    replay = empty_predictor_replay(
        jax.tree.map(lambda value: value[0], first), 8, jax.random.PRNGKey(7)
    )
    replay, _ = sample_and_store(replay, first)
    replay, _ = sample_and_store(replay, _intent_rows(2))
    # When: a third real compact batch is inserted after sampling its supervision mixture.
    replay, _ = sample_and_store(replay, _intent_rows(3))
    # Then: the capacity stays fixed and only the two newest batches remain stored.
    assert int(replay.size) == 8
    np.testing.assert_array_equal(
        np.sort(np.asarray(replay.data.target.generated_event[:, 0])), [2, 2, 2, 2, 3, 3, 3, 3]
    )
