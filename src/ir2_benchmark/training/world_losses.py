"""Masked posterior fitting objectives for the event world model."""

from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jax import Array

from ir2_benchmark.config import HostConfig
from ir2_benchmark.models.world_model import (
    RSSMState,
    WorldObservation,
    WorldPrediction,
    initial_rssm,
)
from ir2_benchmark.training.b_state import BTrainingConfig
from ir2_benchmark.training.imagination import local_tile
from ir2_benchmark.training.replay import ReplayError, ReplaySequence, read_field
from ir2_benchmark.types import EnvState

MAP_SAMPLE_STREAM: Final = 0x4D41534B


class WorldTarget(NamedTuple):
    """Real measured labels are loss-only inputs, including true termination versus truncation."""

    local_states: Array
    local_mask: Array
    movement: Array
    delivered: Array
    delivery_mask: Array
    reward: Array
    terminated: Array
    valid: Array


class WorldLoss(NamedTuple):
    """Each component is normalized over valid labels, including empty-mask cases."""

    total: Array
    dynamics_kl: Array
    representation_kl: Array
    local: Array
    movement: Array
    delivery: Array
    reward: Array
    continuation: Array


def masked_mean(value: Array, mask: Array) -> Array:
    """Exclude padding exactly; even an entirely empty batch contributes finite zero."""
    selected = jnp.where(mask, value, 0)
    return jnp.sum(selected) / jnp.maximum(jnp.sum(mask), 1)


def local_change_sample_mask(  # noqa: PLR0913, PLR0917
    before: Array,
    after: Array,
    position: Array,
    valid_shape: Array,
    sensor_radius: float,
    key: Array,
    unchanged_rate: float,
) -> Array:
    """Select every changed state and a seeded unchanged subset within the valid sensing circle.

    Tiles are aligned to the actual event-end position. Comparing the persistent
    event-before map includes paid map updates as well as physical observations.
    Observation-time refreshes with no state change remain unchanged labels.
    """
    radius = before.shape[0] // 2
    yy, xx = jnp.indices(before.shape)
    x, y = position[0] + xx - radius, position[1] + yy - radius
    valid = (
        (x >= 0)
        & (x < valid_shape[1])
        & (y >= 0)
        & (y < valid_shape[0])
        & ((xx - radius) ** 2 + (yy - radius) ** 2 <= sensor_radius**2)
    )
    probability = jnp.asarray(unchanged_rate, dtype=jnp.float32)
    unchanged = jax.random.bernoulli(key, probability, before.shape)
    return valid & ((before != after) | unchanged)


def _categorical_kl(left: Array, right: Array) -> Array:
    """Sum independent category KLs into one event-level divergence."""
    log_left = jax.nn.log_softmax(left, -1)
    log_right = jax.nn.log_softmax(right, -1)
    return jnp.sum(jnp.exp(log_left) * (log_left - log_right), axis=(-1, -2))


def _cross_entropy(logits: Array, label: Array) -> Array:
    """Integer labels avoid allocating a one-hot map target."""
    return -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), label[..., None], -1)[..., 0]


def _binary_loss(logits: Array, labels: Array) -> Array:
    """Numerically stable binary likelihood for directed delivery/continuation labels."""
    return jax.nn.softplus(logits) - labels.astype(jnp.float32) * logits


def world_loss(
    prediction: WorldPrediction, target: WorldTarget, free_nats: float = 1.0
) -> WorldLoss:
    """Balance prior dynamics/representation KL; time truncation never labels continuation false."""
    valid = target.valid
    dynamics = _categorical_kl(
        jax.lax.stop_gradient(prediction.posterior_logits), prediction.prior_logits
    )
    representation = _categorical_kl(
        prediction.posterior_logits, jax.lax.stop_gradient(prediction.prior_logits)
    )
    dynamics = masked_mean(jnp.maximum(dynamics, free_nats), valid)
    representation = masked_mean(jnp.maximum(representation, free_nats), valid)
    local = masked_mean(
        _cross_entropy(prediction.local_logits, target.local_states),
        target.local_mask & valid[..., None, None],
    )
    movement = masked_mean(
        _cross_entropy(prediction.movement_logits, jnp.maximum(target.movement, 0)),
        valid & (target.movement >= 0),
    )
    delivery = masked_mean(
        _binary_loss(prediction.delivery_logits, target.delivered),
        target.delivery_mask & valid[..., None],
    )
    reward = masked_mean((prediction.reward - target.reward) ** 2, valid)
    continuation = masked_mean(
        _binary_loss(prediction.continuation_logit, ~target.terminated), valid
    )
    total = (
        0.8 * dynamics + 0.2 * representation + local + movement + delivery + reward + continuation
    )
    return WorldLoss(
        total, dynamics, representation, local, movement, delivery, reward, continuation
    )


class WorldBatch(NamedTuple):
    """Compact event windows and small central features; no [time,N,H,W] device tensor."""

    actions: Array
    observations: WorldObservation
    targets: WorldTarget
    valid: Array
    loss_mask: Array


def _world_observation(
    state: EnvState, agent: Array, radius: int, config: HostConfig
) -> WorldObservation:
    """Central replay measurement is model-only; true positions never enter actor_features."""
    scale = jnp.maximum(state.valid_shape[::-1] - 1, 1)
    robots = jnp.concatenate(
        (
            state.positions / scale,
            (state.balances / max(config.initial_budget, 1))[:, None],
            state.active_mask[:, None],
        ),
        -1,
    )
    coverage = jnp.sum(state.covered & state.component) / jnp.sum(state.component)
    return WorldObservation(
        jax.nn.one_hot(local_tile(state.local_maps[agent], state.positions[agent], radius), 3),
        robots,
        jnp.asarray([coverage, state.round / config.max_rounds]),
    )


def _world_batch(sequence: ReplaySequence, radius: int, config: HostConfig) -> WorldBatch:
    """Decode only measured actor-centered tiles from host metadata into the training sequence."""
    transitions = [entry for entry in sequence.transitions if entry is not None]
    if not transitions:
        raise ReplayError("sampled_sequence_has_no_real_transitions")  # noqa: EM101
    first = transitions[0].after_fields
    side = 2 * radius + 1
    actions, tiles, robots, globals_, moves, deliveries, rewards, terminated = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    masks = []
    for transition in sequence.transitions:
        frame = transition.after_fields if transition is not None else first
        actions.append(
            transition.action
            if transition is not None
            else np.zeros_like(sequence.transitions[0].action)
        )
        tiles.append(
            read_field(frame, "world_tile")
            if transition is not None
            else np.full((side, side), 2, np.int32)
        )
        masks.append(
            read_field(frame, "world_local_mask")
            if transition is not None
            else np.zeros((side, side), bool)
        )
        robots.append(read_field(frame, "world_robots"))
        globals_.append(read_field(frame, "world_global"))
        moves.append(
            -1
            if config.navigation_mode != "grid_path"
            else int(read_field(frame, "world_movement"))
            if transition is not None
            else 0
        )
        deliveries.append(
            read_field(frame, "world_delivered")
            if transition is not None
            else np.zeros(config.n_slots, bool)
        )
        rewards.append(transition.reward if transition is not None else 0)
        terminated.append(transition.terminated if transition is not None else False)
    valid = jnp.asarray(sequence.valid)
    labels = jnp.asarray(np.stack(tiles), jnp.int32)
    target = WorldTarget(
        labels,
        jnp.asarray(np.stack(masks)),
        jnp.asarray(moves),
        jnp.asarray(np.stack(deliveries)),
        jnp.broadcast_to(
            jnp.arange(config.n_slots) < config.n_agents, (len(actions), config.n_slots)
        ),
        jnp.asarray(rewards),
        jnp.asarray(terminated),
        jnp.asarray(sequence.loss_mask),
    )
    return WorldBatch(
        jnp.asarray(np.stack(actions)),
        WorldObservation(
            jax.nn.one_hot(labels, 3),
            jnp.asarray(np.stack(robots)),
            jnp.asarray(np.stack(globals_)),
        ),
        target,
        valid,
        jnp.asarray(sequence.loss_mask),
    )


def posterior_sequence(
    world: TrainState, batch: WorldBatch, key: Array, options: BTrainingConfig
) -> tuple[WorldLoss, WorldPrediction]:
    """Burn in on measured events, then fit the action-conditioned recurrent posterior sequence."""
    initial = initial_rssm(
        deterministic=options.hidden, categories=options.categories, classes=options.classes
    )

    def event(previous: RSSMState, index: Array) -> tuple[RSSMState, WorldPrediction]:
        observation = jax.tree_util.tree_map(lambda value: value[index], batch.observations)
        prediction = world.apply_fn(
            {"params": world.params},
            previous,
            batch.actions[index],
            observation,
            jax.random.fold_in(key, index),
        )
        latent = jax.tree_util.tree_map(
            lambda new, old: jnp.where(batch.valid[index], new, old), prediction.state, previous
        )
        latent = jax.tree_util.tree_map(
            lambda value: jnp.where(index < options.burn_in, jax.lax.stop_gradient(value), value),
            latent,
        )
        return latent, prediction._replace(state=latent)

    _, predictions = jax.lax.scan(event, initial, jnp.arange(batch.actions.shape[0]))
    return world_loss(predictions, batch.targets), predictions


def update_world(
    world: TrainState, batches: tuple[WorldBatch, ...], key: Array, options: BTrainingConfig
) -> tuple[TrainState, Array]:
    """Accumulate the declared logical batch using physical microbatch one."""
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *batches)

    def microbatch(
        carry: tuple[FrozenDict, Array], item: tuple[Array, WorldBatch]
    ) -> tuple[tuple[FrozenDict, Array], None]:
        gradient, total = carry
        index, batch = item

        def objective(params: FrozenDict) -> Array:
            loss, _ = posterior_sequence(
                world.replace(params=params), batch, jax.random.fold_in(key, index), options
            )
            return loss.total

        value, current = jax.value_and_grad(objective)(world.params)
        gradient = jax.tree_util.tree_map(
            lambda accumulated, update: accumulated + update / len(batches), gradient, current
        )
        return (gradient, total + value / len(batches)), None

    initial = (jax.tree_util.tree_map(jnp.zeros_like, world.params), jnp.float32(0))
    (gradient, total), _ = jax.lax.scan(microbatch, initial, (jnp.arange(len(batches)), stacked))
    return world.apply_gradients(grads=gradient), total
