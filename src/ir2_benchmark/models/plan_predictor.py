"""Receiver-visible recurrent stochastic intent estimation, without label inputs."""

from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from ir2_benchmark.codecs.packet import Codec
from ir2_benchmark.types import LocalObservation

UNKNOWN: Final = 2


class PlanInput(NamedTuple):
    """Coordinates in measurement are normalized using valid_shape, never padding."""

    context: Array
    measurement: Array
    measurement_mask: Array
    measurement_kind: Array
    age: Array
    last_message_event: Array


class PlanMemory(NamedTuple):
    """A separate state belongs to every receiver-neighbor pair."""

    deterministic: Array
    stochastic: Array
    last_message_event: Array


class PlanEstimate(NamedTuple):
    """Normalized current intention with uncertainty and explicit missingness."""

    goal: Array
    positions: Array
    variance: Array
    length_logits: Array
    last_message_event: Array
    missing: Array


class PlanPrediction(NamedTuple):
    """Prior/posterior logits are training outputs, never privileged inputs."""

    memory: PlanMemory
    estimate: PlanEstimate
    prior_logits: Array
    posterior_logits: Array


class ZOHMemory(NamedTuple):
    """Only a genuine PLAN measurement creates a complete held intention."""

    goal: Array
    positions: Array
    valid_length: Array
    last_message_event: Array
    has_plan: Array


def initial_plan_memory(
    batch_shape: tuple[int, ...] = (), hidden: int = 256, categories: int = 16, classes: int = 16
) -> PlanMemory:
    """Cold starts contain no true neighbor state."""
    return PlanMemory(
        jnp.zeros((*batch_shape, hidden), jnp.float32),
        jnp.zeros((*batch_shape, categories, classes), jnp.float32),
        jnp.full(batch_shape, -1, jnp.int32),
    )


def plan_inputs(observation: LocalObservation) -> PlanInput:
    """Enable measurements only for independently fresh accepted position/goal/plan fields."""
    neighbors = jnp.arange(observation.peer_positions.shape[0])
    plan = observation.fresh_plan
    position = observation.fresh_position
    goal = observation.fresh_goal
    scale = jnp.maximum(observation.valid_shape[::-1] - 1, 1)
    horizon = observation.peer_plans.shape[-2]
    lengths = jax.nn.one_hot(observation.peer_plan_lengths, horizon + 1)
    measurement = jnp.concatenate(
        (
            observation.peer_positions / scale,
            observation.peer_goals / scale,
            (observation.peer_plans / scale).reshape((neighbors.shape[0], -1)),
            lengths,
        ),
        -1,
    )
    mask = jnp.concatenate(
        (
            jnp.repeat(position[:, None], 2, -1),
            jnp.repeat(goal[:, None], 2, -1),
            jnp.repeat(
                (plan[:, None] & (jnp.arange(horizon) < observation.peer_plan_lengths[:, None])),
                2,
                -1,
            ),
            jnp.repeat(plan[:, None], horizon + 1, -1),
        ),
        -1,
    )
    last = jnp.maximum(
        observation.peer_position_event,
        jnp.maximum(observation.peer_goal_event, observation.peer_plan_event),
    )
    context = jnp.concatenate(
        (
            observation.position / scale,
            observation.own_goal / scale,
            jnp.asarray(
                [jnp.log1p(observation.balance), jnp.mean(observation.local_map != UNKNOWN)]
            ),
        )
    )
    return PlanInput(
        jnp.broadcast_to(context, (neighbors.shape[0], context.shape[0])),
        measurement,
        mask,
        jnp.where(
            plan, Codec.PLAN, jnp.where(goal, Codec.GOAL, jnp.where(position, Codec.POSITION, 0))
        ),
        jnp.where(last >= 0, observation.event - last, -1),
        last,
    )


class PlanPredictor(nn.Module):
    """A categorical RSSM whose measurement update is gated by real received fields."""

    horizon: int = 5
    hidden: int = 256
    categories: int = 16
    classes: int = 16
    variance_floor: float = 1e-4

    @nn.compact
    def __call__(self, memory: PlanMemory, inputs: PlanInput, key: Array) -> PlanPrediction:
        """Advance from local context; unavailable measurements leave the prior unchanged."""
        age = jnp.log1p(jnp.maximum(inputs.age, 0))[..., None]
        previous = memory.stochastic.reshape((*memory.stochastic.shape[:-2], -1))
        recurrent_inputs = jnp.concatenate(
            (previous, inputs.context, age, (inputs.age < 0)[..., None]), -1
        )
        # Flax 0.10.2 GRUCell does not expose precision; its internal Dense uses this scope.
        with jax.default_matmul_precision("highest"):
            deterministic, _ = nn.GRUCell(self.hidden)(memory.deterministic, recurrent_inputs)
        shape = (*deterministic.shape[:-1], self.categories, self.classes)
        prior = nn.Dense(
            self.categories * self.classes, name="prior", precision=jax.lax.Precision.HIGHEST
        )(deterministic).reshape(shape)
        measurement = jnp.concatenate(
            (
                jnp.where(inputs.measurement_mask, inputs.measurement, 0),
                inputs.measurement_mask.astype(jnp.float32),
                jax.nn.one_hot(inputs.measurement_kind, 5),
            ),
            -1,
        )
        measurement = nn.silu(
            nn.Dense(128, name="measurement", precision=jax.lax.Precision.HIGHEST)(measurement)
        )
        posterior = nn.Dense(
            self.categories * self.classes, name="posterior", precision=jax.lax.Precision.HIGHEST
        )(jnp.concatenate((deterministic, measurement), -1)).reshape(shape)
        observed = jnp.any(inputs.measurement_mask, -1)
        posterior = jnp.where(observed[..., None, None], posterior, prior)
        probabilities = jax.nn.softmax(posterior, -1)
        sample = jax.nn.one_hot(jax.random.categorical(key, posterior, axis=-1), self.classes)
        stochastic = probabilities + jax.lax.stop_gradient(sample - probabilities)
        latent = jnp.concatenate((deterministic, stochastic.reshape((*shape[:-2], -1))), -1)
        latent = nn.silu(
            nn.Dense(self.hidden, name="decode", precision=jax.lax.Precision.HIGHEST)(latent)
        )
        coordinates = jax.nn.sigmoid(
            nn.Dense(2 * (self.horizon + 1), name="means", precision=jax.lax.Precision.HIGHEST)(
                latent
            )
        )
        coordinates = coordinates.reshape((*shape[:-2], self.horizon + 1, 2))
        variance = jax.nn.softplus(
            nn.Dense(2 * (self.horizon + 1), name="variance", precision=jax.lax.Precision.HIGHEST)(
                latent
            )
        )
        variance = variance.reshape(coordinates.shape) + self.variance_floor
        lengths = nn.Dense(self.horizon + 1, name="lengths", precision=jax.lax.Precision.HIGHEST)(
            latent
        )
        last_event = jnp.where(observed, inputs.last_message_event, memory.last_message_event)
        estimate = PlanEstimate(
            coordinates[..., 0, :],
            coordinates[..., 1:, :],
            variance,
            lengths,
            last_event,
            last_event < 0,
        )
        return PlanPrediction(
            PlanMemory(deterministic, stochastic, last_event), estimate, prior, posterior
        )


def initial_zoh(batch_shape: tuple[int, ...] = (), horizon: int = 5) -> ZOHMemory:
    """Missing plans remain invalid; no hidden spawn coordinates are synthesized."""
    return ZOHMemory(
        jnp.zeros((*batch_shape, 2)),
        jnp.zeros((*batch_shape, horizon, 2)),
        jnp.zeros(batch_shape, jnp.int32),
        jnp.full(batch_shape, -1, jnp.int32),
        jnp.zeros(batch_shape, jnp.bool_),
    )


def update_zoh(  # noqa: PLR0913, PLR0917
    memory: ZOHMemory,
    goal: Array,
    positions: Array,
    valid_length: Array,
    generated_event: Array,
    is_plan: Array,
) -> ZOHMemory:
    """Hold a plan's original timestamp; POSITION cannot manufacture a complete plan."""
    newer = is_plan & ((~memory.has_plan) | (generated_event > memory.last_message_event))
    return ZOHMemory(
        jnp.where(newer[..., None], goal, memory.goal),
        jnp.where(newer[..., None, None], positions, memory.positions),
        jnp.where(newer, valid_length, memory.valid_length),
        jnp.where(newer, generated_event, memory.last_message_event),
        memory.has_plan | newer,
    )
