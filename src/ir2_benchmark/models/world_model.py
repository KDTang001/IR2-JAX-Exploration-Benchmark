"""Action-conditioned categorical RSSM; centralized latent never enters the local actor."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from ir2_benchmark.models.encoders import MapEncoder


class RSSMState(NamedTuple):
    """Deterministic recurrent state and factored one-hot stochastic state."""

    deterministic: Array
    stochastic: Array


class WorldObservation(NamedTuple):
    """Training-only posterior measurement, built from an actual replay event's after state."""

    local_tile: Array
    robot_features: Array
    global_features: Array


class WorldPrediction(NamedTuple):
    """Local map decoder, legal path progress distribution, directed delivery, and reward."""

    state: RSSMState
    prior_logits: Array
    posterior_logits: Array
    local_logits: Array
    movement_logits: Array
    delivery_logits: Array
    reward: Array
    continuation_logit: Array


def initial_rssm(
    batch_shape: tuple[int, ...] = (),
    deterministic: int = 256,
    categories: int = 16,
    classes: int = 16,
) -> RSSMState:
    """Initialize unknown central dynamics without injecting actual future state."""
    return RSSMState(
        jnp.zeros((*batch_shape, deterministic), jnp.float32),
        jax.nn.one_hot(jnp.zeros((*batch_shape, categories), jnp.int32), classes),
    )


def latent_features(state: RSSMState) -> Array:
    """Central critic/model features; this function is never an actor feature adapter."""
    return jnp.concatenate(
        (state.deterministic, state.stochastic.reshape(*state.stochastic.shape[:-2], -1)), -1
    )


def _sample(logits: Array, key: Array) -> Array:
    """Straight-through categorical latent used by model learning, not by policy actions."""
    probabilities = jax.nn.softmax(logits, -1)
    sample = jax.nn.one_hot(jax.random.categorical(key, logits), logits.shape[-1])
    return probabilities + jax.lax.stop_gradient(sample - probabilities)


class WorldModel(nn.Module):
    """One ordered robot-event transition, shared between posterior fitting and imagination."""

    n_agents: int
    tile_side: int
    max_distance: int
    deterministic: int = 256
    categories: int = 16
    classes: int = 16

    def setup(self) -> None:
        """Use the same recurrent transition and decoder in measured and imagined paths."""
        self.action_encoder = nn.Dense(self.deterministic, precision=jax.lax.Precision.HIGHEST)
        self.recurrent = nn.GRUCell(self.deterministic)
        self.prior_head = nn.Dense(
            self.categories * self.classes, precision=jax.lax.Precision.HIGHEST
        )
        self.map_encoder = MapEncoder(128)
        self.observation_encoder = nn.Dense(self.deterministic, precision=jax.lax.Precision.HIGHEST)
        self.posterior_head = nn.Dense(
            self.categories * self.classes, precision=jax.lax.Precision.HIGHEST
        )
        self.decoder = nn.Dense(self.deterministic, precision=jax.lax.Precision.HIGHEST)
        self.local_head = nn.Dense(
            self.tile_side * self.tile_side * 3, precision=jax.lax.Precision.HIGHEST
        )
        self.motion_head = nn.Dense(self.max_distance + 1, precision=jax.lax.Precision.HIGHEST)
        self.delivery_head = nn.Dense(self.n_agents, precision=jax.lax.Precision.HIGHEST)
        self.reward_head = nn.Dense(1, precision=jax.lax.Precision.HIGHEST)
        self.continuation_head = nn.Dense(1, precision=jax.lax.Precision.HIGHEST)

    def _transition(self, previous: RSSMState, action: Array) -> tuple[Array, Array]:
        features = jnp.concatenate(
            (previous.stochastic.reshape(*previous.stochastic.shape[:-2], -1), action), -1
        )
        # One scoped policy covers Flax 0.10.2 GRU internals in posterior and imagined paths.
        with jax.default_matmul_precision("highest"):
            deterministic, _ = self.recurrent(
                previous.deterministic, nn.silu(self.action_encoder(features))
            )
        logits = self.prior_head(deterministic).reshape(
            *deterministic.shape[:-1], self.categories, self.classes
        )
        return deterministic, logits

    def _decode(self, state: RSSMState, prior: Array, posterior: Array) -> WorldPrediction:
        hidden = nn.silu(self.decoder(latent_features(state)))
        local = self.local_head(hidden).reshape(
            *hidden.shape[:-1], self.tile_side, self.tile_side, 3
        )
        return WorldPrediction(
            state,
            prior,
            posterior,
            local,
            self.motion_head(hidden),
            self.delivery_head(hidden),
            self.reward_head(hidden)[..., 0],
            self.continuation_head(hidden)[..., 0],
        )

    def __call__(
        self, previous: RSSMState, action: Array, observation: WorldObservation, key: Array
    ) -> WorldPrediction:
        """Fit a posterior using replay measurements; targets still only enter the loss."""
        deterministic, prior = self._transition(previous, action)
        encoded = jnp.concatenate(
            (
                self.map_encoder(observation.local_tile),
                observation.robot_features.reshape(*deterministic.shape[:-1], -1),
                observation.global_features,
            ),
            -1,
        )
        measurement = nn.silu(self.observation_encoder(encoded))
        posterior = self.posterior_head(jnp.concatenate((deterministic, measurement), -1))
        posterior = posterior.reshape(*deterministic.shape[:-1], self.categories, self.classes)
        return self._decode(RSSMState(deterministic, _sample(posterior, key)), prior, posterior)

    def imagine(self, previous: RSSMState, action: Array, key: Array) -> WorldPrediction:
        """Predict from history/action only: no truth, posterior observation, or future fault input."""  # noqa: E501
        deterministic, prior = self._transition(previous, action)
        return self._decode(RSSMState(deterministic, _sample(prior, key)), prior, prior)


class ImaginedCritic(nn.Module):
    """Centralized value head for model-based updates; independent of the decentralized actor."""

    hidden: int = 256

    @nn.compact
    def __call__(self, features: Array) -> Array:
        """Regress centralized event value from frozen model features."""
        hidden = nn.silu(nn.Dense(self.hidden, precision=jax.lax.Precision.HIGHEST)(features))
        hidden = nn.silu(nn.Dense(self.hidden, precision=jax.lax.Precision.HIGHEST)(hidden))
        return nn.Dense(1, precision=jax.lax.Precision.HIGHEST)(hidden)[..., 0]
