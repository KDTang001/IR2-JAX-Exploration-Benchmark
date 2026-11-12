"""Central value network with a distinct privileged feature type."""

from typing import NamedTuple

import jax.numpy as jnp
from flax import linen as nn
from jax import Array, lax

from ir2_benchmark.models.encoders import MapEncoder


class PrivilegedFeatures(NamedTuple):
    """Joint-map channels and true joint robot/time state, never actor inputs."""

    map_channels: Array
    robot_features: Array
    global_features: Array


class Critic(nn.Module):
    """One team value per ordered robot event."""

    hidden: int = 256

    @nn.compact
    def __call__(self, features: PrivilegedFeatures) -> Array:
        """Estimate team return from the separately typed centralized input."""
        encoded = MapEncoder()(features.map_channels)
        robots = features.robot_features.reshape((*features.robot_features.shape[:-2], -1))
        hidden = jnp.concatenate((encoded, robots, features.global_features), -1)
        for _ in range(2):
            hidden = nn.silu(nn.Dense(self.hidden, precision=lax.Precision.HIGHEST)(hidden))
        return nn.Dense(1, precision=lax.Precision.HIGHEST)(hidden)[..., 0]
