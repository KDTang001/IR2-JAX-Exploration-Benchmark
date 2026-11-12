"""Approved fixed communication control, distinct from learned message choices."""

import jax.numpy as jnp
from jax import Array

from ir2_benchmark.codecs.packet import Codec
from ir2_benchmark.navigation.candidates import UNKNOWN
from ir2_benchmark.types import LocalObservation


def fixed_communication(
    observation: LocalObservation, *, plan_every_step: bool = False
) -> tuple[Array, Array]:
    """Use PLAN every action for pretraining, or the benchmark's fourth-action cycle."""
    if plan_every_step:
        return jnp.ones((), bool), jnp.asarray(Codec.PLAN, jnp.int32)
    opportunity = observation.round // 4
    cycle = jnp.array([Codec.PLAN, Codec.MAP_DELTA, Codec.MAP_DELTA, Codec.POSITION])
    codec = cycle[opportunity % 4]
    empty = ~jnp.any((observation.local_map < UNKNOWN) & (observation.observed_event >= 0))
    codec = jnp.where((codec == Codec.MAP_DELTA) & empty, Codec.PLAN, codec)
    return observation.round % 4 == 0, codec
