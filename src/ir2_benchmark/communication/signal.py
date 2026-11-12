"""Analytic grid-intersection propagation: a new model, not numeric IR2 reproduction."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class SegmentLengths(NamedTuple):
    """Physical lengths along a line, measured in map-coordinate units."""

    distance: Array
    obstacle_distance: Array
    free_distance: Array


class Signal(NamedTuple):
    """Path lengths, received strength and strict-threshold reachability."""

    distance: Array
    obstacle_distance: Array
    free_distance: Array
    strength: Array
    reachable: Array


def segment_lengths(grid: Array, start_xy: Array, end_xy: Array) -> SegmentLengths:
    """Intersect integer-centered cells exactly using O(H+W) grid crossings.

    Endpoints must be inside the validated map. Coincident crossings have zero
    measure, so walls touched only at a corner contribute no obstacle distance.
    """
    height, width = grid.shape
    start = start_xy.astype(jnp.float32)
    delta = end_xy.astype(jnp.float32) - start
    safe_delta = jnp.where(delta != 0, delta, 1)
    tx = (jnp.arange(width - 1, dtype=jnp.float32) + 0.5 - start[0]) / safe_delta[0]
    ty = (jnp.arange(height - 1, dtype=jnp.float32) + 0.5 - start[1]) / safe_delta[1]
    tx = jnp.where(delta[0] != 0, tx, 0)
    ty = jnp.where(delta[1] != 0, ty, 0)
    crossings = jnp.sort(jnp.clip(jnp.concatenate((jnp.asarray([0.0, 1.0]), tx, ty)), 0, 1))
    intervals = crossings[1:] - crossings[:-1]
    midpoints = start + ((crossings[1:] + crossings[:-1]) * 0.5)[:, None] * delta
    cells = jnp.floor(midpoints + 0.5).astype(jnp.int32)
    distance = jnp.linalg.norm(delta)
    obstacle = distance * jnp.sum(intervals * (grid[cells[:, 1], cells[:, 0]] == 1))
    return SegmentLengths(distance, obstacle, distance - obstacle)


# Physical constants are explicit so controlled propagation comparisons remain jittable.
def link_signal(  # noqa: PLR0913, PLR0917
    grid: Array,
    start_xy: Array,
    end_xy: Array,
    transmit_power: float = -20.0,
    reference_loss: float = 31.0,
    gamma_free: float = 2.0,
    gamma_obstacle: float = 4.0,
    reference_distance: float = 35.0,
    threshold: float = -70.0,
) -> Signal:
    """Apply the approved piecewise loss, with no extra random noise."""
    lengths = segment_lengths(grid, start_xy, end_xy)
    obstacle_loss = jnp.where(
        lengths.obstacle_distance > 0,
        10 * gamma_obstacle * jnp.log10(jnp.maximum(lengths.obstacle_distance, 1)),
        0,
    )
    free_loss = jnp.where(
        lengths.free_distance >= reference_distance,
        10
        * gamma_free
        * jnp.log10(jnp.maximum(lengths.free_distance, reference_distance) / reference_distance),
        0,
    )
    strength = transmit_power - reference_loss - obstacle_loss - free_loss
    return Signal(*lengths, strength, strength > threshold)
