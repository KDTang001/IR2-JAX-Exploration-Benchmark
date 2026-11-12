"""JAX visibility using cached static supercover geometry, never sampled rays."""

from functools import lru_cache
from math import floor
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray


class SensorGeometry(NamedTuple):
    """Host-precomputed disk centers and every intervening closed cell."""

    target_offsets: NDArray[np.int32]
    blocker_offsets: NDArray[np.int32]
    blocker_mask: NDArray[np.bool_]


class SensorResult(NamedTuple):
    """Knowledge and physical footprint; received knowledge never changes the latter."""

    local_map: Array
    visible: Array
    observed_free: Array


def _supercover(dx: int, dy: int) -> list[tuple[int, int]]:
    """Traverse integer center-to-center segments; exact ties include both side cells."""
    ax, ay = abs(dx), abs(dy)
    stepx, stepy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    x, y = 0, 0
    ix, iy = 0, 0
    cells: list[tuple[int, int]] = []
    while ix < ax or iy < ay:
        # Cross multiplication compares the next half-cell boundary times exactly.
        cross_x = (2 * ix + 1) * ay
        cross_y = (2 * iy + 1) * ax
        if ix == ax or (iy < ay and cross_y < cross_x):
            y += stepy
            iy += 1
        elif iy == ay or cross_x < cross_y:
            x += stepx
            ix += 1
        else:
            cells.extend(((x + stepx, y), (x, y + stepy)))
            x += stepx
            y += stepy
            ix += 1
            iy += 1
        cells.append((x, y))
    return [cell for cell in cells if cell != (dx, dy)]


@lru_cache(maxsize=4)
def build_sensor_geometry(radius: float) -> SensorGeometry:
    """Cache O(R³) relative geometry on the host; pass it as a PyTree to jit/vmap."""
    reach = floor(radius)
    targets = [
        (x, y)
        for y in range(-reach, reach + 1)
        for x in range(-reach, reach + 1)
        if x * x + y * y <= radius * radius
    ]
    # Two passes avoid retaining millions of Python coordinate tuples at R=80.
    length = max(len(_supercover(x, y)) or 1 for x, y in targets)
    offsets = np.zeros((len(targets), length, 2), dtype=np.int32)
    mask = np.zeros((len(targets), length), dtype=bool)
    for index, (x, y) in enumerate(targets):
        ray = _supercover(x, y)
        if ray:
            offsets[index, : len(ray)] = ray
            mask[index, : len(ray)] = True
    centers = np.asarray(targets, dtype=np.int32)
    for array in (centers, offsets, mask):
        array.setflags(write=False)
    return SensorGeometry(centers, offsets, mask)


def visible_mask(
    truth: Array,
    position: Array,
    geometry: SensorGeometry,
    valid_shape: tuple[int, int] | Array | None = None,
) -> Array:
    """Evaluate every disk target's exact blockers in parallel on the device."""
    height, width = truth.shape
    limits = jnp.asarray(truth.shape if valid_shape is None else valid_shape)
    targets = jnp.asarray(geometry.target_offsets) + position
    blockers = jnp.asarray(geometry.blocker_offsets) + position
    bx, by = blockers[..., 0], blockers[..., 1]
    blocker_in_bounds = (bx >= 0) & (bx < limits[1]) & (by >= 0) & (by < limits[0])
    occupied = truth[jnp.clip(by, 0, height - 1), jnp.clip(bx, 0, width - 1)] == 1
    blocked = jnp.any(jnp.asarray(geometry.blocker_mask) & (occupied | ~blocker_in_bounds), axis=1)
    tx, ty = targets[:, 0], targets[:, 1]
    valid = (tx >= 0) & (tx < limits[1]) & (ty >= 0) & (ty < limits[0]) & ~blocked
    # Invalid targets use a dropped sentinel, never clipped real cells.
    indices = jnp.where(valid, ty * width + tx, height * width)
    return (
        jnp.zeros(height * width, dtype=bool)
        .at[indices]
        .set(valid, mode="drop")
        .reshape(truth.shape)
    )


def sense(
    truth: Array,
    local_map: Array,
    position: Array,
    geometry: SensorGeometry,
    valid_shape: tuple[int, int] | Array | None = None,
) -> SensorResult:
    """Write visible truth without changing inputs or forgetting unseen local cells."""
    visible = visible_mask(truth, position, geometry, valid_shape)
    return SensorResult(jnp.where(visible, truth, local_map), visible, visible & (truth == 0))
