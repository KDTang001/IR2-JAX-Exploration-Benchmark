"""JAX implementation of the original IR2 staircase ray sensor."""

from functools import lru_cache
from itertools import product

import jax.numpy as jnp
import numpy as np
from jax import Array

from ir2_benchmark.sensing.jax_sensor import SensorResult


@lru_cache(maxsize=8)
def _geometry(radius: float, height: int, width: int) -> tuple[np.ndarray, ...]:
    """Cache rays and position-dependent NumPy endpoint rounding on the host."""
    if not np.isfinite(radius) or radius < 0:
        message = "Sensor radius must be finite and nonnegative"
        raise ValueError(message)
    angle = 0.0
    deltas = []
    while angle < 2 * np.pi:
        deltas.append((np.cos(angle) * radius, np.sin(angle) * radius))
        angle += 0.5 / 180 * np.pi
    delta = np.asarray(deltas)
    lower = np.floor(delta).astype(np.int32)
    # Rounding absolute endpoints can differ from rounding relative offsets:
    # adding an integer can erase the last few bits near a half-integer.
    x = np.arange(width)
    y = np.arange(height)
    choose_x = (np.rint(x[:, None] + delta[:, 0]) - x[:, None] - lower[:, 0]).astype(np.int32)
    choose_y = (np.rint(y[:, None] + delta[:, 1]) - y[:, None] - lower[:, 1]).astype(np.int32)
    endpoints = lower[:, None, None, :] + np.asarray(
        list(product((0, 1), repeat=2)), dtype=np.int32
    ).reshape(1, 2, 2, 2)
    unique, indices = np.unique(endpoints.reshape(-1, 2), axis=0, return_inverse=True)
    length = max(1, int(np.abs(unique).sum(axis=1).max()))
    offsets = np.zeros((len(unique), length, 2), dtype=np.int32)
    mask = np.zeros((len(unique), length), dtype=bool)
    for index, (ex, ey) in enumerate(unique):
        dx, dy = abs(int(ex)), abs(int(ey))
        error = dx - dy
        sx, sy = (1 if ex > 0 else -1), (1 if ey > 0 else -1)
        px, py = 0, 0
        for step in range(dx + dy):
            offsets[index, step] = (px, py)
            mask[index, step] = True
            if error > 0:
                px += sx
                error -= 2 * dy
            else:
                py += sy
                error += 2 * dx
    return offsets, mask, indices.reshape(-1, 2, 2), choose_x, choose_y


def sense_ir2(
    truth: Array,
    local_map: Array,
    position: Array,
    radius: float,
    valid_shape: tuple[int, int] | Array | None = None,
) -> SensorResult:
    """Match ``sensor.sensor_work`` for integer poses; radius must be static.

    Truth uses 0=free, 1=obstacle; local maps additionally use 2=unknown.
    Each ray reveals up to nine obstacle cells, stopping before the tenth
    obstacle, the first subsequent free cell, its endpoint, or the map border.
    """
    height, width = truth.shape
    offsets, masks, choices, choose_x, choose_y = _geometry(radius, height, width)
    pose = jnp.asarray(position, dtype=jnp.int32)
    ray = jnp.arange(len(choices))
    cx = jnp.asarray(choose_x)[jnp.clip(pose[0], 0, width - 1)]
    cy = jnp.asarray(choose_y)[jnp.clip(pose[1], 0, height - 1)]
    selected = jnp.asarray(choices)[ray, cx, cy]
    cells = jnp.asarray(offsets)[selected] + pose
    x, y = cells[..., 0], cells[..., 1]
    limits = jnp.asarray(truth.shape if valid_shape is None else valid_shape)
    inside = (x >= 0) & (x < limits[1]) & (y >= 0) & (y < limits[0])
    occupied = truth[jnp.clip(y, 0, height - 1), jnp.clip(x, 0, width - 1)] == 1
    collisions = jnp.cumsum(occupied, axis=1)
    stop = ~inside | (collisions >= 10) | (~occupied & (collisions > 0))  # noqa: PLR2004 - legacy wall penetration limit
    revealed = jnp.asarray(masks)[selected] & (jnp.cumsum(stop, axis=1) == 0)
    indices = jnp.where(revealed, y * width + x, height * width)
    visible = (
        jnp.zeros(height * width, dtype=bool)
        .at[indices.reshape(-1)]
        .set(revealed.reshape(-1), mode="drop")
        .reshape(truth.shape)
    )
    return SensorResult(jnp.where(visible, truth, local_map), visible, visible & (truth == 0))
