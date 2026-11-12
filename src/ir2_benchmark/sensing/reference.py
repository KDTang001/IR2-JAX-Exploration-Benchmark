"""Independent NumPy sensor oracle using line-segment/AABB intersections."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class SensorResult(NamedTuple):
    """Updated knowledge and physical sensing footprint, without coverage bookkeeping."""

    local_map: NDArray[np.uint8]
    visible: NDArray[np.bool_]
    observed_free: NDArray[np.bool_]


def _intersects_cell(sx: int, sy: int, tx: int, ty: int, cx: int, cy: int) -> bool:  # noqa: PLR0913, PLR0917
    """Test the closed segment against a closed unit square by slab clipping."""
    enter, leave = 0.0, 1.0
    for source, delta, center in ((sx, tx - sx, cx), (sy, ty - sy, cy)):
        if delta == 0:
            if not center - 0.5 <= source <= center + 0.5:
                return False
        else:
            first = (center - 0.5 - source) / delta
            second = (center + 0.5 - source) / delta
            enter = max(enter, min(first, second))
            leave = min(leave, max(first, second))
            if enter > leave:
                return False
    return True


def visible_mask(
    truth: NDArray[np.uint8],
    position: NDArray[np.int32],
    radius: float,
    valid_shape: tuple[int, int] | None = None,
) -> NDArray[np.bool_]:
    """Reveal target centers in the disk unless another occupied closed cell intersects."""
    height, width = truth.shape if valid_shape is None else valid_shape
    sx, sy = int(position[0]), int(position[1])
    visible = np.zeros(truth.shape, dtype=bool)
    reach = int(np.floor(radius))
    for ty in range(max(0, sy - reach), min(height, sy + reach + 1)):
        for tx in range(max(0, sx - reach), min(width, sx + reach + 1)):
            if (tx - sx) ** 2 + (ty - sy) ** 2 > radius**2:
                continue
            blocked = False
            for cy in range(min(sy, ty), max(sy, ty) + 1):
                for cx in range(min(sx, tx), max(sx, tx) + 1):
                    if (cx, cy) in ((sx, sy), (tx, ty)):
                        continue
                    if truth[cy, cx] == 1 and _intersects_cell(sx, sy, tx, ty, cx, cy):
                        blocked = True
                        break
                if blocked:
                    break
            visible[ty, tx] = not blocked
    return visible


def sense(
    truth: NDArray[np.uint8],
    local_map: NDArray[np.uint8],
    position: NDArray[np.int32],
    radius: float,
    valid_shape: tuple[int, int] | None = None,
) -> SensorResult:
    """Update only physically visible cells, preserving all previous knowledge elsewhere."""
    visible = visible_mask(truth, position, radius, valid_shape)
    return SensorResult(np.where(visible, truth, local_map), visible, visible & (truth == 0))
