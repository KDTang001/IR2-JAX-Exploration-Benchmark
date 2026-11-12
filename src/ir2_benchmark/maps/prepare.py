"""Preserve IR2 occupancy and select connected, reproducible spawn positions."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

FREE: Final = 0
OCCUPIED: Final = 1
UNKNOWN: Final = 2
ANCHOR_ALGORITHM: Final = "208-nearest-centroid-row-major-v1"
Scalar = TypeVar("Scalar", bound=np.generic)


@dataclass(frozen=True, slots=True)
class MapError(ValueError):
    """A concrete source map cannot meet the map contract."""

    source: str
    reason: str

    def __str__(self) -> str:
        """Describe the concrete typed failure."""
        return f"{self.source}: {self.reason}"


def immutable(array: NDArray[Scalar]) -> NDArray[Scalar]:
    """Copy onto an immutable bytes buffer, including against setflags(write=True)."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class MapData:
    """Unpadded binary occupancy [y,x] with an anchor in (x,y) coordinates."""

    occupancy: NDArray[np.uint8]
    anchor: tuple[int, int]
    source: str = "generated"

    def __post_init__(self) -> None:
        """Validate and freeze this value at its input boundary."""
        grid = np.asarray(self.occupancy)
        if grid.ndim != 2 or not grid.size or not np.all((grid == FREE) | (grid == OCCUPIED)):  # noqa: PLR2004
            raise MapError(self.source, "invalid_binary_occupancy")
        x, y = self.anchor
        h, w = grid.shape
        if not (0 <= x < w and 0 <= y < h) or grid[y, x] != FREE:
            raise MapError(self.source, "anchor_not_free_or_outside")
        object.__setattr__(self, "occupancy", immutable(grid.astype(np.uint8)))

    @property
    def valid_shape(self) -> tuple[int, int]:
        """Return the original (height,width), never a resized geometry."""
        return self.occupancy.shape


@dataclass(frozen=True, slots=True)
class PreparedMap:
    """Padded map and anchor-connected scoring component, all immutable."""

    occupancy: NDArray[np.uint8]
    valid_mask: NDArray[np.bool_]
    component: NDArray[np.bool_]
    anchor: tuple[int, int]
    valid_shape: tuple[int, int]


def read_ir2_pixels(path: str | Path) -> NDArray[np.uint8]:
    """Read grayscale source pixels so rejected marker metadata can still be audited."""
    source = str(path)
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise MapError(source, f"corrupt_image:{error}") from error
    return immutable(pixels)


def load_ir2_map(path: str | Path) -> MapData:
    """Decode IR2's >150 free rule and choose a marker pixel nearest its centroid."""
    return decode_ir2_pixels(read_ir2_pixels(path), str(path))


def decode_ir2_pixels(pixels: NDArray[np.uint8], source: str) -> MapData:
    """Decode one already-read grayscale image so audits hash exactly the same source pixels."""
    marker_yx = np.argwhere(pixels == 208)  # noqa: PLR2004
    if not len(marker_yx):
        raise MapError(source, "missing_marker_208")
    distance = np.sum((marker_yx - marker_yx.mean(axis=0)) ** 2, axis=1)
    y, x = marker_yx[int(np.argmin(distance))]
    return MapData((pixels <= 150).astype(np.uint8), (int(x), int(y)), source)  # noqa: PLR2004


def connected_component(grid: NDArray[np.uint8], anchor: tuple[int, int]) -> NDArray[np.bool_]:
    """Flood-fill four-neighbor free cells using horizontal spans instead of per-cell queues."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    pending = deque([anchor])
    while pending:
        x, y = pending.pop()
        if visited[y, x] or grid[y, x] != FREE:
            continue
        left = x
        right = x
        while left > 0 and grid[y, left - 1] == FREE and not visited[y, left - 1]:
            left -= 1
        while right + 1 < w and grid[y, right + 1] == FREE and not visited[y, right + 1]:
            right += 1
        visited[y, left : right + 1] = True
        for neighbor_y in (y - 1, y + 1):
            if 0 <= neighbor_y < h:
                available = (grid[neighbor_y, left : right + 1] == FREE) & ~visited[
                    neighbor_y, left : right + 1
                ]
                starts = np.flatnonzero(available & ~np.r_[False, available[:-1]])
                pending.extend((left + int(offset), neighbor_y) for offset in starts)
    return immutable(visited)


def prepare_map(data: MapData, buckets: tuple[int, ...] = (512, 768, 1024)) -> PreparedMap:
    """Pad only right/bottom with occupied cells; oversize inputs get an explicit extra bucket."""
    h, w = data.valid_shape
    required = max(h, w)
    eligible = [side for side in buckets if side >= required]
    side = min(eligible) if eligible else required
    occupancy = np.full((side, side), OCCUPIED, dtype=np.uint8)
    occupancy[:h, :w] = data.occupancy
    valid = np.zeros((side, side), dtype=np.bool_)
    valid[:h, :w] = True
    component = np.zeros((side, side), dtype=np.bool_)
    component[:h, :w] = connected_component(data.occupancy, data.anchor)
    return PreparedMap(
        immutable(occupancy), immutable(valid), immutable(component), data.anchor, (h, w)
    )


def select_spawns(
    data: MapData,
    n_agents: int,
    seed: int,
    initial_radius: int = 10,
) -> NDArray[np.int32]:
    """Sample different connected free cells; grow the Euclidean anchor radius only as needed."""
    if not 1 <= n_agents <= 6 or initial_radius < 0:  # noqa: PLR2004
        raise MapError(data.source, "invalid_spawn_parameters")
    yx = np.argwhere(connected_component(data.occupancy, data.anchor))
    if len(yx) < n_agents:
        raise MapError(data.source, f"insufficient_component:{len(yx)}<{n_agents}")
    xy = yx[:, ::-1]
    squared = np.sum((xy - np.asarray(data.anchor)) ** 2, axis=1)
    needed_radius = int(np.ceil(np.sqrt(np.partition(squared, n_agents - 1)[n_agents - 1])))
    radius = max(initial_radius, needed_radius)
    candidates = xy[squared <= radius**2]
    chosen = np.random.default_rng(seed).choice(len(candidates), size=n_agents, replace=False)
    return immutable(candidates[chosen].astype(np.int32))
