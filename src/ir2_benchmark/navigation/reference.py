"""Independent deque BFS oracle; all route storage is bounded by map area."""

from collections import deque
from typing import Final, NamedTuple

import numpy as np
from numpy.typing import NDArray

NEIGHBORS: Final = ((0, -1), (1, 0), (0, 1), (-1, 0))


class SearchTree(NamedTuple):
    """Distances (-1 when unreachable) and flattened predecessor indices."""

    distances: NDArray[np.int32]
    parents: NDArray[np.int32]


class PathResult(NamedTuple):
    """Route includes its source; unused positions repeat the terminal position."""

    path: NDArray[np.int32]
    length: int
    valid: bool


class NavigationResult(NamedTuple):
    """Executed endpoint and a snapshot of future own-action positions."""

    position: NDArray[np.int32]
    distance: int
    invalid_action: bool
    goal: NDArray[np.int32]
    path: NDArray[np.int32]
    path_length: int
    plan: NDArray[np.int32]
    plan_valid_length: int


def bfs(known_map: NDArray[np.uint8], start: NDArray[np.int32]) -> SearchTree:
    """Search only known-free cells, discovering each cell once in fixed neighbor order."""
    height, width = known_map.shape
    sx, sy = int(start[0]), int(start[1])
    distances = np.full(known_map.shape, -1, dtype=np.int32)
    parents = np.full(known_map.size, -1, dtype=np.int32)
    if not (0 <= sx < width and 0 <= sy < height and known_map[sy, sx] == 0):
        return SearchTree(distances, parents)
    queue = deque([(sx, sy)])
    distances[sy, sx] = 0
    parents[sy * width + sx] = sy * width + sx
    while queue:
        x, y = queue.popleft()
        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and known_map[ny, nx] == 0
                and distances[ny, nx] < 0
            ):
                distances[ny, nx] = distances[y, x] + 1
                parents[ny * width + nx] = y * width + x
                queue.append((nx, ny))
    return SearchTree(distances, parents)


def shortest_path(
    known_map: NDArray[np.uint8],
    start: NDArray[np.int32],
    goal: NDArray[np.int32],
    tree: SearchTree | None = None,
) -> PathResult:
    """Return the full exact path, or a masked zero-distance wait when unreachable."""
    search = bfs(known_map, start) if tree is None else tree
    height, width = known_map.shape
    gx, gy = int(goal[0]), int(goal[1])
    valid = 0 <= gx < width and 0 <= gy < height and search.distances[gy, gx] >= 0
    length = int(search.distances[gy, gx]) if valid else 0
    path = np.tile(goal if valid else start, (known_map.size, 1)).astype(np.int32)
    if valid:
        node = gy * width + gx
        for index in range(length, -1, -1):
            path[index] = (node % width, node // width)
            node = int(search.parents[node])
    return PathResult(path, length, bool(valid))


def navigate(  # noqa: PLR0913, PLR0917
    known_map: NDArray[np.uint8],
    start: NDArray[np.int32],
    goal: NDArray[np.int32],
    max_distance: int,
    plan_horizon: int,
    tree: SearchTree | None = None,
) -> NavigationResult:
    """Follow at most max_distance edges and forecast repeated capped movement."""
    route = shortest_path(known_map, start, goal, tree)
    distance = min(route.length, max_distance)
    indices = np.minimum(np.arange(1, plan_horizon + 1) * max_distance, route.length)
    valid_length = min(plan_horizon, (route.length + max_distance - 1) // max_distance)
    return NavigationResult(
        route.path[distance],
        distance,
        not route.valid,
        np.asarray(goal if route.valid else start, dtype=np.int32),
        route.path,
        route.length,
        route.path[indices],
        valid_length,
    )
