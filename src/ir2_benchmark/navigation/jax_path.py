"""Exact JAX BFS with a map-area queue and first-discovery predecessor tree."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, lax


class SearchTree(NamedTuple):
    """Full-map graph distances (-1 unreachable) and flattened predecessors."""

    distances: Array
    parents: Array


class PathResult(NamedTuple):
    """Source-inclusive route padded by its endpoint, length measured in edges."""

    path: Array
    length: Array
    valid: Array


class NavigationResult(NamedTuple):
    """Executed movement and the pre-movement intention snapshot."""

    position: Array
    distance: Array
    invalid_action: Array
    goal: Array
    path: Array
    path_length: Array
    plan: Array
    plan_valid_length: Array


class _QueueState(NamedTuple):
    head: Array
    tail: Array
    queue: Array
    distances: Array
    parents: Array


def bfs(known_map: Array, start: Array, goal: Array | None = None) -> SearchTree:
    """Visit each known-free cell once, optionally stopping when goal is discovered.

    Neighbors are expanded up/right/down/left. A tree made with a goal is partial
    and may include the remainder of its FIFO chunk; omit goal when sharing a tree.
    """
    height, width = known_map.shape
    size = height * width
    start = jnp.asarray(start, dtype=jnp.int32)
    sx, sy = start[0], start[1]
    source = jnp.clip(sy, 0, height - 1) * width + jnp.clip(sx, 0, width - 1)
    flat = known_map.reshape(-1)
    valid_start = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height) & (flat[source] == 0)
    distances = jnp.full(size, -1, jnp.int32).at[source].set(jnp.where(valid_start, 0, -1))
    parents = jnp.full(size, -1, jnp.int32).at[source].set(jnp.where(valid_start, source, -1))
    initial = _QueueState(
        jnp.int32(0),
        valid_start.astype(jnp.int32),
        jnp.zeros(size, jnp.int32).at[0].set(source),
        distances,
        parents,
    )
    offsets = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=jnp.int32)
    chunk = min(256, size)
    ranks = jnp.arange(4 * chunk)

    def searching(state: _QueueState) -> Array:
        remaining = state.head < state.tail
        if goal is not None:
            gx, gy = goal[0], goal[1]
            target = jnp.clip(gy, 0, height - 1) * width + jnp.clip(gx, 0, width - 1)
            valid_goal = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height) & (flat[target] == 0)
            remaining &= valid_goal & (state.distances[target] < 0)
        return remaining

    def expand(state: _QueueState) -> _QueueState:
        count = jnp.minimum(chunk, state.tail - state.head)
        nodes = state.queue[jnp.minimum(state.head + jnp.arange(chunk), size - 1)]
        nx = (nodes % width)[:, None] + offsets[:, 0]
        ny = (nodes // width)[:, None] + offsets[:, 1]
        targets = (jnp.clip(ny, 0, height - 1) * width + jnp.clip(nx, 0, width - 1)).reshape(-1)
        discover = (
            (
                (jnp.arange(chunk)[:, None] < count)
                & (nx >= 0)
                & (nx < width)
                & (ny >= 0)
                & (ny < height)
            ).reshape(-1)
            & (flat[targets] == 0)
            & (state.distances[targets] < 0)
        )
        # Stable cell sorting identifies the earliest (FIFO node, U/R/D/L) proposal.
        keys = jnp.where(discover, targets, size)
        order = jnp.argsort(keys, stable=True)
        sorted_keys = keys[order]
        first_sorted = (sorted_keys < size) & (
            (ranks == 0) | (sorted_keys != jnp.roll(sorted_keys, 1))
        )
        first = jnp.zeros(4 * chunk, bool).at[order].set(first_sorted)
        prefix = jnp.cumsum(first, dtype=jnp.int32)
        slots = jnp.where(first, state.tail + prefix - 1, size)
        unique_targets = jnp.where(first, targets, size)
        parents = jnp.broadcast_to(nodes[:, None], (chunk, 4)).reshape(-1)
        lengths = state.distances[parents] + 1
        return _QueueState(
            state.head + count,
            state.tail + prefix[-1],
            state.queue.at[slots].set(targets, mode="drop"),
            state.distances.at[unique_targets].set(lengths, mode="drop"),
            state.parents.at[unique_targets].set(parents, mode="drop"),
        )

    result = lax.while_loop(searching, expand, initial)
    return SearchTree(result.distances.reshape(known_map.shape), result.parents)


def shortest_path(
    known_map: Array,
    start: Array,
    goal: Array,
    tree: SearchTree | None = None,
) -> PathResult:
    """Construct the full route using a supplied tree or an exact targeted BFS."""
    start = jnp.asarray(start, dtype=jnp.int32)
    goal = jnp.asarray(goal, dtype=jnp.int32)
    search = bfs(known_map, start, goal) if tree is None else tree
    height, width = known_map.shape
    gx, gy = goal[0], goal[1]
    target = jnp.clip(gy, 0, height - 1) * width + jnp.clip(gx, 0, width - 1)
    graph_distance = search.distances.reshape(-1)[target]
    valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height) & (graph_distance >= 0)
    length = jnp.where(valid, graph_distance, 0)
    endpoint = jnp.where(valid, goal, start)
    path = jnp.broadcast_to(endpoint, (height * width, 2))

    def backtrack(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        index, node, route = carry
        route = route.at[index].set(jnp.array([node % width, node // width], jnp.int32))
        return index - 1, search.parents[node], route

    initial = (jnp.where(valid, length, -1), target, path)
    _, _, route = lax.while_loop(lambda carry: carry[0] >= 0, backtrack, initial)
    return PathResult(route, length, valid)


def navigate(  # noqa: PLR0913, PLR0917
    known_map: Array,
    start: Array,
    goal: Array,
    max_distance: int,
    plan_horizon: int,
    tree: SearchTree | None = None,
) -> NavigationResult:
    """Move along at most max_distance edges, retaining a valid unfinished goal.

    plan_horizon controls output shape and must be static under jit. Plans describe
    this event and subsequent own actions assuming the same route is followed.
    """
    route = shortest_path(known_map, start, goal, tree)
    distance = jnp.minimum(route.length, max_distance)
    indices = jnp.minimum(jnp.arange(1, plan_horizon + 1) * max_distance, route.length)
    valid_length = jnp.minimum(plan_horizon, (route.length + max_distance - 1) // max_distance)
    return NavigationResult(
        route.path[distance],
        distance,
        ~route.valid,
        jnp.where(route.valid, goal, start),
        route.path,
        route.length,
        route.path[indices],
        valid_length,
    )
