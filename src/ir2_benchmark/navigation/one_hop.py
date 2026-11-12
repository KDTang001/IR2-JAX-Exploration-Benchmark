"""Local collision-free graph edges and bounded, receiver-owned node history."""

import jax
import jax.numpy as jnp
from jax import Array, lax

from ir2_benchmark.navigation.candidates import CandidateSet
from ir2_benchmark.navigation.jax_path import NavigationResult


def merge_nodes(nodes: Array, count: Array, incoming: Array, present: Array) -> tuple[Array, Array]:
    """Deduplicate exact coordinates; deterministically evict oldest nodes at capacity."""
    capacity = nodes.shape[0]

    def insert(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        stored, total = carry
        valid = jnp.arange(capacity) < jnp.minimum(total, capacity)
        exists = jnp.any(valid & jnp.all(stored == incoming[index], axis=-1))
        accept = present[index] & ~exists
        slot = total % capacity
        stored = stored.at[slot].set(jnp.where(accept, incoming[index], stored[slot]))
        return stored, total + accept.astype(jnp.int32)

    return lax.fori_loop(0, incoming.shape[0], insert, (nodes, count))


def clear_segment(local_map: Array, start: Array, end: Array) -> Array:
    """Check the IR2 staircase Bresenham segment, including both free endpoints."""
    height, width = local_map.shape
    dx, dy = jnp.abs(end - start)
    direction = jnp.where(end > start, 1, -1)

    def free(point: Array) -> Array:
        x, y = point
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height)
        return valid & (local_map[jnp.clip(y, 0, height - 1), jnp.clip(x, 0, width - 1)] == 0)

    def move(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        point, error, clear = carry
        horizontal = error > 0
        point = point + jnp.where(
            horizontal, jnp.array([direction[0], 0]), jnp.array([0, direction[1]])
        )
        error = error + jnp.where(horizontal, -2 * dy, 2 * dx)
        return point, error, clear & free(point)

    _, _, clear = lax.while_loop(
        lambda carry: carry[2] & jnp.any(carry[0] != end),
        move,
        (start, dx - dy, free(start) & free(end)),
    )
    return clear


def graph_candidates(  # noqa: PLR0913
    local_map: Array,
    position: Array,
    valid_shape: Array,
    nodes: Array,
    count: Array,
    *,
    max_candidates: int,
    nodes_per_axis: int,
    local_radius: int,
    global_radius: int,
) -> CandidateSet:
    """Prioritize retained long edges, then nearest local nodes; reject blocked edges."""
    x = jnp.rint(jnp.linspace(0, valid_shape[1] - 1, nodes_per_axis)).astype(jnp.int32)
    y = jnp.rint(jnp.linspace(0, valid_shape[0] - 1, nodes_per_axis)).astype(jnp.int32)
    grid = jnp.stack(jnp.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    unique_x = (jnp.arange(nodes_per_axis) == 0) | (x != jnp.roll(x, 1))
    unique_y = (jnp.arange(nodes_per_axis) == 0) | (y != jnp.roll(y, 1))
    delta = grid - position
    grid_valid = (unique_x[:, None] & unique_y[None, :]).reshape(-1)
    grid_valid &= jnp.all((delta >= -local_radius) & (delta < local_radius), axis=-1)
    # A historical lattice node is already represented by the dense grid.
    node_delta = nodes - position
    in_window = jnp.all((node_delta >= -local_radius) & (node_delta < local_radius), axis=-1)
    on_grid = jnp.any(nodes[:, 0, None] == x, axis=-1) & jnp.any(nodes[:, 1, None] == y, axis=-1)
    node_valid = jnp.arange(nodes.shape[0]) < jnp.minimum(count, nodes.shape[0])
    node_valid &= ~(in_window & on_grid)
    points = jnp.concatenate((grid, nodes), axis=0)
    enabled = jnp.concatenate((grid_valid, node_valid))
    enabled &= jnp.all((points >= 0) & (points < valid_shape[::-1]), axis=-1)
    enabled &= (
        local_map[
            points[:, 1].clip(0, local_map.shape[0] - 1),
            points[:, 0].clip(0, local_map.shape[1] - 1),
        ]
        == 0
    )
    squared = jnp.sum(jnp.square(points - position), axis=-1)
    enabled &= (squared > 0) & (squared <= global_radius**2)
    long_history = (jnp.arange(points.shape[0]) >= grid.shape[0]) & (squared > local_radius**2)

    def neighbors(allowed: Array) -> tuple[Array, Array]:
        order = jnp.argsort(jnp.where(allowed, squared, jnp.iinfo(jnp.int32).max), stable=True)
        selected = order[:max_candidates]
        targets = points[selected]
        mask = allowed[selected] & jax.vmap(lambda p: clear_segment(local_map, position, p))(
            targets
        )
        return targets, mask

    # Blocked historical edges must not occupy all slots and hide legal local edges.
    far, far_mask = neighbors(enabled & long_history)
    near, near_mask = neighbors(enabled & ~long_history)
    combined, allowed = jnp.concatenate((far, near)), jnp.concatenate((far_mask, near_mask))
    if max_candidates:
        indices = jnp.nonzero(allowed, size=max_candidates, fill_value=0)[0]
        targets = combined[indices]
        mask = jnp.arange(max_candidates) < jnp.sum(allowed)
    else:
        targets, mask = near, near_mask
    targets = jnp.where(mask[:, None], targets, position)
    distance = jnp.where(
        mask, jnp.linalg.norm((targets - position).astype(jnp.float32), axis=-1), 0
    )
    return CandidateSet(
        jnp.concatenate((position[None], targets)),
        jnp.concatenate((jnp.ones(1, bool), mask)),
        jnp.concatenate((jnp.zeros(1, jnp.float32), distance)),
    )


def navigate_edge(
    position: Array, goal: Array, candidates: CandidateSet, horizon: int
) -> NavigationResult:
    """Execute exactly one offered graph edge; a PLAN commits only this endpoint."""
    selected = jnp.all(candidates.positions == goal, axis=-1) & candidates.mask
    valid = jnp.any(selected)
    endpoint = jnp.where(valid, goal, position)
    distance = jnp.linalg.norm((endpoint - position).astype(jnp.float32))
    moving = valid & jnp.any(endpoint != position)
    return NavigationResult(
        endpoint,
        distance,
        ~valid,
        endpoint,
        jnp.stack((position, endpoint)),
        moving.astype(jnp.int32),
        jnp.broadcast_to(endpoint, (horizon, 2)),
        moving.astype(jnp.int32),
    )
