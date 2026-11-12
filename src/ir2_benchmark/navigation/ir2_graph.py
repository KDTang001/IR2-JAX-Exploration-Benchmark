"""Fixed-shape IR2 graph state; maps use 0 free, 1 obstacle, 2 unknown.

Edges are directed and include self edges. Collision testing deliberately does
not inspect the destination pixel, matching graph_generator.py and node.py.
Capacity exhaustion is sticky and blocks every observation; callers must report
``overflow`` as an error rather than continuing an incomplete graph.
"""

# Original port signatures and literal map codes intentionally remain recognizable.
# ruff: noqa: PLR0913, PLR0917, PLR2004, FBT003

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from ir2_benchmark.config import HostConfig


class GraphObservation(NamedTuple):
    """Original six-feature graph and masks (True means blocked)."""

    node_inputs: Array
    edge_inputs: Array
    current_index: Array
    node_padding_mask: Array
    edge_padding_mask: Array
    edge_mask: Array


class GraphMemory(NamedTuple):
    """Receiver-owned persistent graph, utility sets, and independent own route."""

    nodes: Array
    valid: Array
    edges: Array
    edge_order: Array
    global_nodes: Array
    global_valid: Array
    global_edges: Array
    history: Array
    history_count: Array
    consumed_history: Array
    observable: Array
    frontier_valid: Array
    utility: Array
    zero_utility: Array
    step: Array
    overflow: Array
    path_failed: Array


def clear_segment(local_map: Array, start: Array, end: Array) -> Array:
    """Original staircase Bresenham: exclude endpoint, stop at map exterior."""
    return _clear_segment(local_map, start, end, jnp.array(True))


def _clear_segment(local_map: Array, start: Array, end: Array, enabled: Array) -> Array:
    """Evaluate exact staircase pixels in blocks, excluding the destination.

    At Manhattan step t, horizontal_count = (2*dx*t + dx+dy-1)//(2*(dx+dy)).
    This integer formula preserves the original vertical-first equal-error tie.
    """
    start, end = start.astype(jnp.int32), end.astype(jnp.int32)
    dx, dy = jnp.abs(end - start)
    length = dx + dy
    denominator_length = jnp.maximum(length, 1)
    direction = jnp.where(end > start, 1, -1)
    start_inside = jnp.all((start >= 0) & (start < jnp.array(local_map.shape[::-1])))

    def block(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        offset, clear = carry
        t = offset + jnp.arange(32)
        horizontal = (2 * dx * t + denominator_length - 1) // (2 * denominator_length)
        x = start[0] + direction[0] * horizontal
        y = start[1] + direction[1] * (t - horizontal)
        inside = (x >= 0) & (y >= 0) & (x < local_map.shape[1]) & (y < local_map.shape[0])
        checked = inside & (t < length)
        free = (
            local_map[
                jnp.clip(y, 0, local_map.shape[0] - 1), jnp.clip(x, 0, local_map.shape[1] - 1)
            ]
            == 0
        )
        return offset + 32, clear & jnp.all(~checked | free)

    return lax.while_loop(
        lambda carry: carry[1] & start_inside & (carry[0] < length),
        block,
        (jnp.array(0, jnp.int32), enabled),
    )[1]


def find_frontier(local_map: Array, valid_shape: Array, resolution: int = 4) -> tuple[Array, Array]:
    """Original min block reduction and eight-neighbor frontier predicate."""
    height, width = local_map.shape
    ph, pw = (-height) % resolution, (-width) % resolution
    yy, xx = jnp.indices(local_map.shape)
    source = jnp.where((yy < valid_shape[0]) & (xx < valid_shape[1]), local_map, 1)
    encoded = jnp.where(source == 0, 255, jnp.where(source == 1, 1, 127))
    # skimage.block_reduce pads incomplete blocks with zero.
    encoded = jnp.pad(encoded, ((0, ph), (0, pw)), constant_values=0)
    reduced = encoded.reshape(
        (height + ph) // resolution, resolution, (width + pw) // resolution, resolution
    ).min(axis=(1, 3))
    unknown = jnp.pad((reduced == 127).astype(jnp.int32), 1)
    h, w = reduced.shape
    count = sum(
        unknown[dy : dy + h, dx : dx + w]
        for dy in range(3)
        for dx in range(3)
        if (dy, dx) != (1, 1)
    )
    enabled = (reduced == 255) & (count > 1) & (count < 8)
    x, y = jnp.meshgrid(jnp.arange(w) * resolution, jnp.arange(h) * resolution, indexing="ij")
    return jnp.stack((x, y), axis=-1).reshape(-1, 2), enabled.T.reshape(-1)


def _insert(
    nodes: Array, valid: Array, incoming: Array, present: Array
) -> tuple[Array, Array, Array]:
    """Append first occurrences in input order into free slots, without eviction."""
    if incoming.shape[0] == 0:
        return nodes, valid, jnp.array(False)
    ids = jnp.arange(incoming.shape[0])
    order = jnp.lexsort((ids, ~present, incoming[:, 1], incoming[:, 0]))
    sorted_points = incoming[order]
    first = (ids == 0) | jnp.any(sorted_points != jnp.roll(sorted_points, 1, axis=0), axis=1)
    first_present = jnp.zeros_like(present).at[order].set(first & present[order])
    exists = jnp.any(jnp.all(incoming[:, None] == nodes[None], axis=-1) & valid[None], axis=1)
    accept = first_present & ~exists
    rank = jnp.cumsum(accept) - 1
    room = jnp.sum(~valid)
    slots = jnp.nonzero(~valid, size=nodes.shape[0], fill_value=0)[0]
    destination = jnp.where(
        accept & (rank < room), slots[jnp.clip(rank, 0, nodes.shape[0] - 1)], nodes.shape[0]
    )
    nodes = nodes.at[destination].set(incoming, mode="drop")
    valid = valid.at[destination].set(True, mode="drop")
    return nodes, valid, jnp.sum(accept) > room


def _lattice(valid_shape: Array, size: int) -> Array:
    x = jnp.rint(jnp.linspace(0, valid_shape[1] - 1, size)).astype(jnp.int32)
    y = jnp.rint(jnp.linspace(0, valid_shape[0] - 1, size)).astype(jnp.int32)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    return jnp.stack((xx, yy), axis=-1).reshape(-1, 2)


def _visible(
    local_map: Array, nodes: Array, valid: Array, frontiers: Array, enabled: Array, radius: int
) -> Array:
    span = (radius + 3) // 4
    dx, dy = jnp.meshgrid(jnp.arange(-span, span + 1), jnp.arange(-span, span + 1), indexing="ij")
    offsets = jnp.stack((dx, dy), axis=-1).reshape(-1, 2)
    frontier_height, frontier_width = (local_map.shape[0] + 3) // 4, (local_map.shape[1] + 3) // 4

    def row(point: Array, active: Array) -> Array:
        def visible() -> Array:
            cells = point // 4 + offsets
            inside = jnp.all(
                (cells >= 0) & (cells < jnp.array([frontier_width, frontier_height])), axis=1
            )
            indices = jnp.clip(
                cells[:, 0] * frontier_height + cells[:, 1], 0, frontiers.shape[0] - 1
            )
            targets = cells * 4
            nearby = (
                active
                & inside
                & enabled[indices]
                & (jnp.sum((targets - point) ** 2, axis=1) < radius**2)
            )
            count = jnp.sum(nearby)
            selected = jnp.nonzero(nearby, size=offsets.shape[0], fill_value=0)[0]

            def trace(block: Array, visible: Array) -> Array:
                ranks = block * 32 + jnp.arange(32)
                rays = selected[jnp.minimum(ranks, selected.shape[0] - 1)]
                clear = jax.vmap(
                    lambda target, active_ray: _clear_segment(local_map, point, target, active_ray)
                )(targets[rays], ranks < count)
                return visible.at[indices[rays]].max(clear)

            return lax.fori_loop(0, (count + 31) // 32, trace, jnp.zeros_like(enabled))

        return lax.cond(active, visible, lambda: jnp.zeros_like(enabled))

    return jax.vmap(row)(nodes, valid)


def _knn_row(
    local_map: Array,
    nodes: Array,
    valid: Array,
    point: Array,
    k: int,
    inherited: Array,
    max_distance: float = float("inf"),
    *,
    all_nodes: bool = False,
    active: Array | bool = True,
) -> tuple[Array, Array]:
    """Count attempted neighbors, including colliding edges, as original does."""
    distance = jnp.sum((nodes - point).astype(jnp.float32) ** 2, axis=1)
    order = jnp.lexsort((nodes[:, 1], nodes[:, 0], jnp.where(valid, distance, jnp.inf)))
    budget = k - jnp.sum(inherited)
    ranked = valid[order] if all_nodes else valid[order] & ~inherited[order]
    rank = jnp.cumsum(ranked) - 1
    # The original per-node loop executes once even when global edges use the
    # entire K budget; preserve that behavior (it can produce K+1 edges).
    considered = ranked & (rank < (budget if all_nodes else jnp.maximum(budget, 1)))
    # custom KNN's over-length branch continues before the break condition.
    # If fewer than K points lie inside the radius it examines the entire list.
    stop = ranked & (rank >= budget - 1) & (distance[order] <= max_distance**2)
    stop_index = jnp.min(jnp.where(stop, jnp.arange(nodes.shape[0]), nodes.shape[0]))
    scanned = valid[order] & (jnp.arange(nodes.shape[0]) <= stop_index)
    attempted = jnp.zeros_like(valid).at[order].set(scanned)
    eligible = (
        jnp.zeros_like(valid).at[order].set(considered) & (distance <= max_distance**2) & active
    )
    clear = jax.vmap(
        lambda target, active_ray: _clear_segment(local_map, point, target, active_ray)
    )(nodes, eligible)
    return inherited | clear, attempted


def rebuild_edges(
    local_map: Array,
    memory: GraphMemory,
    *,
    config: HostConfig,
    global_min: float | None = None,
    global_max: float | None = None,
) -> Array:
    """Rebuild every dense row, optionally using the rendezvous fallback radii."""
    minimum = config.graph_local_radius if global_min is None else global_min
    maximum = config.graph_global_radius if global_max is None else global_max
    return _dense_edges(
        local_map,
        memory.nodes,
        memory.valid,
        memory.global_nodes,
        memory.global_valid,
        memory.global_edges,
        config.max_candidates + 1,
        minimum,
        maximum,
        all_nodes=True,
    )[0]


def _ordered_edges(nodes: Array, edges: Array, inherited: Array, k: int) -> Array:
    """Store dictionary insertion order: retained globals before dense KNN.

    Both implementations standardize neighbor ties by x then y; inherited
    globals originate from a set and therefore use coordinate order alone.
    """
    distance = jnp.sum((nodes[:, None] - nodes[None]).astype(jnp.float32) ** 2, axis=-1)
    group = jnp.where(inherited, 0, jnp.where(edges, 1, 2))
    x = jnp.broadcast_to(nodes[None, :, 0], distance.shape)
    y = jnp.broadcast_to(nodes[None, :, 1], distance.shape)
    order = jnp.lexsort((y, x, jnp.where(inherited, 0.0, distance), group), axis=1)[:, :k]
    return jnp.where(jnp.take_along_axis(edges, order, axis=1), order, -1)


def rebuild_graph_edges(
    local_map: Array,
    memory: GraphMemory,
    *,
    config: HostConfig,
    global_min: float | None = None,
    global_max: float | None = None,
) -> GraphMemory:
    """Refresh adjacency AND outgoing insertion order for rendezvous fallback."""
    minimum = config.graph_local_radius if global_min is None else global_min
    maximum = config.graph_global_radius if global_max is None else global_max
    edges, _, inherited = _dense_edges(
        local_map,
        memory.nodes,
        memory.valid,
        memory.global_nodes,
        memory.global_valid,
        memory.global_edges,
        config.max_candidates + 1,
        minimum,
        maximum,
        all_nodes=True,
    )
    return memory._replace(
        edges=edges,
        edge_order=_ordered_edges(memory.nodes, edges, inherited, config.max_candidates + 1),
        overflow=memory.overflow | jnp.any(jnp.sum(edges, axis=1) > config.max_candidates + 1),
    )


def _dense_edges(
    local_map: Array,
    nodes: Array,
    valid: Array,
    globals_: Array,
    global_valid: Array,
    global_edges: Array,
    k: int,
    minimum: float,
    maximum: float,
    *,
    all_nodes: bool = False,
) -> tuple[Array, Array, Array]:
    matches = jnp.all(nodes[:, None] == globals_[None, :], axis=-1) & global_valid[None, :]
    indices = jnp.argmax(matches, axis=1)
    global_present = jnp.any(matches, axis=1)
    mapped = (
        global_edges[indices[:, None], indices[None, :]]
        & global_present[:, None]
        & global_present[None, :]
    )
    same_coordinates = jnp.all(nodes[:, None] == nodes[None], axis=-1) & valid[None]
    first_coordinate = jnp.argmax(same_coordinates, axis=1)
    distance = jnp.sum((nodes[:, None] - nodes[None, :]).astype(jnp.float32) ** 2, axis=-1)
    inherited = mapped & (distance > minimum**2) & (distance <= maximum**2)
    inherited &= jnp.arange(nodes.shape[0])[None] == first_coordinate[None]
    x = jnp.broadcast_to(nodes[None, :, 0], distance.shape)
    y = jnp.broadcast_to(nodes[None, :, 1], distance.shape)
    order = jnp.lexsort((y, x, jnp.where(inherited, distance, jnp.inf)), axis=1)
    rank = jnp.argsort(order, axis=1)
    inherited &= rank < k
    edges, attempted = jax.vmap(
        lambda point, inherited_row, active: _knn_row(
            local_map, nodes, valid, point, k, inherited_row, all_nodes=all_nodes, active=active
        )
    )(nodes, inherited, valid)
    # The source stores edges by coordinate but observations resolve each
    # destination to its FIRST node-array index, including duplicate start nodes.
    edges = (
        jnp.zeros_like(edges)
        .at[jnp.arange(nodes.shape[0])[:, None], first_coordinate[None]]
        .max(edges)
    )
    return edges & valid[:, None] & valid[None, :], attempted & valid[:, None], inherited


def _global_update(
    local_map: Array,
    nodes: Array,
    valid: Array,
    edges: Array,
    added: Array,
    config: HostConfig,
    bidirectional: bool,  # noqa: FBT001
) -> Array:
    def update() -> Array:
        def row(i: int) -> tuple[Array, Array]:
            return _knn_row(
                local_map,
                nodes,
                valid,
                nodes[i],
                config.max_candidates + 1,
                jnp.zeros_like(valid),
                config.graph_global_radius,
                active=valid[i] & jnp.any(added),
            )

        new_rows, attempted = jax.vmap(row)(jnp.arange(nodes.shape[0]))
        affected = added | jnp.any(attempted & added[:, None], axis=0)
        result = jnp.where(affected[:, None], new_rows, edges) & valid[:, None] & valid[None, :]
        return result | result.T if bidirectional else result

    return lax.cond(jnp.any(added), update, lambda: edges)


def initialize_graph(
    local_map: Array, position: Array, valid_shape: Array, *, config: HostConfig
) -> GraphMemory:
    """Generate all known-free lattice nodes, prepend start, and initialize sets."""
    size, global_size = config.graph_padding_size, config.graph_capacity
    grid = _lattice(valid_shape, config.graph_nodes_per_axis)
    incoming = jnp.concatenate((position[None], grid))
    present = jnp.concatenate((jnp.ones(1, bool), local_map[grid[:, 1], grid[:, 0]] == 0))
    source_indices = jnp.nonzero(present, size=size, fill_value=0)[0]
    nodes = incoming[source_indices]
    valid = jnp.arange(size) < jnp.sum(present)
    nodes = jnp.where(valid[:, None], nodes, 0)
    overflow = jnp.sum(present) >= size
    frontiers, frontier_valid = find_frontier(local_map, valid_shape)
    observable = _visible(
        local_map, nodes, valid, frontiers, frontier_valid, config.graph_utility_radius
    )
    utility = jnp.sum(observable, axis=1)
    memory = GraphMemory(
        nodes,
        valid,
        jnp.zeros((size, size), bool),
        jnp.full((size, config.max_candidates + 1), -1, jnp.int32),
        jnp.zeros((global_size, 2), jnp.int32),
        jnp.zeros(global_size, bool),
        jnp.zeros((global_size, global_size), bool),
        jnp.zeros((config.max_rounds + 1, 2), jnp.int32).at[0].set(position),
        jnp.array(1, jnp.int32),
        # Env.route_node contains start; Env's global route history starts empty.
        jnp.array(1, jnp.int32),
        observable,
        frontier_valid,
        utility,
        utility == 0,
        jnp.array(0, jnp.int32),
        overflow,
        jnp.array(False),
    )
    return rebuild_graph_edges(local_map, memory, config=config)


def _reachable(local_map: Array, position: Array, radius: int, seed: Array) -> Array:
    """Exact four-connected flood inside the original half-open local window."""
    yy, xx = jnp.indices((2 * radius, 2 * radius))
    yy, xx = yy + position[1] - radius, xx + position[0] - radius
    inside = (xx >= 0) & (yy >= 0) & (xx < local_map.shape[1]) & (yy < local_map.shape[0])
    allowed = (
        local_map[jnp.clip(yy, 0, local_map.shape[0] - 1), jnp.clip(xx, 0, local_map.shape[1] - 1)]
        == 0
    ) & inside
    relative = seed - position + radius
    seed_inside = jnp.all((relative >= 0) & (relative < 2 * radius))
    relative = jnp.clip(relative, 0, 2 * radius - 1)
    reached = jnp.zeros_like(allowed).at[relative[1], relative[0]].set(seed_inside) & allowed

    def expand(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        old, _ = carry
        padded = jnp.pad(old, 1)
        new = old | (
            allowed & (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:])
        )
        return new, jnp.any(new != old)

    reached = lax.while_loop(lambda carry: carry[1], expand, (reached, jnp.array(True)))[0]
    return (
        jnp.zeros_like(local_map, bool)
        .at[jnp.clip(yy, 0, local_map.shape[0] - 1), jnp.clip(xx, 0, local_map.shape[1] - 1)]
        .max(reached & inside)
    )


def _local_grid(
    local_map: Array, position: Array, radius: int, grid: Array, memory: GraphMemory
) -> Array:
    """Match the source's old-node lookup, including its negative-index fallback."""
    allowed = (local_map[grid[:, 1], grid[:, 0]] == 0) & jnp.all(
        (grid >= position - radius) & (grid < position + radius), axis=1
    )
    old_match = memory.valid & jnp.all(memory.nodes == position, axis=1)
    old_index = jnp.where(
        jnp.any(old_match), jnp.argmax(old_match), jnp.maximum(jnp.sum(memory.valid) - 1, 0)
    )
    candidate_match = allowed & jnp.all(grid == memory.nodes[old_index], axis=1)
    last_candidate = jnp.max(jnp.where(allowed, jnp.arange(grid.shape[0]), 0))
    seed_index = jnp.where(jnp.any(candidate_match), jnp.argmax(candidate_match), last_candidate)
    component = _reachable(local_map, position, radius, grid[seed_index])
    return allowed & component[grid[:, 1], grid[:, 0]]


def _graph_reachable(edges: Array, start: Array, enabled: Array) -> Array:
    reached = (jnp.arange(edges.shape[0]) == start) & enabled

    def expand(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        old, _ = carry
        new = old | jnp.any(edges & old[:, None], axis=0)
        return new, jnp.any(new != old)

    return lax.while_loop(lambda carry: carry[1], expand, (reached, enabled))[0]


def merge_global_graph(
    memory: GraphMemory,
    local_map: Array,
    frontiers: Array,
    frontier_valid: Array,
    positions: Array,
    position_valid: Array,
    *,
    config: HostConfig,
    enabled: Array | bool = True,
) -> GraphMemory:
    """Original sequential connectivity-preserving global-node merger.

    Both the reference and port traverse coordinate sets in x/y order.
    """
    nodes, valid, edges = memory.global_nodes, memory.global_valid, memory.global_edges
    in_old = jnp.any(
        jnp.all(nodes[:, None] == memory.nodes[None, :], axis=-1) & memory.valid[None], axis=1
    )
    near_frontier = jnp.any(
        (
            jnp.sum((nodes[:, None] - frontiers[None]).astype(jnp.float32) ** 2, axis=-1)
            < config.graph_frontier_protect_radius**2
        )
        & frontier_valid[None],
        axis=1,
    )
    near_robot = jnp.any(
        (
            jnp.sum((nodes[:, None] - positions[None]).astype(jnp.float32) ** 2, axis=-1)
            < config.graph_unique_radius**2
        )
        & position_valid[None],
        axis=1,
    )
    protected = (in_old & near_frontier) | near_robot

    candidates = valid & ~protected
    coordinate_order = jnp.lexsort((nodes[:, 1], nodes[:, 0], ~candidates))

    def merge(iteration: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        i = coordinate_order[iteration]
        active, current_edges, merged = carry
        distance = jnp.sum((nodes - nodes[i]).astype(jnp.float32) ** 2, axis=1)
        removed = (
            active
            & ~merged
            & (distance < config.graph_unique_radius**2)
            & (jnp.arange(nodes.shape[0]) != i)
        )
        proceed = active[i] & ~merged[i] & jnp.any(removed)

        def attempt() -> tuple[Array, Array, Array]:
            criteria = (
                removed | jnp.any(current_edges & removed[:, None], axis=0) | current_edges[i]
            )
            trial_valid = active & ~removed
            trial_edges = current_edges & trial_valid[:, None] & trial_valid[None]
            row, attempted = _knn_row(
                local_map,
                nodes,
                trial_valid,
                nodes[i],
                config.max_candidates + 1,
                jnp.zeros_like(active),
                config.graph_global_radius,
                active=proceed & enabled,
            )
            trial_edges = trial_edges.at[i].set(row)
            trial_edges |= trial_edges.T
            reached = _graph_reachable(trial_edges, jnp.array(i), proceed & enabled)
            success = jnp.all(~(criteria & trial_valid) | reached)
            # Source rollback restores only the saved outgoing dictionaries;
            # reverse edges introduced elsewhere by symmetrization survive.
            restored_edges = jnp.where((criteria | attempted)[:, None], current_edges, trial_edges)
            return (
                jnp.where(success, trial_valid, active),
                jnp.where(success, trial_edges, restored_edges),
                merged | (success & (removed | (jnp.arange(nodes.shape[0]) == i))),
            )

        return lax.cond(proceed, attempt, lambda: carry)

    def iteration(
        carry: tuple[Array, tuple[Array, Array, Array]],
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        index, graph = carry
        return index + 1, merge(index, graph)

    _, (valid, edges, _) = lax.while_loop(
        lambda carry: enabled & (carry[0] < jnp.sum(candidates)),
        iteration,
        (jnp.array(0, jnp.int32), (valid, edges, protected)),
    )
    return memory._replace(global_valid=valid, global_edges=edges)


def _shortest_route(
    nodes: Array,
    valid: Array,
    edges: Array,
    start: Array,
    destination: Array,
    *,
    enabled: Array | bool = True,
) -> tuple[Array, Array]:
    """Euclidean shortest path as fixed indices, returned in start-first order."""
    size = nodes.shape[0]
    distance = jnp.full(size, jnp.inf).at[start].set(jnp.where(enabled, 0.0, jnp.inf))
    parent = jnp.full(size, -1, jnp.int32).at[start].set(start)
    weights = jnp.sqrt(jnp.sum((nodes[:, None] - nodes[None, :]).astype(jnp.float32) ** 2, axis=-1))

    def relax(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        costs, parents, closed = carry
        heuristic = jnp.sqrt(jnp.sum((nodes - nodes[destination]).astype(jnp.float32) ** 2, axis=1))
        current = jnp.lexsort(
            (nodes[:, 1], nodes[:, 0], jnp.where(valid & ~closed, costs + heuristic, jnp.inf))
        )[0]
        active = jnp.isfinite(costs[current]) & ~closed[current] & ~closed[destination]
        better = edges[current] & ~closed & (costs[current] + weights[current] < costs) & active
        return (
            jnp.where(better, costs[current] + weights[current], costs),
            jnp.where(better, current, parents),
            closed.at[current].set(closed[current] | active),
        )

    distance, parent, _ = lax.while_loop(
        lambda carry: (
            enabled & ~carry[2][destination] & jnp.any(valid & ~carry[2] & jnp.isfinite(carry[0]))
        ),
        relax,
        (distance, parent, jnp.zeros(size, bool)),
    )

    def backtrack(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        indices, cursor, length = carry
        active = cursor >= 0
        indices = indices.at[length].set(jnp.maximum(cursor, 0))
        return (
            indices,
            jnp.where(active & (cursor != start), parent[jnp.maximum(cursor, 0)], -1),
            length + active.astype(jnp.int32),
        )

    indices, _, length = lax.while_loop(
        lambda carry: (carry[1] >= 0) & (carry[2] < size),
        backtrack,
        (
            jnp.zeros(size, jnp.int32),
            jnp.where(jnp.isfinite(distance[destination]), destination, -1),
            jnp.array(0, jnp.int32),
        ),
    )
    return indices[jnp.clip(length - 1 - jnp.arange(size), 0, size - 1)], jnp.arange(size) < length


def _offshoots(
    memory: GraphMemory, local_map: Array, position: Array, own_local: Array, *, config: HostConfig
) -> GraphMemory:
    nodes, valid = memory.nodes, memory.valid
    first_coordinate = jnp.argmax(
        jnp.all(nodes[:, None] == nodes[None], axis=-1) & valid[None], axis=1
    )
    own_local &= jnp.arange(nodes.shape[0]) == first_coordinate
    own_local &= valid & (
        jnp.sum((nodes - position).astype(jnp.float32) ** 2, axis=1)
        <= config.graph_offshoot_max_radius**2
    )
    current = jnp.argmin(
        jnp.where(valid, jnp.sum((nodes - position) ** 2, axis=1), jnp.iinfo(jnp.int32).max)
    )
    # The source compares the unsorted LOCAL INDEX (not its utility) against
    # own utility. Preserve that surprising condition rather than fixing it.
    coordinate_order = jnp.lexsort((nodes[:, 1], nodes[:, 0]))
    local_index = (
        jnp.zeros(nodes.shape[0], jnp.int32)
        .at[coordinate_order]
        .set(jnp.cumsum(own_local[coordinate_order]) - 1)
    )
    candidates = (
        own_local
        & ~memory.zero_utility
        & (local_index > memory.utility[current])
        & jnp.any(memory.global_valid)
    )
    order = jnp.lexsort(
        (nodes[:, 1], nodes[:, 0], jnp.where(candidates, -memory.utility, jnp.iinfo(jnp.int32).max))
    )
    old_global_valid = memory.global_valid

    def visit(
        i: int, carry: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        globals_, enabled, overflow, added_count = carry
        index = order[i]
        nearby = jnp.any(
            enabled
            & (
                jnp.sum((globals_ - nodes[index]).astype(jnp.float32) ** 2, axis=1)
                < config.graph_offshoot_unique_radius**2
            )
        )
        eligible = (
            own_local[index]
            & ~memory.zero_utility[index]
            & (local_index[index] > memory.utility[current])
            & ~nearby
            & (added_count < config.graph_offshoot_count)
        )

        def append() -> tuple[Array, Array, Array, Array]:
            direct = _clear_segment(local_map, position, nodes[index], eligible)
            indices, path_valid = lax.cond(
                direct,
                lambda: (jnp.full(nodes.shape[0], index), jnp.arange(nodes.shape[0]) == 0),
                lambda: _shortest_route(
                    nodes, valid, memory.edges, current, index, enabled=eligible & ~direct
                ),
            )
            path_valid &= eligible
            new_nodes, new_valid, extra = _insert(globals_, enabled, nodes[indices], path_valid)
            return new_nodes, new_valid, overflow | extra, added_count + jnp.sum(path_valid)

        return lax.cond(eligible, append, lambda: carry)

    def iteration(
        carry: tuple[Array, tuple[Array, Array, Array, Array]],
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        index, graph = carry
        return index + 1, visit(index, graph)

    _, (globals_, enabled, overflow, _) = lax.while_loop(
        lambda carry: (
            (carry[0] < jnp.sum(candidates)) & (carry[1][3] < config.graph_offshoot_count)
        ),
        iteration,
        (
            jnp.array(0, jnp.int32),
            (memory.global_nodes, memory.global_valid, memory.overflow, jnp.array(0, jnp.int32)),
        ),
    )
    edges = _global_update(
        local_map,
        globals_,
        enabled,
        memory.global_edges,
        enabled & ~old_global_valid,
        config,
        False,
    )
    return memory._replace(
        global_nodes=globals_, global_valid=enabled, global_edges=edges, overflow=overflow
    )


def update_graph(  # noqa: PLR0915
    memory: GraphMemory,
    local_map: Array,
    position: Array,
    valid_shape: Array,
    peer_positions: Array,
    peer_valid: Array,
    incoming_nodes: Array,
    incoming_valid: Array,
    *,
    config: HostConfig,
    record_visit: bool = True,
    extend_offshoots: bool = True,
) -> GraphMemory:
    """Update from own sensing and delivered data, never teammate hidden state."""
    history, history_count = memory.history, memory.history_count
    overflow = memory.overflow
    if record_visit:
        overflow |= history_count >= history.shape[0]
        history = history.at[jnp.minimum(history_count, history.shape[0] - 1)].set(position)
        history_count += 1
        matches = memory.valid & jnp.all(memory.nodes == position, axis=1)
        visited = (jnp.arange(memory.nodes.shape[0]) == jnp.argmax(matches)) & jnp.any(matches)
        memory = memory._replace(
            zero_utility=memory.zero_utility | visited,
            observable=memory.observable & ~visited[:, None],
            utility=jnp.where(visited, 0, memory.utility),
        )
    old_global_valid = memory.global_valid
    history_ids = jnp.arange(history.shape[0])
    new_history = (history_ids < history_count) & (history_ids >= memory.consumed_history)
    global_nodes, global_valid, extra = _insert(
        memory.global_nodes, old_global_valid, history, new_history
    )
    overflow |= extra
    # Import delivered data only; own route is independent of fusion.
    if config.global_graph_fusion:
        global_nodes, global_valid, extra = _insert(
            global_nodes, global_valid, incoming_nodes, incoming_valid
        )
        overflow |= extra
    touched = (
        jnp.any(
            jnp.all(global_nodes[:, None] == history[None], axis=-1) & new_history[None], axis=1
        )
        & global_valid
    )
    if config.global_graph_fusion:
        touched |= (
            jnp.any(
                jnp.all(global_nodes[:, None] == incoming_nodes[None], axis=-1)
                & incoming_valid[None],
                axis=1,
            )
            & global_valid
        )
    global_edges = _global_update(
        local_map, global_nodes, global_valid, memory.global_edges, touched, config, True
    )
    memory = memory._replace(
        global_nodes=global_nodes,
        global_valid=global_valid,
        global_edges=global_edges,
        history=history,
        history_count=history_count,
        consumed_history=history_count,
        overflow=overflow,
    )
    frontiers, frontier_valid = find_frontier(local_map, valid_shape)
    positions = jnp.concatenate((position[None], peer_positions))
    positions_valid = jnp.concatenate((jnp.ones(1, bool), peer_valid))
    step = memory.step + int(record_visit)
    if config.graph_merge_enabled:
        do_merge = (step > 0) & (
            (step % config.graph_merge_every == 0)
            | (jnp.sum(global_valid) > config.graph_merge_threshold)
        )
        memory = lax.cond(
            do_merge,
            lambda: merge_global_graph(
                memory,
                local_map,
                frontiers,
                frontier_valid,
                positions,
                positions_valid,
                config=config,
                enabled=do_merge,
            ),
            lambda: memory,
        )
    grid = _lattice(valid_shape, config.graph_nodes_per_axis)
    own_grid = _local_grid(local_map, position, config.graph_local_radius, grid, memory)

    def peer_component(pair: tuple[Array, Array]) -> Array:
        return lax.cond(
            pair[1],
            lambda: _local_grid(local_map, pair[0], config.graph_peer_radius, grid, memory),
            lambda: jnp.zeros(grid.shape[0], bool),
        )

    peer_grids = lax.map(peer_component, (peer_positions, peer_valid))
    grid_valid = own_grid | jnp.any(peer_grids, axis=0)
    desired = jnp.concatenate((memory.global_nodes, positions, grid))
    desired_valid = jnp.concatenate((memory.global_valid, positions_valid, grid_valid))
    retained = memory.valid & jnp.any(
        jnp.all(memory.nodes[:, None] == desired[None], axis=-1) & desired_valid[None], axis=1
    )
    nodes, valid, extra = _insert(memory.nodes, retained, desired, desired_valid)
    same = retained
    observable_new = _visible(
        local_map, nodes, valid & ~same, frontiers, frontier_valid, config.graph_utility_radius
    )
    newly_visible = _visible(
        local_map,
        nodes,
        same & ~memory.zero_utility,
        frontiers,
        frontier_valid & ~memory.frontier_valid,
        config.graph_utility_radius,
    )
    observable = jnp.where(
        same[:, None], (memory.observable & frontier_valid[None]) | newly_visible, observable_new
    )
    utility = jnp.sum(observable, axis=1)
    zero = jnp.where(same, memory.zero_utility | (utility == 0), utility == 0)
    new_edges, attempted, inherited = _dense_edges(
        local_map,
        nodes,
        valid,
        memory.global_nodes,
        memory.global_valid,
        memory.global_edges,
        config.max_candidates + 1,
        config.graph_local_radius,
        config.graph_global_radius,
    )
    removed = memory.valid & ~retained
    added = valid & ~retained
    affected_removed = jnp.any(memory.edges & removed[:, None], axis=0) & retained
    nearby_global = jnp.any(
        jnp.all(nodes[:, None] == memory.global_nodes[None], axis=-1) & memory.global_valid[None],
        axis=1,
    ) & (
        jnp.sum((nodes - position).astype(jnp.float32) ** 2, axis=1)
        <= (2 * config.sensor_radius) ** 2
    )
    affected = (
        added
        | affected_removed
        | (jnp.any(removed) & nearby_global)
        | jnp.any(attempted & added[:, None], axis=0)
    )
    old_edges = memory.edges & retained[:, None] & retained[None]
    edges = jnp.where(affected[:, None], new_edges, old_edges)
    old_order = memory.edge_order
    old_order_valid = (old_order >= 0) & jnp.take_along_axis(
        old_edges, jnp.maximum(old_order, 0), axis=1
    )
    compact_order = jnp.argsort(~old_order_valid, axis=1, stable=True)
    old_order = jnp.take_along_axis(
        jnp.where(old_order_valid, old_order, -1), compact_order, axis=1
    )
    edge_order = jnp.where(
        affected[:, None],
        _ordered_edges(nodes, new_edges, inherited, config.max_candidates + 1),
        old_order,
    )
    # The original retains surviving node order and appends new coordinates.
    # Compact holes accordingly; this matters because it masks node INDEX zero.
    ids = jnp.arange(nodes.shape[0])
    permutation = jnp.lexsort(
        (
            nodes[:, 1],
            nodes[:, 0],
            jnp.where(retained, ids, nodes.shape[0]),
            jnp.where(retained, 0, jnp.where(valid, 1, 2)),
        )
    )
    inverse = jnp.argsort(permutation)
    edge_order = edge_order[permutation]
    edge_order = jnp.where(edge_order >= 0, inverse[jnp.maximum(edge_order, 0)], -1)
    nodes, valid = nodes[permutation], valid[permutation]
    edges = edges[permutation[:, None], permutation[None]]
    memory = memory._replace(
        nodes=nodes,
        valid=valid,
        edges=edges,
        edge_order=edge_order,
        observable=observable[permutation],
        frontier_valid=frontier_valid,
        utility=utility[permutation],
        zero_utility=zero[permutation],
        step=step,
        overflow=memory.overflow
        | extra
        | (jnp.sum(valid) >= config.graph_padding_size)
        | jnp.any(jnp.sum(edges, axis=1) > config.max_candidates + 1),
    )
    if extend_offshoots:
        own_local = jnp.any(jnp.all(nodes[:, None] == grid[None], axis=-1) & own_grid[None], axis=1)
        memory = lax.cond(
            jnp.any(memory.global_valid),
            lambda: _offshoots(memory, local_map, position, own_local, config=config),
            lambda: memory,
        )
    return memory


def graph_observation(
    memory: GraphMemory,
    position: Array,
    peer_positions: Array,
    peer_valid: Array,
    *,
    config: HostConfig,
) -> GraphObservation:
    """Pack full graph and original nearest-node route visit counts."""
    distance = jnp.sum((memory.nodes - position).astype(jnp.float32) ** 2, axis=1)
    current = jnp.argmin(jnp.where(memory.valid, distance, jnp.inf))
    history_distances = jnp.sum(
        (memory.history[:, None] - memory.nodes[None]).astype(jnp.float32) ** 2, axis=-1
    )
    history_nearest = jnp.argmin(jnp.where(memory.valid[None], history_distances, jnp.inf), axis=1)
    guidepost = (
        jnp.zeros(memory.nodes.shape[0], jnp.float32)
        .at[history_nearest]
        .add((jnp.arange(memory.history.shape[0]) < memory.history_count).astype(jnp.float32))
    )
    occupied = jnp.any(
        jnp.all(memory.nodes[:, None] == peer_positions[None], axis=-1) & peer_valid[None], axis=1
    ).astype(jnp.float32)
    occupied = occupied.at[current].set(-1.0)
    features = jnp.concatenate(
        (
            memory.nodes.astype(jnp.float32) / config.graph_coordinate_scale,
            (memory.utility / 50.0)[:, None],
            jnp.zeros((memory.nodes.shape[0], 1)),
            guidepost[:, None],
            occupied[:, None],
        ),
        axis=1,
    )
    features = jnp.where(memory.valid[:, None], features, 0.0)
    stored_order = memory.edge_order[current]
    order = jnp.maximum(stored_order, 0)
    # Original workers mask node index 0 even if that is a real neighbor; the
    # network separately suppresses candidate SLOT zero. Both are observable.
    failure = memory.overflow | memory.path_failed
    edge_valid = (stored_order >= 0) & (order != 0) & ~failure
    return GraphObservation(
        features, order, current, ~memory.valid | failure, ~edge_valid, ~memory.edges | failure
    )
