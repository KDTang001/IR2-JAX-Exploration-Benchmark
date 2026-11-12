"""Original IR2 A* route-based rendezvous utility using receiver-local data."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from ir2_benchmark.navigation.ir2_graph import _clear_segment


class RendezvousResult(NamedTuple):
    """Utility and explicit route failure/rebuild signals for the environment."""

    utility: Array
    unreachable: Array
    repaired: Array


def _astar(  # noqa: PLR0913 - explicit graph state and batch execution gate
    nodes: Array, valid: Array, edges: Array, start: Array, goal: Array, *, enabled: Array
) -> tuple[Array, Array, Array]:
    """Return a reversed shortest route; break equal costs by coordinates (x, y)."""
    capacity = len(nodes)
    coordinates = nodes.astype(jnp.float32)
    heuristic = jnp.linalg.norm(coordinates - coordinates[goal], axis=1)
    costs = jnp.full(capacity, jnp.inf).at[start].set(0)
    opened = jnp.zeros(capacity, bool).at[start].set(valid[start] & enabled)
    closed = jnp.zeros(capacity, bool)
    parents = jnp.full(capacity, -1, jnp.int32).at[start].set(start)

    def condition(state: tuple[Array, Array, Array, Array, Array]) -> Array:
        _, opened, _, _, found = state
        return jnp.any(opened) & ~found

    def expand(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        costs, opened, closed, parents, _ = state
        priority = jnp.where(opened, costs + heuristic, jnp.inf)
        index = jnp.lexsort((nodes[:, 1], nodes[:, 0], priority))[0]
        proposals = costs[index] + jnp.linalg.norm(coordinates - coordinates[index], axis=1)
        improve = edges[index] & valid & ~closed & (proposals < costs)
        costs = jnp.where(improve, proposals, costs)
        parents = jnp.where(improve, index, parents)
        opened = (opened | improve).at[index].set(False)
        closed = closed.at[index].set(True)
        return costs, opened, closed, parents, index == goal

    _, _, _, parents, found = lax.while_loop(
        condition,
        expand,
        (costs, opened, closed, parents, jnp.array(False)),  # noqa: FBT003 - JAX scalar boolean
    )
    route = jnp.full(capacity, start, jnp.int32)

    def trace_condition(state: tuple[Array, Array, Array]) -> Array:
        _, count, index = state
        return found & (start != goal) & (count < capacity) & (index >= 0)

    def trace_step(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        route, count, index = state
        return route.at[count].set(index), count + 1, jnp.where(index == start, -1, parents[index])

    route, count, _ = lax.while_loop(
        trace_condition, trace_step, (route, jnp.array(0, jnp.int32), goal)
    )
    return route, count, found


def _inflate_route(  # noqa: PLR0913, PLR0917 - explicit JAX state arrays
    nodes: Array,
    valid: Array,
    local_map: Array,
    route: Array,
    count: Array,
    magnitude: Array,
    utility: Array,
) -> Array:
    """Visit actual densified route samples, without a path-by-node ray tensor."""
    coords = nodes[route].astype(jnp.float32)
    insertions = jnp.floor(jnp.linalg.norm(coords[1:] - coords[:-1], axis=1) / 30).astype(jnp.int32)
    total = count + jnp.sum(jnp.where(jnp.arange(len(insertions)) < count - 1, insertions, 0))

    def paint(point: Array, index: Array | int, values: Array, active: Array) -> Array:
        nearby = valid & active & (jnp.sum((nodes - point) ** 2, axis=1) <= 50**2)
        visible = jax.vmap(lambda end, enabled: _clear_segment(local_map, point, end, enabled))(
            nodes, nearby
        )
        value = magnitude - index * (magnitude / jnp.maximum(total, 1)) + 0.1
        return jnp.maximum(values, jnp.where(visible, value, 0))

    utility = paint(coords[0], 0, utility, count > 0)
    if len(nodes) == 1:
        return utility

    def segment(index: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        values, sample_index = state
        divisions = insertions[index] + 1
        # Evaluate interpolation in the original forward direction, then visit
        # samples backwards so the peer end receives the largest utility.
        start, end = coords[index + 1], coords[index]
        active = index < count - 1

        def sample(j: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            values, sample_index = carry
            point = jnp.rint(start + ((divisions - j) / divisions) * (end - start))
            sample_index += 1
            return paint(point, sample_index, values, active & (j <= divisions)), sample_index

        return lax.fori_loop(1, jnp.where(active, divisions + 1, 1), sample, (values, sample_index))

    return lax.fori_loop(0, jnp.maximum(count - 1, 0), segment, (utility, 0))[0]


def rendezvous_utility(  # noqa: PLR0913, PLR0917 - explicit public JAX state arrays
    nodes: Array,
    valid: Array,
    edges: Array,
    current_index: Array,
    peer_positions: Array,
    peer_enabled: Array,
    surplus: Array,
    local_map: Array,
    valid_shape: Array,
    repair_edges: Array | None = None,
) -> RendezvousResult:
    """Reproduce legacy map-delta route inflation, gating and graph fallbacks.

    ``surplus`` and peer gating are supplied by the receiver's delivered map
    cache. No ground truth or undisclosed peer knowledge is accessed here.
    The caller must abort an episode with an unreachable eligible peer and
    persist ``repair_edges`` when ``repaired`` is true.
    """
    nodes = jnp.asarray(nodes)
    valid = jnp.asarray(valid)
    edges = jnp.asarray(edges)
    current_index = jnp.asarray(current_index)
    peer_positions = jnp.asarray(peer_positions)
    peer_enabled = jnp.asarray(peer_enabled)
    surplus = jnp.asarray(surplus)
    local_map = jnp.asarray(local_map)
    valid_shape = jnp.asarray(valid_shape)
    capacity = len(nodes)
    area = jnp.prod(valid_shape).astype(jnp.float32)
    utility = jnp.zeros(capacity, jnp.float32)
    unreachable = jnp.zeros(len(peer_positions), bool)
    repaired = jnp.array(False)  # noqa: FBT003 - JAX scalar boolean
    fallback = edges if repair_edges is None else jnp.asarray(repair_edges)

    def peer(
        index: Array, state: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        values, failures, did_repair, graph = state
        enabled = peer_enabled[index] & (surplus[index] >= area * 0.01) & ~jnp.any(failures)

        def generate() -> tuple[Array, Array, Array, Array]:
            distance = jnp.sum((nodes - peer_positions[index]).astype(jnp.float32) ** 2, axis=1)
            goal = jnp.argmin(jnp.where(valid, distance, jnp.inf))
            route, count, found = _astar(nodes, valid, graph, current_index, goal, enabled=enabled)
            route, count, found = lax.cond(
                found,
                lambda: (route, count, found),
                lambda: _astar(
                    nodes, valid, graph | graph.T, current_index, goal, enabled=enabled & ~found
                ),
            )
            needs_repair = enabled & ~found
            route, count, found = lax.cond(
                needs_repair,
                lambda: _astar(
                    nodes, valid, fallback | fallback.T, current_index, goal, enabled=needs_repair
                ),
                lambda: (route, count, found),
            )
            updated = lax.cond(
                found,
                lambda: _inflate_route(
                    nodes,
                    valid,
                    local_map,
                    route,
                    jnp.where(enabled & found, count, 0),
                    surplus[index] * 8 / area,
                    values,
                ),
                lambda: values,
            )
            return (
                updated,
                failures.at[index].set(~found),
                did_repair | needs_repair,
                jnp.where(needs_repair, fallback, graph),
            )

        return lax.cond(enabled, generate, lambda: state)

    utility, unreachable, repaired, _ = lax.fori_loop(
        0, len(peer_positions), peer, (utility, unreachable, repaired, edges)
    )
    own_radius = jnp.sum((nodes - nodes[current_index]) ** 2, axis=1) <= 30**2
    utility = jnp.where(valid & ~(own_radius & ~jnp.any(unreachable)), utility, 0)
    return RendezvousResult(utility, unreachable, repaired)
