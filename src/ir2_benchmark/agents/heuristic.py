"""Nearest frontier by actual local graph distance, with deterministic coordinate ties."""

import jax.numpy as jnp
from jax import Array, lax

from ir2_benchmark.agents.fixed_comm import fixed_communication
from ir2_benchmark.config import HostConfig
from ir2_benchmark.navigation.candidates import frontier_mask
from ir2_benchmark.types import Action, LocalObservation, make_action


def _graph_waypoint(observation: LocalObservation) -> Array:
    """Dijkstra on the receiver's graph, seeded only through legal first actions."""
    graph = observation.graph
    if graph is None:
        return observation.position
    nodes = graph.node_inputs[:, :2]
    valid = ~graph.node_padding_mask.astype(bool)
    candidates = observation.candidates
    count, slots = nodes.shape[0], graph.edge_inputs.shape[0]
    node_order = jnp.lexsort((jnp.arange(count), nodes[:, 1], nodes[:, 0]))
    node_rank = jnp.empty(count, jnp.int32).at[node_order].set(jnp.arange(count))
    points = candidates.positions
    slot_order = jnp.lexsort((jnp.arange(slots), points[:, 1], points[:, 0]))
    slot_rank = jnp.empty(slots, jnp.int32).at[slot_order].set(jnp.arange(slots))
    legal = candidates.mask & (candidates.distances > 0)
    legal &= ~graph.edge_padding_mask.astype(bool)
    legal &= ~graph.edge_mask[graph.current_index, graph.edge_inputs].astype(bool)
    first = jnp.min(
        jnp.where(
            (jnp.arange(count)[:, None] == graph.edge_inputs[None]) & legal[None],
            slot_rank[None],
            slots,
        ),
        axis=1,
    )
    distances = jnp.where(
        valid & (first < slots),
        jnp.linalg.norm(nodes - nodes[graph.current_index], axis=1),
        jnp.inf,
    )
    # Every future observation suppresses node zero as an action destination.
    visited = ~valid | (jnp.arange(count) == graph.current_index) | (jnp.arange(count) == 0)
    frontier = valid & (graph.node_inputs[:, 2] > 0)

    def pending(carry: tuple[Array, Array, Array, Array]) -> Array:
        distance, _first, done, selected = carry
        return (selected == slots) & jnp.any(~done & jnp.isfinite(distance))

    def expand(carry: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        distance, first_hop, done, selected = carry
        scores = jnp.where(done, jnp.inf, distance)
        index = jnp.argmin(jnp.where(scores == jnp.min(scores), node_rank, count))
        onward = distance[index] + jnp.linalg.norm(nodes - nodes[index], axis=1)
        reachable = ~done & ~graph.edge_mask[index].astype(bool)
        better = (onward < distance) | ((onward == distance) & (first_hop[index] < first_hop))
        improved = reachable & better
        return (
            jnp.where(improved, onward, distance),
            jnp.where(improved, first_hop[index], first_hop),
            done.at[index].set(True),
            jnp.where(frontier[index], first_hop[index], selected),
        )

    _, _, _, selected = lax.while_loop(
        pending, expand, (distances, first, visited, jnp.asarray(slots, jnp.int32))
    )
    return jnp.where(
        selected < slots, points[slot_order[jnp.minimum(selected, slots - 1)]], observation.position
    )


def nearest_frontier_action(observation: LocalObservation, _config: HostConfig) -> Action:
    """Use offered graph edges; retain the explicit legacy grid reference policy."""
    if observation.graph is not None:
        send, codec = fixed_communication(observation)
        return make_action(_graph_waypoint(observation), send, codec)
    if observation.tree is None:
        candidates = observation.candidates
        frontier = frontier_mask(observation.local_map, observation.valid_shape)
        points = candidates.positions
        eligible = candidates.mask & (candidates.distances > 0)
        at_frontier = eligible & frontier[points[:, 1], points[:, 0]]
        nearest = jnp.argmin(jnp.where(at_frontier, candidates.distances, jnp.inf))
        fallback = jnp.argmin(jnp.where(eligible, candidates.distances, jnp.inf))
        index = jnp.where(jnp.any(at_frontier), nearest, fallback)
        send, codec = fixed_communication(observation)
        waypoint = jnp.where(jnp.any(eligible), points[index], observation.position)
        return make_action(waypoint, send, codec)
    distances = observation.tree.distances
    eligible = frontier_mask(observation.local_map, observation.valid_shape) & (distances > 0)
    scores = jnp.where(eligible, distances, jnp.iinfo(jnp.int32).max)
    index = jnp.argmin(scores.reshape(-1))
    width = observation.local_map.shape[1]
    goal = jnp.where(
        jnp.any(eligible), jnp.array([index % width, index // width]), observation.position
    )
    send, codec = fixed_communication(observation)
    return make_action(goal, send, codec)
