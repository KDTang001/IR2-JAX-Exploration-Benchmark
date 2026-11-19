"""IR2 graph observations derived exclusively from receiver-owned knowledge."""

import jax
import jax.numpy as jnp
from jax import Array, lax

from ir2_benchmark.codecs.packet import Codec, Packet, decode_payload_origin
from ir2_benchmark.codecs.standard import decode_map_delta
from ir2_benchmark.config import HostConfig
from ir2_benchmark.maps.prepare import UNKNOWN
from ir2_benchmark.models.ir2_policy import action_padding_mask
from ir2_benchmark.navigation.candidates import CandidateSet
from ir2_benchmark.navigation.ir2_graph import (
    GraphMemory,
    graph_observation,
    rebuild_graph_edges,
    update_graph,
)
from ir2_benchmark.navigation.ir2_rendezvous import rendezvous_utility
from ir2_benchmark.types import LocalFields, LocalObservation


def delivered_map_nodes(fields: LocalFields, config: HostConfig) -> tuple[Array, Array]:
    """Only actual MAP_DELTA payload cells can introduce received graph nodes."""

    def decode(buffer: Array, length: Array, present: Array) -> tuple[Array, Array]:
        packet = Packet(buffer, length)
        delta = decode_map_delta(packet)
        enabled = present & (decode_payload_origin(packet).codec == Codec.MAP_DELTA)
        return delta.coordinates, delta.valid_mask & (delta.states == 0) & enabled

    points, present = jax.vmap(decode)(
        fields.inbox_buffers, fields.inbox_lengths, fields.inbox_present
    )
    points, present = points.reshape(-1, 2), present.reshape(-1)
    x = jnp.rint(jnp.linspace(0, fields.valid_shape[1] - 1, config.graph_nodes_per_axis))
    y = jnp.rint(jnp.linspace(0, fields.valid_shape[0] - 1, config.graph_nodes_per_axis))
    present &= jnp.all((points >= 0) & (points < fields.valid_shape[::-1]), axis=-1)
    present &= jnp.any(points[:, 0, None] == x, axis=-1)
    present &= jnp.any(points[:, 1, None] == y, axis=-1)
    present &= (
        fields.local_map[
            points[:, 1].clip(0, fields.local_map.shape[0] - 1),
            points[:, 0].clip(0, fields.local_map.shape[1] - 1),
        ]
        == 0
    )
    return points, present & config.global_graph_fusion


def ir2_observation(fields: LocalFields, config: HostConfig) -> LocalObservation:
    """Keep original graph features separate from budget/message extension fields."""
    if fields.graph_memory is None or fields.peer_maps is None:
        detail = "IR2 graph mode requires persistent graph and received map state"
        raise ValueError(detail)
    peers = fields.peer_positions
    peer_valid = (fields.peer_position_event >= 0) & (jnp.arange(config.n_slots) != fields.agent_id)
    incoming, incoming_valid = delivered_map_nodes(fields, config)

    def refresh(carry: tuple[Array, GraphMemory]) -> tuple[Array, GraphMemory]:
        _, current = carry
        updated = update_graph(
            current,
            fields.local_map,
            fields.position,
            fields.valid_shape,
            peers,
            peer_valid,
            incoming,
            incoming_valid,
            config=config,
            record_visit=False,
            extend_offshoots=False,
        )
        return jnp.bool_(0), updated

    # A one-pass while preserves an actual branch under vmap's conditional batching.
    _, memory = lax.while_loop(
        lambda carry: carry[0],
        refresh,
        (jnp.any(fields.inbox_present), fields.graph_memory),
    )
    graph = graph_observation(memory, fields.position, peers, peer_valid, config=config)
    # Original surplus counts positive differences in the 1/127/255 encoding.
    codes = jnp.array([255, 1, 127], jnp.int16)
    own_map = codes[fields.local_map]
    peer_maps = codes[fields.peer_maps]
    yy, xx = jnp.indices(fields.local_map.shape)
    inside = (yy < fields.valid_shape[0]) & (xx < fields.valid_shape[1])
    surplus = jnp.sum((own_map[None] > peer_maps) & inside[None], axis=(-2, -1))
    # PLAN-only contact supplies no map belief from which to compute IR2's surplus.
    has_peer_map = jnp.any((fields.peer_maps != UNKNOWN) & inside[None], axis=(-2, -1))
    surplus = jnp.where(has_peer_map, surplus, 0)
    # Contact evidence is a received packet, never a privileged connectivity graph.
    contacted = (fields.message_ages >= 0) & (fields.message_ages <= config.n_agents)
    eligible = peer_valid & ~contacted & (surplus >= jnp.prod(fields.valid_shape) * 0.01)
    abort_unreachable = config.rendezvous_mode == "strict_ir2"

    def meeting(current: GraphMemory) -> tuple[GraphMemory, Array]:
        repaired = rebuild_graph_edges(
            fields.local_map,
            current,
            config=config,
            global_min=0,
            global_max=10 * config.sensor_radius,
        )
        result = rendezvous_utility(
            current.nodes,
            current.valid,
            current.edges,
            graph.current_index,
            peers,
            eligible,
            surplus,
            fields.local_map,
            fields.valid_shape,
            repaired.edges,
            abort_unreachable=abort_unreachable,
        )
        current = jax.tree.map(
            lambda new, old: jnp.where(result.repaired, new, old), repaired, current
        )
        return current._replace(
            path_failed=current.path_failed | (abort_unreachable & jnp.any(result.unreachable))
        ), result.utility

    def meet(carry: tuple[Array, GraphMemory, Array]) -> tuple[Array, GraphMemory, Array]:
        _, current, _ = carry
        updated, values = meeting(current)
        return jnp.bool_(0), updated, values

    _, memory, utility = lax.while_loop(
        lambda carry: carry[0],
        meet,
        (jnp.any(eligible), memory, jnp.zeros(memory.nodes.shape[0], jnp.float32)),
    )
    graph = graph_observation(memory, fields.position, peers, peer_valid, config=config)
    graph = graph._replace(node_inputs=graph.node_inputs.at[:, 3].set(utility))
    points = memory.nodes[graph.edge_inputs]
    allowed = ~action_padding_mask(graph.edge_padding_mask)
    allowed &= ~(memory.overflow | memory.path_failed)
    distances = jnp.linalg.norm((points - fields.position).astype(jnp.float32), axis=-1)
    candidates = CandidateSet(points, allowed, jnp.where(allowed, distances, 0))
    fields = fields._replace(graph_memory=memory)
    return LocalObservation(*fields, candidates, None, graph)
