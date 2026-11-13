"""Read-only local observations; unread paid messages are overlaid without consumption."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from ir2_benchmark.codecs.packet import Codec, Packet, decode_payload_origin
from ir2_benchmark.codecs.standard import decode_map_delta, decode_plan, decode_position
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.ir2_observation import ir2_observation
from ir2_benchmark.navigation.candidates import waypoint_candidates
from ir2_benchmark.navigation.jax_path import bfs
from ir2_benchmark.navigation.one_hop import graph_candidates, merge_nodes
from ir2_benchmark.types import EnvState, LocalFields, LocalObservation


class LocalKnowledge(NamedTuple):
    """Persistent local maps and paid peer estimates, with no privileged fields."""

    local_map: Array
    observed_event: Array
    peer_positions: Array
    peer_goals: Array
    peer_plans: Array
    peer_plan_lengths: Array
    peer_position_event: Array
    peer_goal_event: Array
    peer_plan_event: Array
    peer_maps: Array | None = None


def merge_local_messages(  # noqa: PLR0913 - real map bounds differ from static array shape
    initial: LocalKnowledge,
    buffers: Array,
    lengths: Array,
    present: Array,
    config: HostConfig,
    *,
    valid_shape: Array | None = None,
) -> LocalKnowledge:
    """Overlay paid packets within real map bounds, excluding static bucket padding.

    Omitting valid_shape treats the array as an unpadded map. Internal callers
    always pass the real geometry, including replay and imagined observations.
    """
    limits = jnp.asarray(initial.local_map.shape if valid_shape is None else valid_shape)

    def merge(sender: int, known: LocalKnowledge) -> LocalKnowledge:
        packet = Packet(buffers[sender], lengths[sender])
        origin = decode_payload_origin(packet)
        author, event = origin.author.astype(jnp.int32), origin.generated_event.astype(jnp.int32)
        point = decode_position(packet)
        plan = decode_plan(packet, config.plan_horizon)
        position_new = ((origin.codec == Codec.POSITION) | (origin.codec == Codec.PLAN)) & (
            event > known.peer_position_event[author]
        )
        goal_new = ((origin.codec == Codec.GOAL) | (origin.codec == Codec.PLAN)) & (
            event > known.peer_goal_event[author]
        )
        plan_new = (origin.codec == Codec.PLAN) & (event > known.peer_plan_event[author])
        positions = known.peer_positions.at[author].set(
            jnp.where(position_new, point, known.peer_positions[author])
        )
        goals = known.peer_goals.at[author].set(
            jnp.where(
                goal_new,
                jnp.where(origin.codec == Codec.PLAN, plan.goal, point),
                known.peer_goals[author],
            )
        )
        plans = known.peer_plans.at[author].set(
            jnp.where(plan_new, plan.positions, known.peer_plans[author])
        )
        plan_lengths = known.peer_plan_lengths.at[author].set(
            jnp.where(plan_new, plan.valid_length, known.peer_plan_lengths[author])
        )
        delta = decode_map_delta(packet)
        height, width = known.local_map.shape

        def cell(index: int, grids: tuple[Array, Array]) -> tuple[Array, Array]:
            local_map, times = grids
            x, y = delta.coordinates[index]
            valid = (
                delta.valid_mask[index]
                & (origin.codec == Codec.MAP_DELTA)
                & (x >= 0)
                & (x < limits[1])
                & (y >= 0)
                & (y < limits[0])
            )
            cx, cy = jnp.clip(x, 0, width - 1), jnp.clip(y, 0, height - 1)
            update = valid & (event >= times[cy, cx])
            value = delta.states[index].astype(local_map.dtype)
            return (
                local_map.at[cy, cx].set(jnp.where(update, value, local_map[cy, cx])),
                times.at[cy, cx].set(jnp.where(update, event, times[cy, cx])),
            )

        local_map, times = lax.fori_loop(0, 47, cell, (known.local_map, known.observed_event))
        peer_maps = known.peer_maps
        if peer_maps is not None:

            def peer_cell(index: int, maps: Array) -> Array:
                x, y = delta.coordinates[index]
                valid = (
                    delta.valid_mask[index]
                    & (origin.codec == Codec.MAP_DELTA)
                    & (x >= 0)
                    & (x < limits[1])
                    & (y >= 0)
                    & (y < limits[0])
                )
                cx, cy = jnp.clip(x, 0, width - 1), jnp.clip(y, 0, height - 1)
                value = jnp.where(valid, delta.states[index], maps[author, cy, cx])
                return maps.at[author, cy, cx].set(value.astype(maps.dtype))

            peer_maps = lax.fori_loop(0, 47, peer_cell, peer_maps)
        return LocalKnowledge(
            local_map,
            times,
            positions,
            goals,
            plans,
            plan_lengths,
            known.peer_position_event.at[author].set(
                jnp.where(position_new, event, known.peer_position_event[author])
            ),
            known.peer_goal_event.at[author].set(
                jnp.where(goal_new, event, known.peer_goal_event[author])
            ),
            known.peer_plan_event.at[author].set(
                jnp.where(plan_new, event, known.peer_plan_event[author])
            ),
            peer_maps,
        )

    return lax.fori_loop(
        0,
        config.n_slots,
        lambda sender, known: lax.cond(
            present[sender], lambda value: merge(sender, value), lambda value: value, known
        ),
        initial,
    )


def observation_from_local_fields(fields: LocalFields, config: HostConfig) -> LocalObservation:
    """Derive navigation features from public cognitive fields, never an EnvState."""
    if config.navigation_mode == "ir2_graph":
        return ir2_observation(fields, config)
    if config.navigation_mode == "ir2_one_hop":
        nodes, count = merge_nodes(
            fields.graph_nodes, fields.graph_count, fields.position[None], jnp.ones(1, bool)
        )
        if config.global_graph_fusion:
            # Only explicitly received map cells may introduce teammate graph nodes.
            x = jnp.rint(
                jnp.linspace(0, fields.valid_shape[1] - 1, config.graph_nodes_per_axis)
            ).astype(jnp.int32)
            y = jnp.rint(
                jnp.linspace(0, fields.valid_shape[0] - 1, config.graph_nodes_per_axis)
            ).astype(jnp.int32)

            def receive(sender: int, history: tuple[Array, Array]) -> tuple[Array, Array]:
                packet = Packet(fields.inbox_buffers[sender], fields.inbox_lengths[sender])
                origin = decode_payload_origin(packet)
                delta = decode_map_delta(packet)
                points = delta.coordinates
                present = (
                    delta.valid_mask
                    & fields.inbox_present[sender]
                    & (origin.codec == Codec.MAP_DELTA)
                )
                present &= (delta.states == 0) & jnp.all(
                    (points >= 0) & (points < fields.valid_shape[::-1]), axis=-1
                )
                present &= jnp.any(points[:, 0, None] == x, axis=-1) & jnp.any(
                    points[:, 1, None] == y, axis=-1
                )
                present &= (
                    fields.local_map[
                        points[:, 1].clip(0, fields.local_map.shape[0] - 1),
                        points[:, 0].clip(0, fields.local_map.shape[1] - 1),
                    ]
                    == 0
                )
                return merge_nodes(*history, points, present)

            nodes, count = lax.fori_loop(0, config.n_slots, receive, (nodes, count))
        fields = fields._replace(graph_nodes=nodes, graph_count=count)
        candidates = graph_candidates(
            fields.local_map,
            fields.position,
            fields.valid_shape,
            nodes,
            count,
            max_candidates=config.max_candidates,
            nodes_per_axis=config.graph_nodes_per_axis,
            local_radius=config.graph_local_radius,
            global_radius=config.graph_global_radius,
        )
        return LocalObservation(*fields, candidates, None)
    tree = bfs(fields.local_map, fields.position)
    candidates = waypoint_candidates(
        fields.local_map, fields.position, fields.own_goal, config.max_candidates, tree
    )
    return LocalObservation(*fields, candidates, tree)


def observe(state: EnvState, agent_id: Array, config: HostConfig) -> LocalObservation:
    """Return a deterministic observation without consuming inboxes or touching truth."""
    initial = LocalKnowledge(
        state.local_maps[agent_id],
        state.observed_event[agent_id],
        state.peer_positions[agent_id],
        state.peer_goals[agent_id],
        state.peer_plans[agent_id],
        state.peer_plan_lengths[agent_id],
        state.peer_position_event[agent_id],
        state.peer_goal_event[agent_id],
        state.peer_plan_event[agent_id],
        None if state.peer_maps is None else state.peer_maps[agent_id],
    )
    known = merge_local_messages(
        initial,
        state.inbox.buffers[agent_id],
        state.inbox.lengths[agent_id],
        state.inbox.present[agent_id],
        config,
        valid_shape=state.valid_shape,
    )
    ages = jnp.where(
        state.inbox.received_event[agent_id] >= 0,
        state.event - state.inbox.received_event[agent_id],
        -1,
    )
    fields = LocalFields(
        agent_id,
        state.event,
        state.round,
        state.positions[agent_id],
        state.valid_shape,
        state.balances[agent_id],
        known.local_map,
        known.observed_event,
        state.goals[agent_id],
        state.inbox.buffers[agent_id],
        state.inbox.lengths[agent_id],
        state.inbox.present[agent_id],
        ages,
        known.peer_positions,
        known.peer_goals,
        known.peer_plans,
        known.peer_plan_lengths,
        known.peer_position_event,
        known.peer_goal_event,
        known.peer_plan_event,
        known.peer_position_event > state.peer_position_event[agent_id],
        known.peer_goal_event > state.peer_goal_event[agent_id],
        known.peer_plan_event > state.peer_plan_event[agent_id],
        state.graph_nodes[agent_id],
        state.graph_count[agent_id],
        None
        if state.graph_memory is None
        else jax.tree.map(lambda value: value[agent_id], state.graph_memory),
        known.peer_maps,
    )
    return observation_from_local_fields(fields, config)
