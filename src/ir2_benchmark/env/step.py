"""Ordered events: lock intention, move, sense, transmit from the new position, score."""

import jax
import jax.numpy as jnp
from jax import Array, lax

from ir2_benchmark.codecs.packet import Codec, Packet, make_packet
from ir2_benchmark.codecs.standard import (
    encode_goal,
    encode_map_delta,
    encode_plan,
    encode_position,
    select_map_delta,
)
from ir2_benchmark.communication.channel import broadcast, consume_inbox
from ir2_benchmark.communication.faults import fault_links
from ir2_benchmark.communication.propagation import LinkContext
from ir2_benchmark.communication.sending import prepare_message
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.rewards import RewardContext
from ir2_benchmark.navigation.ir2_graph import find_frontier, update_graph
from ir2_benchmark.navigation.jax_path import NavigationResult, navigate
from ir2_benchmark.navigation.one_hop import merge_nodes, navigate_edge
from ir2_benchmark.presets import resolve_preset
from ir2_benchmark.sensing.ir2_sensor import sense_ir2
from ir2_benchmark.sensing.jax_sensor import build_sensor_geometry, sense
from ir2_benchmark.types import Action, EnvState, LocalObservation, StepInfo, StepResult


def lock_packet(  # noqa: PLR0913, PLR0917
    action: Action,
    observation: LocalObservation,
    route: NavigationResult,
    sequence: Array,
    cursor: Array,
    config: HostConfig,
) -> tuple[Packet, Array]:
    """Every standard byte comes from decision-time local knowledge and intention."""
    args = (observation.agent_id, sequence, observation.event, config.n_agents)

    def plan() -> Packet:
        return encode_plan(
            observation.position, route.goal, route.plan, route.plan_valid_length, *args
        )

    def position_payload() -> tuple[Packet, Array]:
        return encode_position(observation.position, *args), cursor

    def goal_payload() -> tuple[Packet, Array]:
        return encode_goal(route.goal, *args), cursor

    def plan_payload() -> tuple[Packet, Array]:
        return plan(), cursor

    def map_payload() -> tuple[Packet, Array]:
        selected = select_map_delta(observation.local_map, observation.observed_event, cursor)
        packet = encode_map_delta(selected.coordinates, selected.states, selected.count, *args)
        return packet, selected.next_cursor

    def custom_payload() -> tuple[Packet, Array]:
        return make_packet(action.codec, action.payload, action.payload_length, *args), cursor

    index = jnp.where(
        (action.codec >= Codec.POSITION) & (action.codec <= Codec.MAP_DELTA), action.codec - 1, 4
    )
    packet, next_cursor = lax.switch(
        index, (position_payload, goal_payload, plan_payload, map_payload, custom_payload)
    )
    if config.exchange_mode == "addressed":
        if action.recipients is None or action.recipients.shape != (config.n_slots,):
            raise ValueError("addressed exchange requires one boolean recipient per robot slot")
        mask = jnp.sum(action.recipients.astype(jnp.uint32) << jnp.arange(config.n_slots))
        packet = packet._replace(buffer=packet.buffer.at[14].set(mask.astype(jnp.uint8)))
    elif action.recipients is not None:
        raise ValueError("explicit recipients require exchange_mode='addressed'")
    return packet, next_cursor


def _coverage(state: EnvState) -> Array:
    """Compute physical coverage only on the host-verified scoring component."""
    return jnp.sum(state.covered & state.component) / jnp.sum(state.component)


def _advance(
    state: EnvState,
    action: Action,
    config: HostConfig,
    key: Array,
    observation: LocalObservation | None,
) -> StepResult:
    agent = state.current_agent
    local = observe(state, agent, config) if observation is None else observation
    route = (
        navigate_edge(local.position, action.waypoint, local.candidates, config.plan_horizon)
        if config.navigation_mode != "grid_path"
        else navigate(
            local.local_map,
            local.position,
            action.waypoint,
            config.max_distance,
            config.plan_horizon,
            local.tree,
        )
    )
    graph_nodes, graph_count = merge_nodes(
        local.graph_nodes, local.graph_count, route.position[None], jnp.ones(1, bool)
    )
    message_intent = prepare_message(local, action, config)
    packet, next_cursor = lock_packet(
        message_intent, local, route, state.sequences[agent], state.map_cursors[agent], config
    )
    positions = state.positions.at[agent].set(route.position)
    if config.navigation_mode == "ir2_graph":
        sensed = sense_ir2(
            state.truth, local.local_map, route.position, config.sensor_radius, state.valid_shape
        )
    else:
        sensed = sense(
            state.truth,
            local.local_map,
            route.position,
            build_sensor_geometry(config.sensor_radius),
            state.valid_shape,
        )
    graph_memory = state.graph_memory
    peer_maps = state.peer_maps
    if config.navigation_mode == "ir2_graph":
        memory = update_graph(
            local.graph_memory,
            sensed.local_map,
            route.position,
            state.valid_shape,
            local.peer_positions,
            (local.peer_position_event >= 0) & (jnp.arange(config.n_slots) != agent),
            jnp.zeros((1, 2), jnp.int32),
            jnp.zeros(1, bool),
            config=config,
        )
        graph_memory = jax.tree.map(
            lambda full, part: full.at[agent].set(part), graph_memory, memory
        )
        peer_maps = peer_maps.at[agent].set(local.peer_maps)
    physical = state.physical_seen.at[agent].set(state.physical_seen[agent] | sensed.observed_free)
    covered = state.covered | (sensed.observed_free & state.component)
    signal_function = resolve_preset("signal", config.signal_model)
    episode_key = jax.random.fold_in(jax.random.fold_in(state.key, state.episode), 0x53534E)
    event_key = jax.random.fold_in(jax.random.fold_in(key, state.event), 0x534947)
    signals = jax.vmap(
        lambda receiver, receiver_position: signal_function(
            LinkContext(
                state.truth,
                route.position,
                receiver_position,
                jax.random.fold_in(event_key, receiver),
                episode_key,
            ),
            config,
        )
    )(jnp.arange(config.n_slots), positions)
    base = signals.reachable
    burst = (state.round >= config.burst_start) & (
        state.round < config.burst_start + config.burst_duration
    )
    external = fault_links(
        key,
        config.fault_mode,
        state.episode,
        state.event,
        agent,
        config.n_slots,
        config.dropout_probability,
        state.isolated_mask,
        burst,
    )
    sent = broadcast(
        consume_inbox(state.inbox, agent),
        packet,
        agent,
        state.balances[agent],
        message_intent.send,
        base,
        external,
        state.active_mask,
        config.plan_horizon,
        config.charge_mode,
        config.exchange_mode,
    )
    event = state.event + 1
    next_state = state._replace(
        positions=positions,
        local_maps=state.local_maps.at[agent].set(sensed.local_map),
        observed_event=state.observed_event.at[agent].set(
            jnp.where(sensed.visible, state.event, local.observed_event)
        ),
        physical_seen=physical,
        covered=covered,
        balances=state.balances.at[agent].set(sent.balance),
        inbox=sent.inbox,
        goals=state.goals.at[agent].set(route.goal),
        plans=state.plans.at[agent].set(route.plan),
        plan_lengths=state.plan_lengths.at[agent].set(route.plan_valid_length),
        plan_generated_event=state.plan_generated_event.at[agent].set(state.event),
        sequences=state.sequences.at[agent].add(sent.eligible.astype(jnp.int32)),
        map_cursors=state.map_cursors.at[agent].set(
            jnp.where(sent.eligible, next_cursor, state.map_cursors[agent])
        ),
        peer_positions=state.peer_positions.at[agent].set(local.peer_positions),
        peer_goals=state.peer_goals.at[agent].set(local.peer_goals),
        peer_plans=state.peer_plans.at[agent].set(local.peer_plans),
        peer_plan_lengths=state.peer_plan_lengths.at[agent].set(local.peer_plan_lengths),
        peer_position_event=state.peer_position_event.at[agent].set(local.peer_position_event),
        peer_goal_event=state.peer_goal_event.at[agent].set(local.peer_goal_event),
        peer_plan_event=state.peer_plan_event.at[agent].set(local.peer_plan_event),
        event=event,
        round=event // config.n_agents,
        current_agent=event % config.n_agents,
        spent_bytes=state.spent_bytes.at[agent].add(sent.spent),
        travel_distance=state.travel_distance.at[agent].add(route.distance),
        graph_nodes=state.graph_nodes.at[agent].set(graph_nodes),
        graph_count=state.graph_count.at[agent].set(graph_count),
        graph_memory=graph_memory,
        peer_maps=peer_maps,
    )
    before, after = _coverage(state), _coverage(next_state)
    terminated = after >= config.coverage_target
    truncated = (event >= config.n_agents * config.max_rounds) & ~terminated
    next_state = next_state._replace(terminated=terminated, truncated=truncated)
    observed_frontiers, explore_utility, meeting_utility = (jnp.float32(0),) * 3
    if config.reward_model == "ir2_individual" or config.reward_graph_statistics:
        old_merged = jnp.min(
            jnp.where(state.active_mask[:, None, None], state.local_maps, 2), axis=0
        )
        new_merged = jnp.min(
            jnp.where(state.active_mask[:, None, None], next_state.local_maps, 2), axis=0
        )
        _, old_frontiers = find_frontier(old_merged, state.valid_shape)
        _, new_frontiers = find_frontier(new_merged, state.valid_shape)
        observed_frontiers = jnp.sum(old_frontiers & ~new_frontiers).astype(jnp.float32)
        matches = local.graph_memory.valid & jnp.all(
            local.graph_memory.nodes == route.position, axis=-1
        )
        selected = jnp.argmax(matches)
        explore_utility = jnp.where(jnp.any(matches), local.graph_memory.utility[selected], 0.0)
        meeting_utility = jnp.where(jnp.any(matches), local.graph.node_inputs[selected, 3], 0.0)
    reward_result = resolve_preset("reward", config.reward_model)(
        RewardContext(
            before,
            after,
            route.distance,
            sent.spent,
            terminated,
            observed_frontiers,
            explore_utility,
            meeting_utility,
        ),
        config,
    )
    info = StepInfo(
        agent,
        jnp.bool_(1),
        route.invalid_action,
        route.distance.astype(jnp.float32),
        sent.spent,
        sent.delivered,
        packet.buffer,
        packet.length,
        before,
        after,
        reward_result.terms,
        base & state.active_mask & (jnp.arange(config.n_slots) != agent),
        external & state.active_mask & (jnp.arange(config.n_slots) != agent),
        signals.strength,
        signals.strength_valid & state.active_mask & (jnp.arange(config.n_slots) != agent),
    )
    return StepResult(next_state, reward_result.reward, terminated, truncated, info)


def step_agent(
    state: EnvState,
    action: Action,
    config: HostConfig,
    key: Array | None = None,
    observation: LocalObservation | None = None,
) -> StepResult:
    """Execute one event; an optional observation must belong to this state and actor.

    The optional argument reuses a freshly computed BFS, never an earlier event's
    cached observation. Callers must retain the observe-policy-step event order.
    """

    def frozen() -> StepResult:
        coverage = _coverage(state)
        info = StepInfo(
            state.current_agent,
            jnp.bool_(0),
            jnp.bool_(0),
            jnp.float32(0),
            jnp.int32(0),
            jnp.zeros(config.n_slots, bool),
            jnp.zeros(256, jnp.uint8),
            jnp.int32(0),
            coverage,
            coverage,
            jnp.zeros(6, jnp.float32),
            jnp.zeros(config.n_slots, bool),
            jnp.zeros(config.n_slots, bool),
            jnp.zeros(config.n_slots, jnp.float32),
            jnp.zeros(config.n_slots, bool),
        )
        return StepResult(state, jnp.float32(0), state.terminated, state.truncated, info)

    return lax.cond(
        state.terminated | state.truncated,
        frozen,
        lambda: _advance(state, action, config, state.key if key is None else key, observation),
    )
