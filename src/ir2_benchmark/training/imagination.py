"""Closed public-state imagination with exact navigation, packet costs, and inbox semantics."""

from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import Array, lax

from ir2_benchmark.agents.actor import ActorOutput, IndexAction, sample_communication
from ir2_benchmark.agents.fixed_comm import fixed_communication
from ir2_benchmark.communication.channel import Inbox, broadcast, consume_inbox
from ir2_benchmark.communication.sending import prepare_message, validate_builtin_policy
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import (
    LocalKnowledge,
    merge_local_messages,
    observation_from_local_fields,
)
from ir2_benchmark.env.step import lock_packet
from ir2_benchmark.models.encoders import (
    ActorFeatures,
    actor_features,
    no_plan_estimate,
    selected_plan_features,
    zoh_estimate,
)
from ir2_benchmark.models.plan_predictor import PlanEstimate, PlanMemory, plan_inputs
from ir2_benchmark.models.world_model import RSSMState, WorldPrediction, latent_features
from ir2_benchmark.navigation.ir2_graph import update_graph
from ir2_benchmark.navigation.jax_path import navigate
from ir2_benchmark.navigation.one_hop import merge_nodes, navigate_edge
from ir2_benchmark.training.dreamer import ImaginedBatch
from ir2_benchmark.types import Action, EnvState, LocalFields, LocalObservation, make_action

if TYPE_CHECKING:
    from ir2_benchmark.navigation.ir2_graph import GraphMemory


class CognitiveState(NamedTuple):
    """Modeled public state: no truth, scoring component, physical coverage, or fault schedule."""

    positions: Array
    active_mask: Array
    local_maps: Array
    observed_event: Array
    valid_shape: Array
    balances: Array
    inbox: Inbox
    goals: Array
    plans: Array
    plan_lengths: Array
    plan_generated_event: Array
    sequences: Array
    map_cursors: Array
    peer_positions: Array
    peer_goals: Array
    peer_plans: Array
    peer_plan_lengths: Array
    peer_position_event: Array
    peer_goal_event: Array
    peer_plan_event: Array
    event: Array
    round: Array
    current_agent: Array
    terminated: Array
    truncated: Array
    spent_bytes: Array
    travel_distance: Array
    graph_nodes: Array
    graph_count: Array
    graph_memory: "GraphMemory | None"
    peer_maps: Array | None


class ImaginedState(NamedTuple):
    """Only the critic/model uses latent; each actor's memory remains local to that robot."""

    cognition: CognitiveState
    latent: RSSMState
    actor_memory: Array


class ImaginedTransition(NamedTuple):
    """Exact modeled bookkeeping and sampled dynamics from one ordered event."""

    state: ImaginedState
    reward: Array
    continuation: Array
    valid: Array
    spent: Array
    delivered: Array
    distance: Array


class PolicyCarry(NamedTuple):
    """Per-robot actor and per-receiver/neighbor predictor memory with no central latent."""

    actor_memory: Array
    plan_memory: PlanMemory


class PolicyDecision(NamedTuple):
    """Local sampled action and the exact actor cutoff retained for score-function learning."""

    action: Action
    index_action: IndexAction
    features: ActorFeatures
    old_memory: Array
    plan_features: Array
    carry: PolicyCarry
    estimate: PlanEstimate


ImaginedEvent: TypeAlias = tuple[
    ActorFeatures, Array, Array, IndexAction, Array, Array, Array, Array
]


def policy_decision(  # noqa: PLR0913, PLR0917
    actor: TrainState,
    predictor: TrainState,
    observation: LocalObservation,
    carry: PolicyCarry,
    config: HostConfig,
    key: Array,
    *,
    learned_communication: bool = True,
    predicted_plans: bool = True,
    use_neighbor_plan: bool = True,
    greedy: bool = False,
) -> PolicyDecision:
    """Shared real/imaged actor adapter; model latent and true neighbor state are not arguments."""
    plan_key, motion_key, communication_key = jax.random.split(key, 3)
    agent = observation.agent_id
    receiver_memory = jax.tree_util.tree_map(lambda value: value[agent], carry.plan_memory)
    prediction = predictor.apply_fn(
        {"params": predictor.params}, receiver_memory, plan_inputs(observation), plan_key
    )
    estimate = prediction.estimate if predicted_plans else zoh_estimate(observation)
    if not use_neighbor_plan:
        estimate = no_plan_estimate(observation)
    features = actor_features(
        observation,
        config.n_agents,
        estimate,
        initial_budget=config.initial_budget,
        max_rounds=config.max_rounds,
        patch_radius=int(config.sensor_radius),
        use_neighbor_plan=use_neighbor_plan,
        candidate_attention="neighbor_keys" in actor.params,
    )
    memory = carry.actor_memory[agent]
    # Flax methods are selected by name, avoiding a second policy implementation for route B.
    new_memory, logits = actor.apply_fn({"params": actor.params}, features, memory, method="encode")
    waypoint_index = (
        jnp.argmax(logits) if greedy else jax.random.categorical(motion_key, logits)
    ).astype(jnp.int32)
    waypoint = observation.candidates.positions[waypoint_index]
    plan = selected_plan_features(observation, waypoint, config.max_distance, config.plan_horizon)
    gate, types = actor.apply_fn({"params": actor.params}, new_memory, plan, method="communication")
    output = ActorOutput(new_memory, logits, gate, types)
    index = sample_communication(output, waypoint_index, communication_key)
    if greedy:
        index = IndexAction(waypoint_index, gate >= 0, jnp.argmax(types).astype(jnp.int32))
    if learned_communication:
        send, codec = index.send, index.message_type + 1
    else:
        send, codec = fixed_communication(observation)
        index = IndexAction(waypoint_index, send, codec - 1)
    validate_builtin_policy(config, learned_communication)
    action = prepare_message(observation, make_action(waypoint, send, codec), config)
    index = IndexAction(waypoint_index, action.send, action.codec - 1)
    next_plans = jax.tree_util.tree_map(
        lambda full, part: full.at[agent].set(part), carry.plan_memory, prediction.memory
    )
    return PolicyDecision(
        action,
        index,
        features,
        memory,
        plan,
        PolicyCarry(carry.actor_memory.at[agent].set(new_memory), next_plans),
        estimate,
    )


def cognitive_from_environment(state: EnvState) -> CognitiveState:
    """Copy the real event-before cognitive cutoff, omitting all privileged future inputs."""
    return CognitiveState(*(getattr(state, name) for name in CognitiveState._fields))


def imagined_observe(state: CognitiveState, config: HostConfig) -> LocalObservation:
    """Rebuild local knowledge/navigation every event, including newly received simulated maps."""
    agent = state.current_agent
    known = LocalKnowledge(
        state.local_maps[agent],
        state.observed_event[agent],
        state.peer_positions[agent],
        state.peer_goals[agent],
        state.peer_plans[agent],
        state.peer_plan_lengths[agent],
        state.peer_position_event[agent],
        state.peer_goal_event[agent],
        state.peer_plan_event[agent],
        None if state.peer_maps is None else state.peer_maps[agent],
    )
    known = merge_local_messages(
        known,
        state.inbox.buffers[agent],
        state.inbox.lengths[agent],
        state.inbox.present[agent],
        config,
    )
    ages = jnp.where(
        state.inbox.received_event[agent] >= 0, state.event - state.inbox.received_event[agent], -1
    )
    fields = LocalFields(
        agent,
        state.event,
        state.round,
        state.positions[agent],
        state.valid_shape,
        state.balances[agent],
        known.local_map,
        known.observed_event,
        state.goals[agent],
        state.inbox.buffers[agent],
        state.inbox.lengths[agent],
        state.inbox.present[agent],
        ages,
        known.peer_positions,
        known.peer_goals,
        known.peer_plans,
        known.peer_plan_lengths,
        known.peer_position_event,
        known.peer_goal_event,
        known.peer_plan_event,
        known.peer_position_event > state.peer_position_event[agent],
        known.peer_goal_event > state.peer_goal_event[agent],
        known.peer_plan_event > state.peer_plan_event[agent],
        state.graph_nodes[agent],
        state.graph_count[agent],
        None
        if state.graph_memory is None
        else jax.tree.map(lambda value: value[agent], state.graph_memory),
        known.peer_maps,
    )
    return observation_from_local_fields(fields, config)


def encode_world_action(observation: LocalObservation, action: Action, config: HostConfig) -> Array:
    """Use the same causal action descriptor for real model fitting and simulated transitions."""
    scale = jnp.maximum(observation.valid_shape[::-1] - 1, 1)
    return jnp.concatenate(
        (
            jax.nn.one_hot(observation.agent_id, config.n_slots),
            observation.position / scale,
            action.waypoint / scale,
            jnp.asarray(
                [
                    action.send,
                    observation.balance / max(config.initial_budget, 1),
                    observation.round / config.max_rounds,
                ]
            ),
            jax.nn.one_hot(action.codec - 1, 5),
        )
    )


def local_tile(local_map: Array, position: Array, radius: int) -> Array:
    """Read a fixed window with unknown exterior; no hidden real map is required."""
    padded = jnp.pad(local_map, radius, constant_values=2)
    return lax.dynamic_slice(padded, (position[1], position[0]), (2 * radius + 1, 2 * radius + 1))


def _update_tile(  # noqa: PLR0913, PLR0917
    local_map: Array,
    times: Array,
    position: Array,
    logits: Array,
    event: Array,
    valid_shape: Array,
    sensor_radius: float,
) -> tuple[Array, Array]:
    """Only predicted new observations change unknown cells; unmodeled map cells stay intact."""
    side = logits.shape[-2]
    radius = side // 2
    yy, xx = jnp.indices((side, side))
    x, y = position[0] + xx - radius, position[1] + yy - radius
    h, w = local_map.shape
    cx, cy = jnp.clip(x, 0, w - 1), jnp.clip(y, 0, h - 1)
    predicted = jnp.argmax(logits, axis=-1).astype(jnp.uint8)
    allowed = (
        (x >= 0)
        & (x < valid_shape[1])
        & (y >= 0)
        & (y < valid_shape[0])
        & ((xx - radius) ** 2 + (yy - radius) ** 2 <= sensor_radius**2)
        & (predicted < 2)  # noqa: PLR2004
        & (local_map[cy, cx] == 2)  # noqa: PLR2004
    )

    # Valid coordinates are unique; the one-past-end sentinel drops all invalid entries.
    indices = jnp.where(allowed, y * w + x, h * w).reshape(-1)
    updated = local_map.reshape(-1).at[indices].set(predicted.reshape(-1), mode="drop")
    observed = times.reshape(-1).at[indices].set(event, mode="drop")
    return updated.reshape(h, w), observed.reshape(h, w)


def apply_prediction(  # noqa: PLR0913, PLR0917
    initial: ImaginedState,
    action: Action,
    prediction: WorldPrediction,
    config: HostConfig,
    key: Array,
    observation: LocalObservation | None = None,
) -> ImaginedTransition:
    """Apply controlled model predictions through exact public rules, without calling real env.step."""  # noqa: E501

    def advance() -> ImaginedTransition:
        state = initial.cognition
        agent = state.current_agent
        obs = imagined_observe(state, config) if observation is None else observation
        message_intent = prepare_message(obs, action, config)
        route = (
            navigate_edge(obs.position, action.waypoint, obs.candidates, config.plan_horizon)
            if config.navigation_mode != "grid_path"
            else navigate(
                obs.local_map,
                obs.position,
                action.waypoint,
                config.max_distance,
                config.plan_horizon,
                obs.tree,
            )
        )
        packet, cursor = lock_packet(
            message_intent, obs, route, state.sequences[agent], state.map_cursors[agent], config
        )
        move_key, delivery_key, continuation_key = jax.random.split(key, 3)
        if config.navigation_mode != "grid_path":
            distance, position = route.distance, route.position
        else:
            possible = jnp.arange(config.max_distance + 1) <= jnp.minimum(
                route.path_length, config.max_distance
            )
            index = jax.random.categorical(
                move_key, jnp.where(possible, prediction.movement_logits, -jnp.inf)
            ).astype(jnp.int32)
            distance, position = index.astype(jnp.float32), route.path[index]
        graph_nodes, graph_count = merge_nodes(
            obs.graph_nodes, obs.graph_count, position[None], jnp.ones(1, bool)
        )
        local_map, observed = _update_tile(
            obs.local_map,
            obs.observed_event,
            position,
            prediction.local_logits,
            state.event,
            state.valid_shape,
            config.sensor_radius,
        )
        graph_memory, peer_maps = state.graph_memory, state.peer_maps
        if config.navigation_mode == "ir2_graph":
            memory = update_graph(
                obs.graph_memory,
                local_map,
                position,
                state.valid_shape,
                obs.peer_positions,
                (obs.peer_position_event >= 0) & (jnp.arange(config.n_slots) != agent),
                jnp.zeros((1, 2), jnp.int32),
                jnp.zeros(1, bool),
                config=config,
            )
            graph_memory = jax.tree.map(
                lambda full, part: full.at[agent].set(part), graph_memory, memory
            )
            peer_maps = peer_maps.at[agent].set(obs.peer_maps)
        reachable = jax.random.bernoulli(delivery_key, jax.nn.sigmoid(prediction.delivery_logits))
        sent = broadcast(
            consume_inbox(state.inbox, agent),
            packet,
            agent,
            state.balances[agent],
            message_intent.send,
            reachable,
            jnp.ones(config.n_slots, bool),
            state.active_mask,
            config.plan_horizon,
            config.charge_mode,
            config.exchange_mode,
        )
        event = state.event + 1
        continuation = jax.nn.sigmoid(prediction.continuation_logit)
        terminated = ~jax.random.bernoulli(continuation_key, continuation)
        truncated = (event >= config.max_rounds * config.n_agents) & ~terminated
        next_state = state._replace(
            positions=state.positions.at[agent].set(position),
            local_maps=state.local_maps.at[agent].set(local_map),
            observed_event=state.observed_event.at[agent].set(observed),
            balances=state.balances.at[agent].set(sent.balance),
            inbox=sent.inbox,
            goals=state.goals.at[agent].set(route.goal),
            plans=state.plans.at[agent].set(route.plan),
            plan_lengths=state.plan_lengths.at[agent].set(route.plan_valid_length),
            plan_generated_event=state.plan_generated_event.at[agent].set(state.event),
            sequences=state.sequences.at[agent].add(sent.eligible.astype(jnp.int32)),
            map_cursors=state.map_cursors.at[agent].set(
                jnp.where(sent.eligible, cursor, state.map_cursors[agent])
            ),
            peer_positions=state.peer_positions.at[agent].set(obs.peer_positions),
            peer_goals=state.peer_goals.at[agent].set(obs.peer_goals),
            peer_plans=state.peer_plans.at[agent].set(obs.peer_plans),
            peer_plan_lengths=state.peer_plan_lengths.at[agent].set(obs.peer_plan_lengths),
            peer_position_event=state.peer_position_event.at[agent].set(obs.peer_position_event),
            peer_goal_event=state.peer_goal_event.at[agent].set(obs.peer_goal_event),
            peer_plan_event=state.peer_plan_event.at[agent].set(obs.peer_plan_event),
            event=event,
            round=event // config.n_agents,
            current_agent=event % config.n_agents,
            terminated=terminated,
            truncated=truncated,
            spent_bytes=state.spent_bytes.at[agent].add(sent.spent),
            travel_distance=state.travel_distance.at[agent].add(distance),
            graph_nodes=state.graph_nodes.at[agent].set(graph_nodes),
            graph_count=state.graph_count.at[agent].set(graph_count),
            graph_memory=graph_memory,
            peer_maps=peer_maps,
        )
        # Survival is sampled once; training uses this binary outcome, never probability times outcome.  # noqa: E501
        discount_alive = (~terminated & ~truncated).astype(jnp.float32)
        return ImaginedTransition(
            initial._replace(cognition=next_state, latent=prediction.state),
            prediction.reward,
            discount_alive,
            jnp.bool_(True),  # noqa: FBT003
            sent.spent,
            sent.delivered,
            distance,
        )

    def frozen() -> ImaginedTransition:
        return ImaginedTransition(
            initial,
            jnp.float32(0),
            jnp.float32(0),
            jnp.bool_(False),  # noqa: FBT003
            jnp.int32(0),
            jnp.zeros(config.n_slots, bool),
            jnp.float32(0),
        )

    return lax.cond(initial.cognition.terminated | initial.cognition.truncated, frozen, advance)


def imagine_rollout(  # noqa: PLR0913, PLR0917
    initial: ImaginedState,
    carry: PolicyCarry,
    actor: TrainState,
    predictor: TrainState,
    world: TrainState,
    config: HostConfig,
    key: Array,
    events: int,
    *,
    learned_communication: bool = True,
    predicted_plans: bool = True,
    use_neighbor_plan: bool = True,
) -> tuple[ImaginedState, PolicyCarry, ImaginedBatch]:
    """Scan modeled events only; exact candidates and paid inboxes feed each next local actor."""

    def event(
        previous: tuple[ImaginedState, PolicyCarry], index: Array
    ) -> tuple[tuple[ImaginedState, PolicyCarry], ImaginedEvent]:
        state, policy_carry = previous
        event_key = jax.random.fold_in(key, index)
        policy_key, model_key, transition_key = jax.random.split(event_key, 3)
        obs = imagined_observe(state.cognition, config)
        decision = policy_decision(
            actor,
            predictor,
            obs,
            policy_carry,
            config,
            policy_key,
            learned_communication=learned_communication,
            predicted_plans=predicted_plans,
            use_neighbor_plan=use_neighbor_plan,
        )
        prediction = world.apply_fn(
            {"params": world.params},
            state.latent,
            encode_world_action(obs, decision.action, config),
            model_key,
            method="imagine",
        )
        transition = apply_prediction(
            state, decision.action, prediction, config, transition_key, obs
        )
        next_carry = jax.tree_util.tree_map(
            lambda new, old: jnp.where(transition.valid, new, old), decision.carry, policy_carry
        )
        next_state = transition.state._replace(actor_memory=next_carry.actor_memory)
        continuation = transition.continuation
        item = (
            decision.features,
            decision.old_memory,
            decision.plan_features,
            decision.index_action,
            latent_features(state.latent),
            transition.reward,
            continuation,
            transition.valid,
        )
        return (next_state, next_carry), item

    (final, final_carry), items = lax.scan(event, (initial, carry), jnp.arange(events))
    features, memories, plans, actions, central, rewards, continuation, valid = items
    return (
        final,
        final_carry,
        ImaginedBatch(
            features,
            memories,
            plans,
            actions,
            central,
            latent_features(final.latent),
            rewards,
            continuation,
            valid,
        ),
    )
