"""Separate decentralized observations, explicit environment state and training labels."""

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from ir2_benchmark.codecs.packet import Codec
from ir2_benchmark.communication.channel import Inbox
from ir2_benchmark.navigation.candidates import CandidateSet
from ir2_benchmark.navigation.jax_path import SearchTree

if TYPE_CHECKING:
    from ir2_benchmark.navigation.ir2_graph import GraphMemory, GraphObservation


class MapInput(NamedTuple):
    """Device-ready binary occupancy and host-verified scoring component."""

    occupancy: Array
    component: Array
    valid_shape: Array
    anchor: Array


class Action(NamedTuple):
    """An arbitrary local waypoint and one optional paid packet intent."""

    waypoint: Array
    send: Array
    codec: Array
    payload: Array
    payload_length: Array
    recipients: Array | None = None


def make_action(
    waypoint: Array,
    send: Array = False,  # noqa: FBT002
    codec: Array = Codec.PLAN,
    payload: Array | None = None,
    payload_length: Array = 0,
    recipients: Array | None = None,
) -> Action:
    """Construct an action; standard codecs ignore custom payload fields."""
    return Action(
        jnp.asarray(waypoint, jnp.int32),
        jnp.asarray(send, bool),
        jnp.asarray(codec, jnp.int32),
        jnp.zeros(240, jnp.uint8) if payload is None else payload,
        jnp.asarray(payload_length, jnp.int32),
        None if recipients is None else jnp.asarray(recipients, bool),
    )


class LocalFields(NamedTuple):
    """Public cognitive state, usable by real and imagined local observations."""

    agent_id: Array
    event: Array
    round: Array
    position: Array
    valid_shape: Array
    balance: Array
    local_map: Array
    observed_event: Array
    own_goal: Array
    inbox_buffers: Array
    inbox_lengths: Array
    inbox_present: Array
    message_ages: Array
    peer_positions: Array
    peer_goals: Array
    peer_plans: Array
    peer_plan_lengths: Array
    peer_position_event: Array
    peer_goal_event: Array
    peer_plan_event: Array
    fresh_position: Array
    fresh_goal: Array
    fresh_plan: Array
    graph_nodes: Array
    graph_count: Array
    graph_memory: "GraphMemory | None"
    peer_maps: Array | None


class LocalObservation(NamedTuple):
    """Only own state and information learned through paid incoming packets."""

    agent_id: Array
    event: Array
    round: Array
    position: Array
    valid_shape: Array
    balance: Array
    local_map: Array
    observed_event: Array
    own_goal: Array
    inbox_buffers: Array
    inbox_lengths: Array
    inbox_present: Array
    message_ages: Array
    peer_positions: Array
    peer_goals: Array
    peer_plans: Array
    peer_plan_lengths: Array
    peer_position_event: Array
    peer_goal_event: Array
    peer_plan_event: Array
    fresh_position: Array
    fresh_goal: Array
    fresh_plan: Array
    graph_nodes: Array
    graph_count: Array
    graph_memory: "GraphMemory | None"
    peer_maps: Array | None
    candidates: CandidateSet
    tree: SearchTree | None
    graph: "GraphObservation | None" = None


class EnvState(NamedTuple):
    """Complete immutable simulator state; never pass this structure to a local actor."""

    truth: Array
    component: Array
    valid_shape: Array
    positions: Array
    active_mask: Array
    local_maps: Array
    observed_event: Array
    physical_seen: Array
    covered: Array
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
    key: Array
    episode: Array
    isolated_mask: Array
    spent_bytes: Array
    travel_distance: Array
    graph_nodes: Array
    graph_count: Array
    graph_memory: "GraphMemory | None" = None
    peer_maps: Array | None = None


class PlanLabel(NamedTuple):
    """Training-only current intentions, distinct from later actual trajectories."""

    goals: Array
    positions: Array
    valid_lengths: Array
    generated_event: Array


class PrivilegedState(NamedTuple):
    """Explicit centralized critic input, kept separate from decentralized observations."""

    truth: Array
    component: Array
    positions: Array
    active_mask: Array
    covered: Array
    balances: Array
    event: Array
    round: Array


class StepInfo(NamedTuple):
    """Execution and accounting fields for logging, never an actor observation."""

    agent_id: Array
    active: Array
    invalid_action: Array
    distance: Array
    spent: Array
    delivered: Array
    packet_buffer: Array
    packet_length: Array
    coverage_before: Array
    coverage_after: Array
    reward_terms: Array | None = None
    physical_links: Array | None = None
    external_links: Array | None = None
    signal_strength: Array | None = None
    signal_strength_valid: Array | None = None


class StepResult(NamedTuple):
    """A single ordered robot event, with termination and time limit kept distinct."""

    state: EnvState
    reward: Array
    terminated: Array
    truncated: Array
    info: StepInfo


def privileged_state(state: EnvState) -> PrivilegedState:
    """Build centralized features explicitly rather than extending local observations."""
    return PrivilegedState(
        state.truth,
        state.component,
        state.positions,
        state.active_mask,
        state.covered,
        state.balances,
        state.event,
        state.round,
    )


def plan_labels(state: EnvState) -> PlanLabel:
    """Expose latest generated intentions only to the training loss/logger."""
    return PlanLabel(state.goals, state.plans, state.plan_lengths, state.plan_generated_event)


def validate_graph_state(state: EnvState) -> None:
    """Host boundary: never optimize or score a truncated/unreachable IR2 graph."""
    if state.graph_memory is None:
        return
    if bool(jnp.any(state.graph_memory.overflow)):
        detail = (
            "IR2 graph/history capacity exhausted; increase the declared capacity before running"
        )
        raise ValueError(detail)
    if bool(jnp.any(state.graph_memory.path_failed)):
        detail = "IR2 rendezvous graph remains disconnected after the original repair procedure"
        raise ValueError(detail)
