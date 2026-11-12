"""Pure route-B real-event collection across independent physical environment lanes."""

from collections.abc import Callable
from functools import cache
from math import ceil
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import Array

from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.communication.faults import fold_key
from ir2_benchmark.communication.sending import prepare_message
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.models.world_model import WorldObservation
from ir2_benchmark.training.b_state import BTrainingConfig
from ir2_benchmark.training.imagination import (
    PolicyCarry,
    PolicyDecision,
    encode_world_action,
    local_tile,
    policy_decision,
)
from ir2_benchmark.training.world_losses import (
    MAP_SAMPLE_STREAM,
    _world_observation,
    local_change_sample_mask,
)
from ir2_benchmark.types import Action, EnvState, LocalObservation, StepResult


class RealEvent(NamedTuple):
    """One measured transition and compact replay supervision, with no map time axis."""

    result: StepResult
    decision: PolicyDecision
    observation: LocalObservation
    action: Action
    measurement: WorldObservation
    encoded: Array
    local_mask: Array


RealCollector = Callable[
    [EnvState, PolicyCarry, TrainState, TrainState, Array, Array, Array], RealEvent
]


def real_event(  # noqa: PLR0913, PLR0917
    state: EnvState,
    carry: PolicyCarry,
    actor: TrainState,
    predictor: TrainState,
    key: Array,
    warmup: Array,
    enabled: Array,
    *,
    config: HostConfig,
    options: BTrainingConfig,
) -> RealEvent:
    """Preserve a disabled lane exactly while measuring enabled actual sensing/message changes."""
    radius = ceil(config.sensor_radius)
    observation = observe(state, state.current_agent, config)
    decision = policy_decision(
        actor,
        predictor,
        observation,
        carry,
        config,
        key,
        learned_communication=options.learned_communication,
        predicted_plans=options.predicted_plans,
        use_neighbor_plan=options.use_neighbor_plan,
    )
    action = jax.lax.cond(
        warmup, lambda: nearest_frontier_action(observation, config), lambda: decision.action
    )
    action = prepare_message(observation, action, config)
    # The simulator's terminal branch freezes all event/fault counters on disabled lanes.
    gated = state._replace(terminated=state.terminated | ~enabled)
    result = step_agent(gated, action, config, state.key, observation)
    restored = jax.tree.map(lambda new, old: jnp.where(enabled, new, old), result.state, state)
    result = result._replace(
        state=restored, terminated=restored.terminated, truncated=restored.truncated
    )
    accepted = enabled & result.info.active
    memory = jax.tree.map(lambda new, old: jnp.where(accepted, new, old), decision.carry, carry)
    decision = decision._replace(carry=memory)
    measurement = _world_observation(result.state, observation.agent_id, radius, config)
    position = result.state.positions[observation.agent_id]
    mask_key = fold_key(
        state.key, MAP_SAMPLE_STREAM, state.episode, state.event, observation.agent_id
    )
    mask = local_change_sample_mask(
        local_tile(state.local_maps[observation.agent_id], position, radius),
        jnp.argmax(measurement.local_tile, -1),
        position,
        state.valid_shape,
        config.sensor_radius,
        mask_key,
        options.unchanged_sample_rate,
    )
    return RealEvent(
        result,
        decision,
        observation,
        action,
        measurement,
        encode_world_action(observation, action, config),
        mask,
    )


@cache
def make_real_collector(config: HostConfig, options: BTrainingConfig) -> RealCollector:
    """Share weights, but vmap complete independent EnvState/carry/key/mask lane arguments."""

    def single(  # noqa: PLR0913, PLR0917
        state: EnvState,
        carry: PolicyCarry,
        actor: TrainState,
        predictor: TrainState,
        key: Array,
        warmup: Array,
        enabled: Array,
    ) -> RealEvent:
        return real_event(
            state, carry, actor, predictor, key, warmup, enabled, config=config, options=options
        )

    return jax.jit(jax.vmap(single, in_axes=(0, 0, None, None, 0, None, 0)))
