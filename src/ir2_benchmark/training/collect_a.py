"""Compiled real-environment event sampling; only compact training features persist."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax import Array

from ir2_benchmark.agents.actor import (
    Actor,
    ActorOutput,
    IndexAction,
    action_statistics,
    sample_communication,
)
from ir2_benchmark.agents.critic import Critic, PrivilegedFeatures
from ir2_benchmark.agents.fixed_comm import fixed_communication
from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.communication.sending import prepare_message, validate_builtin_policy
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.evaluation.diagnostics import PlanDiagnostics, measure_plan_estimate
from ir2_benchmark.models.encoders import (
    actor_features,
    no_plan_estimate,
    selected_plan_features,
    zoh_estimate,
)
from ir2_benchmark.models.plan_predictor import PlanInput, PlanMemory, PlanPredictor, plan_inputs
from ir2_benchmark.training.plan_losses import PlanTarget, target_from_labels
from ir2_benchmark.training.storage import CollectionTrace, RolloutBatch, StandardAction
from ir2_benchmark.types import EnvState, make_action, plan_labels


class Parameters(NamedTuple):
    """Predictor parameters remain frozen throughout collection and PPO epochs."""

    actor: FrozenDict
    critic: FrozenDict
    predictor: FrozenDict


class SamplerState(NamedTuple):
    """Only the current physical/local maps live in the carry, never along time."""

    environment: EnvState
    actor_memory: Array
    plan_memory: PlanMemory
    key: Array
    real_events: Array


class PlanTrainingBatch(NamedTuple):
    """Receiver-visible input and privileged targets remain separate structures."""

    memory: PlanMemory
    inputs: PlanInput
    target: PlanTarget
    key: Array


class Collection(NamedTuple):
    """Per-environment ordered axes [B,T]; flatten only after computing event GAE."""

    sampler: SamplerState
    rollout: RolloutBatch
    plans: PlanTrainingBatch
    bootstrap: Array
    trace: CollectionTrace
    plan_diagnostics: PlanDiagnostics
    executed_actions: StandardAction


Collector = Callable[[SamplerState, Parameters, Array, Array], Collection]
TrainingFrame = tuple[
    RolloutBatch, PlanTrainingBatch, CollectionTrace, PlanDiagnostics, StandardAction
]


def privileged_features(state: EnvState, config: HostConfig) -> PrivilegedFeatures:
    """Use bounded neural critic features; physical simulation retains its full geometry."""
    channels = jnp.stack((state.truth, state.covered, state.component), -1).astype(jnp.float32)
    image = jax.image.resize(channels, (32, 32, 3), method="linear")
    scale = jnp.maximum(state.valid_shape[::-1] - 1, 1)
    robots = jnp.concatenate(
        (
            state.positions / scale,
            (state.balances / jnp.maximum(config.initial_budget, 1))[:, None],
            state.active_mask[:, None],
        ),
        -1,
    )
    return PrivilegedFeatures(
        image,
        robots,
        jnp.asarray([state.round / config.max_rounds, state.current_agent / config.n_agents]),
    )


def make_collector(  # noqa: PLR0913
    config: HostConfig,
    actor: Actor,
    critic: Critic,
    predictor: PlanPredictor,
    *,
    length: int = 128,
    learned_communication: bool = True,
    predict_neighbors: bool = True,
    warmup: bool = False,
    compute_values: bool = True,
    use_neighbor_plan: bool = True,
    fixed_plan: bool = False,
) -> Collector:
    """Build scan within each independent environment, then vmap over environments."""
    validate_builtin_policy(config, learned_communication)

    def one(
        sampler: SamplerState, params: Parameters, limit: Array, version: Array
    ) -> tuple[SamplerState, TrainingFrame, Array]:
        def event(carry: SamplerState, _unused: None) -> tuple[SamplerState, TrainingFrame]:
            env = carry.environment
            agent = env.current_agent
            observation = observe(env, agent, config)
            next_key, motion_key, communication_key, prediction_key = jax.random.split(carry.key, 4)
            valid = ~(env.terminated | env.truncated) & (carry.real_events < limit)
            prior_memory = jax.tree.map(lambda value: value[agent], carry.plan_memory)
            inputs = plan_inputs(observation)
            prediction = predictor.apply(
                {"params": params.predictor}, prior_memory, inputs, prediction_key
            )
            estimate = (
                (prediction.estimate if predict_neighbors else zoh_estimate(observation))
                if use_neighbor_plan
                else no_plan_estimate(observation)
            )
            features = actor_features(
                observation,
                config.n_agents,
                estimate,
                initial_budget=config.initial_budget,
                max_rounds=config.max_rounds,
                patch_radius=int(config.sensor_radius),
                use_neighbor_plan=use_neighbor_plan,
                candidate_attention=actor.candidate_attention,
            )
            memory = carry.actor_memory[agent]
            updated_memory, waypoint_logits = actor.apply(
                {"params": params.actor}, features, memory, method=actor.encode
            )
            waypoint_index = jax.random.categorical(motion_key, waypoint_logits).astype(jnp.int32)
            waypoint = observation.candidates.positions[waypoint_index]
            if warmup:
                waypoint = nearest_frontier_action(observation, config).waypoint
            route = selected_plan_features(
                observation, waypoint, config.max_distance, config.plan_horizon
            )
            gate, types = actor.apply(
                {"params": params.actor}, updated_memory, route, method=actor.communication
            )
            output = ActorOutput(updated_memory, waypoint_logits, gate, types)
            index_action = sample_communication(output, waypoint_index, communication_key)
            if not learned_communication or warmup:
                send, codec = fixed_communication(observation, plan_every_step=fixed_plan)
                index_action = IndexAction(waypoint_index, send, codec - 1)
            action = prepare_message(
                observation,
                make_action(waypoint, index_action.send, index_action.message_type + 1),
                config,
            )
            index_action = IndexAction(waypoint_index, action.send, action.codec - 1)
            statistics = action_statistics(
                output,
                index_action,
                features.candidate_mask,
                learned_communication=learned_communication,
            )
            critic_input = privileged_features(env, config)
            value = (
                critic.apply({"params": params.critic}, critic_input)
                if compute_values
                else jnp.float32(0)
            )
            target = target_from_labels(
                plan_labels(env),
                env.valid_shape,
                env.active_mask & (jnp.arange(config.n_slots) != agent),
            )
            target = target._replace(available=target.available & valid)
            result = step_agent(env, action, config, key=env.key, observation=observation)
            next_environment = jax.tree.map(
                lambda new, old: jnp.where(valid, new, old), result.state, env
            )
            next_actor_memory = carry.actor_memory.at[agent].set(
                jnp.where(valid, updated_memory, memory)
            )
            next_plan_memory = jax.tree.map(
                lambda full, new, old: full.at[agent].set(jnp.where(valid, new, old)),
                carry.plan_memory,
                prediction.memory,
                prior_memory,
            )
            updated = SamplerState(
                next_environment,
                next_actor_memory,
                next_plan_memory,
                jnp.where(valid, next_key, carry.key),
                carry.real_events + valid.astype(jnp.int32),
            )
            rollout = RolloutBatch(
                features,
                critic_input,
                memory,
                route,
                index_action,
                statistics.log_prob,
                value,
                jnp.where(valid, result.reward, 0),
                result.terminated,
                result.truncated,
                valid,
                agent,
                version,
                features.candidates,
                features.candidate_mask,
            )
            trace = CollectionTrace(
                env.event,
                agent,
                valid,
                next_environment.positions,
                next_environment.goals,
                next_environment.balances,
                jnp.where(valid, result.info.spent, 0),
                result.info.delivered & valid,
                result.info.coverage_after,
                jnp.where(valid, result.reward, 0),
                result.terminated,
                result.truncated,
                result.info.invalid_action & valid,
            )
            diagnostics = measure_plan_estimate(
                estimate,
                env,
                agent,
                config,
                event_valid=valid,
                has_uncertainty=predict_neighbors and use_neighbor_plan,
            )
            return updated, (
                rollout,
                PlanTrainingBatch(prior_memory, inputs, target, prediction_key),
                trace,
                diagnostics,
                StandardAction(action.waypoint, action.send, action.codec),
            )

        final, (rollout, plans, trace, diagnostics, actions) = jax.lax.scan(
            event, sampler, None, length=length
        )
        bootstrap = (
            critic.apply({"params": params.critic}, privileged_features(final.environment, config))
            if compute_values
            else jnp.float32(0)
        )
        bootstrap = jnp.where(
            final.environment.terminated | final.environment.truncated, 0, bootstrap
        )
        return final, (rollout, plans, trace, diagnostics, actions), bootstrap

    @jax.jit
    def collect(
        sampler: SamplerState, params: Parameters, limits: Array, version: Array
    ) -> Collection:
        final, (rollout, plans, trace, diagnostics, actions), bootstrap = jax.vmap(
            one, in_axes=(0, None, 0, None)
        )(sampler, params, limits, version)
        return Collection(final, rollout, plans, bootstrap, trace, diagnostics, actions)

    return collect
