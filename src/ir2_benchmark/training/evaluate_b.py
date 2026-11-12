"""Real route-B evaluation and paired-action open-loop world-model diagnostics."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import cache, partial
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
from jax import Array

from ir2_benchmark.agents.actor import Actor, initial_memory
from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.communication.faults import KeyPurpose, fold_key, sample_isolated
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import _sample_spawns, reset
from ir2_benchmark.env.rollout import EventTrace, RolloutResult
from ir2_benchmark.env.scenarios import load_scenario
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.evaluation.diagnostics import PlanDiagnostics, measure_plan_estimate
from ir2_benchmark.evaluation.metrics import EpisodeOutcome, summarize_episode
from ir2_benchmark.models.plan_predictor import PlanMemory, PlanPredictor, initial_plan_memory
from ir2_benchmark.models.world_model import initial_rssm
from ir2_benchmark.presets import verify_preset_manifest
from ir2_benchmark.training.b_state import BTrainingConfig, _cognition, _initialize
from ir2_benchmark.training.b_types import EvaluationSummary, ModelDiagnostic
from ir2_benchmark.training.checkpoint import load_checkpoint
from ir2_benchmark.training.dreamer import ImaginedBatch
from ir2_benchmark.training.imagination import (
    ImaginedState,
    ImaginedTransition,
    PolicyCarry,
    apply_prediction,
    cognitive_from_environment,
    encode_world_action,
    imagine_rollout,
    imagined_observe,
    policy_decision,
)
from ir2_benchmark.training.replay import ReplayBuffer, ReplayError
from ir2_benchmark.types import EnvState, MapInput, StepResult


class Parameters(NamedTuple):
    """Only inference weights cross the compiled evaluation boundary."""

    actor: FrozenDict
    predictor: FrozenDict


class InferenceSettings(NamedTuple):
    """Training quotas and optimizer settings do not affect compiled policy inference."""

    hidden: int
    categories: int
    classes: int
    learned_communication: bool
    predicted_plans: bool
    use_neighbor_plan: bool
    candidate_attention: bool


EvaluationKernel: TypeAlias = Callable[
    [Parameters, MapInput, Array, Array, Array, Array], tuple[RolloutResult, Array]
]


def inference_settings(settings: BTrainingConfig) -> InferenceSettings:
    """Extract exactly the model structure and policy switches used during evaluation."""
    return InferenceSettings(
        settings.hidden,
        settings.categories,
        settings.classes,
        settings.learned_communication,
        settings.predicted_plans,
        settings.use_neighbor_plan,
        settings.candidate_attention,
    )


def load_parameters(checkpoint: str | Path) -> tuple[Parameters, BTrainingConfig]:
    """Verify and read a checkpoint once, returning only actor/predictor weights and settings."""
    path = Path(checkpoint)
    configuration = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    verify_preset_manifest(
        HostConfig(**configuration["environment"]), configuration.get("implementations")
    )
    config_hash = sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    recovered = serialization.msgpack_restore(load_checkpoint(path, config_hash))
    parameters = Parameters(
        freeze(jax.tree_util.tree_map(jnp.asarray, recovered["actor"]["params"])),
        freeze(jax.tree_util.tree_map(jnp.asarray, recovered["predictor"]["params"])),
    )
    configuration["training"].setdefault("candidate_attention", False)
    return parameters, BTrainingConfig(**configuration["training"])


def get_evaluation_kernel(config: HostConfig, settings: BTrainingConfig) -> EvaluationKernel:
    """Return the persistent batched kernel; every argument has its own leading batch axis."""
    return _cached_evaluation_kernel(config, inference_settings(settings))


@cache
def _cached_evaluation_kernel(config: HostConfig, settings: InferenceSettings) -> EvaluationKernel:
    """Capture architecture/configuration only, keeping weights and scenario arrays dynamic."""
    actor_model = Actor(hidden=settings.hidden, candidate_attention=settings.candidate_attention)
    predictor_model = PlanPredictor(
        config.plan_horizon, settings.hidden, settings.categories, settings.classes
    )

    def single(  # noqa: PLR0913, PLR0917
        parameters: Parameters,
        map_data: MapInput,
        policy_key: Array,
        spawns: Array,
        fault_key: Array,
        isolated_mask: Array,
    ) -> tuple[RolloutResult, Array]:
        actor = TrainState.create(
            apply_fn=actor_model.apply, params=parameters.actor, tx=optax.identity()
        )
        predictor = TrainState.create(
            apply_fn=predictor_model.apply, params=parameters.predictor, tx=optax.identity()
        )
        initial = reset(map_data, config, policy_key, spawns=spawns)
        initial = initial._replace(key=fault_key, isolated_mask=isolated_mask)
        carry = PolicyCarry(
            initial_memory((config.n_slots,), settings.hidden),
            initial_plan_memory(
                (config.n_slots, config.n_slots),
                settings.hidden,
                settings.categories,
                settings.classes,
            ),
        )

        def event(
            previous: tuple[EnvState, PolicyCarry], _unused: None
        ) -> tuple[tuple[EnvState, PolicyCarry], tuple[EventTrace, PlanDiagnostics]]:
            state, memory = previous
            observation = observe(state, state.current_agent, config)
            decision = policy_decision(
                actor,
                predictor,
                observation,
                memory,
                config,
                fold_key(
                    policy_key, KeyPurpose.POLICY, state.episode, state.event, state.current_agent
                ),
                learned_communication=settings.learned_communication,
                predicted_plans=settings.predicted_plans,
                use_neighbor_plan=settings.use_neighbor_plan,
                greedy=False,
            )
            result = step_agent(state, decision.action, config, fault_key, observation)
            diagnostic = measure_plan_estimate(
                decision.estimate,
                state,
                state.current_agent,
                config,
                event_valid=result.info.active,
                has_uncertainty=settings.predicted_plans and settings.use_neighbor_plan,
            )
            updated_memory = jax.tree_util.tree_map(
                lambda new, old: jnp.where(result.info.active, new, old), decision.carry, memory
            )
            updated, info = result.state, result.info
            trace = EventTrace(
                state.event,
                info.agent_id,
                info.active,
                updated.positions,
                updated.goals,
                updated.balances,
                info.spent,
                info.delivered,
                info.coverage_after,
                result.reward,
                info.invalid_action,
                result.terminated,
                result.truncated,
            )
            return (updated, updated_memory), (trace, diagnostic)

        (final, _), (trace, diagnostics) = jax.lax.scan(
            event, (initial, carry), None, length=config.max_rounds * config.n_agents
        )
        return RolloutResult(final, trace, diagnostics), initial.physical_seen

    return jax.jit(jax.vmap(single))


def _evaluate_policy(  # noqa: PLR0913, PLR0917
    actor: TrainState,
    predictor: TrainState,
    config: HostConfig,
    options: BTrainingConfig,
    seed: int,
    map_path: Path | None,
) -> EvaluationSummary:
    """Execute a held-out real environment; modeled returns never serve as evaluation scores."""
    state = reset(load_scenario(config, seed, map_path), config, jax.random.PRNGKey(seed))
    carry = PolicyCarry(
        initial_memory((config.n_slots,), options.hidden),
        initial_plan_memory(
            (config.n_slots, config.n_slots), options.hidden, options.categories, options.classes
        ),
    )

    @jax.jit
    def event(state: EnvState, carry: PolicyCarry, key: Array) -> tuple[StepResult, PolicyCarry]:
        obs = observe(state, state.current_agent, config)
        decision = policy_decision(
            actor,
            predictor,
            obs,
            carry,
            config,
            key,
            learned_communication=options.learned_communication,
            predicted_plans=options.predicted_plans,
            use_neighbor_plan=options.use_neighbor_plan,
            greedy=False,
        )
        result = step_agent(state, decision.action, config, state.key, obs)
        return result, decision.carry

    rounds, events = [], []
    initial_coverage = float(jnp.sum(state.covered) / jnp.sum(state.component))
    while not bool(state.terminated | state.truncated):
        result, carry = event(
            state, carry, jax.random.fold_in(jax.random.PRNGKey(seed), state.event)
        )
        state = result.state
        coverage = float(result.info.coverage_after)
        events.append(coverage)
        if int(state.current_agent) == 0 or bool(state.terminated | state.truncated):
            rounds.append(coverage)
    if not rounds:
        rounds.append(initial_coverage)
    outcome = EpisodeOutcome.SUCCESS if bool(state.terminated) else EpisodeOutcome.TIMEOUT
    metrics = summarize_episode(
        rounds,
        horizon=config.max_rounds,
        n_agents=config.n_agents,
        outcome=outcome,
        event_coverage=events or None,
        threshold=config.coverage_target,
    )
    return {**asdict(metrics), "decision_rule": "stochastic_with_fixed_seed"}


def model_diagnostics(
    world: TrainState,
    config: HostConfig,
    options: BTrainingConfig,
    seed: int,
    map_path: Path | None,
) -> list[ModelDiagnostic]:
    """Compare a shared real action script with open-loop modeled execution at 1/3/5 rounds."""
    real = reset(load_scenario(config, seed, map_path), config, jax.random.PRNGKey(seed))
    simulated = ImaginedState(
        cognitive_from_environment(real),
        initial_rssm(
            deterministic=options.hidden, categories=options.categories, classes=options.classes
        ),
        initial_memory((config.n_slots,), options.hidden),
    )

    @jax.jit
    def paired(
        real: EnvState, simulated: ImaginedState, key: Array
    ) -> tuple[StepResult, ImaginedTransition]:
        real_obs = observe(real, real.current_agent, config)
        action = nearest_frontier_action(real_obs, config)
        actual = step_agent(real, action, config, real.key, real_obs)
        model_obs = imagined_observe(simulated.cognition, config)
        prediction = world.apply_fn(
            {"params": world.params},
            simulated.latent,
            encode_world_action(model_obs, action, config),
            jax.random.fold_in(key, 1),
            method="imagine",
        )
        modeled = apply_prediction(
            simulated, action, prediction, config, jax.random.fold_in(key, 2), model_obs
        )
        return actual, modeled

    result: list[ModelDiagnostic] = []
    reward_error = delivery_error = 0.0
    for event in range(5 * config.n_agents):
        actual, modeled = paired(
            real, simulated, jax.random.fold_in(jax.random.PRNGKey(seed), event)
        )
        real, simulated = actual.state, modeled.state
        reward_error += abs(float(actual.reward) - float(modeled.reward))
        delivery_error += float(jnp.mean(actual.info.delivered != modeled.delivered))
        if event + 1 in (config.n_agents, 3 * config.n_agents, 5 * config.n_agents):
            result.append(
                {
                    "rounds": (event + 1) // config.n_agents,
                    "position_error": float(
                        jnp.mean(
                            jnp.linalg.norm(real.positions - simulated.cognition.positions, axis=-1)
                        )
                    ),
                    "local_map_disagreement": float(
                        jnp.mean(real.local_maps != simulated.cognition.local_maps)
                    ),
                    "reward_mae": reward_error / (event + 1),
                    "delivery_error": delivery_error / (event + 1),
                }
            )
    return result


def run_evaluation(  # noqa: PLR0913
    checkpoint: str | Path,
    map_data: MapInput,
    config: HostConfig,
    settings: BTrainingConfig,
    key: Array,
    *,
    spawns: Array | None = None,
    fault_key: Array | None = None,
    isolated_mask: Array | None = None,
) -> RolloutResult:
    """Run a checkpoint policy in the real simulator with paired evaluator-controlled scenario seeds."""  # noqa: E501
    parameters, trained = load_parameters(checkpoint)
    settings = replace(settings, candidate_attention=trained.candidate_attention)
    selected_spawns = (
        _sample_spawns(map_data, config, key, jnp.int32(0))
        if spawns is None
        else jnp.asarray(spawns, jnp.int32)
    )
    fault = key if fault_key is None else fault_key
    isolated = (
        sample_isolated(fault, jnp.int32(0), jnp.arange(config.n_slots) < config.n_agents)
        if isolated_mask is None
        else jnp.asarray(isolated_mask, bool)
    )
    batched = jax.tree_util.tree_map(
        lambda value: value[None], (parameters, map_data, key, selected_spawns, fault, isolated)
    )
    result, _ = get_evaluation_kernel(config, settings)(*batched)
    return jax.tree_util.tree_map(lambda value: value[0], result)


class ModelUpdateBenchmark(NamedTuple):
    """One actual model optimizer step and its effective replay training-label count."""

    state: TrainState
    loss: Array
    updates: Array
    valid_training_events: Array


class ImaginationBenchmark(NamedTuple):
    """Actual modeled rollout with frozen post-termination events excluded from throughput."""

    final_state: ImaginedState
    carry: PolicyCarry
    batch: ImaginedBatch
    valid_events: Array


@dataclass(frozen=True, slots=True)
class BenchmarkMetadata:
    """Declare measured batch sizes and the actual replay cutoff used for imagination."""

    world_logical_batch: int
    world_physical_microbatch: int
    imagination_logical_origins: int
    imagination_physical_batch: int
    maximum_imagined_events: int
    origin: str
    episode_id: int
    origin_index: int
    sequence_length: int
    burn_in: int


class BThroughputFunctions(NamedTuple):
    """Setup is outside measurement; each callable evaluates the same reproducible full kernel."""

    world_update: Callable[[], ModelUpdateBenchmark]
    imagination: Callable[[], ImaginationBenchmark]
    metadata: BenchmarkMetadata


def benchmark_functions(
    checkpoint: str | Path,
    map_data: MapInput,
    config: HostConfig,
    settings: BTrainingConfig,
    key: Array,
) -> BThroughputFunctions:
    """Restore real replay/parameters and expose genuine model-update and closed-imagination kernels."""  # noqa: E501
    from ir2_benchmark.training.world_losses import (  # noqa: PLC0415
        WorldBatch,
        _world_batch,
        posterior_sequence,
        update_world,
    )

    checkpoint = Path(checkpoint)
    configuration = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    verify_preset_manifest(
        HostConfig(**configuration["environment"]), configuration.get("implementations")
    )
    config_hash = sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    recovered = serialization.msgpack_restore(load_checkpoint(checkpoint, config_hash))
    real = reset(map_data, config, key)
    actor, predictor, world, _, _, _ = _initialize(config, settings, real, key)
    actor = serialization.from_state_dict(actor, recovered["actor"])
    predictor = serialization.from_state_dict(predictor, recovered["predictor"])
    world = serialization.from_state_dict(world, recovered["world"])
    replay_path = (
        checkpoint.parent / "replay" / f"{map_data.occupancy.shape[0]}x{config.n_agents}.npz"
        if configuration["manifest_hash"]
        else checkpoint.parent / "replay.npz"
    )
    replay = ReplayBuffer.load(replay_path)
    if not replay.has_sequence(settings.burn_in):
        raise ReplayError("benchmark_replay_has_no_postburnin_labels")  # noqa: EM101
    sequences = []
    while len(sequences) < settings.logical_batch:
        sequence = replay.sample(settings.sequence_length, settings.burn_in)
        if np.any(sequence.loss_mask):
            sequences.append(sequence)
    batches = tuple(
        _world_batch(sequence, int(config.sensor_radius), config) for sequence in sequences
    )
    sequence = sequences[0]
    _, predictions = posterior_sequence(world, batches[0], key, settings)
    cutoff = settings.burn_in
    origin_index = sequence.start_index + cutoff
    frame = replay.reconstruct(sequence.episode_id, origin_index)
    latent = (
        jax.tree_util.tree_map(lambda value: value[cutoff - 1], predictions.state)
        if cutoff
        else initial_rssm(
            deterministic=settings.hidden, categories=settings.categories, classes=settings.classes
        )
    )
    carry = PolicyCarry(
        jnp.asarray(frame.field("all_actor_memory")),
        PlanMemory(
            *(jnp.asarray(frame.field(f"all_plan_memory_{name}")) for name in PlanMemory._fields)
        ),
    )
    initial = ImaginedState(_cognition(frame), latent, carry.actor_memory)
    jax.block_until_ready((initial, carry, actor, predictor, world, batches, key))

    @jax.jit
    def update(
        world: TrainState, batches: tuple[WorldBatch, ...], key: Array
    ) -> ModelUpdateBenchmark:
        state, loss = update_world(world, batches, key, settings)
        return ModelUpdateBenchmark(
            state, loss, jnp.int32(1), sum(batch.loss_mask.sum() for batch in batches)
        )

    @jax.jit
    def imagine(  # noqa: PLR0913, PLR0917
        initial: ImaginedState,
        carry: PolicyCarry,
        actor: TrainState,
        predictor: TrainState,
        world: TrainState,
        key: Array,
    ) -> ImaginationBenchmark:
        final, final_carry, batch = imagine_rollout(
            initial,
            carry,
            actor,
            predictor,
            world,
            config,
            key,
            settings.imagination_rounds * config.n_agents,
            learned_communication=settings.learned_communication,
            predicted_plans=settings.predicted_plans,
            use_neighbor_plan=settings.use_neighbor_plan,
        )
        return ImaginationBenchmark(final, final_carry, batch, batch.valid.sum())

    return BThroughputFunctions(
        partial(update, world, batches, key),
        partial(imagine, initial, carry, actor, predictor, world, key),
        BenchmarkMetadata(
            world_logical_batch=len(batches),
            world_physical_microbatch=1,
            imagination_logical_origins=1,
            imagination_physical_batch=1,
            maximum_imagined_events=settings.imagination_rounds * config.n_agents,
            origin="actual_replay_burn_in",
            episode_id=sequence.episode_id,
            origin_index=origin_index,
            sequence_length=settings.sequence_length,
            burn_in=cutoff,
        ),
    )
