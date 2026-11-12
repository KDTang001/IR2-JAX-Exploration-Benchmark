"""Immutable route-A state, configuration and real-scenario initialization."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jax import Array

from ir2_benchmark.agents.actor import Actor, initial_memory
from ir2_benchmark.agents.critic import Critic
from ir2_benchmark.communication.faults import KeyPurpose, fold_key
from ir2_benchmark.communication.sending import validate_builtin_policy
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.models.encoders import actor_features, selected_plan_features
from ir2_benchmark.models.plan_predictor import PlanPredictor, initial_plan_memory, plan_inputs
from ir2_benchmark.presets import preset_manifest
from ir2_benchmark.training.collect_a import (
    PlanTrainingBatch,
    SamplerState,
    privileged_features,
)
from ir2_benchmark.training.plan_losses import target_from_labels
from ir2_benchmark.training.ppo import optimizer
from ir2_benchmark.training.predictor_replay import PredictorReplay, empty_predictor_replay
from ir2_benchmark.types import MapInput, plan_labels, validate_graph_state


@dataclass(frozen=True, slots=True)
class TrainAConfig:
    """Formal defaults; the micro CLI explicitly overrides warmup and event quota."""

    events: int = 2_000_000
    warmup_events: int = 100_000
    rollout_length: int = 128
    sequence_length: int = 32
    ppo_epochs: int = 4
    ppo_minibatch_sequences: int = 0
    ppo_target_kl: float = 0.0
    batch_size: int = 4
    physical_env_count: int = 4
    hidden: int = 256
    candidate_attention: bool = True
    fixed_plan: bool = False
    curriculum_easy_episodes: int = 0
    learning_rate: float = 3e-4
    predictor_replay_capacity: int = 4096
    predictor_recent_fraction: float = 0.5
    communication_mode: Literal["fixed", "learned"] = "learned"
    neighbor_estimator: Literal["zoh", "predict", "none"] = "predict"
    suite: Literal["micro", "formal"] = "formal"

    def __post_init__(self) -> None:
        """Reject invalid experiment settings before compiled sampling."""
        if self.ppo_minibatch_sequences < 0 or not 0 <= self.ppo_target_kl < float("inf"):
            raise TrainingError(
                field="ppo", detail="minibatch count and finite KL must be nonnegative"
            )
        if self.curriculum_easy_episodes < 0 or (
            self.fixed_plan and self.communication_mode != "fixed"
        ):
            raise TrainingError(
                field="pretraining",
                detail="require nonnegative curriculum and fixed communication for fixed PLAN",
            )
        if not 0 <= self.warmup_events < self.events:
            raise TrainingError(field="events", detail="require 0 <= warmup < total real events")
        if (
            min(
                self.rollout_length,
                self.sequence_length,
                self.ppo_epochs,
                self.batch_size,
                self.physical_env_count,
                self.hidden,
            )
            < 1
        ):
            raise TrainingError(field="shape", detail="positive training dimensions required")
        if (
            self.predictor_replay_capacity < self.batch_size * self.rollout_length
            or not 0 < self.predictor_recent_fraction <= 1
        ):
            raise TrainingError(
                field="predictor_replay",
                detail="capacity must hold a rollout and recent fraction must be in (0,1]",
            )
        if self.communication_mode not in ("fixed", "learned") or self.neighbor_estimator not in (
            "zoh",
            "predict",
            "none",
        ):
            raise TrainingError(field="variant", detail="unknown communication or estimator mode")


@dataclass(frozen=True, slots=True)
class TrainingError(ValueError):
    """An explicit experiment contract or data-source failure."""

    field: str
    detail: str

    def __str__(self) -> str:
        """Keep actionable training configuration failures visible."""
        return f"{self.field}: {self.detail}"


class TrainingState(NamedTuple):
    """Full optimizer/PRNG/environment state, not only inference parameters."""

    actor: TrainState
    critic: TrainState
    predictor: TrainState
    sampler: SamplerState
    updates: Array
    predictor_version: Array
    replay: PredictorReplay


class StageSeconds(TypedDict):
    """Measured host wall durations, including first-use compilation and synchronization."""

    collect: float
    update: float
    checkpoint: float
    validation: float
    recording: float


@dataclass(frozen=True, slots=True)
class TrainASummary:
    """Executed real interactions and concrete output paths."""

    real_events: int
    warmup_events: int
    updates: int
    episodes: int
    actor_parameter_change: float
    checkpoint: str
    configuration_hash: str
    session_real_events: int = 0
    session_seconds: float = 0.0
    session_events_per_second: float = 0.0
    stage_seconds: StageSeconds | None = None
    physical_env_count: int = 1


def configuration_hash(config: HostConfig, settings: TrainAConfig) -> str:
    """Bind all dynamics and learning settings; output directories are irrelevant."""
    training = asdict(settings)
    for name in ("ppo_minibatch_sequences", "ppo_target_kl"):
        if not training[name]:
            training.pop(name)
    payload = json.dumps(
        {
            "environment": asdict(config),
            "training": training,
            "implementations": preset_manifest(config),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_configuration(
    destination: Path, config: HostConfig, settings: TrainAConfig, identity: str
) -> None:
    """Persist the effective immutable A settings independently of output-only recording."""
    (destination / "configuration.json").write_text(
        json.dumps(
            {
                "environment": asdict(config),
                "implementations": preset_manifest(config),
                "training": asdict(settings),
                "configuration_hash": identity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def actor_parameter_change(before: FrozenDict, after: FrozenDict) -> float:
    """Report the existing Euclidean actor-update magnitude after device synchronization."""
    squares = jax.tree.leaves(
        jax.tree.map(lambda old, new: jnp.sum(jnp.square(old - new)), before, after)
    )
    return float(jnp.sqrt(sum(squares)))


def initialize_training(
    map_data: MapInput, config: HostConfig, settings: TrainAConfig, key: Array
) -> TrainingState:
    """Initialize shared models and independent per-environment/robot/pair carries."""
    validate_builtin_policy(config, settings.communication_mode == "learned")
    actor = Actor(settings.hidden, candidate_attention=settings.candidate_attention)
    critic = Critic(settings.hidden)
    predictor = PlanPredictor(config.plan_horizon, settings.hidden)
    root, actor_key, critic_key, predictor_key = jax.random.split(key, 4)
    keys = jax.random.split(root, settings.batch_size)
    environments = jax.vmap(lambda sample_key: reset(map_data, config, sample_key))(keys)
    validate_graph_state(environments)
    first = jax.tree.map(lambda value: value[0], environments)
    observation = observe(first, first.current_agent, config)
    features = actor_features(
        observation,
        config.n_agents,
        initial_budget=config.initial_budget,
        max_rounds=config.max_rounds,
        patch_radius=int(config.sensor_radius),
        use_neighbor_plan=settings.neighbor_estimator != "none",
        candidate_attention=settings.candidate_attention,
    )
    route = selected_plan_features(
        observation, observation.position, config.max_distance, config.plan_horizon
    )
    actor_parameters = actor.init(
        actor_key, features, initial_memory(hidden=settings.hidden), route
    )["params"]
    critic_parameters = critic.init(critic_key, privileged_features(first, config))["params"]
    pair_memory = initial_plan_memory((config.n_slots,), settings.hidden)
    predictor_parameters = predictor.init(
        predictor_key, pair_memory, plan_inputs(observation), predictor_key
    )["params"]
    tx = optimizer(settings.learning_rate)
    sampler = SamplerState(
        environments,
        initial_memory((settings.batch_size, config.n_slots), settings.hidden),
        initial_plan_memory((settings.batch_size, config.n_slots, config.n_slots), settings.hidden),
        keys,
        jnp.zeros(settings.batch_size, jnp.int32),
    )
    return TrainingState(
        TrainState.create(apply_fn=actor.apply, params=actor_parameters, tx=tx),
        TrainState.create(apply_fn=critic.apply, params=critic_parameters, tx=tx),
        TrainState.create(apply_fn=predictor.apply, params=predictor_parameters, tx=tx),
        sampler,
        jnp.int32(0),
        jnp.int32(0),
        empty_predictor_replay(
            PlanTrainingBatch(
                pair_memory,
                plan_inputs(observation),
                target_from_labels(plan_labels(first), first.valid_shape, first.active_mask),
                predictor_key,
            ),
            settings.predictor_replay_capacity,
            fold_key(key, KeyPurpose.PLAN_REPLAY),
        ),
    )


def _limits(total: int, batch_size: int) -> Array:
    """Distribute an exact event quota, including a final nondivisible remainder."""
    return jnp.asarray(
        [total // batch_size + int(index < total % batch_size) for index in range(batch_size)]
    )


def _reset_finished(  # noqa: PLR0913, PLR0917
    state: TrainingState,
    map_data: MapInput,
    config: HostConfig,
    settings: TrainAConfig,
    limits: Array,
    scenario_sampler: Callable[[int], MapInput] | None,
) -> TrainingState:
    """Load new episode maps only on the host, then reset through the real JAX reset."""
    sampler = state.sampler
    for index in range(settings.batch_size):
        environment = jax.tree.map(lambda value, index=index: value[index], sampler.environment)
        finished = bool(environment.terminated | environment.truncated)
        if finished and int(sampler.real_events[index]) < int(limits[index]):
            episode = int(environment.episode) + 1
            scenario_id = episode * settings.batch_size + index
            selected = map_data if scenario_sampler is None else scenario_sampler(scenario_id)
            if selected.occupancy.shape != environment.truth.shape:
                raise TrainingError(
                    field="bucket",
                    detail="map bucket changed; dispatch through a separate bucket sampler state",
                )
            key, next_key = jax.random.split(sampler.key[index])
            new_environment = reset(selected, config, key, episode=episode)
            sampler = SamplerState(
                jax.tree.map(
                    lambda full, new, index=index: full.at[index].set(new),
                    sampler.environment,
                    new_environment,
                ),
                sampler.actor_memory.at[index].set(0),
                jax.tree.map(
                    lambda full, new, index=index: full.at[index].set(new),
                    sampler.plan_memory,
                    initial_plan_memory((config.n_slots, config.n_slots), settings.hidden),
                ),
                sampler.key.at[index].set(next_key),
                sampler.real_events,
            )
    return state._replace(sampler=sampler)
