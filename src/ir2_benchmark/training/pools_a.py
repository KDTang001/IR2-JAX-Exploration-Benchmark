"""Persistent formal episode pools and atomic full-state checkpoint payloads."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Final, NamedTuple, assert_never

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training.train_state import TrainState
from jax import Array
from numpy.typing import NDArray

from ir2_benchmark.agents.actor import initial_memory
from ir2_benchmark.communication.faults import KeyPurpose, fold_key, sample_isolated
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.reset import reset
from ir2_benchmark.maps.sampler import TrainingMapSampler, TrainingSample
from ir2_benchmark.models.plan_predictor import initial_plan_memory
from ir2_benchmark.training.checkpoint import load_checkpoint, save_checkpoint
from ir2_benchmark.training.collect_a import PlanTrainingBatch, SamplerState
from ir2_benchmark.training.ppo import PPOMetrics, Returns, optimize_ppo
from ir2_benchmark.training.predictor_replay import PredictorReplay, sample_and_store
from ir2_benchmark.training.state_a import StageSeconds, TrainingError
from ir2_benchmark.training.storage import RolloutBatch
from ir2_benchmark.training.train_a import (
    TrainAConfig,
    initialize_training,
)

PACKED_POOL_SCHEMA: Final = 3

SOURCES: Final = ("ir2", "procedural")
FAMILIES: Final = ("easy", "difficult", "rooms", "corridor", "hybrid")
PPOUpdate = Callable[
    [TrainState, TrainState, RolloutBatch, Returns], tuple[TrainState, TrainState, PPOMetrics]
]
PlanUpdate = Callable[[TrainState, PlanTrainingBatch], tuple[TrainState, Array]]


class ModelState(NamedTuple):
    """A single parameter/optimizer copy is shared across all nine environment pools."""

    actor: TrainState
    critic: TrainState
    predictor: TrainState
    updates: Array
    predictor_version: Array
    replay: PredictorReplay


class PoolState(NamedTuple):
    """A native (bucket,N) group with persistent independent physical lanes.

    sample_indices is host metadata; -1 identifies an unassigned pending lane.
    Such lanes contain an unused shape template and never reach a collector.
    """

    bucket: int
    n_agents: int
    sample_indices: tuple[int, ...]
    sampler: SamplerState


class FormalState(NamedTuple):
    """All pools, schedule cursor, model optimizers and future policy keys are checkpointed."""

    models: ModelState
    pools: tuple[PoolState, ...]
    cursor: int
    real_events: int
    root_key: Array
    source_episodes: Array
    source_events: Array
    family_episodes: Array
    family_events: Array
    best_auc: float
    last_validation: int


def _pool(
    sample: TrainingSample,
    maps: TrainingMapSampler,
    config: HostConfig,
    settings: TrainAConfig,
    root_key: Array,
) -> PoolState:
    """Create one real native-bucket episode with independently named policy/fault streams."""
    data = maps.load(sample)
    environment = reset(
        data,
        replace(config, n_agents=sample.n_agents, agent_bucket=6),
        jax.random.PRNGKey(sample.spawn_seed),
        episode=sample.index,
    )
    fault_key = jax.random.PRNGKey(sample.fault_seed)
    environment = environment._replace(
        key=fault_key,
        isolated_mask=sample_isolated(fault_key, sample.index, environment.active_mask),
    )
    sampler = SamplerState(
        jax.tree.map(lambda value: value[None], environment),
        initial_memory((1, 6), settings.hidden),
        initial_plan_memory((1, 6, 6), settings.hidden),
        fold_key(root_key, KeyPurpose.POLICY, sample.index)[None],
        jnp.zeros(1, jnp.int32),
    )
    count = settings.physical_env_count
    return PoolState(
        sample.bucket,
        sample.n_agents,
        (sample.index, *(-1 for _ in range(count - 1))),
        jax.tree.map(lambda value: jnp.repeat(value, count, axis=0), sampler),
    )


def _save(path: Path, state: FormalState, identity: str) -> None:
    """Checkpoint all in-flight pools and their restore-template sample identities atomically."""
    metadata = {
        "pool_schema": PACKED_POOL_SCHEMA,
        "pool_samples": [list(pool.sample_indices) for pool in state.pools],
    }
    payload = serialization.msgpack_serialize(
        {"metadata": metadata, "state": _pack_booleans(serialization.to_state_dict(state))}
    )
    save_checkpoint(path, payload, identity)


def _pack_booleans(state: dict) -> dict:
    """Losslessly pack large Boolean maps on device before checkpoint transfer."""

    def pack(value: object) -> object:
        if isinstance(value, (Array, np.ndarray)) and value.dtype == bool and value.size >= 2**20:
            return {"packed_bool": jnp.packbits(value.reshape(-1)), "shape": list(value.shape)}
        return value

    return jax.tree.map(pack, state)


def _unpack_booleans(state: dict) -> dict:
    """Restore Boolean maps before Flax checks the complete state template."""

    def packed(value: object) -> bool:
        return isinstance(value, dict) and set(value) == {"packed_bool", "shape"}

    def unpack(value: object) -> object:
        if isinstance(value, dict) and packed(value):
            shape = tuple(value["shape"])
            return (
                np.unpackbits(value["packed_bool"], count=int(np.prod(shape)))
                .astype(bool)
                .reshape(shape)
            )
        return value

    return jax.tree.map(unpack, state, is_leaf=packed)


def _restore_array(leaf: Array | NDArray[np.generic] | float) -> Array | float:
    """Restore device arrays while keeping host schedule/selection metadata as Python values."""
    match leaf:
        case np.ndarray():
            return jnp.asarray(leaf)
        case Array() | int() | float():
            return leaf
        case unreachable:
            assert_never(unreachable)


def initialize_formal(  # noqa: PLR0913, PLR0917
    maps: TrainingMapSampler,
    config: HostConfig,
    settings: TrainAConfig,
    key: Array,
    identity: str,
    resume: str | Path | None,
) -> FormalState:
    """Construct or exactly restore every bucket pool and the one shared model state."""
    first = maps.schedule(0)
    first_config = replace(config, n_agents=first.n_agents, agent_bucket=6)
    template = initialize_training(
        maps.load(first), first_config, replace(settings, batch_size=1), key
    )
    models = ModelState(
        template.actor,
        template.critic,
        template.predictor,
        template.updates,
        template.predictor_version,
        template.replay,
    )
    state = FormalState(
        models,
        (),
        0,
        0,
        key,
        jnp.zeros(2, jnp.int32),
        jnp.zeros(2, jnp.int32),
        jnp.zeros(5, jnp.int32),
        jnp.zeros(5, jnp.int32),
        -1.0,
        0,
    )
    if resume is not None:
        restored = serialization.msgpack_restore(load_checkpoint(resume, identity))
        schema = restored["metadata"].get("pool_schema")
        if schema not in (2, PACKED_POOL_SCHEMA):
            raise TrainingError(field="checkpoint", detail="incompatible physical-lane schema")
        if schema == PACKED_POOL_SCHEMA:
            restored["state"] = _unpack_booleans(restored["state"])
        pools = tuple(
            _pool(maps.schedule(indices[0]), maps, config, settings, key)._replace(
                sample_indices=tuple(indices)
            )
            for indices in restored["metadata"]["pool_samples"]
        )
        state = serialization.from_state_dict(state._replace(pools=pools), restored["state"])
        state = jax.tree.map(_restore_array, state)
    return state


def write_summary(  # noqa: PLR0913
    state: FormalState,
    settings: TrainAConfig,
    checkpoint: Path,
    identity: str,
    destination: Path,
    *,
    session_real_events: int = 0,
    session_seconds: float = 0.0,
    stage_seconds: StageSeconds | None = None,
) -> None:
    """Report actual episode and event mixture counts without conflating their weights."""
    summary = {
        "real_events": state.real_events,
        "warmup_events": min(settings.warmup_events, state.real_events),
        "updates": int(state.models.updates),
        "schedule_cursor": state.cursor,
        "pools": len(state.pools),
        "physical_env_count": settings.physical_env_count,
        "assigned_lanes": sum(index >= 0 for pool in state.pools for index in pool.sample_indices),
        "source_episode_counts": dict(zip(SOURCES, map(int, state.source_episodes), strict=True)),
        "source_event_counts": dict(zip(SOURCES, map(int, state.source_events), strict=True)),
        "family_episode_counts": dict(zip(FAMILIES, map(int, state.family_episodes), strict=True)),
        "family_event_counts": dict(zip(FAMILIES, map(int, state.family_events), strict=True)),
        "checkpoint": str(checkpoint),
        "configuration_hash": identity,
        "best_validation_auc": state.best_auc if state.last_validation else None,
        "last_validation_event": state.last_validation,
        "session_real_events": session_real_events,
        "session_seconds": session_seconds,
        "session_events_per_second": session_real_events / session_seconds
        if session_seconds
        else 0.0,
        "stage_seconds": stage_seconds,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def learn_models(  # noqa: PLR0913
    models: ModelState,
    batch: RolloutBatch,
    plan_batch: PlanTrainingBatch,
    returns: Returns,
    settings: TrainAConfig,
    *,
    warming: bool,
    update_ppo: PPOUpdate,
    update_plan: PlanUpdate,
    diagnose: Callable,
    shuffle_key: Array,
) -> tuple[ModelState, Array, dict]:
    """Keep predictor updates strictly after all frozen-feature PPO epochs."""
    metrics = {}
    if not warming:
        updated_actor, updated_critic, metrics = optimize_ppo(
            models.actor,
            models.critic,
            batch,
            returns,
            epochs=settings.ppo_epochs,
            minibatch_sequences=settings.ppo_minibatch_sequences,
            target_kl=settings.ppo_target_kl,
            key=shuffle_key,
            update=update_ppo,
            diagnose=diagnose,
        )
        models = models._replace(actor=updated_actor, critic=updated_critic)
    plan_value = jnp.float32(0)
    if settings.neighbor_estimator == "predict":
        replay, training_plans = sample_and_store(
            models.replay, plan_batch, settings.predictor_recent_fraction
        )
        updated_predictor, plan_value = update_plan(models.predictor, training_plans)
        models = models._replace(
            predictor=updated_predictor,
            predictor_version=models.predictor_version + 1,
            replay=replay,
        )
    return models._replace(updates=models.updates + 1), plan_value, metrics
