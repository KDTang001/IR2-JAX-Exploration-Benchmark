"""Formal map/robot-bucket dispatch with shared models and persistent episode pools."""

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from ir2_benchmark.agents.actor import Actor
from ir2_benchmark.agents.critic import Critic
from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.validation import ValidationEvaluator
from ir2_benchmark.maps.sampler import TrainingMapSampler
from ir2_benchmark.models.plan_predictor import PlanPredictor
from ir2_benchmark.training.collect_a import (
    Parameters,
    make_collector,
)
from ir2_benchmark.training.dispatch_a import account_events, activate_pool
from ir2_benchmark.training.pools_a import (
    _save,
    initialize_formal,
    learn_models,
    write_summary,
)
from ir2_benchmark.training.ppo import ppo_diagnostics, ppo_update
from ir2_benchmark.training.recording_a import close_recorder, record_collection, recorder_for
from ir2_benchmark.training.selection_a import prepare_output, record_training_total, validate_due
from ir2_benchmark.training.storage import TrainingSlice, pack_collections
from ir2_benchmark.training.train_a import (
    TrainAConfig,
    TrainingError,
    _plan_update,
)
from ir2_benchmark.types import validate_graph_state
from ir2_benchmark.visualization.recording import ModelVersions

if TYPE_CHECKING:
    from ir2_benchmark.training.state_a import StageSeconds
    from ir2_benchmark.visualization.recording import RecordingConfig


def collection_length(capacity: int, lanes: int, maximum: int) -> int:
    """Drop quota-only scan tails, using a bounded set of compiled lengths."""
    needed = (capacity + lanes - 1) // lanes
    if not 0 < needed <= maximum:
        message = "collection quota exceeds the configured scan capacity"
        raise ValueError(message)
    return min(maximum, max(min(16, maximum), 1 << (needed - 1).bit_length()))


# The synchronized collect/update/persist phases stay visible in one host loop.
def train_a_formal(  # noqa: PLR0913, PLR0915
    maps: TrainingMapSampler,
    config: HostConfig,
    settings: TrainAConfig,
    key: Array,
    output: str | Path,
    *,
    resume: str | Path | None = None,
    stop_after_updates: int | None = None,
    validation: ValidationEvaluator | None = None,
    validation_interval: int = 100_000,
    recording: "RecordingConfig | None" = None,
) -> Path:
    """Sample maps 50/50 by episodes and keep separate (bucket,N) episode carries.

    Logical batches contain batch_size*rollout_length real events. Each compiled
    collector vmaps physical_env_count independent native-bucket environments;
    parameters remain frozen until the complete logical batch has been collected.
    """
    session_start = perf_counter()
    if settings.suite != "formal":
        raise TrainingError(field="suite", detail="the bucket driver requires suite=formal")
    destination, identity = prepare_output(
        maps, config, settings, output, validation, validation_interval
    )
    state = initialize_formal(maps, config, settings, key, identity, resume)
    recorder = recorder_for(destination, recording)
    session_start_events = state.real_events
    stage_seconds: StageSeconds = {
        "collect": 0.0,
        "update": 0.0,
        "checkpoint": 0.0,
        "validation": 0.0,
        "recording": 0.0,
    }
    actor = Actor(settings.hidden, candidate_attention=settings.candidate_attention)
    critic = Critic(settings.hidden)
    predictor = PlanPredictor(config.plan_horizon, settings.hidden)
    update_ppo = jax.jit(
        lambda a, c, batch, returns: ppo_update(
            a,
            c,
            batch,
            returns,
            actor,
            critic,
            learned_communication=settings.communication_mode == "learned",
            normalize_advantages=False,
        )
    )
    diagnose = jax.jit(
        lambda a, c, batch, returns: ppo_diagnostics(
            a,
            c,
            batch,
            returns,
            learned_communication=settings.communication_mode == "learned",
            moments=True,
        )
    )
    update_plan = jax.jit(lambda p, batch: _plan_update(predictor, p, batch))

    # Omit only quota-frozen scan tails; logical batches retain batch_size*rollout_length events.
    chunk_length = min(
        settings.rollout_length,
        (settings.batch_size * settings.rollout_length + settings.physical_env_count - 1)
        // settings.physical_env_count,
    )
    collectors = {}
    checkpoint = destination / "checkpoint"
    state = validate_due(state, checkpoint, identity, validation, validation_interval)
    start_updates = int(state.models.updates)
    with (
        (destination / "updates.jsonl").open("a", encoding="utf-8") as log,
        (destination / "map-episodes.jsonl").open("a", encoding="utf-8") as episodes,
    ):
        while state.real_events < settings.events:
            interval_start = perf_counter()
            warming = state.real_events < settings.warmup_events
            boundary = min(
                settings.warmup_events if warming else settings.events,
                state.last_validation + validation_interval,
            )
            target = min(
                settings.batch_size * settings.rollout_length, boundary - state.real_events
            )
            collections: list[TrainingSlice] = []
            count = 0
            sample_slots = 0
            first_use_keys = []
            recording_seconds = 0.0
            collect_start = perf_counter()
            while count < target:
                state, index, pool_key = activate_pool(state, maps, config, settings, episodes)
                pool = state.pools[index]
                pool_config = replace(config, n_agents=pool.n_agents, agent_bucket=6)
                capacity = min(target - count, chunk_length * settings.physical_env_count)
                length = collection_length(capacity, settings.physical_env_count, chunk_length)
                cache_key = (*pool_key, warming, length)
                if cache_key not in collectors:
                    first_use_keys.append(cache_key)
                    collectors[cache_key] = make_collector(
                        pool_config,
                        actor,
                        critic,
                        predictor,
                        length=length,
                        learned_communication=settings.communication_mode == "learned",
                        predict_neighbors=settings.neighbor_estimator == "predict",
                        use_neighbor_plan=settings.neighbor_estimator != "none",
                        fixed_plan=settings.fixed_plan,
                        warmup=warming,
                    )
                quotas = jnp.asarray(
                    [
                        capacity // settings.physical_env_count
                        + int(lane < capacity % settings.physical_env_count)
                        for lane in range(settings.physical_env_count)
                    ],
                    jnp.int32,
                )
                limit = pool.sampler.real_events + quotas
                result = collectors[cache_key](
                    pool.sampler,
                    Parameters(
                        state.models.actor.params,
                        state.models.critic.params,
                        state.models.predictor.params,
                    ),
                    limit,
                    state.models.predictor_version,
                )
                valid_count = int(jnp.sum(result.rollout.valid))
                validate_graph_state(result.sampler.environment)
                sample_slots += result.rollout.valid.size
                recording_seconds += record_collection(
                    recorder,
                    pool.sampler,
                    result,
                    pool_config,
                    episode_indices=pool.sample_indices,
                    training_events=state.real_events,
                    versions=ModelVersions(
                        int(state.models.actor.step),
                        int(state.models.predictor.step),
                        "warmup" if warming else "train",
                    ),
                )
                state = account_events(state, index, result, maps, valid_count)
                if valid_count:
                    collections.append(
                        TrainingSlice(result.rollout, result.plans, result.bootstrap)
                    )
                    count += valid_count
                del pool, result
            collect_seconds = perf_counter() - collect_start - recording_seconds
            update_start = perf_counter()
            batch, plan_batch, returns = pack_collections(
                collections, target, sequence_length=settings.sequence_length
            )
            del collections
            models, plan_value, ppo_metrics = learn_models(
                state.models,
                batch,
                plan_batch,
                returns,
                settings,
                warming=warming,
                update_ppo=update_ppo,
                update_plan=update_plan,
                diagnose=diagnose,
                shuffle_key=jax.random.fold_in(state.root_key, state.models.updates),
            )
            state = state._replace(models=models)
            jax.block_until_ready(models)
            update_seconds = perf_counter() - update_start
            checkpoint_start = perf_counter()
            _save(checkpoint, state, identity)
            checkpoint_seconds = perf_counter() - checkpoint_start
            validation_start = perf_counter()
            state = validate_due(state, checkpoint, identity, validation, validation_interval)
            validation_seconds = perf_counter() - validation_start
            interval_stages: StageSeconds = {
                "collect": collect_seconds,
                "update": update_seconds,
                "checkpoint": checkpoint_seconds,
                "validation": validation_seconds,
                "recording": recording_seconds,
            }
            stage_seconds["collect"] += collect_seconds
            stage_seconds["update"] += update_seconds
            stage_seconds["checkpoint"] += checkpoint_seconds
            stage_seconds["validation"] += validation_seconds
            stage_seconds["recording"] += recording_seconds
            log.write(
                json.dumps(
                    {
                        "real_events": state.real_events,
                        "valid_events": count,
                        "warmup": warming,
                        "plan_loss": float(plan_value),
                        "schedule_cursor": state.cursor,
                        "pools": len(state.pools),
                        "updates": int(state.models.updates),
                        "ppo": ppo_metrics,
                        "interval_real_events": count,
                        "interval_seconds": perf_counter() - interval_start,
                        "stage_seconds": interval_stages,
                        "physical_env_count": settings.physical_env_count,
                        "sample_slots": sample_slots,
                        "padding_slots": sample_slots - count,
                        "first_use_collector_keys": first_use_keys,
                        "packed_sequence_slots": batch.valid.size,
                        "packed_batch_bytes": sum(value.nbytes for value in jax.tree.leaves(batch)),
                        "device_memory": jax.devices()[0].memory_stats(),
                    }
                )
                + "\n"
            )
            log.flush()
            del batch, plan_batch, returns, models
            if (
                stop_after_updates is not None
                and int(state.models.updates) - start_updates >= stop_after_updates
            ):
                break
    stage_seconds["recording"] += close_recorder(recorder, state.real_events)
    write_summary(
        state,
        settings,
        checkpoint,
        identity,
        destination,
        session_real_events=state.real_events - session_start_events,
        session_seconds=perf_counter() - session_start,
        stage_seconds=stage_seconds,
    )
    record_training_total(checkpoint, state.real_events)
    return checkpoint
