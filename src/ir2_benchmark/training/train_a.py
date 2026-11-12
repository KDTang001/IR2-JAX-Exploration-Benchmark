"""Real-interaction plan-assisted PPO, resumable at immutable rollout boundaries."""

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax import Array

from ir2_benchmark.agents.actor import Actor
from ir2_benchmark.agents.critic import Critic
from ir2_benchmark.config import HostConfig
from ir2_benchmark.models.plan_predictor import PlanPredictor
from ir2_benchmark.training.checkpoint import load_checkpoint, save_checkpoint
from ir2_benchmark.training.collect_a import (
    Parameters,
    make_collector,
)
from ir2_benchmark.training.plan_losses import plan_update as _plan_update
from ir2_benchmark.training.ppo import optimize_ppo, ppo_diagnostics, ppo_update
from ir2_benchmark.training.predictor_replay import mix_live_predictor_events
from ir2_benchmark.training.recording_a import close_recorder, record_collection, recorder_for
from ir2_benchmark.training.state_a import (
    StageSeconds,
    TrainAConfig,
    TrainASummary,
    TrainingError,
    TrainingState,
    _limits,
    _reset_finished,
    actor_parameter_change,
    configuration_hash,
    initialize_training,
    write_configuration,
)
from ir2_benchmark.training.storage import pack_collections
from ir2_benchmark.types import MapInput, validate_graph_state
from ir2_benchmark.visualization.recording import ModelVersions

if TYPE_CHECKING:
    from ir2_benchmark.visualization.recording import RecordingConfig

__all__ = [
    "TrainAConfig",
    "TrainASummary",
    "TrainingError",
    "TrainingState",
    "configuration_hash",
    "initialize_training",
    "train_a",
]


# The synchronized collect/update/persist phases stay visible in one host loop.
def train_a(  # noqa: PLR0913, PLR0915
    map_data: MapInput,
    config: HostConfig,
    settings: TrainAConfig,
    key: Array,
    output: str | Path,
    *,
    resume: str | Path | None = None,
    scenario_sampler: Callable[[int], MapInput] | None = None,
    stop_after_updates: int | None = None,
    recording: "RecordingConfig | None" = None,
) -> TrainASummary:
    """Run real warmup/rollout/PPO/predictor phases and checkpoint every boundary.

    Formal runs require the caller's balanced training-manifest sampler. Repeating
    the supplied development map is an explicit micro-suite behavior only.
    """
    session_start = perf_counter()
    if settings.suite == "formal" and scenario_sampler is None:
        raise TrainingError(
            field="maps", detail="formal training requires a balanced training-manifest sampler"
        )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    identity = configuration_hash(config, settings)
    write_configuration(destination, config, settings, identity)
    state = initialize_training(map_data, config, settings, key)
    if resume is not None:
        state = serialization.from_bytes(state, load_checkpoint(resume, identity))
        state = jax.tree.map(jnp.asarray, state)
    recorder = recorder_for(destination, recording)
    session_start_events = int(jnp.sum(state.sampler.real_events))
    stage_seconds: StageSeconds = {
        "collect": 0.0,
        "update": 0.0,
        "checkpoint": 0.0,
        "validation": 0.0,
        "recording": 0.0,
    }
    used_collectors: set[bool] = set()
    initial_actor = jax.tree.map(np.asarray, state.actor.params)
    actor = Actor(settings.hidden, candidate_attention=settings.candidate_attention)
    critic = Critic(settings.hidden)
    predictor = PlanPredictor(config.plan_horizon, settings.hidden)
    collectors = {
        warmup: make_collector(
            config,
            actor,
            critic,
            predictor,
            length=settings.rollout_length,
            learned_communication=settings.communication_mode == "learned",
            predict_neighbors=settings.neighbor_estimator == "predict",
            use_neighbor_plan=settings.neighbor_estimator != "none",
            fixed_plan=settings.fixed_plan,
            warmup=warmup,
        )
        for warmup in (False, True)
    }
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
    start_updates = int(state.updates)
    checkpoint = destination / "checkpoint"
    with (destination / "updates.jsonl").open("a", encoding="utf-8") as log:
        while int(jnp.sum(state.sampler.real_events)) < settings.events:
            interval_start = perf_counter()
            real_events = int(jnp.sum(state.sampler.real_events))
            warming = real_events < settings.warmup_events
            limits = _limits(
                settings.warmup_events if warming else settings.events, settings.batch_size
            )
            collect_start = perf_counter()
            first_use = warming not in used_collectors
            used_collectors.add(warming)
            state = _reset_finished(state, map_data, config, settings, limits, scenario_sampler)
            initial_sampler = state.sampler
            collection = collectors[warming](
                state.sampler,
                Parameters(state.actor.params, state.critic.params, state.predictor.params),
                limits,
                state.predictor_version,
            )
            state = state._replace(sampler=collection.sampler)
            validate_graph_state(collection.sampler.environment)
            valid_events = int(jnp.sum(collection.rollout.valid))
            collect_seconds = perf_counter() - collect_start
            recording_seconds = record_collection(
                recorder,
                initial_sampler,
                collection,
                config,
                episode_indices=tuple(
                    int(episode) * settings.batch_size + lane
                    for lane, episode in enumerate(initial_sampler.environment.episode)
                ),
                training_events=real_events,
                versions=ModelVersions(
                    int(state.actor.step),
                    int(state.predictor.step),
                    "warmup" if warming else "train",
                ),
            )
            update_start = perf_counter()
            raw = collection.rollout
            batch, _, returns = pack_collections(
                [collection], int(jnp.sum(raw.valid)), sequence_length=settings.sequence_length
            )
            ppo_metrics = {}
            if not warming:
                updated_actor, updated_critic, ppo_metrics = optimize_ppo(
                    state.actor,
                    state.critic,
                    batch,
                    returns,
                    epochs=settings.ppo_epochs,
                    minibatch_sequences=settings.ppo_minibatch_sequences,
                    target_kl=settings.ppo_target_kl,
                    key=jax.random.fold_in(key, state.updates),
                    update=update_ppo,
                    diagnose=diagnose,
                )
                state = state._replace(actor=updated_actor, critic=updated_critic)
            plan_value = jnp.float32(0)
            if settings.neighbor_estimator == "predict":
                replay, training_plans = mix_live_predictor_events(
                    state.replay, collection.plans, raw.valid, settings.predictor_recent_fraction
                )
                updated_predictor, plan_value = update_plan(state.predictor, training_plans)
                state = state._replace(
                    predictor=updated_predictor,
                    predictor_version=state.predictor_version + 1,
                    replay=replay,
                )
            state = state._replace(updates=state.updates + 1)
            jax.block_until_ready((state.actor, state.critic, state.predictor, state.replay))
            update_seconds = perf_counter() - update_start
            checkpoint_start = perf_counter()
            save_checkpoint(checkpoint, serialization.to_bytes(state), identity)
            checkpoint_seconds = perf_counter() - checkpoint_start
            interval_stages: StageSeconds = {
                "collect": collect_seconds,
                "update": update_seconds,
                "checkpoint": checkpoint_seconds,
                "validation": 0.0,
                "recording": recording_seconds,
            }
            stage_seconds["collect"] += collect_seconds
            stage_seconds["update"] += update_seconds
            stage_seconds["checkpoint"] += checkpoint_seconds
            stage_seconds["recording"] += recording_seconds
            log.write(
                json.dumps(
                    {
                        "real_events": int(jnp.sum(state.sampler.real_events)),
                        "warmup": warming,
                        "plan_loss": float(plan_value),
                        "valid_events": valid_events,
                        "updates": int(state.updates),
                        "ppo": ppo_metrics,
                        "interval_real_events": valid_events,
                        "interval_seconds": perf_counter() - interval_start,
                        "stage_seconds": interval_stages,
                        "physical_env_count": settings.batch_size,
                        "sample_slots": raw.valid.size,
                        "padding_slots": raw.valid.size - valid_events,
                        "first_use_collector_keys": [warming] if first_use else [],
                    }
                )
                + "\n"
            )
            log.flush()
            if (
                stop_after_updates is not None
                and int(state.updates) - start_updates >= stop_after_updates
            ):
                break
    stage_seconds["recording"] += close_recorder(recorder, int(jnp.sum(state.sampler.real_events)))
    parameter_change = actor_parameter_change(initial_actor, state.actor.params)
    session_real_events = int(jnp.sum(state.sampler.real_events)) - session_start_events
    session_seconds = perf_counter() - session_start
    summary = TrainASummary(
        int(jnp.sum(state.sampler.real_events)),
        min(settings.warmup_events, int(jnp.sum(state.sampler.real_events))),
        int(state.updates),
        int(jnp.sum(state.sampler.environment.episode)) + settings.batch_size,
        parameter_change,
        str(checkpoint),
        identity,
        session_real_events=session_real_events,
        session_seconds=session_seconds,
        session_events_per_second=session_real_events / session_seconds,
        stage_seconds=stage_seconds,
        physical_env_count=settings.batch_size,
    )
    (destination / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary
