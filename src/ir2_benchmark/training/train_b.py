"""Real replay -> posterior world fitting -> closed imagination -> discrete policy updates."""

from __future__ import annotations

# SIZE_OK: One ordered host driver owns native-bucket dispatch and checkpoint consistency.
import json
import shutil
from dataclasses import asdict, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax import Array

from ir2_benchmark.communication.faults import KeyPurpose, fold_key
from ir2_benchmark.communication.sending import validate_builtin_policy
from ir2_benchmark.config import HostConfig, load_config
from ir2_benchmark.models.plan_predictor import PlanMemory
from ir2_benchmark.models.world_model import initial_rssm
from ir2_benchmark.presets import preset_manifest
from ir2_benchmark.training.b_collect import make_real_collector
from ir2_benchmark.training.b_pools import activate_pool, micro_pool, new_pool, refresh_micro
from ir2_benchmark.training.b_state import (
    BTrainingConfig,
    Counters,
    TrainingPool,
    _cognition,
    _decision_extras,
    _frame,
    _initialize,
    _named_fields,
    _real_policy_batch,
    _update_predictor,
    plan_training_batch,
)
from ir2_benchmark.training.checkpoint import load_checkpoint, save_checkpoint
from ir2_benchmark.training.dreamer import (
    DreamerConfig,
    ImaginedBatch,
    pad_policy_batch,
    update_imagined,
)
from ir2_benchmark.training.evaluate_b import _evaluate_policy, model_diagnostics
from ir2_benchmark.training.evaluate_b import benchmark_functions as benchmark_functions  # noqa: PLC0414
from ir2_benchmark.training.evaluate_b import run_evaluation as run_evaluation  # noqa: PLC0414
from ir2_benchmark.training.imagination import (
    ImaginedState,
    PolicyCarry,
    imagine_rollout,
)
from ir2_benchmark.training.replay import ReplayBuffer, ReplayError, read_field
from ir2_benchmark.training.storage import StandardAction
from ir2_benchmark.training.world_losses import (
    _world_batch,
    posterior_sequence,
    update_world,
)
from ir2_benchmark.types import plan_labels, validate_graph_state
from ir2_benchmark.visualization.recording import ModelVersions, RecordedEvent, TrainingRecorder

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

    from flax.training.train_state import TrainState

    from ir2_benchmark.evaluation.validation import ValidationEvaluator
    from ir2_benchmark.maps.sampler import TrainingMapSampler
    from ir2_benchmark.training.b_types import CheckpointValue, TrainingSummary
    from ir2_benchmark.visualization.recording import RecordingConfig


def train_b(  # noqa: C901, PLR0912, PLR0913, PLR0915
    config: HostConfig,
    seed: int,
    events: int,
    output: str | Path,
    *,
    options: BTrainingConfig | None = None,
    map_path: Path | None = None,
    resume: Path | None = None,
    sampler: TrainingMapSampler | None = None,
    validation: ValidationEvaluator | None = None,
    stop_after_updates: int | None = None,
    recording: RecordingConfig | None = None,
) -> TrainingSummary:
    """Train on actual events, update from modeled events, then evaluate separately in the real simulator."""  # noqa: E501
    if events < 1:
        raise ReplayError("real_event_budget_must_be_positive")  # noqa: EM101
    session_start = perf_counter()
    stage_seconds = dict.fromkeys(
        ("collect", "update", "checkpoint", "validation", "recording"), 0.0
    )
    if stop_after_updates is not None and stop_after_updates < 1:
        raise ReplayError("stop_after_updates_must_be_positive")  # noqa: EM101
    if options is None:
        options = (
            BTrainingConfig()
            if sampler is not None
            else BTrainingConfig(
                logical_batch=1, update_every=64, warmup_events=256, development=map_path is None
            )
        )
    validate_builtin_policy(config, options.learned_communication)
    if not (
        0 <= options.burn_in < options.sequence_length
        and options.logical_batch >= 1
        and options.update_every >= 1
    ):
        raise ReplayError("invalid_B_sequence_or_batch_settings")  # noqa: EM101
    if map_path is None and sampler is None and not options.development:
        raise ReplayError("formal_training_requires_explicit_training_map_source")  # noqa: EM101
    if (
        sampler is not None
        and not options.development
        and events >= options.validation_interval
        and validation is None
    ):
        raise ReplayError("formal_training_requires_fixed_validation_manifest")  # noqa: EM101
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    recorder = (
        TrainingRecorder(root / "gifs", recording, training_seed=seed, route="B")
        if recording
        else None
    )
    base_config = config
    configuration = {
        "environment": asdict(config),
        "implementations": preset_manifest(config),
        "training": asdict(options),
        "seed": seed,
        "event_budget": events,
        "map_path": str(map_path) if map_path else None,
        "physical_microbatch": 1,
        "manifest_hash": sampler.manifest_hash if sampler else None,
        "validation_manifest_hash": validation.manifest_hash if validation else None,
        "evaluation_decision_rule": "stochastic_with_fixed_seed",
    }
    config_hash = sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    (root / "config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    required_capacity = (
        options.physical_env_count * config.max_rounds * (6 if sampler else config.n_agents)
    )
    if options.replay_capacity < required_capacity:
        reason = f"replay_capacity_requires_{required_capacity}_events_for_concurrent_open_episodes"
        raise ReplayError(reason)
    key = jax.random.PRNGKey(seed)
    pools: dict[str, TrainingPool] = {}
    schedule_cursor = 0
    pool_key = "micro"
    if sampler is not None:
        sample = sampler.schedule(schedule_cursor)
        first_pool = new_pool(sample, sampler, base_config, options, seed)
        pool_key = f"{sample.bucket}x{sample.n_agents}"
        pools[pool_key] = first_pool
        config = first_pool.config
        schedule_cursor += 1
    else:
        pools[pool_key] = micro_pool(config, options, seed, map_path)
    pool = pools[pool_key]
    state, carry, latent = jax.tree.map(
        lambda value: value[0], (pool.state, pool.carry, pool.latent)
    )
    actor, predictor, world, critic, carry, latent = _initialize(config, options, state, key)
    replay = pool.replay
    counters = Counters()
    episode = 0
    actor_delta = 0.0
    last_model_loss = None
    best_auc = None
    last_validation_event = 0
    update_pending = False
    saved_replays: dict[str, tuple[int, str]] = {}
    source_events = {"ir2": 0, "procedural": 0}
    family_events = dict.fromkeys(("easy", "difficult", "rooms", "corridor", "hybrid"), 0)
    source_episodes = dict.fromkeys(source_events, 0)
    family_episodes = dict.fromkeys(family_events, 0)
    if sampler is not None:
        source_episodes[sample.source] += 1
        family_episodes[sample.family] += 1

    def payload() -> Mapping[str, CheckpointValue]:
        return {
            "lane_schema": 2,
            "update_pending": update_pending,
            "actor": actor,
            "predictor": predictor,
            "world": world,
            "critic": critic,
            "carry": carry,
            "latent": latent,
            "state": state,
            "key": key,
            "counters": jnp.asarray(counters, jnp.int32),
            "episode": jnp.asarray(episode),
            "schedule_cursor": schedule_cursor,
            "pool_key": pool_key,
            "actor_delta": actor_delta,
            "last_model_loss": last_model_loss,
            "source_events": source_events,
            "family_events": family_events,
            "source_episodes": source_episodes,
            "family_episodes": family_episodes,
            "best_auc": best_auc,
            "last_validation_event": last_validation_event,
            "pools": {
                name: {
                    "state": pool.state,
                    "carry": pool.carry,
                    "latent": pool.latent,
                    "episode": pool.episode,
                    "events": pool.events,
                    "sample_indices": jnp.asarray(pool.sample_indices),
                    "lane_episodes": jnp.asarray(pool.lane_episodes),
                    "lane_cursor": pool.lane_cursor,
                    "config": asdict(pool.config),
                }
                for name, pool in pools.items()
            },
        }

    def persist() -> None:
        checkpoint_start = perf_counter()
        jax.block_until_ready((actor.params, predictor.params, world.params, critic.params))
        save_checkpoint(root / "checkpoint", serialization.to_bytes(payload()), config_hash)
        if sampler is not None:
            (root / "replay").mkdir(exist_ok=True)
        for name, pool in pools.items():
            identity = (pool.events, pool.replay.sampling_state)
            if saved_replays.get(name) == identity:
                continue
            path = root / "replay.npz" if sampler is None else root / "replay" / f"{name}.npz"
            pool.replay.save(path)
            saved_replays[name] = identity
        stage_seconds["checkpoint"] += perf_counter() - checkpoint_start

    if resume is not None:
        recovered = serialization.msgpack_restore(
            load_checkpoint(resume / "checkpoint", config_hash)
        )
        if recovered["lane_schema"] != 2:  # noqa: PLR2004
            raise ReplayError("incompatible_B_physical_lane_checkpoint_schema")  # noqa: EM101
        # Legacy checkpoints retain their original resume behavior; no phase is inferred.
        update_pending = recovered.get("update_pending", False)
        if not isinstance(update_pending, bool):
            raise ReplayError("invalid_B_pending_update_marker")  # noqa: EM101
        actor, predictor, world, critic = (
            serialization.from_state_dict(template, recovered[name])
            for name, template in (
                ("actor", actor),
                ("predictor", predictor),
                ("world", world),
                ("critic", critic),
            )
        )
        carry = serialization.from_state_dict(carry, recovered["carry"])
        latent = serialization.from_state_dict(latent, recovered["latent"])
        state = serialization.from_state_dict(state, recovered["state"])
        key = jnp.asarray(recovered["key"])
        counters = Counters(*(int(value) for value in recovered["counters"]))
        episode = int(recovered["episode"])
        actor_delta, last_model_loss = float(recovered["actor_delta"]), recovered["last_model_loss"]
        source_events, family_events = recovered["source_events"], recovered["family_events"]
        source_episodes, family_episodes = (
            recovered["source_episodes"],
            recovered["family_episodes"],
        )
        best_auc, last_validation_event = (
            recovered["best_auc"],
            int(recovered["last_validation_event"]),
        )
        schedule_cursor = int(recovered["schedule_cursor"])
        pool_key = recovered["pool_key"]
        pools = {
            name: TrainingPool(
                HostConfig(**record["config"]),
                jax.tree.map(jnp.asarray, serialization.from_state_dict(state, record["state"])),
                jax.tree.map(jnp.asarray, serialization.from_state_dict(carry, record["carry"])),
                jax.tree.map(jnp.asarray, serialization.from_state_dict(latent, record["latent"])),
                ReplayBuffer.load(
                    resume / "replay" / f"{name}.npz" if sampler else resume / "replay.npz"
                ),
                int(record["episode"]),
                tuple(map(int, record["sample_indices"])),
                tuple(map(int, record["lane_episodes"])),
                int(record["events"]),
                int(record["lane_cursor"]),
            )
            for name, record in recovered["pools"].items()
        }
        config, replay = pools[pool_key].config, pools[pool_key].replay

    session_initial_events = counters.real_events
    session_initial_updates = counters.model_updates

    def maybe_validate() -> None:
        nonlocal best_auc, last_validation_event
        if (
            validation is not None
            and counters.real_events // options.validation_interval
            > last_validation_event // options.validation_interval
        ):
            persist()
            validation_start = perf_counter()
            score = validation(root / "checkpoint", counters.real_events)
            stage_seconds["validation"] += perf_counter() - validation_start
            if not np.isfinite(score.mean_auc) or not 0 <= score.mean_auc <= 1:
                raise ReplayError("validation_auc_requires_finite_fraction")  # noqa: EM101
            selected = best_auc is None or score.mean_auc > best_auc
            if selected:
                best_auc = score.mean_auc
                (root / "best").mkdir(exist_ok=True)
                shutil.copyfile(root / "checkpoint", root / "best" / "checkpoint")
                shutil.copyfile(root / "config.json", root / "best" / "config.json")
                (root / "best" / "summary.json").write_text(
                    json.dumps(
                        {
                            "route": "B",
                            **counters._asdict(),
                            "selected_at_events": counters.real_events,
                            "total_training_events": counters.real_events,
                            "validation_auc": score.mean_auc,
                            "validation_manifest_hash": score.manifest_hash,
                            "checkpoint": str(root / "best" / "checkpoint"),
                            "formal_complete": False,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            last_validation_event = counters.real_events
            with (root / "validation.jsonl").open("a", encoding="utf-8") as validation_log:
                validation_log.write(
                    json.dumps(
                        {
                            "real_events": counters.real_events,
                            "milestone": counters.real_events // options.validation_interval,
                            "mean_auc": score.mean_auc,
                            "scenario_hash": score.manifest_hash,
                            "results": str(score.output),
                            "evaluated": score.evaluated,
                            "selected": selected,
                            "selected_checkpoint": str(root / "best" / "checkpoint"),
                        }
                    )
                    + "\n"
                )
            persist()

    maybe_validate()
    radius = int(np.ceil(config.sensor_radius))

    world_update = jax.jit(lambda world, batches, key: update_world(world, batches, key, options))
    posterior = jax.jit(lambda world, batch, key: posterior_sequence(world, batch, key, options))
    predictor_update = jax.jit(_update_predictor)
    actor_update = jax.jit(
        lambda actor, critic, batch: update_imagined(
            actor,
            critic,
            batch,
            DreamerConfig(
                entropy=options.entropy, learned_communication=options.learned_communication
            ),
        )
    )

    @partial(jax.jit, static_argnames=("config",))
    def imagination(  # noqa: PLR0913, PLR0917
        initial: ImaginedState,
        carry: PolicyCarry,
        actor: TrainState,
        predictor: TrainState,
        world: TrainState,
        key: Array,
        config: HostConfig,
    ) -> tuple[ImaginedState, PolicyCarry, ImaginedBatch]:
        return imagine_rollout(
            initial,
            carry,
            actor,
            predictor,
            world,
            config,
            key,
            options.imagination_rounds * config.n_agents,
            learned_communication=options.learned_communication,
            predicted_plans=options.predicted_plans,
            use_neighbor_plan=options.use_neighbor_plan,
        )

    start = perf_counter()
    log_path = root / "updates.jsonl"
    if sampler is not None and resume is None:
        with (root / "training_maps.jsonl").open("w", encoding="utf-8") as maps_log:
            maps_log.write(json.dumps(asdict(sampler.schedule(0)), sort_keys=True) + "\n")
    with log_path.open("a" if resume else "w", encoding="utf-8") as log:
        interval_start = perf_counter()
        interval_initial_events = counters.real_events
        interval_stages = stage_seconds.copy()
        while (update_pending or counters.real_events < events) and (
            stop_after_updates is None
            or counters.model_updates - session_initial_updates < stop_after_updates
        ):
            if not update_pending:
                collect_start = perf_counter()
                recording_before = stage_seconds["recording"]
                current = pools[pool_key]
                if sampler is not None and (
                    counters.real_events % options.update_every == 0
                    or bool(jnp.any(current.state.terminated | current.state.truncated))
                ):
                    pool_key, schedule_cursor, assigned = activate_pool(
                        pools, schedule_cursor, sampler, base_config, options, seed
                    )
                    if assigned:
                        with (root / "training_maps.jsonl").open("a", encoding="utf-8") as maps_log:
                            for sample in assigned:
                                maps_log.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
                                source_episodes[sample.source] += 1
                                family_episodes[sample.family] += 1
                elif sampler is None:
                    pools[pool_key] = refresh_micro(current, options, seed, map_path)
                pool = pools[pool_key]
                config, replay = pool.config, pool.replay
                if bool(jnp.any(pool.state.terminated)):
                    raise ReplayError("development_map_already_complete_at_spawn")  # noqa: EM101
                warmup = counters.real_events < options.warmup_events
                remaining = min(
                    options.physical_env_count,
                    events - counters.real_events,
                    options.update_every - counters.real_events % options.update_every,
                    options.warmup_events - counters.real_events if warmup else events,
                    options.validation_interval - counters.real_events % options.validation_interval
                    if validation is not None
                    else events,
                )
                enabled = (
                    jnp.arange(options.physical_env_count) - pool.lane_cursor
                ) % options.physical_env_count < remaining
                identifiers = pool.sample_indices if sampler else pool.lane_episodes
                event_keys = jnp.stack(
                    [
                        fold_key(
                            jax.random.PRNGKey(seed),
                            KeyPurpose.POLICY,
                            identifiers[lane],
                            pool.state.event[lane],
                            pool.state.current_agent[lane],
                        )
                        for lane in range(options.physical_env_count)
                    ]
                )
                if recorder is not None:
                    recording_start = perf_counter()
                    for lane in np.flatnonzero(np.asarray(enabled)):
                        index = identifiers[lane]
                        if recorder.should_record(index):
                            birth = jax.tree.map(lambda value, lane=lane: value[lane], pool.state)
                            recorder.begin_episode(
                                f"B-{pool_key}-episode{pool.lane_episodes[lane]}",
                                birth,
                                config,
                                training_events=counters.real_events,
                                episode_index=index,
                                partial=int(birth.event) > 0,
                            )
                    stage_seconds["recording"] += perf_counter() - recording_start
                measured = make_real_collector(config, options)(
                    pool.state,
                    pool.carry,
                    actor,
                    predictor,
                    event_keys,
                    jnp.asarray(warmup),
                    enabled,
                )
                jax.block_until_ready(measured)
                validate_graph_state(measured.result.state)
                valid = np.asarray(measured.result.info.active) & np.asarray(enabled)
                for offset, lane in enumerate(np.flatnonzero(valid), start=1):
                    before_state, before_carry = jax.tree.map(
                        lambda value, index=lane: value[index], (pool.state, pool.carry)
                    )
                    entry = jax.tree.map(lambda value, index=lane: value[index], measured)
                    result, decision, obs = entry.result, entry.decision, entry.observation
                    extras = (
                        *_decision_extras(decision, obs, before_carry, before_state),
                        ("policy_valid", np.asarray(not warmup)),
                    )
                    after_extras = (
                        ("world_tile", np.asarray(jnp.argmax(entry.measurement.local_tile, -1))),
                        ("world_robots", np.asarray(entry.measurement.robot_features)),
                        ("world_global", np.asarray(entry.measurement.global_features)),
                        ("world_movement", np.asarray(result.info.distance)),
                        ("world_delivered", np.asarray(result.info.delivered)),
                        ("world_local_mask", np.asarray(entry.local_mask)),
                    )
                    replay.append(
                        pool.lane_episodes[lane],
                        _frame(before_state, extras),
                        _frame(result.state, after_extras),
                        np.asarray(entry.encoded),
                        agent_id=int(obs.agent_id),
                        reward=float(result.reward),
                        terminated=bool(result.terminated),
                        truncated=bool(result.truncated),
                        plan_labels=_named_fields("label", plan_labels(before_state)),
                    )
                    if sampler is not None:
                        sample = sampler.schedule(pool.sample_indices[lane])
                        source_events[sample.source] += 1
                        family_events[sample.family] += 1
                    if recorder is not None and recorder.should_record(identifiers[lane]):
                        recording_start = perf_counter()
                        identity = f"B-{pool_key}-episode{pool.lane_episodes[lane]}"
                        expected = RecordedEvent(
                            before_state.event,
                            obs.agent_id,
                            result.state.positions,
                            result.state.goals,
                            result.state.balances,
                            result.info.spent,
                            result.info.delivered,
                            result.info.coverage_after,
                            result.terminated,
                            result.truncated,
                        )
                        recorder.append_actions(
                            identity,
                            jax.tree.map(
                                lambda value: value[None],
                                StandardAction(
                                    entry.action.waypoint, entry.action.send, entry.action.codec
                                ),
                            ),
                            np.ones(1, bool),
                            jax.tree.map(lambda value: value[None], expected),
                            versions=ModelVersions(
                                int(actor.step),
                                int(predictor.step),
                                "warmup" if warmup else "train",
                                world_model=int(world.step),
                            ),
                        )
                        if bool(result.terminated | result.truncated):
                            recorder.end_episode(
                                identity,
                                "terminated" if bool(result.terminated) else "truncated",
                                counters.real_events + offset,
                            )
                        stage_seconds["recording"] += perf_counter() - recording_start
                count = int(valid.sum())
                pool = replace(
                    pool,
                    state=measured.result.state,
                    carry=measured.decision.carry,
                    events=pool.events + count,
                    lane_cursor=(pool.lane_cursor + count) % options.physical_env_count,
                )
                pools[pool_key] = pool
                state, carry, latent = jax.tree.map(
                    lambda value: value[0], (pool.state, pool.carry, pool.latent)
                )
                episode = pool.episode
                counters = counters._replace(real_events=counters.real_events + count)
                stage_seconds["collect"] += (
                    perf_counter() - collect_start - stage_seconds["recording"] + recording_before
                )
                update_pending = (
                    counters.real_events % options.update_every == 0
                    and replay.has_sequence(options.burn_in)
                )
                maybe_validate()
            if not update_pending:
                continue
            update_pending = False
            update_start = perf_counter()
            sequences = []
            while len(sequences) < options.logical_batch:
                sequence = replay.sample(options.sequence_length, options.burn_in)
                if np.any(sequence.loss_mask):
                    sequences.append(sequence)
            batches = tuple(_world_batch(sequence, radius, config) for sequence in sequences)
            key, model_key, plan_key, imagine_key = jax.random.split(key, 4)
            world, model_loss_value = world_update(world, batches, model_key)
            last_model_loss = float(model_loss_value)
            predictor_frames = tuple(
                entry.before_fields
                for sequence in sequences
                for entry in sequence.transitions
                if entry is not None
            )
            predictor, predictor_loss = predictor_update(
                predictor, plan_training_batch(predictor_frames), plan_key
            )
            counters = counters._replace(
                model_updates=counters.model_updates + 1,
                predictor_updates=counters.predictor_updates + 1,
            )
            if counters.real_events <= options.warmup_events:
                jax.block_until_ready((world.params, predictor.params))
                stage_seconds["update"] += perf_counter() - update_start
                log.write(
                    json.dumps(
                        {
                            **counters._asdict(),
                            "phase": "model_warmup",
                            "model_loss": last_model_loss,
                            "physical_env_count": options.physical_env_count,
                            "interval_real_events": counters.real_events - interval_initial_events,
                            "interval_seconds": perf_counter() - interval_start,
                            "stage_seconds": {
                                name: value - interval_stages[name]
                                for name, value in stage_seconds.items()
                            },
                        }
                    )
                    + "\n"
                )
                log.flush()
                interval_start = perf_counter()
                interval_initial_events = counters.real_events
                interval_stages = stage_seconds.copy()
                continue
            policy_batches = []
            if options.real_updates:
                while len(policy_batches) < options.logical_batch:
                    sequence = replay.sample(options.sequence_length, options.burn_in)
                    real_frames = tuple(
                        entry.before_fields for entry in sequence.transitions if entry is not None
                    )
                    if not any(
                        bool(read_field(frame, "policy_valid"))
                        for frame in real_frames[options.burn_in :]
                    ):
                        continue
                    batch = _world_batch(sequence, radius, config)
                    _, predictions = posterior(
                        world, batch, jax.random.fold_in(model_key, len(policy_batches))
                    )
                    policy_batches.append(
                        pad_policy_batch(
                            _real_policy_batch(real_frames, sequence, predictions),
                            options.sequence_length - options.burn_in,
                        )
                    )
            else:
                for origin, (sequence, batch) in enumerate(zip(sequences, batches, strict=True)):
                    _, predictions = posterior(world, batch, jax.random.fold_in(model_key, origin))
                    cutoff = options.burn_in
                    frame = replay.reconstruct(sequence.episode_id, sequence.start_index + cutoff)
                    cognition = _cognition(frame)
                    latent = (
                        jax.tree_util.tree_map(
                            lambda value, index=cutoff - 1: value[index], predictions.state
                        )
                        if cutoff
                        else initial_rssm(
                            deterministic=options.hidden,
                            categories=options.categories,
                            classes=options.classes,
                        )
                    )
                    seed_carry = PolicyCarry(
                        jnp.asarray(frame.field("all_actor_memory")),
                        PlanMemory(
                            *(
                                jnp.asarray(frame.field(f"all_plan_memory_{name}"))
                                for name in PlanMemory._fields
                            )
                        ),
                    )
                    initial = ImaginedState(cognition, latent, seed_carry.actor_memory)
                    _, _, trajectory = imagination(
                        initial,
                        seed_carry,
                        actor,
                        predictor,
                        world,
                        jax.random.fold_in(imagine_key, origin),
                        config,
                    )
                    policy_batches.append(trajectory)
            imagined_count = (
                0
                if options.real_updates
                else sum(int(batch.valid.sum()) for batch in policy_batches)
            )
            policy_batch = (
                policy_batches[0]
                if options.logical_batch == 1
                else jax.tree_util.tree_map(lambda *values: jnp.stack(values), *policy_batches)
            )
            actor, critic, statistics = actor_update(actor, critic, policy_batch)
            actor_delta += float(statistics.actor_parameter_delta)
            counters = counters._replace(
                imagined_events=counters.imagined_events + imagined_count,
                actor_updates=counters.actor_updates + 1,
            )
            jax.block_until_ready((actor.params, critic.params, world.params, predictor.params))
            stage_seconds["update"] += perf_counter() - update_start
            record = {
                **counters._asdict(),
                "model_loss": last_model_loss,
                "predictor_loss": float(predictor_loss),
                "actor_loss": float(statistics.actor_loss),
                "critic_loss": float(statistics.critic_loss),
                "actor_parameter_delta": float(statistics.actor_parameter_delta),
                "real_calls_during_actor_update": 0,
                "policy_trajectories": len(policy_batches),
                "physical_env_count": options.physical_env_count,
                "interval_real_events": counters.real_events - interval_initial_events,
                "interval_seconds": perf_counter() - interval_start,
                "stage_seconds": {
                    name: value - interval_stages[name] for name, value in stage_seconds.items()
                },
            }
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            interval_start = perf_counter()
            interval_initial_events = counters.real_events
            interval_stages = stage_seconds.copy()
            if counters.actor_updates % 10 == 0:
                persist()
    jax.block_until_ready(actor.params)
    elapsed = perf_counter() - start
    persist()
    if best_auc is not None:
        best_summary_path = root / "best" / "summary.json"
        best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
        best_summary["total_training_events"] = counters.real_events
        best_summary["real_events"] = counters.real_events
        best_summary_path.write_text(json.dumps(best_summary, indent=2) + "\n", encoding="utf-8")
    end_evaluation_start = perf_counter()
    completed = counters.real_events >= events
    evaluation = (
        _evaluate_policy(actor, predictor, config, options, seed + 10000, map_path)
        if sampler is None and completed
        else None
    )
    comparison = {
        "real_evaluation": evaluation,
        "model_open_loop": model_diagnostics(world, config, options, seed + 20000, map_path)
        if sampler is None and completed
        else [],
        "imagined_only_actor_parameter_delta": actor_delta if not options.real_updates else None,
        "real_environment_calls_during_actor_updates": 0,
        "update_source": "real_replay" if options.real_updates else "world_model",
    }
    stage_seconds["validation"] += perf_counter() - end_evaluation_start
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    if recorder is not None:
        recording_start = perf_counter()
        recorder.close(counters.real_events)
        stage_seconds["recording"] += perf_counter() - recording_start
    summary: TrainingSummary = {
        "route": "B",
        "total_training_events": counters.real_events,
        **counters._asdict(),
        "config_hash": config_hash,
        "physical_microbatch": 1,
        "physical_env_count": options.physical_env_count,
        "session_real_events": counters.real_events - session_initial_events,
        "session_seconds": perf_counter() - session_start,
        "session_events_per_second": 0.0,
        "stage_seconds": stage_seconds,
        "development": options.development,
        "model_loss": last_model_loss,
        "seconds": elapsed,
        "actor_parameter_delta": actor_delta,
        "evaluation": evaluation,
        "checkpoint": str(root / "checkpoint"),
        "formal_complete": False,
        "schedule_cursor": schedule_cursor,
        "group_events": {name: pool.events for name, pool in pools.items()},
        "best_validation_auc": best_auc,
        "last_validation_event": last_validation_event,
        "best_checkpoint": str(root / "best" / "checkpoint") if best_auc is not None else None,
        "source_event_counts": source_events,
        "family_event_counts": family_events,
        "source_episode_counts": source_episodes,
        "family_episode_counts": family_episodes,
        "replay": {
            name: {
                "size": pool.replay.size,
                "capacity": pool.replay.capacity,
                "evicted_events": pool.replay.evicted_events,
                "evicted_episodes": pool.replay.evicted_episodes,
            }
            for name, pool in pools.items()
        }
        if sampler
        else {
            "micro": {
                "size": replay.size,
                "capacity": replay.capacity,
                "evicted_events": replay.evicted_events,
                "evicted_episodes": replay.evicted_episodes,
            }
        },
        "evaluation_status": (
            "real_development_episode" if sampler is None else "requires_independent_V2_suite"
        )
        if completed
        else "stopped_at_update_boundary",
    }
    summary["session_events_per_second"] = (
        summary["session_real_events"] / summary["session_seconds"]
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def train_b_handler(args: argparse.Namespace) -> int:
    """CLI integration hook; method JSON selects fixed/predicted/real-update ablations."""
    options_path = getattr(args, "method_config", None)
    options = (
        BTrainingConfig(**json.loads(Path(options_path).read_text(encoding="utf-8")))
        if options_path
        else None
    )
    result = train_b(
        load_config(args.config),
        args.seed,
        args.events,
        args.output,
        options=options,
        map_path=getattr(args, "map", None),
        resume=getattr(args, "resume", None),
    )
    print(json.dumps(result, indent=2))  # noqa: T201
    return 0
