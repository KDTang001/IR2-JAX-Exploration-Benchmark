"""Adapt actual compact A training events to the shared deterministic GIF recorder."""

from __future__ import annotations

import json
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import numpy as np

from ir2_benchmark.visualization.recording import RecordedEvent, TrainingRecorder

if TYPE_CHECKING:
    from pathlib import Path

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.training.collect_a import Collection, SamplerState
    from ir2_benchmark.visualization.recording import ModelVersions, RecordingConfig


def recorder_for(output: Path, options: RecordingConfig | None) -> TrainingRecorder | None:
    """Keep recording options outside learning identity; reuse explicit CLI seed metadata."""
    if options is None:
        return None
    experiment = output / "experiment.json"
    seed = (
        json.loads(experiment.read_text(encoding="utf-8")).get("training_seed")
        if experiment.exists()
        else None
    )
    return TrainingRecorder(output / "gifs", options, training_seed=seed, route="A")


def record_collection(  # noqa: PLR0913
    recorder: TrainingRecorder | None,
    initial: SamplerState,
    result: Collection,
    config: HostConfig,
    *,
    episode_indices: tuple[int, ...],
    training_events: int,
    versions: ModelVersions,
) -> float:
    """Replay selected actual actions without sampling policy/RNG or changing training state."""
    if recorder is None:
        return 0.0
    started = perf_counter()
    trace = result.trace
    final_training_events = training_events + int(np.sum(np.asarray(trace.valid)))
    for lane, episode_index in enumerate(episode_indices):
        if not recorder.should_record(episode_index) or not bool(
            np.any(np.asarray(trace.valid[lane]))
        ):
            continue
        snapshot = jax.tree.map(lambda value, lane=lane: value[lane], initial.environment)
        identifier = (
            f"A-bucket{snapshot.truth.shape[0]}-N{config.n_agents}"
            f"-lane{lane}-episode{episode_index}"
        )
        recorder.begin_episode(
            identifier,
            snapshot,
            config,
            training_events=training_events,
            episode_index=episode_index,
            partial=int(snapshot.event) > 0,
        )
        actions = jax.tree.map(lambda value, lane=lane: value[lane], result.executed_actions)
        expected = RecordedEvent(*(getattr(trace, field)[lane] for field in RecordedEvent._fields))
        recorder.append_actions(identifier, actions, trace.valid[lane], expected, versions=versions)
        final = result.sampler.environment
        if bool(final.terminated[lane]) or bool(final.truncated[lane]):
            reason = "terminated" if bool(final.terminated[lane]) else "truncated"
            recorder.end_episode(identifier, reason, final_training_events)
    return perf_counter() - started


def close_recorder(recorder: TrainingRecorder | None, training_events: int) -> float:
    """Flush truthful partial clips when a run pauses or reaches its real-event quota."""
    if recorder is None:
        return 0.0
    started = perf_counter()
    recorder.close(training_events)
    return perf_counter() - started
