"""Exact environment replay of captured real actions, with no actor call or training RNG use."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from functools import cache
from time import perf_counter
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from ir2_benchmark.maps.manifest import obstacle_hash
from ir2_benchmark.visualization.recording import RecordedEvent, checked_event
from ir2_benchmark.visualization.render import (
    Delivery,
    FrameData,
    frame_events,
    render_frame,
    save_views,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jax import Array

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.types import EnvState
    from ir2_benchmark.visualization.recording import EpisodeCapture, RecordingConfig

ReplayKernel: TypeAlias = (
    "Callable[[EnvState, Array, Array, Array], tuple[EnvState, RecordedEvent, Array, Array]]"
)
FrameKernel: TypeAlias = "Callable[[EnvState], tuple[Array, ...]]"


@cache
def replay_step(config: HostConfig) -> ReplayKernel:
    """Cache replay using the recorded navigation mode's actual executed path."""
    import jax  # noqa: PLC0415

    from ir2_benchmark.env.observe import observe  # noqa: PLC0415
    from ir2_benchmark.env.step import step_agent  # noqa: PLC0415
    from ir2_benchmark.navigation.jax_path import navigate  # noqa: PLC0415
    from ir2_benchmark.navigation.one_hop import navigate_edge  # noqa: PLC0415
    from ir2_benchmark.types import make_action  # noqa: PLC0415

    def one(
        state: EnvState, waypoint: Array, send: Array, codec: Array
    ) -> tuple[EnvState, RecordedEvent, Array, Array]:
        local = observe(state, state.current_agent, config)
        route = (
            navigate_edge(local.position, waypoint, local.candidates, config.plan_horizon)
            if config.navigation_mode != "grid_path"
            else navigate(
                local.local_map,
                local.position,
                waypoint,
                config.max_distance,
                config.plan_horizon,
                local.tree,
            )
        )
        result = step_agent(
            state, make_action(waypoint, send, codec), config, key=state.key, observation=local
        )
        updated, info = result.state, result.info
        evidence = RecordedEvent(
            state.event,
            info.agent_id,
            updated.positions,
            updated.goals,
            updated.balances,
            info.spent,
            info.delivered,
            info.coverage_after,
            result.terminated,
            result.truncated,
        )
        return updated, evidence, route.path[: config.max_distance + 1], route.distance

    return jax.jit(one)


@cache
def frame_fields(config: HostConfig) -> FrameKernel:
    """Overlay paid inbox knowledge for images without computing extra navigation searches."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from ir2_benchmark.env.observe import LocalKnowledge, merge_local_messages  # noqa: PLC0415

    def fields(state: EnvState) -> tuple[Array, ...]:
        def local(agent: Array) -> tuple[Array, Array, Array]:
            initial = LocalKnowledge(
                state.local_maps[agent],
                state.observed_event[agent],
                state.peer_positions[agent],
                state.peer_goals[agent],
                state.peer_plans[agent],
                state.peer_plan_lengths[agent],
                state.peer_position_event[agent],
                state.peer_goal_event[agent],
                state.peer_plan_event[agent],
            )
            known = merge_local_messages(
                initial,
                state.inbox.buffers[agent],
                state.inbox.lengths[agent],
                state.inbox.present[agent],
                config,
                valid_shape=state.valid_shape,
            )
            return known.local_map, known.peer_positions, known.peer_position_event

        maps, peers, times = jax.vmap(local)(jnp.arange(config.n_agents))
        coverage = jnp.sum(state.covered & state.component) / jnp.sum(state.component)
        return (
            maps,
            peers,
            times,
            state.positions[: config.n_agents],
            state.balances[: config.n_agents],
            state.travel_distance[: config.n_agents],
            coverage,
            state.event,
            state.round,
            state.valid_shape,
        )

    return jax.jit(fields)


def snapshot_frame(state: EnvState, config: HostConfig) -> FrameData:
    """Transfer only selected frame beliefs/compact fields, never hidden truth for local images."""
    import jax  # noqa: PLC0415

    maps, peers, times, poses, balances, distances, coverage, event, round_id, shape = (
        jax.device_get(frame_fields(config)(state))
    )
    return FrameData(
        maps,
        peers,
        times,
        poses,
        balances,
        distances,
        float(coverage),
        int(event),
        int(round_id),
        tuple(int(value) for value in shape),
    )


def export_capture(  # noqa: PLR0913, PLR0917
    capture: EpisodeCapture,
    output: Path,
    options: RecordingConfig,
    reason: str,
    training_events: int,
    training_seed: int | None,
    route: str | None,
) -> Path:
    """Verify every real event, then emit merged/local GIFs, last PNGs and provenance metadata."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    started = perf_counter()
    state = jax.tree.map(jnp.asarray, capture.initial)
    start = int(capture.initial.event)
    stop = start + sum(len(chunk.waypoint) for chunk in capture.chunks)
    selected = frame_events(
        start, stop, capture.config.n_agents, options.frame_stride_rounds, options.max_frames
    )
    last = capture.chunks[-1].expected if capture.chunks else None
    terminal = (
        bool(last.terminated[-1] | last.truncated[-1])
        if last is not None
        else bool(state.terminated | state.truncated)
    )
    partial = capture.partial or not terminal
    title = f"{route or 'Recorded'} real training episode {capture.episode_index}"
    paths = [
        [tuple(int(value) for value in point)]
        for point in np.asarray(capture.initial.positions)[: capture.config.n_agents]
    ]
    deliveries, frames, versions = [], {}, []

    def add_frame() -> None:
        stamp = (
            versions[-1]["actor"]
            if versions
            else (capture.chunks[0].versions.actor if capture.chunks else None)
        )
        phase = (
            versions[-1]["phase"]
            if versions
            else (capture.chunks[0].versions.phase if capture.chunks else "no-actions")
        )
        images = render_frame(
            snapshot_frame(state, capture.config),
            [np.asarray(path) for path in paths],
            deliveries,
            f"{title} {phase} actor-step={stamp}",
            partial=partial,
            max_rounds=capture.config.max_rounds,
            max_side=options.max_side,
        )
        for label, image in images.items():
            frames.setdefault(label, []).append(image)
        deliveries.clear()

    add_frame()
    step = replay_step(capture.config)
    for chunk in capture.chunks:
        versions.append(
            {
                "start_event": int(chunk.expected.event[0]),
                "end_event": int(chunk.expected.event[-1]) + 1,
                **asdict(chunk.versions),
            }
        )
        for index in range(len(chunk.waypoint)):
            state, evidence, path, distance = step(
                state, chunk.waypoint[index], chunk.send[index], chunk.codec[index]
            )
            actual, vertices, moved = jax.device_get((evidence, path, distance))
            checked_event(actual, chunk.expected, index)
            agent = int(actual.agent_id)
            paths[agent].extend(
                tuple(int(value) for value in point) for point in vertices[1 : int(moved) + 1]
            )
            deliveries.extend(
                Delivery(
                    agent,
                    int(receiver),
                    tuple(actual.positions[agent]),
                    tuple(actual.positions[receiver]),
                )
                for receiver in np.flatnonzero(actual.delivered[: capture.config.n_agents])
            )
            if int(actual.event) + 1 in selected:
                add_frame()
    output.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(capture.episode_id.encode()).hexdigest()[:12]
    stem = f"episode_{capture.episode_index:06d}_{token}_from_{start:06d}"
    artifacts = save_views(frames, output, stem, options.duration_ms)
    height, width = np.asarray(capture.initial.valid_shape).tolist()
    metadata = {
        "source": "actual_real_training_actions",
        "replay_verified": True,
        "episode_id": capture.episode_id,
        "episode_index": capture.episode_index,
        "route": route,
        "training_seed": training_seed,
        "partial": partial,
        "reason": reason,
        "initial_event": start,
        "end_event": stop,
        "valid_real_events": stop - start,
        "training_events_start": capture.training_start,
        "training_events_end": training_events,
        "training_events_added_by_replay": 0,
        "fault_root_key": np.asarray(capture.initial.key).tolist(),
        "environment_episode": int(capture.initial.episode),
        "map_hash": obstacle_hash(np.asarray(capture.initial.truth)[:height, :width]),
        "valid_shape": [height, width],
        "config": asdict(capture.config),
        "recording": asdict(options),
        "frame_events": selected.tolist(),
        "model_versions": versions,
        "views": artifacts,
        "discrete_check": "exact",
        "float_check": {"rtol": 1e-6, "atol": 1e-7},
        "recording_seconds": perf_counter() - started,
    }
    path = output / f"{stem}.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path
