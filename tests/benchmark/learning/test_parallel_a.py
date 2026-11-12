"""Real physical vmap lanes, exact quotas, schedule allocation and checkpoint continuation."""

# Tiny quotas and explicit geometric expectations are independent acceptance oracles.
# ruff: noqa: PLR2004

import json
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from PIL import Image

from ir2_benchmark.agents.actor import Actor
from ir2_benchmark.agents.critic import Critic
from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.evaluation.scenarios import EvaluationMap, make_scenario, write_scenario_manifest
from ir2_benchmark.evaluation.validation import build_validation_callback
from ir2_benchmark.maps.manifest import canonical_obstacle_hash, obstacle_hash, scenario_hash
from ir2_benchmark.maps.sampler import ManifestSource, TrainingMapSampler
from ir2_benchmark.models.plan_predictor import PlanPredictor
from ir2_benchmark.training.checkpoint import load_checkpoint
from ir2_benchmark.training.collect_a import Parameters, make_collector
from ir2_benchmark.training.driver_a import train_a_formal
from ir2_benchmark.training.state_a import TrainAConfig, TrainingError, initialize_training
from ir2_benchmark.training.storage import pack_collections
from ir2_benchmark.types import MapInput, make_action

Tree = TypeVar("Tree")


def _assert_same(actual: Tree, expected: Tree) -> None:
    """Keep discrete/RNG evidence exact and retain the established floating-point tolerances."""
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if np.issubdtype(np.asarray(left).dtype, np.inexact):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(("quotas", "max_rounds"), [((2, 0), 3), ((3, 2), 3), ((3, 2), 1)])
def test_collection_when_independent_lane_limits_match_single_lane_and_actions_replay(
    quotas: tuple[int, int], max_rounds: int
) -> None:
    # Given: real lanes with uneven quotas, a zero-quota case and a finite task horizon.
    grid = jnp.ones((7, 7), jnp.uint8).at[1:6, 1:6].set(0)
    data = MapInput(grid, grid == 0, jnp.asarray([7, 7]), jnp.asarray([3, 3]))
    config = HostConfig(
        sensor_radius=1,
        max_distance=1,
        plan_horizon=2,
        max_candidates=3,
        max_rounds=max_rounds,
        spawn_radius=1,
    )
    settings = TrainAConfig(
        events=8,
        warmup_events=0,
        batch_size=2,
        physical_env_count=2,
        hidden=8,
        rollout_length=8,
        predictor_replay_capacity=16,
        suite="micro",
    )
    state = initialize_training(data, config, settings, jax.random.PRNGKey(11))
    parameters = Parameters(state.actor.params, state.critic.params, state.predictor.params)
    collector = make_collector(
        config, Actor(8), Critic(8), PlanPredictor(2, 8), length=8, warmup=True, fixed_plan=True
    )
    # When: the real vmapped collector records its actual heuristic-overridden actions.
    result = collector(state.sampler, parameters, jnp.asarray(quotas), jnp.int32(0))
    # Then: padding advances no RNG/state, and every lane equals its singleton execution.
    np.testing.assert_array_equal(
        np.sum(result.rollout.valid, axis=1), np.minimum(quotas, config.n_agents * max_rounds)
    )
    for lane in range(2):
        single = jax.tree.map(lambda value, lane=lane: value[lane : lane + 1], state.sampler)
        expected = collector(single, parameters, jnp.asarray([quotas[lane]]), jnp.int32(0))
        actual = jax.tree.map(lambda value, lane=lane: value[lane : lane + 1], result)
        _assert_same(actual, expected)
    if quotas[1] == 0:
        for before, after in zip(
            jax.tree.leaves(state.sampler), jax.tree.leaves(result.sampler), strict=True
        ):
            np.testing.assert_array_equal(before[1], after[1])
    short = make_collector(
        config, Actor(8), Critic(8), PlanPredictor(2, 8), length=3, warmup=True, fixed_plan=True
    )(
        state.sampler, parameters, jnp.asarray(quotas), jnp.int32(0)
    )
    count = int(jnp.sum(result.rollout.valid))
    assert bool(jnp.all(result.executed_actions.send[result.rollout.valid]))
    np.testing.assert_array_equal(result.executed_actions.codec[result.rollout.valid], 3)
    sequences, _, _ = pack_collections([result], count, sequence_length=2)
    assert int(sequences.valid.sum()) == count
    for row in range(sequences.valid.shape[0]):
        agents = np.asarray(sequences.actor_ids[row])[np.asarray(sequences.valid[row])]
        assert len(np.unique(agents)) == 1
    _assert_same(
        (result.sampler, result.bootstrap, pack_collections([result], count)),
        (short.sampler, short.bootstrap, pack_collections([short], count)),
    )
    for full, prefix in zip(
        jax.tree.leaves((result.trace, result.executed_actions, result.plan_diagnostics)),
        jax.tree.leaves((short.trace, short.executed_actions, short.plan_diagnostics)),
        strict=True,
    ):
        _assert_same(full[np.asarray(result.trace.valid)], prefix[np.asarray(short.trace.valid)])
    replay = jax.tree.map(lambda value: value[0], state.sampler.environment)
    for event in range(int(jnp.sum(result.trace.valid[0]))):
        observation = observe(replay, replay.current_agent, config)
        expected_waypoint = nearest_frontier_action(observation, config).waypoint
        saved = jax.tree.map(lambda value, event=event: value[0, event], result.executed_actions)
        np.testing.assert_array_equal(saved.waypoint, expected_waypoint)
        replay = step_agent(
            replay, make_action(saved.waypoint, saved.send, saved.codec), config, key=replay.key
        ).state
    _assert_same(replay, jax.tree.map(lambda value: value[0], result.sampler.environment))


def _sources(directory: Path) -> tuple[ManifestSource, ...]:
    """Write actual native-bucket PNGs with small connected rooms for real lane tests."""
    directory.mkdir()
    sources = []
    for source in ("ir2", "procedural"):
        strata = (
            (("easy", 768), ("difficult", 512))
            if source == "ir2"
            else tuple(
                (family, side)
                for family in ("rooms", "corridor", "hybrid")
                for side in (512, 768, 1024)
            )
        )
        rows = []
        for family, side in strata:
            grid = np.ones((side, side), np.uint8)
            center = side // 2
            grid[center - 4 : center + 5, center - 4 : center + 5] = 0
            pixels = np.where(grid == 0, 255, 0).astype(np.uint8)
            pixels[center, center] = 208
            name = f"{source}-{family}-{side}.png"
            Image.fromarray(pixels).save(directory / name)
            rows.append(
                {
                    "path": name,
                    "split": "train",
                    "family": family,
                    "template": family,
                    "bucket": side,
                    "occupancy_hash": obstacle_hash(grid),
                    "canonical_hash": canonical_obstacle_hash(grid),
                    "scenario_hash": scenario_hash(grid, (center, center)),
                }
            )
        manifest = directory / f"{source}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        sources.append(ManifestSource(directory, manifest, source))
    return tuple(sources)


def test_formal_when_physical_lanes_resume_preserves_schedule_and_exact_event_boundaries(
    tmp_path: Path,
) -> None:
    # Given: real native maps, independent physical lanes and a fixed real validation task.
    maps = TrainingMapSampler(_sources(tmp_path / "maps"), seed=11)
    config = HostConfig(
        sensor_radius=1, max_distance=1, max_rounds=1, max_candidates=2, spawn_radius=1
    )
    settings = TrainAConfig(
        events=9,
        warmup_events=3,
        rollout_length=2,
        batch_size=3,
        physical_env_count=2,
        ppo_epochs=1,
        hidden=8,
        neighbor_estimator="zoh",
        suite="formal",
    )
    pixels = np.zeros((7, 7), np.uint8)
    pixels[1:6, 1:6] = 255
    pixels[3, 3] = 208
    image = tmp_path / "validation.png"
    Image.fromarray(pixels).save(image)
    occupancy = np.where(pixels > 0, 0, 1).astype(np.uint8)
    declared = EvaluationMap(
        str(image),
        obstacle_hash(occupancy),
        "validation",
        "procedural",
        "rooms",
        7,
        scenario_hash(occupancy, (3, 3)),
    )
    manifest = tmp_path / "validation.jsonl"
    write_scenario_manifest(manifest, (make_scenario(declared, config, 19),))
    full_validation = build_validation_callback(manifest, tmp_path / "full-val", training_seed=11)
    split_validation = build_validation_callback(manifest, tmp_path / "split-val", training_seed=11)
    key = jax.random.PRNGKey(11)
    full = train_a_formal(
        maps,
        config,
        settings,
        key,
        tmp_path / "full",
        validation=full_validation,
        validation_interval=4,
    )
    # When: pause after a warmup-boundary batch and restore all live and pending lanes.
    boundary = train_a_formal(
        maps,
        config,
        settings,
        key,
        tmp_path / "split",
        stop_after_updates=1,
        validation=split_validation,
        validation_interval=4,
    )
    resumed = train_a_formal(
        maps,
        config,
        settings,
        key,
        tmp_path / "split",
        resume=boundary,
        validation=split_validation,
        validation_interval=4,
    )
    # Then: actual source counts, schedule order, boundary quotas and every checkpoint leaf agree.
    configuration = json.loads((full.parent / "configuration.json").read_text(encoding="utf-8"))
    identity = configuration["configuration_hash"]
    expected = serialization.msgpack_restore(load_checkpoint(full, identity))
    actual = serialization.msgpack_restore(load_checkpoint(resumed, identity))
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    episodes = [
        json.loads(row) for row in (full.parent / "map-episodes.jsonl").read_text().splitlines()
    ]
    assert [row["index"] for row in episodes] == list(range(len(episodes)))
    summary = json.loads((full.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["real_events"] == 9
    assert summary["physical_env_count"] == 2
    assert sum(summary["source_event_counts"].values()) == 9
    assert sum(summary["source_episode_counts"].values()) == len(episodes)
    resumed_summary = json.loads((resumed.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["session_real_events"] == 9
    assert resumed_summary["session_real_events"] == 6
    assert resumed_summary["session_events_per_second"] == pytest.approx(
        6 / resumed_summary["session_seconds"]
    )
    assert resumed_summary["stage_seconds"]["collect"] > 0
    updates = [json.loads(row) for row in (full.parent / "updates.jsonl").read_text().splitlines()]
    assert [row["real_events"] for row in updates] == [3, 4, 8, 9]
    assert [row["interval_real_events"] for row in updates] == [3, 1, 4, 1]
    assert all(row["sample_slots"] - row["padding_slots"] == row["valid_events"] for row in updates)
    for pool in expected["state"]["pools"].values():
        assert np.shape(pool["sampler"]["environment"]["event"]) == (2,)


def test_settings_when_physical_environment_count_is_nonpositive_rejects() -> None:
    # Given: a requested physical group cannot contain any environment.
    # When / Then: configuration parsing rejects it before allocating a map or model.
    with pytest.raises(TrainingError, match="positive training dimensions"):
        replace(TrainAConfig(), physical_env_count=0)
