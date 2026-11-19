"""Frozen validation artifacts and CUDA-compatible birth/coverage data contracts."""

# ruff: noqa: PLR2004

import json
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import jax
import numpy as np
import pytest

from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import HostConfig, load_config
from ir2_benchmark.env.reset import from_prepared_map, reset
from ir2_benchmark.evaluation.validation import validate_manifest
from ir2_benchmark.maps.prepare import MapData, prepare_map

ROOT = Path(__file__).resolve().parents[3]


def test_published_validation_is_current_graph_config_and_validation_only() -> None:
    # Given: the actual published registry and fixed model-selection artifacts.
    frozen = ROOT / "artifacts/maps/validation-scenarios.jsonl"
    config = replace(load_config(ROOT / "configs/receiver_ir2_graph.json"), initial_budget=8192)
    selection = json.loads(frozen.with_suffix(".selection.json").read_text())
    assert selection["manifest_hash"] == sha256(frozen.read_bytes()).hexdigest()
    candidates = {}
    for source in manifest_sources(ROOT / "artifacts/maps/sources.json"):
        for line in source.manifest.read_text().splitlines():
            row = json.loads(line)
            if row["split"] == "validation" and row.get("exclusion_reason") is None:
                family = row.get("family", row.get("template"))
                candidates[source.source, family, row["occupancy_hash"]] = row["scenario_hash"]
    # When: loading the exact committed workload through the production validator.
    scenes = validate_manifest(frozen)
    # Then: every scene belongs to validation and has the current full graph configuration.
    assert len(scenes) == 25
    assert Counter((s.source, s.family) for s in scenes) == {
        ("ir2", "easy"): 5,
        ("ir2", "difficult"): 5,
        ("procedural", "rooms"): 5,
        ("procedural", "corridor"): 5,
        ("procedural", "hybrid"): 5,
    }
    assert selection["base_config"] == json.loads(json.dumps(asdict(config)))
    for scene in scenes:
        assert candidates[scene.source, scene.family, scene.map_hash] == scene.map_scenario_hash
        assert scene.config == replace(config, n_agents=scene.n_agents, agent_bucket=6)
        assert not Path(scene.map_path).is_absolute()


@pytest.mark.parametrize("bucket", [512, 768, 1024])
@pytest.mark.parametrize("n_agents", [2, 4, 6])
def test_birth_and_physical_denominator_preserve_component_and_padding(
    bucket: int, n_agents: int
) -> None:
    # Given: a hand-countable anchor component and a separate free room in a native bucket.
    grid = np.ones((18, 25), np.uint8)
    grid[2:7, 2:11] = 0
    grid[10:16, 16:23] = 0
    prepared = prepare_map(MapData(grid, (2, 2)), buckets=(bucket,))
    data = from_prepared_map(prepared)
    config = HostConfig(
        n_agents=n_agents, agent_bucket=6, navigation_mode="grid_path", sensor_radius=1
    )
    key = jax.random.PRNGKey(13)
    # When: reset is actually jitted and vmapped over two independent episode seeds.
    birth = jax.jit(jax.vmap(lambda k: reset(data, config, k)))
    state = birth(jax.random.split(key, 2))
    repeat = birth(jax.random.split(key, 2))
    jax.block_until_ready(state)
    # Then: spawns are distinct/in-component, unreachable cells and padding never score.
    np.testing.assert_array_equal(state.positions, repeat.positions)
    assert int(prepared.component.sum()) == 45
    for positions in np.asarray(state.positions[:, :n_agents]):
        assert len(set(map(tuple, positions))) == n_agents
        assert np.all(prepared.component[positions[:, 1], positions[:, 0]])
    np.testing.assert_array_equal(np.asarray(state.valid_shape), [[18, 25], [18, 25]])
    assert not np.asarray(state.covered)[:, 10:16, 16:23].any()
    assert not np.asarray(state.covered)[:, 18:].any()
    assert not np.asarray(state.covered)[:, :, 25:].any()
    assert np.all(np.asarray(state.truth)[:, 18:] == 1)
    assert not np.asarray(state.physical_seen)[:, n_agents:].any()
    np.testing.assert_array_equal(np.asarray(state.component).sum(axis=(1, 2)), [45, 45])


def test_completion_scores_anchor_component_instead_of_disconnected_free_area() -> None:
    # Given: 45 reachable cells plus 42 free cells behind walls in another component.
    grid = np.ones((18, 25), np.uint8)
    grid[2:7, 2:11] = 0
    grid[10:16, 16:23] = 0
    data = from_prepared_map(prepare_map(MapData(grid, (2, 2)), buckets=(32,)))
    config = HostConfig(n_agents=2, navigation_mode="grid_path", sensor_radius=8)
    spawns = np.array([[2, 2], [10, 6]], np.int32)
    # When: the initial scans see the whole reachable room, but not the disconnected room.
    state = jax.jit(lambda: reset(data, config, jax.random.PRNGKey(9), spawns))()
    # Then: 45/45 completes the task; neither 45/87 nor padded cells affect the denominator.
    assert int(state.covered.sum()) == 45
    assert int(state.component.sum()) == 45
    assert int((state.truth == 0).sum()) == 87
    assert bool(state.terminated)
