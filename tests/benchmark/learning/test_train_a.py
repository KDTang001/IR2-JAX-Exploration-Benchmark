"""Actual small-map interaction, optimization, and exact boundary restoration."""

# Exact event quotas and gradient comparisons are independent acceptance expectations.
# ruff: noqa: PLR2004

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from PIL import Image

from ir2_benchmark.commands.evaluate import checkpoint_method
from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.scenarios import EvaluationMap, make_scenario, write_scenario_manifest
from ir2_benchmark.evaluation.validation import ValidationEvaluator, build_validation_callback
from ir2_benchmark.maps.manifest import canonical_obstacle_hash, obstacle_hash, scenario_hash
from ir2_benchmark.maps.sampler import ManifestSource, TrainingMapSampler
from ir2_benchmark.training.checkpoint import CheckpointError, load_checkpoint
from ir2_benchmark.training.driver_a import train_a_formal
from ir2_benchmark.training.evaluate_a import evaluate_a
from ir2_benchmark.training.train_a import (
    TrainAConfig,
    configuration_hash,
    initialize_training,
    train_a,
)
from ir2_benchmark.types import MapInput


def _scenario() -> tuple[MapInput, HostConfig, TrainAConfig]:
    occupancy = jnp.ones((12, 12), jnp.uint8).at[1:11, 1:11].set(0)
    data = MapInput(occupancy, occupancy == 0, jnp.asarray([12, 12]), jnp.asarray([5, 5]))
    config = HostConfig(
        n_agents=2,
        sensor_radius=2,
        max_distance=1,
        max_rounds=12,
        initial_budget=90,
        max_candidates=4,
        spawn_radius=2,
    )
    settings = TrainAConfig(
        events=32,
        warmup_events=8,
        rollout_length=4,
        ppo_epochs=1,
        batch_size=2,
        hidden=16,
        suite="micro",
    )
    return data, config, settings


def test_training_when_real_interactions_update_actor_and_write_checkpoint(tmp_path: Path) -> None:
    # Given: a real connected map, true environment dynamics and a small interaction quota.
    data, config, settings = _scenario()
    settings = replace(
        settings,
        communication_mode="fixed",
        fixed_plan=True,
        ppo_minibatch_sequences=2,
        ppo_target_kl=0.02,
    )
    # When: run heuristic warmup then actual recurrent PPO interaction and updates.
    summary = train_a(data, config, settings, jax.random.PRNGKey(11), tmp_path)
    # Then: actual events include warmup and trained actor parameters change.
    assert summary.real_events == 32
    assert summary.warmup_events == 8
    assert summary.actor_parameter_change > 0
    assert Path(summary.checkpoint).is_file()
    assert (tmp_path / "updates.jsonl").is_file()
    rows = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    learned = [row["ppo"] for row in rows if not row["warmup"]]
    assert all(row["optimizer_steps"] > 0 for row in learned)
    assert all(np.isfinite(row["approx_kl"]) and row["mean_actor_grad_norm"] > 0 for row in learned)


def test_training_when_resume_matches_uninterrupted_next_batches(tmp_path: Path) -> None:
    # Given: identical model, environment, optimization and random root configurations.
    data, config, settings = _scenario()
    settings = replace(settings, ppo_minibatch_sequences=2, ppo_target_kl=0.02)
    key = jax.random.PRNGKey(7)
    uninterrupted = train_a(data, config, settings, key, tmp_path / "full")
    boundary = train_a(data, config, settings, key, tmp_path / "split", stop_after_updates=2)
    # When: restore every optimizer, PRNG, environment and pair-memory field at the boundary.
    resumed = train_a(data, config, settings, key, tmp_path / "split", resume=boundary.checkpoint)
    template = initialize_training(data, config, settings, key)
    identity = configuration_hash(config, settings)
    full = serialization.from_bytes(template, load_checkpoint(uninterrupted.checkpoint, identity))
    split = serialization.from_bytes(template, load_checkpoint(resumed.checkpoint, identity))
    assert int(full.replay.size) == 32
    assert bool(jnp.any(full.replay.data.target.available[:32]))
    # Then: the next training trajectory and parameter updates agree with uninterrupted execution.
    for expected, actual in zip(jax.tree.leaves(full), jax.tree.leaves(split), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_checkpoint_when_configuration_identity_changes(tmp_path: Path) -> None:
    # Given: a real checkpoint is tied to its original dynamics and optimizer settings.
    data, config, settings = _scenario()
    summary = train_a(data, config, settings, jax.random.PRNGKey(2), tmp_path, stop_after_updates=1)
    # When / Then: incompatible recovery is a concrete error, never a replacement policy.
    with pytest.raises(CheckpointError):
        load_checkpoint(summary.checkpoint, "different-environment-and-budget")


def _validation_fixture(data: MapInput, config: HostConfig, directory: Path) -> ValidationEvaluator:
    """Freeze one genuine small-map validation workload for the shared runner."""
    directory.mkdir(parents=True, exist_ok=True)
    pixels = np.where(np.asarray(data.occupancy) == 0, 255, 0).astype(np.uint8)
    x, y = map(int, data.anchor)
    pixels[y, x] = 208
    image = directory / "validation-map.png"
    Image.fromarray(pixels).save(image)
    validation_map = EvaluationMap(
        str(image),
        obstacle_hash(np.asarray(data.occupancy)),
        "validation",
        "procedural",
        "rooms",
        12,
        scenario_hash(np.asarray(data.occupancy), (x, y)),
    )
    manifest = directory / "validation.jsonl"
    write_scenario_manifest(manifest, (make_scenario(validation_map, config, 19),))
    return build_validation_callback(manifest, directory / "results", training_seed=3)


@pytest.mark.parametrize("candidate_attention", [False, True])
def test_evaluation_when_checkpoint_runs_on_fresh_real_episode(
    tmp_path: Path, *, candidate_attention: bool
) -> None:
    # Given: trained parameters from an actual small-map training run.
    data, config, settings = _scenario()
    settings = replace(settings, candidate_attention=candidate_attention)
    trained = train_a(data, config, settings, jax.random.PRNGKey(3), tmp_path / "train")
    if not candidate_attention:
        metadata_path = tmp_path / "train" / "configuration.json"
        metadata = json.loads(metadata_path.read_text())
        del metadata["training"]["candidate_attention"]
        metadata_path.write_text(json.dumps(metadata))
    # When: an independently seeded real evaluation episode uses the restored actor.
    output = evaluate_a(
        trained.checkpoint, data, config, settings, jax.random.PRNGKey(19), tmp_path / "evaluation"
    )
    # Then: evaluation produces an actual per-event trajectory and final local maps.
    assert output.is_file()
    assert (output.parent / "events.jsonl").stat().st_size > 0
    assert (output.parent / "trajectory.npz").is_file()
    assert (output.parent / "maps.png").is_file()
    validate = _validation_fixture(data, config, tmp_path / "validation")
    score = validate(Path(trained.checkpoint), trained.real_events)
    with np.load(next((score.output / "events").glob("*.npz"))) as actual:
        coverage = actual["coverage"][actual["active"]]
    rounds = coverage[config.n_agents - 1 :: config.n_agents]
    if len(coverage) % config.n_agents:
        rounds = np.r_[rounds, coverage[-1]]
    expected = (rounds.sum() + (config.max_rounds - len(rounds)) * coverage[-1]) / config.max_rounds
    assert score.mean_auc == pytest.approx(expected)
    assert score.manifest_hash == hashlib.sha256(validate.manifest.read_bytes()).hexdigest()
    assert score.evaluated == 1
    assert score.real_events == 32


def _formal_sources(directory: Path) -> tuple[ManifestSource, ...]:
    """Real image fixtures include every required map-source/template/native-bucket stratum."""
    directory.mkdir()
    manifests = []
    for source in ("ir2", "procedural"):
        records = []
        strata = (
            [("easy", 768), ("difficult", 512)]
            if source == "ir2"
            else [
                (family, side)
                for family in ("rooms", "corridor", "hybrid")
                for side in (512, 768, 1024)
            ]
        )
        for family, side in strata:
            grid = np.ones((side, side), np.uint8)
            grid[1:-1, 1:-1] = 0
            pixels = np.where(grid == 0, 255, 0).astype(np.uint8)
            pixels[side // 2, side // 2] = 208
            name = f"{source}-{family}-{side}.png"
            Image.fromarray(pixels).save(directory / name)
            records.append(
                {
                    "path": name,
                    "split": "train",
                    "family": family,
                    "template": family,
                    "bucket": side,
                    "occupancy_hash": obstacle_hash(grid),
                    "canonical_hash": canonical_obstacle_hash(grid),
                    "scenario_hash": scenario_hash(grid, (side // 2, side // 2)),
                }
            )
        manifest = directory / f"{source}.jsonl"
        manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        manifests.append(ManifestSource(directory, manifest, source))
    return tuple(manifests)


def test_formal_driver_when_native_buckets_share_models_and_resume_all_pools(
    tmp_path: Path,
) -> None:
    # Given: actual 512/768/1024 image maps and a valid balanced hierarchical sampler.
    sources = _formal_sources(tmp_path / "maps")
    seed = next(
        value
        for value in range(100)
        if TrainingMapSampler(sources, value).schedule(0).bucket
        != TrainingMapSampler(sources, value).schedule(1).bucket
    )
    maps = TrainingMapSampler(sources, seed)
    config = HostConfig(
        sensor_radius=2,
        max_distance=1,
        max_rounds=1,
        max_candidates=2,
        initial_budget=90,
        spawn_radius=1,
    )
    settings = TrainAConfig(
        events=8,
        warmup_events=0,
        rollout_length=2,
        ppo_epochs=1,
        batch_size=2,
        hidden=8,
        neighbor_estimator="zoh",
        suite="formal",
        physical_env_count=1,
    )
    key = jax.random.PRNGKey(17)
    validation_data, validation_config, _ = _scenario()
    fixed_validation = _validation_fixture(
        validation_data, validation_config, tmp_path / "fixed-validation"
    )
    full_validation = build_validation_callback(
        fixed_validation.manifest, tmp_path / "full-validation", training_seed=17
    )
    split_validation = build_validation_callback(
        fixed_validation.manifest, tmp_path / "split-validation", training_seed=17
    )
    full = train_a_formal(
        maps,
        config,
        settings,
        key,
        tmp_path / "formal-full",
        validation=full_validation,
        validation_interval=4,
    )
    # When: pause after one logical real batch, then restore every native-bucket episode pool.
    boundary = train_a_formal(
        maps,
        config,
        settings,
        key,
        tmp_path / "formal-split",
        stop_after_updates=1,
        validation=split_validation,
        validation_interval=4,
    )
    resumed = train_a_formal(
        maps,
        config,
        settings,
        key,
        tmp_path / "formal-split",
        resume=boundary,
        validation=split_validation,
        validation_interval=4,
    )
    identity = json.loads((full.parent / "configuration.json").read_text())["configuration_hash"]
    expected = serialization.msgpack_restore(load_checkpoint(full, identity))["state"]
    actual = serialization.msgpack_restore(load_checkpoint(resumed, identity))["state"]
    # Then: actual events and hierarchy identity survive restart, with distinct native pools.
    summary = json.loads((full.parent / "summary.json").read_text())
    assert summary["real_events"] == 8
    assert summary["pools"] >= 2
    assert summary["last_validation_event"] == 8
    assert (full.parent / "best" / "checkpoint").is_file()
    selected = json.loads((full.parent / "best" / "selection.json").read_text(encoding="utf-8"))
    method, _ = checkpoint_method(full.parent / "best" / "checkpoint")
    best_summary = json.loads((full.parent / "best" / "summary.json").read_text(encoding="utf-8"))
    assert method.real_training_events == 8
    assert best_summary["total_training_events"] == 8
    assert best_summary["selected_at_events"] == selected["real_events"]
    assert method.selected_at_events == selected["real_events"]
    assert method.training_seed == 17
    assert sum(summary["source_event_counts"].values()) == 8
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
