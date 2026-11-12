"""Actual A training GIF capture preserves learning and truthfully labels resumed prefixes."""

# Literal event counts and optimizer versions are independent acceptance expectations.
# ruff: noqa: PLR2004

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from PIL import Image

from ir2_benchmark.config import HostConfig
from ir2_benchmark.training.checkpoint import load_checkpoint
from ir2_benchmark.training.state_a import TrainAConfig, configuration_hash
from ir2_benchmark.training.train_a import train_a
from ir2_benchmark.types import MapInput
from ir2_benchmark.visualization.recording import RecordingConfig


def test_training_when_gifs_capture_resumed_real_prefix_does_not_change_learning(
    tmp_path: Path,
) -> None:
    # Given: a real baseline and a captured warmup prefix of the same genuine episode.
    occupancy = jnp.ones((7, 7), jnp.uint8).at[1:6, 1:6].set(0)
    data = MapInput(occupancy, occupancy == 0, jnp.asarray([7, 7]), jnp.asarray([3, 3]))
    config = HostConfig(
        n_agents=2,
        sensor_radius=1,
        max_distance=1,
        plan_horizon=2,
        max_rounds=3,
        max_candidates=3,
        spawn_radius=1,
    )
    settings = TrainAConfig(
        events=4,
        warmup_events=2,
        rollout_length=2,
        ppo_epochs=1,
        batch_size=1,
        physical_env_count=1,
        hidden=8,
        predictor_replay_capacity=4,
        neighbor_estimator="zoh",
        suite="micro",
    )
    key = jax.random.PRNGKey(11)
    baseline = train_a(data, config, settings, key, tmp_path / "baseline")
    output = tmp_path / "captured"
    output.mkdir()
    (output / "experiment.json").write_text('{"training_seed":11}', encoding="utf-8")
    recording = RecordingConfig(
        every_episodes=1, frame_stride_rounds=1, max_frames=5, duration_ms=100, max_side=192
    )
    prefix = train_a(data, config, settings, key, output, stop_after_updates=1, recording=recording)
    # When: training resumes its actual sampler and records only the newly executed suffix.
    captured = train_a(
        data, config, settings, key, output, resume=prefix.checkpoint, recording=recording
    )
    # Then: learning matches capture disabled; replay contributes zero training events.
    identity = configuration_hash(config, settings)
    expected = serialization.msgpack_restore(load_checkpoint(baseline.checkpoint, identity))
    actual = serialization.msgpack_restore(load_checkpoint(captured.checkpoint, identity))
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(expected["sampler"]["key"], actual["sampler"]["key"])
    np.testing.assert_array_equal(expected["sampler"]["real_events"], [4])
    assert captured.real_events == 4
    assert captured.session_real_events == 2
    assert captured.actor_parameter_change > 0
    assert captured.stage_seconds is not None
    assert captured.stage_seconds["recording"] > 0
    evidence = [
        json.loads(path.read_text(encoding="utf-8")) for path in (output / "gifs").glob("*.json")
    ]
    assert sorted(record["initial_event"] for record in evidence) == [0, 2]
    assert sum(record["valid_real_events"] for record in evidence) == 4
    assert all(record["source"] == "actual_real_training_actions" for record in evidence)
    assert all(record["replay_verified"] for record in evidence)
    assert all(record["training_events_added_by_replay"] == 0 for record in evidence)
    assert all(record["partial"] for record in evidence)
    assert all(record["training_seed"] == 11 for record in evidence)
    phases = {version["phase"] for record in evidence for version in record["model_versions"]}
    assert phases == {"warmup", "train"}
    gifs = tuple((output / "gifs").glob("*.gif"))
    assert gifs
    for path in gifs:
        with Image.open(path) as image:
            assert image.format == "GIF"
            assert image.n_frames >= 1
