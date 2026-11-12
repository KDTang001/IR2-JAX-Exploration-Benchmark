"""Actual checkpoint loading keeps a bounded, hash-verified inference parameter window."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.batching import ParameterCache, prepare_task
from ir2_benchmark.evaluation.errors import ModelExecutionError
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.scenarios import EvaluationMap, Method, make_scenario, tasks


def test_real_parameter_cache_is_bounded_and_rejects_changed_checkpoint(tmp_path: Path) -> None:
    # Given: three actual initialized-model checkpoints with the same inference structure.
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415
    from flax import serialization  # noqa: PLC0415

    from ir2_benchmark.training.checkpoint import save_checkpoint  # noqa: PLC0415
    from ir2_benchmark.training.train_a import (  # noqa: PLC0415
        TrainAConfig,
        configuration_hash,
        initialize_training,
    )

    config = HostConfig(
        sensor_radius=1, max_distance=1, max_rounds=2, plan_horizon=2, max_candidates=3
    )
    scene = make_scenario(
        EvaluationMap("not-loaded.png", "occupancy", "test", "ir2", "rooms", 7, "anchor"), config, 0
    )
    settings = TrainAConfig(
        events=8,
        warmup_events=0,
        rollout_length=2,
        ppo_epochs=1,
        batch_size=1,
        hidden=8,
        predictor_replay_capacity=4,
        suite="micro",
    )
    from ir2_benchmark.types import MapInput  # noqa: PLC0415

    grid = jnp.ones((7, 7), jnp.uint8).at[1:6, 1:6].set(0)
    data = MapInput(grid, grid == 0, jnp.asarray([7, 7]), jnp.asarray([3, 3]))
    state = initialize_training(data, config, settings, jax.random.PRNGKey(1))
    models = {
        name: {"params": serialization.to_state_dict(model.params)}
        for name, model in (
            ("actor", state.actor),
            ("critic", state.critic),
            ("predictor", state.predictor),
        )
    }
    identity = configuration_hash(config, settings)
    prepared_models = []
    for index in range(3):
        root = tmp_path / str(index)
        root.mkdir()
        checkpoint = root / "checkpoint"
        payload = jax.tree.map(lambda value, index=index: jnp.asarray(value) + index * 0.01, models)
        save_checkpoint(checkpoint, serialization.to_bytes(payload), identity)
        (root / "configuration.json").write_text(
            json.dumps(
                {
                    "environment": asdict(config),
                    "training": asdict(settings),
                    "configuration_hash": identity,
                }
            ),
            encoding="utf-8",
        )
        method = Method(
            "A_learned_predict",
            "A",
            "learned",
            "predict",
            index,
            str(checkpoint),
            file_hash(checkpoint),
            0,
        )
        prepared_models.append(prepare_task(next(tasks((scene,), (method,)))))
    # When: loading more real bundles than one two-lane execution window can retain.
    cache = ParameterCache(2)
    first = cache.load(prepared_models[0])
    assert cache.load(prepared_models[0]) is first
    cache.load(prepared_models[1])
    cache.load(prepared_models[2])
    # Then: inference grouping ignores weights, LRU storage stays bounded, and changed files fail.
    assert len({entry.group for entry in prepared_models}) == 1
    assert cache.size == 2  # noqa: PLR2004
    assert cache.load(prepared_models[0]) is not first
    path = Path(prepared_models[0].task.method.checkpoint)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ModelExecutionError, match="checkpoint_content_hash_mismatch"):
        cache.load(prepared_models[0])
