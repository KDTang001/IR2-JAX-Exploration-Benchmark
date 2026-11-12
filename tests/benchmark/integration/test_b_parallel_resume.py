"""Same-quota B recovery preserves discrete state exactly and floating learners numerically."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import serialization
from PIL import Image

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.scenarios import EvaluationMap, make_scenario, write_scenario_manifest
from ir2_benchmark.evaluation.validation import (
    ValidationEvaluator,
    ValidationScore,
    build_validation_callback,
)
from ir2_benchmark.maps.manifest import canonical_obstacle_hash, obstacle_hash, scenario_hash
from ir2_benchmark.maps.sampler import ManifestSource, TrainingMapSampler
from ir2_benchmark.training.b_state import BTrainingConfig
from ir2_benchmark.training.checkpoint import load_checkpoint
from ir2_benchmark.training.replay import ReplayBuffer, ReplayError, read_field
from ir2_benchmark.training.train_b import train_b
from ir2_benchmark.visualization.recording import RecordingConfig


def test_parallel_partial_update_resume_matches_uninterrupted_same_quota(tmp_path: Path) -> None:
    # Given: odd boundaries require masked physical tails; total quota remains21 in both sessions.
    config = HostConfig(sensor_radius=1, max_distance=2, max_candidates=4, max_rounds=3)
    options = BTrainingConfig(
        hidden=8,
        categories=2,
        classes=3,
        physical_env_count=4,
        logical_batch=2,
        update_every=7,
        sequence_length=3,
        burn_in=0,
        warmup_events=5,
        imagination_rounds=1,
        development=True,
    )
    full, split = tmp_path / "full", tmp_path / "split"
    recording = RecordingConfig(every_episodes=1, frame_stride_rounds=1, max_frames=3)
    expected = train_b(config, 11, 21, full, options=options)
    # When: stopping after a real model/actor update, then advancing the same original quota.
    partial = train_b(
        config, 11, 21, split, options=options, stop_after_updates=1, recording=recording
    )
    actual = train_b(config, 11, 21, split, options=options, resume=split, recording=recording)
    # Then: real counts are never rounded toP, and resumed work is counted only in this session.
    assert partial["real_events"] == 7  # noqa: PLR2004
    assert actual["real_events"] == expected["real_events"] == 21  # noqa: PLR2004
    assert actual["session_real_events"] == 14  # noqa: PLR2004
    assert actual["model_updates"] == expected["model_updates"] == 3  # noqa: PLR2004
    assert actual["actor_updates"] == expected["actor_updates"] == 3  # noqa: PLR2004
    assert actual["imagined_events"] == expected["imagined_events"]
    assert actual["actor_parameter_delta"] == expected["actor_parameter_delta"]
    recovered = []
    for directory, summary in ((full, expected), (split, actual)):
        recovered.append(
            serialization.msgpack_restore(
                load_checkpoint(directory / "checkpoint", summary["config_hash"])
            )
        )
    for observed, reference in zip(
        jax.tree_util.tree_leaves(recovered[1]),
        jax.tree_util.tree_leaves(recovered[0]),
        strict=True,
    ):
        observed_array, reference_array = np.asarray(observed), np.asarray(reference)
        assert observed_array.shape == reference_array.shape
        assert observed_array.dtype == reference_array.dtype
        # Use route A's existing FP32 recovery contract; discrete state stays exact.
        if np.issubdtype(reference_array.dtype, np.floating):
            np.testing.assert_allclose(observed_array, reference_array, rtol=1e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(observed_array, reference_array)
    records = [json.loads(line) for line in (split / "updates.jsonl").read_text().splitlines()]
    assert [row["real_events"] for row in records] == [7, 14, 21]
    assert all(row["physical_env_count"] == 4 for row in records)  # noqa: PLR2004
    replay = ReplayBuffer.load(split / "replay.npz")
    policy_flags = [
        bool(read_field(row.before_fields, "policy_valid"))
        for rows in replay._episodes.values()  # noqa: SLF001
        for row in rows
    ]
    assert sum(policy_flags) == 16  # noqa: PLR2004
    assert replay.sampling_state == ReplayBuffer.load(full / "replay.npz").sampling_state
    assert actual["config_hash"] == expected["config_hash"]
    assert actual["stage_seconds"]["recording"] > 0
    assert tuple((split / "gifs").rglob("*.gif"))


@pytest.fixture
def native_fixture_sampler(tmp_path: Path) -> TrainingMapSampler:
    """Real tiny fixture PNGs exercise native padding/load paths, not formal template quality."""
    grid = np.zeros((9, 9), np.uint8)
    pixels = np.full((9, 9), 195, np.uint8)
    pixels[4, 4] = 208
    Image.fromarray(pixels).save(tmp_path / "fixture.png")
    common = {
        "path": "fixture.png",
        "split": "train",
        "occupancy_hash": obstacle_hash(grid),
        "canonical_hash": canonical_obstacle_hash(grid),
        "scenario_hash": scenario_hash(grid, (4, 4)),
    }
    ir2 = [
        {**common, "family": family, "bucket": bucket}
        for family, bucket in (("easy", 512), ("difficult", 768))
    ]
    procedural = [
        {**common, "template": family, "size": bucket}
        for family in ("rooms", "corridor", "hybrid")
        for bucket in (512, 768, 1024)
    ]
    first, second = tmp_path / "ir2.jsonl", tmp_path / "procedural.jsonl"
    first.write_text("\n".join(map(json.dumps, ir2)), encoding="utf-8")
    second.write_text("\n".join(map(json.dumps, procedural)), encoding="utf-8")
    return TrainingMapSampler(
        (ManifestSource(tmp_path, first, "ir2"), ManifestSource(tmp_path, second, "procedural")),
        seed=0,
    )


def test_nine_native_pool_checkpoint_resumes_forward_learning_under_same_quota(
    tmp_path: Path,
    native_fixture_sampler: TrainingMapSampler,
) -> None:
    # Given: seed0 introduces all9native(bucket,N) groups in its first20actual map assignments.
    sampler = native_fixture_sampler
    config = HostConfig(
        sensor_radius=1, max_distance=2, max_candidates=4, max_rounds=1, agent_bucket=6
    )
    options = BTrainingConfig(
        hidden=8,
        categories=2,
        classes=3,
        physical_env_count=2,
        logical_batch=1,
        update_every=128,
        sequence_length=2,
        burn_in=0,
        warmup_events=0,
        imagination_rounds=1,
        development=True,
    )
    full, split = tmp_path / "formal-full", tmp_path / "formal-split"
    expected = train_b(config, 0, 256, full, options=options, sampler=sampler)
    # When: the original256event quota is paused at its first optimizer boundary and then resumed.
    partial = train_b(config, 0, 256, split, options=options, sampler=sampler, stop_after_updates=1)
    saved = serialization.msgpack_restore(
        load_checkpoint(split / "checkpoint", partial["config_hash"])
    )
    assert len(saved["pools"]) == 9  # noqa: PLR2004
    assert partial["real_events"] == 128  # noqa: PLR2004
    actual = train_b(config, 0, 256, split, options=options, sampler=sampler, resume=split)
    # Then: the remaining actual events and world/actor updates produce the uninterrupted state.
    assert actual["session_real_events"] == 128  # noqa: PLR2004
    assert actual["model_updates"] == actual["actor_updates"] == 2  # noqa: PLR2004
    assert actual["actor_parameter_delta"] > partial["actor_parameter_delta"] > 0
    states = [
        serialization.msgpack_restore(load_checkpoint(path / "checkpoint", expected["config_hash"]))
        for path in (full, split)
    ]
    for first, second in zip(
        jax.tree_util.tree_leaves(states[0]), jax.tree_util.tree_leaves(states[1]), strict=True
    ):
        first_array, second_array = np.asarray(first), np.asarray(second)
        assert first_array.shape == second_array.shape
        assert first_array.dtype == second_array.dtype
        if np.issubdtype(first_array.dtype, np.floating):
            np.testing.assert_allclose(first_array, second_array, rtol=1e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(first_array, second_array)
    for name in states[0]["pools"]:
        before = ReplayBuffer.load(full / "replay" / f"{name}.npz")
        after = ReplayBuffer.load(split / "replay" / f"{name}.npz")
        assert before.size == after.size
        assert before.sampling_state == after.sampling_state
    assignments = [
        json.loads(line) for line in (split / "training_maps.jsonl").read_text().splitlines()
    ]
    assert [row["index"] for row in assignments] == list(range(actual["schedule_cursor"]))
    assert sum(actual["source_event_counts"].values()) == actual["real_events"]
    assert sum(actual["source_episode_counts"].values()) == actual["schedule_cursor"]


def test_native_validation_persists_only_changed_replays(
    tmp_path: Path,
    native_fixture_sampler: TrainingMapSampler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: nine real native pools reach validation before their first learning update.
    config = HostConfig(
        sensor_radius=1, max_distance=2, max_candidates=4, max_rounds=1, agent_bucket=6
    )
    options = BTrainingConfig(
        hidden=8,
        categories=2,
        classes=3,
        physical_env_count=2,
        logical_batch=1,
        update_every=128,
        sequence_length=2,
        burn_in=0,
        warmup_events=0,
        imagination_rounds=1,
        validation_interval=128,
        development=True,
    )
    grid = np.zeros((11, 11), np.uint8)
    pixels = np.full(grid.shape, 195, np.uint8)
    pixels[5, 5] = 208
    map_path = tmp_path / "validation.png"
    Image.fromarray(pixels).save(map_path)
    validation_map = EvaluationMap(
        str(map_path), obstacle_hash(grid), "validation", "procedural", "rooms", 64,
        scenario_hash(grid, (5, 5)),
    )
    manifest = tmp_path / "validation.jsonl"
    write_scenario_manifest(manifest, (make_scenario(validation_map, config, 31),))
    output = tmp_path / "training"
    saved: dict[Path, list[tuple[int, str]]] = {}
    validated_paths: list[Path] = []
    original_save = ReplayBuffer.save
    original_validate = ValidationEvaluator.__call__

    def record_save(buffer: ReplayBuffer, path: str | Path) -> None:
        original_save(buffer, path)
        saved.setdefault(Path(path), []).append(
            (buffer.size + buffer.evicted_events, buffer.sampling_state)
        )

    def check_snapshot_then_validate(
        evaluator: ValidationEvaluator, checkpoint: Path, real_events: int
    ) -> ValidationScore:
        assert real_events == 128  # noqa: PLR2004
        assert len(saved) == 9  # noqa: PLR2004
        assert all(len(records) == 1 for records in saved.values())
        validated_paths.extend(saved)
        return original_validate(evaluator, checkpoint, real_events)

    monkeypatch.setattr(ReplayBuffer, "save", record_save)
    monkeypatch.setattr(ValidationEvaluator, "__call__", check_snapshot_then_validate)
    # When: real validation is followed by learning and a final save without more collection.
    result = train_b(
        config, 0, 128, output, options=options, sampler=native_fixture_sampler,
        validation=build_validation_callback(manifest, tmp_path / "selection", training_seed=0),
    )
    # Then: only the active pool is saved again, including its changed replay sampling state.
    assert result["real_events"] == 128  # noqa: PLR2004
    assert result["model_updates"] == result["actor_updates"] == 1
    checkpoint = serialization.msgpack_restore(
        load_checkpoint(output / "checkpoint", result["config_hash"])
    )
    active = output / "replay" / f"{checkpoint['pool_key']}.npz"
    assert len(saved[active]) == 2  # noqa: PLR2004
    before, after = saved[active]
    assert before[0] == after[0] == int(checkpoint["pools"][checkpoint["pool_key"]]["events"])
    assert before[1] != after[1]
    restored = ReplayBuffer.load(active)
    assert restored.size + restored.evicted_events == after[0]
    assert restored.sampling_state == after[1]
    assert len(validated_paths) == 9  # noqa: PLR2004
    assert set(saved) == set(validated_paths)
    assert all(len(records) == 1 for path, records in saved.items() if path != active)
    assert sum(map(len, saved.values())) == 10  # noqa: PLR2004


def test_concurrent_open_episode_capacity_requires_explicit_sufficient_setting(
    tmp_path: Path,
) -> None:
    # Given: four2-agent lanes can retain24open events, but the configured replay can hold only23.
    config = HostConfig(max_rounds=3)
    options = BTrainingConfig(physical_env_count=4, replay_capacity=23, development=True)
    # When/then: reject insufficient declared capacity before loading maps.
    with pytest.raises(ReplayError, match="replay_capacity_requires_24_events"):
        train_b(config, 0, 1, tmp_path, options=options)
