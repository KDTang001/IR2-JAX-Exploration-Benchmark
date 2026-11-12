"""Bounded grouping and genuine batched episodes preserve identity and independent keys."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.batching import (
    BoundedTaskQueue,
    GroupKey,
    PreparedTask,
)
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    make_scenario,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash

if TYPE_CHECKING:
    from pathlib import Path
    from typing import NoReturn

    from jax import Array

    from ir2_benchmark.env.rollout import RolloutResult
    from ir2_benchmark.evaluation.batching import EvaluationKernel, ModelParameters
    from ir2_benchmark.evaluation.episode import LoadedLane
    from ir2_benchmark.types import MapInput


def prepared(index: int, budget: int = 8192) -> PreparedTask:
    """Create only task metadata; queue tests do not allocate maps or model parameters."""
    config = HostConfig(initial_budget=budget)
    scene = make_scenario(
        EvaluationMap("not-loaded.png", "occupancy", "test", "ir2", "rooms", 512, "anchor"),
        config,
        index,
    )
    method = Method("heuristic", "heuristic", "fixed", "none", -1, "", "", 0)
    task = next(tasks((scene,), (method,)))
    return PreparedTask(task, GroupKey(config, 512, "heuristic", ()), None)


def test_queue_flushes_same_group_without_loading_a_whole_manifest() -> None:
    # Given: two interleaved static configurations and a two-task capacity per batch.
    queue = BoundedTaskQueue(batch_size=2, max_pending=3)
    first, other, same = prepared(0), prepared(1, 2048), prepared(2)
    queue.append(first)
    queue.append(other)
    # When: a matching scenario arrives later in the stream.
    ready = queue.append(same)
    # Then: only the matching group is dispatched; the other remains bounded metadata.
    assert ready == ((first, same),)
    assert queue.pending_count == 1
    assert queue.finish() == ((other,),)


def test_queue_bound_flushes_oldest_partial_group_instead_of_accumulating_tasks() -> None:
    # Given: distinct groups that cannot fill a three-lane batch immediately.
    queue = BoundedTaskQueue(batch_size=3, max_pending=3)
    entries = tuple(prepared(index, 1024 * (index + 1)) for index in range(4))
    for entry in entries[:3]:
        queue.append(entry)
    # When: another group would exceed the declared pending metadata limit.
    ready = queue.append(entries[3])
    # Then: the oldest actual partial batch is emitted without scanning the remaining workload.
    assert ready == ((entries[0],),)
    assert queue.pending_count == 3  # noqa: PLR2004


def test_group_identity_does_not_include_checkpoint_hash_or_scenario_seed() -> None:
    # Given: metadata for another checkpoint and a distinct scenario seed with identical structure.
    first, second = prepared(0), prepared(1)
    changed = replace(second.task, method=replace(second.task.method, checkpoint_hash="different"))
    second = replace(second, task=changed)
    # When: both records are queued under the same static inference group.
    queue = BoundedTaskQueue(batch_size=2, max_pending=2)
    queue.append(first)
    ready = queue.append(second)
    # Then: different learned weights/keys do not force an otherwise compatible group apart.
    assert ready == ((first, second),)


def write_real_workload(directory: Path) -> Path:
    """Persist distinct genuine maps and seeds in the same native compilation bucket."""
    directory.mkdir(parents=True, exist_ok=True)
    config = HostConfig(
        sensor_radius=1,
        max_distance=1,
        max_rounds=2,
        coverage_target=1,
        plan_horizon=2,
        max_candidates=3,
        spawn_radius=1,
    )
    scenes = []
    for index in range(3):
        pixels = np.full((7, 12), 127, np.uint8)
        pixels[2 + index % 2, 1:11] = 195
        pixels[2 + index % 2, 1] = 208
        path = directory / f"map-{index}.png"
        Image.fromarray(pixels).save(path)
        occupancy = (pixels <= 150).astype(np.uint8)  # noqa: PLR2004
        anchor = (1, 2 + index % 2)
        row = EvaluationMap(
            str(path),
            obstacle_hash(occupancy),
            "test",
            "ir2",
            "corridor",
            12,
            scenario_hash(occupancy, anchor),
        )
        scenes.append(make_scenario(row, config, index))
    method = Method("heuristic", "heuristic", "fixed", "none", -1, "", "", 0)
    manifest = directory / "tasks.jsonl"
    write_task_manifest(manifest, tasks(scenes, (method,)))
    return manifest


def test_batch_two_matches_serial_real_trajectories_and_reuses_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: three actual tasks, requiring a full batch and a genuine tail batch.
    jax = pytest.importorskip("jax")
    from ir2_benchmark.evaluation import episode  # noqa: PLC0415
    from ir2_benchmark.evaluation.runner import run_manifest  # noqa: PLC0415

    manifest = write_real_workload(tmp_path / "source")
    # When: the same fixed workload runs with batch1 and batch2.
    serial = run_manifest(manifest, tmp_path / "serial", batch_size=1)
    select_kernel = episode.evaluation_kernel
    batches, compilations = [], []

    def inspect_kernel(prepared: PreparedTask) -> EvaluationKernel:
        kernel = select_kernel(prepared)

        def call(  # noqa: PLR0913, PLR0917
            parameters: ModelParameters,
            map_data: MapInput,
            policy_key: Array,
            spawns: Array,
            fault_key: Array,
            isolated_mask: Array,
        ) -> tuple[RolloutResult, Array]:
            batches.append(map_data.occupancy.shape[0])
            result = kernel(parameters, map_data, policy_key, spawns, fault_key, isolated_mask)
            jax.block_until_ready(result)
            compilations.append(kernel._cache_size())  # noqa: SLF001
            return result

        return call

    monkeypatch.setattr(episode, "evaluation_kernel", inspect_kernel)
    batched = run_manifest(manifest, tmp_path / "batch", batch_size=2)
    run_manifest(manifest, tmp_path / "batch-repeat", batch_size=2)
    # Then: each independent scenario retains its exact physical/cognitive/cost trajectory.
    assert (serial.executed, batched.executed, batched.failures) == (3, 3, 0)
    assert batches == [2, 1, 2, 1]
    assert len(set(compilations)) == 1
    for path in sorted((tmp_path / "serial" / "results").glob("*.json")):
        left = json.loads(path.read_text(encoding="utf-8"))["result"]
        right = json.loads(
            (tmp_path / "batch" / "results" / path.name).read_text(encoding="utf-8")
        )["result"]
        for field in ("result_id", "scenario_id", "outcome", "emitted_bytes", "configuration_hash"):
            assert left[field] == right[field]
        with np.load(left["trace_file"]) as a, np.load(right["trace_file"]) as b:
            assert a.files == b.files
            for field in a.files:
                np.testing.assert_array_equal(a[field], b[field])


def test_bad_map_lane_is_saved_without_discarding_other_real_episodes(tmp_path: Path) -> None:
    # Given: one missing map in a real three-task workload with batch size two.
    pytest.importorskip("jax")
    from ir2_benchmark.evaluation.runner import run_manifest  # noqa: PLC0415

    manifest = write_real_workload(tmp_path / "source")
    (manifest.parent / "map-1.png").unlink()
    # When: preprocessing rejects that lane before invoking the compatible remaining lanes.
    summary = run_manifest(manifest, tmp_path / "out", batch_size=2)
    # Then: the failed task has no invented trace and both genuine surviving episodes finish.
    assert (summary.executed, summary.failures) == (3, 1)
    rows = [
        json.loads(path.read_text())["result"] for path in (tmp_path / "out/results").glob("*.json")
    ]
    assert sorted(row["outcome"] for row in rows) == ["map_error", "timeout", "timeout"]
    assert sum(bool(row["trace_file"]) for row in rows) == 2  # noqa: PLR2004


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (MemoryError("allocator failed"), "oom"),
        (RuntimeError("kernel failed"), "environment_error"),
        (TypeError("invalid model"), "model_error"),
    ],
)
def test_compute_failure_records_the_whole_batch_without_serial_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: MemoryError | RuntimeError | TypeError,
    outcome: str,
) -> None:
    # Given: genuine lane preprocessing, with an injected failure at the device-compute boundary.
    pytest.importorskip("jax")
    from ir2_benchmark.evaluation import runner  # noqa: PLC0415

    manifest = write_real_workload(tmp_path / "source")
    sizes = []

    def fail(lanes: tuple[LoadedLane, ...]) -> NoReturn:
        sizes.append(len(lanes))
        raise error

    monkeypatch.setattr(runner, "execute_batch", fail)
    # When: one complete batch and one real tail batch fail.
    summary = runner.run_manifest(manifest, tmp_path / "out", batch_size=2)
    # Then: each lane retains the actual failure, and no single-lane fallback is attempted.
    assert sizes == [2, 1]
    assert (summary.executed, summary.failures) == (3, 3)
    for path in (tmp_path / "out/results").glob("*.json"):
        row = json.loads(path.read_text())["result"]
        assert (row["outcome"], row["error"], row["trace_file"]) == (outcome, str(error), "")
