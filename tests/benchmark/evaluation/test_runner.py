"""Real missing-file failures test resumability without mocking a model or its score."""

import json
from pathlib import Path

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.records import completed_result
from ir2_benchmark.evaluation.runner import run_manifest
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    make_scenario,
    tasks,
    write_task_manifest,
)


def test_resume_keeps_one_error_result_and_does_not_hide_failure_status(tmp_path: Path) -> None:
    # Given: a real absent checkpoint, handled before any JAX runtime is needed.
    scenario = make_scenario(
        EvaluationMap("not-opened.png", "map", "test", "ir2", "rooms", 512, "fixture-anchor"),
        HostConfig(),
        0,
    )
    method = Method(
        "A", "A", "fixed", "zoh", 0, str(tmp_path / "absent-checkpoint"), "expected", 2000000
    )
    task = next(iter(tasks((scenario,), (method,))))
    manifest = tmp_path / "tasks.jsonl"
    write_task_manifest(manifest, (task,))
    output = tmp_path / "evaluation"
    # When: the failed workload is executed and then resumed.
    first, second = run_manifest(manifest, output), run_manifest(manifest, output)
    # Then: one unique result remains, and a resumed error still returns a failure count.
    assert (first.executed, first.failures) == (1, 1)
    assert (second.executed, second.resumed, second.failures) == (0, 1, 1)
    paths = tuple((output / "results").glob("*.json"))
    assert len(paths) == 1
    row = json.loads(paths[0].read_text(encoding="utf-8"))["result"]
    assert row["outcome"] == "model_error"
    assert row["metrics"] is None


def test_corrupt_saved_result_is_reexecuted_instead_of_counted_complete(tmp_path: Path) -> None:
    # Given: a malformed interrupted result file for a real missing checkpoint task.
    scenario = make_scenario(
        EvaluationMap("not-opened.png", "map", "test", "ir2", "rooms", 512, "fixture-anchor"),
        HostConfig(),
        0,
    )
    method = Method("A", "A", "fixed", "zoh", 0, str(tmp_path / "absent"), "expected", 2000000)
    task = next(iter(tasks((scenario,), (method,))))
    manifest, output = tmp_path / "tasks.jsonl", tmp_path / "out"
    write_task_manifest(manifest, (task,))
    path = output / "results" / f"{task.result_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"result":', encoding="utf-8")
    # When: the same saved workload is resumed.
    summary = run_manifest(manifest, output)
    # Then: the incomplete record is replaced by a complete explicit error record.
    assert summary.executed == 1
    assert summary.resumed == 0
    assert completed_result(path, task)
