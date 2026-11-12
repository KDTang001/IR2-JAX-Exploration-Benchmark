"""Actual release export recognizes both route-specific best-validation metadata schemas."""

import json
from pathlib import Path
from typing import Literal

import pytest

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.release import write_release_manifest
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    make_scenario,
    tasks,
    write_task_manifest,
)


@pytest.mark.parametrize(
    ("route", "field"), [("A", "best_validation_auc"), ("B", "validation_auc")]
)
@pytest.mark.parametrize("auc", [None, 0.0, 0.625])
@pytest.mark.parametrize("directory", ["best", "latest"])
def test_release_selection_when_exporting_route_specific_summaries(
    tmp_path: Path, route: Literal["A", "B"], field: str, auc: float | None, directory: str
) -> None:
    # Given: an opaque checkpoint fixture and the exact summary schema written by its route.
    checkpoint = tmp_path / directory / "checkpoint"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"opaque checkpoint fixture; no model execution")
    summary = {
        field: auc,
        "real_events": 2000000,
        "total_training_events": 2000000,
        "selected_at_events": 1800000,
    }
    (checkpoint.parent / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    scenario = make_scenario(
        EvaluationMap("unused.png", "occupancy", "test", "ir2", "rooms", 512, "anchor"),
        HostConfig(),
        0,
    )
    method = Method(
        f"{route}_fixed_zoh",
        route,
        "fixed",
        "zoh",
        3,
        str(checkpoint),
        file_hash(checkpoint),
        2000000,
        selected_at_events=1800000,
    )
    manifest = tmp_path / "tasks.jsonl"
    write_task_manifest(manifest, tasks((scenario,), (method,)))
    report = tmp_path / "report"
    report.mkdir()
    (report / "report.json").write_text(
        json.dumps({"formal_complete": False, "limited": True}), encoding="utf-8"
    )
    # When: the real release writer reads the checkpoint and route-specific sidecar.
    output = write_release_manifest(tmp_path, report, manifest)
    exported = json.loads(output.read_text(encoding="utf-8"))["checkpoints"][method.checkpoint_hash]
    # Then: zero AUC is valid selection evidence, while latest directories remain unselected.
    expected = (
        "fixed_validation_coverage_auc"
        if directory == "best" and auc is not None
        else "latest_checkpoint_unselected"
    )
    assert exported["model_selection"] == expected
    assert exported["actual_hash"] == method.checkpoint_hash
    assert exported["total_training_events"] == 2000000  # noqa: PLR2004
    assert exported["selected_at_events"] == 1800000  # noqa: PLR2004
