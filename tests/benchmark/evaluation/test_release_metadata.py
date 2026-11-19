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


def test_installed_release_does_not_inherit_unrelated_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ir2_benchmark.evaluation import release  # noqa: PLC0415

    package = tmp_path / "site-packages" / "ir2_benchmark"
    module = package / "evaluation" / "release.py"
    module.parent.mkdir(parents=True)
    module.write_text("# installed source", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("unrelated", encoding="utf-8")
    monkeypatch.setattr(release, "__file__", str(module))
    (tmp_path / "report.json").write_text(
        json.dumps({"formal_complete": False, "limited": True}), encoding="utf-8",
    )
    output = release.write_release_manifest(tmp_path, tmp_path, None)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["code_revision"] is None
    assert payload["runtime_lock"] is None
    assert payload["configuration_files"] == {}
    assert payload["provenance"]["kind"] == "installed_distribution"
    assert payload["source_hashes"] == {"evaluation/release.py": file_hash(module)}


@pytest.mark.parametrize("declared", ["matching", "wrong", None])
def test_local_wheel_hash_is_observed_and_compared(tmp_path: Path, declared: str | None) -> None:
    from ir2_benchmark.evaluation.release import _wheel_provenance  # noqa: PLC0415

    archive = tmp_path / "package.whl"
    archive.write_bytes(b"archive bytes")
    digest = file_hash(archive)
    hashes = {} if declared is None else {"sha256": digest if declared == "matching" else declared}
    actual = _wheel_provenance({"url": archive.as_uri(), "archive_info": {"hashes": hashes}})
    assert actual["wheel_sha256"] == digest
    expected = {
        "matching": "verified_archive_hash",
        "wrong": "archive_hash_mismatch",
        None: "archive_hashed_no_declared_hash",
    }
    assert actual["wheel_status"] == expected[declared]
    archive.unlink()
    assert _wheel_provenance({"url": archive.as_uri()})["wheel_sha256"] is None


def test_runtime_identity_without_backend_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from ir2_benchmark.evaluation.release import _runtime_provenance  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", None)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "observed-fixture")
    actual = _runtime_provenance()
    assert actual["executable"] == sys.executable
    assert actual["prefix"] == sys.prefix
    assert actual["conda_default_env"] == "observed-fixture"
    assert actual["jax_devices"] == {}
    assert actual["jax_backend_status"] == "not_observed_initialized"
    assert actual["installed_distributions_sha256"] == hashlib.sha256(
        json.dumps(actual["installed_distributions"], sort_keys=True).encode()
    ).hexdigest()
