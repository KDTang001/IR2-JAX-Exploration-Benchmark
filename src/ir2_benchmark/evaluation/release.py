"""Audit manifests bind real artifacts and keep unfinished formal work explicitly incomplete."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from shutil import which

from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.scenarios import read_task_manifest


def write_release_manifest(results: Path, report: Path, manifest: Path | None) -> Path:
    """Record artifact identities and actual available training counters."""
    package = Path(__file__).resolve().parents[1]
    repository = package.parent.parent
    sources = {
        path.relative_to(package).as_posix(): file_hash(path)
        for path in sorted(package.rglob("*.py"))
    }
    code_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    packages = {}
    for name in ("jax", "jaxlib", "flax", "optax", "numpy", "pillow", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    checkpoints, maps, configurations = {}, set(), set()
    if manifest is not None:
        for task in read_task_manifest(manifest):
            maps.add(task.scenario.map_hash)
            configurations.add(task.scenario.config_hash)
            method = task.method
            if method.checkpoint and method.checkpoint_hash not in checkpoints:
                path = Path(method.checkpoint)
                summary_path = path.parent / "summary.json"
                counters = (
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary_path.exists()
                    else {}
                )
                checkpoints[method.checkpoint_hash] = {
                    "path": str(path),
                    "exists": path.is_file(),
                    "actual_hash": file_hash(path) if path.is_file() else None,
                    "real_events": counters.get("real_events"),
                    "imagined_events": counters.get("imagined_events"),
                    "model_selection": (
                        "external_pretrained_reference"
                        if method.route == "baseline"
                        else (
                            "fixed_validation_coverage_auc"
                            if (
                                counters.get("best_validation_auc") is not None
                                or counters.get("validation_auc") is not None
                            )
                            and path.parent.name == "best"
                            else "latest_checkpoint_unselected"
                        )
                    ),
                    "training_seed": method.training_seed,
                    "selected_at_events": method.selected_at_events,
                    "total_training_events": method.real_training_events,
                }
    git = which("git")
    revision = (
        subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False
        ).stdout.strip()
        if git
        else "unavailable"
    )
    report_data = json.loads((report / "report.json").read_text(encoding="utf-8"))
    files = [repository / "requirements-server.txt", repository / "pyproject.toml"]
    files.extend(sorted((repository / "configs").rglob("*.json")))
    payload = {
        "formal_release_complete": bool(report_data["formal_complete"]),
        "core_complete": report_data.get("core_complete", False),
        "auxiliary_complete": report_data.get("auxiliary_complete", False),
        "missing_auxiliary_conditions": report_data.get("missing_auxiliary_conditions", []),
        "workload_scope": report_data.get("workload_scope", {"status": "unverified_scope"}),
        "code_revision": revision,
        "code_hash": code_hash,
        "source_hashes": sources,
        "python": platform.python_version(),
        "packages": packages,
        "runtime_lock": "approved conda ex snapshot + requirements-server.txt",
        "configuration_files": {
            str(path.relative_to(repository)): file_hash(path) for path in files if path.is_file()
        },
        "scenario_manifest": str(manifest) if manifest else None,
        "scenario_manifest_hash": file_hash(manifest) if manifest else None,
        "map_hashes": sorted(maps),
        "configuration_hashes": sorted(configurations),
        "checkpoints": checkpoints,
        "results": str(results),
        "errors": str(report / "failures.csv"),
        "remaining_result_ids": str(report / "pending_tasks.jsonl"),
        "missing_matrix_conditions": report_data.get("missing_matrix_conditions", []),
        "limited": report_data["limited"],
    }
    output = report / "release_manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output
