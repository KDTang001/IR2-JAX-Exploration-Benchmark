"""Audit manifests bind real artifacts and keep unfinished formal work explicitly incomplete."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, distributions, version
from pathlib import Path
from shutil import which
from urllib.parse import urlsplit
from urllib.request import url2pathname

from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.scenarios import read_task_manifest


def _wheel_provenance(direct_url: dict | None) -> dict:
    """Hash only an accessible local archive, never infer its bytes from RECORD."""
    result = {"wheel_sha256": None, "wheel_status": "archive_not_verified"}
    if not direct_url:
        return result
    url = urlsplit(direct_url.get("url", ""))
    if url.scheme != "file" or url.netloc not in ("", "localhost"):
        return result
    archive = Path(url2pathname(url.path))
    if archive.suffix != ".whl" or not archive.is_file():
        return result
    actual = file_hash(archive)
    info = direct_url.get("archive_info", {})
    declared = info.get("hashes", {}).get("sha256")
    if declared is None and info.get("hash", "").startswith("sha256="):
        declared = info["hash"].removeprefix("sha256=")
    return {
        "wheel_sha256": actual,
        "wheel_declared_sha256": declared,
        "wheel_status": (
            "verified_archive_hash" if declared == actual
            else "archive_hash_mismatch" if declared else "archive_hashed_no_declared_hash"
        ),
    }


def _runtime_provenance() -> dict:
    """Observe runtime identity without initializing an accelerator backend."""
    inventory = sorted(
        (item.metadata.get("Name", ""), item.version) for item in distributions()
    )
    bridge = sys.modules.get("jax._src.xla_bridge")
    backends = getattr(bridge, "_backends", {})
    return {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "installed_distributions": inventory,
        "installed_distributions_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()
        ).hexdigest(),
        "jax_backend_status": "observed_initialized" if backends else "not_observed_initialized",
        "jax_devices": {
            name: [str(device) for device in backend.devices()]
            for name, backend in backends.items()
        },
    }


def write_release_manifest(results: Path, report: Path, manifest: Path | None) -> Path:
    """Record artifact identities and actual available training counters."""
    package = Path(__file__).resolve().parents[1]
    candidate = package.parent.parent
    repository = (
        candidate
        if package.parent.name == "src"
        and (candidate / "pyproject.toml").is_file()
        and (candidate / ".git").exists()
        else None
    )
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
    revision = None
    dirty = None
    if git and repository is not None:
        result = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False
        )
        revision = result.stdout.strip() if result.returncode == 0 else None
        status = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain"], cwd=repository,
            capture_output=True, text=True, check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    installed = None
    try:
        metadata = distribution("ir2-benchmark")
    except PackageNotFoundError:
        pass
    else:
        record = metadata.read_text("RECORD")
        direct_url = metadata.read_text("direct_url.json")
        origin = json.loads(direct_url) if direct_url else None
        location = Path(metadata.locate_file("ir2_benchmark")).resolve()
        installed = {
            "version": metadata.version,
            "requires": metadata.requires,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None,
            "direct_url": origin,
            "package_location": str(location),
            "import_path_match": location == package,
            **_wheel_provenance(origin),
        }
    report_data = json.loads((report / "report.json").read_text(encoding="utf-8"))
    files = []
    if repository is not None:
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
        "runtime_lock": None,
        "runtime": _runtime_provenance(),
        "provenance": {
            "kind": "source_checkout" if repository is not None else "installed_distribution",
            "source_path": str(package),
            "revision_status": "observed_checkout" if revision else "unavailable",
            "working_tree_dirty": dirty,
            "distribution": installed,
            "runtime_lock_status": "not_verified",
        },
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
