"""Freeze new validation task identities for the declared C1024 capacity check."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ir2_benchmark.config import ConfigError, HostConfig, load_config
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    EvaluationTask,
    make_scenario,
    read_task_manifest,
    tasks,
    write_task_manifest,
)


def capacity_tasks(
    parent: tuple[EvaluationTask, ...], extended: HostConfig
) -> tuple[EvaluationTask, ...]:
    """Keep the frozen five-map/two-method workload and change only graph storage."""
    expected = {
        (source, family, method)
        for source, family in (
            ("ir2", "easy"), ("ir2", "difficult"), ("procedural", "rooms"),
            ("procedural", "corridor"), ("procedural", "hybrid"),
        )
        for method in ("nf_mixed", "ir2_mixed")
    }
    actual = {
        (task.scenario.source, task.scenario.family, task.method.method_id) for task in parent
    }
    if len(parent) != len(expected) or actual != expected:
        raise ConfigError(field="capacity.tasks", detail="requires the ten frozen validation tasks")
    planned = []
    for task in parent:
        scene = task.scenario
        changed = {
            name for name, value in asdict(scene.config).items()
            if value != getattr(extended, name)
        }
        if changed != {"graph_capacity", "graph_padding_size"} or (
            scene.config.graph_capacity, scene.config.graph_padding_size,
            extended.graph_capacity, extended.graph_padding_size,
        ) != (400, 400, 1024, 1024):
            raise ConfigError(
                field="capacity.config", detail="only C400 to C1024 storage is allowed"
            )
        if (
            scene.split != "validation" or not scene.limited
            or (scene.n_agents, scene.budget, scene.fault_mode, scene.max_rounds)
            != (4, 8192, 0, 512)
            or scene.drop_probability != 0 or scene.budget_mode != "per_robot"
            or task.method.route != "baseline" or task.method.communication != "mixed"
            or task.method.real_training_events != 0
        ):
            raise ConfigError(
                field="capacity.scope", detail="not the declared validation cost subset"
            )
        data = EvaluationMap(
            scene.map_path, scene.map_hash, scene.split, scene.source, scene.family,
            scene.bucket, scene.map_scenario_hash,
        )
        original = make_scenario(data, scene.config, 0, limited=True)
        if original != scene or next(tasks((original,), (task.method,))) != task:
            raise ConfigError(field="capacity.identity", detail="parent identity or seed changed")
        replacement = make_scenario(data, extended, 0, limited=True)
        planned.append(next(tasks((replacement,), (task.method,))))
    return tuple(planned)


def main() -> None:
    """Write the new manifest and explicit old/new identity mapping without executing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(".capacity.json")
    if args.output.exists() or sidecar.exists():
        raise ConfigError(field="capacity.output", detail="refusing to overwrite a frozen plan")
    parent = tuple(read_task_manifest(args.parent_manifest))
    planned = capacity_tasks(parent, load_config(args.config))
    write_task_manifest(args.output, planned)
    sidecar.write_text(json.dumps({
        "name": "capacity-check-v1-c1024", "status": "frozen_not_executed",
        "parent_manifest_sha256": file_hash(args.parent_manifest),
        "config_file_sha256": file_hash(args.config),
        "task_manifest_sha256": file_hash(args.output), "task_count": len(planned),
        "changed_fields": ["graph_capacity", "graph_padding_size"],
        "task_mapping": [
            {"parent_result_id": old.result_id, "result_id": new.result_id,
             "map_hash": new.scenario.map_hash, "family": new.scenario.family,
             "method_id": new.method.method_id}
            for old, new in zip(parent, planned, strict=True)
        ],
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
