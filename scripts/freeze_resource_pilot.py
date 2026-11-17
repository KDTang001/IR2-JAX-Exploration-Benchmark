"""Freeze resource-pilot-v1 validation tasks; never execute an episode."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ir2_benchmark.agents.baselines import make_baseline_method
from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import ConfigError, load_config
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    make_scenario,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.maps.manifest import (
    canonical_obstacle_hash,
    obstacle_hash,
    scenario_hash,
)
from ir2_benchmark.maps.prepare import load_ir2_map

if TYPE_CHECKING:
    from ir2_benchmark.evaluation.scenarios import EvaluationTask
    from ir2_benchmark.maps.sampler import ManifestSource

STRATA: Final = (
    ("ir2", "easy"),
    ("ir2", "difficult"),
    ("procedural", "rooms"),
    ("procedural", "corridor"),
    ("procedural", "hybrid"),
)
CONDITIONS: Final = ((0, 0.0, 64), (1, 0.3, 64), (2, 0.0, 64))


@dataclass(frozen=True, slots=True)
class SelectedMap:
    """Verified native geometry and content identities for one validation stratum."""

    map_data: EvaluationMap
    png_hash: str
    canonical_hash: str
    valid_shape: tuple[int, int]
    manifest: str
    manifest_hash: str
    source_path: str


def select_validation_maps(sources: tuple[ManifestSource, ...]) -> tuple[SelectedMap, ...]:  # noqa: C901
    """Select minimum occupancy hash before scoring; fail on invalid winners, never replace."""
    candidates = {key: [] for key in STRATA}
    forbidden = set()
    for source in sources:
        if source.source not in ("ir2", "procedural"):
            raise ConfigError(field="pilot.source", detail="unknown source registry label")
        for line in source.manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") in ("test", "ood"):
                forbidden.add(row.get("canonical_hash"))
            if row.get("split") != "validation" or row.get("exclusion_reason") is not None:
                continue
            family = row.get("family", row.get("template"))
            key = (source.source, family)
            if (
                key not in candidates
                or row.get("shift", "none") != "none"
                or row.get("source", source.source) != source.source
                or row.get("template", family) != family
            ):
                raise ConfigError(
                    field="pilot.stratum", detail="unexpected validation source/family/shift"
                )
            relative = Path(row["path"])
            valid_origin = (
                relative.parts[:2] == ("train", family) and row.get("origin") == "train"
                if source.source == "ir2"
                else relative.name.startswith(f"validation_{family}_") and "template" in row
            )
            if not valid_origin or relative.is_absolute() or ".." in relative.parts:
                raise ConfigError(
                    field="pilot.source", detail="validation source metadata/path mismatch"
                )
            candidates[key].append((row, source))
    selected = []
    for key, choices in candidates.items():
        if not choices:
            raise ConfigError(
                field="pilot.stratum", detail=f"missing retained validation stratum: {key}"
            )
        row, source = min(choices, key=lambda item: (item[0]["occupancy_hash"], item[0]["path"]))
        if row["canonical_hash"] in forbidden:
            raise ConfigError(
                field="pilot.split", detail="validation winner overlaps test/OOD canonical group"
            )
        path = (source.root / row["path"]).resolve(strict=True)
        data = load_ir2_map(path)
        actual = {
            "occupancy_hash": obstacle_hash(data.occupancy),
            "canonical_hash": canonical_obstacle_hash(data.occupancy),
            "scenario_hash": scenario_hash(data.occupancy, data.anchor),
        }
        if any(row[name] != digest for name, digest in actual.items()):
            raise ConfigError(field="pilot.identity", detail=f"map content mismatch: {path}")
        shape = data.occupancy.shape
        required = max(shape)
        bucket = next((side for side in (512, 768, 1024) if side >= required), required)
        declared_bucket = int(row.get("bucket", row.get("size")))
        if declared_bucket != bucket or (
            "valid_shape" in row and tuple(row["valid_shape"]) != shape
        ):
            raise ConfigError(field="pilot.geometry", detail=f"native geometry mismatch: {path}")
        selected.append(
            SelectedMap(
                EvaluationMap(
                    str(path),
                    actual["occupancy_hash"],
                    "validation",
                    *key,
                    bucket,
                    actual["scenario_hash"],
                ),
                file_hash(path),
                actual["canonical_hash"],
                shape,
                str(source.manifest),
                file_hash(source.manifest),
                row["path"],
            )
        )
    return tuple(selected)


def plan_resource_pilot(
    registry: Path,
    config_path: Path,
    checkpoint: Path,
) -> tuple[tuple[SelectedMap, ...], tuple[EvaluationTask, ...]]:
    """Build exactly 240 limited tasks with independent faults and eight fixed methods."""
    base = load_config(config_path)
    expected = {
        "max_rounds": 512,
        "coverage_target": 0.99,
        "graph_capacity": 400,
        "rendezvous_mode": "reachable_only",
        "burst_start": 128,
    }
    if any(getattr(base, name) != value for name, value in expected.items()):
        raise ConfigError(
            field="pilot.config", detail="resource-pilot-v1 requires the main benchmark config"
        )
    selected = select_validation_maps(manifest_sources(registry))
    methods = tuple(
        make_baseline_method(
            motion,
            communication,
            checkpoint.resolve(strict=True) if motion == "ir2_adapted" else None,
        )
        for motion in ("nearest_frontier", "ir2_adapted")
        for communication in ("off", "plan", "map", "mixed")
    )
    scenarios = (
        make_scenario(
            item.map_data,
            replace(
                base,
                n_agents=4,
                agent_bucket=6,
                initial_budget=budget,
                fault_mode=mode,
                dropout_probability=probability,
                burst_duration=duration,
            ),
            0,
            limited=True,
        )
        for item in selected
        for budget in (2048, 8192)
        for mode, probability, duration in CONDITIONS
    )
    return selected, tuple(tasks(scenarios, methods))


def main() -> None:
    """Write the frozen plan and selection evidence after the final source pin only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("artifacts/maps/sources.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected, planned = plan_resource_pilot(args.sources, args.config, args.checkpoint)
    if args.output.exists() or args.output.with_suffix(".selection.json").exists():
        raise ConfigError(
            field="pilot.output", detail="refusing to overwrite an existing frozen plan"
        )
    count = write_task_manifest(args.output, planned)
    evidence = {
        "name": "resource-pilot-v1",
        "status": "planning_only",
        "limited": True,
        "task_count": count,
        "scenario_seed": 0,
        "history_capacity": 513,
        "selection_rule": "minimum occupancy_hash per retained validation stratum before scores",
        "selected_maps": [asdict(item) for item in selected],
        "registry_hash": file_hash(args.sources),
        "config_file_hash": file_hash(args.config),
        "checkpoint_hash": file_hash(args.checkpoint),
        "planner_hash": file_hash(Path(__file__)),
        "task_manifest_hash": file_hash(args.output),
        "implementations": planned[0].scenario.implementations,
    }
    args.output.with_suffix(".selection.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
