"""Frozen validation-only model selection shared by both learning routes."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from ir2_benchmark.config import ConfigError
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.report import family_balanced_mean, load_report_rows
from ir2_benchmark.evaluation.runner import run_manifest
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    identity,
    make_scenario,
    read_scenario_manifest,
    tasks,
    write_scenario_manifest,
    write_task_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.evaluation.scenarios import Scenario
    from ir2_benchmark.maps.sampler import ManifestSource


@dataclass(frozen=True, slots=True)
class ValidationScore:
    """Actual coverage-AUC score with its frozen manifest and executed result location."""

    mean_auc: float
    manifest_hash: str
    evaluated: int
    output: Path
    real_events: int


def build_validation_manifest(
    sources: tuple[ManifestSource, ...],
    config: HostConfig,
    output: Path,
    *,
    maps_per_family: int = 5,
    seed: int = 0,
) -> Path:
    """Choose hash-first maps from each validation family, one fixed scenario per map."""
    groups: dict[tuple[str, str], list[EvaluationMap]] = defaultdict(list)
    for source in sources:
        for line in source.manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") != "validation" or row.get("exclusion_reason") is not None:
                continue
            family = row.get("family", row.get("template"))
            groups[source.source, family].append(
                EvaluationMap(
                    str((source.root / row["path"]).resolve()),
                    row["occupancy_hash"],
                    "validation",
                    source.source,
                    family,
                    int(row.get("bucket", row.get("size"))),
                    row["scenario_hash"],
                )
            )
    required = (
        ("ir2", "easy"),
        ("ir2", "difficult"),
        ("procedural", "rooms"),
        ("procedural", "corridor"),
        ("procedural", "hybrid"),
    )
    selected = []
    for key in required:
        candidates = sorted(groups[key], key=lambda row: row.map_hash)
        if len(candidates) < maps_per_family:
            raise ConfigError(
                field="validation", detail=f"insufficient declared validation maps:{key}"
            )
        selected.extend(candidates[:maps_per_family])
    scenarios = tuple(
        make_scenario(row, replace(config, n_agents=(2, 4, 6)[index % 3], agent_bucket=6), seed)
        for index, row in enumerate(selected)
    )
    write_scenario_manifest(output, scenarios)
    selection = {
        "rule": "first occupancy hashes within each of five source/family strata",
        "maps_per_family": maps_per_family,
        "seed": seed,
        "scenarios": len(scenarios),
        "split": "validation",
        "manifest_hash": file_hash(output),
        "base_config": asdict(config),
        "search_budget_shared_across_methods": True,
    }
    output.with_suffix(".selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    return output


def validate_manifest(path: Path) -> tuple[Scenario, ...]:
    """Reject any test/training row or changed config before training/model selection starts."""
    scenarios = tuple(read_scenario_manifest(path))
    if not scenarios:
        raise ConfigError(field="validation", detail="empty fixed validation manifest")
    for scenario in scenarios:
        if scenario.split != "validation" or scenario.config_hash != identity(
            asdict(scenario.config)
        ):
            raise ConfigError(
                field="validation",
                detail="requires validation-only rows and intact configuration hashes",
            )
    return scenarios


def evaluate_validation(
    checkpoint: Path, manifest: Path, output: Path, *, real_events: int, training_seed: int = 0
) -> ValidationScore:
    """Execute fixed validation; runtime failures never become selection scores."""
    from ir2_benchmark.commands.evaluate import checkpoint_method  # noqa: PLC0415

    scenarios = validate_manifest(manifest)
    method, _ = checkpoint_method(checkpoint, training_seed=training_seed, real_events=real_events)
    destination = output / f"events-{real_events}-{method.checkpoint_hash[:12]}"
    workload = destination / "tasks.jsonl"
    write_task_manifest(workload, tasks(scenarios, (method,)))
    execution = run_manifest(workload, destination)
    if execution.failures:
        raise ConfigError(
            field="validation", detail=f"validation execution failures:{execution.failures}"
        )
    rows, corrupt = load_report_rows(destination)
    score = family_balanced_mean(rows)
    if corrupt or len(rows) != len(scenarios) or score is None:
        raise ConfigError(
            field="validation", detail="incomplete validation cannot select a checkpoint"
        )
    result = ValidationScore(score, file_hash(manifest), len(rows), destination, real_events)
    payload = asdict(result) | {"output": str(result.output)}
    (destination / "validation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return result


@dataclass(frozen=True, slots=True)
class ValidationEvaluator:
    """A frozen selection workload callable, shared by A/B checkpoint schedules."""

    manifest: Path
    output: Path
    training_seed: int
    manifest_hash: str

    def __call__(self, checkpoint: Path, real_events: int) -> ValidationScore:
        """Return actual validation measurements and reject a changed selection workload."""
        if file_hash(self.manifest) != self.manifest_hash:
            raise ConfigError(
                field="validation", detail="fixed validation manifest changed during training"
            )
        return evaluate_validation(
            checkpoint,
            self.manifest,
            self.output,
            real_events=real_events,
            training_seed=self.training_seed,
        )


def build_validation_callback(
    manifest: Path, output: Path, *, training_seed: int = 0
) -> ValidationEvaluator:
    """Validate and freeze one model-selection workload before either training route begins."""
    validate_manifest(manifest)
    return ValidationEvaluator(manifest, output, training_seed, file_hash(manifest))
