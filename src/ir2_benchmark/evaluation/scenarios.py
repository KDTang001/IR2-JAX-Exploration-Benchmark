"""Immutable paired scenarios and checkpoint-bound task manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from ir2_benchmark.config import FAULT_BURST, HostConfig
from ir2_benchmark.evaluation.budgets import BudgetMode, allocate_budget
from ir2_benchmark.presets import preset_manifest, verify_preset_manifest

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from ir2_benchmark.maps.sampler import ManifestSource


JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


def identity(value: JsonValue) -> str:
    """Hash canonical JSON for portable manifest/configuration identities."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationMap:
    """An explicit retained test/OOD map; no failed input is substituted."""

    path: str
    map_hash: str
    split: str
    source: str
    family: str
    bucket: int
    scenario_hash: str


@dataclass(frozen=True, slots=True)
class Scenario:
    """Environment identity shared by all methods, independent of their run order."""

    scenario_id: str
    map_path: str
    map_hash: str
    split: str
    source: str
    family: str
    bucket: int
    spawn_seed: int
    fault_seed: int
    n_agents: int
    budget: int
    fault_mode: int
    drop_probability: float
    isolation_ids: tuple[int, ...]
    burst_start: int
    burst_duration: int
    max_rounds: int
    config_hash: str
    config: HostConfig
    map_scenario_hash: str
    limited: bool = False
    budget_mode: BudgetMode = "per_robot"
    requested_team_bytes: int | None = None
    unallocated_bytes: int = 0
    implementations: dict | None = None


@dataclass(frozen=True, slots=True)
class Method:
    """One trained repetition, including the actual checkpoint identity."""

    method_id: str
    route: Literal["A", "B", "heuristic", "baseline"]
    communication: str
    estimator: str
    training_seed: int
    checkpoint: str
    checkpoint_hash: str
    real_training_events: int
    decision_rule: str = "stochastic_with_fixed_seed"
    selected_at_events: int | None = None


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    """A unique paired execution, suitable for resumable independent dispatch."""

    result_id: str
    scenario: Scenario
    method: Method


def evaluation_maps(sources: tuple[ManifestSource, ...]) -> tuple[EvaluationMap, ...]:
    """Select only explicitly retained original/procedural test and OOD entries."""
    maps = []
    for source in sources:
        for line in source.manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") not in ("test", "ood") or row.get("exclusion_reason") is not None:
                continue
            family = row.get("family", row.get("template"))
            if row.get("shift", "none") != "none":
                family = f"{family}:{row['shift']}"
            maps.append(
                EvaluationMap(
                    str((source.root / row["path"]).resolve()),
                    row["occupancy_hash"],
                    row["split"],
                    source.source,
                    family,
                    int(row.get("bucket", row.get("size"))),
                    row["scenario_hash"],
                )
            )
    return tuple(sorted(maps, key=lambda row: (row.source, row.family, row.map_hash)))


def make_scenario(  # noqa: PLR0913
    map_data: EvaluationMap,
    config: HostConfig,
    scenario_seed: int,
    *,
    limited: bool = False,
    budget_mode: BudgetMode = "per_robot",
    requested_team_bytes: int | None = None,
    unallocated_bytes: int = 0,
) -> Scenario:
    """Derive spawn/fault seeds from map identity and scenario counter, never a global RNG."""
    digest = bytes.fromhex(identity([map_data.map_hash, map_data.scenario_hash, scenario_seed]))
    spawn_seed, fault_seed = (int(value) & 0x7FFFFFFF for value in np.frombuffer(digest[:8], "<u4"))
    ranks = np.random.default_rng(fault_seed).permutation(config.n_agents)
    isolated = tuple(sorted(int(value) for value in ranks[: config.n_agents // 2]))
    payload = {
        "map_hash": map_data.map_hash,
        "map_scenario_hash": map_data.scenario_hash,
        "split": map_data.split,
        "spawn_seed": spawn_seed,
        "fault_seed": fault_seed,
        "config": asdict(config),
        "implementations": preset_manifest(config),
    }
    if budget_mode != "per_robot":
        payload.update(
            budget_mode=budget_mode,
            requested_team_bytes=requested_team_bytes,
            unallocated_bytes=unallocated_bytes,
        )
    return Scenario(
        identity(payload),
        map_data.path,
        map_data.map_hash,
        map_data.split,
        map_data.source,
        map_data.family,
        map_data.bucket,
        spawn_seed,
        fault_seed,
        config.n_agents,
        config.initial_budget,
        config.fault_mode,
        config.dropout_probability,
        isolated if config.fault_mode == FAULT_BURST else (),
        config.burst_start,
        config.burst_duration,
        config.max_rounds,
        identity(asdict(config)),
        config,
        map_data.scenario_hash,
        limited,
        budget_mode,
        requested_team_bytes,
        unallocated_bytes,
        preset_manifest(config),
    )


def scenario_matrix(  # noqa: PLR0913
    maps: Iterable[EvaluationMap],
    base: HostConfig,
    *,
    seeds: tuple[int, ...],
    n_agents: tuple[int, ...],
    budgets: tuple[int, ...],
    conditions: tuple[tuple[int, float, int], ...],
    budget_mode: BudgetMode = "per_robot",
) -> Iterator[Scenario]:
    """Stream every declared map/seed/N/budget/condition without silently limiting the matrix."""
    for map_data in maps:
        for seed in seeds:
            for agents in n_agents:
                for budget in (0,) if budget_mode == "unlimited" else budgets:
                    allocation = allocate_budget(budget_mode, budget, agents, base.max_rounds)
                    for mode, probability, duration in conditions:
                        config = replace(
                            base,
                            n_agents=agents,
                            agent_bucket=6,
                            initial_budget=allocation.per_robot,
                            fault_mode=mode,
                            dropout_probability=probability,
                            burst_duration=duration,
                        )
                        yield make_scenario(
                            map_data,
                            config,
                            seed,
                            budget_mode=budget_mode,
                            requested_team_bytes=allocation.requested_team_bytes,
                            unallocated_bytes=allocation.unallocated_bytes,
                        )


def tasks(scenarios: Iterable[Scenario], methods: tuple[Method, ...]) -> Iterator[EvaluationTask]:
    """Keep method iteration out of scenario RNG and bind each result to real weights."""
    for scenario in scenarios:
        for method in methods:
            selected_method = (
                replace(method, decision_rule="heuristic")
                if method.route == "heuristic"
                else method
            )
            method_identity = {
                name: value
                for name, value in asdict(selected_method).items()
                if name != "checkpoint"
            }
            result_id = identity([scenario.scenario_id, method_identity])
            yield EvaluationTask(result_id, scenario, selected_method)


def write_task_manifest(path: Path, planned: Iterable[EvaluationTask]) -> int:
    """Persist the complete requested workload before evaluation begins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for task in planned:
            stream.write(json.dumps(asdict(task), sort_keys=True) + "\n")
            count += 1
    return count


def read_task_manifest(path: Path) -> Iterator[EvaluationTask]:
    """Restore typed scenario/configuration records from the exact persisted workload."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            scenario = row["scenario"]
            scenario["config"] = HostConfig(**scenario["config"])
            verify_preset_manifest(scenario["config"], scenario.get("implementations"))
            scenario["isolation_ids"] = tuple(scenario["isolation_ids"])
            yield EvaluationTask(row["result_id"], Scenario(**scenario), Method(**row["method"]))


def write_scenario_manifest(path: Path, scenarios: Iterable[Scenario]) -> int:
    """Freeze policy-independent validation or evaluation scenarios before model selection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for scenario in scenarios:
            stream.write(json.dumps(asdict(scenario), sort_keys=True) + "\n")
            count += 1
    return count


def read_scenario_manifest(path: Path) -> Iterator[Scenario]:
    """Restore a model-independent frozen scenario sequence."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row["config"] = HostConfig(**row["config"])
            verify_preset_manifest(row["config"], row.get("implementations"))
            row["isolation_ids"] = tuple(row["isolation_ids"])
            yield Scenario(**row)
