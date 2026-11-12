"""Verify full map/scenario coverage against an independent registry with compact host bitsets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import FAULT_BURST, HostConfig, load_config
from ir2_benchmark.evaluation import scope
from ir2_benchmark.evaluation.budgets import (
    BudgetAllocation,
    BudgetMode,
    allocate_budget,
    auxiliary_kind,
)
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Scenario,
    evaluation_maps,
    identity,
    make_scenario,
    tasks,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ir2_benchmark.evaluation.scenarios import EvaluationTask, Method

MapIdentity: TypeAlias = tuple[str, str, str, str, str, int]


@dataclass(frozen=True, slots=True)
class WorkloadScope:
    """Independent input identities, missing block counts and bounded examples."""

    status: str = "unverified_scope"
    registry_path: str | None = None
    registry_hash: str | None = None
    declaration_hash: str | None = None
    source_manifest_hashes: tuple[tuple[str, str, str], ...] = ()
    retained_maps: int = 0
    retained_map_hash: str | None = None
    scenario_seeds: tuple[int, ...] = ()
    base_config_hash: str | None = None
    base_config_path: str | None = None
    expected_core_blocks: int | None = None
    expected_auxiliary_blocks: int | None = None
    missing_core_blocks: int | None = None
    missing_auxiliary_blocks: int | None = None
    invalid_core_tasks: int = 0
    invalid_auxiliary_tasks: int = 0
    core_examples: tuple[str, ...] = ()
    auxiliary_examples: tuple[str, ...] = ()
    core_scope_complete: bool = False
    auxiliary_scope_complete: bool = False


def map_identity(data: EvaluationMap) -> MapIdentity:
    """Include origin, family, split, anchor and native bucket rather than occupancy alone."""
    return data.map_hash, data.scenario_hash, data.split, data.source, data.family, data.bucket


@lru_cache(maxsize=256)
def geometry_matches(actual: HostConfig, base: HostConfig) -> bool:
    """Allow declared axes and ignore burst-start metadata outside Burst conditions."""
    return (
        replace(
            actual,
            n_agents=base.n_agents,
            initial_budget=base.initial_budget,
            fault_mode=base.fault_mode,
            dropout_probability=base.dropout_probability,
            burst_duration=base.burst_duration,
            burst_start=actual.burst_start
            if actual.fault_mode == FAULT_BURST
            else base.burst_start,
        )
        == base
    )


@lru_cache(maxsize=256)
def expected_scene(
    data: EvaluationMap,
    config: HostConfig,
    seed: int,
    mode: BudgetMode,
    allocation: BudgetAllocation,
) -> Scenario:
    """Reuse production seed/ID semantics while keeping only a small metadata window."""
    return make_scenario(
        data,
        config,
        seed,
        budget_mode=mode,
        requested_team_bytes=allocation.requested_team_bytes,
        unallocated_bytes=allocation.unallocated_bytes,
    )


class WorkloadAudit:
    """A streaming mutable bitset auditor stores no per-task or device-model objects."""

    __slots__ = (
        "_auxiliary",
        "_base",
        "_bits",
        "_conditions",
        "_core",
        "_declared_methods",
        "_invalid",
        "_lookup",
        "_maps",
        "_methods",
        "_proof",
        "_seeds",
    )

    def __init__(self, registry: Path | None, declaration: dict, core: Path | None) -> None:
        """Load the independently declared retained pool and baseline before reading task rows."""
        self._proof = WorkloadScope(declaration_hash=file_hash(core) if core is not None else None)
        self._base: HostConfig | None = None
        self._maps: tuple[EvaluationMap, ...] = ()
        self._methods: dict[tuple[str, int], Method] = {}
        self._declared_methods = scope.declared_method_fields(declaration)
        self._seeds = tuple(declaration.get("scenario_seeds", ()))
        self._lookup: dict[tuple[MapIdentity, int, int], int] = {}
        self._conditions = {
            key: index
            for index, key in enumerate(
                sorted(
                    set(scope.core_conditions(declaration))
                    | {key for _, key in scope.auxiliary_conditions(declaration)}
                )
            )
        }
        self._core = set(scope.core_conditions(declaration))
        self._auxiliary = {key for _, key in scope.auxiliary_conditions(declaration)}
        self._invalid = [0, 0]
        self._bits = np.empty((0, 0), np.uint8)
        if registry is None:
            return
        sources = manifest_sources(registry)
        self._maps = evaluation_maps(sources)
        self._proof = replace(
            self._proof,
            registry_path=str(registry.resolve()),
            registry_hash=file_hash(registry),
            source_manifest_hashes=tuple(
                (source.source, str(source.manifest), file_hash(source.manifest))
                for source in sources
            ),
            retained_maps=len(self._maps),
            scenario_seeds=self._seeds,
            retained_map_hash=identity([list(map_identity(data)) for data in self._maps]),
        )
        if not self._maps or not self._seeds:
            self._proof = replace(self._proof, status="unverified_map_or_seed_scope")
            return
        if core is None or "base_config" not in declaration:
            self._proof = replace(self._proof, status="unverified_base_config")
            return
        baseline = core.parent / declaration["base_config"]
        self._base = load_config(baseline)
        self._proof = replace(
            self._proof,
            status="verified",
            base_config_hash=file_hash(baseline),
            base_config_path=str(baseline.resolve()),
        )
        for index, data in enumerate(self._maps):
            for offset, seed in enumerate(self._seeds):
                scene = make_scenario(data, self._base, seed)
                self._lookup[map_identity(data), scene.spawn_seed, scene.fault_seed] = (
                    index * len(self._seeds) + offset
                )
        width = len(self._maps) * len(self._seeds)
        self._bits = np.zeros((len(self._conditions), (width + 7) // 8), np.uint8)

    def observe(self, task: EvaluationTask) -> None:
        """Validate an actual manifest row before marking one expected map/seed bit."""
        if self._base is None:
            return
        scene, condition = task.scenario, scope.condition_for_task(task)
        auxiliary = scene.budget_mode != "per_robot" or bool(auxiliary_kind(task.method))
        signature = (
            scene.map_hash,
            scene.map_scenario_hash,
            scene.split,
            scene.source,
            scene.family,
            scene.bucket,
        )
        column = self._lookup.get((signature, scene.spawn_seed, scene.fault_seed))
        if (
            column is None
            or condition not in self._conditions
            or scene.limited
            or not geometry_matches(scene.config, self._base)
        ):
            self._invalid[auxiliary] += 1
            return
        index, offset = divmod(column, len(self._seeds))
        allocation = allocate_budget(
            scene.budget_mode, condition.budget or 0, scene.n_agents, self._base.max_rounds
        )
        if scene.config.initial_budget != allocation.per_robot:
            self._invalid[auxiliary] += 1
            return
        expected = expected_scene(
            self._maps[index], scene.config, self._seeds[offset], scene.budget_mode, allocation
        )
        canonical = next(tasks((expected,), (task.method,)))
        if (
            replace(expected, map_path=scene.map_path) != scene
            or canonical.result_id != task.result_id
            or canonical.method != task.method
        ):
            self._invalid[auxiliary] += 1
            return
        method = replace(task.method, checkpoint="")
        key = method.method_id, method.training_seed
        if (
            any(
                getattr(method, field) != value
                for field, value in self._declared_methods.get(method.method_id, ())
            )
            or self._methods.setdefault(key, method) != method
        ):
            self._invalid[auxiliary] += 1
            return
        row = self._conditions[condition]
        self._bits[row, column // 8] |= np.uint8(1 << (column % 8))

    def finish(self) -> WorkloadScope:
        """Count missing cells and retain at most four examples for each independent scope."""
        if self._base is None:
            return self._proof
        width = len(self._maps) * len(self._seeds)
        counts, examples = [0, 0], [[], []]
        for key, row in self._conditions.items():
            absent = np.flatnonzero(np.unpackbits(self._bits[row], bitorder="little")[:width] == 0)
            for role, required in enumerate((self._core, self._auxiliary)):
                if key not in required:
                    continue
                counts[role] += len(absent)
                for column in absent[: max(0, 4 - len(examples[role]))]:
                    index, offset = divmod(int(column), len(self._seeds))
                    examples[role].append(
                        f"{key}:map={map_identity(self._maps[index])}:scenario_seed={self._seeds[offset]}"
                    )
        return replace(
            self._proof,
            expected_core_blocks=len(self._core) * width,
            expected_auxiliary_blocks=len(self._auxiliary) * width,
            missing_core_blocks=counts[0],
            missing_auxiliary_blocks=counts[1],
            invalid_core_tasks=self._invalid[0],
            invalid_auxiliary_tasks=self._invalid[1],
            core_examples=tuple(examples[0]),
            auxiliary_examples=tuple(examples[1]),
            core_scope_complete=counts[0] == 0 and self._invalid[0] == 0,
            auxiliary_scope_complete=counts[1] == 0 and self._invalid[1] == 0,
        )
