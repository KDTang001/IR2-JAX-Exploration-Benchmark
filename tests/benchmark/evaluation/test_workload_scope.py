"""Independent retained-map pools and declared seeds are required for a complete workload."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import pytest

from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    EvaluationTask,
    Method,
    evaluation_maps,
    make_scenario,
    scenario_matrix,
    tasks,
)
from ir2_benchmark.evaluation.workload import WorkloadAudit, WorkloadScope

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Case:
    """Two independently declared map identities and two scenario seeds in each scope."""

    registry: Path
    core: Path
    declaration: dict
    base: HostConfig
    maps: tuple[EvaluationMap, ...]
    workload: tuple[EvaluationTask, ...]


@pytest.fixture
def case(tmp_path: Path) -> Case:
    base = HostConfig(agent_bucket=6, sensor_radius=1, max_distance=1, max_rounds=2)
    (tmp_path / "base.json").write_text(json.dumps(asdict(base)), encoding="utf-8")
    sources = []
    for index, (source, split) in enumerate((("ir2", "test"), ("procedural", "ood"))):
        path = tmp_path / f"maps-{index}.jsonl"
        row = {
            "path": f"map-{index}.png",
            "split": split,
            "family": "rooms",
            "bucket": 64,
            "occupancy_hash": f"map-{index}",
            "scenario_hash": f"anchor-{index}",
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        sources.append({"source": source, "root": ".", "manifest": path.name})
    registry = tmp_path / "sources.json"
    registry.write_text(json.dumps(sources), encoding="utf-8")
    declaration = {
        "base_config": "base.json",
        "scenario_seeds": [0, 1],
        "methods": [{"method_id": "A_learned_predict"}],
        "training_seeds": [0],
        "n_agents": [2],
        "budgets": [8192],
        "conditions": [[0, 0, 64]],
        "auxiliary": ["heuristic"],
        "auxiliary_matrix": [
            {
                "category": "heuristic",
                "method_ids": ["heuristic"],
                "training_seeds": [-1],
                "budgets": [8192],
                "conditions": [[0, 0, 64]],
            }
        ],
    }
    core = tmp_path / "core.json"
    core.write_text(json.dumps(declaration), encoding="utf-8")
    maps = evaluation_maps(manifest_sources(registry))
    scenes = scenario_matrix(
        maps, base, seeds=(0, 1), n_agents=(2,), budgets=(8192,), conditions=((0, 0, 64),)
    )
    methods = (
        Method("A_learned_predict", "A", "learned", "predict", 0, "model", "weights", 2000000),
        Method("heuristic", "heuristic", "fixed", "none", -1, "", "", 0),
    )
    return Case(registry, core, declaration, base, maps, tuple(tasks(scenes, methods)))


def audit(case: Case, workload: tuple[EvaluationTask, ...]) -> WorkloadScope:
    """Feed only actual provided rows through the same streaming report audit."""
    state = WorkloadAudit(case.registry, case.declaration, case.core)
    for task in workload:
        state.observe(task)
    return state.finish()


def test_full_small_declaration_proves_both_scopes_and_records_independent_hashes(
    case: Case,
) -> None:
    # Given: all two maps, two scenario seeds and both declared method conditions.
    # When: the task rows are checked against their independently loaded registry.
    proof = audit(case, case.workload)
    # Then: every declared bit is covered, with source/baseline evidence persisted.
    assert proof.status == "verified"
    assert proof.registry_hash == file_hash(case.registry)
    assert proof.base_config_hash == file_hash(case.core.parent / "base.json")
    assert proof.retained_maps == 2  # noqa: PLR2004
    assert len(proof.source_manifest_hashes) == 2  # noqa: PLR2004
    assert (proof.expected_core_blocks, proof.expected_auxiliary_blocks) == (4, 4)
    assert (proof.missing_core_blocks, proof.missing_auxiliary_blocks) == (0, 0)
    assert proof.core_scope_complete
    assert proof.auxiliary_scope_complete


def test_one_map_cannot_stand_in_for_the_declared_full_pool(case: Case) -> None:
    # Given: every condition is present, but all rows from the OOD map were omitted.
    subset = tuple(task for task in case.workload if task.scenario.split == "test")
    # When: the independent pool still contains both retained maps.
    proof = audit(case, subset)
    # Then: map omissions are counted in both scopes and shown as bounded named examples.
    assert (proof.missing_core_blocks, proof.missing_auxiliary_blocks) == (2, 2)
    assert not proof.core_scope_complete
    assert not proof.auxiliary_scope_complete
    assert "ood" in proof.core_examples[0]
    assert "map-1" in proof.auxiliary_examples[0]


def test_missing_scenario_seed_cannot_be_replaced_by_another_seed_or_duplicate(case: Case) -> None:
    # Given: one actual spawn/fault pair per map, duplicated to preserve the original row count.
    subset = tuple(task for index, task in enumerate(case.workload) if index % 4 < 2)  # noqa: PLR2004
    # When: duplicate rows cannot mark the other declared seed's bit.
    proof = audit(case, subset + subset)
    # Then: the missing seed stays missing independently for each map and method scope.
    assert (proof.missing_core_blocks, proof.missing_auxiliary_blocks) == (2, 2)
    assert "scenario_seed=1" in proof.core_examples[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "ood"),
        ("scenario_hash", "wrong-anchor"),
        ("source", "procedural"),
        ("family", "other"),
    ],
)
def test_wrong_map_origin_or_anchor_cannot_fill_the_registered_map_bit(
    case: Case, field: str, value: str
) -> None:
    # Given: a well-formed task for an undeclared map identity replacing one core task.
    changed = replace(case.maps[0], **{field: value})
    scene = make_scenario(changed, case.base, 0)
    altered = next(tasks((scene,), (case.workload[0].method,)))
    # When: rows are matched by occupancy, anchor, split, source and family together.
    proof = audit(case, (altered, *case.workload[1:]))
    # Then: the wrong identity is invalid and leaves one expected core block unfilled.
    assert (proof.invalid_core_tasks, proof.missing_core_blocks) == (1, 1)
    assert not proof.core_scope_complete
    assert proof.auxiliary_scope_complete


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_rounds", 1),
        ("sensor_radius", 2),
        ("max_distance", 2),
        ("max_candidates", 3),
        ("agent_bucket", 4),
    ],
)
def test_baseline_geometry_and_slots_are_verified_independently(
    case: Case, field: str, value: int
) -> None:
    # Given: genuine IDs generated from an altered geometry rather than the declared baseline.
    changed = replace(case.base, **{field: value})
    scene = make_scenario(case.maps[0], changed, 0)
    altered = next(tasks((scene,), (case.workload[0].method,)))
    # When: the condition labels otherwise match every declared axis.
    proof = audit(case, (altered, *case.workload[1:]))
    # Then: altered geometry and slot counts cannot certify the declared baseline.
    assert proof.invalid_core_tasks == 1
    assert not proof.core_scope_complete


def test_normal_ignores_irrelevant_burst_start_and_duration_metadata(case: Case) -> None:
    # Given: Normal tasks with changed unused burst scheduling metadata and recomputed genuine IDs.
    changed = []
    for task in case.workload:
        data = next(data for data in case.maps if data.map_hash == task.scenario.map_hash)
        original = task.scenario
        seed = next(
            seed
            for seed in (0, 1)
            if make_scenario(data, case.base, seed).spawn_seed == original.spawn_seed
        )
        scene = make_scenario(data, replace(case.base, burst_start=777, burst_duration=777), seed)
        changed.append(next(tasks((scene,), (task.method,))))
    # When: only physically relevant baseline/configuration fields are compared.
    proof = audit(case, tuple(changed))
    # Then: irrelevant Normal metadata does not manufacture a missing condition.
    assert proof.core_scope_complete
    assert proof.auxiliary_scope_complete


def test_unverified_registry_or_baseline_never_claims_complete(case: Case) -> None:
    # Given: old report inputs without independent registry or baseline provenance.
    absent = WorkloadAudit(None, case.declaration, case.core).finish()
    no_base = WorkloadAudit(
        case.registry,
        {key: value for key, value in case.declaration.items() if key != "base_config"},
        case.core,
    ).finish()
    # When: statistics may still be written from those old artifacts.
    # Then: neither source can claim verified core or auxiliary coverage.
    assert absent.status == "unverified_scope"
    assert not absent.core_scope_complete
    assert no_base.status == "unverified_base_config"
    assert not no_base.auxiliary_scope_complete


def test_forged_duplicate_task_id_cannot_fill_a_different_maps_block(case: Case) -> None:
    # Given: a correct second-map scenario carrying a copied first-map result ID.
    altered = replace(case.workload[4], result_id=case.workload[0].result_id)
    # When: coverage validates the existing scenario/task identity formula as well as map fields.
    proof = audit(case, (*case.workload[:4], altered, *case.workload[5:]))
    # Then: a duplicate result cannot stand in for a distinct map's execution.
    assert (proof.invalid_core_tasks, proof.missing_core_blocks) == (1, 1)


def test_underfunded_unlimited_label_cannot_satisfy_the_declared_budget_diagnostic(
    case: Case,
) -> None:
    # Given: a newly declared unlimited block but an actual one-byte starting balance.
    declaration = {
        **case.declaration,
        "auxiliary_matrix": [
            *case.declaration["auxiliary_matrix"],
            {
                "category": "unlimited_budget",
                "budget_mode": "unlimited",
                "method_ids": ["A_learned_predict"],
                "budgets": [],
                "conditions": [[0, 0, 64]],
            },
        ],
    }
    case.core.write_text(json.dumps(declaration), encoding="utf-8")
    scene = make_scenario(
        case.maps[0], replace(case.base, initial_budget=1), 0, budget_mode="unlimited"
    )
    task = next(tasks((scene,), (case.workload[0].method,)))
    state = WorkloadAudit(case.registry, declaration, case.core)
    for actual in (*case.workload, task):
        state.observe(actual)
    # Then: a correct tag/ID cannot hide a binding ledger balance in this diagnostic.
    proof = state.finish()
    assert proof.invalid_auxiliary_tasks == 1
    assert proof.core_scope_complete
    assert not proof.auxiliary_scope_complete
