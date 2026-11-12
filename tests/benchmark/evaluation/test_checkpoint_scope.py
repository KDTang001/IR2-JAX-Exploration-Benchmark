"""A trained repetition retains one model identity across all maps and diagnostic conditions."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from test_auxiliary_report import save_hand_calculated_result
from test_workload_scope import audit
from test_workload_scope import case as case  # noqa: PLC0414 - Register the shared pytest fixture.

from ir2_benchmark.evaluation.report import write_report
from ir2_benchmark.evaluation.scenarios import make_scenario, tasks, write_task_manifest
from ir2_benchmark.evaluation.workload import WorkloadAudit

if TYPE_CHECKING:
    from pathlib import Path

    from test_workload_scope import Case


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_hash", "other-weights"),
        ("route", "B"),
        ("communication", "fixed"),
        ("estimator", "zoh"),
        ("real_training_events", 1999999),
        ("decision_rule", "greedy"),
        ("selected_at_events", 1000000),
    ],
)
def test_each_non_path_method_field_is_fixed_across_maps(
    case: Case, field: str, value: str | int
) -> None:
    # Given: a valid second-map task generated from another identity of the same trained repetition.
    changed = list(case.workload)
    original = changed[4]
    method = replace(original.method, **{field: value})
    changed[4] = next(tasks((original.scenario,), (method,)))
    # When: all task/scenario IDs remain individually valid.
    proof = audit(case, tuple(changed))
    # Then: a per-map checkpoint or metadata change cannot count as one trained model.
    assert proof.invalid_core_tasks == 1
    assert not proof.core_scope_complete


@pytest.mark.parametrize(
    ("field", "value"), [("route", "B"), ("communication", "fixed"), ("estimator", "zoh")]
)
def test_explicit_core_method_declaration_is_required(case: Case, field: str, value: str) -> None:
    # Given: a complete workload whose method metadata contradicts the explicit core declaration.
    case.declaration["methods"][0][field] = value
    case.core.write_text(json.dumps(case.declaration), encoding="utf-8")
    # When: every map and scenario seed is still present.
    proof = audit(case, case.workload)
    # Then: matching labels alone cannot satisfy a different declared policy variant.
    assert proof.invalid_core_tasks == 4  # noqa: PLR2004
    assert not proof.core_scope_complete


@pytest.mark.parametrize("mixed_checkpoint", [True, False])
def test_real_report_rejects_mixed_checkpoints_but_allows_same_hash_relocated_path(
    case: Case,
    tmp_path: Path,
    *,
    mixed_checkpoint: bool,
) -> None:
    # Given: real generated task IDs and checksummed hand-calculated report fixtures for both maps.
    changed = list(case.workload)
    original = changed[4]
    method = replace(
        original.method,
        checkpoint="relocated-model",
        checkpoint_hash="other-weights" if mixed_checkpoint else original.method.checkpoint_hash,
    )
    changed[4] = next(tasks((original.scenario,), (method,)))
    for task in changed:
        save_hand_calculated_result(tmp_path, task, 0.25)
    manifest = tmp_path / "tasks.jsonl"
    write_task_manifest(manifest, changed)
    # When: the production writer verifies scope, results, traces and model repetition identity.
    path = write_report(
        tmp_path,
        tmp_path / "report",
        manifest=manifest,
        core=case.core,
        map_sources=case.registry,
        repetitions=5,
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    # Then: only the coherent model identity can complete the declared core matrix.
    assert report["formal_complete"] is (not mixed_checkpoint)
    assert report["workload_scope"]["invalid_core_tasks"] == int(mixed_checkpoint)


def test_same_repetition_cannot_switch_weights_between_core_and_unlimited(case: Case) -> None:
    # Given: an unlimited diagnostic that reuses the declared core method and training repetition.
    declaration = {
        **case.declaration,
        "auxiliary_matrix": [
            *case.declaration["auxiliary_matrix"],
            {
                "category": "unlimited_budget",
                "budget_mode": "unlimited",
                "budgets": [],
                "method_ids": ["A_learned_predict"],
                "conditions": [[0, 0, 64]],
            },
        ],
    }
    case.core.write_text(json.dumps(declaration), encoding="utf-8")
    scene = make_scenario(
        case.maps[0], replace(case.base, initial_budget=512), 0, budget_mode="unlimited"
    )
    method = replace(case.workload[0].method, checkpoint_hash="other-weights")
    changed = next(tasks((scene,), (method,)))
    # When: the valid core rows and individually valid diagnostic use different model identities.
    state = WorkloadAudit(case.registry, declaration, case.core)
    for task in (*case.workload, changed):
        state.observe(task)
    # Then: the diagnostic is invalid despite being in a distinct budget mode and auxiliary scope.
    proof = state.finish()
    assert proof.core_scope_complete
    assert proof.invalid_auxiliary_tasks == 1


def test_real_report_rejects_route_that_disagrees_with_explicit_core_declaration(
    case: Case,
    tmp_path: Path,
) -> None:
    # Given: generated route-A tasks while the same named core method explicitly declares route B.
    case.declaration["methods"][0]["route"] = "B"
    case.core.write_text(json.dumps(case.declaration), encoding="utf-8")
    for task in case.workload:
        save_hand_calculated_result(tmp_path, task, 0.25)
    manifest = tmp_path / "tasks.jsonl"
    write_task_manifest(manifest, case.workload)
    # When: the production report validates the declaration as well as task/result identities.
    path = write_report(
        tmp_path,
        tmp_path / "report",
        manifest=manifest,
        core=case.core,
        map_sources=case.registry,
        repetitions=5,
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    # Then: coverage alone cannot certify the incorrectly named comparison.
    assert not report["formal_complete"]
    assert report["workload_scope"]["invalid_core_tasks"] == 4  # noqa: PLR2004
