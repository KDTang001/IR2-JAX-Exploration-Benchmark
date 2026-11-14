"""Diagnostics cannot inflate core rankings or silently satisfy omitted auxiliary workloads."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.budgets import result_budget_fields
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.report import write_report
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    identity,
    make_scenario,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.evaluation.scope import invalid_auxiliary_quotas, missing_auxiliary_conditions
from ir2_benchmark.evaluation.statistics import ReportRow
from ir2_benchmark.evaluation.tables import write_cost_tables

if TYPE_CHECKING:
    from pathlib import Path

    from ir2_benchmark.evaluation.scenarios import EvaluationTask


def save_hand_calculated_result(root: Path, task: EvaluationTask, auc: float) -> None:
    """Persist a small metric fixture with an actual trace/checksum, without executing a model."""
    trace = root / f"{task.result_id}.npz"
    np.savez(trace, coverage=np.full(4, auc), plan_has_target=np.zeros((4, 2), bool))
    scene, method = task.scenario, task.method
    raw = {
        "result_id": task.result_id,
        "method_id": method.method_id,
        "route": method.route,
        "training_seed": method.training_seed,
        "source": scene.source,
        "family": scene.family,
        "map_hash": scene.map_hash,
        "map_scenario_hash": scene.map_scenario_hash,
        "split": scene.split,
        "n_agents": scene.n_agents,
        "budget": scene.budget,
        "outcome": "timeout",
        "metrics": {"auc": auc, "final_coverage": auc},
        "trace_file": str(trace),
        "trace_hash": file_hash(trace),
        "checkpoint_hash": method.checkpoint_hash,
        "configuration_hash": scene.config_hash,
        "plan_diagnostics": True,
        "decision_rule": method.decision_rule,
        "real_training_events": method.real_training_events,
        **result_budget_fields(scene, method),
    }
    path = root / "results" / f"{task.result_id}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"result": raw, "checksum": identity(raw)}), encoding="utf-8")


def test_missing_auxiliary_block_keeps_formal_incomplete_without_polluting_core_rank(
    tmp_path: Path,
) -> None:
    # Given: a fully covered small core declaration, a heuristic, and an omitted unlimited block.
    scene = make_scenario(
        EvaluationMap("map.png", "map", "test", "ir2", "rooms", 64, "anchor"),
        HostConfig(agent_bucket=6, max_rounds=2),
        0,
    )
    methods = (
        Method("A_learned_predict", "A", "learned", "predict", 0, "", "weights", 2000000),
        Method("heuristic", "heuristic", "fixed", "none", -1, "", "", 0),
    )
    workload = tuple(tasks((scene,), methods))
    manifest = tmp_path / "tasks.jsonl"
    write_task_manifest(manifest, workload)
    for task, score in zip(workload, (0.25, 1.0), strict=True):
        save_hand_calculated_result(tmp_path, task, score)
    declaration = {
        "base_config": "base.json",
        "scenario_seeds": [0],
        "methods": [{"method_id": "A_learned_predict"}],
        "training_seeds": [0],
        "n_agents": [2],
        "budgets": [8192],
        "conditions": [[0, 0, 64]],
        "auxiliary": ["heuristic", "unlimited_budget"],
        "auxiliary_matrix": [
            {
                "category": "heuristic",
                "method_ids": ["heuristic"],
                "training_seeds": [-1],
                "budgets": [8192],
                "conditions": [[0, 0, 64]],
            },
            {
                "category": "unlimited_budget",
                "budget_mode": "unlimited",
                "method_ids": ["A_learned_predict"],
                "budgets": [],
                "conditions": [[0, 0, 64]],
            },
        ],
    }
    core = tmp_path / "core.json"
    core.write_text(json.dumps(declaration), encoding="utf-8")
    (tmp_path / "base.json").write_text(json.dumps(asdict(scene.config)), encoding="utf-8")
    (tmp_path / "maps.jsonl").write_text(
        json.dumps(
            {
                "path": "map.png",
                "split": "test",
                "family": "rooms",
                "bucket": 64,
                "occupancy_hash": "map",
                "scenario_hash": "anchor",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "sources.json"
    registry.write_text(
        json.dumps([{"source": "ir2", "root": ".", "manifest": "maps.jsonl"}]), encoding="utf-8"
    )
    # When: the genuine report writer combines the available verified artifacts.
    report = write_report(
        tmp_path,
        tmp_path / "report",
        manifest=manifest,
        core=core,
        repetitions=5,
        map_sources=registry,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    # Then: core keeps its own score, and the missing diagnostic remains incomplete.
    assert payload["core_complete"]
    assert not payload["auxiliary_complete"]
    assert not payload["formal_complete"]
    assert [row["method_id"] for row in payload["methods"]] == ["A_learned_predict"]
    assert payload["methods"][0]["auc"] == 0.25  # noqa: PLR2004
    assert payload["missing_auxiliary_conditions"]
    with (report.parent / "auxiliary/condition_scores.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["training_seeds"] == "(-1,)"
    assert rows[0]["missing_training_seeds"] == "()"
    # When: the one missing declared block is also supplied, using its own tagged result.
    unlimited = make_scenario(
        EvaluationMap("map.png", "map", "test", "ir2", "rooms", 64, "anchor"),
        replace(scene.config, initial_budget=512),
        0,
        budget_mode="unlimited",
    )
    unlimited_task = next(tasks((unlimited,), (methods[0],)))
    save_hand_calculated_result(tmp_path, unlimited_task, 0.9)
    write_task_manifest(manifest, (*workload, unlimited_task))
    complete_path = write_report(
        tmp_path,
        tmp_path / "complete-report",
        manifest=manifest,
        core=core,
        repetitions=5,
        map_sources=registry,
    )
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    # Then: auxiliary is complete without requiring a training quota or five seeds for heuristic.
    assert complete["core_complete"]
    assert complete["auxiliary_complete"]
    assert complete["formal_complete"]
    assert complete["methods"][0]["auc"] == 0.25  # noqa: PLR2004
    unverified = write_report(
        tmp_path, tmp_path / "unverified-report", manifest=manifest, core=core, repetitions=5
    )
    unverified_data = json.loads(unverified.read_text(encoding="utf-8"))
    assert unverified_data["workload_scope"]["status"] == "unverified_scope"
    assert not unverified_data["formal_complete"]
    assert (unverified.parent / "condition_scores.csv").is_file()


def test_team_cost_table_reports_requested_allocated_and_unallocated_bytes(tmp_path: Path) -> None:
    # Given: an eight-KiB team budget shared equally by six robots.
    row = ReportRow(
        "team",
        "A",
        0,
        "ir2",
        "rooms",
        "map",
        n_agents=6,
        budget=1365,
        budget_mode="team_total",
        requested_team_bytes=8192,
        unallocated_bytes=2,
    )
    # When: exporting the ordinary actual-cost table.
    write_cost_tables((row,), tmp_path)
    # Then: requested 8192, allocated 8190 and leftover 2 remain distinct.
    with (tmp_path / "costs.csv").open(encoding="utf-8") as stream:
        saved = next(csv.DictReader(stream))
    assert (
        saved["requested_team_bytes"],
        saved["initial_team_bytes"],
        saved["unallocated_bytes"],
    ) == ("8192", "8190", "2")


def test_learning_auxiliary_keeps_full_training_quota_while_heuristic_needs_none() -> None:
    # Given: a short no-plan-input learner and an untrained heuristic.
    rows = (
        ReportRow(
            "short",
            "A_learned_none",
            0,
            "ir2",
            "rooms",
            "map",
            route="A",
            real_training_events=4096,
        ),
        ReportRow("script", "heuristic", -1, "ir2", "rooms", "map", route="heuristic"),
    )
    # When: validating auxiliary quotas separately from core algorithms.
    # Then: the learner fails its quota and the heuristic is not asked for two million events.
    assert invalid_auxiliary_quotas(rows, {"real_training_events": 2000000}) == (
        "A_learned_none:0:4096",
    )
    assert missing_auxiliary_conditions((), {"auxiliary": ["heuristic"]}) == (
        "undeclared_scope:heuristic",
    )
