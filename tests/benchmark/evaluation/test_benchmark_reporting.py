"""Fixed-policy reports retain actual costs and distinguish smoke from formal completion."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np
import pytest

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.budgets import auxiliary_kind, result_budget_fields
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.report import load_report_rows, write_report
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    identity,
    make_scenario,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.evaluation.scope import declared_method_fields, is_auxiliary, row_auxiliary_kind
from ir2_benchmark.evaluation.statistics import ReportRow
from ir2_benchmark.evaluation.tables import write_cost_tables
from ir2_benchmark.evaluation.workload import WorkloadAudit

if TYPE_CHECKING:
    from pathlib import Path

    from ir2_benchmark.evaluation.scenarios import EvaluationTask


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """A declared two-map workload with checksummed hand-calculated metric fixtures."""

    root: Path
    manifest: Path
    core: Path
    registry: Path
    workload: tuple[EvaluationTask, ...]


def save_measured_fixture(root: Path, task: EvaluationTask) -> None:
    """Persist known coverage and byte values without executing or inventing a training run."""
    scene, method = task.scenario, task.method
    trace = root / f"{task.result_id}.npz"
    np.savez(trace, coverage=np.array([0.1, 0.2, 0.2, 0.4]))
    raw = (
        asdict(scene)
        | asdict(method)
        | {
            "result_id": task.result_id,
            "outcome": "timeout",
            "metrics": {"auc": 0.3, "final_coverage": 0.4},
            "trace_file": str(trace),
            "trace_hash": file_hash(trace),
            "configuration_hash": scene.config_hash,
            "emitted_bytes": 128,
            "spent_bytes": 64,
            "overlap": 0.25,
            "allocation_concentration": 0.625,
            **result_budget_fields(scene, method),
        }
    )
    path = root / "results" / f"{task.result_id}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"result": raw, "checksum": identity(raw)}), encoding="utf-8")


@pytest.fixture
def benchmark_case(tmp_path: Path) -> BenchmarkCase:
    base = HostConfig(agent_bucket=6, max_rounds=2)
    (tmp_path / "base.json").write_text(json.dumps(asdict(base)), encoding="utf-8")
    method = Method("nf_off", "baseline", "off", "none", -1, "", "", 0, "nearest_frontier")
    declaration = {
        "scope": "benchmark",
        "base_config": "base.json",
        "scenario_seeds": [0],
        "methods": [asdict(method)],
        "training_seeds": [-1],
        "real_training_events": 0,
        "n_agents": [2],
        "budgets": [8192],
        "conditions": [[0, 0, 64]],
    }
    core = tmp_path / "benchmark.json"
    core.write_text(json.dumps(declaration), encoding="utf-8")
    entries, scenes = [], []
    for index in range(2):
        data = EvaluationMap(
            f"map-{index}.png", f"map-{index}", "test", "ir2", "rooms", 64, f"anchor-{index}"
        )
        entries.append(
            {
                "path": data.path,
                "occupancy_hash": data.map_hash,
                "scenario_hash": data.scenario_hash,
                "split": data.split,
                "family": data.family,
                "bucket": data.bucket,
            }
        )
        scenes.append(make_scenario(data, base, 0))
    (tmp_path / "maps.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    registry = tmp_path / "sources.json"
    registry.write_text(
        json.dumps([{"source": "ir2", "root": ".", "manifest": "maps.jsonl"}]), encoding="utf-8"
    )
    workload = tuple(tasks(scenes, (method,)))
    manifest = tmp_path / "tasks.jsonl"
    write_task_manifest(manifest, workload)
    for task in workload:
        save_measured_fixture(tmp_path, task)
    return BenchmarkCase(tmp_path, manifest, core, registry, workload)


@pytest.mark.parametrize("rule", ["nearest_frontier", "ir2_adapted"])
@pytest.mark.parametrize("communication", ["off", "plan", "map", "mixed"])
def test_fixed_baselines_enter_core_without_neighbor_prediction(
    rule: str, communication: str
) -> None:
    # Given: one of the eight fixed methods with no learned neighbor-plan estimator.
    method = Method(
        f"{rule}_{communication}", "baseline", communication, "none", -1, "", "", 0, rule
    )
    row = ReportRow(
        "result",
        method.method_id,
        -1,
        "ir2",
        "rooms",
        "map",
        route="baseline",
        auxiliary_kind="no_neighbor_plan",
    )
    # When: task and legacy-result classifications are calculated.
    classifications = auxiliary_kind(method), row_auxiliary_kind(row), is_auxiliary(row)
    # Then: baseline identity overrides the estimator and stale auxiliary tag.
    assert classifications == ("", "", False)


@pytest.mark.parametrize(("spent", "expected"), [(64, "0.25"), (0, "0.0"), (None, "")])
def test_budget_utilization_uses_paid_bytes_and_preserves_unknown_cost(
    tmp_path: Path, spent: int | None, expected: str
) -> None:
    # Given: physical emission may exceed success-only charges or have no ledger telemetry.
    row = ReportRow(
        "result",
        "nf_plan",
        -1,
        "ir2",
        "rooms",
        "map",
        budget=128,
        emitted_bytes=128,
        spent_bytes=spent,
    )
    # When: costs are exported for two robots with 256 total allocated bytes.
    write_cost_tables((row,), tmp_path)
    # Then: the fee and physical traffic remain independent, with unknown utilization blank.
    with (tmp_path / "costs.csv").open(encoding="utf-8") as stream:
        saved = next(csv.DictReader(stream))
    assert (saved["emitted_bytes"], saved["spent_bytes"], saved["budget_utilization"]) == (
        "128",
        "" if spent is None else str(spent),
        expected,
    )


def test_report_preserves_parallel_metrics_and_fixed_policy_scope(
    benchmark_case: BenchmarkCase,
) -> None:
    # Given: a complete declared baseline workload with no prediction telemetry or training.
    case = benchmark_case
    # When: reporting its verified manifests and per-episode measurements.
    report = write_report(
        case.root,
        case.root / "report",
        manifest=case.manifest,
        core=case.core,
        repetitions=5,
        map_sources=case.registry,
    )
    # Then: fixed methods are core, have no fabricated missing repetitions, and retain metrics.
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["requested_manifest_complete"]
    assert payload["formal_complete"]
    assert payload["uncertainty"] == "conditional_map_bootstrap_of_fixed_policies"
    assert payload["auxiliary_results"] == 0
    assert payload["missing_plan_telemetry"] == []
    assert payload["methods"][0]["missing_training_seeds"] == []
    with (report.parent / "episodes.csv").open(encoding="utf-8") as stream:
        saved = next(csv.DictReader(stream))
    assert (saved["spent_bytes"], saved["overlap"], saved["allocation_concentration"]) == (
        "64",
        "0.25",
        "0.625",
    )


@pytest.mark.parametrize(
    "state", ["limited", "missing", "error", "corrupt", "empty", "duplicate", "mismatched"]
)
def test_requested_completion_stays_separate_from_formal_proof(
    benchmark_case: BenchmarkCase,
    state: Literal["limited", "missing", "error", "corrupt", "empty", "duplicate", "mismatched"],
) -> None:
    # Given: an intentionally limited or incomplete immutable requested workload.
    case = benchmark_case
    first = case.workload[0]
    path = case.root / "results" / f"{first.result_id}.json"
    match state:
        case "limited":
            workload = (replace(first, scenario=replace(first.scenario, limited=True)),)
            write_task_manifest(case.manifest, workload)
            save_measured_fixture(case.root, workload[0])
        case "missing":
            path.unlink()
        case "error" | "mismatched":
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["result"].update(
                {
                    "error": {"outcome": "oom", "metrics": None, "error": "device memory"},
                    "mismatched": {"configuration_hash": "different-config", "metrics": {"auc": 1}},
                }[state]
            )
            envelope["checksum"] = identity(envelope["result"])
            path.write_text(json.dumps(envelope), encoding="utf-8")
        case "corrupt":
            path.write_text("{", encoding="utf-8")
        case "empty":
            write_task_manifest(case.manifest, ())
        case "duplicate":
            write_task_manifest(case.manifest, (*case.workload, first))
        case unreachable:
            assert_never(unreachable)
    # When: the report considers actual requested results separately from the full declared pool.
    report = write_report(
        case.root,
        case.root / "report",
        manifest=case.manifest,
        core=case.core,
        repetitions=5,
        map_sources=case.registry,
    )
    # Then: only the finished limited smoke is requested-complete; other incomplete cases fail.
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["requested_manifest_complete"] == (state == "limited")
    assert payload["duplicate_requested_tasks"] == int(state == "duplicate")
    assert payload["unmatched_requested_results"] == int(state == "mismatched")
    assert payload["methods"][0]["auc"] == (None if state == "empty" else 0.3)
    assert payload["pending_results"] == int(state in ("missing", "corrupt", "mismatched"))
    if state != "duplicate":
        assert not payload["formal_complete"]


def test_declaration_checks_baseline_decision_rule(benchmark_case: BenchmarkCase) -> None:
    # Given: genuine baseline tasks whose declaration claims a different navigation policy.
    case = benchmark_case
    declaration = json.loads(case.core.read_text(encoding="utf-8"))
    declaration["methods"][0]["decision_rule"] = "ir2_adapted"
    audit = WorkloadAudit(case.registry, declaration, case.core)
    # When: full map coverage is audited against explicit policy identity fields.
    for task in case.workload:
        audit.observe(task)
    proof = audit.finish()
    # Then: matching map/seed labels cannot certify the wrong navigation method.
    assert declared_method_fields(declaration)["nf_off"][-1] == ("decision_rule", "ir2_adapted")
    assert proof.invalid_core_tasks == 2  # noqa: PLR2004
    assert not proof.core_scope_complete


def test_loaded_rows_keep_absent_cost_telemetry_unknown(benchmark_case: BenchmarkCase) -> None:
    # Given: a legacy intact record that predates ledger-spending telemetry.
    case = benchmark_case
    path = case.root / "results" / f"{case.workload[0].result_id}.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["result"].pop("spent_bytes")
    envelope["checksum"] = identity(envelope["result"])
    path.write_text(json.dumps(envelope), encoding="utf-8")
    # When: legacy and current checksummed records are loaded together.
    rows, corrupt = load_report_rows(case.root)
    # Then: missing paid bytes remain absent instead of borrowing physical emission.
    assert not corrupt
    assert (
        next(row for row in rows if row.result_id == case.workload[0].result_id).spent_bytes is None
    )
