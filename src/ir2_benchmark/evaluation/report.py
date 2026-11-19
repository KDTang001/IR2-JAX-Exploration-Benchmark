"""Equal-map/equal-family statistics with training-seed and map-level uncertainty."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from ir2_benchmark.evaluation.budgets import auxiliary_kind
from ir2_benchmark.evaluation.figures import plot_facets
from ir2_benchmark.evaluation.plan_quality import write_plan_quality
from ir2_benchmark.evaluation.records import completed_result, file_hash
from ir2_benchmark.evaluation.recovery import write_recovery
from ir2_benchmark.evaluation.scenarios import identity, read_task_manifest
from ir2_benchmark.evaluation.scope import (
    has_missing_diagnostics,
    invalid_auxiliary_quotas,
    invalid_training_quotas,
    is_auxiliary,
    missing_auxiliary_conditions,
    missing_conditions,
    write_auxiliary_views,
)
from ir2_benchmark.evaluation.statistics import (
    MethodSummary,
    ReportRow,
    family_balanced_mean,
    hierarchical_interval,
    summarize_method,
)
from ir2_benchmark.evaluation.tables import write_condition_statistics, write_cost_tables
from ir2_benchmark.evaluation.workload import WorkloadAudit


def load_report_rows(root: Path) -> tuple[tuple[ReportRow, ...], tuple[str, ...]]:
    """Reject corrupt records as named failures instead of silently counting partial rows."""
    rows, corrupt = [], []
    for path in sorted((root / "results").glob("*.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            raw = envelope["result"]
            if envelope["checksum"] != identity(raw) or raw["result_id"] != path.stem:
                corrupt.append(str(path))
                continue
            if raw["outcome"] in ("success", "timeout"):
                trace = Path(raw.get("trace_file", ""))
                if not trace.is_file() or file_hash(trace) != raw.get("trace_hash"):
                    corrupt.append(f"{path}:missing_or_changed_trace")
                    continue
            metrics = raw.get("metrics") or {}
            fields = {name: raw[name] for name in ReportRow.__dataclass_fields__ if name in raw}
            fields.update(
                auc=metrics.get("auc"),
                final_coverage=metrics.get("final_coverage"),
                success_round=metrics.get("success_round"),
            )
            rows.append(ReportRow(**fields))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            corrupt.append(f"{path}:{error}")
    return tuple(rows), tuple(corrupt)


def write_report(  # noqa: PLR0913, PLR0915
    results: Path,
    output: Path,
    *,
    manifest: Path | None = None,
    core: Path | None = None,
    repetitions: int = 1000,
    map_sources: Path | None = None,
) -> Path:
    """Write measured scores/cost/failures and explicitly incomplete workloads."""
    rows, corrupt = load_report_rows(results)
    declaration = json.loads(core.read_text(encoding="utf-8")) if core is not None else {}
    benchmark = declaration.get("scope") == "benchmark"
    expected_seeds = tuple(declaration.get("training_seeds", range(5)))
    expected_methods = tuple(method["method_id"] for method in declaration.get("methods", []))
    output.mkdir(parents=True, exist_ok=True)
    present = {row.result_id for row in rows}
    current = set()
    pending = requested = core_pending = auxiliary_pending = 0
    unmatched_results = 0
    audit = WorkloadAudit(map_sources, declaration, core)
    with (output / "pending_tasks.jsonl").open("w", encoding="utf-8") as stream:
        if manifest is not None:
            for task in read_task_manifest(manifest):
                audit.observe(task)
                requested += 1
                if task.result_id in present and not completed_result(
                    results / "results" / f"{task.result_id}.json", task
                ):
                    unmatched_results += 1
                    present.remove(task.result_id)
                if task.result_id not in present:
                    stream.write(json.dumps({"result_id": task.result_id}) + "\n")
                    pending += 1
                    if task.scenario.budget_mode != "per_robot" or auxiliary_kind(task.method):
                        auxiliary_pending += 1
                    else:
                        core_pending += 1
                current.add(task.result_id)
    workload = audit.finish()
    superseded = tuple(
        row.result_id for row in rows if manifest is not None and row.result_id not in current
    )
    rows = tuple(row for row in rows if manifest is None or row.result_id in current)
    rows = tuple(row for row in rows if row.result_id in present)
    requested_complete = (
        manifest is not None
        and requested > 0
        and pending == 0
        and not corrupt
        and unmatched_results == 0
        and len(rows) == requested
        and all(row.outcome in ("success", "timeout") and row.error is None for row in rows)
    )
    core_rows = tuple(row for row in rows if not is_auxiliary(row))
    auxiliary_rows = tuple(row for row in rows if is_auxiliary(row))
    methods = tuple(sorted(set(expected_methods) | {row.method_id for row in core_rows}))
    summaries = [
        summarize_method(method, core_rows, expected_seeds, repetitions=repetitions)
        for method in methods
    ]
    absent_conditions = missing_conditions(core_rows, declaration)
    bad_training_quotas = invalid_training_quotas(core_rows, declaration)
    absent_auxiliary = missing_auxiliary_conditions(rows, declaration)
    bad_auxiliary_quotas = invalid_auxiliary_quotas(rows, declaration)
    absent_plan_telemetry = write_plan_quality(rows, output)
    absent_recovery_pairs = write_recovery(rows, manifest, output)
    core_complete = (
        manifest is not None
        and requested > 0
        and core_pending == 0
        and workload.core_scope_complete
        and not corrupt
        and not any(row.limited for row in core_rows)
        and all(not summary.missing_training_seeds and not summary.errors for summary in summaries)
        and bool(expected_methods)
        and not absent_conditions
        and not has_missing_diagnostics(core_rows, absent_plan_telemetry, absent_recovery_pairs)
        and not bad_training_quotas
        and all(row.map_scenario_hash and row.decision_rule != "unspecified" for row in core_rows)
    )
    auxiliary_complete = (
        manifest is not None
        and requested > 0
        and not corrupt
        and auxiliary_pending == 0
        and workload.auxiliary_scope_complete
        and not absent_auxiliary
        and not bad_auxiliary_quotas
        and not has_missing_diagnostics(
            auxiliary_rows, absent_plan_telemetry, absent_recovery_pairs
        )
        and all(
            row.outcome in ("success", "timeout")
            and not row.limited
            and row.map_scenario_hash
            and row.decision_rule != "unspecified"
            for row in auxiliary_rows
        )
    )
    complete = core_complete and auxiliary_complete and pending == 0
    payload = {
        "requested_manifest_complete": requested_complete,
        "formal_complete": complete,
        "core_complete": core_complete,
        "auxiliary_complete": auxiliary_complete,
        "workload_scope": asdict(workload),
        "requested": requested,
        "requested_unique_tasks": len(current),
        "duplicate_requested_tasks": requested - len(current),
        "unmatched_requested_results": unmatched_results,
        "completed_results": len(rows),
        "pending_results": pending,
        "corrupt_records": corrupt,
        "limited": any(row.limited for row in rows),
        "methods": [asdict(row) for row in summaries],
        "missing_matrix_conditions": absent_conditions,
        "missing_plan_telemetry": absent_plan_telemetry,
        "missing_recovery_pairs": absent_recovery_pairs,
        "invalid_training_quotas": bad_training_quotas,
        "missing_auxiliary_conditions": absent_auxiliary,
        "invalid_auxiliary_training_quotas": bad_auxiliary_quotas,
        "auxiliary_results": len(auxiliary_rows),
        "superseded_result_ids": superseded,
        "overall_summary_role": "core_methods_only; primary curves by split/N/fault/p/duration",
        "weighting": "scenario_mean_then_equal_maps_then_equal_families"
        if benchmark
        else "scenario_mean_then_equal_maps_then_equal_families_then_training_seeds",
        "uncertainty": "conditional_map_bootstrap_of_fixed_policies"
        if benchmark
        else "training_seed_and_map_hierarchical_bootstrap",
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, selected in (
        ("episodes", rows),
        ("failures", tuple(row for row in rows if row.outcome != "success")),
    ):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ReportRow.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(row) for row in selected)
    write_cost_tables(rows, output)
    write_condition_statistics(core_rows, output, expected_seeds, repetitions)
    plot_facets(core_rows, output)
    write_auxiliary_views(rows, output, expected_seeds, repetitions)
    lines = [
        "# Exploration benchmark report",
        "",
        f"Requested manifest complete: **{requested_complete}**.",
        f"Formal matrix complete: **{complete}**.",
        f"Core complete: **{core_complete}**; auxiliary complete: **{auxiliary_complete}**.",
        f"Episodes: {len(rows)}; missing: {pending}; corrupt: {len(corrupt)}.",
        "",
        "Scores weight maps and families equally after averaging scenarios."
        if benchmark
        else "Scores weight maps, families and training seeds equally after averaging scenarios.",
        "Intervals condition on fixed policies and resample maps within families; "
        "seed -1 is a policy sentinel, with no independent trained repetitions."
        if benchmark
        else "Intervals resample training seeds/maps; one trained model is one repetition.",
        "",
        "| Method | AUC | 95% conditional interval | Success rate | Policy seed sentinel | "
        "Missing sentinels | Errors |"
        if benchmark
        else "| Method | AUC | 95% interval | Success rate | Training seeds | Missing seeds | "
        "Errors |",
        "|---|---:|---|---:|---|---|---:|",
    ]
    lines.extend(
        f"| {row.method_id} | {row.auc} | [{row.ci_low}, {row.ci_high}] | {row.success_rate} | "
        f"{row.training_seeds} | {row.missing_training_seeds} | {row.errors} |"
        for row in summaries
    )
    lines.extend(
        [
            "",
            "Timeout success times remain absent; runtime errors remain in failures.csv.",
            "Cost, cold-start and recovery tables retain negative and nonsignificant outcomes.",
            "Plan quality uses recorded decision-cutoff estimates; missing samples remain missing.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "report.json"


__all__ = (
    "MethodSummary",
    "ReportRow",
    "family_balanced_mean",
    "hierarchical_interval",
    "load_report_rows",
    "missing_conditions",
    "summarize_method",
    "write_report",
)
