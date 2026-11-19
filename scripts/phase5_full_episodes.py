"""Run the predeclared validation-only full-episode acceptance workload."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np

from ir2_benchmark.agents.baselines import make_baseline_method
from ir2_benchmark.config import load_config
from ir2_benchmark.evaluation.episode import execute_episode, summarize_execution
from ir2_benchmark.evaluation.records import completed_result, save_result
from ir2_benchmark.evaluation.runner import _EXECUTION_ERRORS, _error_result
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    make_scenario,
    read_task_manifest,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.types import GraphStateError


def declare(selection: Path, config: Path, checkpoint: Path, output: Path) -> None:
    """Freeze all identities before execution; no map is selected from observed scores."""
    selected = json.loads(selection.read_text(encoding="utf-8"))["maps"]
    base = load_config(config)
    methods = tuple(
        make_baseline_method(motion, communication, checkpoint if motion == "ir2_adapted" else None)
        for motion in ("nearest_frontier", "ir2_adapted")
        for communication in ("off", "plan", "map", "mixed")
    )
    planned = []
    for row, agents in zip(selected, (2, 4, 6), strict=True):
        data = EvaluationMap(**{key: row[key] for key in EvaluationMap.__dataclass_fields__})
        for mode in (0, 1, 2):
            settings = replace(
                base,
                n_agents=agents,
                fault_mode=mode,
                dropout_probability=0.3 if mode == 1 else 0.0,
            )
            planned.extend(tasks((make_scenario(data, settings, 71, limited=True),), methods))
    first = EvaluationMap(**{key: selected[0][key] for key in EvaluationMap.__dataclass_fields__})
    # These probes are explicit additions, never replacements for the 8 KiB main cases.
    for settings, communication in (
        (replace(base, n_agents=2, initial_budget=0), "map"),
        (replace(base, n_agents=2, initial_budget=2048), "plan"),
        (replace(base, n_agents=2, global_graph_fusion=True, fault_mode=2), "map"),
    ):
        selected_methods = tuple(
            method for method in methods if method.communication == communication
        )
        planned.extend(tasks((make_scenario(first, settings, 71, limited=True),), selected_methods))
    write_task_manifest(output / "tasks.jsonl", planned)
    declaration = {
        "scope": "phase5_engineering_acceptance_v1",
        "limited": True,
        "baseline": "be8949ff8b6d25cee0891c48d128e8fee6cc44d3",
        "selection": json.loads(selection.read_text(encoding="utf-8")),
        "main_tasks": 72,
        "explicit_probes": 6,
        "requested": len(planned),
        "round_limit": 512,
        "coverage_target": 0.99,
        "faults": {"normal": 0, "dropout": 0.3, "burst": [128, 192]},
        "methods": [asdict(method) for method in methods],
        "training_events": 0,
        "note": "Bucket/N are paired 512/2,768/4,1024/6; this is not their Cartesian product.",
    }
    (output / "core.json").write_text(json.dumps(declaration, indent=2) + "\n", encoding="utf-8")


def run(manifest: Path, output: Path, limit: int | None, *, warm_full: bool) -> None:
    """Use production kernels/records, adding compact final-state graph diagnostics."""
    counts = Counter()
    for index, task in enumerate(read_task_manifest(manifest)):
        if limit is not None and index >= limit:
            break
        result_path = output / "results" / f"{task.result_id}.json"
        if completed_result(result_path, task):
            counts["resumed"] += 1
            continue
        start = time.perf_counter()
        try:
            execution = execute_episode(task)
        except _EXECUTION_ERRORS as error:
            result = _error_result(task, error)
            save_result(result_path, result)
            counts["executed"] += 1
            counts[result.outcome] += 1
            continue
        state = execution.rollout.final_state
        graph = state.graph_memory
        diagnostic = {
            "result_id": task.result_id,
            "event": int(state.event),
            "round": int(state.round),
            "terminated": bool(state.terminated),
            "truncated": bool(state.truncated),
            "graph_nodes": np.asarray(graph.valid).sum(-1).tolist(),
            "global_nodes": np.asarray(graph.global_valid).sum(-1).tolist(),
            "history_count": np.asarray(graph.history_count).tolist(),
            "overflow": np.asarray(graph.overflow).tolist(),
            "path_failed": np.asarray(graph.path_failed).tolist(),
            "balances": np.asarray(state.balances).tolist(),
            "elapsed_seconds": time.perf_counter() - start,
        }
        if (
            warm_full
            and task.scenario.fault_mode == 0
            and task.method.communication == "mixed"
            and task.scenario.budget == 8192  # noqa: PLR2004 - Declared main budget.
        ):
            start = time.perf_counter()
            repeated = execute_episode(task)
            diagnostic["warm_full_seconds"] = time.perf_counter() - start
            diagnostic["warm_full_active_events"] = int(repeated.rollout.trace.active.sum())
            diagnostic["repeat_positions_equal"] = bool(
                np.array_equal(
                    np.asarray(execution.rollout.trace.positions),
                    np.asarray(repeated.rollout.trace.positions),
                )
            )
        try:
            result = summarize_execution(
                task, execution, output / "traces" / f"{task.result_id}.npz"
            )
        except GraphStateError as error:
            result = _error_result(task, error)
            failed_traces = output / "failed-traces"
            failed_traces.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                failed_traces / f"{task.result_id}.npz",
                **{
                    name: np.asarray(value)
                    for name, value in execution.rollout.trace._asdict().items()
                    if value is not None
                },
            )
        save_result(result_path, result)
        diagnostics = output / "diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        (diagnostics / f"{task.result_id}.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
        )
        counts["executed"] += 1
        counts[result.outcome] += 1
        print(  # noqa: T201
            json.dumps(
                {
                    "index": index,
                    "method": task.method.method_id,
                    "bucket": task.scenario.bucket,
                    "N": task.scenario.n_agents,
                    "fault": task.scenario.fault_mode,
                    "outcome": result.outcome,
                    "seconds": diagnostic["elapsed_seconds"],
                }
            ),
            flush=True,
        )
        (output / "progress.json").write_text(
            json.dumps(dict(counts), indent=2) + "\n", encoding="utf-8"
        )
        # Avoid retaining compiled graphs for a previous native shape/configuration.
        if (index + 1) % 8 == 0:
            jax.clear_caches()


def main() -> int:
    """Declare or execute a fixed acceptance workload, supporting verified resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warm-full", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        run(args.manifest, args.output, args.limit, warm_full=args.warm_full)
    else:
        declare(args.selection, args.config, args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
