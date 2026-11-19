#!/usr/bin/env python3
"""Measure unchanged full-episode runners in a new, bounded B1/B4 window.

The request/manifests are frozen separately. This script does not plan tasks,
change environment code, retry errors, or grant shared-device authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Final, ParamSpec, TypeVar

from phase6_profile_supervisor import cost_worker, query, save
from phase6_profile_supervisor import main as run_cases

if TYPE_CHECKING:
    from collections.abc import Callable

    from ir2_benchmark.evaluation.batching import EvaluationKernel, PreparedTask
    from ir2_benchmark.evaluation.episode import EpisodeExecution, LoadedLane

P = ParamSpec("P")
R = TypeVar("R")
IDLE_UTILIZATION_LIMIT: Final = 10
IDLE_SECONDS: Final = 60


def terminal_graph(lane: LoadedLane, execution: EpisodeExecution) -> dict[str, object]:
    """Record terminal counts and sticky failures, not trajectory maxima."""
    state = execution.rollout.final_state
    memory = state.graph_memory
    if memory is None:
        detail = "Graph diagnostics require ir2_graph execution"
        raise RuntimeError(detail)
    count = lane.prepared.task.scenario.n_agents
    return {
        "result_id": lane.prepared.task.result_id,
        "event_final": int(state.event),
        "local_nodes": memory.valid.sum(-1)[:count].tolist(),
        "global_nodes": memory.global_valid.sum(-1)[:count].tolist(),
        "history_counts": memory.history_count[:count].tolist(),
        "max_outgoing_degrees": memory.edges.sum(-1).max(-1)[:count].tolist(),
        "overflow": memory.overflow[:count].tolist(),
        "path_failed": memory.path_failed[:count].tolist(),
    }


def instrument_worker(output: Path) -> None:
    """Time actual runner boundaries; compilation and execution are device-synchronized."""
    import jax  # noqa: PLC0415 - Worker only; queue must not initialize a GPU.

    from ir2_benchmark.evaluation import episode, runner  # noqa: PLC0415

    rows = []
    compiled = {}
    original_kernel = episode.evaluation_kernel

    def timed(label: str, function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            result = function(*args, **kwargs)
            row = {"phase": label, "seconds": time.perf_counter() - started}
            if label == "load_lane":
                row.update(
                    result_id=result.prepared.task.result_id,
                    component_area=result.component_area,
                )
            if label == "execute_batch":
                row.update(
                    batch_size=len(args[0]),
                    result_ids=[lane.prepared.task.result_id for lane in args[0]],
                )
                if os.environ.get("IR2_GRAPH_DIAGNOSTICS") == "1":
                    diagnostic_started = time.perf_counter()
                    row["graph_terminal"] = [
                        terminal_graph(lane, execution)
                        for lane, execution in zip(args[0], result, strict=True)
                    ]
                    row["graph_diagnostics_seconds"] = time.perf_counter() - diagnostic_started
            rows.append(row)
            save(output, {"pid": os.getpid(), "boundaries": rows})
            return result

        return call

    def select_kernel(prepared: PreparedTask) -> EvaluationKernel:
        kernel = original_kernel(prepared)

        @wraps(kernel)
        def call(*values):  # noqa: ANN002, ANN202 - Preserve the kernel's heterogeneous pytree ABI.
            batch_size = values[2].shape[0]
            key = (id(kernel), batch_size)
            started = time.perf_counter()
            jax.block_until_ready(values)
            rows.append(
                {
                    "phase": "input_ready",
                    "batch_size": batch_size,
                    "seconds": time.perf_counter() - started,
                }
            )
            if key not in compiled:
                started = time.perf_counter()
                compiled[key] = kernel.lower(*values).compile()
                rows.append(
                    {
                        "phase": "lower_compile",
                        "batch_size": batch_size,
                        "seconds": time.perf_counter() - started,
                    }
                )
                save(output, {"pid": os.getpid(), "boundaries": rows})
            started = time.perf_counter()
            result = compiled[key](*values)
            jax.block_until_ready(result)
            rows.append(
                {
                    "phase": "device_execute",
                    "batch_size": batch_size,
                    "seconds": time.perf_counter() - started,
                }
            )
            save(output, {"pid": os.getpid(), "boundaries": rows})
            return result

        return call

    episode.evaluation_kernel = select_kernel
    runner.load_lane = timed("load_lane", runner.load_lane)
    runner.execute_batch = timed("execute_batch", runner.execute_batch)
    runner.summarize_execution = timed("summarize_trace_write", runner.summarize_execution)
    runner.save_result = timed("result_write", runner.save_result)


def supervise(root: Path) -> None:
    """Verify frozen inputs, wait for the authorized resource condition, then run once."""
    request = json.loads((root / "profile-request.json").read_text())
    if (root / "gpu-window.json").exists() or (root / "gpu-window-closed.json").exists():
        detail = "The batch-cost window already exists; refusing to renew or restart it"
        raise RuntimeError(detail)
    for path, expected in request["input_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            detail = f"Frozen input changed: {path}"
            raise RuntimeError(detail)
    now = time.time()
    deadline = now + 14400
    save(
        root / "gpu-window.json",
        {
            "started_epoch": now,
            "deadline_epoch": deadline,
            "started_utc": datetime.now(UTC).isoformat(),
            "authorized_seconds": 14400,
            "includes_compilation_and_waiting": True,
            "authorization": request.get(
                "authorization_summary",
                "User reply: maximum four hours; full-episode B1/B4 pairs only",
            ),
        },
    )
    idle_since = None
    try:
        with (root / "queue.jsonl").open("x") as log:
            while time.time() < deadline - 10:
                processes = query(
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                )
                gpu = query(
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                )
                quiet = not processes and int(gpu.split(",")[0]) < IDLE_UTILIZATION_LIMIT
                idle_since = idle_since or time.time() if quiet else None
                row = {
                    "epoch": time.time(),
                    "gpu": gpu,
                    "processes": processes,
                    "quiet_since": idle_since,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                save(root / "queue-current.json", row)
                if request.get("allow_shared_gpu", False) or (
                    idle_since is not None and time.time() - idle_since >= IDLE_SECONDS
                ):
                    os.environ["IR2_COST_TIMING"] = "1"
                    sys.argv = [sys.argv[0], str(root), "mini"]
                    run_cases()
                    break
                time.sleep(min(15, max(0, deadline - time.time() - 10)))
    finally:
        save(
            root / "gpu-window-closed.json",
            {
                "utc": datetime.now(UTC).isoformat(),
                "epoch": time.time(),
                "deadline_epoch": deadline,
                "finished_exists": (root / "mini-finished.json").exists(),
            },
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cost-worker"]:
        cost_worker()
    else:
        supervise(Path(sys.argv[1]).resolve())
