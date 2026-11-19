#!/usr/bin/env python3
"""Profile the existing Phase 5 workload without changing environment algorithms.

Run with the project interpreter and PYTHONPATH pointing at a frozen source tree:
  python phase6_profile.py --source SOURCE --output NEW_DIRECTORY -- [Phase5 worker args]
Use a separate process per case. The external supervisor owns the GPU deadline.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import importlib
import json
import resource
import runpy
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from ir2_benchmark.evaluation.timing import Timing

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
Result = TypeVar("Result")


def save(path: Path, value: JsonValue) -> None:
    """Checkpoint only complete JSON objects."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def annotate_components() -> None:
    """Add transparent JAX name scopes; preserve original operations and arguments."""
    # Import only after main selects the frozen source and creates its output directory.
    import jax  # noqa: PLC0415

    targets = {
        "ir2_benchmark.sensing.ir2_sensor": {"sense_ir2": "sensing"},
        "ir2_benchmark.navigation.ir2_graph": {
            "initialize_graph": "graph_initialization",
            "update_graph": "graph_maintenance",
            "rebuild_graph_edges": "graph_edges",
            "find_frontier": "frontier_detection",
            "graph_observation": "graph_features",
        },
        "ir2_benchmark.navigation.ir2_rendezvous": {"rendezvous_utility": "rendezvous"},
        "ir2_benchmark.env.ir2_observation": {"ir2_observation": "candidate_generation"},
        "ir2_benchmark.env.observe": {"merge_local_messages": "message_decode"},
        "ir2_benchmark.env.step": {"lock_packet": "message_encode"},
        "ir2_benchmark.communication.channel": {"broadcast": "communication_delivery"},
    }
    for module_name in targets:
        importlib.import_module(module_name)
    for module_name, functions in targets.items():
        module = sys.modules[module_name]
        for name, scope in functions.items():
            original = getattr(module, name)
            wrapped = jax.named_call(original, name=scope)
            # Update imported aliases as well as the defining module.
            for loaded_name, loaded in tuple(sys.modules.items()):
                if loaded_name.startswith("ir2_benchmark.") and loaded is not None:
                    for attribute, value in tuple(vars(loaded).items()):
                        if value is original:
                            setattr(loaded, attribute, wrapped)


def main() -> None:  # noqa: PLR0915
    """Instrument synchronized original workload calls and export compiler evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source / "src"))
    # Frozen-source imports must follow sys.path setup and exclusive output creation.
    import jax  # noqa: PLC0415

    import ir2_benchmark.commands.performance as command  # noqa: PLC0415
    import ir2_benchmark.evaluation.throughput as throughput  # noqa: PLC0415, PLR0402
    from ir2_benchmark.evaluation.timing import Timing  # noqa: PLC0415

    annotate_components()
    phases = iter(("reset", "scripted_events", "nearest_frontier", "ir2_inference", "ir2_rollout"))
    records: list[dict[str, JsonValue]] = []

    def measured(
        function: Callable[[], Result],
        units: Callable[[Result], int],
        *,
        warmups: int = 3,
        repeats: int = 10,
    ) -> Timing:
        """Keep the existing measurement signature while recording synchronized evidence."""
        name = next(phases)
        # JSON lines are the supervisor's live progress protocol.
        print(json.dumps({"phase": name, "status": "compiling"}), flush=True)  # noqa: T201
        save(args.output / "progress.json", {"completed": records, "current": name})
        started = time.perf_counter()
        first = function()
        jax.block_until_ready(first)
        first_seconds = time.perf_counter() - started
        memory = None
        if isinstance(function, functools.partial) and hasattr(function.func, "lower"):
            compiled = function.func.lower(*function.args, **(function.keywords or {})).compile()
            analysis = compiled.memory_analysis()
            memory = {
                field: getattr(analysis, field)
                for field in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "temp_size_in_bytes",
                    "generated_code_size_in_bytes",
                )
            }
            with gzip.open(args.output / f"{name}.optimized-hlo.txt.gz", "wt") as stream:
                stream.write(compiled.as_text())
            save(args.output / f"{name}.compiler-cost.json", compiled.cost_analysis())
        # Timing is outside the profiler. Export overhead is excluded.
        for _ in range(warmups):
            jax.block_until_ready(function())
        samples, counts = [], []
        for _ in range(repeats):
            before = time.perf_counter()
            result = function()
            jax.block_until_ready(result)
            samples.append(time.perf_counter() - before)
            counts.append(units(result))
        timing = Timing(
            first_seconds,
            tuple(samples),
            sum(samples) / repeats,
            counts[-1],
            sum(counts) / sum(samples),
            warmups,
        )
        row = {
            "phase": name,
            "timing": dataclasses.asdict(timing),
            "compiler_memory": memory,
            "device_memory_before_trace": jax.devices()[0].memory_stats(),
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        records.append(row)
        save(args.output / "progress.json", {"completed": records, "current": name + ":trace"})
        # No browser/link handshake. Record a synchronized representative steady call.
        # Preserve the frozen harness's explicit trace/annotation nesting.
        with jax.profiler.trace(str(args.output / name), create_perfetto_trace=True):  # noqa: SIM117
            with jax.profiler.TraceAnnotation(name):
                profiled = function()
                jax.block_until_ready(profiled)
        jax.profiler.save_device_memory_profile(str(args.output / f"{name}.memory.prof"))
        row["device_memory_after_trace"] = jax.devices()[0].memory_stats()
        save(args.output / "progress.json", {"completed": records, "current": None})
        print(json.dumps({"phase": name, "status": "measured", **row}), flush=True)  # noqa: T201
        return timing

    throughput.measure = measured
    command.measure = measured
    sys.argv = [str(args.source / "scripts/phase5_performance.py"), "--worker", *forwarded]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
