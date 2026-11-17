#!/usr/bin/env python3
"""Pair the measured visibility optimization with original source in one device process.

Usage: python phase6_compare.py PHASE6_ROOT CASE_ID
The supervisor must isolate each case and enforce the existing GPU deadline.
"""

from __future__ import annotations

# Imports follow frozen-source selection; private visibility replacement is the experiment.
# ruff: noqa: PLC0415, SLF001
import ast
import dataclasses
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias, TypedDict

    from jax import Array
    from jax.stages import Compiled, Wrapped

    from ir2_benchmark.env.rollout import RolloutResult
    from ir2_benchmark.types import EnvState

    Arguments: TypeAlias = tuple[Array | EnvState, ...]
    FactoryResult: TypeAlias = tuple[Wrapped, Arguments]
    CompiledCase: TypeAlias = tuple[Compiled, Arguments, float]

    class LeafIdentity(TypedDict):
        """Exact comparison evidence for one flattened device-array leaf."""

        leaf: int
        shape: list[int]
        dtype: str
        sha256: str
        exact: bool

    class ParityResult(TypedDict):
        """Successful exact comparison and its individual leaf identities."""

        exact: bool
        leaf_count: int
        leaves: list[LeafIdentity]


# Keep the experimental pairing flow together without changing its execution order.
def main() -> None:  # noqa: C901, PLR0915
    """Compile and measure both visibility implementations with identical case inputs."""
    root, case_id = Path(sys.argv[1]).resolve(), sys.argv[2]
    request = json.loads((root / "profile-request.json").read_text())
    case = next(row for row in request["cases"] if row["case_id"] == case_id)
    output = root / "optimization-pairs" / case_id
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "source-final/src"))
    import jax
    import numpy as np

    from ir2_benchmark.agents.baselines import BaselineSpec, baseline_action, load_policy_parameters
    from ir2_benchmark.config import load_config
    from ir2_benchmark.env.reset import from_prepared_map, reset
    from ir2_benchmark.env.rollout import rollout
    from ir2_benchmark.maps.prepare import load_ir2_map, prepare_map
    from ir2_benchmark.navigation import ir2_graph
    from ir2_benchmark.types import validate_graph_state

    candidate = ir2_graph._visible
    old_file = root / "source-baseline/src/ir2_benchmark/navigation/ir2_graph.py"
    tree = ast.parse(old_file.read_text())
    definition = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_visible"
    )
    namespace = dict(vars(ir2_graph))
    # The baseline function comes from the explicitly frozen local source snapshot.
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(old_file), "exec"), namespace)  # noqa: S102
    original = namespace["_visible"]
    config = dataclasses.replace(
        load_config(root / "source-final/configs/benchmark.json"),
        n_agents=case["n_agents"],
        agent_bucket=6,
    )
    data = from_prepared_map(
        prepare_map(load_ir2_map(case["map"]["path"]), buckets=(case["bucket"],))
    )
    keys = jax.random.split(jax.random.PRNGKey(7), case["batch_size"])
    weights = load_policy_parameters(Path(request["checkpoint"]))
    records = []

    def save() -> None:
        target = output / "comparison.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"case_id": case_id, "records": records}, indent=2) + "\n")
        temporary.replace(target)

    def compare(left: EnvState | RolloutResult, right: EnvState | RolloutResult) -> ParityResult:
        first, first_tree = jax.tree.flatten(left)
        second, second_tree = jax.tree.flatten(right)
        if first_tree != second_tree:
            # Preserve the experimental harness's exact failure contract.
            raise AssertionError("output PyTree changed")  # noqa: EM101, TRY003
        mismatches = []
        leaves: list[LeafIdentity] = []
        for index, (before, after) in enumerate(zip(first, second, strict=True)):
            a, b = np.asarray(before), np.asarray(after)
            exact = np.array_equal(a, b, equal_nan=True)
            if not exact:
                mismatches.append(
                    {
                        "leaf": index,
                        "shape": list(a.shape),
                        "dtype": str(a.dtype),
                        "max_abs_difference": float(
                            np.max(np.abs(a.astype(float) - b.astype(float)))
                        ),
                    }
                )
            leaves.append(
                {
                    "leaf": index,
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                    "exact": exact,
                }
            )
        if mismatches:
            raise AssertionError(json.dumps({"mismatches": mismatches}))
        return {"exact": True, "leaf_count": len(leaves), "leaves": leaves}

    def compile_pair(factory: Callable[[], FactoryResult]) -> list[CompiledCase]:
        compiled = []
        for visible in (original, candidate):
            ir2_graph._visible = visible
            function, arguments = factory()
            before = time.perf_counter()
            executable = function.lower(*arguments).compile()
            compile_seconds = time.perf_counter() - before
            compiled.append((executable, arguments, compile_seconds))
        return compiled

    def measure_pair(label: str, compiled: list[CompiledCase]) -> list[EnvState | RolloutResult]:
        results = [
            jax.block_until_ready(function(*arguments)) for function, arguments, _ in compiled
        ]
        parity = compare(*results)
        for result in results:
            state = result.final_state if hasattr(result, "final_state") else result
            validate_graph_state(state)
        for function, arguments, _ in compiled:
            jax.block_until_ready(function(*arguments))
        times = [[], []]
        # Alternate order to reduce drift/order bias. Synchronize complete outputs.
        for repeat in range(10):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for index in order:
                function, arguments, _ = compiled[index]
                before = time.perf_counter()
                result = function(*arguments)
                jax.block_until_ready(result)
                times[index].append(time.perf_counter() - before)
        memories = []
        for function, _, _ in compiled:
            analysis = function.memory_analysis()
            memories.append(
                {
                    name: getattr(analysis, name)
                    for name in (
                        "argument_size_in_bytes",
                        "output_size_in_bytes",
                        "temp_size_in_bytes",
                        "alias_size_in_bytes",
                    )
                }
            )
        row = {
            "workload": label,
            "parity": parity,
            "baseline_compile_seconds": compiled[0][2],
            "candidate_compile_seconds": compiled[1][2],
            "baseline_samples_seconds": times[0],
            "candidate_samples_seconds": times[1],
            "speedup_median": statistics.median(times[0]) / statistics.median(times[1]),
            "baseline_compiler_memory": memories[0],
            "candidate_compiler_memory": memories[1],
            "memory_scope": (
                "compiler buffer plan per executable; process holds both variants; "
                "no independent NVML peak claim"
            ),
        }
        if hasattr(results[0], "trace"):
            row.update(
                active_events=int(np.asarray(results[0].trace.active).sum()),
                covered_cells=[int(np.asarray(s).sum()) for s in results[0].final_state.covered],
                emitted_bytes=int(np.asarray(results[0].trace.emitted_length).sum()),
                delivery_count=int(np.asarray(results[0].trace.delivered).sum()),
                invalid_actions=int(np.asarray(results[0].trace.invalid_action).sum()),
            )
        records.append(row)
        save()
        # JSON stdout is the supervisor's synchronized progress protocol.
        print(  # noqa: T201
            json.dumps(
                {
                    "case_id": case_id,
                    "workload": label,
                    "parity": parity["exact"],
                    "speedup_median": row["speedup_median"],
                    "baseline_memory": memories[0],
                    "candidate_memory": memories[1],
                }
            ),
            flush=True,
        )
        return results

    reset_pair = compile_pair(
        lambda: (jax.jit(jax.vmap(lambda key: reset(data, config, key))), (keys,))
    )
    states = measure_pair("reset", reset_pair)
    initial = states[0]
    conditions = [(0, 0.0)]
    if case_id == "gpu-768-n4-b4":
        conditions += [(1, 0.3), (2, 0.0)]
    for fault, drop in conditions:
        settings = dataclasses.replace(
            config, fault_mode=fault, dropout_probability=drop, burst_start=2, burst_duration=2
        )
        for motion in ("nearest_frontier", "ir2_adapted"):
            spec = BaselineSpec(motion, "mixed")

            def factory() -> FactoryResult:
                # compile_pair traces synchronously before either loop variable advances.
                function = jax.jit(
                    jax.vmap(
                        lambda state: rollout(
                            state,
                            settings,  # noqa: B023
                            32,
                            policy=lambda observation, cfg: baseline_action(
                                observation,
                                cfg,
                                spec,  # noqa: B023
                                weights,
                            ),
                        )
                    )
                )
                return function, (initial,)

            measure_pair(f"{motion}_mixed_fault{fault}_32events", compile_pair(factory))
    ir2_graph._visible = candidate


if __name__ == "__main__":
    main()
