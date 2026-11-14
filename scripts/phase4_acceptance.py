"""Execute a declared, limited paired baseline workload through the public evaluator."""

# This is an acceptance workload, not a reduced substitute for the formal benchmark.
# ruff: noqa: PLR2004, S101

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np
from PIL import Image

from ir2_benchmark.agents.baselines import load_policy_parameters, make_baseline_method
from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import HostConfig, load_config
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import from_prepared_map, reset
from ir2_benchmark.evaluation.records import completed_result
from ir2_benchmark.evaluation.runner import run_manifest
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    evaluation_maps,
    make_scenario,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash
from ir2_benchmark.maps.prepare import load_ir2_map, prepare_map, select_spawns
from ir2_benchmark.models.ir2_policy import GraphPolicy


def reference_parity(
    cases: tuple[EvaluationMap, ...], config: HostConfig, checkpoint: Path
) -> list:
    """Compare real graph inputs using original Torch tensors and H100 JAX inference."""
    import torch  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location("original_model", root / "model.py")
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    original = module.PolicyNet(6, 128).cpu().eval()
    original64 = module.PolicyNet(6, 128).double().cpu().eval()
    with np.load(checkpoint, allow_pickle=False) as tensors:
        weights = {name: torch.from_numpy(tensors[name]) for name in tensors.files}
        original.load_state_dict(weights)
        original64.load_state_dict(weights)
    parameters = load_policy_parameters(checkpoint)
    forward = jax.jit(GraphPolicy().apply)
    records = []
    for data in cases:
        source = load_ir2_map(data.path)
        positions = select_spawns(source, config.n_agents, 17, config.spawn_radius)
        state = reset(
            from_prepared_map(prepare_map(source, (data.bucket,))),
            config,
            jax.random.PRNGKey(17),
            positions,
        )
        for agent in range(config.n_agents):
            graph = observe(state, agent, config).graph
            assert graph is not None
            arrays = [np.array(x, copy=True) for x in graph]
            with torch.no_grad():
                torch_args = (
                    torch.from_numpy(arrays[0][None]),
                    torch.from_numpy(arrays[1][None, None].astype(np.int64)),
                    torch.from_numpy(arrays[2].reshape(1, 1, 1).astype(np.int64)),
                    torch.from_numpy(arrays[3][None, None]),
                    torch.from_numpy(arrays[4][None, None]),
                    torch.from_numpy(arrays[5][None]),
                )
                expected = original(*torch_args).numpy()[0]
                # Original PolicyNet mutates an action mask; never reuse that argument.
                args64 = tuple(arg.clone() for arg in torch_args)
                args64 = (args64[0].double(), *args64[1:])
                expected64 = original64(*args64).numpy()[0]
            actual = forward(parameters, *graph)
            actual.block_until_ready()
            assert all(device.platform == "gpu" for device in actual.devices())
            with jax.experimental.enable_x64():
                parameters64 = jax.tree.map(lambda x: x.astype(jax.numpy.float64), parameters)
                graph64 = tuple(
                    x.astype(jax.numpy.float64)
                    if jax.numpy.issubdtype(x.dtype, jax.numpy.floating)
                    else x
                    for x in graph
                )
                actual64 = forward(parameters64, *graph64)
                np.testing.assert_allclose(actual64, expected64, atol=1e-10, rtol=1e-10)
            # Trained 400-node graphs amplify FP32 reduction-order rounding. A separate
            # FP64 oracle distinguishes this from conversion/architecture disagreement.
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)
            np.testing.assert_allclose(np.exp(actual), np.exp(expected), atol=5e-5, rtol=0)
            np.testing.assert_array_equal(np.asarray(actual) < -1e7, expected < -1e7)
            assert (
                int(np.argmax(actual))
                == int(np.argmax(expected))
                == int(np.argmax(expected64))
                == int(np.argmax(actual64))
            )
            records.append(
                {
                    "map": data.path,
                    "agent": agent,
                    "max_abs_error": float(np.max(np.abs(np.asarray(actual) - expected))),
                    "fp32_probability_max_abs_error": float(
                        np.max(np.abs(np.exp(actual) - np.exp(expected)))
                    ),
                    "fp64_max_abs_error": float(np.max(np.abs(np.asarray(actual64) - expected64))),
                    "devices": [str(d) for d in actual.devices()],
                }
            )
    return records


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Freeze all tasks before execution, then verify resume, reporting and zero-budget pairing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/benchmark.json")
    base = replace(base, n_agents=2, max_rounds=16, burst_start=4, burst_duration=4)
    small = replace(
        base,
        sensor_radius=8,
        graph_nodes_per_axis=12,
        graph_local_radius=16,
        graph_global_radius=32,
        graph_utility_radius=7,
        graph_peer_radius=8,
        graph_coordinate_scale=64,
        graph_unique_radius=8,
        graph_offshoot_unique_radius=8,
        graph_offshoot_max_radius=16,
        graph_frontier_protect_radius=16,
    )
    pixels = np.full((64, 64), 127, np.uint8)
    pixels[5:28, 5:28] = 195
    pixels[36:59, 36:59] = 195
    pixels[18:23, 20:49] = 195
    pixels[18:49, 44:49] = 195
    pixels[12, 12] = 208
    path = (args.output / "development-map.png").resolve()
    Image.fromarray(pixels).save(path)
    prepared = load_ir2_map(path)
    development = EvaluationMap(
        str(path),
        obstacle_hash(prepared.occupancy),
        "development_test",
        "procedural",
        "micro_rooms",
        64,
        scenario_hash(prepared.occupancy, prepared.anchor),
    )
    registry = args.data_root / "artifacts/maps/sources.json"
    pool = evaluation_maps(manifest_sources(registry))
    selected = []
    for origin, family, split in [
        ("ir2", "complex", "test"),
        ("ir2", "corridor", "test"),
        ("ir2", "hybrid", "test"),
        ("procedural", "rooms", "test"),
        ("procedural", "hybrid:narrow", "ood"),
    ]:
        selected.append(
            min(
                (m for m in pool if (m.source, m.family, m.split) == (origin, family, split)),
                key=lambda m: m.map_hash,
            )
        )
    methods = tuple(
        make_baseline_method(motion, comm, args.checkpoint if motion == "ir2_adapted" else None)
        for motion in ("nearest_frontier", "ir2_adapted")
        for comm in ("off", "plan", "map", "mixed")
    )
    parity = reference_parity(tuple(selected), base, args.checkpoint)
    (args.output / "reference-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    config_file = args.output / "development-config.json"
    config_file.write_text(json.dumps(asdict(small), indent=2))
    command = [
        sys.executable,
        "-m",
        "ir2_benchmark.cli",
        "evaluate",
        "--suite",
        "micro",
        "--config",
        str(config_file),
        "--map",
        str(path),
        "--method-manifest",
        str(root / "configs/evaluation/benchmark-methods.json"),
        "--reference-checkpoint",
        str(args.checkpoint),
        "--scenario-seeds",
        "0",
        "--output",
        str(args.output),
        "--batch-size",
        "2",
        "--max-pending-tasks",
        "32",
    ]
    with (args.output / "cli-smoke.log").open("w") as log:
        subprocess.run(  # noqa: S603 - Fixed interpreter/entrypoint; argv never enters a shell.
            command, stdout=log, stderr=subprocess.STDOUT, check=False, env=os.environ.copy()
        )
    cases = [(development, small), *((m, base) for m in selected)]
    planned = []
    roles = {}
    for data, config in cases:
        for seed in (0, 1):
            for mode, probability in [(0, 0.0), (1, 0.3), (2, 0.0)]:
                scene = make_scenario(
                    data,
                    replace(config, fault_mode=mode, dropout_probability=probability),
                    seed,
                    limited=True,
                )
                for task in tasks((scene,), methods):
                    planned.append(task)
                    roles[task.result_id] = "paired"
    native = next(m for m in selected if m.source == "ir2" and m.family == "hybrid")
    for n in (4, 6):
        scene = make_scenario(native, replace(base, n_agents=n, max_rounds=8), 0, limited=True)
        for task in tasks((scene,), methods):
            planned.append(task)
            roles[task.result_id] = "population"
    zero = make_scenario(development, replace(small, initial_budget=0), 0, limited=True)
    for task in tasks((zero,), methods):
        planned.append(task)
        roles[task.result_id] = "zero_budget"
    assert len(planned) == len(set(roles)) == 312
    manifest = args.output / "tasks.jsonl"
    write_task_manifest(manifest, planned)
    (args.output / "acceptance-index.json").write_text(json.dumps(roles, indent=2))
    start = time.monotonic()
    first = run_manifest(manifest, args.output, batch_size=2, max_pending_tasks=32)
    assert first.requested == 312, first
    second = run_manifest(manifest, args.output, batch_size=2, max_pending_tasks=32)
    assert second.requested == second.resumed == 312
    assert second.executed == 0
    assert second.failures == first.failures
    for task in planned:
        assert completed_result(args.output / "results" / f"{task.result_id}.json", task)
    exogenous = {}
    for task in planned:
        scene = task.scenario
        key = (
            scene.map_hash,
            scene.spawn_seed,
            scene.fault_seed,
            scene.n_agents,
            scene.budget,
            scene.max_rounds,
            scene.fault_mode,
            scene.drop_probability,
        )
        row = json.loads((args.output / "results" / f"{task.result_id}.json").read_text())["result"]
        if not row["trace_file"]:
            assert row["outcome"] not in ("success", "timeout")
            continue
        with np.load(row["trace_file"]) as trace:
            realized = trace["external_links"][trace["active"]]
        if key in exogenous:
            length = min(len(realized), len(exogenous[key]))
            np.testing.assert_array_equal(realized[:length], exogenous[key][:length])
        else:
            exogenous[key] = realized
    for motion in ("nearest_frontier", "ir2_adapted"):
        traces = []
        for task in planned:
            if roles[task.result_id] == "zero_budget" and task.method.decision_rule == motion:
                row = json.loads((args.output / "results" / f"{task.result_id}.json").read_text())[
                    "result"
                ]
                assert row["emitted_bytes"] == row["spent_bytes"] == 0
                with np.load(row["trace_file"]) as trace:
                    traces.append(trace["positions"])
        assert len(traces) == 4
        for trace in traces[1:]:
            np.testing.assert_array_equal(trace, traces[0])
    report_command = [
        sys.executable,
        "-m",
        "ir2_benchmark.cli",
        "report",
        "--results",
        str(args.output),
        "--output",
        str(args.output / "report"),
        "--task-manifest",
        str(manifest),
        "--core",
        str(root / "configs/evaluation/benchmark.json"),
        "--map-sources",
        str(registry),
        "--bootstrap-repetitions",
        "200",
    ]
    with (args.output / "cli-report.log").open("w") as log:
        subprocess.run(  # noqa: S603 - Fixed interpreter/entrypoint; argv never enters a shell.
            report_command, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    report = json.loads((args.output / "report/report.json").read_text())
    assert report["requested_manifest_complete"] == (first.failures == 0)
    assert not report["formal_complete"]
    release = json.loads((args.output / "report/release_manifest.json").read_text())
    assert all(
        v["model_selection"] == "external_pretrained_reference"
        for v in release["checkpoints"].values()
    )
    result = {
        "status": "passed" if first.failures == 0 else "failed",
        "limited": True,
        "tasks": len(planned),
        "roles": dict(Counter(roles.values())),
        "initial": asdict(first),
        "resume": asdict(second),
        "reference_graphs": len(parity),
        "zero_budget_trajectories_equal_within_motion": True,
        "elapsed_seconds": time.monotonic() - start,
    }
    (args.output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)  # noqa: T201
    return int(first.failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
