"""Native PNG trajectory fidelity against the actual IR2 sensor/graph/worker."""

# Oracle objects come from dynamically loaded, untyped legacy modules; JAX
# imports stay inside tests so environments without JAX can skip them cleanly.
# ruff: noqa: ANN001, ANN201, ANN202, PLC0415, PLR2004

import ast
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def test_native_original_ir2_trace_crosses_merge_step(monkeypatch, tmp_path):  # noqa: PLR0915
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.navigation.ir2_graph import (
        find_frontier,
        graph_observation,
        initialize_graph,
        update_graph,
    )
    from ir2_benchmark.sensing.ir2_sensor import sense_ir2

    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setitem(sys.modules, "TRAINING", False)  # noqa: FBT003
    original = importlib.import_module("graph_generator")
    sensor = importlib.import_module("sensor")
    monkeypatch.setattr(original, "NUM_DENSE_COORDS_WIDTH", 50)
    monkeypatch.setattr(original, "MERGE_GLOBAL_GRAPH", True)
    monkeypatch.setattr(original, "MERGE_GLOBAL_GRAPH_EVERY", 10)
    monkeypatch.setattr(original, "GLOBAL_GRAPH_NODE_COORDS_THRESH", 340)
    monkeypatch.setattr(original, "PRUNE_GLOBAL_GRAPH", False)
    helper_spec = importlib.util.spec_from_file_location(
        "native_graph_parity_helpers", Path(__file__).with_name("test_ir2_graph_parity.py")
    )
    assert helper_spec is not None
    assert helper_spec.loader is not None
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)
    tree = ast.parse((ROOT / "env.py").read_text(encoding="utf-8"))
    method = next(
        item
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "find_frontier"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "env.py", "exec"), namespace)  # noqa: S102
    originals = (original, None, namespace["find_frontier"])
    config = HostConfig(
        navigation_mode="ir2_graph",
        graph_capacity=400,
        graph_padding_size=400,
        graph_coordinate_scale=1000.0,
        graph_merge_threshold=340,
        max_rounds=64,
    )
    relative = "DungeonMaps/train/easy/img_9339.png"
    with Image.open(ROOT / relative) as image:
        pixels = np.asarray(image.convert("L"))
    assert pixels.shape == (480, 640)
    # Exactly Env.import_ground_truth: marker pixel number127 and threshold150.
    marker = np.argwhere(pixels == 208)
    position = marker[127, ::-1].astype(np.int32)
    truth = (pixels <= 150).astype(np.int32)
    legacy_truth = np.where(truth == 0, 255, 1)
    legacy_map = np.full(truth.shape, 127, dtype=np.int32)
    local = jnp.full(truth.shape, 2, dtype=jnp.int32)
    sense = jax.jit(lambda known, pose: sense_ir2(jnp.asarray(truth), known, pose, 80).local_map)
    empty_points, empty_valid = jnp.zeros((0, 2), jnp.int32), jnp.zeros(0, bool)
    advance = jax.jit(
        lambda state, known, pose: update_graph(
            state,
            known,
            pose,
            jnp.array(truth.shape),
            empty_points,
            empty_valid,
            empty_points,
            empty_valid,
            config=config,
        )
    )
    observe = jax.jit(
        lambda state, pose: graph_observation(state, pose, empty_points, empty_valid, config=config)
    )
    frontier = jax.jit(lambda known: find_frontier(known, jnp.array(truth.shape)))
    report_path = Path(
        os.environ.get("IR2_NATIVE_PARITY_OUTPUT", str(tmp_path / "native-parity.json"))
    )
    report = {
        "device": str(jax.devices()[0]),
        "map": relative,
        "shape": list(truth.shape),
        "steps_requested": 16,
        "status": "running",
        "step": 0,
        "check": "initialization",
        "records": [],
    }
    began = time.monotonic()
    reference = original.Graph_generator(0, truth.shape, 30, 80, relative)
    reference.route_node = [position.copy()]
    histories = [[[], []]]  # Env global history excludes the initial start.
    old_lengths = [[0, 0]]

    def check(state, known, pose, source_frontiers, step):
        report["step"] = step
        report["check"] = "sensor_map"
        expected_map = np.where(legacy_map == 255, 0, np.where(legacy_map == 1, 1, 2))
        np.testing.assert_array_equal(known, expected_map, err_msg=f"step {step} sensor")
        report["check"] = "frontiers"
        points, enabled = frontier(known)
        np.testing.assert_array_equal(np.asarray(points)[enabled], source_frontiers)
        report["check"] = "node_coordinates"
        nodes, valid = np.asarray(state.nodes), np.asarray(state.valid)
        np.testing.assert_array_equal(nodes[valid], reference.node_coords)
        report["check"] = "utility_and_observable_sets"
        for index, node in enumerate(reference.nodes_list):
            assert int(state.utility[index]) == node.utility, (
                step,
                index,
                tuple(node.coords),
                int(state.utility[index]),
                node.utility,
            )
            assert bool(state.zero_utility[index]) == node.zero_utility_node, (
                step,
                index,
                "zero latch",
            )
            actual_visible = set(
                map(tuple, np.asarray(points)[np.asarray(state.observable[index])])
            )
            assert actual_visible == node.observable_frontiers, (
                step,
                index,
                "observable set",
                actual_visible ^ node.observable_frontiers,
            )
        report["check"] = "directed_edges"
        for index, point in enumerate(reference.node_coords):
            actual_edges = set(map(tuple, nodes[np.asarray(state.edges[index]) & valid]))
            expected_edges = set(reference.graph.edges[tuple(point)])
            assert actual_edges == expected_edges, (
                step,
                tuple(point),
                "edge difference",
                actual_edges ^ expected_edges,
            )
        report["check"] = "global_nodes_and_edges"
        global_nodes, global_valid = np.asarray(state.global_nodes), np.asarray(state.global_valid)
        assert set(map(tuple, global_nodes[global_valid])) == set(
            map(tuple, reference.global_graph_nodes)
        ), (
            step,
            "global node difference",
            set(map(tuple, global_nodes[global_valid]))
            ^ set(map(tuple, reference.global_graph_nodes)),
        )
        for index in np.flatnonzero(global_valid):
            actual_edges = set(
                map(tuple, global_nodes[np.asarray(state.global_edges[index]) & global_valid])
            )
            assert actual_edges == set(reference.global_graph.edges[tuple(global_nodes[index])]), (
                step,
                tuple(global_nodes[index]),
                "global edge difference",
            )
        report["check"] = "all_worker_observation_tensors"
        expected_observation = helpers._worker_observation(reference, pose, [], config)  # noqa: SLF001
        actual_observation = observe(state, jnp.asarray(pose))
        for field, (actual, expected) in enumerate(
            zip(actual_observation, expected_observation, strict=True)
        ):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=0,
                atol=1e-7,
                err_msg=f"step {step}, observation field {field}",
            )
        assert not bool(state.overflow)
        assert not bool(state.path_failed)
        report["records"].append(
            {
                "step": step,
                "position": pose.tolist(),
                "known_cells": int(np.sum(expected_map != 2)),
                "frontiers": len(source_frontiers),
                "nodes": int(valid.sum()),
                "global_nodes": int(global_valid.sum()),
                "edges": int(np.asarray(state.edges).sum()),
            }
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["records"][-1]), flush=True)  # noqa: T201

    try:
        legacy_map = sensor.sensor_work(position, 80, legacy_map, legacy_truth)
        local = sense(local, jnp.asarray(position))
        frontiers = helpers._frontiers(originals, np.asarray(local))  # noqa: SLF001
        reference.generate_graph(position, legacy_map, frontiers)
        memory = jax.jit(
            lambda known, pose: initialize_graph(known, pose, jnp.array(truth.shape), config=config)
        )(local, jnp.asarray(position))
        check(memory, local, position, frontiers, 0)
        for step in range(1, 17):
            outgoing = list(reference.graph.edges[tuple(position)].values())
            candidates = []
            for slot, edge in enumerate(outgoing):
                index = reference.find_index_from_coords(reference.node_coords, edge.to_node)
                if slot != 0 and index != 0 and tuple(edge.to_node) != tuple(position):
                    candidates.append(index)
            assert candidates, (step, "No legal moving action")
            index = min(
                candidates,
                key=lambda i: (
                    -reference.node_utility[i],
                    reference.guidepost[i, 0],
                    -np.linalg.norm(reference.node_coords[i] - position),
                    *reference.node_coords[i],
                ),
            )
            position = reference.node_coords[index].astype(np.int32).copy()
            reference.route_node.append(position.copy())
            histories[0][0].append(position.copy())
            reference.nodes_list[index].set_visited()
            legacy_map = sensor.sensor_work(position, 80, legacy_map, legacy_truth)
            local = sense(local, jnp.asarray(position))
            old_frontiers = frontiers
            frontiers = helpers._frontiers(originals, np.asarray(local))  # noqa: SLF001
            result = reference.update_graph(
                legacy_map,
                frontiers,
                old_frontiers,
                [position],
                histories,
                old_lengths,
                extend_global_graph_towards_fronters=True,
                step=step,
            )
            assert result[0], (step, "Original graph update failed")
            old_lengths = [[len(part) for part in histories[0]]]
            memory = advance(memory, local, jnp.asarray(position))
            check(memory, local, position, frontiers, step)
        report["status"] = "passed"
        assert len({tuple(record["position"]) for record in report["records"]}) >= 8
        assert report["records"][-1]["known_cells"] > report["records"][0]["known_cells"]
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        if report["status"] != "passed":
            report["status"] = "failed"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
