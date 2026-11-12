"""Coordinate-mapped checks against the repository's original IR2 classes."""

# Oracle objects come from dynamically loaded, untyped legacy modules; JAX
# imports stay inside tests so environments without JAX can skip them cleanly.
# ruff: noqa: ANN001, ANN003, ANN201, ANN202, PLC0415, PLR2004

import ast
import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_sparse_visibility_matches_all_rays_across_multiple_blocks():
    import jax
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import _visible, clear_segment, find_frontier

    local = jnp.zeros((64, 64), jnp.uint8).at[8:48, 32].set(1)
    frontiers, _ = find_frontier(local, jnp.asarray(local.shape))
    nodes = jnp.asarray([[4, 4], [24, 24], [52, 52]])
    valid = jnp.asarray([True, False, True])
    enabled = jnp.stack((jnp.zeros(len(frontiers), bool), jnp.ones(len(frontiers), bool)))
    actual = jax.jit(jax.vmap(lambda mask: _visible(local, nodes, valid, frontiers, mask, 70)))(
        enabled
    )
    rays = jax.jit(
        jax.vmap(lambda point: jax.vmap(lambda end: clear_segment(local, point, end))(frontiers))
    )(nodes)
    radius_mask = jnp.sum((nodes[:, None] - frontiers[None]) ** 2, axis=-1) < 70**2
    expected = rays[None] & radius_mask[None] & valid[None, :, None] & enabled[:, None]
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def originals(monkeypatch):
    """Load actual originals and isolate their TRAINING module sentinel."""
    pytest.importorskip("jax")
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setitem(sys.modules, "TRAINING", False)  # noqa: FBT003
    graph = importlib.import_module("graph_generator")
    node = importlib.import_module("node")
    monkeypatch.setattr(graph, "NUM_DENSE_COORDS_WIDTH", 6)
    monkeypatch.setattr(graph, "MERGE_GLOBAL_GRAPH", False)
    monkeypatch.setattr(graph, "PRUNE_GLOBAL_GRAPH", False)
    # Execute the original method AST verbatim without importing Env's unrelated
    # training and plotting dependencies.
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
    return graph, node, namespace["find_frontier"]


def _config(**overrides):
    from ir2_benchmark.config import HostConfig

    values = {
        "graph_padding_size": 80,
        "graph_capacity": 80,
        "graph_nodes_per_axis": 6,
        "max_candidates": 63,
        "graph_merge_enabled": False,
        "max_rounds": 12,
    }
    return HostConfig(**(values | overrides))


def _frontiers(originals, local):
    legacy = np.asarray([255, 1, 127])[local]
    h, w = legacy.shape
    padded = np.pad(legacy, ((0, (-h) % 4), (0, (-w) % 4)), constant_values=0)
    reduced = padded.reshape((h + 3) // 4, 4, (w + 3) // 4, 4).min(axis=(1, 3))
    return originals[2](SimpleNamespace(resolution=4), reduced)


def _assert_graph(memory, original):
    active = np.asarray(memory.valid)
    coords = np.asarray(memory.nodes)[active]
    expected = {tuple(p): i for i, p in enumerate(original.node_coords)}
    assert set(map(tuple, coords)) == set(expected)
    for index in np.flatnonzero(active):
        point = tuple(np.asarray(memory.nodes)[index])
        source_index = expected[point]
        assert int(memory.utility[index]) == original.nodes_list[source_index].utility
        assert (
            bool(memory.zero_utility[index]) == original.nodes_list[source_index].zero_utility_node
        )
        destinations = set(
            map(tuple, np.asarray(memory.nodes)[np.asarray(memory.edges[index]) & active])
        )
        assert destinations == set(original.graph.edges.get(point, {}))


def test_frontier_and_collision_match_original_endpoint_semantics(originals):
    import jax
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import clear_segment, find_frontier

    # Given unknown borders, staircase wall crossings, and occupied endpoints.
    local = np.full((35, 43), 2, dtype=np.int32)
    local[3:30, 3:38] = 0
    local[10:24, 19] = 1
    points = np.array([[5, 5], [19, 10], [25, 20], [6, 28], [0, 0]], np.int32)
    starts = np.repeat(points, len(points), axis=0)
    ends = np.tile(points, (len(points), 1))
    # Exercise the blocked integer staircase across ties, map exits, and starts
    # outside the map, where the original returns clear without entering a ray.
    rng = np.random.default_rng(73)
    starts = np.concatenate((starts, rng.integers(-4, 48, size=(1000, 2), dtype=np.int32)))
    ends = np.concatenate((ends, rng.integers(-4, 48, size=(1000, 2), dtype=np.int32)))
    reference = originals[0].Graph_generator(0, local.shape, 64, 80, "")
    legacy = np.asarray([255, 1, 127])[local]
    expected = [
        not reference.check_collision(a, b, legacy) for a, b in zip(starts, ends, strict=True)
    ]
    # When the port is jitted, it must retain the destination-exclusion quirk.
    clear = jax.jit(jax.vmap(lambda a, b: clear_segment(jnp.asarray(local), a, b)))(starts, ends)
    coordinates, enabled = jax.jit(find_frontier)(jnp.asarray(local), jnp.array(local.shape))
    # Then both the rays and exact frontier coordinate sets match originals.
    np.testing.assert_array_equal(clear, expected)
    np.testing.assert_array_equal(np.asarray(coordinates)[enabled], _frontiers(originals, local))


def test_initialization_and_multistep_updates_preserve_utility_and_guidepost(originals):
    import jax
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import graph_observation, initialize_graph, update_graph

    # Given a progressively revealed map and a route that revisits one node.
    config = _config()
    local = np.full((64, 64), 2, np.int32)
    local[3:43, 3:43] = 0
    position = np.array([10, 10], np.int32)
    original = originals[0].Graph_generator(0, local.shape, 64, 80, "")
    frontiers = _frontiers(originals, local)
    original.route_node = [position.copy()]
    original.generate_graph(position, np.asarray([255, 1, 127])[local], frontiers)
    memory = jax.jit(lambda m, p: initialize_graph(m, p, jnp.array(m.shape), config=config))(
        jnp.asarray(local), jnp.asarray(position)
    )
    _assert_graph(memory, original)
    history = []
    old_length = 0
    empty_points, empty_valid = jnp.zeros((0, 2), jnp.int32), jnp.zeros(0, bool)
    update = jax.jit(
        lambda state, m, p: update_graph(
            state,
            m,
            p,
            jnp.array(m.shape),
            empty_points,
            empty_valid,
            empty_points,
            empty_valid,
            config=config,
            extend_offshoots=False,
        )
    )
    for step, position in enumerate(
        [np.array([13, 13]), np.array([25, 25]), np.array([13, 13])], start=1
    ):
        # When moving, the original marks visited before extending its frontier.
        original.nodes_list[
            original.find_index_from_coords(original.node_coords, position)
        ].set_visited()
        original.route_node.append(position.copy())
        history.append(position.copy())
        old_frontiers = frontiers
        local[3 : 43 + step * 4, 3 : 43 + step * 4] = 0
        frontiers = _frontiers(originals, local)
        original.update_graph(
            np.asarray([255, 1, 127])[local],
            frontiers,
            old_frontiers,
            [position],
            [[history, []]],
            [[old_length, 0]],
            step=step,
        )
        old_length = len(history)
        memory = update(memory, jnp.asarray(local), jnp.asarray(position))
        # Then graph edges, observable counts, zero latches and route counts match.
        _assert_graph(memory, original)
        observation = graph_observation(
            memory, jnp.asarray(position), empty_points, empty_valid, config=config
        )
        for i in np.flatnonzero(memory.valid):
            original_index = original.find_index_from_coords(
                original.node_coords, np.asarray(memory.nodes)[i]
            )
            assert observation.node_inputs[i, 4] == original.guidepost[original_index, 0]
        assert not bool(memory.overflow)
    assert int(memory.history_count) == 4
    assert int(jnp.sum(memory.global_valid)) == 2  # Own movement history survives fusion=False.


def test_merger_matches_original_when_only_one_unprotected_pair(originals):
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import initialize_graph, merge_global_graph

    # Given one close pair and a protected robot position.
    config = _config(graph_unique_radius=30)
    local = np.zeros((96, 96), np.int32)
    position = np.array([90, 90], np.int32)
    reference = originals[0].Graph_generator(0, local.shape, 64, 80, "")
    reference.generate_graph(position, np.full_like(local, 255), np.empty((0, 2), int))
    globals_ = np.array([[10, 10], [20, 10], [90, 90]], np.int32)
    reference.global_graph_nodes = list(globals_)
    for point in globals_:
        reference.find_k_neighbor_custom(
            globals_,
            point,
            np.full_like(local, 255),
            reference.global_graph,
            global_graph_knn_dist_min=0.0,
            max_edge_len=160,
        )
    memory = initialize_graph(
        jnp.asarray(local), jnp.asarray(position), jnp.array(local.shape), config=config
    )
    global_nodes = memory.global_nodes.at[:3].set(globals_)
    global_valid = jnp.arange(config.graph_capacity) < 3
    memory = memory._replace(
        global_nodes=global_nodes,
        global_valid=global_valid,
        global_edges=global_valid[:, None] & global_valid[None],
    )
    # When merging, original and JAX must retain two connected vertices.
    reference.merge_global_graph(np.full_like(local, 255), np.empty((0, 2), int), [position], 30)
    result = merge_global_graph(
        memory,
        jnp.asarray(local),
        jnp.zeros((1, 2), jnp.int32),
        jnp.zeros(1, bool),
        jnp.asarray(position)[None],
        jnp.ones(1, bool),
        config=config,
    )
    # Then the protected location and one close-pair representative remain.
    assert int(jnp.sum(result.global_valid)) == len(reference.global_graph.nodes) == 2
    remaining = np.asarray(result.global_nodes)[result.global_valid]
    assert set(map(tuple, remaining)) == reference.global_graph.nodes
    assert tuple(position) in set(map(tuple, remaining))
    assert np.asarray(result.global_edges)[
        np.ix_(np.asarray(result.global_valid), np.asarray(result.global_valid))
    ].all()


@pytest.mark.usefixtures("originals")
def test_overflow_blocks_observation_instead_of_eviction():
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import graph_observation, initialize_graph

    # Given more free lattice nodes than the configured graph storage.
    config = _config(graph_padding_size=8, max_candidates=7)
    local = jnp.zeros((32, 32), jnp.int32)
    position = jnp.array([1, 1])
    # When capacity is exhausted, no truncated graph is permitted to act.
    memory = initialize_graph(local, position, jnp.array(local.shape), config=config)
    observation = graph_observation(
        memory, position, jnp.zeros((0, 2), jnp.int32), jnp.zeros(0, bool), config=config
    )
    # Then failure is explicit and every action and node is blocked.
    assert bool(memory.overflow)
    assert np.asarray(observation.node_padding_mask).all()
    assert np.asarray(observation.edge_padding_mask).all()


def _worker_observation(original, position, peers, config):
    """Execute original training worker observation methods without its driver."""
    torch = pytest.importorskip("torch")
    tree = ast.parse((ROOT / "multi_robot_worker.py").read_text(encoding="utf-8"))
    methods = [
        item
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for item in cls.body
        if isinstance(item, ast.FunctionDef)
        and item.name in {"get_observations", "calculate_edge_mask"}
    ]
    namespace = {
        "np": np,
        "copy": copy,
        "torch": torch,
        "NODE_COORDS_SCALING_FACTOR": 1 / config.graph_coordinate_scale,
        "NODE_UTILITY_SCALING_FACTOR": 1 / 50,
        "K_SIZE": config.max_candidates + 1,
    }
    exec(  # noqa: S102
        compile(ast.Module(body=methods, type_ignores=[]), "multi_robot_worker.py", "exec"),
        namespace,
    )
    environment = SimpleNamespace(
        generate_rendezvous_utility_layer=lambda *_: (
            0.0,
            np.zeros((len(original.node_coords), 1)),
            True,
        ),
        all_robot_map_belief_area_diff=[None],
        all_rendezvous_utility_inputs=[None],
        all_node_coords=[original.node_coords],
        all_graph=[original.graph.edges],
        all_node_utility=[original.node_utility],
        all_guidepost=[original.guidepost],
        all_robot_positions_belief=[[position, *peers]],
        find_index_from_coords=lambda p, _: original.find_index_from_coords(
            original.node_coords, p
        ),
    )
    worker_class = type(
        "OriginalObservationWorker",
        (),
        {name: namespace[name] for name in ("get_observations", "calculate_edge_mask")},
    )
    worker = worker_class()
    worker.env, worker.device = environment, "cpu"
    worker.node_padding_size, worker.k_size = config.graph_padding_size, config.max_candidates + 1
    observation, success = worker.get_observations(position, 0, 0, 0, plot=False)
    assert success
    return tuple(item.cpu().numpy().squeeze() for item in observation)


def test_full_original_worker_observation_current_nonzero_and_peer(originals):
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import graph_observation, initialize_graph

    # Given current node index != 0 and a distinct peer on the full graph.
    config = _config()
    local = np.zeros((64, 64), np.int32)
    start = np.array([10, 10], np.int32)
    reference = originals[0].Graph_generator(0, local.shape, 64, 80, "")
    reference.route_node = [start]
    reference.generate_graph(start, np.full_like(local, 255), np.empty((0, 2), int))
    memory = initialize_graph(
        jnp.asarray(local), jnp.asarray(start), jnp.array(local.shape), config=config
    )
    position, peer = reference.node_coords[8], reference.node_coords[22]
    # When the actual original worker builds its complete observation.
    expected = _worker_observation(reference, position, [peer], config)
    actual = graph_observation(
        memory, jnp.asarray(position), jnp.asarray(peer)[None], jnp.ones(1, bool), config=config
    )
    # Then all six tensors agree, including edge insertion order and both masks.
    for field, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(field, target, rtol=0, atol=1e-7)
    assert int(actual.current_index) == 8
    assert actual.node_inputs[8, 5] == -1
    assert actual.node_inputs[22, 5] == 1


def test_native_50_point_grid_k30_ties_match_reference(originals, monkeypatch):
    import jax
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import initialize_graph

    # Given the native 640-pixel geometry, 50x50 lattice, K30, and many tied
    # distances at the K cutoff. This is deliberately NOT a tie-free fixture.
    monkeypatch.setattr(originals[0], "NUM_DENSE_COORDS_WIDTH", 50)
    config = _config(
        graph_nodes_per_axis=50, graph_padding_size=400, graph_capacity=400, max_candidates=29
    )
    local = np.full((640, 640), 2, np.int32)
    local[240:400, 240:400] = 0
    start = np.array([319, 319], np.int32)
    reference = originals[0].Graph_generator(0, local.shape, 30, 80, "")
    reference.route_node = [start]
    reference.generate_graph(start, np.asarray([255, 1, 127])[local], _frontiers(originals, local))
    # When the complete JAX graph is compiled on that geometry.
    memory = jax.jit(
        lambda m: initialize_graph(m, jnp.asarray(start), jnp.array(m.shape), config=config)
    )(jnp.asarray(local))
    # Then full candidate sets, ordered outgoing edges, and utility all match.
    _assert_graph(memory, reference)
    for index, point in enumerate(reference.node_coords):
        expected = [
            reference.find_index_from_coords(reference.node_coords, edge.to_node)
            for edge in reference.graph.edges[tuple(point)].values()
        ]
        actual = np.asarray(memory.edge_order[index])
        np.testing.assert_array_equal(actual[actual >= 0], expected)


def test_duplicate_start_node_is_retained_like_original(originals):
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import initialize_graph

    # Given an initial start that lies exactly on the generated lattice.
    config = _config()
    local = np.zeros((64, 64), np.int32)
    position = np.array([13, 13], np.int32)
    reference = originals[0].Graph_generator(0, local.shape, 64, 80, "")
    reference.generate_graph(position, np.full_like(local, 255), np.empty((0, 2), int))
    # When generating the graph, original prepends without deduplicating start.
    memory = initialize_graph(
        jnp.asarray(local), jnp.asarray(position), jnp.array(local.shape), config=config
    )
    # Then the duplicate features remain but edge IDs resolve to the first copy.
    np.testing.assert_array_equal(np.asarray(memory.nodes)[memory.valid], reference.node_coords)
    for index, point in enumerate(reference.node_coords):
        expected = [
            reference.find_index_from_coords(reference.node_coords, edge.to_node)
            for edge in reference.graph.edges[tuple(point)].values()
        ]
        actual = np.asarray(memory.edge_order[index])
        np.testing.assert_array_equal(actual[actual >= 0], expected)


@pytest.mark.parametrize("wall", [False, True])
@pytest.mark.parametrize("duplicate_start", [False, True])
def test_offshoots_and_own_global_graph_match_original(originals, wall, duplicate_start):
    import jax
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import initialize_graph, update_graph

    # Given a revealing frontier, zeroed current-node utility, and optionally a
    # wall that forces an offshoot to use the graph's shortest path.
    config = _config()
    local = np.full((96, 96), 2, np.int32)
    local[3:88, 3:88] = 0
    if wall:
        local[3:58, 47] = 1
    start = np.array([38, 19] if duplicate_start else [10, 10], np.int32)
    position = np.array([19, 19], np.int32)
    frontiers = _frontiers(originals, local)
    reference = originals[0].Graph_generator(0, local.shape, 64, 80, "")
    reference.route_node = [start]
    reference.generate_graph(start, np.asarray([255, 1, 127])[local], frontiers)
    memory = initialize_graph(
        jnp.asarray(local), jnp.asarray(start), jnp.array(local.shape), config=config
    )
    reference.nodes_list[
        reference.find_index_from_coords(reference.node_coords, position)
    ].set_visited()
    reference.route_node.append(position)
    # When both implementations extend global offshoots with the same history.
    reference.update_graph(
        np.asarray([255, 1, 127])[local],
        frontiers,
        frontiers,
        [position],
        [[[position], []]],
        [[0, 0]],
        extend_global_graph_towards_fronters=True,
        step=1,
    )
    empty_points, empty_valid = jnp.zeros((0, 2), jnp.int32), jnp.zeros(0, bool)
    memory = jax.jit(
        lambda state: update_graph(
            state,
            jnp.asarray(local),
            jnp.asarray(position),
            jnp.array(local.shape),
            empty_points,
            empty_valid,
            empty_points,
            empty_valid,
            config=config,
        )
    )(memory)
    # Then local graph, offshoot coordinate membership, and global edges agree.
    _assert_graph(memory, reference)
    actual_nodes = np.asarray(memory.global_nodes)
    assert set(map(tuple, actual_nodes[memory.global_valid])) == set(
        map(tuple, reference.global_graph_nodes)
    )
    for index in np.flatnonzero(memory.global_valid):
        destinations = set(
            map(
                tuple,
                actual_nodes[
                    np.asarray(memory.global_edges[index]) & np.asarray(memory.global_valid)
                ],
            )
        )
        assert destinations == set(reference.global_graph.edges[tuple(actual_nodes[index])])


def test_partial_windows_node_removal_peer_window_and_observation(originals, monkeypatch):
    import jax
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import graph_observation, initialize_graph, update_graph

    # Given a full initial graph that must shrink to own80/peer20 windows.
    monkeypatch.setattr(originals[0], "NUM_DENSE_COORDS_WIDTH", 10)
    config = _config(
        graph_nodes_per_axis=10, graph_padding_size=180, graph_capacity=180, max_candidates=29
    )
    local = np.zeros((256, 256), np.int32)
    start = np.array([10, 10], np.int32)
    reference = originals[0].Graph_generator(0, local.shape, 30, 80, "")
    reference.route_node = [start]
    reference.generate_graph(start, np.full_like(local, 255), np.empty((0, 2), int))
    memory = initialize_graph(
        jnp.asarray(local), jnp.asarray(start), jnp.array(local.shape), config=config
    )
    history, old_count = [], 0
    empty_points, empty_valid = jnp.zeros((0, 2), jnp.int32), jnp.zeros(0, bool)
    run = jax.jit(
        lambda state, position, peer: update_graph(
            state,
            jnp.asarray(local),
            position,
            jnp.array(local.shape),
            peer[None],
            jnp.ones(1, bool),
            empty_points,
            empty_valid,
            config=config,
            extend_offshoots=False,
        )
    )
    for step, (position, peer) in enumerate(
        [(np.array([57, 57]), np.array([198, 198])), (np.array([85, 85]), np.array([227, 227]))],
        start=1,
    ):
        # When movement removes old nodes and adds another local window.
        reference.nodes_list[
            reference.find_index_from_coords(reference.node_coords, position)
        ].set_visited()
        reference.route_node.append(position)
        history.append(position)
        reference.update_graph(
            np.full_like(local, 255),
            np.empty((0, 2), int),
            np.empty((0, 2), int),
            [position, peer],
            [[history, []], [None, None]],
            [[old_count, 0], [0, 0]],
            step=step,
        )
        old_count = len(history)
        memory = run(memory, jnp.asarray(position), jnp.asarray(peer))
        # Then survivor ordering, full adjacency, and original-worker observation
        # still match; stale edges cannot survive deletion in edge_order.
        np.testing.assert_array_equal(np.asarray(memory.nodes)[memory.valid], reference.node_coords)
        _assert_graph(memory, reference)
        expected = _worker_observation(reference, position, [peer], config)
        actual = graph_observation(
            memory, jnp.asarray(position), jnp.asarray(peer)[None], jnp.ones(1, bool), config=config
        )
        for field, target in zip(actual, expected, strict=True):
            np.testing.assert_allclose(field, target, rtol=0, atol=1e-7)


def test_failed_merger_preserves_original_partial_dictionary_rollback(originals):
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import initialize_graph, merge_global_graph

    # Given stale bridge edges across a newly observed wall and an unrelated
    # directed pair. Source rollback only restores copied outgoing dictionaries.
    config = _config(max_candidates=1)
    local = np.zeros((200, 200), np.int32)
    local[:30, 25] = 1
    points = np.array([[10, 10], [20, 10], [60, 10], [120, 10], [180, 10]], np.int32)
    position = points[2]
    reference = originals[0].Graph_generator(0, local.shape, 2, 80, "")
    reference.generate_graph(position, np.asarray([255, 1, 127])[local], np.empty((0, 2), int))
    adjacency = np.eye(5, dtype=bool)
    for a, b in [(0, 1), (1, 0), (1, 2), (2, 1), (3, 4)]:
        adjacency[a, b] = True
    reference.global_graph_nodes = list(points)
    for i, point in enumerate(points):
        reference.global_graph.add_node(tuple(point))
        for j in np.flatnonzero(adjacency[i]):
            reference.global_graph.add_edge(
                tuple(point), tuple(points[j]), np.linalg.norm(point - points[j])
            )
    memory = initialize_graph(
        jnp.asarray(local), jnp.asarray(position), jnp.array(local.shape), config=config
    )
    memory = memory._replace(
        global_nodes=memory.global_nodes.at[:5].set(points),
        global_valid=jnp.arange(config.graph_capacity) < 5,
        global_edges=memory.global_edges.at[:5, :5].set(adjacency),
    )
    # When removing either bridge endpoint disconnects protected-node criteria.
    reference.merge_global_graph(
        np.asarray([255, 1, 127])[local], np.empty((0, 2), int), [position], 30
    )
    memory = merge_global_graph(
        memory,
        jnp.asarray(local),
        jnp.zeros((1, 2), jnp.int32),
        jnp.zeros(1, bool),
        jnp.asarray(position)[None],
        jnp.ones(1, bool),
        config=config,
    )
    # Then the failed transaction retains the original's unrelated reverse edge.
    assert (
        set(map(tuple, np.asarray(memory.global_nodes)[memory.global_valid]))
        == reference.global_graph.nodes
    )
    for i in np.flatnonzero(memory.global_valid):
        actual = set(map(tuple, np.asarray(memory.global_nodes)[memory.global_edges[i]]))
        assert actual == set(
            reference.global_graph.edges[tuple(np.asarray(memory.global_nodes)[i])]
        )


@pytest.mark.usefixtures("originals")
def test_batched_insert_matches_sequential_first_occurrence_and_overflow():
    import jax

    from ir2_benchmark.navigation.ir2_graph import _insert

    # Given holes, repeated incoming coordinates, masked entries, and overflow.
    rng = np.random.default_rng(42)
    nodes = rng.integers(0, 7, size=(4, 8, 2), dtype=np.int32)
    valid = rng.random((4, 8)) > 0.45
    incoming = rng.integers(0, 7, size=(4, 18, 2), dtype=np.int32)
    incoming[:, 3:6] = incoming[:, :3]
    present = rng.random((4, 18)) > 0.2
    expected = []
    for stored, stored_valid, additions, enabled in zip(
        nodes, valid, incoming, present, strict=True
    ):
        points, occupied = stored.copy(), stored_valid.copy()
        overflow = False
        for point, allowed in zip(additions, enabled, strict=True):
            if not allowed or np.any(occupied & np.all(points == point, axis=1)):
                continue
            if occupied.all():
                overflow = True
            else:
                slot = np.flatnonzero(~occupied)[0]
                points[slot], occupied[slot] = point, True
        expected.append((points, occupied, overflow))
    # When all insertion batches are vectorized, preserve sequential semantics.
    actual = jax.jit(jax.vmap(_insert))(nodes, valid, incoming, present)
    # Then identical storage, validity, and explicit failure flags result.
    for field, values in enumerate(actual):
        np.testing.assert_array_equal(values, np.stack([row[field] for row in expected]))


@pytest.mark.usefixtures("originals")
def test_batched_graph_updates_match_independent_divergent_branches():
    import jax
    import jax.numpy as jnp

    from ir2_benchmark.navigation.ir2_graph import initialize_graph, update_graph

    # Given one zero-utility map and one frontier-rich map, offshoot branches
    # differ across robots in exactly the way the collector batches them.
    config = _config(graph_merge_enabled=True, graph_merge_every=1)
    maps = np.zeros((2, 96, 96), np.int32)
    maps[1] = 2
    maps[1, 3:88, 3:88] = 0
    position = jnp.array([19, 19], jnp.int32)
    start = jnp.array([10, 10], jnp.int32)
    states = jax.jit(
        jax.vmap(lambda m: initialize_graph(m, start, jnp.array(m.shape), config=config))
    )(jnp.asarray(maps))
    empty_points, empty_valid = jnp.zeros((0, 2), jnp.int32), jnp.zeros(0, bool)

    def advance(state, known):
        return update_graph(
            state,
            known,
            position,
            jnp.array(known.shape),
            empty_points,
            empty_valid,
            empty_points,
            empty_valid,
            config=config,
        )

    independent = [
        jax.jit(advance)(jax.tree.map(lambda x, i=i: x[i], states), jnp.asarray(maps[i]))
        for i in range(2)
    ]
    # When control flow is batched, inactive branches must not alter results.
    actual = jax.jit(jax.vmap(advance))(states, jnp.asarray(maps))
    # Then every persistent-state tensor equals its separately executed oracle.
    for i, expected in enumerate(independent):
        for got, wanted in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got[i], wanted)
