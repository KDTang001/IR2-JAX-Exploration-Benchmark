"""Compare route utilities with the actual original Env method and graph A*."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def _legacy_types() -> tuple[type, type, type]:
    """Load original environment and graph classes without constructing a world."""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.modules.setdefault("TRAINING", False)
    return (
        importlib.import_module("env").Env,
        importlib.import_module("graph_generator").Graph_generator,
        importlib.import_module("graph").Graph,
    )


def _oracle(  # noqa: PLR0913, PLR0917 - mirrors the original environment inputs
    nodes: np.ndarray,
    edges: np.ndarray,
    positions: np.ndarray,
    enabled: np.ndarray,
    surplus: np.ndarray,
    local: np.ndarray,
    repair: np.ndarray,
) -> tuple[np.ndarray, bool, bool]:
    """Run the actual Env method with controlled graph and delivered-map inputs."""
    env_type, generator_type, graph_type = _legacy_types()

    def make_graph(adjacency: np.ndarray) -> object:
        graph = graph_type()
        for node in nodes:
            graph.add_node(tuple(node))
            graph.edges[tuple(node)] = {}
        for start, end in np.argwhere(adjacency):
            graph.add_edge(
                tuple(nodes[start]), tuple(nodes[end]), np.linalg.norm(nodes[start] - nodes[end])
            )
        return graph

    generator = generator_type.__new__(generator_type)
    generator.graph = make_graph(edges)
    generator.global_graph = graph_type()
    rebuilds = []

    def rebuild(*_args: object, **kwargs: object) -> None:
        assert kwargs["global_graph_knn_dist_max"] == 800  # noqa: PLR2004 - original global repair range
        assert kwargs["global_graph_knn_dist_min"] == 0
        rebuilds.append(True)
        generator.graph = make_graph(repair)

    generator.edge_clear_all_nodes = lambda: None
    generator.find_k_neighbor_all_nodes = rebuild
    env = env_type.__new__(env_type)
    env.all_node_coords = [nodes]
    env.ground_truth_size = local.shape
    env.all_robot_positions_belief = [[nodes[0], *positions]]
    env.group_ids_list = [[0], [i + 1 for i, value in enumerate(enabled) if value]]
    env.all_graph_generator = [generator]
    env.all_graph = [generator.graph.edges]
    env.all_robot_belief = [[np.asarray([255, 1, 127])[local]]]
    env.compute_map_belief_area_diff = lambda _robot_id: [0, *surplus]
    _, values, success = env.generate_rendezvous_utility_layer(0, 0)
    return values[:, 0], success, bool(rebuilds)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse route with off-route probes for LOS and densification."""
    nodes = np.array([[10, 60], [90, 60], [170, 60], [90, 100], [130, 100], [170, 100]], np.int32)
    edges = np.eye(len(nodes), dtype=bool)
    edges[0, 1] = edges[1, 2] = True
    local = np.zeros((130, 200), np.uint8)
    return nodes, edges, local


@pytest.mark.parametrize(
    "case",
    [
        "directed",
        "bidirectional",
        "repair",
        "unreachable",
        "below_threshold",
        "at_threshold",
        "disabled",
        "same_node",
        "wall",
        "unknown",
    ],
)
def test_original_env_rendezvous_route_and_fallback_parity(case: str) -> None:
    nodes, edges, local = _fixture()
    repair = edges.copy()
    positions = nodes[[2]]
    enabled = np.array([True])
    surplus = np.array([800], np.float32)
    if case == "bidirectional":
        edges = edges.T
    elif case in {"repair", "unreachable"}:
        edges = np.eye(len(nodes), dtype=bool)
        if case == "unreachable":
            repair = edges.copy()
    elif case == "below_threshold":
        surplus[:] = local.size * 0.01 - 1
    elif case == "at_threshold":
        surplus[:] = local.size * 0.01
    elif case == "disabled":
        enabled[:] = False
    elif case == "same_node":
        positions = nodes[[0]]
    elif case in {"wall", "unknown"}:
        local[80, 75:190] = 1 if case == "wall" else 2
    expected, success, repaired = _oracle(nodes, edges, positions, enabled, surplus, local, repair)
    jax = pytest.importorskip("jax")
    from ir2_benchmark.navigation.ir2_rendezvous import rendezvous_utility  # noqa: PLC0415

    actual = jax.jit(rendezvous_utility)(
        nodes,
        np.ones(len(nodes), bool),
        edges,
        np.int32(0),
        positions,
        enabled,
        surplus,
        local,
        np.array(local.shape, np.int32),
        repair,
    )
    np.testing.assert_allclose(actual.utility, expected, atol=2e-7, rtol=1e-6)
    assert bool(np.any(actual.unreachable)) == (not success)
    assert bool(actual.repaired) == repaired


def test_batched_multiple_peers_take_maximum_and_keep_repair_for_next_peer() -> None:
    nodes, edges, local = _fixture()
    repair = edges.copy()
    repair[2, 5] = True
    edges = np.eye(len(nodes), dtype=bool)
    positions = nodes[[2, 5]]
    enabled = np.array([True, True])
    surplus = np.array([800, 1200], np.float32)
    worlds = np.stack([local, local.copy()])
    worlds[1, 80, 75:190] = 1
    expected = [
        _oracle(nodes, edges, positions, enabled, surplus, world, repair) for world in worlds
    ]
    jax = pytest.importorskip("jax")
    from ir2_benchmark.navigation.ir2_rendezvous import rendezvous_utility  # noqa: PLC0415

    result = jax.jit(
        jax.vmap(
            lambda world: rendezvous_utility(
                nodes,
                np.ones(len(nodes), bool),
                edges,
                np.int32(0),
                positions,
                enabled,
                surplus,
                world,
                np.array(local.shape, np.int32),
                repair,
            )
        )
    )(worlds)
    np.testing.assert_allclose(
        result.utility, np.stack([value[0] for value in expected]), atol=2e-7, rtol=1e-6
    )
    assert not np.any(result.unreachable)
    assert np.all(result.repaired)


def test_tied_routes_use_coordinates_under_permuted_node_ids() -> None:
    nodes = np.array([[10, 100], [90, 170], [90, 30], [170, 100], [90, 0], [90, 200]], np.int32)
    edges = np.eye(len(nodes), dtype=bool)
    edges[0, [1, 2]] = True
    edges[[1, 2], 3] = True
    permutations = np.array([[0, 1, 2, 3, 4, 5], [0, 2, 1, 3, 5, 4]])
    worlds = nodes[permutations]
    graphs = np.stack([edges[np.ix_(order, order)] for order in permutations])
    local = np.zeros((220, 200), np.uint8)
    positions = nodes[[3]]
    enabled = np.array([True])
    surplus = np.array([1200], np.float32)
    expected = [
        _oracle(world, graph, positions, enabled, surplus, local, graph)[0]
        for world, graph in zip(worlds, graphs, strict=True)
    ]
    # Equal-length upper/lower paths must choose the upper branch by (x, y),
    # even when that branch has the larger array index.
    assert expected[0][4] > 0
    assert expected[0][5] == 0
    jax = pytest.importorskip("jax")
    from ir2_benchmark.navigation.ir2_rendezvous import rendezvous_utility  # noqa: PLC0415

    result = jax.jit(
        jax.vmap(
            lambda world, graph: rendezvous_utility(
                world,
                np.ones(len(nodes), bool),
                graph,
                np.int32(0),
                positions,
                enabled,
                surplus,
                local,
                np.array(local.shape, np.int32),
                graph,
            )
        )
    )(worlds, graphs)
    np.testing.assert_allclose(result.utility, np.stack(expected), atol=2e-7, rtol=1e-6)
    assert not np.any(result.unreachable)
    assert not np.any(result.repaired)


def test_batched_disabled_and_enabled_peers_preserve_original_results() -> None:
    nodes, edges, local = _fixture()
    graphs = np.stack([edges, edges, edges, np.eye(len(nodes), dtype=bool)])
    enabled = np.array([[True], [False], [True], [True]])
    surplus = np.array([[800], [800], [0], [800]], np.float32)
    positions = nodes[[2]]
    expected = [
        _oracle(nodes, graph, positions, active, delta, local, graph)
        for graph, active, delta in zip(graphs, enabled, surplus, strict=True)
    ]
    jax = pytest.importorskip("jax")
    from ir2_benchmark.navigation.ir2_rendezvous import rendezvous_utility  # noqa: PLC0415

    result = jax.jit(
        jax.vmap(
            lambda graph, active, delta: rendezvous_utility(
                nodes,
                np.ones(len(nodes), bool),
                graph,
                np.int32(0),
                positions,
                active,
                delta,
                local,
                np.array(local.shape, np.int32),
                graph,
            )
        )
    )(graphs, enabled, surplus)
    np.testing.assert_allclose(
        result.utility, np.stack([value[0] for value in expected]), atol=2e-7, rtol=1e-6
    )
    np.testing.assert_array_equal(result.unreachable[:, 0], [not value[1] for value in expected])
    np.testing.assert_array_equal(result.repaired, [value[2] for value in expected])


def test_batched_reachable_only_continues_after_unreachable_peer() -> None:
    # Given an unreachable peer before a reachable one, with both eligible.
    nodes, edges, local = _fixture()
    positions = nodes[[5, 2]]
    enabled = np.array([True, True])
    surplus = np.array([800, 1200], np.float32)
    strict, success, repaired = _oracle(nodes, edges, positions, enabled, surplus, local, edges)
    reachable, reachable_success, _ = _oracle(
        nodes, edges, positions[1:], enabled[1:], surplus[1:], local, edges
    )
    assert not success
    assert repaired
    assert reachable_success
    assert reachable[0] == 0
    assert np.any(reachable > 0)
    jax = pytest.importorskip("jax")
    from ir2_benchmark.navigation.ir2_rendezvous import rendezvous_utility  # noqa: PLC0415

    # When the abort gate is a dynamic scalar in a jitted, vmapped solve.
    result = jax.jit(
        jax.vmap(
            lambda abort: rendezvous_utility(
                nodes,
                np.ones(len(nodes), bool),
                edges,
                np.int32(0),
                positions,
                enabled,
                surplus,
                local,
                np.array(local.shape, np.int32),
                edges,
                abort_unreachable=abort,
            )
        )
    )(np.array([True, False]))

    # Then only reachable-only mode retains the later peer's utility, while
    # still zeroing the receiver's own radius and reporting the failed peer.
    np.testing.assert_allclose(result.utility, np.stack([strict, reachable]), atol=2e-7, rtol=1e-6)
    np.testing.assert_array_equal(result.unreachable, [[True, False], [True, False]])
    np.testing.assert_array_equal(result.repaired, [True, True])
    assert result.utility[1, 0] == 0
