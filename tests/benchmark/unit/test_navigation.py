"""Navigation fixtures distinguish real graph distance from straight-line travel."""

import numpy as np
import pytest

from ir2_benchmark.navigation import reference


def test_u_shaped_wall_requires_full_seven_edge_detour() -> None:
    # Given: a narrow U whose target is close in Euclidean distance.
    local = np.ones((5, 6), dtype=np.uint8)
    route = np.array([[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [4, 3], [4, 2], [4, 1]])
    local[route[:, 1], route[:, 0]] = 0
    # When: taking at most two edges toward the target.
    result = reference.navigate(local, route[0], route[-1], 2, 5)
    # Then: the path follows the U and the forecast advances by own-action distances.
    assert result.path_length == 7  # noqa: PLR2004
    assert result.distance == 2  # noqa: PLR2004
    np.testing.assert_array_equal(result.position, [1, 3])
    np.testing.assert_array_equal(result.path[:8], route)
    np.testing.assert_array_equal(result.plan, [[1, 3], [3, 3], [4, 2], [4, 1], [4, 1]])
    assert result.plan_valid_length == 4  # noqa: PLR2004


@pytest.mark.parametrize("goal", [(2, 0), (-1, 0), (3, 3)])
def test_unknown_outside_or_disconnected_goal_waits(goal: tuple[int, int]) -> None:
    # Given: a robot, unknown cell, and disconnected free target.
    local = np.ones((4, 4), dtype=np.uint8)
    local[0, :2] = 0
    local[0, 2] = 2
    local[3, 3] = 0
    # When: an invalid goal is requested.
    result = reference.navigate(local, np.array([0, 0]), np.array(goal), 3, 3)
    # Then: the action is rejected without moving or inventing another path.
    assert result.invalid_action
    assert result.distance == 0
    np.testing.assert_array_equal(result.position, [0, 0])
    np.testing.assert_array_equal(result.plan, [[0, 0]] * 3)


def test_wait_at_own_cell_is_valid() -> None:
    # Given: a robot on a free cell.
    local = np.zeros((3, 3), dtype=np.uint8)
    # When: the goal is its current position.
    result = reference.navigate(local, np.array([1, 1]), np.array([1, 1]), 2, 3)
    # Then: waiting is a valid zero-distance action.
    assert not result.invalid_action
    assert result.distance == 0
    assert result.plan_valid_length == 0


def test_short_goal_moves_only_one_edge() -> None:
    # Given: a neighbor goal and a larger movement budget.
    local = np.zeros((3, 3), dtype=np.uint8)
    # When: navigating to the immediate neighbor.
    result = reference.navigate(local, np.array([1, 1]), np.array([2, 1]), 6, 3)
    # Then: travel stops at the goal.
    assert result.distance == 1
    np.testing.assert_array_equal(result.position, [2, 1])


def test_shortest_path_ties_use_up_right_down_left() -> None:
    # Given: two equal-length routes to the upper-right corner.
    local = np.zeros((3, 3), dtype=np.uint8)
    # When: finding the shortest path.
    result = reference.shortest_path(local, np.array([1, 1]), np.array([2, 0]))
    # Then: up is chosen before right.
    np.testing.assert_array_equal(result.path[:3], [[1, 1], [1, 0], [2, 0]])


def test_long_serpentine_path_is_not_search_truncated() -> None:
    # Given: a corridor whose path greatly exceeds map width plus height.
    local = np.ones((11, 12), dtype=np.uint8)
    local[::2, :] = 0
    for row in range(1, 11, 2):
        local[row, 11 if row % 4 == 1 else 0] = 0
    # When: navigating between the two ends.
    result = reference.shortest_path(local, np.array([0, 0]), np.array([0, 10]))
    # Then: the entire 76-edge route is available.
    assert result.valid
    assert result.length == 76  # noqa: PLR2004


def test_jax_paths_jit_and_vmap_match_deque_oracle() -> None:
    # Given: a maze, asymmetric goals, and an independent deque oracle.
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    from ir2_benchmark.navigation.jax_path import NavigationResult, navigate  # noqa: PLC0415

    local = np.zeros((7, 9), dtype=np.uint8)
    local[1:6, 4] = 1
    local[2, 2] = 2
    start = np.array([1, 3], np.int32)
    goals = np.array([[7, 3], [2, 2], [1, 3], [8, 6], [-1, 0]], np.int32)
    expected = [reference.navigate(local, start, goal, 2, 5) for goal in goals]

    # When: applying the same navigator eagerly, compiled, and vectorized.
    def fn(goal: jax.Array) -> NavigationResult:
        return navigate(jnp.asarray(local), jnp.asarray(start), goal, 2, 5)

    eager = fn(jnp.asarray(goals[0]))
    compiled = jax.jit(fn)(goals[0])
    batched = jax.jit(jax.vmap(fn))(goals)
    # Then: graph distance, tie-break path, endpoint, and plan all match.
    np.testing.assert_array_equal(eager.path, expected[0].path)
    np.testing.assert_array_equal(compiled.path, expected[0].path)
    for index, result in enumerate(expected):
        np.testing.assert_array_equal(batched.position[index], result.position)
        np.testing.assert_array_equal(batched.path[index], result.path)
        np.testing.assert_array_equal(batched.plan[index], result.plan)
        assert int(batched.path_length[index]) == result.path_length
        assert bool(batched.invalid_action[index]) == result.invalid_action


def test_candidates_keep_old_goal_and_exclude_unreachable_unknown_cells() -> None:
    # Given: an old goal in a small component, plus unreachable and unknown cells.
    jax = pytest.importorskip("jax")

    from ir2_benchmark.navigation.candidates import waypoint_candidates  # noqa: PLC0415

    local = np.ones((4, 6), dtype=np.uint8)
    local[0, :3] = 0
    local[1, 0] = 0
    local[0, 3] = 2
    local[3, 5] = 0
    # When: requesting more candidates than can exist.
    result = jax.jit(lambda m: waypoint_candidates(m, np.array([0, 0]), np.array([1, 0]), 6))(local)
    # Then: wait is first, old goal second, and padding has no valid mask.
    np.testing.assert_array_equal(result.positions[:4], [[0, 0], [1, 0], [2, 0], [0, 1]])
    np.testing.assert_array_equal(result.mask, [True, True, True, True, False, False, False])
    np.testing.assert_array_equal(result.distances[:4], [0, 1, 2, 1])


def test_jax_full_search_handles_long_paths_and_shared_tree() -> None:
    # Given: a 76-edge corridor, exceeding any map-dimension search bound.
    jax = pytest.importorskip("jax")

    from ir2_benchmark.navigation.jax_path import bfs, navigate  # noqa: PLC0415

    local = np.ones((11, 12), dtype=np.uint8)
    local[::2, :] = 0
    for row in range(1, 11, 2):
        local[row, 11 if row % 4 == 1 else 0] = 0
    start, goal = np.array([0, 0], np.int32), np.array([0, 10], np.int32)
    expected = reference.navigate(local, start, goal, 9, 5)
    # When: one full search tree is used to construct a compiled movement result.
    result = jax.jit(lambda world: navigate(world, start, goal, 9, 5, bfs(world, start)))(local)
    # Then: there is no short iteration cutoff, and the complete path is retained.
    assert int(result.path_length) == 76  # noqa: PLR2004
    np.testing.assert_array_equal(result.path, expected.path)
    np.testing.assert_array_equal(result.plan, expected.plan)


def test_candidate_ties_use_row_major_order_under_vmap() -> None:
    # Given: four corners equidistant from the center and invalid previous goals.
    jax = pytest.importorskip("jax")

    from ir2_benchmark.navigation.candidates import waypoint_candidates  # noqa: PLC0415

    worlds = np.zeros((2, 3, 3), np.uint8)
    # When: deterministic farthest-point selection runs through a compiled vmap.
    result = jax.jit(
        jax.vmap(lambda world: waypoint_candidates(world, np.array([1, 1]), np.array([-1, -1]), 2))
    )(worlds)
    # Then: the upper-left corner wins first; equal remaining scores keep row order.
    np.testing.assert_array_equal(result.positions, [[[1, 1], [0, 0], [2, 0]]] * 2)


def test_chunked_bfs_full_distances_and_parents_match_single_node_fifo() -> None:
    # Given: a wide wavefront exceeding256 nodes, so chunks straddle BFS levels.
    jax = pytest.importorskip("jax")

    from ir2_benchmark.navigation.jax_path import bfs  # noqa: PLC0415

    local = np.zeros((160, 160), np.uint8)
    local[20:135, 52] = 1
    local[80, 52] = 0
    start = np.array([80, 80], np.int32)
    expected = reference.bfs(local, start)
    # When: the full map is searched by the compiled chunked FIFO.
    actual = jax.jit(lambda world: bfs(world, start))(local)
    # Then: every discovery has the exact single-node distance and tie-break parent.
    np.testing.assert_array_equal(actual.distances, expected.distances)
    np.testing.assert_array_equal(actual.parents, expected.parents)


def test_frontiers_exclude_unknown_padding_outside_rectangular_valid_shape() -> None:
    # Given: a fully known3x5 valid region padded into an8x8 unknown local bucket.
    jax = pytest.importorskip("jax")

    from ir2_benchmark.navigation.candidates import frontier_mask  # noqa: PLC0415

    local = np.full((8, 8), 2, np.uint8)
    local[:3, :5] = 0
    # When: extracting frontiers with the public valid shape.
    frontiers = jax.jit(frontier_mask)(local, np.array([3, 5], np.int32))
    # Then: map exterior is not an unexplored destination.
    np.testing.assert_array_equal(frontiers, np.zeros_like(local, bool))
