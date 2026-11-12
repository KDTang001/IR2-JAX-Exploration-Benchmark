"""Independent geometry fixtures, including closed-cell corner occlusion."""

import numpy as np
import pytest

from ir2_benchmark.sensing import reference


def test_first_wall_is_visible_and_hides_cells_behind_it() -> None:
    # Given: an otherwise empty world and one intervening wall.
    truth = np.zeros((7, 7), dtype=np.uint8)
    truth[3, 3] = 1
    local = np.full_like(truth, 2)
    # When: sensing from the left of that wall.
    result = reference.sense(truth, local, np.array([1, 3]), 5)
    # Then: the first wall is observed, without revealing beyond it or mutating inputs.
    assert result.local_map[3, 2] == 0
    assert result.local_map[3, 3] == 1
    assert result.local_map[3, 4] == 2  # noqa: PLR2004
    assert not result.observed_free[3, 3]
    np.testing.assert_array_equal(local, np.full_like(local, 2))


def test_radius_one_has_exact_cross_footprint() -> None:
    # Given: an empty world with a centered robot.
    truth = np.zeros((5, 5), dtype=np.uint8)
    expected = np.zeros_like(truth, dtype=bool)
    expected[2, 1:4] = True
    expected[1:4, 2] = True
    # When: the sensing radius is exactly one.
    visible = reference.visible_mask(truth, np.array([2, 2]), 1)
    # Then: diagonal cells are outside the Euclidean disk.
    np.testing.assert_array_equal(visible, expected)


@pytest.mark.parametrize("wall", [(2, 1), (1, 2)])
def test_touching_either_corner_wall_blocks_diagonal(wall: tuple[int, int]) -> None:
    # Given: a wall touching the line from (1,1) to (2,2) at its corner.
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[wall[1], wall[0]] = 1
    # When: observing that diagonal target.
    visible = reference.visible_mask(truth, np.array([1, 1]), 3)
    # Then: conservative closed-cell intersection blocks it.
    assert not visible[2, 2]
    assert visible[wall[1], wall[0]]


def test_padding_and_boundary_do_not_create_clipped_observations() -> None:
    # Given: an empty 3x4 world padded to 6x6 with occupied cells.
    truth = np.ones((6, 6), dtype=np.uint8)
    truth[:3, :4] = 0
    # When: sensing at its top left corner with the valid shape supplied.
    visible = reference.visible_mask(truth, np.array([0, 0]), 2, (3, 4))
    # Then: only in-bounds centers inside the disk are visible.
    expected = np.zeros_like(truth, dtype=bool)
    expected[:3, :3] = [[True, True, True], [True, True, False], [True, False, False]]
    np.testing.assert_array_equal(visible, expected)


def test_zero_radius_observes_only_own_cell() -> None:
    # Given: a free world.
    truth = np.zeros((3, 3), dtype=np.uint8)
    # When: observing with a zero-radius sensor.
    visible = reference.visible_mask(truth, np.array([1, 1]), 0)
    # Then: the single co-located cell is visible.
    expected = np.zeros_like(truth, dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(visible, expected)


@pytest.mark.parametrize("radius", [0, 1, 2, 5])
def test_jax_eager_jit_and_vmap_match_independent_intersection_oracle(radius: int) -> None:
    # Given: randomized small maps and asymmetric observation positions.
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    from ir2_benchmark.sensing.jax_sensor import build_sensor_geometry, sense  # noqa: PLC0415

    rng = np.random.default_rng(913)
    truths = (rng.random((6, 7, 9)) < 0.23).astype(np.uint8)  # noqa: PLR2004
    positions = np.array([[1, 3], [0, 0], [8, 6], [4, 2], [2, 5], [3, 3]], np.int32)
    truths[np.arange(6), positions[:, 1], positions[:, 0]] = 0
    locals_ = np.full_like(truths, 2)
    geometry = build_sensor_geometry(radius)
    expected = np.stack(
        [
            reference.sense(world, local, position, radius).local_map
            for world, local, position in zip(truths, locals_, positions, strict=True)
        ]
    )

    # When: sensing eagerly, compiled, and through a batch transform.
    def fn(world: jax.Array, local: jax.Array, position: jax.Array) -> jax.Array:
        return sense(world, local, position, geometry).local_map

    eager = fn(jnp.asarray(truths[0]), jnp.asarray(locals_[0]), jnp.asarray(positions[0]))
    compiled = jax.jit(fn)(truths[0], locals_[0], positions[0])
    batched = jax.jit(jax.vmap(fn))(truths, locals_, positions)
    # Then: all three use exactly the oracle's visible cells.
    np.testing.assert_array_equal(eager, expected[0])
    np.testing.assert_array_equal(compiled, expected[0])
    np.testing.assert_array_equal(batched, expected)


def test_jax_corner_blocker_and_padding_are_conservative() -> None:
    # Given: an exact diagonal corner blocker in a map padded at right and bottom.
    jax = pytest.importorskip("jax")

    from ir2_benchmark.sensing.jax_sensor import build_sensor_geometry, visible_mask  # noqa: PLC0415

    truth = np.ones((6, 6), np.uint8)
    truth[:4, :4] = 0
    truth[1, 2] = 1
    geometry = build_sensor_geometry(4)
    expected = reference.visible_mask(truth, np.array([1, 1]), 4, (4, 4))
    # When: the valid shape is supplied dynamically to a compiled visibility kernel.
    visible = jax.jit(lambda world, shape: visible_mask(world, np.array([1, 1]), geometry, shape))(
        truth, np.array([4, 4], np.int32)
    )
    # Then: neither corner peeking nor padded obstacles become observations.
    np.testing.assert_array_equal(visible, expected)
    assert not visible[2, 2]
    assert not np.any(visible[4:, :])
    assert not np.any(visible[:, 4:])
