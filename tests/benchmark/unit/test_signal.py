"""Physical wall lengths use an independent per-cell slab intersection oracle."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from ir2_benchmark.communication.signal import link_signal, segment_lengths


def _obstacle_length(
    grid: NDArray[np.uint8], start: tuple[int, int], end: tuple[int, int]
) -> float:
    """Independently intersect the line with every occupied cell's closed square."""
    total = 0.0
    distance = math.dist(start, end)
    for y, x in np.argwhere(grid == 1):
        lower, upper = 0.0, 1.0
        for axis, center in enumerate((x, y)):
            difference = end[axis] - start[axis]
            if difference == 0:
                if not center - 0.5 <= start[axis] <= center + 0.5:
                    upper = -1.0
            else:
                limits = (
                    (center - 0.5 - start[axis]) / difference,
                    (center + 0.5 - start[axis]) / difference,
                )
                lower = max(lower, min(limits))
                upper = min(upper, max(limits))
        total += max(0.0, upper - lower) * distance
    return total


@pytest.mark.parametrize(
    ("start", "end", "expected_obstacle"),
    [
        ((0, 3), (6, 3), 1.0),
        ((3, 0), (3, 6), 3.0),
        ((0, 0), (6, 6), math.sqrt(2)),
        ((6, 6), (0, 0), math.sqrt(2)),
        ((3, 3), (3, 3), 0.0),
    ],
)
def test_physical_lengths_when_axis_diagonal_or_colocated(
    start: tuple[int, int],
    end: tuple[int, int],
    expected_obstacle: float,
) -> None:
    # Given: a three-cell vertical wall, each square centered at an integer coordinate.
    grid = jnp.zeros((7, 7), jnp.uint8).at[2:5, 3].set(1)
    # When: exact crossings partition the physical segment.
    eager = segment_lengths(grid, jnp.asarray(start), jnp.asarray(end))
    compiled = jax.jit(segment_lengths)(grid, jnp.asarray(start), jnp.asarray(end))
    # Then: measured obstacle lengths are geometric, with free + obstacle = total.
    assert float(eager.obstacle_distance) == pytest.approx(expected_obstacle, abs=2e-6)
    assert float(eager.distance) == pytest.approx(math.dist(start, end), abs=2e-6)
    assert float(eager.obstacle_distance + eager.free_distance) == pytest.approx(
        math.dist(start, end), abs=2e-6
    )
    np.testing.assert_allclose(compiled, eager, atol=2e-6)


def test_signal_when_touching_wall_only_at_corner() -> None:
    # Given: two walls share only a zero-measure point with the diagonal segment.
    grid = jnp.zeros((5, 5), jnp.uint8).at[1, 2].set(1).at[2, 1].set(1)
    # When: trace the diagonal past the touching corners.
    result = jax.jit(segment_lengths)(grid, jnp.asarray([0, 0]), jnp.asarray([4, 4]))
    # Then: neither corner contributes a fictitious obstacle-cell count.
    assert float(result.obstacle_distance) == pytest.approx(0.0, abs=1e-6)
    assert float(result.free_distance) == pytest.approx(4 * math.sqrt(2), abs=1e-6)


@pytest.mark.parametrize(
    ("size", "occupied", "expected"),
    [
        (36, False, -51.0),
        (71, False, -51.0 - 20 * math.log10(2)),
        (11, True, -91.0),
        (1, True, -51.0),
    ],
)
def test_signal_when_piecewise_loss_or_colocation_applies(
    *,
    size: int,
    occupied: bool,
    expected: float,
) -> None:
    # Given: a known physical distance and an all-free or all-obstacle segment.
    grid = jnp.full((1, size), int(occupied), jnp.uint8)
    # When: apply the approved uncalibrated propagation constants.
    result = jax.jit(link_signal)(grid, jnp.asarray([0, 0]), jnp.asarray([size - 1, 0]))
    # Then: analytic reference arithmetic defines the loss, with no log(0).
    assert math.isfinite(float(result.strength))
    assert float(result.strength) == pytest.approx(expected, abs=1e-5)


def test_signal_when_strength_equals_threshold_is_unreachable() -> None:
    # Given: a short free segment has strength exactly -20 -31 = -51.
    grid = jnp.zeros((2, 2), jnp.uint8)
    # When: threshold equals that received strength.
    result = jax.jit(link_signal)(grid, jnp.asarray([0, 0]), jnp.asarray([1, 0]), threshold=-51.0)
    # Then: equality fails the strict comparison.
    assert not bool(result.reachable)


def test_signal_when_jit_vmap_matches_independent_random_geometry() -> None:
    # Given: asymmetric random maps and segments; the oracle never uses production helpers.
    random = np.random.default_rng(31)
    grids = random.integers(0, 2, size=(12, 7, 9), dtype=np.uint8)
    starts = np.column_stack((random.integers(0, 9, 12), random.integers(0, 7, 12)))
    ends = np.column_stack((random.integers(0, 9, 12), random.integers(0, 7, 12)))
    expected = [
        _obstacle_length(grid, tuple(start), tuple(end))
        for grid, start, end in zip(grids, starts, ends, strict=True)
    ]
    # When: both eager and compiled vectorized paths evaluate the same true maps.
    eager = jax.vmap(segment_lengths)(jnp.asarray(grids), jnp.asarray(starts), jnp.asarray(ends))
    compiled = jax.jit(jax.vmap(segment_lengths))(
        jnp.asarray(grids), jnp.asarray(starts), jnp.asarray(ends)
    )
    # Then: every pair agrees with independent cell intersection geometry.
    np.testing.assert_allclose(eager.obstacle_distance, expected, atol=3e-6)
    np.testing.assert_allclose(compiled.obstacle_distance, expected, atol=3e-6)
    np.testing.assert_allclose(
        compiled.free_distance + compiled.obstacle_distance,
        np.linalg.norm(ends - starts, axis=1),
        atol=3e-6,
    )
