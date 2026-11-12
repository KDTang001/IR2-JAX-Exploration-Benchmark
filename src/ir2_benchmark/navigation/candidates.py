"""Deterministic farthest-point candidates drawn only from local reachable free cells."""

from typing import Final, NamedTuple

import jax.numpy as jnp
from jax import Array, lax

from ir2_benchmark.navigation.jax_path import SearchTree, bfs

UNKNOWN: Final = 2


class CandidateSet(NamedTuple):
    """Wait in slot zero, then at most K waypoints with masks and graph distances."""

    positions: Array
    mask: Array
    distances: Array


def frontier_mask(known_map: Array, valid_shape: Array | tuple[int, int] | None = None) -> Array:
    """Mark known-free cells with an unknown four-neighbor, excluding map exterior."""
    limits = jnp.asarray(known_map.shape if valid_shape is None else valid_shape)
    y, x = jnp.indices(known_map.shape)
    valid = (y < limits[0]) & (x < limits[1])
    unknown = jnp.pad((known_map == UNKNOWN) & valid, 1, constant_values=False)
    adjacent = unknown[:-2, 1:-1] | unknown[2:, 1:-1] | unknown[1:-1, :-2] | unknown[1:-1, 2:]
    return (known_map == 0) & valid & adjacent


def waypoint_candidates(
    known_map: Array,
    position: Array,
    existing_goal: Array,
    max_candidates: int = 64,
    tree: SearchTree | None = None,
) -> CandidateSet:
    """Preserve a valid old goal, then maximize Euclidean separation deterministically.

    Sampling is seeded by the current position and retained goal. Equal scores use
    row-major coordinate order (y, then x). max_candidates is static under jit.
    A shared full BFS tree avoids repeated searches for candidate features/routes.
    """
    position = jnp.asarray(position, jnp.int32)
    existing_goal = jnp.asarray(existing_goal, jnp.int32)
    search = bfs(known_map, position) if tree is None else tree
    height, width = known_map.shape
    ids = jnp.arange(height * width, dtype=jnp.int32)
    coordinates = jnp.stack((ids % width, ids // width), axis=-1)
    distances = search.distances.reshape(-1)
    available = (distances >= 0) & jnp.any(coordinates != position, axis=1)
    separation = jnp.sum((coordinates - position) ** 2, axis=1)
    gx, gy = existing_goal[0], existing_goal[1]
    old_id = jnp.clip(gy, 0, height - 1) * width + jnp.clip(gx, 0, width - 1)
    preserve = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height) & available[old_id]
    positions = jnp.broadcast_to(position, (max_candidates + 1, 2))
    mask = jnp.zeros(max_candidates + 1, bool).at[0].set(True)
    candidate_distances = jnp.zeros(max_candidates + 1, jnp.int32)

    def choose(
        index: int, carry: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array]:
        remaining, nearest, selected, valid, lengths = carry
        farthest = jnp.argmax(jnp.where(remaining, nearest, -1))
        node = jnp.where((index == 1) & preserve, old_id, farthest)
        exists = jnp.any(remaining)
        point = jnp.where(exists, coordinates[node], position)
        selected = selected.at[index].set(point)
        valid = valid.at[index].set(exists)
        lengths = lengths.at[index].set(jnp.where(exists, distances[node], 0))
        remaining = remaining.at[node].set(False)
        nearest = jnp.minimum(nearest, jnp.sum((coordinates - point) ** 2, axis=1))
        return remaining, nearest, selected, valid, lengths

    _, _, selected, valid, lengths = lax.fori_loop(
        1,
        max_candidates + 1,
        choose,
        (available, separation, positions, mask, candidate_distances),
    )
    return CandidateSet(selected, valid, lengths)
