"""Nearest frontier by actual local graph distance, with deterministic coordinate ties."""

import jax.numpy as jnp

from ir2_benchmark.agents.fixed_comm import fixed_communication
from ir2_benchmark.config import HostConfig
from ir2_benchmark.navigation.candidates import frontier_mask
from ir2_benchmark.types import Action, LocalObservation, make_action


def nearest_frontier_action(observation: LocalObservation, _config: HostConfig) -> Action:
    """Use offered graph edges; retain the explicit legacy grid reference policy."""
    if observation.tree is None:
        candidates = observation.candidates
        frontier = frontier_mask(observation.local_map, observation.valid_shape)
        points = candidates.positions
        eligible = candidates.mask & (candidates.distances > 0)
        at_frontier = eligible & frontier[points[:, 1], points[:, 0]]
        nearest = jnp.argmin(jnp.where(at_frontier, candidates.distances, jnp.inf))
        outward = jnp.argmax(jnp.where(eligible, candidates.distances, -1))
        index = jnp.where(jnp.any(at_frontier), nearest, outward)
        send, codec = fixed_communication(observation)
        return make_action(points[index], send, codec)
    distances = observation.tree.distances
    eligible = frontier_mask(observation.local_map, observation.valid_shape) & (distances > 0)
    scores = jnp.where(eligible, distances, jnp.iinfo(jnp.int32).max)
    index = jnp.argmin(scores.reshape(-1))
    width = observation.local_map.shape[1]
    goal = jnp.where(
        jnp.any(eligible), jnp.array([index % width, index // width]), observation.position
    )
    send, codec = fixed_communication(observation)
    return make_action(goal, send, codec)
