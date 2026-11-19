"""Initialize independent robot knowledge and physically sensed team coverage."""

import jax
import jax.numpy as jnp
from jax import Array

from ir2_benchmark.communication.channel import empty_inbox
from ir2_benchmark.communication.faults import KeyPurpose, fold_key, sample_isolated
from ir2_benchmark.config import HostConfig
from ir2_benchmark.maps.prepare import PreparedMap
from ir2_benchmark.navigation.ir2_graph import initialize_graph
from ir2_benchmark.sensing.ir2_sensor import sense_ir2
from ir2_benchmark.sensing.jax_sensor import build_sensor_geometry, sense
from ir2_benchmark.types import EnvState, MapInput


def from_prepared_map(prepared: PreparedMap) -> MapInput:
    """Move host-prepared geometry into the reset function's ordinary array PyTree."""
    return MapInput(
        jnp.asarray(prepared.occupancy),
        jnp.asarray(prepared.component),
        jnp.asarray(prepared.valid_shape, jnp.int32),
        jnp.asarray(prepared.anchor, jnp.int32),
    )


def _sample_spawns(map_data: MapInput, config: HostConfig, key: Array, episode: Array) -> Array:
    """Sample distinct nearby free cells, expanding the birth region only as needed."""
    height, width = map_data.occupancy.shape
    indices = jnp.arange(height * width)
    coordinates = jnp.stack((indices % width, indices // width), axis=-1)
    squared = jnp.sum((coordinates - map_data.anchor) ** 2, axis=1)
    component = map_data.component.reshape(-1)
    sorted_distance = jnp.sort(jnp.where(component, squared, jnp.iinfo(jnp.int32).max))
    radius = jnp.maximum(
        config.spawn_radius,
        jnp.ceil(jnp.sqrt(sorted_distance[config.n_agents - 1])).astype(jnp.int32),
    )
    eligible = component & (squared <= radius * radius)
    scores = jax.random.uniform(fold_key(key, KeyPurpose.SPAWN, episode), (height * width,))
    selected = jnp.argsort(jnp.where(eligible, scores, jnp.inf))[: config.n_agents]
    return (
        jnp.broadcast_to(map_data.anchor, (config.n_slots, 2))
        .at[: config.n_agents]
        .set(coordinates[selected])
        .astype(jnp.int32)
    )


def reset(
    map_data: MapInput,
    config: HostConfig,
    key: Array,
    spawns: Array | None = None,
    episode: Array = 0,
) -> EnvState:
    """Sense once at birth; reset can be jitted/vmapped with static config and shapes."""
    positions = (
        _sample_spawns(map_data, config, key, episode)
        if spawns is None
        else (
            jnp.broadcast_to(map_data.anchor, (config.n_slots, 2))
            .at[: spawns.shape[0]]
            .set(jnp.asarray(spawns, jnp.int32))
        )
    )
    active = jnp.arange(config.n_slots) < config.n_agents
    unknown = jnp.full(map_data.occupancy.shape, 2, jnp.uint8)
    if config.navigation_mode == "ir2_graph":
        sensed = jax.vmap(
            lambda position: sense_ir2(
                map_data.occupancy, unknown, position, config.sensor_radius, map_data.valid_shape
            )
        )(positions)
    else:
        geometry = build_sensor_geometry(config.sensor_radius)
        sensed = jax.vmap(
            lambda position: sense(
                map_data.occupancy, unknown, position, geometry, map_data.valid_shape
            )
        )(positions)
    maps = jnp.where(active[:, None, None], sensed.local_map, unknown)
    physical = sensed.observed_free & active[:, None, None]
    covered = jnp.any(physical, axis=0) & map_data.component
    shape = (config.n_slots, config.n_slots)
    missing = jnp.full(shape, -1, jnp.int32)
    coverage = jnp.sum(covered) / jnp.sum(map_data.component)
    graph_memory = None
    peer_maps = None
    if config.navigation_mode == "ir2_graph":
        graph_memory = jax.vmap(
            lambda local_map, position: initialize_graph(
                local_map, position, map_data.valid_shape, config=config
            )
        )(maps, positions)
        peer_maps = jnp.full((config.n_slots, config.n_slots, *maps.shape[-2:]), 2, jnp.uint8)
    return EnvState(
        map_data.occupancy,
        map_data.component,
        map_data.valid_shape,
        positions,
        active,
        maps,
        jnp.where(active[:, None, None] & sensed.visible, 0, -1).astype(jnp.int32),
        physical,
        covered,
        jnp.where(active, config.initial_budget, 0).astype(jnp.int32),
        empty_inbox(config.n_slots),
        positions,
        jnp.broadcast_to(positions[:, None, :], (config.n_slots, config.plan_horizon, 2)),
        jnp.zeros(config.n_slots, jnp.int32),
        jnp.full(config.n_slots, -1, jnp.int32),
        jnp.zeros(config.n_slots, jnp.int32),
        jnp.zeros(config.n_slots, jnp.int32),
        jnp.zeros((*shape, 2), jnp.int32),
        jnp.zeros((*shape, 2), jnp.int32),
        jnp.zeros((*shape, config.plan_horizon, 2), jnp.int32),
        jnp.zeros(shape, jnp.int32),
        missing,
        missing,
        missing,
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        coverage >= config.coverage_target,
        jnp.bool_(0),
        key,
        jnp.asarray(episode, jnp.int32),
        sample_isolated(key, episode, active),
        jnp.zeros(config.n_slots, jnp.int32),
        jnp.zeros(config.n_slots, jnp.float32),
        jnp.zeros((config.n_slots, config.graph_capacity, 2), jnp.int32).at[:, 0].set(positions),
        active.astype(jnp.int32),
        graph_memory,
        peer_maps,
    )
