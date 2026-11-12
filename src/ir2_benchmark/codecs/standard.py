"""Standard occupancy, waypoint and plan payloads as actual little-endian bytes."""

from typing import Final, NamedTuple

import jax.numpy as jnp
from jax import Array

from ir2_benchmark.codecs.packet import (
    Codec,
    Packet,
    decode_payload_origin,
    make_packet,
    read_uint_le,
    uint_le,
)

UNKNOWN: Final = 2


class DecodedPlan(NamedTuple):
    """The horizon is static; only valid_length positions carry a prediction."""

    source: Array
    goal: Array
    positions: Array
    valid_length: Array
    valid_mask: Array


class DecodedMapDelta(NamedTuple):
    """At most 47 entries, addressed as [x,y], with a static validity mask."""

    coordinates: Array
    states: Array
    count: Array
    valid_mask: Array


class MapDeltaSelection(NamedTuple):
    """A local observation-only selection and its unacknowledged rotation cursor."""

    coordinates: Array
    states: Array
    count: Array
    next_cursor: Array


def encode_position(
    position: Array, sender: Array, sequence: Array, generated_event: Array, n_agents: Array
) -> Packet:
    """Encode x/y as two unsigned 16-bit coordinates."""
    return make_packet(
        Codec.POSITION,
        uint_le(position, 2).reshape(-1),
        4,
        sender,
        sequence,
        generated_event,
        n_agents,
    )


def encode_goal(
    goal: Array, sender: Array, sequence: Array, generated_event: Array, n_agents: Array
) -> Packet:
    """Encode a goal using the same coordinate layout as POSITION."""
    return make_packet(
        Codec.GOAL, uint_le(goal, 2).reshape(-1), 4, sender, sequence, generated_event, n_agents
    )


def decode_position(packet: Packet) -> Array:
    """Decode a structurally valid POSITION or GOAL without consulting a map."""
    b = decode_payload_origin(packet).payload
    return jnp.stack((read_uint_le(b, 0, 2), read_uint_le(b, 2, 2))).astype(jnp.int32)


# Each source/header field is independently mapped by batched device callers.
def encode_plan(  # noqa: PLR0913, PLR0917
    source: Array,
    goal: Array,
    positions: Array,
    valid_length: Array,
    sender: Array,
    sequence: Array,
    generated_event: Array,
    n_agents: Array,
) -> Packet:
    """Serialize every horizon slot; five positions make a 45-byte packet."""
    payload = jnp.concatenate(
        (
            uint_le(source, 2).reshape(-1),
            uint_le(goal, 2).reshape(-1),
            jnp.asarray([valid_length], jnp.uint8),
            uint_le(positions, 2).reshape(-1),
        )
    )
    return make_packet(
        Codec.PLAN, payload, payload.shape[0], sender, sequence, generated_event, n_agents
    )


def decode_plan(packet: Packet, plan_horizon: int = 5) -> DecodedPlan:
    """Decode bounded plan positions, independent of their map validity."""
    b = decode_payload_origin(packet).payload
    source = jnp.stack((read_uint_le(b, 0, 2), read_uint_le(b, 2, 2))).astype(jnp.int32)
    goal = jnp.stack((read_uint_le(b, 4, 2), read_uint_le(b, 6, 2))).astype(jnp.int32)
    data = b[9 : 9 + 4 * plan_horizon].reshape(plan_horizon, 2, 2).astype(jnp.int32)
    positions = data[..., 0] + 256 * data[..., 1]
    count = b[8].astype(jnp.int32)
    return DecodedPlan(source, goal, positions, count, jnp.arange(plan_horizon) < count)


# Entries and their wire header stay separate to support independent vmap axes.
def encode_map_delta(  # noqa: PLR0913, PLR0917
    coordinates: Array,
    states: Array,
    count: Array,
    sender: Array,
    sequence: Array,
    generated_event: Array,
    n_agents: Array,
) -> Packet:
    """Encode up to 47 static entries; only ``count`` entries are transmitted."""
    entries = jnp.concatenate(
        (uint_le(coordinates, 2).reshape(-1, 4), states.astype(jnp.uint8)[:, None]), axis=1
    )
    payload = jnp.concatenate((jnp.asarray([count], jnp.uint8), entries.reshape(-1)))
    return make_packet(
        Codec.MAP_DELTA, payload, 1 + 5 * count, sender, sequence, generated_event, n_agents
    )


def decode_map_delta(packet: Packet) -> DecodedMapDelta:
    """Return exactly 47 slots so receivers can merge maps under jit."""
    b = decode_payload_origin(packet).payload
    data = b[1:236].reshape(47, 5).astype(jnp.int32)
    coordinates = jnp.stack((data[:, 0] + 256 * data[:, 1], data[:, 2] + 256 * data[:, 3]), axis=-1)
    count = b[0].astype(jnp.int32)
    return DecodedMapDelta(coordinates, data[:, 4], count, jnp.arange(47) < count)


def select_map_delta(local_map: Array, observed_event: Array, cursor: Array) -> MapDeltaSelection:
    """Sort by recency then x/y, rotating without assuming previous delivery."""
    height, width = local_map.shape
    y, x = jnp.indices((height, width))
    known = (local_map.reshape(-1) < UNKNOWN) & (observed_event.reshape(-1) >= 0)
    times = jnp.where(known, observed_event.reshape(-1), -1)
    order = jnp.lexsort((y.reshape(-1), x.reshape(-1), -times))
    size = jnp.sum(known, dtype=jnp.int32)
    count = jnp.minimum(size, 47)
    ranks = (cursor + jnp.arange(47)) % jnp.maximum(size, 1)
    selected = order[ranks]
    valid = jnp.arange(47) < count
    coordinates = jnp.stack((selected % width, selected // width), axis=-1)
    return MapDeltaSelection(
        jnp.where(valid[:, None], coordinates, 0),
        jnp.where(valid, local_map.reshape(-1)[selected], 0),
        count,
        (cursor + count) % jnp.maximum(size, 1),
    )
