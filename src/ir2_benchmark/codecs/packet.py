"""A 16-byte header and at most 240 payload bytes, without host callbacks."""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

HEADER_SIZE: Final = 16
MAX_PACKET_SIZE: Final = 256
MAX_PAYLOAD_SIZE: Final = 240
VERSION: Final = 1
HEADER_STRUCT: Final = struct.Struct("<BBBBIIHBB")
COORDINATE_BYTES: Final = 4
MAX_MAP_DELTA_ENTRIES: Final = 47
MAX_AGENTS: Final = 6


class Codec(IntEnum):
    """Standard IDs; IDs 128 through 255 are participant-defined codecs."""

    POSITION = 1
    GOAL = 2
    PLAN = 3
    MAP_DELTA = 4
    FORWARD = 5
    CUSTOM = 128


class Packet(NamedTuple):
    """Only ``length`` bytes of the static uint8 buffer incur a charge."""

    buffer: Array
    length: Array


class Header(NamedTuple):
    """Decoded scalar JAX fields in their wire order."""

    version: Array
    codec: Array
    sender: Array
    flags: Array
    sequence: Array
    generated_event: Array
    payload_length: Array
    recipient_mask: Array
    reserved: Array


class PayloadOrigin(NamedTuple):
    """Effective payload with its original author and generation event."""

    codec: Array
    author: Array
    generated_event: Array
    payload: Array
    payload_length: Array


@dataclass(frozen=True, slots=True)
class PacketError(ValueError):
    """A malformed host-supplied packet or field."""

    field: str
    detail: str

    def __str__(self) -> str:
        """Keep the failing protocol field visible at host boundaries."""
        return f"{self.field}: {self.detail}"


def uint_le(value: Array, width: int) -> Array:
    """Encode scalar or array integers, appending a little-endian byte axis."""
    shifts = jnp.arange(width, dtype=jnp.uint32) * 8
    return ((jnp.asarray(value, jnp.uint32)[..., None] >> shifts) & 255).astype(jnp.uint8)


def read_uint_le(buffer: Array, offset: int, width: int) -> Array:
    """Read unsigned little-endian integers without enabling JAX x64."""
    shifts = jnp.arange(width, dtype=jnp.uint32) * 8
    return jnp.sum(
        buffer[..., offset : offset + width].astype(jnp.uint32) << shifts, axis=-1, dtype=jnp.uint32
    )


def recipient_mask(sender: Array, n_agents: Array) -> Array:
    """Broadcast addresses every other active contiguous robot ID."""
    return ((jnp.uint32(1) << jnp.asarray(n_agents, jnp.uint32)) - 1) ^ (
        jnp.uint32(1) << jnp.asarray(sender, jnp.uint32)
    )


# Wire header fields remain explicit so jit/vmap callers can map each scalar independently.
def make_packet(  # noqa: PLR0913, PLR0917
    codec: Array,
    payload: Array,
    payload_length: Array,
    sender: Array,
    sequence: Array,
    generated_event: Array,
    n_agents: Array,
) -> Packet:
    """Build a bounded device packet; validate its public structure before sending."""
    prefix = jnp.asarray([VERSION, codec, sender, 0], dtype=jnp.uint8)
    header = jnp.concatenate(
        (
            prefix,
            uint_le(sequence, 4),
            uint_le(generated_event, 4),
            uint_le(payload_length, 2),
            jnp.asarray([recipient_mask(sender, n_agents), 0], jnp.uint8),
        )
    )
    padded = jnp.pad(payload.astype(jnp.uint8), (0, MAX_PAYLOAD_SIZE - payload.shape[0]))
    padded = jnp.where(jnp.arange(MAX_PAYLOAD_SIZE) < payload_length, padded, 0)
    return Packet(
        jnp.concatenate((header, padded)), jnp.asarray(HEADER_SIZE + payload_length, jnp.int32)
    )


def decode_header(packet: Packet) -> Header:
    """Decode only the fixed header; validity is a separate device boolean."""
    b = packet.buffer
    return Header(
        b[0],
        b[1],
        b[2],
        b[3],
        read_uint_le(b, 4, 4),
        read_uint_le(b, 8, 4),
        read_uint_le(b, 12, 2),
        b[14],
        b[15],
    )


def validate_packet(
    packet: Packet, n_agents: Array, plan_horizon: int = 5, addressing: str = "broadcast"
) -> Array:
    """Check public structure only, returning a scalar mask under jit/vmap."""
    h = decode_header(packet)
    origin = decode_payload_origin(packet)
    b = origin.payload
    count = b[0].astype(jnp.int32)
    entry_states = b[5:236:5]
    entries_valid = jnp.all((jnp.arange(47) >= count) | (entry_states <= 1))
    standard_valid = (
        (
            ((origin.codec == Codec.POSITION) | (origin.codec == Codec.GOAL))
            & (origin.payload_length == COORDINATE_BYTES)
        )
        | (
            (origin.codec == Codec.PLAN)
            & (origin.payload_length == 9 + 4 * plan_horizon)
            & (b[8] <= plan_horizon)
        )
        | (
            (origin.codec == Codec.MAP_DELTA)
            & (count <= MAX_MAP_DELTA_ENTRIES)
            & (origin.payload_length == 1 + 5 * count)
            & entries_valid
        )
        | (origin.codec >= Codec.CUSTOM)
    )
    allowed = recipient_mask(h.sender, n_agents)
    address_valid = (
        (h.recipient_mask == allowed)
        if addressing == "broadcast"
        else (
            (h.recipient_mask != 0) & ((h.recipient_mask.astype(jnp.uint32) | allowed) == allowed)
        )
    )
    if addressing not in ("broadcast", "addressed"):
        raise ValueError("unknown packet addressing mode")
    return (
        (h.version == VERSION)
        & (h.flags == 0)
        & (h.reserved == 0)
        & (h.sender < n_agents)
        & (n_agents >= 1)
        & (n_agents <= MAX_AGENTS)
        & address_valid
        & (h.payload_length <= MAX_PAYLOAD_SIZE)
        & (origin.payload_length >= 0)
        & (origin.payload_length <= MAX_PAYLOAD_SIZE)
        & (origin.author < n_agents)
        & (packet.length == HEADER_SIZE + h.payload_length)
        & (packet.length >= HEADER_SIZE)
        & (packet.length <= MAX_PACKET_SIZE)
        & standard_valid
    )


def packet_from_bytes(
    raw: bytes, n_agents: int, plan_horizon: int = 5, addressing: str = "broadcast"
) -> Packet:
    """Parse an untrusted host byte string, refusing malformed protocol fields."""
    if not HEADER_SIZE <= len(raw) <= MAX_PACKET_SIZE:
        raise PacketError(field="length", detail="expected between 16 and 256 bytes")
    packet = Packet(
        jnp.asarray(np.frombuffer(raw.ljust(MAX_PACKET_SIZE, b"\x00"), np.uint8)),
        jnp.asarray(len(raw), jnp.int32),
    )
    if not bool(validate_packet(packet, n_agents, plan_horizon, addressing)):
        raise PacketError(
            field="structure", detail="invalid header, codec, payload length or standard payload"
        )
    return packet


def encode_packet(
    header: tuple[int, int, int, int, int, int, int, int, int],
    payload: bytes,
    n_agents: int,
    plan_horizon: int = 5,
    addressing: str = "broadcast",
) -> Packet:
    """Host encoder using the exact documented struct layout."""
    if header[6] != len(payload):
        raise PacketError(field="payload_length", detail="does not match actual payload")
    try:
        wire_header = HEADER_STRUCT.pack(*header)
    except struct.error as error:
        raise PacketError(field="header", detail=str(error)) from error
    return packet_from_bytes(wire_header + payload, n_agents, plan_horizon, addressing)


def decode_packet_host(
    packet: Packet, n_agents: int, plan_horizon: int = 5, addressing: str = "broadcast"
) -> tuple[tuple[int, ...], bytes]:
    """Return checked host fields and only the actual, unpadded payload."""
    raw = np.asarray(packet.buffer, np.uint8).tobytes()[: int(packet.length)]
    packet_from_bytes(raw, n_agents, plan_horizon, addressing)
    return HEADER_STRUCT.unpack(raw[:HEADER_SIZE]), raw[HEADER_SIZE:]


def decode_payload_origin(packet: Packet) -> PayloadOrigin:
    """Unwrap a single charged FORWARD envelope; nesting is never valid."""
    h = decode_header(packet)
    forwarded = h.codec == Codec.FORWARD
    payload = packet.buffer[HEADER_SIZE:]
    unwrapped = jnp.pad(payload[6:], (0, 6))
    return PayloadOrigin(
        jnp.where(forwarded, payload[0], h.codec),
        jnp.where(forwarded, payload[1], h.sender),
        jnp.where(forwarded, read_uint_le(payload, 2, 4), h.generated_event),
        jnp.where(forwarded, unwrapped, payload),
        h.payload_length.astype(jnp.int32) - jnp.where(forwarded, 6, 0),
    )


def forward_packet(
    packet: Packet, sender: Array, sequence: Array, generated_event: Array, n_agents: Array
) -> Packet:
    """Explicit forwarding charges six origin bytes; over-capacity results are invalid."""
    origin = decode_payload_origin(packet)
    envelope = jnp.concatenate(
        (
            jnp.asarray([origin.codec, origin.author], jnp.uint8),
            uint_le(origin.generated_event, 4),
            origin.payload[:234],
        )
    )
    return make_packet(
        Codec.FORWARD,
        envelope,
        origin.payload_length + 6,
        sender,
        sequence,
        generated_event,
        n_agents,
    )
