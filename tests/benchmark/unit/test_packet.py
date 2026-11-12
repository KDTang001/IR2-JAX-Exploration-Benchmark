"""Wire assertions use Python struct and explicit independent payload bytes."""

# Independent wire-size expectations must not be imported from production constants.
# ruff: noqa: PLR2004

import struct
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from ir2_benchmark.codecs.packet import (
    Codec,
    Packet,
    PacketError,
    decode_header,
    decode_packet_host,
    decode_payload_origin,
    encode_packet,
    forward_packet,
    make_packet,
    packet_from_bytes,
    validate_packet,
)
from ir2_benchmark.codecs.standard import (
    decode_map_delta,
    decode_plan,
    decode_position,
    encode_goal,
    encode_map_delta,
    encode_plan,
    encode_position,
    select_map_delta,
)


@pytest.mark.parametrize(("encoder", "codec"), [(encode_position, 1), (encode_goal, 2)])
@pytest.mark.parametrize("coordinate", [(3, 11), (65535, 0)])
def test_coordinate_wire_when_asymmetric_or_maximum(
    encoder: Callable[[Array, int, int, int, int], Packet],
    codec: int,
    coordinate: tuple[int, int],
) -> None:
    # Given: wire expectations do not call any production packing helper.
    expected = struct.pack("<BBBBIIHBBHH", 1, codec, 0, 0, 0x01020304, 19, 4, 14, 0, *coordinate)
    # When: eager and compiled encoders build the real packet.
    eager = encoder(jnp.asarray(coordinate), 0, 0x01020304, 19, 4)
    compiled = jax.jit(encoder)(jnp.asarray(coordinate), 0, 0x01020304, 19, 4)
    # Then: exact wire bytes, length and decoding agree independently.
    assert int(eager.length) == 16 + 2 + 2
    assert bytes(np.asarray(eager.buffer[:20])) == expected
    np.testing.assert_array_equal(compiled.buffer, eager.buffer)
    np.testing.assert_array_equal(decode_position(compiled), coordinate)
    assert bool(jax.jit(validate_packet)(compiled, 4))


def test_plan_wire_when_horizon_is_five() -> None:
    # Given: 4-byte source, 4-byte goal, 1-byte count, and five 4-byte points.
    points = jnp.asarray([[3, 11], [4, 11], [5, 11], [6, 11], [65535, 11]])
    payload = struct.pack("<HHHHB10H", 3, 11, 6, 11, 4, *np.asarray(points).reshape(-1))
    expected = struct.pack("<BBBBIIHBB", 1, 3, 1, 0, 7, 13, 29, 13, 0) + payload
    # When: serialize the supplied plan and decode under jit.
    packet = jax.jit(encode_plan)(points[0], points[3], points, 4, 1, 7, 13, 4)
    decoded = jax.jit(decode_plan)(packet)
    # Then: PLAN45 charges and transmits every horizon slot, with a separate validity count.
    assert int(packet.length) == 16 + 4 + 4 + 1 + 5 * 4 == 45
    assert bytes(np.asarray(packet.buffer[:45])) == expected
    np.testing.assert_array_equal(decoded.positions, points)
    np.testing.assert_array_equal(decoded.source, [3, 11])
    np.testing.assert_array_equal(decoded.goal, [6, 11])
    np.testing.assert_array_equal(decoded.valid_mask, [True, True, True, True, False])


@pytest.mark.parametrize("count", [0, 2, 47])
def test_map_delta_wire_when_count_controls_actual_length(count: int) -> None:
    # Given: a fixed capacity input with independent little-endian entry bytes.
    coordinates = jnp.stack((jnp.arange(47), 65535 - jnp.arange(47)), axis=1)
    states = jnp.arange(47) % 2
    payload = bytes([count]) + b"".join(
        struct.pack("<HHB", i, 65535 - i, i % 2) for i in range(count)
    )
    expected = struct.pack("<BBBBIIHBB", 1, 4, 2, 0, 9, 17, 1 + 5 * count, 11, 0) + payload
    # When: encode and decode without dynamic array sizes.
    packet = jax.jit(encode_map_delta)(coordinates, states, count, 2, 9, 17, 4)
    decoded = jax.jit(decode_map_delta)(packet)
    # Then: padding is uncharged and all real entries round trip.
    assert int(packet.length) == 16 + 1 + 5 * count
    assert bytes(np.asarray(packet.buffer[: int(packet.length)])) == expected
    assert bool(validate_packet(packet, 4))
    np.testing.assert_array_equal(decoded.coordinates[:count], coordinates[:count])
    np.testing.assert_array_equal(decoded.states[:count], states[:count])
    np.testing.assert_array_equal(decoded.valid_mask, np.arange(47) < count)


@pytest.mark.parametrize(("offset", "value"), [(0, 2), (1, 31), (3, 1), (12, 5), (14, 15), (15, 1)])
def test_host_and_device_reject_when_structure_is_invalid(offset: int, value: int) -> None:
    # Given: independently packed POSITION with one malformed field.
    raw = bytearray(struct.pack("<BBBBIIHBBHH", 1, 1, 0, 0, 0, 0, 4, 14, 0, 3, 11))
    raw[offset] = value
    valid = encode_position(jnp.asarray([3, 11]), 0, 0, 0, 4)
    malformed = valid._replace(buffer=valid.buffer.at[offset].set(value))
    # When / Then: the host rejects; the device returns false rather than throwing.
    with pytest.raises(PacketError):
        packet_from_bytes(bytes(raw), 4)
    assert not bool(jax.jit(validate_packet)(malformed, 4))


def test_custom_when_at_capacity_and_beyond() -> None:
    # Given: a participant-defined codec and a full 240-byte opaque payload.
    payload = bytes(range(240))
    header = (1, 201, 0, 0, 2, 3, 240, 2, 0)
    # When: use the host encoder and try one extra byte independently.
    packet = encode_packet(header, payload, 2)
    with pytest.raises(PacketError):
        packet_from_bytes(struct.pack("<BBBBIIHBB", *header) + payload + b"x", 2)
    # Then: all 256 bytes are meaningful, with opaque custom content unchanged.
    assert int(packet.length) == 256
    assert decode_packet_host(packet, 2) == (header, payload)


@pytest.mark.parametrize(
    ("codec", "payload"),
    [
        (3, struct.pack("<HHHHB10H", 3, 11, 6, 11, 6, *([0] * 10))),
        (4, struct.pack("<BHHB", 1, 3, 11, 2)),
        (4, bytes([48]) + bytes(235)),
        (5, struct.pack("<BBI", 5, 0, 7)),
    ],
)
def test_host_when_standard_content_is_malformed(codec: int, payload: bytes) -> None:
    # Given: too many plan positions, manufactured UNKNOWN, excess entries, or nested forwarding.
    raw = struct.pack("<BBBBIIHBB", 1, codec, 0, 0, 0, 0, len(payload), 14, 0) + payload
    # When / Then: structurally malformed content never reaches environment merging.
    with pytest.raises(PacketError):
        packet_from_bytes(raw, 4)


def test_map_delta_when_coordinates_are_structurally_valid_but_outside_a_map() -> None:
    # Given: the protocol permits all uint16 coordinates without access to a hidden map shape.
    coordinates = jnp.asarray([[65535, 3]])
    # When: encode a known-state observation with a large structural coordinate.
    packet = encode_map_delta(coordinates, jnp.asarray([1]), 1, 0, 1, 2, 4)
    # Then: the wire validator accepts it; the environment owns bounds-aware map merging.
    assert bool(validate_packet(packet, 4))
    np.testing.assert_array_equal(decode_map_delta(packet).coordinates[0], [65535, 3])


def test_forward_when_rebroadcast_preserves_charged_origin() -> None:
    # Given: author zero's plan from event 7.
    points = jnp.asarray([[3, 11]] * 5)
    original = encode_plan(points[0], points[0], points, 5, 0, 1, 7, 4)
    # When: two explicit forwarding actions use their own header identity/time.
    forwarded = jax.jit(forward_packet)(original, 1, 3, 20, 4)
    repeated = jax.jit(forward_packet)(forwarded, 2, 4, 21, 4)
    origin = jax.jit(decode_payload_origin)(repeated)
    # Then: one six-byte envelope retains author/time; every relay pays 51 bytes.
    assert int(forwarded.length) == int(repeated.length) == 51
    assert bool(validate_packet(repeated, 4))
    assert (int(origin.codec), int(origin.author), int(origin.generated_event)) == (3, 0, 7)
    assert bytes(np.asarray(repeated.buffer[16:22])) == struct.pack("<BBI", 3, 0, 7)
    assert (int(decode_header(repeated).sender), int(decode_header(repeated).generated_event)) == (
        2,
        21,
    )
    np.testing.assert_array_equal(decode_plan(repeated).positions, points)


def test_forward_when_original_payload_leaves_no_envelope_capacity() -> None:
    # Given: a legal maximum-sized custom packet.
    original = make_packet(Codec.CUSTOM, jnp.arange(240, dtype=jnp.uint8), 240, 0, 1, 7, 4)
    # When: request forwarding with its required six bytes.
    forwarded = jax.jit(forward_packet)(original, 1, 2, 8, 4)
    # Then: invalid capacity is explicit and the transmission validator refuses it.
    assert int(forwarded.length) == 262
    assert not bool(validate_packet(forwarded, 4))


def test_map_selection_when_cursor_rotates_without_acknowledgement() -> None:
    # Given: all known cells have equal timestamps; ties sort x first, then y.
    local = jnp.zeros((8, 8), dtype=jnp.int32)
    times = jnp.full((8, 8), 7, dtype=jnp.int32)
    # When: two consecutive unacknowledged windows rotate through the ordering.
    first = jax.jit(select_map_delta)(local, times, 0)
    second = jax.jit(select_map_delta)(local, times, first.next_cursor)
    # Then: the next window starts at known rank 47 regardless of any recipient.
    np.testing.assert_array_equal(first.coordinates[:3], [[0, 0], [0, 1], [0, 2]])
    np.testing.assert_array_equal(second.coordinates[:3], [[5, 7], [6, 0], [6, 1]])


def test_map_selection_when_unknown_entries_have_newer_timestamps() -> None:
    # Given: unknown cells cannot enter the message even with a high timestamp.
    local = jnp.asarray([[2, 0], [1, 2]])
    times = jnp.asarray([[99, 2], [3, 99]])
    # When: choose a bounded map delta.
    selected = jax.jit(select_map_delta)(local, times, 0)
    # Then: known observations sort newest first, with unfilled entries masked.
    assert int(selected.count) == 2
    np.testing.assert_array_equal(selected.coordinates[:2], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(selected.states[:2], [1, 0])


def test_packet_when_vectorized_matches_individual_encoding() -> None:
    # Given: different asymmetric positions, senders, and sequence counters.
    positions = jnp.asarray([[3, 11], [65535, 1], [8, 2]])
    senders = jnp.arange(3)
    # When: batch encoding shares one compiled kernel.
    packets = jax.jit(jax.vmap(encode_position, in_axes=(0, 0, 0, None, None)))(
        positions, senders, senders + 10, 20, 4
    )
    # Then: each wire image matches an independent struct oracle.
    for i in range(3):
        expected = struct.pack(
            "<BBBBIIHBBHH", 1, 1, i, 0, i + 10, 20, 4, 15 ^ (1 << i), 0, *np.asarray(positions[i])
        )
        assert bytes(np.asarray(packets.buffer[i, :20])) == expected
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(validate_packet, in_axes=(0, None)))(packets, 4), [True, True, True]
    )
