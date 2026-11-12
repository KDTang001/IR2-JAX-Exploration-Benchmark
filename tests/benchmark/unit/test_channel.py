"""One charged broadcast, receiver-specific faults and immutable source inboxes."""

# Independent oracle values and positional device calls deliberately mirror public JAX signatures.
# ruff: noqa: PLR2004, FBT003

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from ir2_benchmark.codecs.packet import Packet, decode_payload_origin, forward_packet
from ir2_benchmark.codecs.standard import encode_plan
from ir2_benchmark.communication.channel import (
    BroadcastResult,
    broadcast,
    consume_inbox,
    empty_inbox,
)
from ir2_benchmark.communication.faults import (
    FaultMode,
    KeyPurpose,
    fault_links,
    fold_key,
    sample_isolated,
)


def _plan(sender: int = 0, sequence: int = 1, event: int = 7) -> Packet:
    points = jnp.asarray([[3, 11], [4, 11], [5, 11], [6, 11], [7, 11]])
    return encode_plan(points[0], points[-1], points, 5, sender, sequence, event, 4)


@pytest.mark.parametrize(
    ("budget", "reachable", "expected_spent", "expected_balance", "expected_delivery"),
    [
        (50, [False, False, False, False], 45, 5, [False, False, False, False]),
        (50, [True, True, True, True], 45, 5, [False, True, True, True]),
        (44, [True, True, True, True], 0, 44, [False, False, False, False]),
    ],
)
def test_broadcast_when_no_receivers_many_receivers_or_insufficient_budget(
    budget: int,
    reachable: list[bool],
    expected_spent: int,
    expected_balance: int,
    expected_delivery: list[bool],
) -> None:
    # Given: one PLAN45 and an independent per-receiver physical reachability vector.
    inbox = empty_inbox(4)
    packet = _plan()
    # When: one compiled broadcast considers all receivers.
    result = jax.jit(broadcast)(
        inbox, packet, 0, budget, True, jnp.asarray(reachable), jnp.ones(4, bool), jnp.ones(4, bool)
    )
    # Then: charging happens once even for loss or three successful deliveries.
    assert int(result.spent) == expected_spent
    assert int(result.balance) == expected_balance
    np.testing.assert_array_equal(result.delivered, expected_delivery)
    np.testing.assert_array_equal(result.inbox.present[:, 0], expected_delivery)
    np.testing.assert_array_equal(inbox.present, np.zeros((4, 4), bool))


def test_broadcast_when_receiver_faults_differ_and_forwarding_is_explicit() -> None:
    # Given: receiver 2 drops source 0's packet; all physical links permit it.
    inbox = empty_inbox(4)
    allowed = jnp.asarray([True, True, False, True])
    # When: source 0 makes exactly one transmission.
    result = broadcast(inbox, _plan(), 0, 100, True, jnp.ones(4, bool), allowed, jnp.ones(4, bool))
    # Then: receiver 1 has no automatic outgoing packet, and failed receiver remains empty.
    np.testing.assert_array_equal(result.inbox.present[:, 0], [False, True, False, True])
    np.testing.assert_array_equal(result.inbox.present[:, 1:], np.zeros((4, 3), bool))
    assert int(result.spent) == 45


def test_broadcast_when_unread_newer_packet_replaces_same_direct_source() -> None:
    # Given: an unread packet from event 7 already occupies receivers' source slots.
    flags = jnp.ones(4, bool)
    first = broadcast(empty_inbox(4), _plan(), 0, 100, True, flags, flags, flags)
    # When: the same sender explicitly transmits a newer packet.
    second = jax.jit(broadcast)(
        first.inbox, _plan(sequence=2, event=8), 0, first.balance, True, flags, flags, flags
    )
    # Then: only unread occupied source slots count overwrites; source contents update.
    np.testing.assert_array_equal(second.inbox.overwritten[:, 0], [0, 1, 1, 1])
    np.testing.assert_array_equal(second.inbox.received_event[:, 0], [-1, 8, 8, 8])
    np.testing.assert_array_equal(first.inbox.overwritten, np.zeros((4, 4), int))


def test_broadcast_when_stale_delivery_cannot_replace_newer_unread_packet() -> None:
    # Given: receivers already hold event 8.
    flags = jnp.ones(4, bool)
    first = broadcast(empty_inbox(4), _plan(sequence=2, event=8), 0, 100, True, flags, flags, flags)
    # When: a stale event 7 is paid for and physically delivered.
    second = broadcast(
        first.inbox, _plan(sequence=3, event=7), 0, first.balance, True, flags, flags, flags
    )
    # Then: delivery does not make stale content fresher or erase the paid charge.
    assert int(second.spent) == 45
    np.testing.assert_array_equal(second.inbox.buffers, first.inbox.buffers)
    np.testing.assert_array_equal(second.overwritten, [False] * 4)


def test_consume_when_receiver_takes_its_event() -> None:
    # Given: several receivers have the same broadcast unread.
    flags = jnp.ones(4, bool)
    received = broadcast(empty_inbox(4), _plan(), 0, 100, True, flags, flags, flags).inbox
    # When: receiver 1 consumes its own inbox during its state transition.
    consumed = jax.jit(consume_inbox)(received, 1)
    # Then: other robots' unread observations and the original state remain intact.
    np.testing.assert_array_equal(consumed.present[:, 0], [False, False, True, True])
    np.testing.assert_array_equal(received.present[:, 0], [False, True, True, True])
    np.testing.assert_array_equal(consumed.buffers, received.buffers)


def test_forward_when_relay_pays_for_origin_and_sends_only_on_its_action() -> None:
    # Given: robot 1 received robot 0's plan at the earlier event.
    flags = jnp.ones(4, bool)
    first = broadcast(empty_inbox(4), _plan(), 0, 100, True, flags, flags, flags)
    forwarded = forward_packet(
        Packet(first.inbox.buffers[1, 0], first.inbox.lengths[1, 0]), 1, 9, 12, 4
    )
    # When: robot 1 takes a new paid transmission action.
    result = broadcast(first.inbox, forwarded, 1, 60, True, flags, flags, flags)
    origin = decode_payload_origin(Packet(result.inbox.buffers[2, 1], result.inbox.lengths[2, 1]))
    # Then: robot 1 pays the complete forward51 while original author/event survive.
    assert int(result.spent) == 16 + 6 + 29 == 51
    assert int(result.balance) == 9
    assert (int(origin.author), int(origin.generated_event)) == (0, 7)
    np.testing.assert_array_equal(result.inbox.present[:, 1], [True, False, True, True])


@pytest.mark.parametrize(
    ("mode", "probability", "isolated", "active", "expected"),
    [
        (FaultMode.NORMAL, 1.0, [True] * 4, True, [True] * 4),
        (FaultMode.DROPOUT, 0.0, [False] * 4, False, [True] * 4),
        (FaultMode.DROPOUT, 1.0, [False] * 4, False, [False] * 4),
        (FaultMode.BURST, 0.0, [True, False, True, False], True, [False] * 4),
        (FaultMode.BURST, 0.0, [True, False, True, False], False, [True] * 4),
    ],
)
def test_faults_when_normal_dropout_endpoints_or_burst(
    *,
    mode: FaultMode,
    probability: float,
    isolated: list[bool],
    active: bool,
    expected: list[bool],
) -> None:
    # Given: source 0 and a declared external degradation condition.
    key = jax.random.PRNGKey(11)
    # When: the channel samples its independent per-receiver external allowances.
    result = jax.jit(fault_links, static_argnames=("n_agents",))(
        key, mode, 2, 17, 0, 4, probability, jnp.asarray(isolated), active
    )
    # Then: Normal adds no loss; only Burst isolates both directions.
    np.testing.assert_array_equal(result, expected)


def test_burst_when_nonisolated_sender_cannot_reach_isolated_receivers() -> None:
    # Given: sender 1 is outside the isolated node set.
    isolated = jnp.asarray([True, False, True, False])
    # When: its burst-period transmission is checked.
    allowed = fault_links(jax.random.PRNGKey(4), FaultMode.BURST, 0, 0, 1, 4, 0.0, isolated, True)
    # Then: only the other nonisolated nodes pass the external link condition.
    np.testing.assert_array_equal(allowed, [False, True, False, True])


def test_random_streams_when_purpose_or_event_changes() -> None:
    # Given: semantic counters and one root key, independent of mutable call count.
    key = jax.random.PRNGKey(13)
    purposes = (KeyPurpose.SPAWN, KeyPurpose.POLICY, KeyPurpose.FAULT)
    expected_key = key
    for value in (KeyPurpose.FAULT, 2, 17, 0, 3):
        expected_key = jax.random.fold_in(expected_key, value)
    # When: build purpose namespaces and repeat an event around unrelated sampling.
    keys = [fold_key(key, purpose, 2, 17, 0, 3) for purpose in purposes]
    before = fault_links(key, FaultMode.DROPOUT, 2, 17, 0, 4, 0.5, jnp.zeros(4, bool), False)
    fault_links(key, FaultMode.DROPOUT, 2, 16, 2, 4, 0.5, jnp.zeros(4, bool), False)
    after = fault_links(key, FaultMode.DROPOUT, 2, 17, 0, 4, 0.5, jnp.zeros(4, bool), False)
    # Then: the exact fold sequence matches the independent oracle and call history is irrelevant.
    np.testing.assert_array_equal(keys[-1], expected_key)
    assert len({tuple(map(int, sample)) for sample in keys}) == 3
    np.testing.assert_array_equal(before, after)
    assert not np.array_equal(fold_key(key, KeyPurpose.FAULT, 2, 18, 0, 3), expected_key)


def test_faults_when_vmap_keeps_each_receiver_key_independent() -> None:
    # Given: events use a fixed sender, per-receiver fold-in and a nontrivial drop rate.
    key = jax.random.PRNGKey(7)
    events = jnp.asarray([4, 7, 12])

    # When: events are sampled in a compiled batch.
    def sample(event: Array) -> Array:
        return fault_links(key, FaultMode.DROPOUT, 2, event, 0, 4, 0.5, jnp.zeros(4, bool), False)

    result = jax.jit(jax.vmap(sample))(events)
    # Then: explicit independent per-receiver key arithmetic defines the outcomes.
    expected = []
    for event in [4, 7, 12]:
        row = []
        for receiver in range(4):
            receiver_key = key
            for value in (2, 2, event, 0, receiver):
                receiver_key = jax.random.fold_in(receiver_key, value)
            row.append(float(jax.random.uniform(receiver_key)) >= 0.5)
        expected.append(row)
    np.testing.assert_array_equal(result, expected)


def test_burst_selection_when_bucket_contains_inactive_robots() -> None:
    # Given: four logical robots in a six-robot static bucket.
    active = jnp.asarray([True, True, True, True, False, False])
    key = jax.random.PRNGKey(17)
    # When: scenario selection is compiled and replayed.
    selected = jax.jit(sample_isolated)(key, 31, active)
    replayed = sample_isolated(key, 31, active)
    # Then: exactly floor(4/2) logical robots are isolated and padding is never chosen.
    assert int(jnp.sum(selected)) == 2
    np.testing.assert_array_equal(selected[4:], [False, False])
    np.testing.assert_array_equal(selected, replayed)


def test_broadcast_when_vmap_compares_different_receiver_conditions() -> None:
    # Given: separate environments have differing external fault masks.
    masks = jnp.asarray([[True, True, False, True], [False] * 4])
    packet = _plan()

    # When: a compiled vectorized broadcast advances both static inboxes.
    def action(mask: Array) -> BroadcastResult:
        return broadcast(
            empty_inbox(4), packet, 0, 50, True, jnp.ones(4, bool), mask, jnp.ones(4, bool)
        )

    results = jax.jit(jax.vmap(action))(masks)
    # Then: independent receiver outcomes never alter the single per-environment fee.
    np.testing.assert_array_equal(results.spent, [45, 45])
    np.testing.assert_array_equal(results.balance, [5, 5])
    np.testing.assert_array_equal(results.delivered, [[False, True, False, True], [False] * 4])


def test_broadcast_when_static_bucket_has_inactive_robot_slots() -> None:
    # Given: four active logical robots in a six-robot packet/inbox bucket.
    active = jnp.asarray([True, True, True, True, False, False])
    # When: every physical link and external condition is permissive.
    result = jax.jit(broadcast)(
        empty_inbox(6), _plan(), 0, 50, True, jnp.ones(6, bool), jnp.ones(6, bool), active
    )
    # Then: only the three other logical robots receive, with one cost for the attempt.
    np.testing.assert_array_equal(result.delivered, [False, True, True, True, False, False])
    assert int(result.spent) == 45
    np.testing.assert_array_equal(result.inbox.present[4:], np.zeros((2, 6), bool))


def test_broadcast_when_wire_sender_does_not_match_acting_robot() -> None:
    # Given: robot 1 attempts to use robot 0's unmodified packet header.
    flags = jnp.ones(4, bool)
    # When: the packet passes through the actual send boundary.
    result = broadcast(empty_inbox(4), _plan(), 1, 50, True, flags, flags, flags)
    # Then: identity validation refuses the attempt without charging or delivering.
    assert not bool(result.eligible)
    assert int(result.balance) == 50
    np.testing.assert_array_equal(result.delivered, [False] * 4)
