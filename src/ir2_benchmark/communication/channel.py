"""One-hop broadcast with a single paid attempt and immutable per-source inboxes."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from ir2_benchmark.codecs.packet import Packet, decode_header, read_uint_le, validate_packet
from ir2_benchmark.communication.ledger import charge_attempt


class Inbox(NamedTuple):
    """Axes are [receiver,direct sender]; consuming clears only ``present``."""

    buffers: Array
    lengths: Array
    present: Array
    received_event: Array
    overwritten: Array


class BroadcastResult(NamedTuple):
    """Physical deliveries may differ per receiver; spent is scalar per attempt."""

    inbox: Inbox
    balance: Array
    spent: Array
    eligible: Array
    delivered: Array
    overwritten: Array


def empty_inbox(n_agents: int) -> Inbox:
    """Allocate static packet slots, including permanently invalid self slots."""
    shape = (n_agents, n_agents)
    return Inbox(
        jnp.zeros((*shape, 256), jnp.uint8),
        jnp.zeros(shape, jnp.int32),
        jnp.zeros(shape, jnp.bool_),
        jnp.full(shape, -1, jnp.int32),
        jnp.zeros(shape, jnp.int32),
    )


def consume_inbox(inbox: Inbox, receiver: Array) -> Inbox:
    """Mark a robot's observed messages consumed only during its state transition."""
    return inbox._replace(present=inbox.present.at[receiver].set(False))


# Distinct immutable ledger, packet and receiver masks are the shared environment boundary.
def broadcast(  # noqa: PLR0913, PLR0917
    inbox: Inbox,
    packet: Packet,
    sender: Array,
    balance: Array,
    requested: Array,
    base_reachable: Array,
    external_allowed: Array,
    active_mask: Array,
    plan_horizon: int = 5,
    charge_mode: str = "attempt",
    addressing: str = "broadcast",
) -> BroadcastResult:
    """Charge once, deliver directly, and replace only fresher unread source slots."""
    n_agents = jnp.sum(active_mask, dtype=jnp.int32)
    header = decode_header(packet)
    if charge_mode not in ("attempt", "success"):
        raise ValueError("unknown charge mode")
    packet_valid = validate_packet(packet, n_agents, plan_horizon, addressing) & (
        header.sender == sender
    )
    attempt = charge_attempt(balance, packet.length, requested, packet_valid & active_mask[sender])
    receivers = jnp.arange(active_mask.shape[0])
    delivered = (
        attempt.eligible
        & base_reachable
        & external_allowed
        & active_mask
        & (receivers != sender)
        & ((header.recipient_mask.astype(jnp.uint32) & (jnp.uint32(1) << receivers)) != 0)
    )
    spent = (
        attempt.spent
        if charge_mode == "attempt"
        else jnp.where(jnp.any(delivered), attempt.spent, 0)
    )
    old = inbox.buffers[:, sender]
    old_event = read_uint_le(old, 8, 4)
    old_sequence = read_uint_le(old, 4, 4)
    newer = (header.generated_event > old_event) | (
        (header.generated_event == old_event) & (header.sequence > old_sequence)
    )
    accepted = delivered & (~inbox.present[:, sender] | newer)
    overwritten = accepted & inbox.present[:, sender]
    updated = Inbox(
        inbox.buffers.at[:, sender].set(jnp.where(accepted[:, None], packet.buffer, old)),
        inbox.lengths.at[:, sender].set(
            jnp.where(accepted, packet.length, inbox.lengths[:, sender])
        ),
        inbox.present.at[:, sender].set(inbox.present[:, sender] | accepted),
        inbox.received_event.at[:, sender].set(
            jnp.where(
                accepted, header.generated_event.astype(jnp.int32), inbox.received_event[:, sender]
            )
        ),
        inbox.overwritten.at[:, sender].add(overwritten.astype(jnp.int32)),
    )
    return BroadcastResult(
        updated, balance - spent, spent, attempt.eligible, delivered, overwritten
    )
