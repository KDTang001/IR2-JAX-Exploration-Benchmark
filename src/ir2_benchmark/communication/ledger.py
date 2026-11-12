"""A broadcast is charged once when attempted, regardless of delivery."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class Attempt(NamedTuple):
    """Budget after an authorized attempt, and its actual byte charge."""

    eligible: Array
    balance: Array
    spent: Array


def charge_attempt(balance: Array, cost: Array, requested: Array, packet_valid: Array) -> Attempt:
    """Authorize only a valid requested packet that fits the remaining budget."""
    eligible = requested & packet_valid & (cost <= balance)
    return Attempt(
        eligible, jnp.where(eligible, balance - cost, balance), jnp.where(eligible, cost, 0)
    )
