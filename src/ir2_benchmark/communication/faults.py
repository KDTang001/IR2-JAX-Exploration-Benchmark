"""Purpose-separated counter-keyed faults independent of transmission history."""

from enum import IntEnum

import jax
import jax.numpy as jnp
from jax import Array


class KeyPurpose(IntEnum):
    """Namespaces must never be shared by spawn, policy and channel sampling."""

    SPAWN = 0
    POLICY = 1
    FAULT = 2
    BURST_SELECTION = 3
    PLAN_REPLAY = 4


class FaultMode(IntEnum):
    """Burst and Dropout are distinct, mutually exclusive conditions."""

    NORMAL = 0
    DROPOUT = 1
    BURST = 2


# Each semantic counter is a distinct random-stream coordinate.
def fold_key(  # noqa: PLR0913, PLR0917
    key: Array,
    purpose: Array,
    episode: Array = 0,
    event: Array = 0,
    sender: Array = 0,
    receiver: Array = 0,
) -> Array:
    """Derive a stream from semantic counters, never from prior packet attempts."""
    for counter in (purpose, episode, event, sender, receiver):
        key = jax.random.fold_in(key, jnp.asarray(counter, jnp.uint32))
    return key


def sample_isolated(key: Array, scenario: Array, active_mask: Array) -> Array:
    """Choose floor(N/2) active robots once for the burst scenario."""
    scenario_key = fold_key(key, KeyPurpose.BURST_SELECTION, scenario)
    scores = jax.random.uniform(scenario_key, active_mask.shape)
    order = jnp.argsort(jnp.where(active_mask, scores, jnp.inf))
    ranks = jnp.zeros(active_mask.shape, jnp.int32).at[order].set(jnp.arange(active_mask.shape[0]))
    return active_mask & (ranks < jnp.sum(active_mask) // 2)


# Scenario counters and external faults remain explicit inputs to the pure channel.
def fault_links(  # noqa: PLR0913, PLR0917
    key: Array,
    mode: Array,
    episode: Array,
    event: Array,
    sender: Array,
    n_agents: int,
    dropout_probability: Array,
    isolated_mask: Array,
    burst_active: Array,
) -> Array:
    """Return external allowance per receiver; physical links are applied separately."""
    keys = jax.vmap(
        lambda receiver: fold_key(key, KeyPurpose.FAULT, episode, event, sender, receiver)
    )(jnp.arange(n_agents))
    draws = jax.vmap(jax.random.uniform)(keys)
    dropout_allowed = draws >= dropout_probability
    burst_allowed = ~(burst_active & (isolated_mask[sender] | isolated_mask))
    return jnp.where(
        mode == FaultMode.DROPOUT,
        dropout_allowed,
        jnp.where(mode == FaultMode.BURST, burst_allowed, jnp.ones(n_agents, jnp.bool_)),
    )
