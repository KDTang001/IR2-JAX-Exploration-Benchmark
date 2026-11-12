"""Physical-link presets; external packet faults are applied separately."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ir2_benchmark.communication.signal import link_signal, segment_lengths

if TYPE_CHECKING:
    from ir2_benchmark.config import HostConfig


class LinkContext(NamedTuple):
    """Privileged physical geometry and explicitly separated random streams."""

    grid: Array
    start: Array
    end: Array
    key: Array
    episode_key: Array


class LinkResult(NamedTuple):
    """Reachability with optional physical signal-strength diagnostics."""

    reachable: Array
    strength: Array
    strength_valid: Array


def attenuation(context: LinkContext, config: HostConfig) -> LinkResult:
    """Preserve the existing analytic path-intersection model."""
    signal = link_signal(
        context.grid,
        context.start,
        context.end,
        config.transmit_power,
        config.reference_loss,
        config.gamma_free,
        config.gamma_obstacle,
        config.reference_distance,
        config.signal_threshold,
    )
    return LinkResult(signal.reachable, signal.strength, jnp.bool_(True))


def ideal(context: LinkContext, config: HostConfig) -> LinkResult:
    """Remove physical attenuation only; budget and faults still apply."""
    return LinkResult(jnp.bool_(True), jnp.float32(0), jnp.bool_(False))


def range(context: LinkContext, config: HostConfig) -> LinkResult:  # noqa: A001
    """Include every endpoint no farther than the configured radius."""
    distance = jnp.linalg.norm((context.end - context.start).astype(jnp.float32))
    return LinkResult(distance <= config.communication_range, jnp.float32(0), jnp.bool_(False))


def los_range(context: LinkContext, config: HostConfig) -> LinkResult:
    """Require both in-range endpoints and zero positive wall penetration."""
    lengths = segment_lengths(context.grid, context.start, context.end)
    # Positive wall-intersection length blocks LOS; a corner touch alone does not.
    reachable = (lengths.distance <= config.communication_range) & (lengths.obstacle_distance == 0)
    return LinkResult(reachable, jnp.float32(0), jnp.bool_(False))


def ir2(context: LinkContext, config: HostConfig) -> LinkResult:
    """skimage.draw.line cells, excluding source, including destination.

    X_g and K are drawn once per episode (the original Env constructs one
    SS_realistic_model). Every pair/direction/event reuses those two values.
    """
    delta = context.end.astype(jnp.int32) - context.start.astype(jnp.int32)
    dx, dy = jnp.abs(delta)
    major, minor = jnp.maximum(dx, dy), jnp.minimum(dx, dy)
    t = jnp.arange(max(context.grid.shape), dtype=jnp.int32)
    short = (2 * minor * t + major) // (2 * jnp.maximum(major, 1))
    offsets = jnp.stack((jnp.where(dx >= dy, t, short), jnp.where(dx >= dy, short, t)), -1)
    cells = context.start + offsets * jnp.where(delta >= 0, 1, -1)
    x = cells[:, 0].clip(0, context.grid.shape[1] - 1)
    y = cells[:, 1].clip(0, context.grid.shape[0] - 1)
    obstacle = jnp.sum((t > 0) & (t <= major) & (context.grid[y, x] != 0))
    distance = jnp.linalg.norm(delta.astype(jnp.float32))
    free = distance - obstacle
    noise = jax.random.uniform(context.episode_key, (2,))
    x_g = config.signal_noise_free_min + noise[0] * (
        config.signal_noise_free_max - config.signal_noise_free_min
    )
    k = config.signal_noise_obstacle_min + noise[1] * (
        config.signal_noise_obstacle_max - config.signal_noise_obstacle_min
    )
    loss = config.reference_loss
    loss += jnp.where(
        obstacle > 0, 10 * config.gamma_obstacle * jnp.log10(jnp.maximum(obstacle, 1)) + k, 0
    )
    loss += jnp.where(
        free >= config.reference_distance,
        10
        * config.gamma_free
        * jnp.log10(jnp.maximum(free, config.reference_distance) / config.reference_distance)
        + x_g,
        0,
    )
    strength = config.transmit_power - loss
    return LinkResult(strength > config.signal_threshold, strength, jnp.bool_(True))
