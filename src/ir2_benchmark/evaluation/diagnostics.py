"""Decision-cutoff intention error telemetry; labels never enter the actor's estimate."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from ir2_benchmark.config import FAULT_BURST

if TYPE_CHECKING:
    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.models.plan_predictor import PlanEstimate
    from ir2_benchmark.types import EnvState


class PlanDiagnostics(NamedTuple):
    """Compact per-event/per-peer errors against latest already-generated intentions."""

    event: Array
    receiver: Array
    target_event: Array
    has_target: Array
    has_prediction: Array
    path_points: Array
    message_age: Array
    phase: Array
    cold_start: Array
    goal_error: Array
    ade: Array
    fde: Array
    nll: Array
    coverage_one_sigma: Array
    has_uncertainty: Array


def measure_plan_estimate(  # noqa: PLR0913
    estimate: PlanEstimate,
    state: EnvState,
    receiver: Array,
    config: HostConfig,
    *,
    event_valid: Array,
    has_uncertainty: bool,
) -> PlanDiagnostics:
    """Compare the exact actor-used estimate with event-before intention labels only.

    ADE/FDE/goal error are map-cell distances. NLL is in normalized coordinates;
    ZOH has no calibrated density, and its NLL remains missing. Zero-length routes
    retain goal diagnostics without inventing valid path points.
    """
    peers = jnp.arange(config.n_slots)
    has_target = (
        event_valid & state.active_mask & (peers != receiver) & (state.plan_generated_event >= 0)
    )
    has_prediction = jnp.asarray(has_uncertainty) | ~estimate.missing
    scale = jnp.maximum(state.valid_shape[::-1] - 1, 1)
    expected = state.plans / scale
    points = jnp.arange(config.plan_horizon)[None] < state.plan_lengths[:, None]
    count = jnp.sum(points, axis=-1)
    differences = estimate.positions - expected
    distances = jnp.linalg.norm(differences * scale, axis=-1)
    measurable = has_target & has_prediction
    has_path = measurable & (count > 0)
    ade = jnp.where(
        has_path, jnp.sum(jnp.where(points, distances, 0), axis=-1) / jnp.maximum(count, 1), jnp.nan
    )
    fde = jnp.where(
        has_path,
        jnp.take_along_axis(distances, jnp.maximum(count - 1, 0)[:, None], axis=-1)[:, 0],
        jnp.nan,
    )
    goal_error = jnp.where(
        measurable, jnp.linalg.norm(estimate.goal * scale - state.goals, axis=-1), jnp.nan
    )
    variance = jnp.maximum(estimate.variance[:, 1:], 1e-12)
    nll_points = jnp.sum(
        0.5 * (differences**2 / variance + jnp.log(2 * jnp.pi * variance)), axis=-1
    )
    nll = jnp.where(
        has_path & has_uncertainty,
        jnp.sum(jnp.where(points, nll_points, 0), axis=-1) / jnp.maximum(count, 1),
        jnp.nan,
    )
    inside = jnp.abs(differences) <= jnp.sqrt(variance)
    calibration = jnp.where(
        has_path & has_uncertainty,
        jnp.sum(inside & points[..., None], axis=(-1, -2)) / jnp.maximum(2 * count, 1),
        jnp.nan,
    )
    burst_end = config.burst_start + config.burst_duration
    phase = jnp.where(
        config.fault_mode == 1,
        1,
        jnp.where(
            config.fault_mode == FAULT_BURST,
            jnp.where(
                state.round < config.burst_start, 0, jnp.where(state.round < burst_end, 1, 2)
            ),
            0,
        ),
    )
    ages = jnp.where(
        estimate.last_message_event >= 0, state.event - estimate.last_message_event, -1
    )
    return PlanDiagnostics(
        state.event,
        receiver,
        state.plan_generated_event,
        has_target,
        has_prediction,
        count,
        ages,
        phase,
        estimate.missing,
        goal_error,
        ade,
        fde,
        nll,
        calibration,
        jnp.full(config.n_slots, has_uncertainty, dtype=bool),
    )
