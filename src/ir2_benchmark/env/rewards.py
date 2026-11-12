"""Event rewards from explicit transition statistics, without state mutation."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from ir2_benchmark.config import HostConfig


class RewardContext(NamedTuple):
    """Measured event statistics; no mutable environment or hidden actor input."""

    coverage_before: Array
    coverage_after: Array
    distance: Array
    spent: Array
    success: Array
    observed_frontiers: Array
    exploration_utility: Array
    rendezvous_utility: Array


class RewardResult(NamedTuple):
    """Six terms: coverage, distance, event, success, frontier, node utility."""

    reward: Array
    terms: Array


def _result(
    coverage: Array | float = 0.0,
    distance: Array | float = 0.0,
    event: Array | float = 0.0,
    success: Array | float = 0.0,
    frontier: Array | float = 0.0,
    utility: Array | float = 0.0,
) -> RewardResult:
    terms = jnp.asarray([coverage, distance, event, success, frontier, utility], jnp.float32)
    return RewardResult(jnp.sum(terms), terms)


def coverage_delta(context: RewardContext, config: HostConfig) -> RewardResult:
    """Original benchmark coverage reward with configurable gain and event cost."""
    cost = (
        1 / (config.n_agents * 512)
        if config.reward_event_cost is None
        else config.reward_event_cost
    )
    gain = config.reward_coverage_weight * (context.coverage_after - context.coverage_before)
    result = _result(coverage=gain, event=-cost)
    # Keep the original arithmetic order, including float32 rounding.
    return RewardResult(gain - cost, result.terms)


def coverage_distance(context: RewardContext, config: HostConfig) -> RewardResult:
    """Trade newly covered area against actual geometric travel."""
    return _result(
        coverage=config.reward_coverage_weight * (context.coverage_after - context.coverage_before),
        distance=-config.reward_distance_weight * context.distance,
        event=-(config.reward_event_cost or 0.0),
    )


def sparse_success(context: RewardContext, config: HostConfig) -> RewardResult:
    """Reward only successful completion, with optional event cost."""
    return _result(
        success=config.reward_success_bonus * context.success,
        event=-(config.reward_event_cost or 0.0),
    )


def ir2_individual(context: RewardContext, config: HostConfig) -> RewardResult:
    """Original single_robot_step formula; no round-wide team completion bonus."""
    frontier = context.observed_frontiers / 25
    exploration = context.exploration_utility / 50
    meeting = context.rendezvous_utility
    distance = context.distance / 512
    result = _result(frontier=frontier, utility=exploration + meeting, distance=-distance)
    return RewardResult(frontier + exploration + meeting - distance, result.terms)
