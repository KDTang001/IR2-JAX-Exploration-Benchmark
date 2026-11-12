"""Local-only features shared by both learning routes.

All neural convolutions/linear layers request HIGHEST input precision, so a GPU
batch-size change cannot silently switch the contracted inputs to reduced mantissas.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from ir2_benchmark.models.ir2_policy import action_padding_mask
from ir2_benchmark.models.plan_predictor import PlanEstimate
from ir2_benchmark.navigation.candidates import frontier_mask
from ir2_benchmark.navigation.ir2_graph import GraphObservation
from ir2_benchmark.navigation.jax_path import navigate
from ir2_benchmark.navigation.one_hop import navigate_edge
from ir2_benchmark.sensing.jax_sensor import build_sensor_geometry
from ir2_benchmark.types import LocalObservation


class ActorFeatures(NamedTuple):
    """No privileged fields: one-hot patch, candidate features and local history."""

    local_patch: Array
    candidates: Array
    candidate_mask: Array
    self_features: Array
    neighbor_features: Array
    neighbor_mask: Array
    graph: GraphObservation | None = None


class MapEncoder(nn.Module):
    """Spatial encoding with bounded output independent of map bucket dimensions."""

    features: int = 128

    @nn.compact
    def __call__(self, image: Array) -> Array:
        """Encode spatial channels without exporting full-map activations to memory."""
        for channels in (32, 64, 64):
            image = nn.silu(
                nn.Conv(channels, (3, 3), strides=(2, 2), precision=jax.lax.Precision.HIGHEST)(
                    image
                )
            )
        return nn.silu(
            nn.Dense(self.features, precision=jax.lax.Precision.HIGHEST)(
                jnp.mean(image, axis=(-3, -2))
            )
        )


class ObservationEncoder(nn.Module):
    """Permutation-invariant pooling over visible neighbor estimates."""

    hidden: int = 256

    @nn.compact
    def __call__(self, features: ActorFeatures, graph_context: Array | None = None) -> Array:
        """Combine only local actor inputs into a shared fixed-width representation."""
        map_features = (
            MapEncoder()(features.local_patch) if graph_context is None else graph_context
        )
        neighbors = nn.silu(
            nn.Dense(128, precision=jax.lax.Precision.HIGHEST)(
                jnp.where(features.neighbor_mask[..., None], features.neighbor_features, 0)
            )
        )
        weights = features.neighbor_mask.astype(jnp.float32)[..., None]
        pooled = jnp.sum(neighbors * weights, axis=-2) / jnp.maximum(jnp.sum(weights, axis=-2), 1)
        own = nn.silu(nn.Dense(64, precision=jax.lax.Precision.HIGHEST)(features.self_features))
        return nn.silu(
            nn.Dense(self.hidden, precision=jax.lax.Precision.HIGHEST)(
                jnp.concatenate((map_features, own, pooled), -1)
            )
        )


def zoh_estimate(observation: LocalObservation) -> PlanEstimate:
    """Expose cached PLAN data with original timestamps and missing-plan flags."""
    scale = jnp.maximum(observation.valid_shape[::-1] - 1, 1)
    horizon = observation.peer_plans.shape[-2]
    present = observation.peer_plan_event >= 0
    lengths = jnp.where(present, observation.peer_plan_lengths, 0)
    return PlanEstimate(
        jnp.where(present[:, None], observation.peer_goals / scale, 0),
        jnp.where(present[:, None, None], observation.peer_plans / scale, 0),
        jnp.full((present.shape[0], horizon + 1, 2), 1e-4),
        jnp.log(jax.nn.one_hot(lengths, horizon + 1) + 1e-8),
        observation.peer_plan_event,
        ~present,
    )


def no_plan_estimate(observation: LocalObservation) -> PlanEstimate:
    """Disable all neighbor intentions while retaining real received position ages."""
    peers, horizon = observation.peer_plans.shape[:2]
    return PlanEstimate(
        jnp.zeros((peers, 2)),
        jnp.zeros((peers, horizon, 2)),
        jnp.ones((peers, horizon + 1, 2)),
        jnp.zeros((peers, horizon + 1)),
        observation.peer_position_event,
        jnp.ones(peers, bool),
    )


def actor_features(  # noqa: PLR0913
    observation: LocalObservation,
    n_agents: int,
    estimate: PlanEstimate | None = None,
    *,
    initial_budget: int = 8192,
    max_rounds: int = 512,
    patch_radius: int = 8,
    use_neighbor_plan: bool = True,
    candidate_attention: bool = True,
) -> ActorFeatures:
    """Build the same local feature schema for real and simulated local observations.

    Bucket dimensions never normalize coordinates. The patch contains remembered
    local knowledge; candidate graph distances retain global reachable-map context.
    """
    scale = jnp.maximum(observation.valid_shape[::-1] - 1, 1)
    position = observation.position / scale
    graph = observation.graph
    if graph is not None:
        patch = jnp.zeros((1, 1), dtype=jnp.int32)
        candidate_data = graph.node_inputs[graph.edge_inputs]
        candidate_mask = ~action_padding_mask(graph.edge_padding_mask)
    else:
        padding = jnp.pad(observation.local_map, patch_radius, constant_values=2)
        patch = jax.lax.dynamic_slice(
            padding,
            (observation.position[1], observation.position[0]),
            (2 * patch_radius + 1, 2 * patch_radius + 1),
        )
        candidates = observation.candidates
        frontier = frontier_mask(observation.local_map, observation.valid_shape)
        points = candidates.positions
        candidate_data = jnp.concatenate(
            (
                points / scale,
                (points - observation.position) / scale,
                (candidates.distances / jnp.sum(scale))[:, None],
                frontier[points[:, 1], points[:, 0]][:, None],
                (jnp.arange(points.shape[0]) == 0)[:, None],
            ),
            -1,
        )
        if candidate_attention:
            geometry = build_sensor_geometry(min(patch_radius, 8))

            def frontier_utility(point: Array) -> Array:
                targets = point + jnp.asarray(geometry.target_offsets)
                blockers = point + jnp.asarray(geometry.blocker_offsets)
                limits = observation.valid_shape[::-1]
                target_valid = jnp.all((targets >= 0) & (targets < limits), axis=-1)
                blocker_valid = jnp.all((blockers >= 0) & (blockers < limits), axis=-1)
                occupied = observation.local_map[blockers[..., 1], blockers[..., 0]] != 0
                blocked = jnp.any(
                    jnp.asarray(geometry.blocker_mask) & (occupied | ~blocker_valid), axis=-1
                )
                return jnp.mean(frontier[targets[:, 1], targets[:, 0]] & target_valid & ~blocked)

            utility = jax.vmap(frontier_utility)(points)
            candidate_data = jnp.concatenate((candidate_data, utility[:, None]), axis=-1)
        candidate_mask = candidates.mask
    own = jnp.concatenate(
        (
            position,
            observation.own_goal / scale,
            jnp.asarray(
                [
                    jnp.log1p(observation.balance) / jnp.log1p(32768),
                    jnp.log1p(initial_budget) / jnp.log1p(32768),
                    observation.round / max_rounds,
                ]
            ),
        )
    )
    intent = (
        (zoh_estimate(observation) if estimate is None else estimate)
        if use_neighbor_plan
        else no_plan_estimate(observation)
    )
    ages = jnp.where(
        intent.last_message_event >= 0, observation.event - intent.last_message_event, -1
    )
    peer_position = jnp.where(
        (observation.peer_position_event >= 0)[:, None], observation.peer_positions / scale, 0
    )
    neighbors = jnp.concatenate(
        (
            intent.goal,
            intent.positions.reshape((intent.positions.shape[0], -1)),
            jnp.sqrt(intent.variance).reshape((intent.goal.shape[0], -1)),
            jax.nn.softmax(intent.length_logits, -1),
            jnp.log1p(jnp.maximum(ages, 0))[:, None],
            intent.missing[:, None],
            peer_position,
            (observation.peer_position_event < 0)[:, None],
        ),
        -1,
    )
    if not use_neighbor_plan:
        neighbors = neighbors.at[:, : 5 * intent.positions.shape[-2] + 5].set(0)
    ids = jnp.arange(intent.goal.shape[0])
    neighbor_mask = (ids < n_agents) & (ids != observation.agent_id)
    if candidate_attention:
        neighbor_mask &= ~intent.missing | (observation.peer_position_event >= 0)
    return ActorFeatures(
        jax.nn.one_hot(patch, 3),
        candidate_data,
        candidate_mask,
        own,
        neighbors,
        neighbor_mask,
        graph,
    )


def selected_plan_features(
    observation: LocalObservation, waypoint: Array, max_distance: int, plan_horizon: int = 5
) -> Array:
    """Encode exactly the declared local action, sharing the real navigation contract."""
    route = (
        navigate_edge(observation.position, waypoint, observation.candidates, plan_horizon)
        if observation.tree is None
        else navigate(
            observation.local_map,
            observation.position,
            waypoint,
            max_distance,
            plan_horizon,
            observation.tree,
        )
    )
    scale = jnp.maximum(observation.valid_shape[::-1] - 1, 1)
    return jnp.concatenate(
        (
            route.goal / scale,
            (route.plan / scale).reshape(-1),
            jnp.asarray([route.plan_valid_length / plan_horizon]),
        )
    )
