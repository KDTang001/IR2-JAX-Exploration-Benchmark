"""Frozen event features preserve the exact sampling distribution during PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from ir2_benchmark.agents.actor import IndexAction
    from ir2_benchmark.agents.critic import PrivilegedFeatures
    from ir2_benchmark.models.encoders import ActorFeatures
    from ir2_benchmark.training.collect_a import Collection, PlanTrainingBatch
    from ir2_benchmark.training.ppo import Returns


class RolloutBatch(NamedTuple):
    """Leading axis is ordered events; actor_id identifies the only optimized actor."""

    actor_features: ActorFeatures
    critic_features: PrivilegedFeatures
    memories: Array
    plan_features: Array
    actions: IndexAction
    old_log_probs: Array
    old_values: Array
    rewards: Array
    terminated: Array
    truncated: Array
    valid: Array
    actor_ids: Array
    predictor_version: Array
    sampled_candidates: Array
    sampled_candidate_mask: Array


@dataclass(frozen=True, slots=True)
class BatchError(ValueError):
    """The update attempted to change the frozen sampling context."""

    field: str

    def __str__(self) -> str:
        """Identify the changed rollout contract field."""
        return f"frozen rollout field changed: {self.field}"


def validate_batch(batch: RolloutBatch) -> None:
    """Check host-owned immutable batch contracts before entering compiled optimization."""
    if not bool(jnp.array_equal(batch.actor_features.candidates, batch.sampled_candidates)):
        raise BatchError(field="candidates")
    if not bool(jnp.array_equal(batch.actor_features.candidate_mask, batch.sampled_candidate_mask)):
        raise BatchError(field="candidate_mask")
    selected = jnp.take_along_axis(
        batch.sampled_candidate_mask, batch.actions.waypoint[..., None], -1
    )[..., 0]
    if not bool(jnp.all(~batch.valid | selected)):
        raise BatchError(field="masked_waypoint")
    versions = jnp.where(batch.valid, batch.predictor_version, batch.predictor_version[0])
    if not bool(jnp.all(versions == versions[0])):
        raise BatchError(field="predictor_version")


def flatten_events(value: Array) -> Array:
    """Pack ordered [environment,event] features after computing per-environment GAE."""
    return value.reshape((-1, *value.shape[2:]))


class TrainingSlice(NamedTuple):
    """Only PPO/predictor inputs survive collection; simulator snapshots do not."""

    rollout: RolloutBatch
    plans: PlanTrainingBatch
    bootstrap: Array


def pack_collections(
    collections: list[Collection | TrainingSlice], target: int, *, sequence_length: int = 1
) -> tuple[RolloutBatch, PlanTrainingBatch, Returns]:
    """Compute event GAE, then optionally group each stream/robot for truncated BPTT.

    Each collector stops at episode termination. Splitting at collection boundaries
    therefore also isolates episodes. Predictor training retains flat real events.
    """
    # PPO imports RolloutBatch; defer this runtime dependency until storage is initialized.
    from ir2_benchmark.training.ppo import event_gae  # noqa: PLC0415

    batches, plans, return_batches = [], [], []
    remaining = target
    for collection in collections:
        raw = collection.rollout
        values = jnp.concatenate((raw.old_values, collection.bootstrap[:, None]), axis=1)
        returns = jax.vmap(event_gae)(raw.rewards, values, raw.terminated, raw.truncated, raw.valid)
        indices = jnp.asarray(np.flatnonzero(np.asarray(raw.valid).ravel())[:remaining])
        remaining -= len(indices)
        if not len(indices):
            continue
        batches.append(
            jax.tree.map(lambda value, indices=indices: flatten_events(value)[indices], raw)
        )
        plans.append(
            jax.tree.map(
                lambda value, indices=indices: flatten_events(value)[indices], collection.plans
            )
        )
        return_batches.append(
            jax.tree.map(lambda value, indices=indices: flatten_events(value)[indices], returns)
        )
    if remaining:
        message = "collection contains fewer real events than the requested batch"
        raise ValueError(message)
    batch = jax.tree.map(lambda *arrays: jnp.concatenate(arrays), *batches)
    plan_batch = jax.tree.map(lambda *arrays: jnp.concatenate(arrays), *plans)
    returns = jax.tree.map(lambda *arrays: jnp.concatenate(arrays), *return_batches)
    validate_batch(batch)
    if sequence_length > 1:
        sequence_indices = []
        sequence_masks = []
        offset = 0
        for collection in collections:
            valid = np.asarray(collection.rollout.valid)
            actors = np.asarray(collection.rollout.actor_ids)
            compact = np.cumsum(valid.ravel()).reshape(valid.shape) - 1 + offset
            for lane in range(valid.shape[0]):
                for agent in np.unique(actors[lane, valid[lane]]):
                    indices = compact[lane, valid[lane] & (actors[lane] == agent)]
                    indices = indices[indices < target]
                    for start in range(0, len(indices), sequence_length):
                        segment = indices[start : start + sequence_length]
                        sequence_indices.append(
                            np.pad(segment, (0, sequence_length - len(segment)), mode="edge")
                        )
                        sequence_masks.append(np.arange(sequence_length) < len(segment))
            offset += int(valid.sum())
        indices = jnp.asarray(np.stack(sequence_indices))
        batch, returns = jax.tree.map(lambda value: value[indices], (batch, returns))
        batch = batch._replace(valid=batch.valid & jnp.asarray(sequence_masks))
    return batch, plan_batch, returns


class StandardAction(NamedTuple):
    """Actual standard-codec action, including heuristic waypoint overrides.

    All three leaves have the same leading [environment,event] axes. Standard
    packets are locked by step_agent; no user payload is needed for replay.
    """

    waypoint: Array
    send: Array
    codec: Array


class CollectionTrace(NamedTuple):
    """Small execution evidence without storing full maps along the time axis."""

    event: Array
    agent_id: Array
    valid: Array
    positions: Array
    goals: Array
    balances: Array
    spent: Array
    delivered: Array
    coverage: Array
    reward: Array
    terminated: Array
    truncated: Array
    invalid_action: Array
