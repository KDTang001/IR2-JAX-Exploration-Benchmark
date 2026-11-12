"""Training targets are separate from every predictor forward-input PyTree."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from flax.core import FrozenDict
    from flax.training.train_state import TrainState

    from ir2_benchmark.models.plan_predictor import PlanPrediction, PlanPredictor
    from ir2_benchmark.training.collect_a import PlanTrainingBatch
    from ir2_benchmark.types import PlanLabel


class PlanTarget(NamedTuple):
    """Current generated intention, not the neighbor's future executed trajectory."""

    goal: Array
    positions: Array
    valid_length: Array
    available: Array
    generated_event: Array


class PlanLoss(NamedTuple):
    """Separately report Gaussian position NLL, length CE and categorical KL."""

    total: Array
    position_nll: Array
    length_ce: Array
    kl: Array


def target_from_labels(labels: PlanLabel, valid_shape: Array, active_mask: Array) -> PlanTarget:
    """Snapshot the labels at the current pre-action event boundary."""
    scale = jnp.maximum(jnp.asarray(valid_shape)[::-1] - 1, 1)
    return PlanTarget(
        labels.goals / scale,
        labels.positions / scale,
        labels.valid_lengths,
        active_mask & (labels.generated_event >= 0),
        labels.generated_event,
    )


def plan_loss(prediction: PlanPrediction, target: PlanTarget, kl_weight: float = 0.01) -> PlanLoss:
    """Missing labels contribute no coordinate, length or KL supervision."""
    estimate = prediction.estimate
    mean = jnp.concatenate((estimate.goal[..., None, :], estimate.positions), -2)
    truth = jnp.concatenate((target.goal[..., None, :], target.positions), -2)
    horizon = target.positions.shape[-2]
    mask = jnp.concatenate(
        (
            jnp.ones((*target.valid_length.shape, 1), bool),
            jnp.arange(horizon) < target.valid_length[..., None],
        ),
        -1,
    )
    mask &= target.available[..., None]
    coordinate_nll = 0.5 * (
        jnp.log(2 * jnp.pi * estimate.variance) + jnp.square(mean - truth) / estimate.variance
    )
    nll = jnp.sum(jnp.where(mask[..., None], coordinate_nll, 0)) / jnp.maximum(2 * jnp.sum(mask), 1)
    length_log_probs = jax.nn.log_softmax(estimate.length_logits, -1)
    ce = -jnp.take_along_axis(length_log_probs, target.valid_length[..., None], -1)[..., 0]
    denominator = jnp.maximum(jnp.sum(target.available), 1)
    ce = jnp.sum(jnp.where(target.available, ce, 0)) / denominator
    posterior_lp = jax.nn.log_softmax(prediction.posterior_logits, -1)
    prior_lp = jax.nn.log_softmax(prediction.prior_logits, -1)
    kl = jnp.sum(jnp.exp(posterior_lp) * (posterior_lp - prior_lp), axis=(-2, -1))
    kl = jnp.sum(jnp.where(target.available, kl, 0)) / denominator
    return PlanLoss(nll + ce + kl_weight * kl, nll, ce, kl)


def plan_update(
    model: PlanPredictor, state: TrainState, batch: PlanTrainingBatch
) -> tuple[TrainState, Array]:
    """Supervise recent real-event intent snapshots after the frozen PPO phase."""

    def objective(params: FrozenDict) -> Array:
        predictions = jax.vmap(
            lambda memory, inputs, key: model.apply({"params": params}, memory, inputs, key)
        )(batch.memory, batch.inputs, batch.key)
        return plan_loss(predictions, batch.target).total

    loss, gradients = jax.value_and_grad(objective)(state.params)
    return state.apply_gradients(grads=gradients), loss
