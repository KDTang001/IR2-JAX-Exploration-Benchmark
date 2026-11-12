"""Real local-input stochastic intent inference and supervised fitting."""

# Independent Gaussian/coordinate fixtures intentionally retain explicit values.
# ruff: noqa: PLR2004

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from jax import Array

from ir2_benchmark.models.plan_predictor import (
    PlanInput,
    PlanPredictor,
    initial_plan_memory,
    initial_zoh,
    update_zoh,
)
from ir2_benchmark.training.plan_losses import PlanTarget, plan_loss


def _inputs() -> PlanInput:
    return PlanInput(
        jnp.zeros(6), jnp.zeros(16), jnp.zeros(16, bool), jnp.int32(0), jnp.int32(-1), jnp.int32(-1)
    )


def test_cold_start_when_no_neighbor_measurement_exists() -> None:
    # Given: no true neighbor position exists in predictor input or memory.
    model = PlanPredictor(hidden=32, categories=2, classes=3)
    memory = initial_plan_memory(hidden=32, categories=2, classes=3)
    inputs, key = _inputs(), jax.random.PRNGKey(8)
    variables = model.init(key, memory, inputs, key)
    # When: run an unmeasured prior transition.
    result = jax.jit(model.apply)(variables, memory, inputs, key)
    # Then: posterior equals the prior, missing remains explicit and variance is positive.
    np.testing.assert_array_equal(result.prior_logits, result.posterior_logits)
    assert bool(result.estimate.missing)
    assert bool(jnp.all(result.estimate.variance >= 1e-4))
    assert int(result.estimate.last_message_event) == -1


def test_zoh_when_position_message_does_not_create_plan() -> None:
    # Given: a receiver has never received a complete PLAN.
    memory = initial_zoh()
    # When: position-only content arrives with no PLAN measurement flag.
    result = jax.jit(update_zoh)(
        memory, jnp.asarray([3.0, 11.0]), jnp.ones((5, 2)), 5, 7, jnp.asarray(0, dtype=bool)
    )
    # Then: the complete-plan cache remains missing with its original timestamp.
    assert not bool(result.has_plan)
    assert int(result.last_message_event) == -1


def test_zoh_when_time_passes_without_new_plan() -> None:
    # Given: the last valid PLAN was generated at event 7.
    first = update_zoh(
        initial_zoh(), jnp.asarray([3.0, 11.0]), jnp.ones((5, 2)), 3, 7, jnp.asarray(1, dtype=bool)
    )
    # When: an event-30 observation contains no new complete plan.
    result = update_zoh(first, jnp.zeros(2), jnp.zeros((5, 2)), 0, 30, jnp.asarray(0, dtype=bool))
    # Then: held coordinates and their generation time remain unchanged.
    np.testing.assert_array_equal(result.goal, [3.0, 11.0])
    assert int(result.last_message_event) == 7
    assert int(result.valid_length) == 3


def test_predictor_when_fitting_fixed_intention_labels() -> None:
    # Given: annotated intention targets supervise an actual local-input stochastic model.
    model = PlanPredictor(hidden=32, categories=2, classes=3)
    memory = initial_plan_memory(hidden=32, categories=2, classes=3)
    inputs, key = _inputs(), jax.random.PRNGKey(9)
    params = model.init(key, memory, inputs, key)["params"]
    target = PlanTarget(
        jnp.asarray([0.2, 0.8]),
        jnp.tile(jnp.asarray([0.3, 0.7]), (5, 1)),
        jnp.int32(4),
        jnp.asarray(1, dtype=bool),
        jnp.int32(1),
    )
    optimizer = optax.adam(3e-3)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(
        current: FrozenDict, state: optax.OptState
    ) -> tuple[FrozenDict, optax.OptState, Array]:
        def loss(candidate: FrozenDict) -> Array:
            return plan_loss(model.apply({"params": candidate}, memory, inputs, key), target).total

        value, gradient = jax.value_and_grad(loss)(current)
        updates, state = optimizer.update(gradient, state, current)
        return optax.apply_updates(current, updates), state, value

    # When: actual gradients fit the fixed intentions without exposing labels to forward inputs.
    values = []
    for _ in range(30):
        params, optimizer_state, value = update(params, optimizer_state)
        values.append(float(value))
    # Then: supervised intent loss decreases, beyond merely being finite.
    assert values[-1] < values[0] - 0.5
