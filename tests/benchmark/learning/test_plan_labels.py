"""Intent labels follow generation-event boundaries and cannot affect forward input."""

import jax
import jax.numpy as jnp
import numpy as np

from ir2_benchmark.codecs.packet import forward_packet
from ir2_benchmark.codecs.standard import encode_plan
from ir2_benchmark.communication.channel import broadcast
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.models.plan_predictor import (
    PlanInput,
    PlanPredictor,
    initial_plan_memory,
    plan_inputs,
)
from ir2_benchmark.training.plan_losses import PlanTarget, plan_loss, target_from_labels
from ir2_benchmark.types import EnvState, MapInput, PlanLabel


def test_targets_when_sender_updates_intention_after_receiver_boundary() -> None:
    # Given: event 1's pre-action snapshot still contains sender zero's event-0 P0.
    before = PlanLabel(
        jnp.asarray([[3, 11], [5, 7]]),
        jnp.zeros((2, 5, 2), jnp.int32),
        jnp.asarray([3, 2]),
        jnp.asarray([0, -1]),
    )
    after = before._replace(
        goals=before.goals.at[0].set(jnp.asarray([9, 4])),
        generated_event=before.generated_event.at[0].set(2),
    )
    # When: generate training targets separately at the two event boundaries.
    p0 = target_from_labels(before, jnp.asarray([20, 40]), jnp.ones(2, bool))
    p1 = target_from_labels(after, jnp.asarray([20, 40]), jnp.ones(2, bool))
    # Then: target coordinates use valid width/height, and no future P1 leaks into P0.
    np.testing.assert_allclose(p0.goal[0], [3 / 39, 11 / 19])
    np.testing.assert_allclose(p1.goal[0], [9 / 39, 4 / 19])
    np.testing.assert_array_equal(p0.available, [True, False])


def test_forward_when_unsent_training_target_changes() -> None:
    # Given: one receiver-visible input and two different privileged training targets.
    model = PlanPredictor(hidden=32, categories=2, classes=3)
    inputs = PlanInput(
        jnp.zeros(6), jnp.zeros(16), jnp.zeros(16, bool), jnp.int32(0), jnp.int32(-1), jnp.int32(-1)
    )
    memory, key = initial_plan_memory(hidden=32, categories=2, classes=3), jax.random.PRNGKey(3)
    variables = model.init(key, memory, inputs, key)
    first = PlanTarget(
        jnp.asarray([0.1, 0.8]),
        jnp.zeros((5, 2)),
        jnp.int32(3),
        jnp.asarray(1, dtype=bool),
        jnp.int32(0),
    )
    second = first._replace(goal=jnp.asarray([0.8, 0.1]))
    # When: forward uses the identical visible input while loss targets differ.
    prediction = model.apply(variables, memory, inputs, key)
    repeated = model.apply(variables, memory, inputs, key)
    # Then: the forward state is identical and only supervised losses depend on the unsent intent.
    np.testing.assert_array_equal(prediction.estimate.goal, repeated.estimate.goal)
    assert not np.isclose(
        float(plan_loss(prediction, first).total), float(plan_loss(prediction, second).total)
    )
    assert "target" not in PlanInput._fields


def test_loss_when_label_is_missing_does_not_supervise_zero_coordinates() -> None:
    # Given: an unlabelled receiver-neighbor pair has arbitrary placeholder coordinates.
    model = PlanPredictor(hidden=32, categories=2, classes=3)
    inputs = PlanInput(
        jnp.zeros(6), jnp.zeros(16), jnp.zeros(16, bool), jnp.int32(0), jnp.int32(-1), jnp.int32(-1)
    )
    memory, key = initial_plan_memory(hidden=32, categories=2, classes=3), jax.random.PRNGKey(2)
    variables = model.init(key, memory, inputs, key)
    target = PlanTarget(
        jnp.ones(2), jnp.ones((5, 2)), jnp.int32(0), jnp.asarray(0, dtype=bool), jnp.int32(-1)
    )
    # When: evaluate all supervised predictor losses with the missing-label mask.
    loss = plan_loss(model.apply(variables, memory, inputs, key), target)
    # Then: placeholder positions do not become zero-coordinate supervision or KL targets.
    np.testing.assert_array_equal(loss, [0.0, 0.0, 0.0, 0.0])


def _forward_to_receiver(plan_event: int) -> tuple[EnvState, HostConfig]:
    """Deliver a paid forwarded PLAN after a more recent POSITION measurement."""
    grid = jnp.zeros((12, 12), jnp.uint8)
    data = MapInput(grid, grid == 0, jnp.asarray([12, 12]), jnp.asarray([4, 3]))
    config = HostConfig(n_agents=3, sensor_radius=1, initial_budget=500, max_candidates=2)
    state = reset(data, config, jax.random.PRNGKey(2), spawns=jnp.asarray([[3, 3], [4, 3], [5, 3]]))
    state = state._replace(
        event=jnp.int32(12),
        peer_positions=state.peer_positions.at[1, 0].set(jnp.asarray([3, 3])),
        peer_goals=state.peer_goals.at[1, 0].set(jnp.asarray([4, 3])),
        peer_plans=state.peer_plans.at[1, 0].set(jnp.asarray([[3, 3]] * 5)),
        peer_plan_lengths=state.peer_plan_lengths.at[1, 0].set(2),
        peer_position_event=state.peer_position_event.at[1, 0].set(10),
        peer_goal_event=state.peer_goal_event.at[1, 0].set(plan_event),
        peer_plan_event=state.peer_plan_event.at[1, 0].set(plan_event),
    )
    original = encode_plan(
        jnp.asarray([3, 3]),
        jnp.asarray([5, 3]),
        jnp.asarray([[4, 3], [5, 3], [5, 3], [5, 3], [5, 3]]),
        2,
        0,
        1,
        9,
        3,
    )
    forwarded = forward_packet(original, 2, 1, 11, 3)
    paid = broadcast(
        state.inbox,
        forwarded,
        2,
        state.balances[2],
        jnp.asarray(1, bool),
        jnp.ones(3, bool),
        jnp.ones(3, bool),
        state.active_mask,
    )
    return state._replace(inbox=paid.inbox, balances=state.balances.at[2].set(paid.balance)), config


def test_measurement_when_old_plan_is_forwarded_again_is_not_new_evidence() -> None:
    # Given: author zero's event-9 PLAN was already accepted; a new relay header replays it.
    state, config = _forward_to_receiver(plan_event=9)
    # When: receiver one overlays the paid stale origin and builds the actual model input.
    observation = observe(state, jnp.int32(1), config)
    inputs = jax.jit(plan_inputs)(observation)
    # Then: no repeated PLAN/POSITION/GOAL becomes a fresh posterior measurement.
    assert not bool(jnp.any(inputs.measurement_mask[0]))
    assert bool(observation.inbox_present[2])


def test_measurement_when_late_plan_is_newer_than_plan_but_older_than_position() -> None:
    # Given: POSITION10 and PLAN7 are cached, and the separately valid PLAN9 arrives late.
    state, config = _forward_to_receiver(plan_event=7)
    # When: modality-specific freshness is used at the receiver's event boundary.
    observation = observe(state, jnp.int32(1), config)
    inputs = jax.jit(plan_inputs)(observation)
    # Then: PLAN/GOAL update without replaying the newer position as a fake new measurement.
    assert bool(observation.fresh_plan[0])
    assert not bool(observation.fresh_position[0])
    np.testing.assert_array_equal(inputs.measurement_mask[0, :2], [False, False])
    np.testing.assert_array_equal(inputs.measurement_mask[0, 2:4], [True, True])
    assert bool(jnp.any(inputs.measurement_mask[0, 4:]))
