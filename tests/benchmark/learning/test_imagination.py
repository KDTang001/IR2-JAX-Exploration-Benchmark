"""Imagined dynamics must pass actual protocol/ledger/local-state rules."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.reset import reset
from ir2_benchmark.models.world_model import WorldPrediction, initial_rssm
from ir2_benchmark.training.imagination import (
    ImaginedState,
    _update_tile,
    apply_prediction,
    cognitive_from_environment,
    imagined_observe,
)
from ir2_benchmark.types import MapInput, make_action


def imagined(budget: int = 50) -> tuple[ImaginedState, HostConfig]:
    config = HostConfig(sensor_radius=1, initial_budget=budget, max_candidates=8, max_distance=2)
    grid = jnp.zeros((9, 9), jnp.uint8)
    data = MapInput(grid, grid == 0, jnp.array([9, 9]), jnp.array([4, 4]))
    real = reset(data, config, jax.random.PRNGKey(2))
    cognition = cognitive_from_environment(real)
    return ImaginedState(
        cognition, initial_rssm(deterministic=8, categories=2, classes=3), jnp.zeros((2, 8))
    ), config


def controlled(state: ImaginedState, delivered: bool = False) -> WorldPrediction:  # noqa: FBT001, FBT002
    latent = state.latent
    logits = jnp.zeros((2, 3))
    local = jnp.zeros((3, 3, 3)).at[..., 2].set(100)
    return WorldPrediction(
        latent,
        logits,
        logits,
        local,
        jnp.array([100.0, -100.0, -100.0]),
        jnp.full(2, 100.0 if delivered else -100.0),
        jnp.float32(0.3),
        jnp.float32(100.0),
    )


def test_one_hop_imagination_reaches_selected_edge_despite_zero_move_logits() -> None:
    state, config = imagined()
    state = state._replace(
        cognition=state.cognition._replace(local_maps=jnp.zeros_like(state.cognition.local_maps))
    )
    obs = imagined_observe(state.cognition, config)
    index = jnp.argmax(jnp.where(obs.candidates.mask, obs.candidates.distances, -1))
    goal = obs.candidates.positions[index]
    result = apply_prediction(
        state, make_action(goal), controlled(state), config, jax.random.PRNGKey(22), observation=obs
    )
    np.testing.assert_array_equal(result.state.cognition.positions[0], goal)
    np.testing.assert_allclose(result.distance, obs.candidates.distances[index])
    assert int(result.state.cognition.plan_lengths[0]) == 1
    assert int(result.state.cognition.graph_count[0]) > 1


def test_imagined_packet_is_rejected_when_balance_below_actual_length() -> None:
    # Given: 44 bytes cannot pay for a 45-byte PLAN despite certain predicted delivery.
    state, config = imagined(44)
    action = make_action(state.cognition.positions[0], send=True)
    # When: the controlled model predicts a delivered message.
    result = jax.jit(
        lambda s: apply_prediction(s, action, controlled(s, True), config, jax.random.PRNGKey(3))  # noqa: FBT003
    )(state)
    # Then: exact ledger eligibility overrides the learned delivery head.
    assert int(result.state.cognition.balances[0]) == 44  # noqa: PLR2004
    assert int(result.spent) == 0
    assert not np.any(result.delivered)


def test_imagined_attempt_is_charged_when_model_predicts_all_loss() -> None:
    # Given: enough budget but no receiver will get this paid attempt.
    state, config = imagined(50)
    action = make_action(state.cognition.positions[0], send=True)
    # When: exact transmission accounting runs under an all-loss prediction.
    result = apply_prediction(state, action, controlled(state), config, jax.random.PRNGKey(4))
    # Then: lost broadcasts are still paid.
    assert int(result.state.cognition.balances[0]) == 5  # noqa: PLR2004
    assert int(result.spent) == 45  # noqa: PLR2004


def test_imagined_success_charging_uses_actual_predicted_delivery():
    state, config = imagined(50)
    config = replace(config, charge_mode="success", send_policy="always", message_codec=1)
    action = make_action(state.cognition.positions[0], send=False)
    lost = apply_prediction(state, action, controlled(state, False), config, jax.random.PRNGKey(4))
    delivered = apply_prediction(
        state, action, controlled(state, True), config, jax.random.PRNGKey(4)
    )
    assert int(lost.spent) == 0
    assert int(delivered.spent) == 20
    assert int(delivered.state.cognition.balances[0]) == 30


def test_candidates_recompute_when_new_local_cells_are_predicted() -> None:
    # Given: the actor initially knows only its own free cell.
    state, config = imagined()
    agent = state.cognition.current_agent
    position = state.cognition.positions[agent]
    maps = jnp.full_like(state.cognition.local_maps, 2).at[agent, position[1], position[0]].set(0)
    state = state._replace(cognition=state.cognition._replace(local_maps=maps))
    initial = imagined_observe(state.cognition, config)
    prediction = controlled(state)._replace(local_logits=jnp.zeros((3, 3, 3)).at[..., 0].set(100))
    # When: one modeled event observes neighboring free cells, then this robot acts again.
    result = apply_prediction(
        state, make_action(position), prediction, config, jax.random.PRNGKey(5)
    )
    actor_again = result.state.cognition._replace(current_agent=agent)
    updated = imagined_observe(actor_again, config)
    # Then: a stale candidate table would retain only wait and fail this check.
    assert int(jnp.sum(initial.candidates.mask)) == 1
    assert int(jnp.sum(updated.candidates.mask)) > 1


def test_imagined_state_has_no_true_map_when_initialized_from_real_state() -> None:
    # Given / When: projection from a true environment at a replay cutoff.
    state, _ = imagined()
    # Then: no truth, component, physical footprint, or future fault field can reach imagination.
    assert not {"truth", "component", "physical_seen", "isolated_mask"} & set(
        state.cognition._fields
    )


def test_scan_freezes_events_when_known_time_limit_reached() -> None:
    # Given: a two-event time limit and five team rounds of requested imagination.
    state, config = imagined()
    config = HostConfig(
        sensor_radius=1, initial_budget=50, max_candidates=8, max_distance=2, max_rounds=1
    )

    def event(
        carry: ImaginedState, _: jax.Array
    ) -> tuple[ImaginedState, tuple[jax.Array, jax.Array]]:
        action = make_action(carry.cognition.positions[carry.cognition.current_agent])
        result = apply_prediction(carry, action, controlled(carry), config, jax.random.PRNGKey(6))
        return result.state, (result.valid, result.state.cognition.event)

    # When: scanning past the exact task horizon.
    final, (valid, events) = jax.jit(lambda s: jax.lax.scan(event, s, jnp.arange(10)))(state)
    # Then: only active events advance, and all remaining modeled steps are frozen padding.
    assert int(valid.sum()) == 2  # noqa: PLR2004
    assert int(final.cognition.event) == 2  # noqa: PLR2004
    np.testing.assert_array_equal(np.asarray(events[2:]), np.full(8, 2))


def test_tile_update_drops_outside_coordinates_when_sensor_at_corner() -> None:
    # Given: a corner sensor, remembered wall, and all-free decoder predictions.
    local = jnp.full((4, 5), 2, jnp.uint8).at[0, 0].set(0).at[0, 1].set(1)
    times = jnp.full((4, 5), -1, jnp.int32)
    logits = jnp.zeros((5, 5, 3)).at[..., 0].set(100)
    # When: scattering a radius-two prediction at (0,0) with a smaller valid_shape.
    updated, observed = jax.jit(
        lambda m, t: _update_tile(
            m, t, jnp.array([0, 0]), logits, jnp.int32(7), jnp.array([3, 4]), 2
        )
    )(local, times)
    # Then: invalid coordinates drop, circle corners stay unknown, remembered wall persists.
    expected = np.array(
        [[0, 1, 0, 2, 2], [0, 0, 2, 2, 2], [0, 2, 2, 2, 2], [2, 2, 2, 2, 2]], dtype=np.uint8
    )
    np.testing.assert_array_equal(updated, expected)
    assert int(observed[0, 1]) == -1
    assert int(observed[1, 1]) == 7  # noqa: PLR2004


def test_survival_is_counted_once_when_probability_one_half() -> None:
    # Given: exactly two equally likely continuation outcomes at p=.5, selected by their keys.
    state, config = imagined()
    prediction = controlled(state)._replace(continuation_logit=jnp.float32(0))
    keys = jax.random.split(jax.random.PRNGKey(91), 64)
    survive = jax.vmap(lambda key: jax.random.bernoulli(jax.random.split(key, 3)[2], 0.5))(keys)
    live_key = keys[jnp.argmax(survive)]
    stop_key = keys[jnp.argmax(~survive)]
    action = make_action(state.cognition.positions[0])
    # When: enumerating the alive and terminated modeled branches through the actual transition.
    live = apply_prediction(state, action, prediction, config, live_key)
    stopped = apply_prediction(state, action, prediction, config, stop_key)
    expected_discount = 0.5 * live.continuation + 0.5 * stopped.continuation
    # Then: binary sampled survival has expectation .5; multiplying by p again would yield .25.
    assert not bool(live.state.cognition.terminated)
    assert bool(stopped.state.cognition.terminated)
    assert float(live.continuation) == 1.0
    assert float(stopped.continuation) == 0.0
    assert float(expected_discount) == 0.5  # noqa: PLR2004
