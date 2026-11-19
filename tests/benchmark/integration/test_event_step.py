"""Real ordered events verify information timing, physical coverage and hard freezing."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ir2_benchmark.codecs.packet import Codec, Packet  # noqa: E402
from ir2_benchmark.codecs.standard import decode_map_delta, decode_plan, encode_map_delta  # noqa: E402
from ir2_benchmark.communication.channel import broadcast  # noqa: E402
from ir2_benchmark.config import HostConfig  # noqa: E402
from ir2_benchmark.env.observe import observe  # noqa: E402
from ir2_benchmark.env.reset import reset  # noqa: E402
from ir2_benchmark.env.step import step_agent  # noqa: E402
from ir2_benchmark.types import EnvState, MapInput, make_action  # noqa: E402


def corridor(config: HostConfig) -> EnvState:
    """Create an incompletely observed corridor with nonoverlapping birth cells."""
    truth = jnp.ones((5, 12), jnp.uint8).at[2, 1:11].set(0)
    data = MapInput(truth, truth == 0, jnp.array([5, 12]), jnp.array([1, 2]))
    return reset(data, config, jax.random.PRNGKey(7), jnp.array([[1, 2], [2, 2]]))


def test_later_robot_uses_new_message_and_earlier_robot_waits_until_next_turn() -> None:
    # Given: two robots and a pre-event observation with no received peer plan.
    config = HostConfig(sensor_radius=1, max_distance=1, coverage_target=1)
    initial = corridor(config)
    old = observe(initial, jnp.int32(1), config)
    # When: robot zero sends a plan, then robot one sends its own plan.
    first = step_agent(initial, make_action(jnp.array([2, 2]), send=True), config)
    later = observe(first.state, jnp.int32(1), config)
    second = step_agent(first.state, make_action(jnp.array([3, 2]), send=True), config)
    earlier_next_turn = observe(second.state, jnp.int32(0), config)
    # Then: same-round access follows event order and preserves generation positions.
    assert not bool(old.inbox_present[0])
    assert bool(later.inbox_present[0])
    assert int(later.peer_plan_event[0]) == 0
    np.testing.assert_array_equal(later.peer_positions[0], [1, 2])
    np.testing.assert_array_equal(first.state.positions[0], [2, 2])
    assert int(earlier_next_turn.peer_plan_event[1]) == 1
    assert not bool(second.state.inbox.present[1, 0])


def test_observe_is_pure_and_ignores_hidden_remote_truth() -> None:
    # Given: identical local knowledge with two different distant hidden walls.
    config = HostConfig(sensor_radius=1, max_distance=1)
    state = corridor(config)
    changed = state._replace(truth=state.truth.at[2, 9].set(1), isolated_mask=~state.isolated_mask)
    # When: observations are requested repeatedly and from both hidden states.
    first = observe(state, jnp.int32(0), config)
    second = observe(state, jnp.int32(0), config)
    hidden = observe(changed, jnp.int32(0), config)
    # Then: every observation field agrees and the inbox remains unconsumed.
    for left, right, remote in zip(
        jax.tree_util.tree_leaves(first),
        jax.tree_util.tree_leaves(second),
        jax.tree_util.tree_leaves(hidden),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left, remote)
    assert int(state.event) == 0


def test_ir2_graph_observation_ignores_unsent_peer_state_and_hidden_truth() -> None:
    config = HostConfig(
        n_agents=2,
        navigation_mode="ir2_graph",
        sensor_radius=4,
        graph_nodes_per_axis=6,
        graph_padding_size=80,
        graph_capacity=80,
    )
    truth = jnp.ones((32, 32), jnp.uint8).at[1:31, 1:31].set(0)
    data = MapInput(truth, truth == 0, jnp.array([32, 32]), jnp.array([3, 3]))
    state = reset(data, config, jax.random.PRNGKey(9), jnp.array([[3, 3], [25, 25]]))
    changed = state._replace(
        truth=state.truth.at[20:28, 20:28].set(1),
        positions=state.positions.at[1].set(jnp.array([28, 28])),
        local_maps=state.local_maps.at[1].set(jnp.zeros_like(truth)),
        goals=state.goals.at[1].set(jnp.array([10, 10])),
        plans=state.plans.at[1].set(jnp.full_like(state.plans[1], 10)),
        balances=state.balances.at[1].set(0),
        isolated_mask=~state.isolated_mask,
    )
    compiled = jax.jit(lambda value: observe(value, jnp.int32(0), config))
    expected, actual = compiled(state), compiled(changed)
    for left, right in zip(
        jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_map_delta_is_locked_before_movement_and_new_sensing() -> None:
    # Given: robot zero knows up to x=2 and will reveal x=3 after moving.
    config = HostConfig(sensor_radius=1, max_distance=1, coverage_target=1)
    state = corridor(config)
    # When: a MAP_DELTA is sent during that movement.
    result = step_agent(
        state, make_action(jnp.array([2, 2]), send=True, codec=Codec.MAP_DELTA), config
    )
    decoded = decode_map_delta(Packet(result.info.packet_buffer, result.info.packet_length))
    coordinates = np.asarray(decoded.coordinates)[np.asarray(decoded.valid_mask)]
    # Then: newly observed x=3 remains absent from the already locked packet.
    assert result.state.local_maps[0, 2, 3] == 0
    assert not np.any(np.all(coordinates == [3, 2], axis=1))


def test_paid_map_overlay_does_not_increase_physical_coverage() -> None:
    # Given: a valid remote map entry for a cell never physically seen by receiver zero.
    config = HostConfig(sensor_radius=1, max_distance=1, coverage_target=1)
    state = corridor(config)
    coordinates = jnp.zeros((47, 2), jnp.int32).at[0].set(jnp.array([8, 2]))
    packet = encode_map_delta(
        coordinates,
        jnp.zeros(47, jnp.uint8),
        jnp.int32(1),
        jnp.int32(1),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(2),
    )
    delivered = broadcast(
        state.inbox,
        packet,
        jnp.int32(1),
        jnp.int32(100),
        jnp.bool_(1),
        jnp.ones(2, bool),
        jnp.ones(2, bool),
        state.active_mask,
    )
    state = state._replace(inbox=delivered.inbox)
    before = int(jnp.sum(state.covered))
    # When: receiver zero observes and then waits, committing the paid overlay.
    observation = observe(state, jnp.int32(0), config)
    result = step_agent(state, make_action(jnp.array([1, 2])), config, observation=observation)
    # Then: knowledge changes while physical footprints and covered counts stay unchanged.
    assert observation.local_map[2, 8] == 0
    assert result.state.local_maps[0, 2, 8] == 0
    assert not bool(result.state.physical_seen[0, 2, 8])
    assert int(jnp.sum(result.state.covered)) == before


def test_all_dropped_transmission_still_charges_actual_packet_bytes() -> None:
    # Given: certain external dropout and sufficient initial balance.
    config = HostConfig(
        sensor_radius=1,
        max_distance=1,
        fault_mode=1,
        dropout_probability=1,
        initial_budget=50,
        coverage_target=1,
    )
    state = corridor(config)
    # When: sending the 45-byte five-position PLAN while waiting.
    result = step_agent(state, make_action(jnp.array([1, 2]), send=True), config)
    # Then: transmission costs 45 bytes even though every receiver drops it.
    assert int(result.info.spent) == 45  # noqa: PLR2004
    assert int(result.state.balances[0]) == 5  # noqa: PLR2004
    assert not np.any(result.info.delivered)


def test_link_reachability_uses_post_move_positions() -> None:
    # Given: a custom threshold that distinguishes distance two from distance three.
    config = HostConfig(
        sensor_radius=3, max_distance=2, signal_threshold=-51.001, coverage_target=1
    )
    # The default reference distance35 makes both short distances reachable; an
    # intervening wall instead distinguishes the old ray from the new ray.
    truth = jnp.ones((8, 10), jnp.uint8).at[:7, 1:9].set(0).at[2, 3:5].set(1)
    data = MapInput(truth, truth == 0, jnp.array([8, 10]), jnp.array([1, 1]))
    state = reset(data, config, jax.random.PRNGKey(3), jnp.array([[1, 2], [5, 2]]))
    # When: robot zero moves up, avoiding positive wall length along the new ray.
    result = step_agent(state, make_action(jnp.array([1, 0]), send=True), config)
    # Then: delivery follows the endpoint ray, rather than the blocked birth ray.
    assert bool(result.info.delivered[1])
    plan = decode_plan(Packet(result.info.packet_buffer, result.info.packet_length))
    np.testing.assert_array_equal(plan.source, [1, 2])


def test_time_limit_freezes_state_without_claiming_coverage_completion() -> None:
    # Given: a one-round limit and a long mostly unknown corridor.
    config = HostConfig(sensor_radius=0, max_distance=1, max_rounds=1, coverage_target=1)
    state = corridor(config)
    state = step_agent(state, make_action(state.positions[0]), config).state
    # When: the last event reaches the limit, followed by another attempted action.
    last = step_agent(state, make_action(state.positions[1]), config)
    frozen = step_agent(last.state, make_action(jnp.array([5, 2]), send=True), config)
    # Then: truncated is distinct and all subsequent movement/accounting is frozen.
    assert bool(last.truncated)
    assert not bool(last.terminated)
    assert not bool(frozen.info.active)
    assert float(frozen.reward) == 0
    for left, right in zip(
        jax.tree_util.tree_leaves(last.state), jax.tree_util.tree_leaves(frozen.state), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_padded_robots_never_observe_move_or_receive_budget() -> None:
    # Given: two real robots in a six-slot bucket.
    config = HostConfig(sensor_radius=1, max_distance=1, agent_bucket=6, coverage_target=1)
    state = corridor(config)
    # When: one complete real round is executed.
    first = step_agent(state, make_action(state.positions[0]), config)
    result = step_agent(first.state, make_action(state.positions[1]), config)
    # Then: padded entries remain inactive and event order returns to robot zero.
    assert int(result.state.current_agent) == 0
    assert int(result.state.round) == 1
    np.testing.assert_array_equal(result.state.local_maps[2:], 2)
    assert not np.any(result.state.physical_seen[2:])
    np.testing.assert_array_equal(result.state.balances[2:], 0)


def test_compiled_event_matches_eager_state() -> None:
    # Given: the same state and action for both execution modes.
    config = HostConfig(sensor_radius=1, max_distance=1, coverage_target=1)
    state = corridor(config)
    action = make_action(jnp.array([2, 2]), send=True)
    # When: executing an eager and compiled event.
    eager = step_agent(state, action, config)
    compiled = jax.jit(lambda s, a: step_agent(s, a, config))(state, action)
    # Then: all state, packet, budget, reward and done fields agree.
    for left, right in zip(
        jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(compiled), strict=True
    ):
        np.testing.assert_allclose(left, right)
