"""One-hop execution and map-message-gated graph fusion."""

# ruff: noqa: FBT001, PLR2004

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ir2_benchmark.codecs.packet import Codec, Packet
from ir2_benchmark.codecs.standard import decode_plan
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env import observe as observation_module
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.models.encoders import selected_plan_features
from ir2_benchmark.navigation.one_hop import clear_segment
from ir2_benchmark.types import EnvState, MapInput, make_action


def fixture(config: HostConfig) -> EnvState:
    """Known map fixture with limited physical birth coverage and two distinct robots."""
    grid = jnp.zeros((64, 64), jnp.uint8)
    state = reset(
        MapInput(grid, grid == 0, jnp.array([64, 64]), jnp.array([9, 9])),
        config,
        jax.random.PRNGKey(2),
        jnp.array([[9, 9], [45, 45]]),
    )
    return state._replace(
        local_maps=jnp.zeros_like(state.local_maps),
        observed_event=jnp.zeros_like(state.observed_event),
    )


def settings() -> HostConfig:
    return HostConfig(
        n_agents=2,
        sensor_radius=1,
        max_distance=1,
        graph_nodes_per_axis=8,
        graph_local_radius=18,
        graph_global_radius=90,
        graph_capacity=64,
        max_candidates=29,
        initial_budget=4096,
        coverage_target=1,
        signal_threshold=-200,
    )


def test_one_hop_reaches_endpoint_counts_euclidean_distance_and_keeps_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr(observation_module, "bfs", forbidden)
    config = settings()
    state = fixture(config)
    obs = observe(state, jnp.int32(0), config)
    goal = jnp.array([18, 18])
    assert obs.tree is None
    assert bool(jnp.any(obs.candidates.mask & jnp.all(obs.candidates.positions == goal, axis=-1)))
    features = selected_plan_features(obs, goal, config.max_distance, config.plan_horizon)
    assert float(features[-1]) == pytest.approx(1 / config.plan_horizon)
    result = jax.jit(lambda s: step_agent(s, make_action(goal, send=True), config))(state)
    np.testing.assert_array_equal(result.state.positions[0], goal)
    assert float(result.info.distance) == pytest.approx(np.sqrt(162))
    assert float(result.state.travel_distance[0]) == pytest.approx(np.sqrt(162))
    assert not bool(result.info.invalid_action)
    packet = decode_plan(
        Packet(result.info.packet_buffer, result.info.packet_length), config.plan_horizon
    )
    assert int(packet.valid_length) == 1
    np.testing.assert_array_equal(packet.positions[0], goal)
    assert int(result.state.graph_count[0]) == 2
    np.testing.assert_array_equal(result.state.graph_nodes[0, :2], [[9, 9], [18, 18]])


@pytest.mark.parametrize("blocker", [1, 2])
def test_blocked_direct_edge_cannot_be_replaced_by_a_detour(blocker: int) -> None:
    config = settings()
    state = fixture(config)
    maps = state.local_maps.at[0, 8:11, 18].set(blocker)
    state = state._replace(local_maps=maps)
    goal = jnp.array([27, 9])
    obs = observe(state, jnp.int32(0), config)
    assert not bool(clear_segment(maps[0], state.positions[0], goal))
    assert not bool(
        jnp.any(obs.candidates.mask & jnp.all(obs.candidates.positions == goal, axis=-1))
    )
    result = step_agent(state, make_action(goal), config)
    assert bool(result.info.invalid_action)
    np.testing.assert_array_equal(result.state.positions[0], state.positions[0])
    assert float(result.info.distance) == 0


def test_free_but_nonadjacent_goal_is_rejected() -> None:
    config = settings()
    state = fixture(config)
    result = step_agent(state, make_action(jnp.array([54, 54])), config)
    assert bool(result.info.invalid_action)
    np.testing.assert_array_equal(result.state.positions[0], state.positions[0])


def test_blocked_history_does_not_hide_a_legal_local_edge() -> None:
    config = replace(settings(), max_candidates=1)
    state = fixture(config)
    state = state._replace(
        local_maps=state.local_maps.at[0, 9, 30].set(1),
        graph_nodes=state.graph_nodes.at[0, 1].set(jnp.array([45, 9])),
        graph_count=state.graph_count.at[0].set(2),
    )
    obs = observe(state, jnp.int32(0), config)
    assert bool(obs.candidates.mask[1])
    assert 0 < float(obs.candidates.distances[1]) < config.graph_local_radius


@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("codec", [Codec.PLAN, Codec.MAP_DELTA])
@pytest.mark.parametrize("drop", [False, True])
def test_graph_fusion_requires_explicit_delivered_map(
    fusion: bool,
    codec: int,
    drop: bool,
) -> None:
    config = replace(
        settings(),
        global_graph_fusion=fusion,
        fault_mode=1 if drop else 0,
        dropout_probability=float(drop),
    )
    state = fixture(config)
    sent = step_agent(state, make_action(jnp.array([18, 18]), send=True, codec=codec), config)
    assert bool(sent.info.delivered[1]) == (not drop)
    obs = observe(sent.state, jnp.int32(1), config)
    imported = fusion and codec == Codec.MAP_DELTA and not drop
    assert (int(obs.graph_count) > 1) == imported
    if imported:
        assert int(sent.info.spent) > 0
        assert bool(
            jnp.any(jnp.all(obs.graph_nodes[: obs.graph_count] == jnp.array([0, 0]), axis=-1))
        )
    received = step_agent(sent.state, make_action(obs.position), config, observation=obs)
    assert int(received.state.graph_count[1]) == int(obs.graph_count)
    np.testing.assert_array_equal(received.state.graph_nodes[1, 0], [45, 45])
