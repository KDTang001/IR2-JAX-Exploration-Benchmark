"""Phase 1 acceptance: physical sensing, paid fusion, failure and isolation boundaries."""

from types import SimpleNamespace
import importlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ir2_benchmark.codecs.packet import Codec
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.evaluation.episode import summarize_execution
from ir2_benchmark.types import MapInput, make_action, validate_graph_state


def world(**options):
    config = HostConfig(
        navigation_mode="ir2_graph", n_agents=2, sensor_radius=8, max_rounds=20,
        graph_nodes_per_axis=12, graph_padding_size=160, graph_capacity=160,
        graph_local_radius=16, graph_global_radius=40, graph_peer_radius=8,
        graph_merge_enabled=False, initial_budget=65536, signal_threshold=-200,
        coverage_target=1, max_candidates=9, **options,
    )
    truth = jnp.zeros((40, 48), jnp.uint8).at[0].set(1).at[-1].set(1)
    truth = truth.at[:, 0].set(1).at[:, -1].set(1)
    data = MapInput(truth, truth == 0, jnp.array(truth.shape), jnp.array([14, 14]))
    return config, reset(data, config, jax.random.PRNGKey(2), jnp.array([[14, 14], [20, 14]]))


@pytest.mark.parametrize("fusion,dropped", [(False, False), (True, False), (True, True)])
def test_ordered_map_episode_matches_sensor_oracle_and_retains_history(fusion, dropped):
    config, state = world(global_graph_fusion=fusion, dropout_probability=float(dropped),
                          fault_mode=int(dropped))
    initial = state

    def advance(value):
        obs = observe(value, value.current_agent, config)
        # Prefer under-visited nodes, then move far; decisions remain receiver-local.
        visits = obs.graph.node_inputs[obs.graph.edge_inputs, 4]
        scores = jnp.where(obs.candidates.mask, obs.candidates.distances - 100 * visits, -1e9)
        goal = jnp.where(jnp.any(obs.candidates.mask), obs.candidates.positions[jnp.argmax(scores)], obs.position)
        return step_agent(value, make_action(goal, send=True, codec=Codec.MAP_DELTA), config, observation=obs)

    advance = jax.jit(advance)
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    sensor = importlib.import_module("sensor")
    truth = np.where(np.asarray(state.truth) == 0, 255, 1)
    physical = np.asarray(state.physical_seen).copy()
    total_spent, delivered, movements = 0, 0, 0
    for _ in range(44):  # 40 live events plus frozen tail, unless coverage completes earlier.
        before = state
        result = advance(state)
        state = result.state
        validate_graph_state(state)
        if bool(result.info.active):
            agent = int(result.info.agent_id)
            pose = np.asarray(state.positions[agent])
            blank = np.full(truth.shape, 127, np.int32)
            footprint = sensor.sensor_work(pose, config.sensor_radius, blank, truth) == 255
            physical[agent] |= footprint
            np.testing.assert_array_equal(state.physical_seen, physical)
            np.testing.assert_array_equal(state.covered, np.any(physical, axis=0))
            assert int(state.graph_memory.history_count[agent]) == int(before.graph_memory.history_count[agent]) + 1
            total_spent += int(result.info.spent)
            delivered += int(np.asarray(result.info.delivered).sum())
            movements += int(np.any(np.asarray(before.positions[agent]) != pose))
        else:
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(state), strict=True):
                np.testing.assert_array_equal(a, b)
            assert float(result.reward) == 0
    assert bool(state.terminated | state.truncated)
    assert movements >= 8
    assert total_spent > 0
    assert int(jnp.sum(initial.balances - state.balances)) == total_spent
    assert (delivered == 0) == dropped
    has_peer_map = bool(jnp.any(state.peer_maps != 2))
    assert has_peer_map == (not dropped)
    assert np.all(np.asarray(state.graph_memory.history_count) > 1)
    if dropped:
        assert np.all(np.asarray(state.peer_position_event) == -1)


@pytest.mark.parametrize("failure", ["history", "path"])
def test_failure_latches_and_evaluation_rejects_invalid_graph(failure, tmp_path):
    config, state = world()
    memory = state.graph_memory
    if failure == "history":
        memory = memory._replace(history_count=memory.history_count.at[0].set(memory.history.shape[1]))
    else:
        memory = memory._replace(path_failed=memory.path_failed.at[0].set(True))
    state = state._replace(graph_memory=memory)
    result = jax.jit(lambda value: step_agent(value, make_action(value.positions[0]), config))(state)
    obs = observe(result.state, jnp.int32(0), config)
    assert not np.any(obs.candidates.mask)
    with pytest.raises(ValueError, match="capacity exhausted|disconnected"):
        validate_graph_state(result.state)
    execution = SimpleNamespace(rollout=SimpleNamespace(final_state=result.state, trace=None))
    task = SimpleNamespace(scenario=None, method=None)
    with pytest.raises(ValueError, match="capacity exhausted|disconnected"):
        summarize_execution(task, execution, tmp_path / "must-not-be-written.json")
    assert not (tmp_path / "must-not-be-written.json").exists()


@pytest.mark.parametrize("terminal", ["terminated", "truncated"])
def test_ir2_terminal_state_freezes_every_field(terminal):
    config, state = world()
    state = state._replace(**{terminal: jnp.bool_(True)})
    result = jax.jit(lambda value: step_agent(value, make_action(jnp.array([25, 25]), send=True), config))(state)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(result.state), strict=True):
        np.testing.assert_array_equal(before, after)
    assert float(result.reward) == 0
    assert not bool(result.info.active)
    assert int(result.info.spent) == 0


def test_nonzero_rendezvous_observation_isolated_from_hidden_peer_truth():
    from dataclasses import replace

    config, initial = world()
    config = replace(config, sensor_radius=40)
    data = MapInput(initial.truth, initial.component, initial.valid_shape, jnp.array([8, 8]))
    state = reset(data, config, jax.random.PRNGKey(3), jnp.array([[8, 8], [40, 32]]))
    state = state._replace(
        peer_positions=state.peer_positions.at[0, 1].set(jnp.array([40, 32])),
        peer_position_event=state.peer_position_event.at[0, 1].set(0),
        peer_maps=state.peer_maps.at[0, 1, 1, 1].set(0),
        event=jnp.int32(100),
    )
    altered = state._replace(
        truth=state.truth.at[25:35, 35:45].set(1),
        positions=state.positions.at[1].set(jnp.array([2, 2])),
        local_maps=state.local_maps.at[1].set(jnp.full_like(state.truth, 2)),
        plans=state.plans.at[1].set(0),
        goals=state.goals.at[1].set(0),
        balances=state.balances.at[1].set(0),
    )
    compiled = jax.jit(lambda value: observe(value, jnp.int32(0), config))
    expected, actual = compiled(state), compiled(altered)
    assert np.any(np.asarray(expected.graph.node_inputs[:, 3]) > 0)
    for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(before, after)
    # A recent received contact suppresses rendezvous without changing the map cache.
    recent = state._replace(inbox=state.inbox._replace(
        received_event=state.inbox.received_event.at[0, 1].set(state.event)))
    np.testing.assert_array_equal(compiled(recent).graph.node_inputs[:, 3], 0)


def test_original_positive_map_difference_includes_encoding_order():
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.modules.setdefault("TRAINING", False)
    env = importlib.import_module("env").Env.__new__(importlib.import_module("env").Env)
    own = np.repeat(np.arange(3, dtype=np.int32), 3).reshape(3, 3)
    peer = np.tile(np.arange(3, dtype=np.int32), 3).reshape(3, 3)
    codes = np.array([255, 1, 127], np.int16)
    expected = env.check_area_discovered_count(codes[own], codes[peer])
    actual = jax.jit(lambda a, b: jnp.sum(jnp.array([255, 1, 127], jnp.int16)[a] >
                                        jnp.array([255, 1, 127], jnp.int16)[b]))(own, peer)
    assert int(actual) == expected == 3
