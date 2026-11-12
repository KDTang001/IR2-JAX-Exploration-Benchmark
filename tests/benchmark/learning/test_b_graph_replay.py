"""Graph replay stays numeric and restores the exact imagined observation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.reset import reset
from ir2_benchmark.models.encoders import ActorFeatures, actor_features
from ir2_benchmark.navigation.ir2_graph import GraphMemory, GraphObservation
from ir2_benchmark.training.b_state import _cognition, _frame, _named_fields, _restore_tensors
from ir2_benchmark.training.imagination import cognitive_from_environment, imagined_observe
from ir2_benchmark.training.replay import ReplayBuffer, ReplayError
from ir2_benchmark.types import MapInput


@pytest.mark.parametrize("mode", ["ir2_graph", "ir2_one_hop"])
def test_graph_and_legacy_replay_roundtrip(tmp_path: Path, mode: str) -> None:
    config = HostConfig(
        navigation_mode=mode,
        n_agents=2,
        sensor_radius=8,
        graph_nodes_per_axis=12,
        graph_padding_size=100,
        graph_capacity=64,
        graph_local_radius=16,
        graph_global_radius=40,
        graph_peer_radius=8,
        graph_merge_enabled=False,
        max_candidates=9,
    )
    grid = jnp.zeros((40, 48), jnp.uint8).at[0].set(1).at[-1].set(1)
    grid = grid.at[:, 0].set(1).at[:, -1].set(1)
    data = MapInput(grid, grid == 0, jnp.array(grid.shape), jnp.array([14, 14]))
    state = reset(data, config, jax.random.PRNGKey(2), jnp.array([[14, 14], [20, 14]]))
    if state.peer_maps is not None:
        state = state._replace(
            peer_maps=state.peer_maps.at[0, 1, 4, 5].set(0).at[1, 0, 6, 7].set(1)
        )
    cognition = cognitive_from_environment(state)
    observation = imagined_observe(cognition, config)
    features = actor_features(observation, config.n_agents)
    frame = _frame(state, _named_fields("actor", features))
    assert all(value.dtype != object for _, value in frame.fields)
    assert "graph_memory_observable" not in dict(frame.fields)
    assert "peer_maps" not in dict(frame.fields)
    if cognition.graph_memory is not None:
        assert frame.field("graph_memory_observable_indices").size == np.count_nonzero(
            cognition.graph_memory.observable
        )
        unknown = 2
        assert frame.field("peer_maps_indices").size == np.count_nonzero(
            cognition.peer_maps != unknown
        )
    replay = ReplayBuffer(2, snapshot_interval=1)
    replay.append(
        episode_id=0,
        before=frame,
        after=frame,
        action=np.zeros(2),
        agent_id=0,
        reward=0.0,
        terminated=False,
        truncated=True,
    )
    path = tmp_path / "replay.zip"
    replay.save(path)
    recovered = ReplayBuffer.load(path).reconstruct(0, 0)
    restored = _cognition(recovered)
    assert jax.tree.structure(restored) == jax.tree.structure(cognition)
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(cognition), strict=True):
        np.testing.assert_array_equal(actual, expected)
    actual_observation = imagined_observe(restored, config)
    for actual, expected in zip(
        jax.tree.leaves(actual_observation), jax.tree.leaves(observation), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    stacked = _restore_tensors(ActorFeatures, (recovered.fields, recovered.fields), "actor")
    expected_features = jax.tree.map(lambda value: jnp.stack((value, value)), features)
    assert jax.tree.structure(stacked) == jax.tree.structure(expected_features)
    for actual, expected in zip(
        jax.tree.leaves(stacked), jax.tree.leaves(expected_features), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    if mode == "ir2_graph":
        assert isinstance(restored.graph_memory, GraphMemory)
        assert isinstance(stacked.graph, GraphObservation)
        legacy_fields = _named_fields("actor", features._replace(graph=None))
        with pytest.raises(ReplayError, match="mixed_graph_and_legacy_actor_features"):
            _restore_tensors(ActorFeatures, (recovered.fields, legacy_fields), "actor")
    else:
        assert restored.graph_memory is None
        assert restored.peer_maps is None
        assert stacked.graph is None
