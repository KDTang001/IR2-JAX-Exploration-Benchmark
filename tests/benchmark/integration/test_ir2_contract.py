"""Strict graph navigation, paid map knowledge, and actual optimizer integration."""

# ruff: noqa: FBT001, PLR2004

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from PIL import Image

from ir2_benchmark.codecs.standard import encode_map_delta, encode_plan
from ir2_benchmark.communication.channel import broadcast
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.ir2_observation import ir2_observation
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.models.world_model import WorldPrediction, initial_rssm
from ir2_benchmark.training.checkpoint import load_checkpoint
from ir2_benchmark.training.imagination import (
    ImaginedState,
    apply_prediction,
    cognitive_from_environment,
    imagined_observe,
)
from ir2_benchmark.training.train_a import (
    TrainAConfig,
    configuration_hash,
    initialize_training,
    train_a,
)
from ir2_benchmark.types import EnvState, LocalFields, MapInput, make_action, validate_graph_state
from ir2_benchmark.visualization.recording import RecordingConfig
from ir2_benchmark.visualization.replay import replay_step


def fixture() -> tuple[MapInput, HostConfig, EnvState]:
    """Provide a small graph with room for both sensing and movement."""
    config = HostConfig(
        navigation_mode="ir2_graph",
        n_agents=2,
        sensor_radius=8,
        max_rounds=8,
        graph_nodes_per_axis=12,
        graph_padding_size=100,
        graph_capacity=64,
        graph_local_radius=16,
        graph_global_radius=40,
        graph_peer_radius=8,
        graph_merge_enabled=False,
        initial_budget=4096,
        signal_threshold=-200,
        coverage_target=1,
        max_candidates=9,
    )
    grid = jnp.zeros((40, 48), jnp.uint8).at[0].set(1).at[-1].set(1)
    grid = grid.at[:, 0].set(1).at[:, -1].set(1)
    data = MapInput(grid, grid == 0, jnp.array(grid.shape), jnp.array([14, 14]))
    state = reset(data, config, jax.random.PRNGKey(2), jnp.array([[14, 14], [20, 14]]))
    return data, config, state


def test_real_and_imagined_edges_share_the_full_graph_contract() -> None:
    _, config, state = fixture()
    obs = observe(state, jnp.int32(0), config)
    assert obs.graph is not None
    assert obs.tree is None
    assert obs.graph.node_inputs.shape == (100, 6)
    assert obs.graph.edge_mask.shape == (100, 100)
    assert float(obs.graph.node_inputs[obs.graph.current_index, 5]) == -1
    index = jnp.argmax(jnp.where(obs.candidates.mask, obs.candidates.distances, -1))
    goal = obs.candidates.positions[index]
    assert bool(obs.candidates.mask[index])
    real = jax.jit(lambda s: step_agent(s, make_action(goal), config))(state)
    validate_graph_state(real.state)
    np.testing.assert_array_equal(real.state.positions[0], goal)
    np.testing.assert_allclose(
        real.info.distance, np.linalg.norm(np.asarray(goal - state.positions[0]))
    )
    assert not bool(real.info.invalid_action)
    assert int(real.state.graph_memory.history_count[0]) == 2
    replayed, _, path, distance = replay_step(config)(state, goal, jnp.bool_(0), jnp.int32(0))
    np.testing.assert_array_equal(replayed.positions, real.state.positions)
    np.testing.assert_array_equal(path, jnp.stack((obs.position, goal)))
    np.testing.assert_allclose(distance, real.info.distance)

    imagined = ImaginedState(
        cognitive_from_environment(state),
        initial_rssm(deterministic=8, categories=2, classes=3),
        jnp.zeros((2, 8)),
    )
    imagined_obs = imagined_observe(imagined.cognition, config)
    for left, right in zip(obs.graph, imagined_obs.graph, strict=True):
        np.testing.assert_array_equal(left, right)
    logits = jnp.zeros((2, 3))
    prediction = WorldPrediction(
        imagined.latent,
        logits,
        logits,
        jnp.zeros((17, 17, 3)).at[..., 2].set(100),
        jnp.array([100.0, -100.0, -100.0]),
        jnp.full(2, -100.0),
        jnp.float32(0),
        jnp.float32(100),
    )
    result = jax.jit(
        lambda s: apply_prediction(s, make_action(goal), prediction, config, jax.random.PRNGKey(8))
    )(imagined)
    np.testing.assert_array_equal(result.state.cognition.positions[0], goal)
    np.testing.assert_allclose(result.distance, real.info.distance)
    assert int(result.state.cognition.graph_memory.history_count[0]) == 2


def test_ir2_individual_reward_uses_decision_time_target_utility():
    from dataclasses import replace

    from ir2_benchmark.navigation.ir2_graph import find_frontier

    _, config, state = fixture()
    config = replace(config, reward_model="ir2_individual")
    obs = observe(state, jnp.int32(0), config)
    index = jnp.argmax(jnp.where(obs.candidates.mask, obs.candidates.distances, -1))
    goal = obs.candidates.positions[index]
    node = obs.graph.edge_inputs[index]
    result = jax.jit(lambda s: step_agent(s, make_action(goal), config))(state)
    _, old = find_frontier(jnp.min(state.local_maps, axis=0), state.valid_shape)
    _, new = find_frontier(jnp.min(result.state.local_maps, axis=0), state.valid_shape)
    expected = (
        jnp.sum(old & ~new) / 25
        + obs.graph_memory.utility[node] / 50
        + obs.graph.node_inputs[node, 3]
        - result.info.distance / 512
    )
    np.testing.assert_allclose(result.reward, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("fusion", "map_message", "delivered", "affordable"),
    [
        (False, True, True, True),
        (True, True, True, True),
        (True, False, True, True),
        (False, False, True, True),
        (True, True, False, True),
        (True, True, True, False),
    ],
)
def test_only_paid_delivered_map_adds_peer_map_and_global_nodes(
    fusion: bool,
    map_message: bool,
    delivered: bool,
    affordable: bool,
) -> None:
    _, config, state = fixture()
    config = replace(config, global_graph_fusion=fusion)
    coordinates = jnp.zeros((47, 2), jnp.int32).at[0].set(jnp.array([4, 4]))
    packet = (
        encode_map_delta(
            coordinates,
            jnp.zeros(47, jnp.uint8),
            jnp.int32(1),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            2,
        )
        if map_message
        else encode_plan(
            state.positions[0],
            state.positions[0],
            jnp.broadcast_to(state.positions[0], (config.plan_horizon, 2)),
            jnp.int32(1),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            2,
        )
    )
    channel = broadcast(
        state.inbox,
        packet,
        jnp.int32(0),
        jnp.int32(4096 if affordable else 0),
        jnp.bool_(True),  # noqa: FBT003
        jnp.ones(2, bool),
        jnp.full(2, delivered),
        state.active_mask,
        config.plan_horizon,
    )
    received = state._replace(inbox=channel.inbox)
    obs = observe(received, jnp.int32(1), config)
    has_map = map_message and delivered and affordable
    assert (int(obs.peer_maps[0, 4, 4]) == 0) == has_map
    imported = jnp.any(
        obs.graph_memory.global_valid
        & jnp.all(obs.graph_memory.global_nodes == jnp.array([4, 4]), axis=-1)
    )
    assert bool(imported) == (fusion and has_map)
    assert int(obs.graph_memory.history_count) == 1
    if has_map:
        assert int(channel.spent) == int(packet.length)
    if fusion and has_map:
        batch = jax.tree.map(lambda left, right: jnp.stack((left, right)), received, state)
        together = jax.jit(jax.vmap(lambda item: observe(item, jnp.int32(1), config)))(batch)
        separate = (obs, observe(state, jnp.int32(1), config))
        for lane, expected in enumerate(separate):
            actual = jax.tree.map(lambda value, lane=lane: value[lane], together)
            for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                if np.issubdtype(left.dtype, np.inexact):
                    np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
                else:
                    np.testing.assert_array_equal(left, right)


def test_plan_only_peer_has_no_map_surplus_utility() -> None:
    data, config, _ = fixture()
    config = replace(config, sensor_radius=40, graph_padding_size=160)
    state = reset(data, config, jax.random.PRNGKey(2), jnp.array([[8, 8], [40, 32]]))
    obs = observe(state, jnp.int32(0), config)
    fields = LocalFields(*(getattr(obs, name) for name in LocalFields._fields))
    fields = fields._replace(
        peer_positions=state.positions,
        peer_position_event=jnp.zeros(config.n_slots, jnp.int32),
        message_ages=jnp.full(config.n_slots, 100),
    )
    actual = jax.jit(lambda value: ir2_observation(value, config))(fields)
    np.testing.assert_array_equal(actual.graph.node_inputs[:, 3], 0)
    assert not bool(actual.graph_memory.path_failed)
    with_map = ir2_observation(
        fields._replace(peer_maps=fields.peer_maps.at[1, 1, 1].set(0)), config
    )
    assert bool(jnp.any(with_map.graph.node_inputs[:, 3] > 0))


def test_graph_training_updates_restores_and_records_real_actions(tmp_path: Path) -> None:
    data, config, _ = fixture()
    config = replace(config, max_rounds=2)
    settings = TrainAConfig(
        suite="micro",
        events=8,
        warmup_events=0,
        batch_size=1,
        physical_env_count=1,
        rollout_length=4,
        sequence_length=2,
        ppo_epochs=1,
        hidden=16,
        predictor_replay_capacity=8,
        communication_mode="fixed",
        fixed_plan=True,
        neighbor_estimator="zoh",
    )
    key = jax.random.PRNGKey(91)
    directory = tmp_path / "graph-training"
    recording = RecordingConfig(
        every_episodes=1, frame_stride_rounds=1, max_frames=3, duration_ms=100, max_side=128
    )
    first = train_a(
        data, config, settings, key, directory, stop_after_updates=1, recording=recording
    )
    assert first.actor_parameter_change > 0
    assert first.real_events == 4
    resumed = train_a(
        data, config, settings, key, directory, resume=first.checkpoint, recording=recording
    )
    assert resumed.real_events == 8
    template = initialize_training(data, config, settings, key)
    restored = serialization.from_bytes(
        template, load_checkpoint(resumed.checkpoint, configuration_hash(config, settings))
    )
    validate_graph_state(restored.sampler.environment)
    assert int(restored.actor.step) == 2
    records = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    assert all(row["ppo"]["mean_actor_grad_norm"] > 0 for row in records)
    clips = list((directory / "gifs").glob("*_merged.gif"))
    assert clips
    for path in clips:
        with Image.open(path) as clip:
            assert clip.n_frames >= 2
            for frame in range(clip.n_frames):
                clip.seek(frame)
                clip.load()
