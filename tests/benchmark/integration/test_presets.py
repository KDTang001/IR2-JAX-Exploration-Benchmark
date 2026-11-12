"""Preset contracts through real events, compiled batching and source comparisons."""

import ast
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ir2_benchmark.codecs.packet import Codec, Packet, decode_header
from ir2_benchmark.communication.propagation import LinkContext, ir2
from ir2_benchmark.config import ConfigError, HostConfig, load_config
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.rewards import RewardContext, RewardResult, ir2_individual
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.presets import (
    preset_manifest,
    register_preset,
    resolve_preset,
    verify_preset_manifest,
)
from ir2_benchmark.types import MapInput, make_action


def corridor(config):
    truth = jnp.ones((5, 20), jnp.uint8).at[2, 1:19].set(0)
    data = MapInput(truth, truth == 0, jnp.array([5, 20]), jnp.array([1, 2]))
    return reset(data, config, jax.random.PRNGKey(7), jnp.array([[1, 2], [2, 2]]))


def test_default_reward_and_wire_remain_compatible():
    config = HostConfig(sensor_radius=1, max_distance=1, coverage_target=1)
    state = corridor(config)
    result = jax.jit(lambda s: step_agent(s, make_action(jnp.array([2, 2]), True), config))(state)
    expected = 100 * (result.info.coverage_after - result.info.coverage_before) - 1 / (2 * 512)
    assert result.reward == expected
    assert int(result.info.spent) == 45
    assert bool(result.info.delivered[1])
    assert (
        int(
            decode_header(
                Packet(result.info.packet_buffer, result.info.packet_length)
            ).recipient_mask
        )
        == 2
    )
    assert result.info.reward_terms.shape == (6,)


@pytest.mark.parametrize(
    "charge,drop,expected", [("attempt", 1.0, 45), ("success", 1.0, 0), ("success", 0.0, 45)]
)
def test_success_charge_respects_faults_and_budget(charge, drop, expected):
    config = HostConfig(
        sensor_radius=1,
        coverage_target=1,
        signal_model="ideal",
        charge_mode=charge,
        fault_mode=1,
        dropout_probability=drop,
    )
    state = corridor(config)
    call = jax.jit(lambda s: step_agent(s, make_action(s.positions[0], True), config))
    result = call(state)
    assert int(result.info.spent) == expected
    assert bool(result.info.physical_links[1])
    assert bool(result.info.external_links[1]) == (drop == 0)
    poor = call(state._replace(balances=state.balances.at[0].set(44)))
    assert int(poor.info.spent) == 0 and not bool(jnp.any(poor.info.delivered))


def test_addressed_map_only_updates_actual_recipient_and_not_coverage():
    config = HostConfig(
        n_agents=3,
        sensor_radius=1,
        coverage_target=1,
        exchange_mode="addressed",
        signal_model="ideal",
    )
    truth = jnp.ones((5, 20), jnp.uint8).at[2, 1:19].set(0)
    state = reset(
        MapInput(truth, truth == 0, jnp.array([5, 20]), jnp.array([1, 2])),
        config,
        jax.random.PRNGKey(0),
        jnp.array([[1, 2], [7, 2], [12, 2]]),
    )
    action = make_action(
        state.positions[0], True, Codec.MAP_DELTA, recipients=jnp.array([False, True, False])
    )
    call = jax.jit(lambda s, a: step_agent(s, a, config))
    result = call(state, action)
    np.testing.assert_array_equal(result.info.delivered, [False, True, False])
    assert observe(result.state, jnp.int32(1), config).local_map[2, 1] == 0
    assert observe(result.state, jnp.int32(2), config).local_map[2, 1] == 2
    np.testing.assert_array_equal(result.state.covered, state.covered)
    malformed = call(state, action._replace(recipients=jnp.array([True, True, False])))
    assert int(malformed.info.spent) == 0 and not bool(jnp.any(malformed.info.delivered))


def test_sending_presets_and_codec_override():
    base = HostConfig(sensor_radius=1, coverage_target=1, signal_model="ideal", message_codec=1)
    for mode, round_index, expected in [
        ("none", 0, 0),
        ("always", 1, 20),
        ("periodic", 0, 20),
        ("periodic", 1, 0),
    ]:
        config = replace(base, send_policy=mode, send_period=2)
        state = corridor(config)._replace(round=jnp.int32(round_index))
        result = step_agent(state, make_action(state.positions[0], send=mode == "none"), config)
        assert int(result.info.spent) == expected


def test_reward_presets_success_once_timeout_and_batching():
    config = HostConfig(
        sensor_radius=1, coverage_target=0.9, reward_model="sparse_success", reward_success_bonus=7
    )
    state = corridor(config)
    state = state._replace(covered=state.component)
    action = make_action(state.positions[0])
    call = jax.jit(lambda s: step_agent(s, action, config))
    result = call(state)
    assert float(result.reward) == 7
    frozen = call(result.state)
    assert float(frozen.reward) == 0 and not bool(jnp.any(frozen.info.physical_links))
    batch = jax.jit(jax.vmap(lambda s: step_agent(s, action, config)))(
        jax.tree.map(lambda x: jnp.stack([x, x]), state)
    )
    np.testing.assert_array_equal(batch.reward, [7, 7])
    timeout = replace(config, max_rounds=1)
    state = corridor(timeout)._replace(event=jnp.int32(1), current_agent=jnp.int32(1))
    result = step_agent(state, make_action(state.positions[1]), timeout)
    assert bool(result.truncated) and float(result.reward) == 0
    distance = replace(
        config,
        reward_model="coverage_distance",
        reward_coverage_weight=0.0,
        reward_distance_weight=2.0,
    )
    state = corridor(distance)
    result = step_agent(state, make_action(jnp.array([2, 2])), distance)
    assert float(result.reward) == -2 * float(result.info.distance)


def test_physical_models_wall_range_and_strength_validity():
    grid = jnp.zeros((9, 15), jnp.uint8).at[:, 7].set(1)
    ctx = LinkContext(
        grid, jnp.array([2, 4]), jnp.array([12, 4]), jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    )
    config = HostConfig(communication_range=10)
    assert bool(resolve_preset("signal", "ideal")(ctx, config).reachable)
    assert bool(resolve_preset("signal", "range")(ctx, config).reachable)
    assert not bool(
        resolve_preset("signal", "range")(ctx, replace(config, communication_range=9)).reachable
    )
    assert not bool(resolve_preset("signal", "los_range")(ctx, config).reachable)
    assert not bool(resolve_preset("signal", "range")(ctx, config).strength_valid)
    assert bool(resolve_preset("signal", "attenuation")(ctx, config).strength_valid)


def test_ir2_signal_matches_original_all_octants_and_episode_noise():
    from ss_realistic_model import SS_realistic_model

    rng = np.random.default_rng(42)
    grid = (rng.random((33, 41)) < 0.2).astype(np.uint8)
    starts = rng.integers([0, 0], [41, 33], (128, 2))
    ends = rng.integers([0, 0], [41, 33], (128, 2))
    starts[0], ends[0] = [1, 1], [1, 1]
    config = HostConfig(
        signal_model="ir2",
        reference_distance=3,
        signal_noise_free_min=2,
        signal_noise_free_max=2,
        signal_noise_obstacle_min=3,
        signal_noise_obstacle_max=3,
    )
    reference = SS_realistic_model(dist_o=3, X_g_min=2, X_g_max=2, K_min=3, K_max=3)
    truth = np.where(grid == 0, 255, 1)
    expected = [reference.is_within_signal_strength(truth, a, b) for a, b in zip(starts, ends)]
    key = jax.random.PRNGKey(8)
    call = jax.jit(
        jax.vmap(lambda a, b: ir2(LinkContext(jnp.asarray(grid), a, b, key, key), config))
    )
    actual = call(jnp.asarray(starts), jnp.asarray(ends))
    np.testing.assert_array_equal(actual.reachable, expected)
    # Sweep thresholds around each independent reference strength, not only one binary threshold.
    from skimage.draw import line

    strengths = []
    for a, b in zip(starts, ends):
        x, y = line(*a, *b)
        obstacles = np.sum(truth[y[1:], x[1:]] != 255)
        free = np.linalg.norm(a - b) - obstacles
        loss = 31 + (40 * np.log10(obstacles) + 3 if obstacles > 0 else 0)
        loss += 20 * np.log10(free / 3) + 2 if free >= 3 else 0
        strengths.append(-20 - loss)
    np.testing.assert_allclose(actual.strength, strengths, atol=2e-5)
    noisy = replace(config, signal_noise_free_max=5, signal_noise_obstacle_max=6)
    ctx = LinkContext(jnp.asarray(grid), jnp.array([1, 1]), jnp.array([39, 31]), key, key)
    assert ir2(ctx, noisy).strength == ir2(ctx._replace(key=jax.random.PRNGKey(99)), noisy).strength


def test_ir2_individual_formula_from_original_source():
    tree = ast.parse((Path(__file__).resolve().parents[3] / "env.py").read_text(encoding="utf-8"))
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "individual_reward" for t in n.targets)
    )
    expression = compile(ast.Expression(assignment.value), "original_ir2_reward", "eval")
    for count, utility, meeting, distance in [
        (0, 0, 0.0, 0.0),
        (12, 70, 0.4, 35.0),
        (3, 5, 0.125, 100.0),
    ]:
        expected = eval(
            expression,
            {
                "all_num_new_frontiers": [count],
                "robot_id": 0,
                "new_pose_explore_util": utility,
                "new_pose_rendezvous_util": meeting,
                "dist_travelled": distance,
            },
        )
        ctx = RewardContext(0.0, 0.0, distance, 0, False, count, utility, meeting)
        assert float(ir2_individual(ctx, HostConfig()).reward) == pytest.approx(expected)


def custom_reward(context, config):
    value = -dict(config.custom_parameters)["byte_cost"] * context.spent
    return RewardResult(jnp.float32(value), jnp.zeros(6, jnp.float32).at[0].set(value))


def test_custom_registration_json_identity_and_real_compiled_step(tmp_path):
    import json

    register_preset("reward", "test_bytes", custom_reward, version="1")
    config = HostConfig(
        sensor_radius=1, reward_model="test_bytes", custom_parameters=(("byte_cost", 0.5),)
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    restored = load_config(path)
    assert config == restored and hash(config) == hash(restored)
    state = corridor(config)
    result = jax.jit(lambda s: step_agent(s, make_action(s.positions[0], True), config))(state)
    assert float(result.reward) == -22.5
    manifest = preset_manifest(config)
    assert manifest["reward"]["version"] == "1"
    with pytest.raises(ValueError, match="mismatch"):
        verify_preset_manifest(config, {})
    with pytest.raises(ValueError, match="already registered"):
        register_preset("reward", "test_bytes", custom_reward, version="2")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal_model": "missing"},
        {"send_period": 0},
        {"communication_range": -1},
        {"charge_mode": "free"},
        {"reward_model": "ir2_individual"},
    ],
)
def test_invalid_config_fails_before_compiling(kwargs):
    with pytest.raises(ConfigError):
        HostConfig(**kwargs)


def test_a_collector_records_effective_fixed_messages():
    from ir2_benchmark.agents.actor import Actor
    from ir2_benchmark.agents.critic import Critic
    from ir2_benchmark.models.plan_predictor import PlanPredictor
    from ir2_benchmark.training.collect_a import Parameters, make_collector
    from ir2_benchmark.training.state_a import TrainAConfig, initialize_training

    config = HostConfig(
        sensor_radius=1,
        plan_horizon=2,
        max_candidates=3,
        coverage_target=1,
        send_policy="periodic",
        send_period=2,
        message_codec=1,
    )
    truth = jnp.ones((7, 20), jnp.uint8).at[1:6, 1:19].set(0)
    data = MapInput(truth, truth == 0, jnp.array([7, 20]), jnp.array([2, 2]))
    settings = TrainAConfig(
        events=4,
        warmup_events=0,
        rollout_length=4,
        sequence_length=2,
        predictor_replay_capacity=8,
        batch_size=1,
        physical_env_count=1,
        hidden=8,
        communication_mode="fixed",
        suite="micro",
    )
    state = initialize_training(data, config, settings, jax.random.PRNGKey(5))
    with pytest.raises(ValueError, match="disabled"):
        make_collector(config, Actor(8), Critic(8), PlanPredictor(2, 8))
    collect = make_collector(
        config,
        Actor(8),
        Critic(8),
        PlanPredictor(2, 8),
        length=4,
        learned_communication=False,
        warmup=True,
    )
    result = collect(
        state.sampler,
        Parameters(state.actor.params, state.critic.params, state.predictor.params),
        jnp.array([4]),
        jnp.int32(0),
    )
    np.testing.assert_array_equal(result.executed_actions.send[0], [True, True, False, False])
    np.testing.assert_array_equal(result.executed_actions.codec[0], [1, 1, 1, 1])
    np.testing.assert_array_equal(result.trace.spent[0], [20, 20, 0, 0])
