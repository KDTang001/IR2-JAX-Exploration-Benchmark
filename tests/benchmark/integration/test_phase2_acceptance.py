"""Independent wire/ledger/knowledge oracles for separate communication conditions."""

# The wire oracle keeps literal protocol constants and explicit device arguments.
# ruff: noqa: PLR2004, FBT003, PLC0415

import json
import os
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from ir2_benchmark.codecs.packet import Codec
from ir2_benchmark.codecs.standard import encode_map_delta
from ir2_benchmark.communication.channel import broadcast
from ir2_benchmark.config import ConfigError, HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.types import EnvState, MapInput, make_action


def corridor(config: HostConfig) -> EnvState:
    truth = np.ones((7, 64), np.uint8)
    truth[3, 1:63] = 0
    starts = np.array([[5, 3], [14, 3], [25, 3], [45, 3]], np.int32)[: config.n_agents]
    data = MapInput(
        jnp.asarray(truth), jnp.asarray(truth == 0), jnp.array(truth.shape), jnp.array(starts[0])
    )
    return reset(data, config, jax.random.PRNGKey(17), jnp.asarray(starts))


def scan_host(
    truth: NDArray[np.uint8],
    position: NDArray[np.int32],
    known: NDArray[np.uint8],
    times: NDArray[np.int32],
    event: int,
) -> None:
    x, y = position
    for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        known[y + dy, x + dx] = truth[y + dy, x + dx]
        times[y + dy, x + dx] = event


def folded_draw(root: jax.Array, episode: int, event: int, sender: int, receiver: int) -> float:
    # Explicit public counter recipe; never calls fault_links/fold_key.
    key = root
    for value in [2, episode, event, sender, receiver]:
        key = jax.random.fold_in(key, value)
    return float(jax.random.uniform(key))


@pytest.mark.parametrize("mode", ["normal", "dropout", "burst"])
@pytest.mark.parametrize("charge", ["attempt", "success"])
def test_separate_conditions_recompute_each_event(mode: str, charge: str) -> None:  # noqa: C901, PLR0912, PLR0915 - one event-by-event independent oracle
    config = HostConfig(
        navigation_mode="grid_path",
        n_agents=4,
        agent_bucket=6,
        sensor_radius=1,
        max_distance=1,
        max_rounds=8,
        coverage_target=1,
        initial_budget=8192,
        signal_model="ideal",
        charge_mode=charge,
        fault_mode={"normal": 0, "dropout": 1, "burst": 2}[mode],
        dropout_probability=0.3 if mode == "dropout" else 0,
        burst_start=2,
        burst_duration=2,
    )
    state = corridor(config)
    truth = np.asarray(state.truth)
    maps = np.full((4, *truth.shape), 2, np.uint8)
    times = np.full(maps.shape, -1, np.int32)
    physical = np.zeros(maps.shape, bool)
    for i, pos in enumerate(np.asarray(state.positions[:4])):
        scan_host(truth, pos, maps[i], times[i], 0)
        physical[i] = maps[i] == 0
    positions = np.zeros((4, 6, 2), np.int32)
    goals = positions.copy()
    plans = np.zeros((4, 6, 5, 2), np.int32)
    stamps = np.full((3, 4, 6), -1, np.int32)
    pending = [[None] * 4 for _ in range(4)]
    balances = [8192] * 4
    cursors = [0] * 4
    sequences = [0] * 4
    records = []
    step = jax.jit(lambda s, a: step_agent(s, a, config))
    read = jax.jit(lambda s: observe(s, s.current_agent, config))
    for event in range(32):
        sender, round_index = event % 4, event // 4
        # Merge only previously delivered bytes, independently of production decoders.
        for raw in pending[sender]:
            if raw is None:
                continue
            h = struct.unpack("<BBBBIIHBB", raw[:16])
            codec, author, generated = h[1], h[2], h[5]
            payload = raw[16:]
            if codec in (1, 3) and generated > stamps[0, sender, author]:
                positions[sender, author] = struct.unpack("<HH", payload[:4])
                stamps[0, sender, author] = generated
            if codec in (2, 3) and generated > stamps[1, sender, author]:
                goals[sender, author] = struct.unpack(
                    "<HH", payload[4:8] if codec == 3 else payload[:4]
                )
                stamps[1, sender, author] = generated
            if codec == 3 and generated > stamps[2, sender, author]:
                plans[sender, author] = np.array(struct.unpack("<10H", payload[9:29])).reshape(5, 2)
                stamps[2, sender, author] = generated
            if codec == 4:
                for k in range(payload[0]):
                    x, y, value = struct.unpack_from("<HHB", payload, 1 + 5 * k)
                    if generated >= times[sender, y, x]:
                        maps[sender, y, x], times[sender, y, x] = value, generated
        pending[sender] = [None] * 4
        observation = read(state)
        np.testing.assert_array_equal(observation.local_map, maps[sender])
        np.testing.assert_array_equal(observation.peer_positions, positions[sender])
        np.testing.assert_array_equal(observation.peer_goals, goals[sender])
        np.testing.assert_array_equal(observation.peer_plans, plans[sender])
        before = np.asarray(state.positions[sender]).copy()
        target = before + np.array([1 if round_index % 2 == 0 else -1, 0])
        codec = round_index % 4 + 1
        result = step(state, make_action(target, send=True, codec=codec))
        raw = bytes(np.asarray(result.info.packet_buffer)[: int(result.info.packet_length)])
        h = struct.unpack("<BBBBIIHBB", raw[:16])
        assert h == (
            1,
            codec,
            sender,
            0,
            sequences[sender],
            event,
            len(raw) - 16,
            15 ^ (1 << sender),
            0,
        )
        if codec in (1, 2):
            assert raw[16:] == struct.pack("<HH", *(before if codec == 1 else target))
            assert len(raw) == 20
        elif codec == 3:
            expected = struct.pack("<HHHHB10H", *before, *target, 1, *np.tile(target, 5))
            assert raw[16:] == expected
            assert len(raw) == 45
        else:
            candidates = [(x, y) for y, x in np.argwhere((maps[sender] < 2) & (times[sender] >= 0))]
            candidates.sort(key=lambda p: (-times[sender, p[1], p[0]], *p))
            count = min(len(candidates), 47)
            chosen = [candidates[(cursors[sender] + k) % len(candidates)] for k in range(count)]
            expected = bytes([count]) + b"".join(
                struct.pack("<HHB", x, y, maps[sender, y, x]) for x, y in chosen
            )
            assert raw[16:] == expected
            assert len(raw) == 17 + 5 * count
            cursors[sender] = (cursors[sender] + count) % len(candidates)
        sequences[sender] += 1
        allowed = np.ones(6, bool)
        if mode == "dropout":
            allowed = np.array(
                [folded_draw(state.key, 0, event, sender, r) >= 0.3 for r in range(6)]
            )
        elif mode == "burst" and 2 <= round_index < 4:
            isolated = np.asarray(state.isolated_mask)
            allowed = ~(isolated[sender] | isolated)
        delivered = allowed & np.array([r < 4 and r != sender for r in range(6)])
        expected_fee = len(raw) if charge == "attempt" or delivered.any() else 0
        balances[sender] -= expected_fee
        np.testing.assert_array_equal(result.info.delivered, delivered)
        np.testing.assert_array_equal(
            result.info.physical_links, [r < 4 and r != sender for r in range(6)]
        )
        assert int(result.info.spent) == expected_fee
        np.testing.assert_array_equal(result.state.balances[:4], balances)
        assert not bool(result.info.invalid_action)
        np.testing.assert_array_equal(result.state.positions[sender], target)
        scan_host(truth, target, maps[sender], times[sender], event)
        for dx in (-1, 0, 1):
            physical[sender, target[1], target[0] + dx] = True
        np.testing.assert_array_equal(result.state.local_maps[sender], maps[sender])
        np.testing.assert_array_equal(result.state.physical_seen[:4], physical)
        for receiver in np.flatnonzero(delivered[:4]):
            pending[receiver][sender] = raw
        assert not np.asarray(result.state.inbox.present[4:]).any()
        records.append(
            {
                "event": event,
                "round": round_index,
                "sender": sender,
                "wire_hex": raw.hex(),
                "spent": expected_fee,
                "delivered": delivered.tolist(),
                "balance": balances[sender],
            }
        )
        state = result.state
    assert bool(state.truncated)
    assert not bool(state.terminated)
    assert all(balance >= 0 for balance in balances)
    output = os.environ.get("IR2_PHASE2_EVIDENCE")
    if output:
        folder = Path(output)
        folder.mkdir(exist_ok=True)
        (folder / f"{mode}-{charge}.json").write_text(
            json.dumps(
                {
                    "condition": mode,
                    "charge_mode": charge,
                    "events": 32,
                    "isolated": np.asarray(state.isolated_mask).tolist(),
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def test_map_payload_outside_valid_shape_never_reveals_padding() -> None:
    config = HostConfig(
        n_agents=2,
        navigation_mode="ir2_graph",
        sensor_radius=1,
        graph_nodes_per_axis=6,
        graph_padding_size=80,
        graph_capacity=80,
    )
    truth = jnp.ones((12, 20), jnp.uint8).at[1:7, 1:11].set(0)
    data = MapInput(truth, truth == 0, jnp.array([8, 12]), jnp.array([3, 3]))
    state = reset(data, config, jax.random.PRNGKey(0), jnp.array([[3, 3], [8, 3]]))
    packet = encode_map_delta(jnp.array([[5, 3], [15, 3]]), jnp.array([0, 0]), 2, 1, 0, 1, 2)
    flags = jnp.ones(2, bool)
    received = broadcast(state.inbox, packet, 1, 100, True, flags, flags, flags)
    obs = jax.jit(lambda s: observe(s, jnp.int32(0), config))(state._replace(inbox=received.inbox))
    assert int(received.spent) == 27
    assert int(obs.local_map[3, 5]) == 0
    assert int(obs.local_map[3, 15]) == 2
    assert int(obs.peer_maps[1, 3, 15]) == 2
    # Replay images and imagined observations share the same receiving boundary.
    from ir2_benchmark.training.imagination import cognitive_from_environment, imagined_observe
    from ir2_benchmark.visualization.replay import frame_fields

    state = state._replace(inbox=received.inbox)
    imagined = jax.jit(lambda s: imagined_observe(s, config))(cognitive_from_environment(state))
    frames = frame_fields(config)(state)
    assert int(imagined.local_map[3, 15]) == 2
    assert int(frames[0][0, 3, 15]) == 2


@pytest.mark.parametrize("mode", [0, 2])
def test_non_dropout_config_rejects_mixed_probability(mode: int) -> None:
    with pytest.raises(ConfigError, match="separate condition"):
        HostConfig(fault_mode=mode, dropout_probability=0.3)


@pytest.mark.parametrize("signal_model", ["ideal", "range", "los_range", "attenuation", "ir2"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_physical_model_and_external_fault_are_separate(signal_model: str, mode: int) -> None:
    config = HostConfig(
        n_agents=4,
        navigation_mode="grid_path",
        sensor_radius=1,
        max_distance=1,
        coverage_target=1,
        signal_model=signal_model,
        communication_range=20,
        reference_distance=3,
        signal_threshold=-64,
        fault_mode=mode,
        dropout_probability=0.3 if mode == 1 else 0,
        burst_start=0,
        burst_duration=1,
    )
    state = corridor(config)
    result = jax.jit(
        lambda s: step_agent(s, make_action(s.positions[0], True, Codec.POSITION), config)
    )(state)
    distances = np.linalg.norm(np.asarray(state.positions) - np.asarray(state.positions[0]), axis=1)
    if signal_model == "ideal":
        base = np.ones(4, bool)
    elif signal_model in ("range", "los_range"):
        base = distances <= 20
    else:
        strengths = -51 - 20 * np.log10(np.maximum(distances, 3) / 3)
        base = strengths > -64
        np.testing.assert_allclose(result.info.signal_strength, strengths, atol=2e-5)
    base[0] = False
    if mode == 0:
        external = np.ones(4, bool)
    elif mode == 1:
        external = np.array([folded_draw(state.key, 0, 0, 0, r) >= 0.3 for r in range(4)])
    else:
        isolated = np.asarray(state.isolated_mask)
        external = ~(isolated[0] | isolated)
    external[0] = False
    np.testing.assert_array_equal(result.info.physical_links, base)
    np.testing.assert_array_equal(result.info.external_links, external)
    np.testing.assert_array_equal(result.info.delivered, base & external)
    assert int(result.info.spent) == 20


@pytest.mark.parametrize("probability", [0.1, 0.3, 0.5])
def test_dropout_rate_and_receiver_joint_frequency(probability: float) -> None:
    from ir2_benchmark.communication.faults import fault_links

    call = jax.jit(
        jax.vmap(
            lambda event: fault_links(
                jax.random.PRNGKey(91), 1, 7, event, 0, 4, probability, jnp.zeros(4, bool), False
            )
        )
    )
    allowed = np.asarray(call(jnp.arange(4096)))
    loss = 1 - allowed.mean(axis=0)
    assert np.all(np.abs(loss - probability) < 6 * np.sqrt(probability * (1 - probability) / 4096))
    for a in range(4):
        for b in range(a):
            assert abs(np.mean(allowed[:, a] & allowed[:, b]) - (1 - probability) ** 2) < 0.04


def test_zero_budget_can_receive_move_and_sense() -> None:
    config = HostConfig(
        n_agents=2,
        navigation_mode="grid_path",
        sensor_radius=1,
        max_distance=1,
        signal_model="ideal",
        coverage_target=1,
    )
    state = corridor(config)
    state = state._replace(balances=state.balances.at[1].set(0))
    sent = step_agent(state, make_action(state.positions[0], True, Codec.POSITION), config)
    expected = observe(sent.state, jnp.int32(1), config)
    np.testing.assert_array_equal(expected.peer_positions[0], state.positions[0])
    goal = sent.state.positions[1] + jnp.array([1, 0])
    result = jax.jit(lambda s: step_agent(s, make_action(goal, True), config))(sent.state)
    assert int(result.info.spent) == 0
    assert not np.any(result.info.delivered)
    assert float(result.info.distance) == 1
    np.testing.assert_array_equal(result.state.positions[1], goal)
    assert int(result.state.local_maps[1, 3, int(goal[0]) + 1]) == 0


def test_custom_wire_and_explicit_forwarding_use_paid_event_boundary() -> None:
    from ir2_benchmark.codecs.packet import forward_packet

    config = HostConfig(
        n_agents=3,
        navigation_mode="grid_path",
        sensor_radius=1,
        max_distance=1,
        signal_model="ideal",
        coverage_target=1,
    )
    state = corridor(config)
    flags = jnp.array([True, True, False])
    # A direct path to robot 2 is blocked for this attempt; there is no automatic relay.
    from ir2_benchmark.codecs.standard import encode_plan

    original = encode_plan(
        state.positions[0], state.positions[0], jnp.tile(state.positions[0], (5, 1)), 1, 0, 0, 0, 3
    )
    received = broadcast(state.inbox, original, 0, 100, True, flags, flags, state.active_mask)
    assert int(received.spent) == 45
    assert not bool(received.inbox.present[2, 0])
    state = state._replace(inbox=received.inbox, event=jnp.int32(1), current_agent=jnp.int32(1))
    forwarded = forward_packet(original, 1, 0, 1, 3)
    action = make_action(
        state.positions[1], True, Codec.FORWARD, forwarded.buffer[16:], forwarded.length - 16
    )
    result = jax.jit(lambda s: step_agent(s, action, config))(state)
    assert int(result.info.spent) == 51
    obs = observe(result.state, jnp.int32(2), config)
    np.testing.assert_array_equal(obs.peer_positions[0], state.positions[0])
    assert int(obs.peer_plan_event[0]) == 0
    custom = make_action(
        result.state.positions[2], True, 128, jnp.arange(240, dtype=jnp.uint8), 240
    )
    sent = jax.jit(lambda s: step_agent(s, custom, config))(result.state)
    assert int(sent.info.spent) == 256
    view = observe(sent.state, jnp.int32(0), config)
    np.testing.assert_array_equal(view.inbox_buffers[2, 16:], np.arange(240, dtype=np.uint8))
    np.testing.assert_array_equal(view.local_map, sent.state.local_maps[0])


@pytest.mark.parametrize("n_agents", [2, 3, 4, 6])
def test_burst_selects_exact_active_population_once(n_agents: int) -> None:
    from ir2_benchmark.communication.faults import sample_isolated

    active = jnp.arange(6) < n_agents
    key = jax.random.PRNGKey(17)
    selected = jax.jit(sample_isolated)(key, 5, active)
    assert int(jnp.sum(selected)) == n_agents // 2
    assert not np.any(np.asarray(selected)[n_agents:])
    scenario_key = key
    for counter in [3, 5, 0, 0, 0]:
        scenario_key = jax.random.fold_in(scenario_key, counter)
    scores = np.asarray(jax.random.uniform(scenario_key, (6,)))
    expected = np.zeros(6, bool)
    expected[np.argsort(scores[:n_agents])[: n_agents // 2]] = True
    np.testing.assert_array_equal(selected, expected)


@pytest.mark.parametrize("defect", ["reserved", "sender", "length", "map_state"])
def test_malformed_wire_never_charges_or_changes_inbox(defect: str) -> None:
    from ir2_benchmark.communication.channel import empty_inbox

    packet = encode_map_delta(jnp.array([[3, 3]]), jnp.array([0]), 1, 0, 0, 0, 2)
    if defect == "length":
        packet = packet._replace(length=packet.length - 1)
    else:
        offset, value = {"reserved": (15, 1), "sender": (2, 1), "map_state": (21, 2)}[defect]
        packet = packet._replace(buffer=packet.buffer.at[offset].set(value))
    flags = jnp.ones(2, bool)
    initial = empty_inbox(2)
    result = jax.jit(broadcast)(initial, packet, 0, 100, True, flags, flags, flags)
    assert int(result.spent) == 0
    assert int(result.balance) == 100
    assert not np.any(result.delivered)
    for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(result.inbox), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("duration", [64, 128])
def test_formal_burst_round_boundaries_and_recovery(duration: int) -> None:
    config = HostConfig(
        n_agents=4,
        navigation_mode="grid_path",
        sensor_radius=1,
        signal_model="ideal",
        fault_mode=2,
        burst_start=128,
        burst_duration=duration,
        coverage_target=1,
    )
    initial = corridor(config)
    call = jax.jit(
        lambda state: step_agent(
            state, make_action(state.positions[state.current_agent], True, Codec.POSITION), config
        )
    )
    for round_index in [127, 128, 128 + duration - 1, 128 + duration]:
        for sender in [0, 2]:
            state = initial._replace(
                event=jnp.int32(4 * round_index + sender),
                round=jnp.int32(round_index),
                current_agent=jnp.int32(sender),
            )
            result = call(state)
            expected = np.arange(4) != sender
            if 128 <= round_index < 128 + duration:
                isolated = np.asarray(initial.isolated_mask)
                expected &= ~(isolated[sender] | isolated)
            np.testing.assert_array_equal(result.info.delivered, expected)
            assert int(result.info.spent) == 20
