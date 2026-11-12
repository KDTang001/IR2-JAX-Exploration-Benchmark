"""Independent NumPy/struct event oracle versus real eager, compiled and batch scans."""

import struct
from typing import NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ir2_benchmark.config import HostConfig  # noqa: E402
from ir2_benchmark.env.reset import reset  # noqa: E402
from ir2_benchmark.env.rollout import rollout, scripted_rollout  # noqa: E402
from ir2_benchmark.navigation import reference as navigation  # noqa: E402
from ir2_benchmark.sensing import reference as sensing  # noqa: E402
from ir2_benchmark.types import Action, LocalObservation, MapInput, make_action  # noqa: E402


class OracleTrace(NamedTuple):
    """Store independently computed post-event fields for every scripted action."""

    positions: np.ndarray
    local_maps: np.ndarray
    balances: np.ndarray
    inbox_buffers: np.ndarray
    inbox_present: np.ndarray
    physical_seen: np.ndarray
    covered: np.ndarray
    goals: np.ndarray
    plans: np.ndarray
    plan_lengths: np.ndarray
    sequences: np.ndarray
    spent_bytes: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


def reference_trace(
    truth: np.ndarray, starts: np.ndarray, goals: np.ndarray, send: np.ndarray, config: HostConfig
) -> OracleTrace:
    """Independent Normal-channel oracle: NumPy sensing/deque paths/struct packet bytes."""
    n = config.n_agents
    positions = starts.copy()
    maps = np.full((n, *truth.shape), 2, np.uint8)
    physical = np.zeros_like(maps, bool)
    for agent in range(n):
        sensed = sensing.sense(truth, maps[agent], positions[agent], config.sensor_radius)
        maps[agent], physical[agent] = sensed.local_map, sensed.observed_free
    covered = physical.any(axis=0)
    balances = np.full(n, config.initial_budget, np.int32)
    buffers = np.zeros((n, n, 256), np.uint8)
    present = np.zeros((n, n), bool)
    own_goals = starts.copy()
    plans = np.repeat(starts[:, None, :], config.plan_horizon, axis=1)
    lengths = np.zeros(n, np.int32)
    sequences, spent = np.zeros(n, np.int32), np.zeros(n, np.int32)
    terminated = covered.sum() / (truth == 0).sum() >= config.coverage_target
    truncated = False
    snapshots = []
    for event, (goal, requested) in enumerate(zip(goals, send, strict=True)):
        if not (terminated or truncated):
            agent = event % n
            source = positions[agent].copy()
            route = navigation.navigate(
                maps[agent], source, goal, config.max_distance, config.plan_horizon
            )
            present[agent] = False
            own_goals[agent], plans[agent], lengths[agent] = (
                route.goal,
                route.plan,
                route.plan_valid_length,
            )
            payload = struct.pack("<HHHHB", *source, *route.goal, route.plan_valid_length)
            payload += b"".join(struct.pack("<HH", *point) for point in route.plan)
            header = struct.pack(
                "<BBBBIIHBB",
                1,
                3,
                agent,
                0,
                sequences[agent],
                event,
                9 + 4 * config.plan_horizon,
                ((1 << n) - 1) ^ (1 << agent),
                0,
            )
            raw = header + payload
            positions[agent] = route.position
            sensed = sensing.sense(truth, maps[agent], route.position, config.sensor_radius)
            maps[agent] = sensed.local_map
            physical[agent] |= sensed.observed_free
            covered |= sensed.observed_free
            if requested and len(raw) <= balances[agent]:
                balances[agent] -= len(raw)
                spent[agent] += len(raw)
                sequences[agent] += 1
                for receiver in range(n):
                    if receiver != agent:
                        buffers[receiver, agent] = np.frombuffer(raw.ljust(256, b"\0"), np.uint8)
                        present[receiver, agent] = True
            terminated = covered.sum() / (truth == 0).sum() >= config.coverage_target
            truncated = event + 1 >= config.max_rounds * n and not terminated
        snapshots.append(
            tuple(
                value.copy()
                for value in (
                    positions,
                    maps,
                    balances,
                    buffers,
                    present,
                    physical,
                    covered,
                    own_goals,
                    plans,
                    lengths,
                    sequences,
                    spent,
                    np.asarray(terminated),
                    np.asarray(truncated),
                )
            )
        )
    return OracleTrace(*(np.stack(values) for values in zip(*snapshots, strict=True)))


@pytest.mark.parametrize("n_agents", [2, 4, 6])
def test_scripted_reference_eager_jit_and_batch_agree_each_event(n_agents: int) -> None:
    # Given: random obstacle fixtures, Normal links guaranteed by a low test threshold.
    config = HostConfig(
        n_agents=n_agents,
        sensor_radius=2,
        max_distance=2,
        max_rounds=3,
        initial_budget=91,
        coverage_target=1,
        signal_threshold=-1e6,
    )
    rng = np.random.default_rng(108 + n_agents)
    truth = np.ones((9, 11), np.uint8)
    truth[1:-1, 1:-1] = (rng.random((7, 9)) < 0.12).astype(np.uint8)  # noqa: PLR2004
    truth[1:3, 1:8] = 0
    starts = np.array([[1 + agent, 1] for agent in range(n_agents)], np.int32)
    component = navigation.bfs(truth, starts[0]).distances >= 0
    truth[(truth == 0) & ~component] = 1
    goals = np.stack(
        (rng.integers(0, 11, 5 * n_agents), rng.integers(0, 9, 5 * n_agents)), axis=1
    ).astype(np.int32)
    send = np.arange(5 * n_agents) % 3 != 2  # noqa: PLR2004
    actions = Action(
        jnp.asarray(goals),
        jnp.asarray(send),
        jnp.full(len(goals), 3, jnp.int32),
        jnp.zeros((len(goals), 240), jnp.uint8),
        jnp.zeros(len(goals), jnp.int32),
    )
    data = MapInput(jnp.asarray(truth), jnp.asarray(component), jnp.array(truth.shape), starts[0])
    initial = reset(data, config, jax.random.PRNGKey(2), jnp.asarray(starts))
    expected = reference_trace(truth, starts, goals, send, config)
    # When: the same fixed script runs eagerly, compiled, and in two independent environments.
    eager = scripted_rollout(initial, actions, config)[1].state
    compiled = jax.jit(lambda state: scripted_rollout(state, actions, config)[1].state)(initial)
    batched_initial = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), initial)
    batched = jax.jit(jax.vmap(lambda state: scripted_rollout(state, actions, config)[1].state))(
        batched_initial
    )
    # Then: every physical/cognitive/accounting event matches independently computed values.
    actuals = [
        eager,
        compiled,
        jax.tree_util.tree_map(lambda value: value[0], batched),
        jax.tree_util.tree_map(lambda value: value[1], batched),
    ]
    for actual in actuals:
        fields = (
            actual.positions,
            actual.local_maps,
            actual.balances,
            actual.inbox.buffers,
            actual.inbox.present,
            actual.physical_seen,
            actual.covered,
            actual.goals,
            actual.plans,
            actual.plan_lengths,
            actual.sequences,
            actual.spent_bytes,
            actual.terminated,
            actual.truncated,
        )
        for observed, oracle in zip(fields, expected, strict=True):
            np.testing.assert_array_equal(observed, oracle)


def test_scan_policy_observes_same_round_delivery_inside_each_event() -> None:
    # Given: robot zero broadcasts; robot one only broadcasts if it sees that new packet.
    config = HostConfig(sensor_radius=1, max_distance=1, max_rounds=2, coverage_target=1)
    truth = jnp.ones((5, 12), jnp.uint8).at[2, 1:11].set(0)
    data = MapInput(truth, truth == 0, jnp.array([5, 12]), jnp.array([1, 2]))
    initial = reset(data, config, jax.random.PRNGKey(1), jnp.array([[1, 2], [2, 2]]))

    def policy(observation: LocalObservation, _config: HostConfig) -> Action:
        request = (observation.agent_id == 0) | observation.inbox_present[0]
        return make_action(observation.position, request)

    # When: both decisions are taken by a policy invoked from inside the temporal scan.
    result = jax.jit(lambda state: rollout(state, config, 2, policy))(initial)
    # Then: robot one's newly informed decision also incurs a real45-byte transmission.
    np.testing.assert_array_equal(result.trace.spent, [45, 45])


def test_coverage_completion_freezes_remainder_of_round() -> None:
    # Given: three free cells, two observed at birth, and a known route to the last one.
    config = HostConfig(sensor_radius=0, max_distance=2, coverage_target=1)
    truth = jnp.ones((3, 5), jnp.uint8).at[1, 1:4].set(0)
    data = MapInput(truth, truth == 0, jnp.array([3, 5]), jnp.array([1, 1]))
    initial = reset(data, config, jax.random.PRNGKey(0), jnp.array([[1, 1], [2, 1]]))
    initial = initial._replace(local_maps=initial.local_maps.at[0, 1, 1:4].set(0))

    def policy(_observation: LocalObservation, _config: HostConfig) -> Action:
        return make_action(jnp.array([3, 1]), send=True)

    # When: robot zero reaches the last unseen free cell before robot one's event.
    result = jax.jit(lambda state: rollout(state, config, 4, policy))(initial)
    # Then: only that first event counts, and later entries preserve its final state.
    np.testing.assert_array_equal(result.trace.active, [True, False, False, False])
    assert bool(result.final_state.terminated)
    assert not bool(result.final_state.truncated)
    assert int(result.final_state.event) == 1
    np.testing.assert_array_equal(
        result.trace.positions[1:], result.trace.positions[:1].repeat(3, axis=0)
    )
