"""Local recurrent policies and executed-action likelihoods."""

# Independent network fixtures and conditional distributions use explicit numeric expectations.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from ir2_benchmark.agents.actor import (
    Actor,
    ActorOutput,
    IndexAction,
    action_statistics,
    initial_memory,
    masked_log_probs,
    sample_communication,
)
from ir2_benchmark.agents.critic import Critic, PrivilegedFeatures
from ir2_benchmark.codecs.packet import Codec
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.models.encoders import ActorFeatures, actor_features
from ir2_benchmark.navigation.ir2_graph import GraphObservation
from ir2_benchmark.training.ppo import PPOMetrics, Returns, optimize_ppo, recurrent_outputs
from ir2_benchmark.training.storage import RolloutBatch
from ir2_benchmark.types import MapInput, make_action


def _features() -> ActorFeatures:
    return ActorFeatures(
        jnp.zeros((9, 9, 3)),
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([True, True, False]),
        jnp.zeros(6),
        jnp.zeros((2, 8)),
        jnp.asarray([False, True]),
    )


@pytest.mark.parametrize("stop", ["none", "minibatch", "epoch"])
def test_sequence_minibatches_preserve_order_mask_tail_and_stop_kl(stop: str) -> None:
    features = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (5, 3, *value.shape)), _features()
    )
    ids = jnp.broadcast_to(jnp.arange(5)[:, None], (5, 3))
    times = jnp.broadcast_to(jnp.arange(3), (5, 3))
    zeros = jnp.zeros((5, 3))
    flags = jnp.ones((5, 3), bool).at[-1, -1].set(False)
    batch = RolloutBatch(
        features,
        PrivilegedFeatures(zeros, zeros, zeros),
        initial_memory((5, 3), 16),
        jnp.zeros((5, 3, 13)),
        IndexAction(times, flags, ids),
        zeros,
        zeros,
        zeros,
        ~flags,
        ~flags,
        flags,
        ids,
        times,
        features.candidates,
        features.candidate_mask,
    )
    seen = []

    def update(
        actor: int,
        critic: int,
        mini: RolloutBatch,
        _targets: Returns,
    ) -> tuple[int, int, PPOMetrics]:
        np.testing.assert_array_equal(mini.predictor_version, np.tile(np.arange(3), (2, 1)))
        np.testing.assert_array_equal(mini.actor_ids[:, 0], mini.actor_ids[:, -1])
        seen.extend(np.asarray(mini.actor_ids)[np.asarray(mini.valid)].tolist())
        kl = 0.03 if stop == "minibatch" and actor == 1 else 0.0
        return actor + 1, critic + 1, PPOMetrics(*map(jnp.asarray, (0, 0, 1, 1, kl, 0, 1, 1)))

    def diagnose(
        _actor: int,
        _critic: int,
        _full: RolloutBatch,
        _targets: Returns,
    ) -> dict[str, Array]:
        return {"approx_kl": jnp.float32(0.03 if stop == "epoch" else 0)}

    actor, critic, metrics = optimize_ppo(
        0,
        0,
        batch,
        Returns(zeros, zeros),
        epochs=4,
        minibatch_sequences=2,
        target_kl=0.02,
        key=jax.random.PRNGKey(3),
        update=update,
        diagnose=diagnose,
    )
    expected = {"none": 12, "minibatch": 1, "epoch": 3}[stop]
    assert actor == critic == metrics["optimizer_steps"] == expected
    assert metrics["early_stopped"] == (stop != "none")
    if stop == "none":
        assert metrics["sample_uses"] == 14 * 4
        assert [seen.count(i) for i in range(5)] == [12, 12, 12, 12, 8]


@pytest.mark.parametrize("graph_mode", [False, True])
def test_recurrent_loss_reaches_earlier_observation_without_crossing_robot(
    *,
    graph_mode: bool,
) -> None:
    actor = Actor(hidden=16)
    single = _features()
    if graph_mode:
        single = single._replace(
            graph=GraphObservation(
                jnp.arange(18, dtype=jnp.float32).reshape(3, 6) / 18,
                jnp.arange(3),
                jnp.int32(0),
                jnp.zeros(3, bool),
                jnp.zeros(3, bool),
                jnp.zeros((3, 3), bool),
            )
        )
    features = jax.tree.map(lambda value: jnp.broadcast_to(value, (2, 3, *value.shape)), single)
    memories = initial_memory((2, 3), 16)
    plans = jnp.zeros((2, 3, 13))
    params = actor.init(jax.random.PRNGKey(19), single, memories[0, 0], plans[0, 0])["params"]
    zeros = jnp.zeros((2, 3))
    flags = jnp.ones((2, 3), bool)
    indices = zeros.astype(jnp.int32)
    batch = RolloutBatch(
        features,
        PrivilegedFeatures(zeros, zeros, zeros),
        memories,
        plans,
        IndexAction(indices, flags, indices),
        zeros,
        zeros,
        zeros,
        ~flags,
        ~flags,
        flags,
        indices,
        indices,
        features.candidates,
        features.candidate_mask,
    )

    def last_decision(own: Array) -> Array:
        changed = batch._replace(actor_features=features._replace(self_features=own))
        return recurrent_outputs(actor, params, changed).send_logit[0, -1]

    gradient = jax.jit(jax.grad(last_decision))(features.self_features)
    assert float(jnp.linalg.norm(gradient[0, 0])) > 0
    np.testing.assert_array_equal(gradient[1], 0)
    original = recurrent_outputs(actor, params, batch)
    changed = recurrent_outputs(actor, params, batch._replace(memories=memories.at[:, 1:].set(99)))
    np.testing.assert_allclose(original.send_logit, changed.send_logit)
    if graph_mode:

        def unrolled(parameters: dict) -> Array:
            memory = memories[:, 0]
            for time in range(3):
                output = actor.apply(
                    {"params": parameters},
                    jax.tree.map(lambda value, time=time: value[:, time], features),
                    memory,
                    plans[:, time],
                )
                memory = output.memory
            return output.send_logit.sum() + output.waypoint_logits[:, 1].sum()

        def scanned(parameters: dict) -> Array:
            output = recurrent_outputs(actor, parameters, batch)
            return output.send_logit[:, -1].sum() + output.waypoint_logits[:, -1, 1].sum()

        expected = jax.jit(jax.value_and_grad(unrolled))(params)
        actual = jax.jit(jax.value_and_grad(scanned))(params)
        for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            np.testing.assert_allclose(before, after, rtol=1e-4, atol=2e-6)


def test_candidate_attention_masks_missing_peers_and_preserves_permutations() -> None:
    actor, memory = Actor(hidden=16), initial_memory(hidden=16)
    features = _features()._replace(
        neighbor_features=jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 16,
        candidate_mask=jnp.ones(3, bool),
        neighbor_mask=jnp.ones(2, bool),
    )
    variables = actor.init(jax.random.PRNGKey(21), features, memory, jnp.zeros(13))
    # Remove the pooled-message path so this check requires candidate-level attention.
    variables = {
        "params": {
            **variables["params"],
            "encoder": jax.tree.map(jnp.zeros_like, variables["params"]["encoder"]),
        }
    }

    def scores(peers: Array, mask: Array) -> Array:
        changed = features._replace(neighbor_features=peers, neighbor_mask=mask)
        return actor.apply(variables, changed, memory, jnp.zeros(13)).waypoint_logits

    expected = scores(features.neighbor_features, features.neighbor_mask)
    permuted = scores(features.neighbor_features[::-1], features.neighbor_mask[::-1])
    np.testing.assert_allclose(expected, permuted, atol=1e-6)
    missing = jnp.zeros(2, bool)
    first = scores(features.neighbor_features, missing)
    second = scores(features.neighbor_features + 100, missing)
    assert bool(jnp.all(jnp.isfinite(first)))
    np.testing.assert_allclose(first, second)
    gradient = jax.grad(lambda peers: scores(peers, features.neighbor_mask)[1])(
        features.neighbor_features
    )
    assert float(jnp.linalg.norm(gradient)) > 0
    masked_gradient = jax.grad(lambda peers: scores(peers, missing)[1])(features.neighbor_features)
    np.testing.assert_array_equal(masked_gradient, 0)


def test_candidate_frontier_utility_respects_local_wall_occlusion() -> None:
    grid = jnp.zeros((7, 7), jnp.uint8)
    data = MapInput(grid, grid == 0, jnp.asarray([7, 7]), jnp.asarray([2, 3]))
    config = HostConfig(n_agents=2, sensor_radius=1, max_candidates=2)
    observation = observe(reset(data, config, jax.random.PRNGKey(1)), jnp.int32(0), config)
    local_map = grid.at[:, 6].set(2)
    points = jnp.broadcast_to(jnp.asarray([2, 3]), observation.candidates.positions.shape)
    observation = observation._replace(
        local_map=local_map, candidates=observation.candidates._replace(positions=points)
    )
    clear = actor_features(observation, 2, patch_radius=4)
    blocked = actor_features(
        observation._replace(local_map=local_map.at[:, 4].set(1)), 2, patch_radius=4
    )
    assert float(clear.candidates[0, -1]) > 0
    assert float(blocked.candidates[0, -1]) == 0
    assert not bool(jnp.any(clear.neighbor_mask))


def test_mask_when_all_caller_candidates_are_invalid() -> None:
    # Given: only the mandatory wait action may remain legal.
    logits = jnp.asarray([2.0, 9.0, 4.0])
    # When: compute the normalized masked distribution.
    probabilities = jnp.exp(jax.jit(masked_log_probs)(logits, jnp.zeros(3, bool)))
    # Then: illegal waypoints have exactly zero probability.
    np.testing.assert_array_equal(probabilities, [1.0, 0.0, 0.0])


def test_policy_when_seed_is_repeated() -> None:
    # Given: one initialized real Flax recurrent actor.
    actor, features, memory = Actor(hidden=32), _features(), initial_memory(hidden=32)
    variables = actor.init(jax.random.PRNGKey(1), features, memory, jnp.zeros(13))
    output = actor.apply(variables, features, memory, jnp.zeros(13))
    # When: sample communication twice with the identical semantic policy key.
    first = sample_communication(output, jnp.int32(1), jax.random.PRNGKey(7))
    second = sample_communication(output, jnp.int32(1), jax.random.PRNGKey(7))
    # Then: actions are reproducible and masked motion remains impossible.
    np.testing.assert_array_equal(first, second)
    assert float(jnp.exp(masked_log_probs(output.waypoint_logits, features.candidate_mask))[2]) == 0


def test_memory_when_other_robot_carry_changes() -> None:
    # Given: a batch axis represents two separate per-robot recurrent memories.
    actor = Actor(hidden=32)
    features = jax.tree.map(lambda value: jnp.stack((value, value)), _features())
    memory = initial_memory((2,), 32)
    variables = actor.init(jax.random.PRNGKey(2), features, memory, jnp.zeros((2, 13)))
    # When: only robot 1's own carry changes.
    first = actor.apply(variables, features, memory, jnp.zeros((2, 13)))
    second = actor.apply(variables, features, memory.at[1].set(17), jnp.zeros((2, 13)))
    # Then: robot zero's action distribution and memory are unchanged.
    np.testing.assert_allclose(first.memory[0], second.memory[0])
    np.testing.assert_allclose(first.waypoint_logits[0], second.waypoint_logits[0])


def test_type_gradient_when_send_is_false() -> None:
    # Given: an executed motion and a suppressed communication gate.
    action = IndexAction(jnp.int32(0), jnp.asarray(0, dtype=bool), jnp.int32(2))

    def likelihood(types: Array) -> Array:
        output = ActorOutput(jnp.zeros(1), jnp.zeros(2), jnp.float32(0.3), types)
        return action_statistics(output, action, jnp.ones(2, bool)).log_prob

    # When: differentiate the actual joint action likelihood.
    gradient = jax.grad(likelihood)(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    # Then: an unexecuted message type receives exactly no gradient.
    np.testing.assert_array_equal(gradient, [0.0, 0.0, 0.0, 0.0])


def test_fixed_communication_when_gate_heads_are_overridden() -> None:
    # Given: fixed communication choices share the learned motion policy.
    action = IndexAction(jnp.int32(0), jnp.asarray(1, dtype=bool), jnp.int32(2))

    def likelihood(heads: Array) -> Array:
        output = ActorOutput(jnp.zeros(1), jnp.zeros(2), heads[0], heads[1:])
        return action_statistics(
            output, action, jnp.ones(2, bool), learned_communication=False
        ).log_prob

    # When: differentiate through fixed communication's suppressed likelihood terms.
    gradient = jax.grad(likelihood)(jnp.arange(5, dtype=jnp.float32))
    # Then: neither forced gate nor forced message type is optimized by PPO.
    np.testing.assert_array_equal(gradient, np.zeros(5))


def test_privileged_critic_when_true_joint_inputs_change() -> None:
    # Given: actor and critic receive incompatible explicit input structures.
    actor, features, memory = Actor(hidden=32), _features(), initial_memory(hidden=32)
    variables = actor.init(jax.random.PRNGKey(3), features, memory, jnp.zeros(13))
    critic = Critic(hidden=32)
    privileged = PrivilegedFeatures(jnp.zeros((9, 9, 2)), jnp.zeros((2, 4)), jnp.zeros(2))
    critic_variables = critic.init(jax.random.PRNGKey(4), privileged)
    before = actor.apply(variables, features, memory, jnp.zeros(13))
    # When: the centralized critic sees a changed true robot state.
    changed = privileged._replace(robot_features=jnp.ones((2, 4)))
    critic.apply(critic_variables, changed)
    after = actor.apply(variables, features, memory, jnp.zeros(13))
    # Then: no privileged field can enter the local actor API or change its output.
    assert set(ActorFeatures._fields).isdisjoint(PrivilegedFeatures._fields)
    np.testing.assert_array_equal(before.memory, after.memory)


def test_local_features_when_equal_budget_fractions_hide_different_actual_bytes() -> None:
    # Given: the same actual local observation with two distinct initially full byte budgets.
    grid = jnp.zeros((7, 7), jnp.uint8)
    data = MapInput(grid, grid == 0, jnp.asarray([7, 7]), jnp.asarray([3, 3]))
    config = HostConfig(n_agents=2, sensor_radius=1, max_candidates=2, initial_budget=8192)
    state = reset(data, config, jax.random.PRNGKey(0), spawns=jnp.asarray([[2, 2], [3, 2]]))
    observation = observe(state, jnp.int32(0), config)
    # When: encode both balances with the same fixed byte normalization scale.
    low = actor_features(observation, 2, initial_budget=8192, patch_radius=1)
    high = actor_features(
        observation._replace(balance=jnp.int32(32768)), 2, initial_budget=32768, patch_radius=1
    )
    # Then: the actor can distinguish eight KiB from thirty-two KiB even at fraction one.
    assert not np.allclose(low.self_features, high.self_features)
    np.testing.assert_array_equal(low.local_patch, high.local_patch)


def test_no_plan_control_when_neighbor_intentions_change_preserves_received_position() -> None:
    # Given: receiver one actually receives source zero's paid POSITION packet.
    grid = jnp.zeros((7, 7), jnp.uint8)
    data = MapInput(grid, grid == 0, jnp.asarray([7, 7]), jnp.asarray([3, 3]))
    config = HostConfig(n_agents=2, sensor_radius=1, max_candidates=2, initial_budget=256)
    state = reset(data, config, jax.random.PRNGKey(0), spawns=jnp.asarray([[2, 2], [3, 2]]))
    received = step_agent(
        state,
        make_action(jnp.asarray([2, 2]), send=jnp.asarray(1, bool), codec=Codec.POSITION),
        config,
    ).state
    observation = observe(received, jnp.int32(1), config)
    # When: different neighbor intentions are supplied to the disabled-plan feature adapter.
    changed_plan = observation._replace(
        peer_goals=observation.peer_goals.at[0].set(jnp.asarray([6, 6])),
        peer_plans=observation.peer_plans.at[0].set(jnp.asarray([[5, 5]] * 5)),
        peer_plan_event=observation.peer_plan_event.at[0].set(1),
    )
    baseline = actor_features(observation, 2, patch_radius=1, use_neighbor_plan=False)
    ignored = actor_features(changed_plan, 2, patch_radius=1, use_neighbor_plan=False)
    position_changed = actor_features(
        observation._replace(
            peer_positions=observation.peer_positions.at[0].set(jnp.asarray([4, 4]))
        ),
        2,
        patch_radius=1,
        use_neighbor_plan=False,
    )
    # Then: plan information cannot affect any actor feature, while actual position remains usable.
    for first, second in zip(jax.tree.leaves(baseline), jax.tree.leaves(ignored), strict=True):
        np.testing.assert_array_equal(first, second)
    assert not np.array_equal(baseline.neighbor_features, position_changed.neighbor_features)
    np.testing.assert_allclose(position_changed.neighbor_features[0, -3:-1], [4 / 6, 4 / 6])
