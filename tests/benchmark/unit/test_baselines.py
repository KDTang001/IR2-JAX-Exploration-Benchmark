"""Fixed baselines use local graph routes, original weights and explicit communication."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import FrozenDict

from ir2_benchmark.agents.baselines import (
    BaselineSpec,
    Communication,
    Motion,
    baseline_action,
    baseline_communication,
    load_policy_parameters,
    make_baseline_method,
    spec_from_method,
)
from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.codecs.packet import Codec
from ir2_benchmark.config import ConfigError, HostConfig
from ir2_benchmark.evaluation.batching import ParameterCache, PreparedTask, prepare_task
from ir2_benchmark.evaluation.errors import ModelExecutionError
from ir2_benchmark.evaluation.scenarios import EvaluationMap, Method, make_scenario, tasks
from ir2_benchmark.models.ir2_policy import action_padding_mask
from ir2_benchmark.navigation.candidates import CandidateSet
from ir2_benchmark.navigation.ir2_graph import GraphObservation
from ir2_benchmark.types import LocalFields, LocalObservation


def _observation(
    points: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    utility: list[int],
    candidates: list[int],
    *,
    current: int = 0,
) -> LocalObservation:
    """Supply only public graph knowledge; an all-free map cannot explain these graph routes."""
    nodes, indices = jnp.asarray(points), jnp.asarray(candidates)
    adjacency = jnp.eye(len(points), dtype=bool)
    for origin, destination in edges:
        adjacency = adjacency.at[origin, destination].set(True)
    features = jnp.zeros((len(points), 6)).at[:, :2].set(nodes).at[:, 2].set(utility)
    graph = GraphObservation(
        features,
        indices,
        jnp.int32(current),
        jnp.zeros(len(points), bool),
        indices == 0,
        ~adjacency,
    )
    fields = LocalFields(
        agent_id=jnp.int32(0),
        event=jnp.int32(0),
        round=jnp.int32(0),
        position=nodes[current],
        valid_shape=jnp.array([16, 16]),
        balance=jnp.int32(8192),
        local_map=jnp.zeros((16, 16), jnp.uint8),
        observed_event=jnp.zeros((16, 16), jnp.int32),
        own_goal=nodes[current],
        inbox_buffers=jnp.zeros((2, 256), jnp.uint8),
        inbox_lengths=jnp.zeros(2, jnp.int32),
        inbox_present=jnp.zeros(2, bool),
        message_ages=jnp.full(2, -1),
        peer_positions=jnp.zeros((2, 2), jnp.int32),
        peer_goals=jnp.zeros((2, 2), jnp.int32),
        peer_plans=jnp.zeros((2, 5, 2), jnp.int32),
        peer_plan_lengths=jnp.zeros(2, jnp.int32),
        peer_position_event=jnp.full(2, -1),
        peer_goal_event=jnp.full(2, -1),
        peer_plan_event=jnp.full(2, -1),
        fresh_position=jnp.zeros(2, bool),
        fresh_goal=jnp.zeros(2, bool),
        fresh_plan=jnp.zeros(2, bool),
        graph_nodes=nodes,
        graph_count=jnp.int32(len(points)),
        graph_memory=None,
        peer_maps=None,
    )
    offered = CandidateSet(
        nodes[indices],
        ~action_padding_mask(graph.edge_padding_mask),
        jnp.linalg.norm((nodes[indices] - nodes[current]).astype(jnp.float32), axis=1),
    )
    return LocalObservation(*fields, offered, None, graph)


def _prepared(method: Method, config: HostConfig | None = None) -> PreparedTask:
    """Prepare actual metadata without loading a map or running an environment."""
    scene = make_scenario(
        EvaluationMap("unused.png", "map", "test", "ir2", "rooms", 512, "anchor"),
        HostConfig(navigation_mode="ir2_graph") if config is None else config,
        0,
    )
    return prepare_task(next(tasks((scene,), (method,))))


def _original_tensors() -> dict[str, np.ndarray]:
    """Build the original 90-tensor shape contract without a torch dependency."""
    tensors = {
        "initial_embedding.weight": np.arange(128 * 6, dtype=np.float32).reshape(128, 6),
        "initial_embedding.bias": np.zeros(128, np.float32),
        "current_embedding.weight": np.zeros((128, 256), np.float32),
        "current_embedding.bias": np.zeros(128, np.float32),
        "pointer.w_query": np.eye(128, dtype=np.float32),
        "pointer.w_key": np.eye(128, dtype=np.float32),
    }
    for prefix in (*[f"encoder.layers.{i}" for i in range(6)], "decoder.layers.0"):
        for name in ("w_query", "w_key", "w_value", "w_out"):
            shape = (8, 16, 128) if name == "w_out" else (8, 128, 16)
            tensors[f"{prefix}.multiHeadAttention.{name}"] = np.zeros(shape, np.float32)
        for layer, inputs, outputs in ((0, 128, 512), (2, 512, 128)):
            tensors[f"{prefix}.feedForward.{layer}.weight"] = np.zeros(
                (outputs, inputs), np.float32
            )
            tensors[f"{prefix}.feedForward.{layer}.bias"] = np.zeros(outputs, np.float32)
        for name in ("normalization1", "normalization2"):
            tensors[f"{prefix}.{name}.normalizer.weight"] = np.ones(128, np.float32)
            tensors[f"{prefix}.{name}.normalizer.bias"] = np.zeros(128, np.float32)
    return tensors


def test_frontier_beyond_neighbors_uses_shortest_route_instead_of_farthest_edge() -> None:
    # Given: no immediate frontier, and a farther outgoing neighbor leads to a longer route.
    observation = _observation(
        [(2, 2), (3, 2), (2, 4), (4, 2), (2, 7)],
        [(0, 1), (0, 2), (1, 3), (2, 4)],
        [0, 0, 0, 1, 1],
        [0, 1, 2],
    )
    # When: a compiled policy selects its next legal one-hop action.
    action = jax.jit(nearest_frontier_action, static_argnums=1)(observation, HostConfig())
    # Then: it heads toward the two-unit graph route, without jumping to the frontier.
    np.testing.assert_array_equal(action.waypoint, [3, 2])


def test_frontier_selection_uses_graph_distance_instead_of_euclidean_goal_distance() -> None:
    # Given: the closest-looking frontier requires a long directed detour.
    observation = _observation(
        [(5, 5), (7, 5), (5, 8), (6, 5), (7, 9), (5, 9)],
        [(0, 1), (1, 4), (4, 3), (0, 2), (2, 5)],
        [0, 0, 0, 1, 0, 1],
        [0, 1, 2],
    )
    # When: selecting a frontier using only the known graph.
    action = nearest_frontier_action(observation, HostConfig())
    # Then: the four-unit route wins over the detour to a geometrically nearer target.
    np.testing.assert_array_equal(action.waypoint, [5, 8])


def test_original_slot_zero_exclusion_is_respected_with_current_node_zero() -> None:
    # Given: slot zero offers a closer frontier but the original policy contract excludes it.
    observation = _observation([(2, 2), (3, 2), (2, 5)], [(0, 1), (0, 2)], [0, 1, 1], [1, 2, 0])
    # When: choosing a frontier with node index zero as the current position.
    action = nearest_frontier_action(observation, HostConfig())
    # Then: the legal second slot wins rather than the excluded first slot.
    np.testing.assert_array_equal(action.waypoint, [2, 5])


def test_equal_distance_frontiers_use_target_coordinates() -> None:
    # Given: two equally distant positive-utility goals in reverse coordinate order.
    observation = _observation([(2, 2), (3, 2), (2, 3)], [(0, 1), (0, 2)], [0, 1, 1], [0, 1, 2])
    # When: selecting the nearest frontier.
    action = nearest_frontier_action(observation, HostConfig())
    # Then: target coordinates, rather than node or candidate order, break the tie.
    np.testing.assert_array_equal(action.waypoint, [2, 3])


def test_padded_positive_utility_cannot_attract_a_route() -> None:
    # Given: a short apparent route ends at a padded graph node with positive feature values.
    observation = _observation(
        [(2, 2), (3, 2), (2, 4), (4, 2), (2, 7)],
        [(0, 1), (0, 2), (1, 3), (2, 4)],
        [0, 0, 0, 1, 1],
        [0, 1, 2],
    )
    graph = observation.graph
    assert graph is not None
    observation = observation._replace(
        graph=graph._replace(node_padding_mask=graph.node_padding_mask.at[3].set(True))
    )
    # When: selecting a route using the full padding contract.
    action = nearest_frontier_action(observation, HostConfig())
    # Then: only the reachable valid frontier is considered.
    np.testing.assert_array_equal(action.waypoint, [2, 4])


@pytest.mark.parametrize("candidates", [[0, 1, 2], [0, 2, 1]])
def test_equal_routes_to_one_frontier_use_first_edge_coordinate_ties(candidates: list[int]) -> None:
    # Given: two equal-length routes and candidate orders with opposite coordinate ordering.
    observation = _observation(
        [(3, 3), (3, 2), (2, 3), (2, 2)],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        [0, 0, 0, 1],
        candidates,
    )
    # When: choosing the first edge of the shortest route.
    action = nearest_frontier_action(observation, HostConfig())
    # Then: x/y coordinate order determines the first action.
    np.testing.assert_array_equal(action.waypoint, [2, 3])


@pytest.mark.parametrize("utility", [[0, 0, 1], [1, 0, 0], [0, 0, 0]])
def test_no_reachable_frontier_waits_instead_of_moving_arbitrarily(utility: list[int]) -> None:
    # Given: a legal outgoing edge but no reachable noncurrent positive-utility node.
    observation = _observation([(2, 2), (3, 2), (9, 9)], [(0, 1)], utility, [0, 1])
    # When: choosing a frontier action.
    action = nearest_frontier_action(observation, HostConfig())
    # Then: waiting preserves a legal action even when utility is absent or unreachable.
    np.testing.assert_array_equal(action.waypoint, [2, 2])


@pytest.mark.parametrize("utility", [[1, 0, 0, 0], [0, 0, 0, 1]])
def test_node_zero_cannot_be_a_future_target_or_transit_node(utility: list[int]) -> None:
    # Given: a nonzero current node and a frontier at, or reachable only through, node zero.
    observation = _observation(
        [(5, 2), (2, 2), (3, 2), (6, 2)],
        [(1, 2), (2, 0), (0, 3)],
        utility,
        [1, 2],
        current=1,
    )
    # When: selecting a route whose later actions must obey the original node-zero mask.
    action = nearest_frontier_action(observation, HostConfig())
    # Then: the forbidden destination cannot attract a move into a dead-end route.
    np.testing.assert_array_equal(action.waypoint, [2, 2])


@pytest.mark.parametrize(
    ("communication", "expected"),
    [
        ("off", []),
        ("plan", [Codec.PLAN] * 4),
        ("map", [Codec.MAP_DELTA] * 4),
        ("mixed", [Codec.PLAN, Codec.MAP_DELTA, Codec.MAP_DELTA, Codec.POSITION]),
    ],
)
def test_fixed_communication_sends_at_own_four_action_intervals(
    communication: Communication, expected: list[Codec]
) -> None:
    # Given: sixteen own actions, covering a full mixed cycle.
    rounds = jnp.arange(16)
    # When: computing fixed communication in a compiled batch.
    sends, codecs = jax.jit(jax.vmap(lambda index: baseline_communication(index, communication)))(
        rounds
    )
    # Then: off never emits, and other codecs follow the declared fourth-action schedule.
    np.testing.assert_array_equal(np.flatnonzero(sends), [] if not expected else [0, 4, 8, 12])
    np.testing.assert_array_equal(np.asarray(codecs)[np.asarray(sends)], expected)


@pytest.mark.parametrize("motion", ["nearest_frontier", "ir2_adapted"])
@pytest.mark.parametrize("communication", ["off", "plan", "map", "mixed"])
def test_methods_keep_untrained_canonical_identities(
    tmp_path: Path, motion: Motion, communication: Communication
) -> None:
    # Given: an external asset only for the adapted IR2 motion.
    checkpoint = tmp_path / "reference.npz"
    checkpoint.write_bytes(b"hash-bound external reference")
    # When: making a method and parsing its static inference spec.
    method = make_baseline_method(
        motion, communication, checkpoint if motion == "ir2_adapted" else None
    )
    # Then: the explicit route and identity cannot be mistaken for a training repetition.
    prefix = "nf" if motion == "nearest_frontier" else "ir2"
    assert (
        method.method_id,
        method.route,
        method.estimator,
        method.training_seed,
        method.real_training_events,
    ) == (f"{prefix}_{communication}", "baseline", "none", -1, 0)
    assert spec_from_method(method) == BaselineSpec(motion, communication)
    assert bool(method.checkpoint_hash) == (motion == "ir2_adapted")


def test_nearest_cache_needs_no_checkpoint_or_learner_configuration_file() -> None:
    # Given: an actual prepared nearest-frontier method with no asset path.
    prepared = _prepared(make_baseline_method("nearest_frontier", "off"))
    cache = ParameterCache(1)
    # When: loading the baseline's model parameters.
    parameters = cache.load(prepared)
    # Then: no synthetic parameter bundle occupies the bounded working window.
    assert parameters == ()
    assert cache.size == 0


def test_original_npz_conversion_preserves_tensors_and_freezes_variables(tmp_path: Path) -> None:
    # Given: exactly the original PolicyNet tensor names and layouts.
    path, tensors = tmp_path / "reference.npz", _original_tensors()
    np.savez(path, **tensors)
    # When: loading the safe NumPy transport rather than a pickled object tree.
    variables = load_policy_parameters(path)
    # Then: the original linear layout is transposed and attention/pointer values are preserved.
    assert isinstance(variables, FrozenDict)
    np.testing.assert_array_equal(
        variables["params"]["initial_embedding"]["kernel"], tensors["initial_embedding.weight"].T
    )
    np.testing.assert_array_equal(
        variables["params"]["pointer"]["w_query"], tensors["pointer.w_query"]
    )


def test_non_float32_original_tensor_transport_is_rejected(tmp_path: Path) -> None:
    # Given: a transport with original names but a tensor converted to the wrong precision.
    path, tensors = tmp_path / "reference.npz", _original_tensors()
    tensors["pointer.w_query"] = tensors["pointer.w_query"].astype(np.float64)
    np.savez(path, **tensors)
    # When/Then: loading fails rather than silently changing the original tensor contract.
    with pytest.raises(ModelExecutionError, match="requires_90_float32_tensors"):
        load_policy_parameters(path)


def test_ir2_cache_shares_weights_across_communication_and_rechecks_every_hit(
    tmp_path: Path,
) -> None:
    # Given: one actual original-format reference and two different fixed communication methods.
    path = tmp_path / "reference.npz"
    np.savez(path, **_original_tensors())
    plan = _prepared(make_baseline_method("ir2_adapted", "plan", path))
    mixed = _prepared(make_baseline_method("ir2_adapted", "mixed", path))
    cache = ParameterCache(1)
    original = cache.load(plan)
    # When: reusing parameters across static communication variants, then changing their asset.
    reused = cache.load(mixed)
    path.write_bytes(path.read_bytes() + b"changed")
    # Then: the same tensors are shared, while a cache hit cannot hide changed checkpoint bytes.
    assert reused is original
    assert cache.size == 1
    with pytest.raises(ModelExecutionError, match="checkpoint_content_hash_mismatch"):
        cache.load(mixed)


def test_ir2_missing_checkpoint_hash_is_rejected_before_loading(tmp_path: Path) -> None:
    # Given: a manifest entry whose external reference is no longer hash-bound.
    path = tmp_path / "reference.npz"
    path.write_bytes(b"reference")
    method = replace(make_baseline_method("ir2_adapted", "off", path), checkpoint_hash="")
    # When/Then: preparing it fails before any model loader can infer or invent weights.
    with pytest.raises(ConfigError, match="requires checkpoint and hash"):
        _prepared(method)


def test_baselines_reject_legacy_observation_mode() -> None:
    # Given: the old one-hop configuration lacks the full receiver-owned graph.
    method = make_baseline_method("nearest_frontier", "off")
    # When/Then: preparing a new baseline requires the declared graph observation contract.
    with pytest.raises(ConfigError, match="requires ir2_graph"):
        _prepared(method, HostConfig())


def test_ir2_without_parameters_does_not_initialize_a_random_policy() -> None:
    # Given: a valid graph observation and no original pretrained variables.
    observation = _observation([(2, 2), (3, 2)], [(0, 1)], [0, 1], [0, 1])
    # When/Then: adapted IR2 fails explicitly rather than constructing replacement weights.
    with pytest.raises(ModelExecutionError, match="original_policy_parameters_missing"):
        baseline_action(observation, HostConfig(), BaselineSpec("ir2_adapted", "off"))
