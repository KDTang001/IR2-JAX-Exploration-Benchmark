"""Weight-identical regression against the repository's original PyTorch PolicyNet."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ir2_benchmark.agents.actor import Actor, initial_memory, masked_log_probs
from ir2_benchmark.models.encoders import ActorFeatures
from ir2_benchmark.models.ir2_policy import (
    GraphPolicy,
    action_padding_mask,
    convert_torch_policy_weights,
)
from ir2_benchmark.navigation.ir2_graph import GraphObservation


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("mask_case", ["neighbors", "dead_end", "mixed"])
def test_policy_matches_original_weights(batch_size: int, mask_case: str) -> None:
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "original_ir2_model", Path(__file__).resolve().parents[3] / "model.py"
    )
    assert spec is not None
    assert spec.loader is not None
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(731)
    reference = original.PolicyNet(6, 128).cpu().eval()
    policy = GraphPolicy()
    rng = np.random.default_rng(513)
    n_nodes = 11
    nodes = rng.normal(size=(batch_size, n_nodes, 6)).astype(np.float32)
    node_mask = np.zeros((batch_size, n_nodes), dtype=bool)
    node_mask[:, -2:] = True
    nodes[:, -2:] = 0
    edge_blocking_probability = 0.55
    adjacency = rng.random((batch_size, n_nodes, n_nodes)) > edge_blocking_probability
    adjacency[:, np.arange(n_nodes), np.arange(n_nodes)] = False
    adjacency[:, -1, :] = True  # Exercise original finite-mask all-blocked attention.
    edges = np.tile(np.array([4, 7, 2, 0, 0], dtype=np.int64), (batch_size, 1))
    current = np.full(batch_size, 4, dtype=np.int64)
    edge_mask = np.tile(np.array([False, False, False, True, True]), (batch_size, 1))
    if mask_case == "dead_end":
        edge_mask[:, 1:] = True
    elif mask_case == "mixed":
        edge_mask[0, 1:] = True

    # Relabel every graph, including current/candidate indices and padded nodes.
    for b in range(batch_size):
        permutation = rng.permutation(n_nodes)
        inverse = np.argsort(permutation)
        nodes[b] = nodes[b, permutation]
        node_mask[b] = node_mask[b, permutation]
        adjacency[b] = adjacency[b][permutation][:, permutation]
        edges[b] = inverse[edges[b]]
        current[b] = inverse[current[b]]

    t_nodes = torch.from_numpy(nodes)
    t_node_mask = torch.from_numpy(node_mask[:, None, :])
    t_adjacency = torch.from_numpy(adjacency)
    with torch.no_grad():
        expected_nodes = reference.encode_graph(t_nodes, t_node_mask, t_adjacency)
        expected = reference(
            t_nodes,
            torch.from_numpy(edges[:, None, :]),
            torch.from_numpy(current[:, None, None]),
            t_node_mask,
            torch.from_numpy(edge_mask[:, None, :].copy()),
            t_adjacency,
        ).numpy()
    with jax.default_device(jax.devices("cpu")[0]):
        variables = convert_torch_policy_weights(reference.state_dict())
        args = tuple(map(jnp.asarray, (nodes, edges, current, node_mask, edge_mask, adjacency)))
        actual_nodes = policy.apply(
            variables, args[0], args[3], args[5], method=policy.encode_graph
        )
        actual = jax.jit(policy.apply)(variables, *args)
        np.testing.assert_allclose(actual_nodes, expected_nodes.numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        if batch_size == 1:
            single = policy.apply(variables, *(arg[0] for arg in args))
            np.testing.assert_allclose(single, expected[0], atol=1e-5, rtol=1e-5)


def test_native_initialization_and_feature_methods() -> None:
    policy = GraphPolicy()
    args = (
        jnp.zeros((4, 6)),
        jnp.array([2, 0, 1]),
        jnp.array(2),
        jnp.array([False, False, False, True]),
        jnp.array([False, False, True]),
        jnp.zeros((4, 4), dtype=bool),
    )
    variables = policy.init(jax.random.PRNGKey(0), *args)
    params = variables["params"]
    linear_layers = [params["initial_embedding"], params["current_embedding"]]
    for layer_name in (*[f"encoder_{i}" for i in range(policy.n_layers)], "decoder_0"):
        linear_layers.extend((params[layer_name]["ff1"], params[layer_name]["ff2"]))
    for layer in linear_layers:
        bound = layer["kernel"].shape[0] ** -0.5
        assert np.max(np.abs(layer["kernel"])) <= bound
        assert np.max(np.abs(layer["bias"])) <= bound
        assert np.any(np.asarray(layer["bias"]) != 0)
    nodes = policy.apply(variables, args[0], args[3], args[5], method=policy.encode_graph)
    context, candidates = policy.apply(
        variables, nodes, args[1], args[2], args[3], method=policy.encode_candidates
    )
    assert context.shape == (128,)
    assert candidates.shape == (3, 128)
    logits = policy.apply(variables, context, candidates, args[4], method=policy.pointer_logits)
    np.testing.assert_allclose(jax.nn.log_softmax(logits), policy.apply(variables, *args))
    assert np.isfinite(np.asarray(logits)).all()


def test_action_mask_retains_original_batch_fallback() -> None:
    blocked = jnp.array([[False, True, True], [False, False, True]])
    np.testing.assert_array_equal(
        action_padding_mask(blocked), [[True, True, True], [True, False, True]]
    )
    np.testing.assert_array_equal(action_padding_mask(blocked[0]), [False, True, True])
    np.testing.assert_array_equal(
        jax.vmap(action_padding_mask)(blocked), [[False, True, True], [True, False, True]]
    )


def test_strict_actor_preserves_graph_policy_and_uses_full_graph() -> None:
    graph = GraphObservation(
        jax.random.normal(jax.random.PRNGKey(9), (5, 6)),
        jnp.array([0, 1, 2]),
        jnp.array(0),
        jnp.zeros(5, dtype=bool),
        jnp.zeros(3, dtype=bool),
        jnp.zeros((5, 5), dtype=bool),
    )
    features = ActorFeatures(
        jnp.zeros((1, 1, 3)),
        graph.node_inputs[graph.edge_inputs],
        ~action_padding_mask(graph.edge_padding_mask),
        jnp.zeros(7),
        jnp.zeros((2, 12)),
        jnp.array([False, True]),
        graph,
    )
    actor = Actor()
    memory = initial_memory()
    variables = actor.init(jax.random.PRNGKey(12), features, memory, method=actor.encode)
    carry, logits = actor.apply(variables, features, memory, method=actor.encode)
    expected = GraphPolicy().apply({"params": variables["params"]["graph_policy"]}, *graph)
    np.testing.assert_allclose(jax.nn.log_softmax(logits), expected, atol=1e-5, rtol=1e-5)
    changed = features._replace(self_features=jnp.ones(7), neighbor_features=jnp.ones((2, 12)))
    new_carry, unchanged_logits = actor.apply(variables, changed, memory + 1, method=actor.encode)
    np.testing.assert_array_equal(logits, unchanged_logits)
    assert not np.allclose(carry, new_carry)
    # Node 4 is neither the current node nor any action candidate.
    changed_graph = graph._replace(node_inputs=graph.node_inputs.at[4].set(jnp.arange(6) * 5.0))
    _, changed_logits = actor.apply(
        variables, features._replace(graph=changed_graph), memory, method=actor.encode
    )
    assert not np.allclose(np.asarray(logits)[1:], np.asarray(changed_logits)[1:])
    assert np.isneginf(np.asarray(masked_log_probs(logits, features.candidate_mask))[0])


def test_masked_log_probs_only_enables_wait_for_empty_action_set() -> None:
    logp = masked_log_probs(jnp.array([10.0, 2.0, 3.0]), jnp.array([False, True, True]))
    assert np.isneginf(np.asarray(logp)[0])
    np.testing.assert_allclose(jnp.exp(logp).sum(), 1.0)
    fallback = masked_log_probs(jnp.full(3, -jnp.inf), jnp.zeros(3, dtype=bool))
    np.testing.assert_array_equal(jnp.exp(fallback), [1.0, 0.0, 0.0])
